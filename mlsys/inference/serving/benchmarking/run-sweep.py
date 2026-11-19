#!/usr/bin/env python3
"""Submit up to two serial Slurm lanes using the existing experiment runners.

From benchmarking/:
  python3 run-sweep.py check
  python3 run-sweep.py submit --output outputs/k3-refresh-YYYYMMDD

Every job waits for all earlier jobs in its lane, including failed/cancelled
jobs. Slurm enforces the limit; this script exits after submission. The limit
applies to this sweep, not other jobs or separately submitted sweeps.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CONFIG_KEYS = ('engine', 'prefill_engine', 'decode_engine', 'benchmark', 'deployment')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def command(case, action, config_dir):
    runner = 'run-1p2d.py' if 'prefill_engine' in case else 'run.py'
    cmd = [sys.executable, str(HERE / runner), action]
    for key in CONFIG_KEYS:
        if key in case:
            cmd += ['--' + key.replace('_', '-'), str(config_dir / case[key])]
    return cmd


def load(manifest, config_dir):
    lanes = json.loads(manifest.read_text())['lanes']
    if not 1 <= len(lanes) <= 2 or not all(lanes):
        raise ValueError('Use one or two nonempty lanes')
    names, configs, max_gpus = set(), {}, 0
    for lane in lanes:
        gpu_counts = []
        for case in lane:
            name = case['name']
            if name in names or Path(name).name != name or name in ('.', '..'):
                raise ValueError(f'Invalid or duplicate case name: {name}')
            names.add(name)
            for key in CONFIG_KEYS:
                if key in case:
                    filename = case[key]
                    if Path(filename).name != filename:
                        raise ValueError('Config filenames must be relative basenames')
                    configs[filename] = json.loads((config_dir / filename).read_text())
            deployment = configs[case['deployment']]
            if any(arg.startswith(('--dependency', '-d', '--hold')) for arg in deployment['slurm']['extra_args']):
                raise ValueError('Remove existing dependencies/holds from deployment extra_args')
            gpu_counts.append(deployment['slurm']['nodes'] * deployment['slurm']['gpus_per_node'])
            # Validate every experiment before submitting any jobs.
            subprocess.run(command(case, 'check', config_dir), check=True)
        max_gpus += max(gpu_counts)
    print(f'Validated {len(names)} experiments; at most {len(lanes)} concurrent, {max_gpus} GPUs.', flush=True)
    return lanes, configs, max_gpus


def submit(lanes, configs, max_gpus, output):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'configs'
    snapshot.mkdir()
    for name, value in configs.items():
        write(snapshot / name, value)
    write(output / 'sweep.json', {'lanes': lanes})
    shutil.copy2(__file__, output / 'run-sweep.py')
    record = {'max_parallel': len(lanes), 'max_gpus': max_gpus, 'jobs': []}
    write(output / 'jobs.json', record)
    previous = [[] for _ in lanes]
    # Interleave submission so both lanes can start promptly.
    for index in range(max(map(len, lanes))):
        for lane_id, lane in enumerate(lanes):
            if index >= len(lane):
                continue
            case = copy.deepcopy(lane[index])
            deployment = copy.deepcopy(configs[case['deployment']])
            dependencies = previous[lane_id].copy()
            if dependencies:
                deployment['slurm']['extra_args'].append('--dependency=afterany:' + ':'.join(dependencies))
            case['deployment'] = case['name'] + '-deployment.json'
            write(snapshot / case['deployment'], deployment)
            root = output / case['name']
            cmd = command(case, 'submit', snapshot) + ['--output', str(root)]
            print(f'Submitting lane {lane_id + 1}: {case["name"]}', flush=True)
            subprocess.run(cmd, check=True)
            job_id = (root / 'job-id.txt').read_text().strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError(f'Unexpected Slurm job ID: {job_id}')
            previous[lane_id].append(job_id)
            record['jobs'].append(dict(lane=lane_id + 1, name=case['name'], job_id=job_id,
                                       dependencies=dependencies, output=str(root)))
            write(output / 'jobs.json', record)
    print(f'All jobs submitted; IDs and dependencies: {output / "jobs.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('check', 'submit'))
    parser.add_argument('--manifest', type=Path, default=HERE / 'configs/sweep-k3.json')
    parser.add_argument('--config-dir', type=Path, default=HERE / 'local')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'submit' and args.output is None:
        parser.error('submit requires a new --output directory; do not resubmit an existing sweep')
    lanes, configs, max_gpus = load(args.manifest, args.config_dir.resolve())
    if args.action == 'submit':
        submit(lanes, configs, max_gpus, args.output.resolve())


if __name__ == '__main__':
    main()
