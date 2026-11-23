#!/usr/bin/env python3
"""One 8-GPU prefill worker and two 8-GPU decode workers, with Dynamo."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
from run import benchmark_command, container_args, flags, sha256, stop, validate
from worker import check_model


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def engine_command(engine, deployment, rank, addresses):
    role = 'prefill' if rank < 2 else 'decode'
    cmd = ['python3', '-m', 'dynamo.vllm', '--model', engine['model']['path'] or '<MODEL_PATH>']
    cmd += flags(engine['vllm_args'])
    cmd += flags({'served-model-name': engine['model']['served_name'],
                  'speculative-config': dict(engine['speculation'], model=engine['model']['draft_path'] or '<DRAFT_PATH>'),
                  'nnodes': 2, 'node-rank': rank % 2, 'master-addr': addresses[rank // 2 * 2],
                  'headless': rank % 2 == 1,
                  'namespace': deployment['dynamo']['namespace'], 'disaggregation-mode': role})
    return cmd + flags(deployment['dynamo']['worker_args'])


def runtime_env(deployment, addresses):
    env = dict(os.environ, **deployment['dynamo']['env'])
    env.update({'ETCD_ENDPOINTS': f'http://{addresses[0]}:2379',
                'NATS_SERVER': f'nats://{addresses[0]}:4222',
                'DYN_TCP_RPC_HOST': socket.gethostbyname(socket.gethostname()),
                'HF_HOME': '/cache/huggingface', 'XDG_CACHE_HOME': '/cache',
                'VLLM_CACHE_ROOT': '/cache/vllm'})
    return env


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as r:
        return r.read()


def wait_ready(urls, processes, seconds):
    deadline = time.monotonic() + seconds
    pending = set(urls)
    while pending:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f'Step exited during readiness with code {process.returncode}')
        for url in tuple(pending):
            try:
                get(url)
                pending.remove(url)
                print('Ready:', url, flush=True)
            except (OSError, TimeoutError):
                pass
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Readiness timed out: {sorted(pending)}')
            time.sleep(5)


def wait_port(host, port, processes, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f'Mooncake process exited before {host}:{port} was ready')
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f'Mooncake readiness timed out: {host}:{port}')


def mooncake_owner_command(deployment, address, master):
    mc = deployment['dynamo']['mooncake']
    owner = deployment['dynamo']['mooncake_store']
    return ['mooncake_client', f'--host={address}:50053', '--port=50052',
            f'--master_server_address={master}:50051',
            f'--metadata_server={mc["metadata_server"]}',
            f'--protocol={mc["protocol"]}', f'--device_names={mc["device_name"]}',
            f'--global_segment_size={owner["global_segment_size"]}',
            f'--local_buffer_size={owner["local_buffer_size"]}',
            '--enable_offload=true', '--start_offload_rpc_server=true']


def worker_with_mooncake_owner(root, deployment, addresses, rank, env, cmd):
    owner = deployment['dynamo']['mooncake_store']
    owner_env = dict(env, **owner['env'])
    base = Path(owner_env['MOONCAKE_OFFLOAD_FILE_STORAGE_PATH'])
    base.mkdir(parents=True, exist_ok=True)
    capacity = int(owner_env['MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES'])
    filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(base)], text=True).strip()
    if filesystem in ('tmpfs', 'ramfs', 'nfs', 'nfs4', 'lustre', 'overlay'):
        raise RuntimeError(f'Mooncake SSD path must be on local disk, got {filesystem}: {base}')
    if shutil.disk_usage(base).free < capacity + 16 * 1024**3:
        raise RuntimeError(f'Insufficient free SSD space for {capacity} bytes at {base}')
    path = base / f'k3-{os.environ["SLURM_JOB_ID"]}-rank{rank}'
    path.mkdir(exist_ok=False)
    owner_env['MOONCAKE_OFFLOAD_FILE_STORAGE_PATH'] = str(path)
    owner_cmd = mooncake_owner_command(deployment, addresses[rank], addresses[0])
    write(root / f'mooncake-owner-{rank}.json', {
        'command': owner_cmd, 'env': {k: owner_env[k] for k in owner['env']},
        'filesystem': filesystem, 'free_bytes': shutil.disk_usage(base).free})
    if owner['prefer_local_segment']:
        env['MOONCAKE_PREFERRED_SEGMENT'] = f'{addresses[rank]}:50053'
    processes = []
    try:
        wait_port(addresses[0], 50051, [], 120)
        with (root / f'mooncake-owner-{rank}.log').open('w') as log:
            processes.append(subprocess.Popen(owner_cmd, env=owner_env, stdout=log, stderr=subprocess.STDOUT))
        wait_port(addresses[rank], 50052, processes, 600)
        processes.append(subprocess.Popen(cmd, env=env))
        while processes[-1].poll() is None:
            if processes[0].poll() is not None:
                raise RuntimeError('Mooncake pool owner exited while the engine was running')
            time.sleep(2)
        if processes[-1].returncode:
            raise RuntimeError(f'Engine exited {processes[-1].returncode}')
    finally:
        for process in reversed(processes):
            stop(process)
        shutil.rmtree(path)


def worker(root):
    deployment = read(root / 'deployment.json')
    addresses = read(root / 'allocation.json')['addresses']
    rank = int(os.environ['SLURM_PROCID'])
    role = 'prefill' if rank < 2 else 'decode'
    engine = read(root / f'engine-{role}.json')
    env = runtime_env(deployment, addresses)
    env.update(engine['env'])
    env.update({'NCCL_SOCKET_IFNAME': deployment['network_interface'],
                'GLOO_SOCKET_IFNAME': deployment['network_interface'],
                'VLLM_HOST_IP': addresses[rank], 'DYN_SYSTEM_PORT': '8001',
                'VLLM_NIXL_SIDE_CHANNEL_HOST': addresses[rank],
                'VLLM_NIXL_SIDE_CHANNEL_PORT': '5600',
                'MOONCAKE_CONFIG_PATH': str(root / f'mooncake-{rank}.json')})
    mc = dict(deployment['dynamo']['mooncake'], master_server_address=f'{addresses[0]}:50051')
    write(env['MOONCAKE_CONFIG_PATH'], mc)
    for key in ('path', 'draft_path'):
        check_model(engine['model'][key])
    # Fail before loading weights if the image/site UCX environment cannot
    # initialize a GPU-capable NIXL backend.
    subprocess.run(['python3', '-c',
        'from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config; '
        'agent = NixlWrapper("startup-check", nixl_agent_config(num_threads=4)); '
        'from mooncake.store import MooncakeDistributedStore; store = MooncakeDistributedStore(); '
        'print("NIXL backend and Mooncake store initialized", flush=True)'], env=env, check=True, timeout=60)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True)
    if len(devices.splitlines()) != 4 or any('GB300' not in line for line in devices.splitlines()):
        raise RuntimeError(f'Expected four GB300 GPUs: {devices}')
    cmd = engine_command(engine, deployment, rank, addresses)
    versions = {n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'flashinfer-python', 'nixl', 'ai-dynamo', 'ai-dynamo-runtime', 'mooncake-transfer-engine-cuda13')}
    write(root / f'runtime-{rank}.json', {'role': role, 'hostname': socket.gethostname(), 'gpus': devices, 'versions': versions, 'command': cmd})
    print(shlex.join(cmd), flush=True)
    if deployment['dynamo'].get('mooncake_store'):
        return worker_with_mooncake_owner(root, deployment, addresses, rank, env, cmd)
    os.execvpe(cmd[0], cmd, env)


def infra(root):
    deployment = read(root / 'deployment.json')
    addresses = read(root / 'allocation.json')['addresses']
    head = addresses[0]
    env = runtime_env(deployment, addresses)
    env['ETCD_UNSUPPORTED_ARCH'] = 'arm64'
    # Dynamo warns against spanning NUMA nodes in the HTTP frontend.
    numa_cpus = set()
    for span in Path('/sys/devices/system/node/node0/cpulist').read_text().strip().split(','):
        ends = [int(n) for n in span.split('-')]
        numa_cpus.update(range(ends[0], ends[-1] + 1))
    frontend_cpus = sorted(numa_cpus & os.sched_getaffinity(0))
    if not frontend_cpus:
        raise RuntimeError('No allocated CPUs on NUMA node 0')
    (root / 'nats.conf').write_text('port: 4222\nmax_payload: 32MB\n')
    commands = {
        'etcd': ['etcd', '--name=benchmark', '--data-dir=/tmp/k3-etcd',
                 f'--listen-client-urls=http://{head}:2379', f'--advertise-client-urls=http://{head}:2379',
                 f'--listen-peer-urls=http://{head}:2380', f'--initial-advertise-peer-urls=http://{head}:2380',
                 f'--initial-cluster=benchmark=http://{head}:2380'],
        'nats': ['nats-server', '-c', str(root / 'nats.conf')],
        'mooncake-master': ['mooncake_master', '--rpc_port=50051', '--default_kv_lease_ttl=60000',
                            '--eviction_high_watermark_ratio=0.95', '--eviction_ratio=0.10'],
        'frontend': ['taskset', '--cpu-list', ','.join(map(str, frontend_cpus)),
                     'python3', '-m', 'dynamo.frontend', '--http-host=0.0.0.0', '--http-port=8000',
                     '--namespace', deployment['dynamo']['namespace']] + flags(deployment['dynamo']['frontend_args']),
    }
    if deployment['dynamo']['mooncake'].get('enable_offload'):
        commands['mooncake-master'].append('--enable_offload=true')
    processes = []
    try:
        for name, cmd in commands.items():
            with (root / f'{name}.log').open('w') as log:
                processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
            if name == 'etcd':
                wait_ready([f'http://{head}:2379/health'], processes, 60)
        while True:
            for p in processes:
                if p.poll() is not None:
                    raise RuntimeError(f'Infrastructure process exited {p.returncode}')
            time.sleep(2)
    finally:
        for p in reversed(processes):
            stop(p)


def benchmark(root):
    bench, deployment = read(root / 'benchmark.json'), read(root / 'deployment.json')
    engine = read(root / 'engine.json')
    addresses = read(root / 'allocation.json')['addresses']
    if importlib.metadata.version('agentx-harness') != bench['harness_version']:
        raise RuntimeError('AgentX version mismatch')
    cmd = benchmark_command(engine, bench, addresses[0])
    cmd += ['--server-metrics'] + [f'http://{addresses[i]}:8001/metrics' for i in (0, 2, 4)]
    env = dict(os.environ, **bench['env'], HF_HOME='/cache/huggingface', XDG_CACHE_HOME='/cache', CUDA_VISIBLE_DEVICES='')
    write(root / 'benchmark-runtime.json', {'command': cmd, 'env': bench['env']})
    print(shlex.join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)


def job(root):
    deployment, bench = read(root / 'deployment.json'), read(root / 'benchmark.json')
    engine = read(root / 'engine.json')
    nodes = sorted(subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).splitlines())
    if len(nodes) != 6 or len({n.split('-t')[0] for n in nodes}) != 1:
        raise RuntimeError(f'Expected six trays in one NVLink domain: {nodes}')
    addresses = [socket.gethostbyname(n) for n in nodes]
    write(root / 'allocation.json', {'job_id': os.environ['SLURM_JOB_ID'], 'nodes': nodes, 'addresses': addresses})
    common = ['srun', '--mpi=none', '--cpu-bind=none', '--kill-on-bad-exit=1', '--mem=0']
    cpu = common + ['--overlap', '--nodes=1', '--ntasks=1', '--gres=none', f'--nodelist={nodes[0]}']
    cpu_env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_GPUS') and k not in ('SLURM_MEM_PER_GPU', 'SLURM_TRES_PER_TASK')}
    cpu_env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    mounts = container_args(engine, deployment['containers']['engine'], root, 'server', deployment.get('cache_dir'))
    entry = ['python3', '/experiment/run-1p2d.py']
    infra_cmd = cpu + [f'--output={root}/infra.log'] + mounts + entry + ['infra', '/experiment']
    servers_cmd = common + ['--nodes=6', '--ntasks=6', '--ntasks-per-node=1', '--gpus-per-node=4', '--distribution=block',
                             '--nodelist=' + ','.join(nodes), f'--output={root}/server-%t.log'] + mounts + entry + ['worker', '/experiment']
    bench_cmd = cpu + [f'--output={root}/benchmark.log'] + container_args(engine, deployment['containers']['benchmark'], root, 'benchmark', deployment.get('cache_dir')) + entry + ['benchmark', '/experiment']
    (root / 'launch-commands.txt').write_text('\n'.join(shlex.join(c) for c in (infra_cmd, servers_cmd, bench_cmd)) + '\n')
    processes = []
    try:
        processes.append(subprocess.Popen(infra_cmd, env=cpu_env))
        wait_ready([f'http://{addresses[0]}:2379/health'], processes, 120)
        processes.append(subprocess.Popen(servers_cmd))
        wait_ready([f'http://{addresses[i]}:8001/health' for i in (0, 2, 4)], processes, deployment['startup_seconds'])
        wait_ready([f'http://{addresses[0]}:8000/v1/models'], processes, 120)
        models = json.loads(get(f'http://{addresses[0]}:8000/v1/models'))
        if not any(m['id'] == engine['model']['served_name'] for m in models['data']):
            raise RuntimeError(f'Model missing from frontend: {models}')
        client = subprocess.Popen(bench_cmd, env=cpu_env)
        processes.append(client)
        deadline = time.monotonic() + bench['timeout_seconds']
        while client.poll() is None:
            if any(p.poll() is not None for p in processes[:-1]):
                raise RuntimeError('Serving step exited during measurement')
            if time.monotonic() >= deadline:
                raise TimeoutError('Benchmark timed out')
            time.sleep(5)
        if client.returncode:
            raise RuntimeError(f'Benchmark exited {client.returncode}')
        write(root / 'completed.json', {'job_id': os.environ['SLURM_JOB_ID'], 'gpu_count': 24})
    finally:
        for p in reversed(processes):
            stop(p)


def validate_1p2d(engines, bench, deployment, local_files=False):
    mc = deployment['dynamo']['mooncake']
    owner = deployment['dynamo'].get('mooncake_store')
    if mc.get('mode') == 'standalone-store' or owner:
        if not owner or mc.get('mode') != 'standalone-store' or mc['global_segment_size'] != 0:
            raise ValueError('Standalone Mooncake requires a per-node owner and zero pool allocation in engine ranks')
        if not mc['enable_offload'] or not owner['enable_offload']:
            raise ValueError('Enable SSD offloading in both the requester and owner configs')
        owner_env = owner['env']
        capacity = int(owner_env['MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES'])
        if capacity <= 0 or int(owner_env['MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE']) != capacity:
            raise ValueError('Set matching positive Mooncake SSD advertised and disk eviction capacities')
        if local_files:
            path = owner_env['MOONCAKE_OFFLOAD_FILE_STORAGE_PATH']
            if not path or not Path(path).is_absolute():
                raise ValueError('Set an absolute local SSD path for Mooncake')
            if not any(not m.get('readonly', True) and Path(path).is_relative_to(m['target'])
                       for m in deployment['containers']['engine']['mounts']):
                raise ValueError('Mount the Mooncake SSD path read-write in the engine container')
    if deployment['slurm']['nodes'] != 6 or deployment['slurm']['gpus_per_node'] != 4:
        raise ValueError('1P2D requires six nodes with four GPUs each')
    if deployment['patches'] or deployment['server_command_prefix']:
        raise ValueError('The 1P2D launcher does not support patches or server_command_prefix')
    if bench.get('kind', 'agentx') != 'agentx':
        raise ValueError('The 1P2D launcher runs AgentX benchmarks')
    prefill_spec = copy.deepcopy(engines['prefill']['speculation'])
    decode_spec = engines['decode']['speculation']
    # Prefill still builds the draft context KV consumed by speculative decode.
    if prefill_spec is not None:
        prefill_spec.pop('prefill_context_kv_only', None)
    if engines['prefill']['model'] != engines['decode']['model'] or prefill_spec != decode_spec:
        raise ValueError('Prefill and decode must use the same models and speculation settings, except prefill_context_kv_only on prefill')
    worker_deployment = copy.deepcopy(deployment)
    worker_deployment['slurm']['nodes'] = 2
    for engine in engines.values():
        if engine['vllm_args']['tensor-parallel-size'] != 8 or engine['vllm_args'].get('data-parallel-size', 1) != 1:
            raise ValueError('Each 1P2D worker requires TP8 / DP1')
        if engine['network']['port'] != 8000:
            raise ValueError('The 1P2D frontend listens on port 8000')
        validate(engine, bench, worker_deployment, local_files=local_files)


def submit(engines, bench, deployment, root):
    validate_1p2d(engines, bench, deployment, local_files=True)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('run.py', 'worker.py'):
        shutil.copy2(HERE / name, root / name)
    shutil.copy2(__file__, root / 'run-1p2d.py')
    for role, engine in engines.items():
        write(root / f'engine-{role}.json', engine)
    write(root / 'engine.json', engines['decode'])
    write(root / 'deployment.json', deployment)
    write(root / 'benchmark.json', bench)
    write(root / 'images.json', {role: {'path': c['image'], 'sha256': sha256(Path(c['image']))} for role, c in deployment['containers'].items()})
    script = root / 'job.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 ' + shlex.quote(str(root / 'run-1p2d.py')) + ' job ' + shlex.quote(str(root)) + '\n')
    slurm = deployment['slurm']
    cmd = ['sbatch', '--parsable', '--exclusive', '--mem=0'] + flags({
        'account': slurm['account'], 'partition': slurm.get('partition'), 'time': slurm['time'],
        'nodes': 6, 'ntasks-per-node': 1, 'gpus-per-node': 4, 'job-name': 'k3-1p2d',
        'output': str(root / 'slurm-%j.log')}) + slurm['extra_args'] + [str(script)]
    (root / 'submit-command.txt').write_text(shlex.join(cmd) + '\n')
    job_id = subprocess.check_output(cmd, text=True).strip()
    (root / 'job-id.txt').write_text(job_id + '\n')
    print(f'Submitted {job_id}; artifacts: {root}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('preview', 'check', 'submit'):
        command = commands.add_parser(action)
        command.add_argument('--prefill-engine', type=Path, required=True)
        command.add_argument('--decode-engine', type=Path, required=True)
        command.add_argument('--benchmark', type=Path, required=True)
        command.add_argument('--deployment', type=Path, required=True)
        if action == 'submit':
            command.add_argument('--output', type=Path, required=True)
    for action in ('job', 'worker', 'infra', 'benchmark'):
        commands.add_parser(action).add_argument('output', type=Path)
    args = parser.parse_args()
    if args.action in ('preview', 'check', 'submit'):
        engines = {'prefill': read(args.prefill_engine), 'decode': read(args.decode_engine)}
        bench, deployment = read(args.benchmark), read(args.deployment)
        validate_1p2d(engines, bench, deployment, local_files=args.action != 'preview')
        if args.action == 'submit':
            submit(engines, bench, deployment, args.output.resolve())
        elif args.action == 'preview':
            addresses = [f'<NODE{i}_IP>' for i in range(6)]
            for rank in range(6):
                role = 'prefill' if rank < 2 else 'decode'
                print(shlex.join(engine_command(engines[role], deployment, rank, addresses)))
        else:
            print('1P2D configs and local paths validated')
    else:
        {'job': job, 'worker': worker, 'infra': infra, 'benchmark': benchmark}[args.action](args.output)
