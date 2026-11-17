#!/usr/bin/env python3
"""Preview or submit one containerized vLLM engine and benchmark."""

import argparse
import hashlib
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
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def flags(values: dict) -> list[str]:
    result = []
    for key, value in values.items():
        if value is False or value is None:
            continue
        result.append("--" + key)
        if value is not True:
            result.append(json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value))
    return result


def server_command(config: dict, deployment: dict, rank: int, head: str) -> list[str]:
    spec = dict(config["speculation"], model=config["model"]["draft_path"] or "<DRAFT_PATH>")
    cmd = ["python3", "-m", "vllm.entrypoints.cli.main", "serve", config["model"]["path"] or "<MODEL_PATH>"]
    cmd += flags(config["vllm_args"])
    cmd += flags({"served-model-name": config["model"]["served_name"], "speculative-config": spec,
                  "nnodes": deployment["slurm"]["nodes"], "node-rank": rank, "master-addr": head,
                  "host": "0.0.0.0", "port": config["network"]["port"], "headless": rank != 0})
    return deployment["server_command_prefix"] + cmd


def benchmark_command(config: dict, bench: dict, head: str) -> list[str]:
    if bench.get("kind", "agentx") == "decode":
        return ["python3", "/experiment/benchmark-decode.py", head]
    return ["agentx", "profile"] + flags({
        "url": f"http://{head}:{config['network']['port']}", "endpoint": "/v1/chat/completions",
        "model": config["model"]["served_name"], "tokenizer": config["model"]["path"] or "<MODEL_PATH>",
        "tokenizer-trust-remote-code": True, "concurrency": bench["concurrency"],
        "public-dataset": bench["dataset"], "benchmark-duration": bench["duration_seconds"],
        "random-seed": bench["seed"], "output-artifact-dir": "/experiment/agentx",
    }) + bench["extra_args"]


def validate(config: dict, bench: dict, deployment: dict, local_files: bool = False) -> None:
    if "benchmark" in config or "vllm_args" in bench:
        raise ValueError("Keep engine and benchmark settings in separate config files")
    if any(key in config for key in ("slurm", "container", "patches", "server_command_prefix", "startup_seconds")) or "container" in bench:
        raise ValueError("Keep allocation, containers, and launcher settings in the deployment config")
    slurm, spec = deployment["slurm"], config["speculation"]
    if slurm["nodes"] * slurm["gpus_per_node"] != config["vllm_args"]["tensor-parallel-size"]:
        raise ValueError("This runner requires one TP group spanning all allocated GPUs")
    kind = bench.get("kind", "agentx")
    if kind not in ("agentx", "decode"):
        raise ValueError(f"Unknown benchmark kind: {kind}")
    if kind == "decode":
        if bench["concurrency"] != 1 or min(bench["repeats"], bench["requests_per_repeat"], bench["output_tokens"]) < 1:
            raise ValueError("Decode benchmark requires concurrency 1 and positive request counts/length")
        if len(bench["prompt_token_ids"]) != 1:
            raise ValueError("Short-context reproduction requires exactly one input token")
    elif bench["concurrency"] < 1 or bench["duration_seconds"] <= 0:
        raise ValueError("Benchmark concurrency and duration must be positive")
    if spec["rejection_sample_method"] == "synthetic":
        if not 1 <= spec["synthetic_acceptance_length"] <= spec["num_speculative_tokens"] + 1:
            raise ValueError("Synthetic acceptance length must be between 1 and draft length + 1")
    reserved = {"nnodes", "node-rank", "master-addr", "headless", "host", "port", "speculative-config", "served-model-name"}
    if reserved.intersection(config["vllm_args"]):
        raise ValueError("Runner-managed arguments must not also appear in vllm_args")
    for patch in deployment["patches"]:
        target = Path(patch["target"])
        if target.is_absolute() or ".." in target.parts or not target.parts or target.parts[0] != "vllm" or target.suffix != ".py":
            raise ValueError("Patch targets must be .py files beneath vllm/, without '..'")
    targets = [patch["target"] for patch in deployment["patches"]]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate patch targets")
    if not local_files:
        return
    cache_dir = deployment.get("cache_dir")
    if cache_dir and (not Path(cache_dir).is_absolute() or any(char in cache_dir for char in (",", ":"))):
        raise ValueError("cache_dir must be an absolute path without ',' or ':'")
    for key in ("path", "draft_path"):
        value = config["model"][key]
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f"Set model.{key} to an existing absolute directory")
        if not (Path(value) / "config.json").is_file():
            raise ValueError(f"Missing model config.json: {value}")
    for label, settings in deployment["containers"].items():
        value = settings["image"]
        if not value or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError(f"Set deployment containers.{label}.image to a prepared absolute .sqsh file")
    for patch in deployment["patches"]:
        if not Path(patch["source"]).is_absolute() or not Path(patch["source"]).is_file():
            raise ValueError("Patch source must be an existing absolute file")
    mounts = deployment["containers"]["engine"]["mounts"] + deployment["containers"]["benchmark"]["mounts"]
    for mount in mounts:
        if not mount["source"] or not Path(mount["source"]).is_absolute() or not Path(mount["source"]).exists():
            raise ValueError("Mount source must be an existing absolute path")
        if not mount["target"] or not Path(mount["target"]).is_absolute():
            raise ValueError("Mount target must be absolute")
    mount_paths = list(config["model"][key] for key in ("path", "draft_path"))
    mount_paths += [mount[key] for mount in mounts for key in ("source", "target")]
    if any("," in path or ":" in path for path in mount_paths):
        raise ValueError("Pyxis mount paths cannot contain ',' or ':'")


def container_args(config: dict, container: dict, output: Path, role: str, cache_dir: str | None = None) -> list[str]:
    mounts = [f"{output}:/experiment"]
    if cache_dir:
        mounts.append(f"{cache_dir}:/cache:rw")
    for key in (("path", "draft_path") if role == "server" else ("path",)):
        path = config["model"][key]
        mounts.append(f"{path}:{path}:ro")
    for mount in container["mounts"]:
        mounts.append(f"{mount['source']}:{mount['target']}:{'ro' if mount.get('readonly', True) else 'rw'}")
    return [f"--container-image={container['image']}",
            "--container-remap-root", "--no-container-mount-home", "--container-workdir=/experiment",
            "--container-mounts=" + ",".join(mounts)]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(config: dict, bench: dict, deployment: dict, output: Path) -> None:
    if deployment.get("cache_dir"):
        Path(deployment["cache_dir"]).mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("run.py", "worker.py", "job.sh"):
        shutil.copy2(ROOT / filename, output / filename)
    if bench.get("kind") == "decode":
        shutil.copy2(ROOT / "benchmark-decode.py", output / "benchmark-decode.py")
    for index, patch in enumerate(deployment["patches"]):
        target = output / "patches" / f"{index}.py"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(patch["source"], target)
        patch["sha256"] = sha256(target)
        patch["snapshot"] = f"/experiment/patches/{index}.py"
    (output / "engine.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")
    (output / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    # Hash images before allocation, not while 16 GPUs wait for host I/O.
    manifest = {role: {"path": settings["image"],
                       "sha256": sha256(Path(settings["image"]))}
                for role, settings in deployment["containers"].items()}
    (output / "images.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "commands.txt").write_text(
        shlex.join(server_command(config, deployment, 0, "<HEAD_IP>")) + "\n" +
        shlex.join(benchmark_command(config, bench, "<HEAD_IP>")) + "\n")


def stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()  # srun propagates termination to its own job step.
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def job(config: dict, bench: dict, deployment: dict, output: Path) -> None:
    # srun sorts its nodelist; the allocation's original order can differ.
    nodes = sorted(subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines())
    slurm = deployment["slurm"]
    if len(nodes) != slurm["nodes"] or len({node.split("-t")[0] for node in nodes}) != 1:
        raise RuntimeError(f"Expected {slurm['nodes']} MIS CDG trays in one NVLink domain: {nodes}")
    head = socket.gethostbyname(nodes[0])
    (output / "allocation.json").write_text(json.dumps({"job_id": os.environ["SLURM_JOB_ID"], "nodes": nodes, "head": head}, indent=2))
    # One process per node, with access to the full exclusive allocation.
    # Explicit --cpus-per-task implies --exact in srun; only set it on request.
    common = ["srun", "--mpi=none", "--kill-on-bad-exit=1", "--cpu-bind=none"]
    common += flags({"cpus-per-task": slurm.get("cpus_per_task")})
    server = common + [f"--nodes={len(nodes)}", f"--ntasks={len(nodes)}", "--ntasks-per-node=1",
                       "--nodelist=" + ",".join(nodes), "--distribution=block",
                       f"--gpus-per-node={slurm['gpus_per_node']}",
                       "--mem=0", f"--output={output}/server-%t.log"]
    server += container_args(config, deployment["containers"]["engine"], output, "server", deployment.get("cache_dir")) + ["python3", "/experiment/worker.py", "server", head]
    client = common + ["--overlap", "--nodes=1", "--ntasks=1", f"--nodelist={nodes[0]}",
                       "--gres=none", "--mem=0", f"--output={output}/benchmark.log"]
    client += container_args(config, deployment["containers"]["benchmark"], output, "benchmark", deployment.get("cache_dir")) + ["python3", "/experiment/worker.py", "benchmark", head]
    (output / "launch-commands.txt").write_text(shlex.join(server) + "\n" + shlex.join(client) + "\n")
    server_process = client_process = None
    try:
        server_process = subprocess.Popen(server)
        deadline = time.monotonic() + deployment["startup_seconds"]
        while True:
            if server_process.poll() is not None:
                raise RuntimeError("Server step exited before readiness; inspect server logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server readiness timed out")
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://{head}:{config['network']['port']}/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(5)
        # The batch allocation exports GPU request variables. Do not let srun
        # inherit those into the CPU-only client step alongside --gres=none.
        client_env = {key: value for key, value in os.environ.items()
                      if not key.startswith("SLURM_GPUS") and key not in ("SLURM_MEM_PER_GPU", "SLURM_TRES_PER_TASK")}
        client_env["NVIDIA_VISIBLE_DEVICES"] = "void"
        client_process = subprocess.Popen(client, env=client_env)
        deadline = time.monotonic() + bench["timeout_seconds"]
        while client_process.poll() is None:
            if server_process.poll() is not None:
                raise RuntimeError("Server step exited during measurement")
            if time.monotonic() >= deadline:
                raise TimeoutError("Benchmark timed out, including warmup and postprocessing")
            time.sleep(5)
        if client_process.returncode:
            raise RuntimeError(f"Benchmark exited {client_process.returncode}; inspect benchmark.log")
        (output / "completed.json").write_text(json.dumps({"benchmark_exit_code": 0, "note": "Inspect benchmark artifacts before using metrics."}))
    finally:
        stop(client_process)
        stop(server_process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "check", "submit", "job"))
    parser.add_argument("--engine", type=Path, required=True, help="vLLM engine JSON")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark/workload JSON")
    parser.add_argument("--deployment", type=Path, required=True, help="Slurm allocation, containers, and launcher JSON")
    parser.add_argument("--output", type=Path, help="New shared-filesystem directory for the run")
    args = parser.parse_args()
    config = json.loads(args.engine.read_text())
    bench = json.loads(args.benchmark.read_text())
    deployment = json.loads(args.deployment.read_text())
    validate(config, bench, deployment, local_files=args.action in ("check", "submit"))
    if args.action == "preview":
        print("# Server rank 0; other ranks add --headless")
        print(shlex.join(server_command(config, deployment, 0, "<HEAD_IP>")))
        print("\n# Benchmark")
        print(shlex.join(benchmark_command(config, bench, "<HEAD_IP>")))
    elif args.action == "check":
        print("Local paths and config valid. Container compatibility and GPU startup remain untested.")
    elif args.action == "job":
        job(config, bench, deployment, args.engine.resolve().parent)
    else:
        if args.output is None:
            parser.error("submit requires --output pointing to a new shared-filesystem directory")
        output = args.output.resolve()
        if any(char in str(output) for char in (",", ":")):
            parser.error("Pyxis run directory cannot contain ',' or ':'")
        prepare(config, bench, deployment, output)
        slurm = deployment["slurm"]
        cmd = ["sbatch", "--parsable", "--exclusive", "--mem=0"] + flags({
            "job-name": f"{config['name']}-{bench['name']}",
            "account": slurm.get("account"), "partition": slurm.get("partition"),
            "time": slurm["time"], "nodes": slurm["nodes"], "ntasks-per-node": 1,
            "gpus-per-node": slurm["gpus_per_node"], "cpus-per-task": slurm.get("cpus_per_task"),
            "output": str(output / "slurm-%j.log"),
        }) + slurm["extra_args"] + [str(output / "job.sh"), str(output)]
        (output / "submit-command.txt").write_text(shlex.join(cmd) + "\n")
        job_id = subprocess.check_output(cmd, text=True).strip()
        (output / "job-id.txt").write_text(job_id + "\n")
        print(f"Submitted {job_id}; artifacts: {output}")


if __name__ == "__main__":
    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError) as error:
        sys.exit(str(error))
