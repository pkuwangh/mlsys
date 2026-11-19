#!/usr/bin/env python3
"""Container entrypoint; shared by the serving and benchmark images."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

from run import benchmark_command, server_command


def patch_vllm(patches: list[dict]) -> dict:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Server image does not contain the vLLM Python package")
    installed = Path(next(iter(spec.submodule_search_locations)))
    if not patches:
        return {"installed_package": str(installed), "patches": []}
    overlay = Path(tempfile.mkdtemp(prefix="vllm-instrumentation-"))
    # Retain the image's extensions and package data. Only selected Python files
    # are replaced; omit cached bytecode so patched sources are recompiled.
    shutil.copytree(installed, overlay / "vllm", copy_function=os.symlink,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for patch in patches:
        source = Path(patch["snapshot"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != patch["sha256"]:
            raise RuntimeError(f"Patch snapshot changed: {source}")
        target = overlay / patch["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        shutil.copy2(source, target)
    os.environ["PYTHONPATH"] = str(overlay) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return {"installed_package": str(installed), "overlay": str(overlay), "patches": patches}


def check_model(path: str) -> None:
    root = Path(path)
    if not (root / "config.json").is_file():
        raise RuntimeError(f"Model config not readable inside container: {root}")
    indexes = list(root.glob("*.safetensors.index.json"))
    if indexes:
        for index in indexes:
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                if not (root / name).is_file():
                    raise RuntimeError(f"Missing shard or broken symlink: {root / name}")
    elif not list(root.glob("*.safetensors")):
        raise RuntimeError(f"No safetensors checkpoint found in {root}")


def main() -> None:
    config = json.loads(Path("/experiment/engine.json").read_text())
    deployment = json.loads(Path("/experiment/deployment.json").read_text())
    role, head = sys.argv[1:]
    if deployment.get("cache_dir"):
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["XDG_CACHE_HOME"] = "/cache"
        os.environ["VLLM_CACHE_ROOT"] = "/cache/vllm"
    else:
        os.environ.setdefault("HF_HOME", "/experiment/hf-cache")
    versions = {}
    for package in ("vllm", "torch", "agentx-harness", "aiperf", "flashinfer-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if role == "server":
        os.environ.update(config["env"])
        interface = deployment["network_interface"]
        if interface not in dict(socket.if_nameindex()).values():
            raise RuntimeError(f"Network interface {interface} not present")
        os.environ["NCCL_SOCKET_IFNAME"] = interface
        os.environ["GLOO_SOCKET_IFNAME"] = interface
        rank = int(os.environ["SLURM_PROCID"])
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True)
        if len(devices.splitlines()) != deployment["slurm"]["gpus_per_node"] or any("GB300" not in line for line in devices.splitlines()):
            raise RuntimeError(f"Expected four GB300 GPUs per tray, got: {devices}")
        for key in (("path", "draft_path") if config["speculation"] is not None else ("path",)):
            check_model(config["model"][key])
        overlay = patch_vllm(deployment["patches"])
        cmd = server_command(config, deployment, rank, head)
        metadata = {"versions": versions, "hostname": socket.gethostname(), "gpus": devices,
                    "vllm_source": overlay, "env": config["env"], "command": cmd}
        Path(f"/experiment/runtime-{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with Path(f"/experiment/gpu-topology-{rank}.txt").open("w") as output:
            subprocess.run(["nvidia-smi", "topo", "-m"], stdout=output, stderr=subprocess.STDOUT, check=True)
    elif role == "benchmark":
        bench = json.loads(Path("/experiment/benchmark.json").read_text())
        if bench.get("kind", "agentx") == "agentx":
            expected = bench["harness_version"]
            if versions.get("agentx-harness") != expected:
                raise RuntimeError(f"Expected agentx-harness {expected}; image contains {versions}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cmd = benchmark_command(config, bench, head)
        Path("/experiment/benchmark-runtime.json").write_text(json.dumps({"versions": versions, "command": cmd}, indent=2) + "\n")
    else:
        raise ValueError(f"Unknown worker role: {role}")
    print(json.dumps({"role": role, "command": cmd}), flush=True)
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
