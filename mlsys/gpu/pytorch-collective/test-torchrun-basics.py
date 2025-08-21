#!/usr/bin/env python3

import os
import socket
import torch

def get_hostname() -> str:
    return socket.gethostname()

def get_pid() -> int:
    return os.getpid()

def get_world_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0))
    )

def get_gpu_info() -> tuple[int, int]:
    if torch.cuda.is_available():
        return torch.cuda.device_count(), torch.cuda.current_device()
    else:
        return 0, -1

if __name__ == "__main__":
    hostname = get_hostname()
    pid = get_pid()
    print(f"starting on {hostname} pid={pid}", flush=True)

    world_size, rank, local_rank = get_world_info()
    gpu_count, gpu_index = get_gpu_info()

    print(f"host={hostname} pid={pid} world_size={world_size} rank={rank} local_rank={local_rank}: {gpu_count} GPUs", flush=True)
