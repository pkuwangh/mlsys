#!/usr/bin/env python3

import argparse
import os
import time
import torch
import torch.distributed


def get_msg_size_str(msg_size):
    if msg_size < 1024:
        return f"{msg_size}B"
    elif msg_size < 1024 * 1024:
        return f"{msg_size // 1024}KB"
    elif msg_size < 1024 * 1024 * 1024:
        return f"{msg_size // 1024 // 1024}MB"
    else:
        return f"{msg_size // 1024 // 1024 // 1024}GB"


def test_sendrecv(data, msg_size, num_iters):
    rank = torch.distributed.get_rank()
    half_world_size = torch.distributed.get_world_size() // 2
    torch.distributed.barrier()
    start_time = time.time()
    for _ in range(num_iters):
        if rank < half_world_size:
            torch.distributed.send(data, dst=(rank + half_world_size))
        else:
            torch.distributed.recv(data, src=(rank - half_world_size))
    torch.distributed.barrier()
    end_time = time.time()
    bandwidth = msg_size * num_iters / (end_time - start_time) / 1e9
    print(f"SendRecv: msg_size={get_msg_size_str(msg_size)} BW={bandwidth} GB/s")


def test_all(args):
    for num in [1, 32, 1024, 32 * 1024, 1024 * 1024]:
        data = torch.randn(1024, num).cuda()
        msg_size = data.numel() * data.element_size()
        # print(f"Testing msg size {msg_size}Byte ...")
        # send-recv
        test_sendrecv(data, msg_size, (1024 * 1024 // num))


def main(args):
    # initialize process group
    MY_RANK = os.environ.get("RANK", None)
    MY_WORLD_SIZE = os.environ.get("WORLD_SIZE", None)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(MY_WORLD_SIZE),
        rank=int(MY_RANK),
    )
    # run tests for different msg sizes and comm patterns
    test_all(args)
    # cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    main(args)
