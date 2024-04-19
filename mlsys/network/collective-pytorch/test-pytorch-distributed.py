#!/usr/bin/env python3

import argparse
import os
import time
import torch
import torch.distributed

from datetime import timedelta
from typing import Tuple


def get_data_size_str(data_size):
    if data_size < 1024:
        data_size_str = f"{data_size:4d}  B"
    elif data_size < 1024 * 1024:
        data_size_str = f"{data_size // 1024:4d} KB"
    elif data_size < 1024 * 1024 * 1024:
        data_size_str = f"{data_size // 1024 // 1024:4d} MB"
    else:
        data_size_str = f"{data_size // 1024 // 1024 // 1024:4d} GB"
    return f"{data_size_str:<7s}"


def get_bw_str(bw_in_gbyte):
    if bw_in_gbyte < 1:
        bw_str = f"{bw_in_gbyte * 1024:4.0f} MB/s"
    else:
        bw_str = f"{bw_in_gbyte:4.0f} GB/s"
    return f"{bw_str:<9s}"


def get_time_str(elapsed_time):
    if elapsed_time >= 1:
        time_str = f"{elapsed_time:4.1f}  s"
    elif elapsed_time >= 0.001:
        time_str = f"{elapsed_time * 1000:4.0f} ms"
    else:
        time_str = f"{elapsed_time * 1000:4.2f} ms"
    return f"{time_str:<7s}"


def dump_bandwidth(
    group,
    test_name,
    data_size,
    num_iters,
    elapsed_time,
    algo_bandwidth,
    bus_bandwidth,
):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            f"{torch.distributed.get_backend(group)} {test_name:<14s}:"
            f"  data_size={get_data_size_str(data_size)},"
            f"  num_iters={num_iters:>4d},"
            f"  time={elapsed_time:4.1f}s,"
            f"  Alog-BW={get_bw_str(algo_bandwidth)},"
            f"  Bus-BW={get_bw_str(bus_bandwidth)}"
        )


def print_rank_0(line):
    if torch.distributed.get_rank() == 0:
        print(line)


def get_basic_info() -> Tuple[int, int]:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return (rank, world_size)


def start() -> float:
    torch.distributed.barrier()
    return time.time()


def end(start_time: float, data_size: int, num_iters: int) -> Tuple[float, float]:
    torch.distributed.barrier()
    elapsed_time = time.time() - start_time
    bandwidth = data_size * num_iters / elapsed_time / 1e9
    return (elapsed_time, bandwidth)


def test_sendrecv(group, tensor, data_size, num_iters):
    (rank, world_size) = get_basic_info()
    half_world_size = world_size // 2
    start_time = start()
    for _ in range(num_iters):
        if rank < half_world_size:
            torch.distributed.send(tensor, dst=(rank + half_world_size), group=group)
        else:
            torch.distributed.recv(tensor, src=(rank - half_world_size), group=group)
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    dump_bandwidth(group, "SendRecv", data_size, num_iters, elapsed_time, bandwidth, bandwidth)


def test_gather(group, tensor, data_size, num_iters):
    (rank, world_size) = get_basic_info()
    send_tensor = tensor
    if rank == 0:
        recv_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        recv_tensors = None
    start_time = start()
    for _ in range(num_iters):
        torch.distributed.gather(send_tensor, recv_tensors, dst=0, group=group)
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    bus_bandwidth = bandwidth
    dump_bandwidth(group, "Gather", data_size, num_iters, elapsed_time, bandwidth, bus_bandwidth)


def test_all_gather(group, tensor, data_size, num_iters):
    (rank, world_size) = get_basic_info()
    send_tensor = tensor
    recv_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    start_time = start()
    for _ in range(num_iters):
        torch.distributed.all_gather(recv_tensors, send_tensor, group=group)
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    bus_bandwidth = bandwidth * (world_size - 1) / world_size
    dump_bandwidth(group, "AllGather", data_size, num_iters, elapsed_time, bandwidth, bus_bandwidth)


def test_reduce(group, tensor, data_size, num_iters):
    (rank, world_size) = get_basic_info()
    start_time = start()
    for _ in range(num_iters):
        torch.distributed.reduce(tensor, dst=0, op=torch.distributed.ReduceOp.SUM, group=group)
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    bus_bandwidth = bandwidth
    dump_bandwidth(group, "Reduce", data_size, num_iters, elapsed_time, bandwidth, bus_bandwidth)


def test_reduce_scatter(group, tensor, data_size, num_iters):
    if torch.distributed.get_backend(group) == "gloo":
        return
    (rank, world_size) = get_basic_info()
    tensor_size = list(tensor.size())
    chunk_size = [x for x in tensor_size[:-1]] + [tensor_size[-1] // world_size]
    input = [torch.randn(chunk_size).cuda() for _ in range(world_size)]
    output = torch.empty_like(input[0])
    start_time = start()
    for _ in range(num_iters):
        torch.distributed.reduce_scatter(
            output, input, op=torch.distributed.ReduceOp.SUM, group=group
        )
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    bus_bandwidth = bandwidth * (world_size - 1) / world_size
    dump_bandwidth(
        group, "ReduceScatter", data_size, num_iters, elapsed_time, bandwidth, bus_bandwidth
    )


def test_all_reduce(group, tensor, data_size, num_iters):
    (rank, world_size) = get_basic_info()
    start_time = start()
    for _ in range(num_iters):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    (elapsed_time, bandwidth) = end(start_time, data_size, num_iters)
    bus_bandwidth = bandwidth * 2 * (world_size - 1) / world_size
    dump_bandwidth(group, "AllReduce", data_size, num_iters, elapsed_time, bandwidth, bus_bandwidth)


def test_all(args, groups):
    world_size = torch.distributed.get_world_size()
    size_and_iters = {}
    tensor_n = 1024 * 1024
    num_iters = 8
    while tensor_n > 0:
        size_and_iters[tensor_n] = num_iters
        tensor_n //= 32
        num_iters = num_iters * 32 // 8
    # print(size_and_iters)
    tests = [
        {"func": test_sendrecv, "data_size_scale": 1},
        {"func": test_gather, "data_size_scale": world_size},
        {"func": test_all_gather, "data_size_scale": world_size},
        {"func": test_reduce, "data_size_scale": 1},
        {"func": test_reduce_scatter, "data_size_scale": 1},
        {"func": test_all_reduce, "data_size_scale": 1},
    ]
    for group_info in groups:
        group = group_info["group"]
        group_iter_scale = group_info["group_iter_scale"]
        for test_info in tests:
            for num in sorted(size_and_iters.keys()):
                data = torch.randn(num // test_info["data_size_scale"], 1024)
                if torch.distributed.get_backend(group) == "nccl":
                    tensor = data.cuda()
                else:
                    tensor = data
                # un-scaled size
                data_size = 1024 * num * tensor.element_size()
                # iterations
                num_iters = size_and_iters[num] / group_iter_scale
                if num_iters < 0.01:
                    continue
                num_iters = max(1, int(num_iters))
                # test
                test_info["func"](group, tensor, data_size, num_iters)
            print_rank_0("----")
        print_rank_0("====")


def main(args):
    # initialize process group
    MY_RANK = os.environ.get("RANK", None)
    MY_WORLD_SIZE = os.environ.get("WORLD_SIZE", None)
    # nccl group
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(MY_WORLD_SIZE),
        rank=int(MY_RANK),
    )
    # gloo group
    gloo_group = torch.distributed.new_group(
        ranks=list(range(int(MY_WORLD_SIZE))),
        timeout=timedelta(seconds=300),
        backend="gloo",
    )
    groups = [
        {
            "group": torch.distributed.group.WORLD,
            "group_iter_scale": 1,
        },
        {
            "group": gloo_group,
            "group_iter_scale": 350,
        },
    ]
    # run tests for different msg sizes and comm patterns
    test_all(args, groups)
    # cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    main(args)
