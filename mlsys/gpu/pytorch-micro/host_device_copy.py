#!/usr/bin/env python3

import torch
import time


def test_host_device_copy(m, n, k, num_iterations, devices, warmup=False):
    # Create random matrices
    A = torch.randn(m, n, k, device=devices[0])
    B = torch.randn(m, n, k, device=devices[1])
    tensor_size = A.nelement() * A.element_size()
    if warmup:
        print(f"tensor storage size: {tensor_size / (1 << 30)} GB")

    # measure 1st iteration
    start_1st = time.time()
    B.copy_(A, non_blocking=False)
    if devices[0].startswith("cuda") or devices[1].startswith("cuda"):
        torch.cuda.synchronize()

    end_1st = time.time()

    for _ in range(num_iterations):
        B.copy_(A, non_blocking=False)
    if devices[0].startswith("cuda") or devices[1].startswith("cuda"):
        torch.cuda.synchronize()
    end_iters = time.time()

    if warmup:
        return

    bw_1st = tensor_size / (end_1st - start_1st) / (1 << 30)
    bw = num_iterations * tensor_size / (end_iters - end_1st) / (1 << 30)
    print(f"bandwidth, 1st run: {bw_1st:.1f} GB/s")
    print(f"bandwidth, rep run: {bw:.1f} GB/s")


test_host_device_copy(1024, 1024, 1024, 1, ["cpu", "cuda:0"], warmup=True)
print("\nHost-to-Host copy", flush=True)
test_host_device_copy(1024, 1024, 1024, 10, ["cpu", "cpu"], warmup=True)
test_host_device_copy(1024, 1024, 1024, 100, ["cpu", "cpu"])
print("\nHost-to-Device copy", flush=True)
test_host_device_copy(1024, 1024, 1024, 10, ["cpu", "cuda:0"], warmup=True)
test_host_device_copy(1024, 1024, 1024, 100, ["cpu", "cuda:0"])
print("\nDevice-to-Host copy", flush=True)
test_host_device_copy(1024, 1024, 1024, 10, ["cuda:0", "cpu"], warmup=True)
test_host_device_copy(1024, 1024, 1024, 100, ["cuda:0", "cpu"])
print("\nDevice-to-Device copy", flush=True)
test_host_device_copy(1024, 1024, 1024, 10, ["cuda:0", "cuda:0"], warmup=True)
test_host_device_copy(1024, 1024, 1024, 100, ["cuda:0", "cuda:0"])
