#!/usr/bin/env python3

import torch
import time

from utils import get_device_properties, print_flops


def test_matmul(
    m: int,
    n: int,
    p: int,
    num_iterations: int,
    device_props: dict[str, float | int | str],
    warmup=False,
):
    # Create random matrices
    A = torch.randn(m, n).cuda().half()
    B = torch.randn(n, p).cuda().half()
    torch.cuda.synchronize()

    # measure 1st iteration
    start_1st = time.time()
    C = torch.matmul(A, B)
    torch.cuda.synchronize()

    end_1st = time.time()

    for _ in range(num_iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end_iters = time.time()
    print(f"output shape {C.shape}")

    if warmup:
        return

    n_flop_per_iter = (m * n * p) * 2
    tflops_1st = n_flop_per_iter / (end_1st - start_1st) / 1e12
    tflops = num_iterations * n_flop_per_iter / (end_iters - end_1st) / 1e12

    print(f"m={m} n={n} p={p}")
    if device_props.get("tensor_core_count", 0) > 0:
        print_flops(tflops_1st, "tc_fp16_flops", device_pros, prefix="1st run")
        print_flops(tflops, "tc_fp16_flops", device_pros, prefix="rep run")
    else:
        print_flops(tflops_1st, "fp16_flops", device_pros, prefix="1st run")
        print_flops(tflops, "fp16_flops", device_pros, prefix="rep run")


device_pros = get_device_properties(verbose=1)

test_matmul(8192, 8192, 8192, 10, device_pros, warmup=True)
print("Matrix multiplication using torch.matmul", flush=True)
test_matmul(512, 512, 512, 100000, device_pros)
test_matmul(1024, 1024, 1024, 10000, device_pros)
test_matmul(2048, 2048, 2048, 10000, device_pros)
test_matmul(4096, 4096, 4096, 1000, device_pros)
test_matmul(8192, 8192, 8192, 100, device_pros)
test_matmul(16384, 16384, 16384, 10, device_pros)
