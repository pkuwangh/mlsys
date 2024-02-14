#!/usr/bin/env python3

import torch
import time

from utils import print_flops

def test_matmul(m, n, p, num_iterations, warmup=False):
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

    if warmup:
        return

    n_flop_per_iter = (m * n * p) * 2
    tflops_1st = n_flop_per_iter / (end_1st - start_1st) / 1e12
    tflops = num_iterations * n_flop_per_iter / (end_iters - end_1st) / 1e12

    print(f"m={m} n={n} p={p})")
    print_flops(tflops_1st, "tc_fp16_flops", prefix="1st run")
    print_flops(tflops, "tc_fp16_flops", prefix="rep run")


test_matmul(8192, 8192, 8192, 10, warmup=True)
print("Matrix multiplication using torch.matmul", flush=True)
test_matmul(512, 512, 512, 10000)
test_matmul(1024, 1024, 1024, 1000)
test_matmul(2048, 2048, 2048, 1000)
test_matmul(4096, 4096, 4096, 100)
test_matmul(8192, 8192, 8192, 10)
