#!/usr/bin/env python3

import torch
import time


def test_matmul(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    num_iterations: int,
):
    # Create random matrices
    A = torch.randn(m, k).cuda().to(dtype)
    B = torch.randn(k, n).cuda().to(dtype)
    torch.cuda.synchronize()

    # warmup
    _WARMUP_ITERS = 1
    for _ in range(_WARMUP_ITERS):
        C = torch.matmul(A, B)
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

    n_flop_per_iter = (m * k * n) * 2
    tflops_1st = n_flop_per_iter / (end_1st - start_1st) / 1e12
    tflops = num_iterations * n_flop_per_iter / (end_iters - end_1st) / 1e12

    print(f"{C.shape=} {C.dtype=} m={m} k={k} n={n} dtype={dtype} tflops_1st={tflops_1st:.1f} tflops={tflops:.1f}", flush=True)


# initialize cuda
torch.cuda.init()
torch.cuda.set_device(0)

# check configuration
print(f"{torch.backends.cuda.matmul.allow_tf32=}")
print(f"{torch.get_float32_matmul_precision()=}")
print(f"{torch.are_deterministic_algorithms_enabled()=}")

# TF32 has slight lower precision than FP32 but can be run on Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
# Even TF32 is allowed, `highest` precision will still prevent Tensor Cores from being used
torch.set_float32_matmul_precision('high')

# test
print("Matrix multiplication using torch.matmul", flush=True)
test_matmul(4096, 8192, 8192, torch.float32, 100)
test_matmul(4096, 8192, 8192, torch.float16, 100)
test_matmul(4096, 8192, 8192, torch.bfloat16, 100)
