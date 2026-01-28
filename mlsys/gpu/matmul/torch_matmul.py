#!/usr/bin/env python3

import os
import time

import torch


def test_matmul(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    num_iterations: int,
):
    # Create random matrices
    A = torch.randn(m, k, dtype=dtype).cuda()
    B = torch.randn(k, n, dtype=dtype).cuda()
    torch.cuda.synchronize()

    # warmup
    _WARMUP_ITERS = 100
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

    print(
        f"{C.shape=} {C.dtype=} m={m} k={k} n={n} dtype={dtype} tflops_1st={tflops_1st:.1f} tflops={tflops:.1f}",
        flush=True,
    )


# initialize cuda
torch.cuda.init()
torch.cuda.set_device(int(os.getenv("RANK", 0)))


# test
test_configs = {
    "without Tensor Cores": {
        "allow_tf32": False,
        "fp32_precision": "highest",
        "allow_fp16_reduced_precision": False,  # seems not working
        "dtypes": [torch.float32],
    },
    "with Tensor Cores": {
        "allow_tf32": True,
        "fp32_precision": "high",
        "allow_fp16_reduced_precision": True,
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
    },
}

def main() -> None:
    for name, config in test_configs.items():
        print(f"Matrix multiplication using torch.matmul {name}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
        torch.set_float32_matmul_precision(config["fp32_precision"])
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config["allow_fp16_reduced_precision"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["allow_fp16_reduced_precision"]
        for dtype in config["dtypes"]:
            test_matmul(4096, 8192, 8192, dtype, 100)


if __name__ == "__main__":
    main()

    if os.getenv("RANK", None) is not None:
        while True:
            time.sleep(0.1)
            main()

