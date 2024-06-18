#!/usr/bin/env python3

import torch
import time

from utils import get_device_properties, print_flops


def test_nn_linear(m, n, p, num_iterations, warmup=False):
    net = torch.nn.Linear(n, p).cuda().half()
    input = torch.randn(m, n).cuda().half()
    torch.cuda.synchronize()

    # measure 1st iteration
    start_1st = time.time()
    output = net(input)
    torch.cuda.synchronize()

    end_1st = time.time()

    for _ in range(num_iterations):
        output = net(input)
    torch.cuda.synchronize()
    end_iters = time.time()
    print(f"output shape {output.shape}")

    if warmup:
        return

    n_flop_per_iter = (m * n * p) * 2
    tflops_1st = n_flop_per_iter / (end_1st - start_1st) / 1e12
    tflops = num_iterations * n_flop_per_iter / (end_iters - end_1st) / 1e12

    print(f"m={m} n={n} p={p}")
    if get_device_properties().get("tensor_core_count", 0) > 0:
        print_flops(tflops_1st, "tc_fp16_flops", prefix="1st run")
        print_flops(tflops, "tc_fp16_flops", prefix="rep run")
    else:
        print_flops(tflops_1st, "fp16_flops", prefix="1st run")
        print_flops(tflops, "fp16_flops", prefix="rep run")

test_nn_linear(8192, 8192, 8192, 10, warmup=True)
print("Matrix multiplication using torch.nn.Linear", flush=True)
test_nn_linear(512, 512, 512, 100000)
test_nn_linear(1024, 1024, 1024, 10000)
test_nn_linear(2048, 2048, 2048, 10000)
test_nn_linear(4096, 4096, 4096, 1000)
test_nn_linear(8192, 8192, 8192, 100)
test_nn_linear(16384, 16384, 16384, 10)
