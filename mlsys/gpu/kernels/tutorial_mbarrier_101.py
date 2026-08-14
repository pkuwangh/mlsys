#!/usr/bin/env python3

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


@cute.kernel
def test_barrier_kernel(x: cute.Tensor):
    x_g = cutlass.make_array_view(x)

    # mbarrier in SMEM
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)

    if prims.elect_sync():
        prims.mbarrier_init(mbar, 1)

    prims.fence_mbarrier_init()
    prims.barrier_cta_sync()

    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        x_g[0] = 0.0
        cute.printf("PASS thread_idx={} x={}", tidx, x_g[0])
        prims.mbarrier_arrive(mbar)
        while not prims.mbarrier_try_wait_parity(mbar, 1, time_limit=10_000_000):
            pass
        x_g[0] = 2.0
        cute.printf("PASS thread_idx={} x={}", tidx, x_g[0])

    if tidx == 10:
        while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
            pass
        x_g[0] = 1.0
        cute.printf("PASS thread_idx={} x={}", tidx, x_g[0])
        prims.mbarrier_arrive(mbar)


@cute.jit
def test_barrier_host(x: cute.Tensor):
    test_barrier_kernel(x).launch(grid=(1, 1, 1), block=(32, 1, 1))


if __name__ == "__main__":
    # compile the kernel
    compiled = cute.compile(
        test_barrier_host,
        make_fake_compact_tensor(cutlass.Float32, (cutlass.sym_int64(), )),
    )
    # run it
    x_t = torch.zeros(1, dtype=torch.float32, device="cuda")
    compiled(x_t)
    print(x_t)
