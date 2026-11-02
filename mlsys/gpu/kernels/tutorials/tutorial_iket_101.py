"""Trace a two-warp producer/consumer pipeline, without TMA or MMA setup.

warp 0:
  - waits empty[slot]                           - iket range: wait_empty
  - loads into s_ring[32*slot : 32*(slot+1)]    - iket range: load
  - sets full[slot]
warp 1:
  - waits full[slot]                            - iket range: wait_full
  - reads s_ring[32*slot : 32*(slot+1)]         - iket range: compute
  - sets empty[slot]

Run:
  python tutorial_iket_101.py

Open the emitted .pftrace in https://ui.perfetto.dev/.
"""

import argparse

import cutlass
import torch
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


@cute.kernel
def pipeline_kernel(x: cute.Tensor, out: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tid % 32
    warp = cute.arch.warp_idx()
    ring = cutlass.Array(cutlass.Float32, 64, space=cutlass.AddressSpace.smem)
    bars = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    full, empty = bars, bars.subview(2)
    if tid == 0:
        for i in cutlass.range_constexpr(4):
            # Every lane arrives, so all stores/reads precede slot handoff.
            prims.mbarrier_init(bars.subview(i), 32)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync()

    lifetime = iket.range_start("pipeline")
    for tile in range(x.shape[1]):
        slot = tile % 2
        phase = (tile // 2) % 2
        if warp == 0:
            iket.range_push("producer.wait_empty")
            # Phase 1 is already satisfied on first use of an empty slot.
            while not prims.mbarrier_try_wait_parity(empty.subview(slot), phase ^ 1):
                pass
            iket.range_pop()
            iket.range_push("producer.load")
            ring.store(x[block, tile, lane], idx=slot * 32 + lane)
            prims.mbarrier_arrive(full.subview(slot))
            iket.range_pop()
        else:
            iket.range_push("consumer.wait_full")
            while not prims.mbarrier_try_wait_parity(full.subview(slot), phase):
                pass
            iket.range_pop()
            iket.range_push("consumer.compute")
            out[block, tile, lane] = ring.load(idx=slot * 32 + lane) * 2 + 1
            prims.mbarrier_arrive(empty.subview(slot))
            iket.range_pop()
    iket.range_end(lifetime)
    # Both warps finish before shared-memory storage can be released.
    prims.barrier_cta_sync()


@cute.jit
def pipeline_host(x, out):
    pipeline_kernel(x, out).launch(grid=(x.shape[0], 1, 1), block=(64, 1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=8)
    args = parser.parse_args()
    if args.tiles < 1:
        parser.error("--tiles must be positive")
    torch.manual_seed(42)
    x = torch.randn(4, args.tiles, 32, device="cuda")
    out = torch.empty_like(x)
    descriptor = make_fake_compact_tensor(cutlass.Float32, x.shape, stride_order=(2, 1, 0))
    compiled = cute.compile(pipeline_host, descriptor, descriptor, options="--enable-tvm-ffi")
    compiled(x, out)
    torch.testing.assert_close(out, x * 2 + 1, atol=0, rtol=0)
    print(f"PASS: {torch.cuda.get_device_name()} | 4 CTAs, 2 warps, {args.tiles} tiles | out = 2*x+1")


if __name__ == "__main__":
    main()
