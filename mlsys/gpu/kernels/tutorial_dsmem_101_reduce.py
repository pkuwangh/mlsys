"""Reduce FP32 vectors across CTAs using DSMEM and transaction barriers.

Each CTA starts with one partial vector. Peers send their partials to CTA 0,
which sums them using mapa + st.async + mbarrier, not global-memory atomics.

Command:
  python tutorial_dsmem_101_reduce.py
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python tutorial_dsmem_101_reduce.py
"""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

CLUSTER_SIZE = 4
THREADS = 32


@cute.kernel
def reduction_kernel(
    partials: cute.Tensor,
    out: cute.Tensor,
):
    """Each thread reduces one vector element across the cluster."""
    tid, _, _ = cute.arch.thread_idx()
    rank = cute.arch.block_idx_in_cluster()
    mailbox = cutlass.Array(
        cutlass.Float32,
        CLUSTER_SIZE * THREADS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)

    # Arm the receiver before any peer sends data. Count bytes, not stores.
    if tid == 0 and rank == 0:
        prims.mbarrier_init(ready, 1)
        # expect data from all other CTAs
        prims.mbarrier_arrive_expect_tx(ready, (CLUSTER_SIZE - 1) * THREADS * 4)
    prims.fence_mbarrier_init()

    # cluster barrier
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    value = partials[rank, tid]
    if rank != 0:
        # CUTLASS 4.8 lacks prims.st_async. This wrapper emits nvvm.mapa and
        # nvvm.store.async, mapping both pointers to CTA 0.
        # The store takes i32 bits, so bitcast FP32 rather than converting it.
        cute.arch.store_async_dsmem(
            cute.make_ptr(cutlass.Float32, mailbox.data_ptr(rank * THREADS + tid).ir_value()),
            value.bitcast(cutlass.Int32),
            cute.make_ptr(cutlass.Int64, ready.data_ptr().ir_value()),
            peer_cta_rank=0,
        )
    else:
        # st.async completes the expected bytes on the receiving barrier.
        while not prims.mbarrier_try_wait_parity(ready, 0):
            pass
        for sender in cutlass.range_constexpr(1, CLUSTER_SIZE):
            value = value + mailbox.load(idx=sender * THREADS + tid)
        out[0, tid] = value

    # No CTA exits while a peer might still access its shared memory.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()


@cute.jit
def reduction_host(
    partials: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
):
    """Launch one cluster with one partial vector per CTA."""
    reduction_kernel(partials, out).launch(
        grid=(CLUSTER_SIZE, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(CLUSTER_SIZE, 1, 1),
        stream=stream,
    )


def main():
    """Check the reduction and capture one replay for Nsight."""
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("DSMEM requires Hopper or newer")
    descriptor = make_fake_compact_tensor(cutlass.Float32, (CLUSTER_SIZE, THREADS), stride_order=(1, 0))
    partials = torch.arange(CLUSTER_SIZE * THREADS, dtype=torch.float32, device="cuda").reshape(CLUSTER_SIZE, THREADS)
    out = torch.empty((1, THREADS), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    compiled = cute.compile(
        reduction_host,
        descriptor,
        make_fake_compact_tensor(cutlass.Float32, (1, THREADS), stride_order=(1, 0)),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        compiled(partials, out, cuda.CUstream(stream.cuda_stream))

    print(f"{torch.cuda.get_device_name()} | {CLUSTER_SIZE} CTAs | {THREADS} FP32 elements/CTA")
    for seed in (0, 7, -100):
        partials.copy_(torch.arange(partials.numel(), dtype=torch.float32, device="cuda").reshape_as(partials) + seed)
        expected = partials.sum(dim=0, keepdim=True)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    print("Input partials (first four elements per CTA):")
    print(partials[:, :4].cpu())
    print("\ngather: correctness PASS | result on CTA 0")
    print(out[:, :4].cpu())

    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range("gather"):
            graph.replay()
            torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
