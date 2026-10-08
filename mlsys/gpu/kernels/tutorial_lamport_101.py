"""Pass data between CTAs using sentinel-based Lamport synchronization.

CTA 0 produces chunks; CTA 1 consumes each chunk once its values change from
-1 to actual data. The data itself is the ready signal, with no separate flag.
Compare waiting for all chunks at a barrier with consuming them as they arrive.
Two CTAs share a cluster to guarantee co-scheduling.

Command:
  python tutorial_lamport_101.py
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python tutorial_lamport_101.py
"""

import statistics

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

PRODUCE_US = 20
CONSUME_US = 20

CHUNKS = 4
THREADS = 32
SENTINEL = -1
REPLAYS = 5
SAMPLES = 3
MODES = ("barrier", "lamport")


@cute.jit
def delay(ns: cutlass.Int64):
    """Stand in for a chunk of computation."""
    start = cute.arch.globaltimer()
    while cute.arch.globaltimer() - start < ns:
        prims.nanosleep(100)


@cute.kernel
def handoff_kernel(
    x: cute.Tensor,
    mailbox: cute.Tensor,
    out: cute.Tensor,
    stamps: cute.Tensor,
    produce_ns: cutlass.Int64,
    consume_ns: cutlass.Int64,
    LAMPORT: cutlass.Constexpr[bool],
):
    """Publish x + 1 from CTA 0; consume and double it in CTA 1."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    data = cutlass.make_array_view(mailbox)
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    if bid == 0:
        for chunk in range(CHUNKS):
            index = chunk * THREADS + tid
            if tid == 0:
                stamps[chunk, 0] = cute.arch.globaltimer()
            delay(produce_ns)
            value = x[index] + 1
            if tid == 0:
                stamps[chunk, 1] = cute.arch.globaltimer()
            # publish the data with release ordering, i.e. fence before the store
            prims.store_ext(value.ir_value(), data.subview(index), order="release", scope="gpu")
            prims.barrier_cta_sync()

    if cutlass.const_expr(not LAMPORT):
        # Baseline: the consumer waits for the entire producer loop.
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

    if bid == 1:
        for chunk in range(CHUNKS):
            index = chunk * THREADS + tid
            value = cutlass.Int32(SENTINEL)
            # Each lane checks its own payload, not just the first element.
            while value == SENTINEL:
                # load the data with acquire ordering, i.e. fence after the load
                value = prims.load_ext(
                    data.subview(index),
                    dtype=cutlass.Int32,
                    order="acquire",
                    scope="gpu",
                )
                if value == SENTINEL:
                    prims.nanosleep(100)
            # sync all lanes
            prims.barrier_cta_sync()
            if tid == 0:
                stamps[chunk, 2] = cute.arch.globaltimer()
            delay(consume_ns)
            out[index] = value * 2
            if tid == 0:
                stamps[chunk, 3] = cute.arch.globaltimer()
            prims.barrier_cta_sync()


@cute.jit
def handoff_host(
    x: cute.Tensor,
    mailbox: cute.Tensor,
    out: cute.Tensor,
    stamps: cute.Tensor,
    produce_ns: cutlass.Int64,
    consume_ns: cutlass.Int64,
    stream: cuda.CUstream,
    LAMPORT: cutlass.Constexpr[bool],
):
    """Launch one producer CTA and one consumer CTA in the same cluster."""
    handoff_kernel(x, mailbox, out, stamps, produce_ns, consume_ns, LAMPORT).launch(
        grid=(2, 1, 1), block=(THREADS, 1, 1), cluster=(2, 1, 1), stream=stream
    )


def main():
    """Check the handoff, compare timings, and show the per-chunk timeline."""
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("This two-CTA cluster example requires Hopper or newer")
    tensor = make_fake_compact_tensor(cutlass.Int32, (CHUNKS * THREADS,))
    timestamps = make_fake_compact_tensor(cutlass.Int64, (CHUNKS, 4), stride_order=(1, 0))
    x = torch.arange(CHUNKS * THREADS, dtype=torch.int32, device="cuda")
    mailbox, out = torch.empty_like(x), torch.empty_like(x)
    stamps = torch.empty((CHUNKS, 4), dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    graphs = {}
    for mode in MODES:
        compiled = cute.compile(
            handoff_host,
            tensor,
            tensor,
            tensor,
            timestamps,
            cutlass.Int64(0),
            cutlass.Int64(0),
            make_fake_stream(),
            mode == "lamport",
            options="--enable-tvm-ffi",
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            # Re-arm before every replay, after the previous consumer finishes.
            mailbox.fill_(SENTINEL)
            compiled(
                x,
                mailbox,
                out,
                stamps,
                int(PRODUCE_US * 1000),
                int(CONSUME_US * 1000),
                cuda.CUstream(stream.cuda_stream),
            )
        graphs[mode] = graph

    print(f"{torch.cuda.get_device_name()} | {CHUNKS} chunks | produce={PRODUCE_US} us consume={CONSUME_US} us")
    for mode, graph in graphs.items():
        for seed in (0, 100, 1000):
            x.copy_(torch.arange(x.numel(), dtype=torch.int32, device="cuda") + seed)
            out.fill_(-9999)
            graph.replay()
            torch.testing.assert_close(out, (x + 1) * 2, rtol=0, atol=0)
        print(f"{mode}: correctness PASS", flush=True)

    timings = {mode: [] for mode in MODES}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(SAMPLES):
        for mode in MODES if sample % 2 == 0 else reversed(MODES):
            start.record()
            for _ in range(REPLAYS):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            timings[mode].append(start.elapsed_time(end) * 1000 / REPLAYS)

    for mode, graph in graphs.items():
        graph.replay()
        rows = stamps.cpu().tolist()
        origin = rows[0][0]
        print(f"\n{mode}: {statistics.median(timings[mode]):.2f} us/replay (including sentinel reset)")
        print("Chunk | Producer work (us) | Consumer work (us)")
        for chunk, row in enumerate(rows):
            p0, p1, c0, c1 = [(t - origin) / 1000 for t in row]
            print(f"{chunk:5} | {p0:7.2f} - {p1:7.2f} | {c0:7.2f} - {c1:7.2f}")

    # Nsight shows one kernel per mode; the table shows overlap inside it.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        for mode, graph in graphs.items():
            with torch.cuda.nvtx.range(mode):
                graph.replay()
                torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
