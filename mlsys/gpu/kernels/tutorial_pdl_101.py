"""Overlap consecutive kernels with Programmatic Dependent Launch (PDL).

PDL lets a kernel begin independent work before the previous kernel finishes,
then wait before reading its output.

Each kernel prepares independently, waits for its input, does dependent work,
then signals the next kernel before finishing and writing its output.

Command:
  python tutorial_pdl_101.py
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python tutorial_pdl_101.py
"""

import statistics

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

INDEPENDENT_US = 10
DEPENDENT_US = 20
FINISH_US = 10

STAGES = 16
THREADS = 128
BLOCKS = 8
REPLAYS = 5
SAMPLES = 3
MODES = ("serial", "pdl")


@cute.jit
def delay(ns: cutlass.Int64):
    """Hold a controlled phase open without depending on another grid."""
    start = cute.arch.globaltimer()
    while cute.arch.globaltimer() - start < ns:
        prims.nanosleep(100)


@cute.kernel
def stage_kernel(
    x: cute.Tensor,
    out: cute.Tensor,
    stamps: cute.Tensor,
    independent_ns: cutlass.Int64,
    dependent_ns: cutlass.Int64,
    finish_ns: cutlass.Int64,
    PDL: cutlass.Constexpr[bool],
):
    """Do independent work, wait before reading input, and release the next grid."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    # Phase 1: preparation that does not need the previous kernel's output.
    if tid == 0:
        stamps[bid, 0] = cute.arch.globaltimer()
    delay(independent_ns)
    if tid == 0:
        stamps[bid, 1] = cute.arch.globaltimer()
    if cutlass.const_expr(PDL):
        # Wait for the previous kernel to finish and make its output visible.
        prims.griddepcontrol("wait")

    # Phase 2: consume the input and work before releasing the next kernel.
    if tid == 0:
        stamps[bid, 2] = cute.arch.globaltimer()
    value = x[bid * THREADS + tid]
    delay(dependent_ns)
    # as the producer kernel, signal dependent kernel can start as long as scheduling allows
    prims.griddepcontrol("launch_dependents")

    # Phase 3: finish while the next kernel can begin its preparation.
    if tid == 0:
        stamps[bid, 3] = cute.arch.globaltimer()
    delay(finish_ns)
    out[bid * THREADS + tid] = value + 1

    if tid == 0:
        stamps[bid, 4] = cute.arch.globaltimer()


@cute.jit
def stage_host(
    x: cute.Tensor,
    out: cute.Tensor,
    stamps: cute.Tensor,
    independent_ns: cutlass.Int64,
    dependent_ns: cutlass.Int64,
    finish_ns: cutlass.Int64,
    stream: cuda.CUstream,
    PDL: cutlass.Constexpr[bool],
):
    """Opt each consumer launch into programmatic stream serialization."""
    stage_kernel(x, out, stamps, independent_ns, dependent_ns, finish_ns, PDL).launch(
        grid=(stamps.shape[0], 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=PDL,
    )


class Chain:
    """Consecutive kernels on one stream, captured in a CUDA graph."""

    def __init__(self, mode, compiled):
        self.mode = mode
        self.values = torch.empty((STAGES + 1, BLOCKS * THREADS), dtype=torch.int32, device="cuda")
        self.stamps = torch.empty((STAGES, BLOCKS, 5), dtype=torch.int64, device="cuda")
        self.stream = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()
        launch = compiled[mode == "pdl"]
        stream = cuda.CUstream(self.stream.cuda_stream)

        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=self.stream):
            for index in range(STAGES):
                launch(
                    self.values[index],
                    self.values[index + 1],
                    self.stamps[index],
                    int(INDEPENDENT_US * 1000),
                    int(DEPENDENT_US * 1000),
                    int(FINISH_US * 1000),
                    stream,
                )

    def validate(self):
        """Check every stage with changing inputs across repeated graph replays."""
        for seed in (7, -19, 113):
            self.values.fill_(-9999)
            self.values[0].fill_(seed)
            self.graph.replay()
            expected = torch.arange(STAGES + 1, device="cuda", dtype=torch.int32) + seed
            torch.testing.assert_close(self.values, expected[:, None].expand_as(self.values), rtol=0, atol=0)

    def overlap(self):
        """Measure overlap between consecutive stages using CTA-0 timestamps."""
        self.graph.replay()
        stamps = self.stamps.cpu().tolist()
        overlaps = []
        for index in range(STAGES - 1):
            producer = stamps[index][0]
            consumer = stamps[index + 1][0]
            overlaps.append(max(0, min(producer[4], consumer[1]) - max(producer[3], consumer[0])) / 1000)
        return sum(value > 0 for value in overlaps), statistics.mean(overlaps)


def main():
    """Check results, compare timings, and mark one replay per mode for Nsight."""
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("PDL requires Hopper or newer")
    tensor = make_fake_compact_tensor(cutlass.Int32, (BLOCKS * THREADS,))
    stamps = make_fake_compact_tensor(cutlass.Int64, (BLOCKS, 5), stride_order=(1, 0))
    compiled = {
        pdl: cute.compile(
            stage_host,
            tensor,
            tensor,
            stamps,
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            make_fake_stream(),
            pdl,
            options="--enable-tvm-ffi",
        )
        for pdl in (False, True)
    }
    chains = [Chain(mode, compiled) for mode in MODES]
    print(
        f"{torch.cuda.get_device_name()} | {STAGES} stages | {BLOCKS} CTAs | "
        f"independent={INDEPENDENT_US:g} us dependent={DEPENDENT_US:g} us finish={FINISH_US:g} us",
        flush=True,
    )
    for chain in chains:
        chain.validate()
        print(f"{chain.mode}: correctness PASS", flush=True)

    timings = {mode: [] for mode in MODES}
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for sample in range(SAMPLES):
        for chain in chains if sample % 2 == 0 else reversed(chains):
            start.record()
            for _ in range(REPLAYS):
                chain.graph.replay()
            end.record()
            end.synchronize()
            timings[chain.mode].append(start.elapsed_time(end) * 1000 / REPLAYS)

    print("\nMode             | Graph us | vs serial | Early boundaries | Overlap us (CTA 0)")
    baseline = statistics.median(timings["serial"])
    for chain in chains:
        us = statistics.median(timings[chain.mode])
        count, overlap = chain.overlap()
        print(f"{chain.mode:16} | {us:8.2f} | {baseline / us:8.3f}x | {count:2}/{STAGES - 1:<13} | {overlap:8.2f}")
    print("\nOverlap is the mean intersection of successor preparation and predecessor finishing phase.")

    # Nsight captures only these replays, excluding compilation and validation.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        for chain in chains:
            with torch.cuda.nvtx.range(chain.mode):
                chain.graph.replay()
                torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
