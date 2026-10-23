"""Small shared result type and CUDA-graph timing for standalone Wan kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import torch


@dataclass(frozen=True)
class PreparedConvInput:
    """Owned NTHWC next-convolution input, temporal history, and optional skip."""

    padded: torch.Tensor
    cache: torch.Tensor
    residual: torch.Tensor | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Median GPU milliseconds for an operation and its named reference."""

    operation: str
    custom_ms: float
    reference_ms: float

    @property
    def speedup(self) -> float:
        return self.reference_ms / self.custom_ms


def measure(
    label: str,
    reference: Callable,
    custom: Callable,
    *,
    samples: int = 8,
    replays: int = 10,
) -> BenchmarkResult:
    """Measure paired median GPU times, excluding Python and JIT compilation.

    Warm both functions before graph capture. Alternate replay order across
    samples to reduce clock drift bias; retain graph outputs throughout.
    Both callables must be inference-only and use the current CUDA device.
    """
    ref_ms, custom_ms = measure_group((reference, custom), samples=samples, replays=replays)
    return BenchmarkResult(label, custom_ms, ref_ms)


def measure_group(
    functions: tuple[Callable, ...], *, samples: int = 20, replays: int = 10, warmup_ms: float = 500
) -> tuple[float, ...]:
    """Time related paths together, rotating replay order to reduce clock drift.

    Each path gets its own CUDA graph and retained outputs. Compilation and
    allocation during capture are excluded. Choose a multiple of the number
    of paths for samples to give each path every position equally often.
    Condition all captured paths before timing so JIT idle periods do not
    leave the first samples at idle clocks. Warmup is excluded from samples.
    Replay each selected path twice before its sample to condition its own
    cache footprint after switching between paths with different buffers.
    """
    if not functions or samples < 1 or replays < 1 or warmup_ms < 0:
        raise ValueError("Expected nonempty functions, positive samples/replays and nonnegative warmup")
    graphs = []
    outputs = []
    for fn in functions:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(fn())
        graphs.append(graph)
    times = [[] for _ in functions]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if warmup_ms:
        start.record()
        elapsed = 0.0
        while elapsed < warmup_ms:
            for graph in graphs:
                for _ in range(replays):
                    graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
    for sample in range(samples):
        for offset in range(len(functions)):
            index = (sample + offset) % len(functions)
            for _ in range(2):
                graphs[index].replay()
            start.record()
            for _ in range(replays):
                graphs[index].replay()
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end) / replays)
    return tuple(median(values) for values in times)


def print_benchmark_table(results: list[BenchmarkResult], reference_name: str = "Torch reference") -> None:
    """Print aligned timing columns; name non-Torch references explicitly."""
    headers = ("Operation", "Custom (ms)", f"{reference_name} (ms)", "Speedup")
    rows = [
        (
            r.operation,
            f"{r.custom_ms:.4f}",
            f"{r.reference_ms:.4f}",
            f"{r.speedup:.3f}x",
        )
        for r in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(4)]
    for index, row in enumerate([headers, *rows]):
        print(
            " | ".join(
                value.ljust(width) if i == 0 else value.rjust(width)
                for i, (value, width) in enumerate(zip(row, widths))
            ),
            flush=True,
        )
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


def benchmark(label: str, reference: Callable, custom: Callable) -> None:
    """Print a single paired result for the other standalone kernel families."""
    result = measure(label, reference, custom)
    print(
        f"{label}: reference={result.reference_ms:.4f} ms custom={result.custom_ms:.4f} ms "
        f"speedup={result.speedup:.3f}x",
        flush=True,
    )
