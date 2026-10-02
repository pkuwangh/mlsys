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


def measure(label: str, reference: Callable, custom: Callable) -> BenchmarkResult:
    """Measure paired median GPU times, excluding Python and JIT compilation.

    Warm both functions before graph capture. Alternate replay order across
    five samples to reduce clock drift bias; retain graph outputs throughout.
    Both callables must be inference-only and use the current CUDA device.
    """
    graphs = []
    outputs = []
    for fn in (reference, custom):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(fn())
        graphs.append(graph)
    times = [[], []]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(5):
        for index in (0, 1) if sample % 2 == 0 else (1, 0):
            start.record()
            for _ in range(20):
                graphs[index].replay()
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end) / 20)
    ref_ms, custom_ms = (median(values) for values in times)
    return BenchmarkResult(label, custom_ms, ref_ms)


def print_benchmark_table(
    results: list[BenchmarkResult], reference_name: str = "Torch reference"
) -> None:
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
