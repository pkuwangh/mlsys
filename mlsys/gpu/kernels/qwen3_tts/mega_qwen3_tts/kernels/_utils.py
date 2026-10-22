"""Short CUDA-graph measurements and numerical reporting."""

from statistics import median

import torch


def device_arch(device):
    """Select the compiled target for Hopper or datacenter Blackwell."""
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((9, 0), (10, 0), (10, 3)):
        raise ValueError("Expected an SM90, SM100, or SM103 GPU")
    return f"sm_{capability[0]}{capability[1]}a"


@torch.inference_mode()
def measure(functions: dict, samples: int = 3, replays: int = 5, *, profile=False, calls_per_graph=1):
    """Warm and capture each callable; retain outputs and alternate timing order."""
    graphs, outputs = {}, {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_graph):
                outputs[name] = function()
        graphs[name] = graph
    times = {name: [] for name in functions}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for sample in range(samples):
        names = list(functions)
        for name in names if sample % 2 == 0 else reversed(names):
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / (replays * calls_per_graph))
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("benchmark"):
            for name, graph in graphs.items():
                with torch.cuda.nvtx.range(name):
                    graph.replay()
                    torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    return {name: median(values) for name, values in times.items()}


def report_error(name, actual, expected, *, atol=0.02, rtol=0.02, strict=True):
    error = (actual.float() - expected.float()).abs()
    relative_l2 = error.norm() / expected.float().norm().clamp_min(1e-12)
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(
        f"{name}: {'PASS' if passed else 'FAIL'} max_abs={error.max().item():.6g} "
        f"mean_abs={error.mean().item():.6g} relative_l2={relative_l2.item():.6g}",
        flush=True,
    )
    if strict:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return passed


def report_difference(name, actual, reference):
    """Report model-level numerical drift; reject shape errors and non-finite values."""
    if actual.shape != reference.shape:
        raise ValueError(f"{name}: shape mismatch: {actual.shape} vs {reference.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError(f"{name}: non-finite values")
    error = (actual.float() - reference.float()).abs()
    relative_l2 = error.norm() / reference.float().norm().clamp_min(1e-12)
    print(
        f"{name}: max_abs={error.max().item():.6g} mean_abs={error.mean().item():.6g} "
        f"relative_l2={relative_l2.item():.6g}",
        flush=True,
    )


def print_table(headers, rows):
    """Print aligned columns with a header separator."""
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print(" | ".join(value.rjust(width) for value, width in zip(headers, widths)), flush=True)
    print("-+-".join("-" * width for width in widths), flush=True)
    for row in rows:
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)


def print_kernel_summary(results, *, columns, workload):
    """Print per-call microseconds and baseline/fused speedups after a kernel sweep."""
    baselines = [name for name in ("Torch", "Compiled") if name in results[0][1]]
    names = ["Fused", *baselines]
    headers = [*columns, *[f"{name} (us)" for name in names], *[f"vs {name}" for name in baselines]]
    rows = [
        [
            *case,
            *[f"{times[name]:.3f}" for name in names],
            *[f"{times[name] / times['Fused']:.3f}x" for name in baselines],
        ]
        for case, times in results
    ]
    print(f"\nSummary: {workload}, median interleaved CUDA-graph timing per call.")
    print("Speedup=baseline/Fused. Compilation, preparation and validation excluded.")
    print_table(headers, rows)


def print_summary(results, *, workload):
    """Print model latency in milliseconds from timings measured in microseconds."""
    names = list(results[0][1])
    baseline = "Compiled" if "Compiled" in names else "Torch"
    headers = ["Batch", *[f"{name} (ms)" for name in names], f"{baseline}/Fused"]
    rows = [
        [str(batch), *[f"{times[name] / 1000:.3f}" for name in names], f"{times[baseline] / times['Fused']:.3f}x"]
        for batch, times in results
    ]
    print(f"\nSummary: milliseconds per {workload}.")
    print_table(headers, rows)
