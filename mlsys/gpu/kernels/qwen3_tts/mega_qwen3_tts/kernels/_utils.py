"""Short, interleaved CUDA-graph measurements."""

from statistics import median

import torch


@torch.inference_mode()
def measure(functions: dict, samples: int = 3, replays: int = 5, *, profile=False):
    """Warm and capture each callable; retain outputs and alternate timing order."""
    graphs, outputs = {}, {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
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
            times[name].append(start.elapsed_time(end) * 1000 / replays)
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
