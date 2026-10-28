"""CPU checks for shared benchmark reports and decoder output lifetimes."""

import io
import sys
import weakref
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_wan_vae_decoder as decoder
import bench_wan_vae_encoder as encoder
import mega_wan_vae
import torch
from bench_utils import BenchmarkCase, BenchmarkResult, compare_tensors, print_case_header


def check_decoder_validation():
    shape = (1, 3, 5, 2, 2)
    calls = []
    candidates = []

    def run(name):
        assert all(ref() is None for ref in candidates), "Previous decoded candidate remained alive"
        calls.append(name)
        result = torch.zeros(shape, dtype=torch.bfloat16)
        if name != "Diffusers":
            candidates.append(weakref.ref(result))
        return result

    paths = {name: lambda name=name: run(name) for name in ("Diffusers", "Torch", "Mega", "Compiled")}
    output = io.StringIO()
    with redirect_stdout(output):
        print_case_header(2, 6, BenchmarkCase(2, 2, 5, 1))
        failures = decoder.warmup_and_validate(paths, shape, atol=0.05, rtol=0.02)
    text = output.getvalue()
    assert failures == []
    assert calls == ["Diffusers", "Torch", "Mega", "Compiled"]
    assert all(ref() is None for ref in candidates)
    assert "[2/6] 2x2, T=5, N=1" in text
    assert text.index("Warmup/validation: Compiled") < text.index("PASS Mega vs Diffusers")
    assert text.index("PASS Mega") < text.index("PASS Torch eager") < text.index("PASS Torch compiled")
    assert text.count("atol=0.05 rtol=0.02 max_abs=0.000000e+00 mean_abs=0.000000e+00 rms_abs=0.000000e+00") == 3


def check_decoder_failures():
    shape = (1, 3, 5, 2, 2)

    def fail_compile():
        raise RuntimeError("simulated compilation failure")

    paths = {
        "Diffusers": lambda: torch.zeros(shape),
        "Mega": lambda: torch.ones(shape),
        "Compiled": fail_compile,
    }
    output = io.StringIO()
    with redirect_stdout(output):
        failures = decoder.warmup_and_validate(paths, shape, atol=0, rtol=0)
    assert failures == ["Mega FAIL", "Compiled ERROR"]
    assert "Compiled" not in paths
    assert "FAIL Mega vs Diffusers" in output.getvalue()
    assert "output frame" in output.getvalue()
    assert "Compiled ERROR: RuntimeError: simulated compilation failure" in output.getvalue()

    # Allocation failures must still reach the sweep's OOM handling.
    def oom():
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    paths["Compiled"] = oom
    with redirect_stdout(io.StringIO()):
        try:
            decoder.warmup_and_validate(paths, shape, atol=0, rtol=0)
        except torch.cuda.OutOfMemoryError:
            pass
        else:
            raise AssertionError("OOM was swallowed as a compilation failure")


def check_output_contract():
    expected = torch.ones((1, 3, 5, 2, 2))
    for actual in (expected[:, :, :1], expected.double()):
        lines = []
        assert not compare_tensors("Mega vs Diffusers", actual, expected, atol=0, rtol=0, emit=lines.append)
        assert "shape/dtype mismatch" in lines[0]
    for dtype in (torch.bfloat16, torch.float32):
        for value in (float("inf"), -float("inf"), float("nan")):
            nonfinite = torch.full(expected.shape, value, dtype=dtype)
            lines = []
            assert not compare_tensors("nonfinite", nonfinite, nonfinite, atol=0, rtol=0, emit=lines.append)
            assert "FAIL nonfinite" in lines[0]


def check_summary_consistency():
    args = SimpleNamespace(benchmark_repeats=4, dtype="bf16")
    for times, status in (
        ({"Mega": 1.0, "Diffusers": 3.0, "Compiled": 1.5}, "ALL PASS"),
        ({"Diffusers": 3.0}, "Compiled ERROR"),
        ({}, "OOM"),
    ):
        encoded, decoded = io.StringIO(), io.StringIO()
        with redirect_stdout(encoded):
            encoder.print_summary([BenchmarkResult(BenchmarkCase(480, 832, 17, 1), times, status)], 4)
        with redirect_stdout(decoded):
            decoder.print_summary([decoder.Result((1, 3, 17, 480, 832), times, status)], args)
        assert encoded.getvalue().replace("full-encode", "full-decode") == decoded.getvalue()
        assert "vs Diffusers" in decoded.getvalue() and "vs Compiled" in decoded.getvalue()
        assert status in decoded.getvalue()
        if "Mega" in times:
            assert "3.000x" in decoded.getvalue() and "1.500x" in decoded.getvalue()


def check_sweep_reuse():
    # Exercise both real sweep loops with a small CPU-compiled model. Each
    # candidate is constructed once and the second case must reuse its graph.
    for benchmark, suffix in ((encoder, "Encoder"), (decoder, "Decoder")):
        torch.compiler.reset()
        original_limit = torch._dynamo.config.recompile_limit
        resolutions = [f"{size}x{size}" for size in range(16, 81, 8)]
        compilations = []

        def backend(graph, inputs, compilations=compilations):
            compilations.append(graph)
            return graph.forward

        candidates = {"Torch": torch.nn.ReLU(), "Mega": torch.nn.ReLU()}
        candidates["Compiled"] = torch.compile(torch.nn.ReLU(), backend=backend, fullgraph=True, dynamic=False)
        vae = SimpleNamespace(
            use_tiling=False,
            config=SimpleNamespace(patch_size=None, scale_factor_spatial=8),
            clear_cache=lambda: None,
        )
        cases = []

        @torch.inference_mode()
        def run_case(*args, candidates=candidates, benchmark=benchmark, cases=cases):
            models = args[-1]
            assert models == candidates
            height = args[1].height if benchmark is encoder else args[3]
            value = torch.tensor([-1.0, 1.0]).repeat(height)
            torch.testing.assert_close(models["Compiled"](value), value.relu())
            cases.append(models)
            if benchmark is encoder:
                return BenchmarkResult(args[1], {"Mega": 1.0, "Compiled": 1.0})
            return decoder.Result((1, 3, args[2], args[3], args[4]), {"Mega": 1.0, "Compiled": 1.0}, "ALL PASS")

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(
                patch.object(
                    sys, "argv", ["bench", "--batch-size", "1", "--frames", "9", "17", "--resolutions", *resolutions]
                )
            )
            stack.enter_context(patch.object(benchmark, "load_vae", return_value=vae))
            stack.enter_context(patch.object(benchmark, "run_case", side_effect=run_case))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="CPU test"))
            stack.enter_context(patch.object(torch.cuda, "mem_get_info", return_value=(256 * 2**30, 256 * 2**30)))
            stack.enter_context(patch.object(torch.cuda, "empty_cache"))
            reset = stack.enter_context(patch.object(torch.compiler, "reset", wraps=torch.compiler.reset))
            constructors = [
                stack.enter_context(patch.object(benchmark, f"TorchWanVae{suffix}", return_value=candidates["Torch"])),
                stack.enter_context(
                    patch.object(benchmark, f"TorchCompileChunkWanVae{suffix}", return_value=candidates["Compiled"])
                ),
                stack.enter_context(
                    patch.object(
                        benchmark if benchmark is encoder else mega_wan_vae,
                        f"MegaWanVae{suffix}",
                        return_value=candidates["Mega"],
                    )
                ),
            ]
            benchmark.main()
            assert all(constructor.call_count == 1 for constructor in constructors)
            reset.assert_not_called()
        assert len(cases) == 2 * len(resolutions) and all(models is cases[0] for models in cases)
        assert len(compilations) == len(resolutions), "An unchanged chunk shape was recompiled"
        assert torch._dynamo.config.recompile_limit == original_limit, "Sweep leaked its graph limit"
    torch.compiler.reset()


if __name__ == "__main__":
    check_decoder_validation()
    check_decoder_failures()
    check_output_contract()
    check_summary_consistency()
    check_sweep_reuse()
    print("Shared benchmark reporting and decoder validation: PASS")
