"""CPU regression checks for CuTe/Torch compilation and cache accounting."""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bench_utils import report_compilation
from mega_wan_vae.kernels import _compile
from torch._dynamo import utils


def check_cute_accounting():
    # Use an isolated registry and a fake compiler: no DSL import or GPU work.
    fake_cute = SimpleNamespace(compile=lambda value: ("executable", value))
    with (
        patch.object(_compile, "_caches", []),
        patch.object(_compile, "_compiled", 0),
        patch.object(_compile, "_compile_seconds", 0.0),
        patch.object(_compile, "perf_counter", side_effect=(10.0, 12.0, 20.0, 23.0)),
        patch.dict(sys.modules, {"cutlass": SimpleNamespace(cute=fake_cute)}),
    ):

        @_compile.compile_cache(maxsize=2)
        def compile_variant(channels):
            return _compile.compile_kernel(channels)

        start = _compile.compile_stats()
        output = io.StringIO()
        with redirect_stdout(output), report_compilation("Mega", cute=True):
            assert compile_variant(96) is compile_variant(96)
            compile_variant(192)
        delta = _compile.compile_stats() - start
        assert (delta.compiled, delta.cache_hits, delta.seconds) == (2, 1, 5.0)
        family = _compile.compile_family_stats()[f"{compile_variant.__module__.rsplit('.', 1)[-1]}.compile_variant"]
        assert (family.compiled, family.cache_hits, family.seconds) == (2, 1, 5.0)
        assert "CuTe: 2 compile calls (5.000 s), 1 in-process executable-cache hits" in output.getvalue()
        assert "Torch:" not in output.getvalue()
        output = io.StringIO()
        with redirect_stdout(output), report_compilation("Mega", cute=True):
            compile_variant(96)
            compile_variant(192)
        assert "CuTe: 0 compile calls (0.000 s), 2 in-process executable-cache hits" in output.getvalue()
        assert "Torch:" not in output.getvalue()


def check_torch_accounting():
    # Nested phases must not be added to the outer compilation time.
    metrics = {"_compile.compile_inner": [50.0], "nested_phase": [40.0]}
    output = io.StringIO()
    with (
        patch.object(utils, "compilation_time_metrics", metrics),
        redirect_stdout(output),
        report_compilation("Compiled"),
    ):
        metrics["_compile.compile_inner"].extend([1.0, 2.0])
        metrics["nested_phase"].append(2.5)
    assert "Torch: compile/cache setup 3.000 s (2 entries)" in output.getvalue()
    # Verify the timer used by the installed Torch, using only CPU eager backend.
    before = len(utils.compilation_time_metrics.get("_compile.compile_inner", ()))
    output = io.StringIO()
    value = torch.tensor([1.0])
    with redirect_stdout(output), report_compilation("Compiled"):
        compiled = torch.compile(lambda x: x.sin() + 1, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(value), value.sin() + 1)
    assert len(utils.compilation_time_metrics["_compile.compile_inner"]) == before + 1
    assert "(1 entries)" in output.getvalue()


def check_cache_accounting():
    output = io.StringIO()
    with patch.dict(utils.counters, {}, clear=True), redirect_stdout(output):
        utils.counters["inductor"]["fxgraph_cache_hit"] = 100
        with report_compilation("Compiled"):
            utils.counters["inductor"]["fxgraph_cache_hit"] += 2
            utils.counters["inductor"]["fxgraph_cache_miss"] += 1
            utils.counters["aot_autograd"]["autograd_cache_bypass"] += 3
        assert "cache hit/miss/bypass FX=2/1/0, AOT=0/0/3" in output.getvalue()
        assert len(output.getvalue().splitlines()) == 1
        output.seek(0)
        output.truncate()
        with report_compilation("Compiled"):
            pass
        assert not output.getvalue(), "Inactive caches reported stale counters"


if __name__ == "__main__":
    check_cute_accounting()
    check_torch_accounting()
    check_cache_accounting()
    print("Compilation and cache accounting: PASS")
