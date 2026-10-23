"""Shared benchmark policy, reporting, compilation and cache diagnostics.

Backend changes happen only in benchmark setup or an isolated accuracy pass,
never inside production inference. Matmul remains strict in both FP32 modes so
the comparison changes convolution permission alone.
"""

import argparse
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from mega_wan_vae._utils import FeatCache
from mega_wan_vae._utils import canonicalize_cache as raw_canonicalize_cache
from mega_wan_vae.kernels._compile import CompileStats, compile_family_stats, compile_stats

# Automatic FP32 batch scaling and cutoff, applied before rounding.
FP32_BATCH_DIVISOR = 2
STRICT_FP32_EXTRA_BATCH_DIVISOR = 4
MIN_AUTO_BATCH_SIZE = 0.8


def auto_batch_divisor(dtype: str, allow_tf32: bool) -> int:
    """Reduce FP32 batches for storage, and strict FP32 further for runtime."""
    if dtype != "fp32":
        return 1
    return FP32_BATCH_DIVISOR * (1 if allow_tf32 else STRICT_FP32_EXTRA_BATCH_DIVISOR)


# Result reporting


@dataclass(frozen=True)
class BenchmarkCase:
    """One complete NCTHW video workload; resolution is height by width."""

    height: int
    width: int
    frames: int
    batch_size: int

    @property
    def label(self) -> str:
        return f"{self.height}x{self.width}, T={self.frames}, N={self.batch_size}"


@dataclass
class BenchmarkResult:
    """Median full-call CUDA-event times and validation for one workload."""

    case: BenchmarkCase
    timings: dict[str, float]
    status: str = "ALL PASS"


def print_line(line: str) -> None:
    print(line, flush=True)


def print_startup_header(args, *, operation: str, device_name: str, batch_label: str) -> None:
    """Identify the model benchmark and its common run settings."""
    print(
        f"Wan VAE {operation} | {device_name} | {batch_label} | "
        f"{args.dtype.upper()} | repeats={args.benchmark_repeats}",
        flush=True,
    )
    print(
        f"Loading local checkpoint {args.model}"
        if args.model
        else "Instantiating Wan 2.2 architecture with deterministic random weights",
        flush=True,
    )


def print_case_header(index: int, total: int, case: BenchmarkCase) -> None:
    print(f"\n[{index}/{total}] {case.label}", flush=True)


def compare_tensors(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    frame_label: str = "latent frame",
    emit: Callable[[str], None] = print_line,
    diagnostic: bool = False,
) -> bool:
    """Compare output tensors; an alternate emitter can retain text without tensors."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        emit(
            f"  FAIL {name}: shape/dtype mismatch "
            f"actual={tuple(actual.shape)}/{actual.dtype} expected={tuple(expected.shape)}/{expected.dtype}"
        )
        return False
    diff = (actual.detach().float() - expected.detach().float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms_abs = float(torch.sqrt(torch.mean(diff * diff)).item())
    matches = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    matches &= torch.isfinite(actual) & torch.isfinite(expected)
    allclose = bool(matches.all())
    status = "PASS" if allclose else "FAIL"
    if diagnostic:
        status = "DIAG within tolerance" if allclose else "DIAG outside tolerance"
    emit(
        f"  {status} {name}: atol={atol:g} rtol={rtol:g} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}",
    )
    if not allclose:
        failed = ~matches
        count = int(failed.sum().item())
        emit(
            f"    outside tolerance: {count}/{failed.numel()} ({100 * count / failed.numel():.4f}%)",
        )
        if actual.ndim == 5:
            # Both posterior parameters and decoded videos are NCTHW.
            per_frame = failed.sum(dim=(0, 1, 3, 4))
            failing_frames = per_frame.nonzero().flatten()
            first, last = (int(failing_frames[i].item()) for i in (0, -1))
            worst = int(per_frame.argmax().item())
            for t in dict.fromkeys((first, worst, last)):
                emit(
                    f"    {frame_label} {t}/{actual.shape[2] - 1}: "
                    f"failed={int(per_frame[t].item())}/{failed[:, :, t].numel()} "
                    f"max_abs={float(diff[:, :, t].max().item()):.6e} "
                    f"rms_abs={float(diff[:, :, t].square().mean().sqrt().item()):.6e}",
                )
    return bool(allclose)


def print_summary(
    results: list[BenchmarkResult], repeats: int, *, operation: str = "encode", dtype: str = "BF16"
) -> None:
    """Print all measured variants; failed or unavailable runs are never hidden."""
    headers = [
        "Resolution",
        "Frames",
        "Batch",
        "Mega (ms)",
        "Diffusers (ms)",
        "Compiled (ms)",
        "vs Diffusers",
        "vs Compiled",
        "Validation",
    ]
    rows = []
    for result in results:
        times = result.timings
        mega_ms = times.get("Mega")
        rows.append(
            [
                f"{result.case.height}x{result.case.width}",
                str(result.case.frames),
                str(result.case.batch_size),
                *[f"{times[name]:.3f}" if name in times else "-" for name in ("Mega", "Diffusers", "Compiled")],
                *[
                    f"{times[name] / mega_ms:.3f}x" if mega_ms and name in times else "-"
                    for name in ("Diffusers", "Compiled")
                ],
                result.status,
            ]
        )
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print(
        f"\nSummary: {dtype.upper()}, resolution=HxW, actual batch shown, "
        f"median of {repeats} full-{operation} CUDA-event samples.",
        flush=True,
    )
    print("Speedup=baseline/Mega. Input generation, warmup, compilation and validation excluded.")
    for index, row in enumerate([headers, *rows]):
        print(
            " | ".join(
                value.ljust(width) if i in (0, len(headers) - 1) else value.rjust(width)
                for i, (value, width) in enumerate(zip(row, widths))
            ),
            flush=True,
        )
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


# Arithmetic and accuracy policy

# vLLM-Omni 2d9909525: wan_vae_fastpath/_utils.py::encoder_nrmse_limit
# and test_wan_vae_fastpath_kernels.py::test_channels_last_decoder_keeps_layout_and_stays_close.
# These are full-model criteria, not relaxed primitive tolerances. The separate
# strict FP32 diagnostic always retains atol=rtol=1e-4.
TF32_ENCODER_NRMSE = 0.01
TF32_DECODER_MAX_ABS = 0.01


def add_arithmetic_arguments(parser):
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit TF32 convolution for FP32 in both Mega and all timed baselines. Matmul stays strict.",
    )
    parser.add_argument(
        "--accuracy-diagnostics",
        action="store_true",
        help="Also validate TF32 against strict FP32 and print elementwise diagnostics (one extra untimed forward).",
    )


def configure_arithmetic(args, *, model_validation=True):
    if args.dtype == "fp32":
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        policy = "TF32 permitted" if args.allow_tf32 else "strict FP32 (TF32 disabled)"
        print(f"Arithmetic: FP32 storage | convolution={policy} | matmul=strict FP32", flush=True)
        print("All timed implementations use the same convolution permission.", flush=True)
        if args.allow_tf32 and model_validation:
            print(
                "TF32 model criteria (vLLM-Omni): encoder moments/mean/logvar NRMSE<=0.01; "
                "decoder max_abs<=0.01. Reference: Diffusers with TF32 permitted.",
                flush=True,
            )
            if args.accuracy_diagnostics:
                print(
                    "Accuracy diagnostics: extra strict FP32 Diffusers reference and elementwise 1e-4 comparisons.",
                    flush=True,
                )
    else:
        print(f"Arithmetic: {args.dtype.upper()} storage/convolution", flush=True)


def strict_accuracy_reference(args, call):
    """Compute an independent reference before model compilation, outside timing."""
    if args.dtype != "fp32" or not args.allow_tf32 or not args.accuracy_diagnostics:
        return None
    saved = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        print("  Accuracy reference: strict FP32 Diffusers (untimed)", flush=True)
        result = call()
        torch.cuda.synchronize()
        return result
    finally:
        torch.backends.cudnn.allow_tf32 = saved


def check_tf32_model(name, actual, expected, *, operation, emit=print):
    """Apply the local framework's model criteria, without changing kernel gates."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != torch.float32:
        emit(f"  FAIL {name}: TF32 model shape/dtype mismatch")
        return False
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        emit(f"  FAIL {name}: nonfinite TF32 model outputs")
        return False
    if operation == "encode":
        if actual.ndim != 5 or actual.shape[1] % 2:
            emit(f"  FAIL {name}: expected posterior mean/logvar channels")
            return False
        channels = actual.shape[1] // 2
        passed = True
        for label, section in (
            ("moments", slice(None)),
            ("mean", slice(None, channels)),
            ("logvar", slice(channels, None)),
        ):
            a, b = actual[:, section], expected[:, section]
            error = (a - b).square().mean().sqrt()
            nrmse = (error / b.square().mean().sqrt().clamp_min(1e-8)).item()
            ok = math.isfinite(nrmse) and nrmse <= TF32_ENCODER_NRMSE
            passed &= ok
            emit(f"  {'PASS' if ok else 'FAIL'} {name} {label}: NRMSE={nrmse:.6e} limit={TF32_ENCODER_NRMSE:g}")
        return passed
    if operation != "decode":
        raise ValueError("Expected encode or decode")
    difference = actual - expected
    max_abs = difference.abs().max().item()
    mse = difference.square().mean().item()
    psnr = math.inf if mse == 0 else 10 * math.log10(4 / mse)
    passed = max_abs <= TF32_DECODER_MAX_ABS
    emit(
        f"  {'PASS' if passed else 'FAIL'} {name}: max_abs={max_abs:.6e} "
        f"limit={TF32_DECODER_MAX_ABS:g} PSNR={psnr:.2f} dB"
    )
    return passed


def compare_model_outputs(
    name, actual, expected, *, atol, rtol, allow_tf32=False, operation="encode", accuracy_diagnostics=False, **kwargs
):
    if not allow_tf32:
        return compare_tensors(name, actual, expected, atol=atol, rtol=rtol, **kwargs)
    if accuracy_diagnostics:
        compare_tensors(name, actual, expected, atol=atol, rtol=rtol, diagnostic=True, **kwargs)
    return check_tf32_model(
        f"{name} (TF32 model criterion)", actual, expected, operation=operation, emit=kwargs.get("emit", print)
    )


def report_strict_accuracy(name, actual, reference, *, operation):
    """Report strict-gate diagnostics without relabeling approximate TF32 as IEEE."""
    if reference is None:
        return True
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return check_tf32_model(name, actual, reference, operation=operation)
    compare_tensors(
        f"{name} TF32 vs strict FP32",
        actual,
        reference,
        atol=1e-4,
        rtol=1e-4,
        frame_label="latent frame" if operation == "encode" else "output frame",
        diagnostic=True,
    )
    diff = actual.float() - reference.float()
    ref_rms = reference.float().square().mean().sqrt()
    relative_rms = diff.square().mean().sqrt() / ref_rms.clamp_min(torch.finfo(torch.float32).tiny)
    print(f"    relative_rms={relative_rms.item():.6e}; strict gate unchanged, TF32 is approximate", flush=True)
    return check_tf32_model(f"{name} vs strict FP32 (TF32 model criterion)", actual, reference, operation=operation)


def validation_status(failures, args):
    if failures:
        return "; ".join(failures)
    if args.dtype == "fp32" and args.allow_tf32:
        return "TF32 MODEL CRITERIA PASS"
    return "ALL PASS"


def validation_succeeded(status):
    return status in ("ALL PASS", "TF32 MODEL CRITERIA PASS")


# Compilation diagnostics


def _torch_compile_times():
    # Dynamo's outer compilation timer includes tracing and backend compilation.
    # Summing all timer keys would double-count nested compiler phases. Keep the
    # version-dependent diagnostic access here, outside production model code.
    from torch._dynamo import utils

    metrics = getattr(utils, "compilation_time_metrics", None)
    if metrics is None:
        return None
    return tuple(metrics.get("_compile.compile_inner", ()))


@contextmanager
def report_compilation(label: str, *, cute: bool = False):
    """Report compilation triggered by one path's initial warmup/validation."""
    from torch._dynamo.utils import counters

    cache_specs = (("FX", "inductor", "fxgraph_cache"), ("AOT", "aot_autograd", "autograd_cache"))
    before_cache = {group: counters[group].copy() for _, group, _ in cache_specs}
    before_cute = compile_stats()
    details = cute and os.environ.get("WAN_VAE_COMPILE_DETAILS") == "1"
    before_families = compile_family_stats() if details else {}
    before_torch = _torch_compile_times()
    try:
        yield
    finally:
        delta = compile_stats() - before_cute
        if cute:
            print(
                f"  {label}: CuTe: {delta.compiled} compile calls ({delta.seconds:.3f} s), "
                f"{delta.cache_hits} in-process executable-cache hits",
                flush=True,
            )
        if details:
            for name, current in sorted(compile_family_stats().items()):
                family = current - before_families.get(name, CompileStats(0, 0, 0.0))
                if family.compiled or family.cache_hits:
                    print(
                        f"    {name}: {family.compiled} compile calls ({family.seconds:.3f} s), "
                        f"{family.cache_hits} cache hits",
                        flush=True,
                    )
        after_torch = _torch_compile_times()
        torch_parts = []
        if before_torch is None or after_torch is None or after_torch[: len(before_torch)] != before_torch:
            torch_parts.append("compile timing unavailable")
        else:
            times = after_torch[len(before_torch) :]
            if times:
                torch_parts.append(f"compile/cache setup {sum(times):.3f} s ({len(times)} entries)")
        cache_parts = []
        for name, group, prefix in cache_specs:
            delta = counters[group] - before_cache[group]
            hits, misses, bypasses = (delta[f"{prefix}_{kind}"] for kind in ("hit", "miss", "bypass"))
            if hits or misses or bypasses:
                cache_parts.append(f"{name}={hits}/{misses}/{bypasses}")
        if cache_parts:
            torch_parts.append("cache hit/miss/bypass " + ", ".join(cache_parts))
        if torch_parts:
            print(f"  {label}: Torch: " + "; ".join(torch_parts), flush=True)


# Chunk and cache validation


def cache_state_name(feat_cache: FeatCache) -> str:
    first_tensor = next((entry for entry in feat_cache if isinstance(entry, torch.Tensor)), None)
    if first_tensor is None:
        return "cache_t=0"
    return f"cache_t={first_tensor.shape[2]}"


def tensor_blocks(
    actual: torch.Tensor, expected: torch.Tensor, *, max_elements: int = 2**20
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield bounded views without flattening/copying a strided cache tensor."""
    if actual.shape != expected.shape:
        raise ValueError("Comparison shapes must match")
    if max_elements < 1:
        raise ValueError("Comparison block size must be positive")
    if actual.numel() <= max_elements:
        yield actual, expected
        return
    dim = max((d for d in range(actual.ndim) if actual.shape[d] > 1), key=lambda d: actual.stride(d))
    length = max(1, max_elements // (actual.numel() // actual.shape[dim]))
    for start in range(0, actual.shape[dim], length):
        size = min(length, actual.shape[dim] - start)
        yield from tensor_blocks(
            actual.narrow(dim, start, size), expected.narrow(dim, start, size), max_elements=max_elements
        )


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    maximum = torch.zeros((), device=actual.device)
    total = torch.zeros((), dtype=torch.float64, device=actual.device)
    squared = torch.zeros_like(total)
    for a, b in tensor_blocks(actual, expected):
        diff = (a.detach().float() - b.detach().to(device=a.device, dtype=torch.float32)).abs()
        if diff.numel():
            maximum = torch.maximum(maximum, diff.max())
            total += diff.sum(dtype=torch.float64)
            squared += diff.square().sum(dtype=torch.float64)
    count = max(1, actual.numel())
    return maximum.item(), (total / count).item(), (squared / count).sqrt().item()


def cache_tolerance_counts(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    max_elements: int = 2**20,
    allow_bf16_neighbors: bool = False,
) -> tuple[int, int]:
    """Count rejected values, optionally allowing adjacent finite BF16 values."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Cache shape and dtype must match")
    failed = torch.zeros((), dtype=torch.int64, device=actual.device)
    adjacent = torch.zeros_like(failed)
    for a, b in tensor_blocks(actual, expected, max_elements=max_elements):
        b = b.to(a.device)
        finite = torch.isfinite(a) & torch.isfinite(b)
        outside = ~(torch.isclose(a, b, atol=atol, rtol=rtol) & finite)
        failed += outside.sum()
        if allow_bf16_neighbors and actual.dtype == torch.bfloat16:
            # nextafter handles unequal spacing at powers of two. All masks and
            # CPU-reference transfers are bounded, even for multi-GiB caches.
            adjacent += (outside & finite & (torch.nextafter(b, a) == a)).sum()
    return int((failed - adjacent).item()), int(adjacent.item())


class _CpuTensorBuffer:
    """Grow-only CPU storage, reused for synchronous tensor snapshots.

    Five-dimensional views use channels-last storage so eager snapshots can
    also serve as the next chunk's canonical cache. Returned views remain
    valid only until the next copy into this buffer. Keep these large buffers
    pageable instead of pinning an entire multi-GiB cache set.
    """

    def __init__(self) -> None:
        self.storage: torch.Tensor | None = None

    def copy_from(self, source: torch.Tensor) -> torch.Tensor:
        elements = source.numel()
        if self.storage is None or self.storage.dtype != source.dtype or self.storage.numel() < elements:
            self.storage = None
            self.storage = torch.empty(elements, dtype=source.dtype, device="cpu")
        view = self.storage[:elements]
        if source.ndim == 5:
            n, c, t, h, w = source.shape
            view = view.view(n, t, h, w, c).permute(0, 4, 1, 2, 3)
        else:
            view = view.view(source.shape)
        # Complete the transfer before comparison or reuse of either buffer.
        view.copy_(source.detach(), non_blocking=False)
        return view


def validate_cache_chunks(
    chunks: Iterable[torch.Tensor],
    feat_cache: FeatCache,
    eager_step: Callable[[torch.Tensor, FeatCache, int], tuple[torch.Tensor, FeatCache]],
    candidate_step: Callable[[torch.Tensor, FeatCache, int], tuple[torch.Tensor, FeatCache]],
    *,
    iterations: int,
    atol: float,
    rtol: float,
    label: str,
    cache_nthwc: bool = False,
    allow_bf16_neighbors: bool = False,
) -> bool:
    """Replay chunks from eager history using reusable, bounded CPU comparisons.

    Full-output checks separately validate each implementation's accumulated
    history. Only Wan 2.2 Mega encoding uses NTHWC cache entries.
    """
    eager_buffers: list[_CpuTensorBuffer] = []
    eager_output_buffer = _CpuTensorBuffer()
    candidate_buffer = _CpuTensorBuffer()
    print(f"  Validating {iterations} {label} chunks and their caches on CPU", flush=True)
    for i, chunk in enumerate(chunks):
        started = time.perf_counter()
        cache_in = tuple(
            entry.to(chunk.device, copy=True) if isinstance(entry, torch.Tensor) else entry
            for entry in raw_canonicalize_cache(feat_cache)
        )
        # The input now owns its storage, including for CPU-only checks. Drop
        # old snapshot views before growing or overwriting the backing buffers.
        feat_cache = None
        eager_out, eager_cache = eager_step(chunk, cache_in, i)
        # Keep the reference on the host during candidate execution. Retaining
        # both GPU cache sets adds tens of GiB beyond a normal encode's peak.
        eager_out = eager_output_buffer.copy_from(eager_out)
        while len(eager_buffers) < len(eager_cache):
            eager_buffers.append(_CpuTensorBuffer())
        eager_cache = tuple(
            eager_buffers[index].copy_from(entry) if isinstance(entry, torch.Tensor) else entry
            for index, entry in enumerate(eager_cache)
        )
        feat_cache = eager_cache
        candidate_input = (
            tuple(entry.permute(0, 2, 3, 4, 1) if isinstance(entry, torch.Tensor) else entry for entry in cache_in)
            if cache_nthwc
            else cache_in
        )
        candidate_out, candidate_cache = candidate_step(chunk, candidate_input, i)
        del candidate_input
        if cache_nthwc:
            candidate_cache = tuple(
                entry.permute(0, 4, 1, 2, 3) if isinstance(entry, torch.Tensor) else entry for entry in candidate_cache
            )

        actual_cpu = candidate_buffer.copy_from(candidate_out)
        if actual_cpu.shape != eager_out.shape or actual_cpu.dtype != eager_out.dtype:
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} output shape/dtype mismatch "
                f"{label}={tuple(actual_cpu.shape)}/{actual_cpu.dtype} "
                f"eager={tuple(eager_out.shape)}/{eager_out.dtype}"
            )
            return False
        output_rejected, _ = cache_tolerance_counts(actual_cpu, eager_out, atol=atol, rtol=rtol)
        if output_rejected:
            max_abs, mean_abs, rms_abs = tensor_error(actual_cpu, eager_out)
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} output mismatch "
                f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}"
            )
            return False

        actual_cpu = None
        if len(candidate_cache) != len(eager_cache):
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} "
                "cache length mismatch "
                f"{label}={len(candidate_cache)} eager={len(eager_cache)}"
            )
            return False

        for cache_idx, (actual, expected) in enumerate(zip(candidate_cache, eager_cache)):
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    print(
                        f"  rejected: chunk[{i}] cache[{cache_idx}] shape/dtype mismatch "
                        f"{label}={tuple(actual.shape)}/{actual.dtype} "
                        f"eager={tuple(expected.shape)}/{expected.dtype}"
                    )
                    return False
                # Release the previous view before the shared buffer may grow.
                actual_cpu = None
                actual_cpu = candidate_buffer.copy_from(actual)
                failed, adjacent = cache_tolerance_counts(
                    actual_cpu, expected, atol=atol, rtol=rtol, allow_bf16_neighbors=allow_bf16_neighbors
                )
                if failed or adjacent:
                    max_abs, mean_abs, rms_abs = tensor_error(actual_cpu, expected)
                    print(
                        f"  {'rejected' if failed else 'accepted BF16 neighbors'}: "
                        f"chunk[{i}] {cache_state_name(cache_in)} cache[{cache_idx}] "
                        f"shape={tuple(actual.shape)} max_abs={max_abs:.6e} "
                        f"mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e} "
                        f"outside_atol_rtol={failed + adjacent}/{expected.numel()} "
                        f"accepted_adjacent={adjacent} rejected={failed}",
                        flush=True,
                    )
                if failed:
                    return False
            elif type(actual) is not type(expected) or actual != expected:
                print(
                    f"  rejected: chunk[{i}] {cache_state_name(cache_in)} "
                    f"cache[{cache_idx}] mismatch "
                    f"{label}={type(actual).__name__} eager={type(expected).__name__}"
                )
                return False

        # Loop variables otherwise keep the previous candidate cache alive while
        # Python evaluates the next chunk's right-hand side.
        actual = expected = actual_cpu = None
        del eager_cache, eager_out, candidate_cache, candidate_out, cache_in
        print(f"    chunk {i + 1}/{iterations}: PASS ({time.perf_counter() - started:.2f}s wall)", flush=True)

    return True
