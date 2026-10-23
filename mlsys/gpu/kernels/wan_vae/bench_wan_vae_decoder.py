#!/usr/bin/env python3

"""Benchmark Wan 2.1/2.2 decoding: Diffusers, eager/compiled Torch, and custom fusion.

Shapes describe the output video, not the input latent. Latents are seeded
Gaussian tensors; omitting --model uses the Wan 2.2 architecture with random
weights. This measures speed and numerical agreement, not reconstruction quality.
All variants use the same channels-last weights and unnormalized latent input.
Full-decode CUDA-event timings include chunk launches, unpatchify, and clamping;
input generation, compilation, warmup, and validation are excluded.
Eager Torch runs only for validation; Diffusers, compiled Torch and Mega are timed.
Cache validation is disabled by default; opt in with --validate-caches mega
or --validate-caches torch-compiled.

    python bench_wan_vae_decoder.py --resolutions 480x832 --frames 17
    python bench_wan_vae_decoder.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers
    python bench_wan_vae_decoder.py --model /path/to/Wan2.2-TI2V-5B-Diffusers

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        python bench_wan_vae_decoder.py --profile --resolutions 480x832 --frames 17

Use --capture-range-end=repeat to profile multiple cases. Both architectures
use fused preparation and attention post-processing. Wan 2.1 uses raw custom
C96/C192 convolutions and cuDNN for C384. Wan 2.2 also replaces latent input
packing/convolution and integrates C256 convolution post-ops; C512 uses cuDNN.
It supports BF16 or FP32 with explicit TF32 permission and requires CUTLASS DSL.
"""

import argparse
import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from bench_utils import (
    BenchmarkCase,
    BenchmarkResult,
    add_arithmetic_arguments,
    compare_model_outputs,
    configure_arithmetic,
    print_case_header,
    print_startup_header,
    report_compilation,
    report_strict_accuracy,
    strict_accuracy_reference,
    validate_cache_chunks,
    validation_status,
    validation_succeeded,
)
from bench_utils import print_summary as print_result_table
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from wan_vae_reference import TorchCompileChunkWanVaeDecoder, TorchWanVaeDecoder

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TOLERANCES = {"bf16": (0.05, 0.02), "fp32": (1e-4, 1e-4)}


@dataclass
class Result:
    """Full-decode latency and correctness for one output-video shape."""

    shape: tuple[int, int, int, int, int]
    timings: dict[str, float]
    status: str


def resolution(value: str) -> tuple[int, int]:
    """Parse a positive HEIGHTxWIDTH argument."""
    try:
        height, width = map(int, value.lower().split("x"))
        if height <= 0 or width <= 0:
            raise ValueError
        return height, width
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected positive HEIGHTxWIDTH") from error


def parse_args() -> argparse.Namespace:
    """Parse output shapes and model settings, showing defaults in --help."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arithmetic_arguments(parser)
    parser.add_argument("--model", help="Local Diffusers model directory; omit for synthetic weights.")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16", help="Weights and latent dtype.")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch for every case.")
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[17],
        help="Decoded frame counts (4n+1).",
    )
    parser.add_argument(
        "--resolutions",
        type=resolution,
        nargs="+",
        default=[(720, 1280)],
        metavar="HEIGHTxWIDTH",
        help="Decoded video resolutions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synthetic weights and latent seed.")
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Full-decode timing samples.")
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the compiled Torch baseline; Mega may still use torch.compile internally.",
    )
    parser.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include custom decoder fusions (BF16 or FP32 with the selected TF32 permission).",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run an extra forward per variant in CUDA profiler ranges."
    )
    parser.add_argument(
        "--validate-caches",
        choices=("mega", "torch-compiled"),
        help="Opt into expensive chunk/cache validation against eager for the selected implementation (default: disabled).",
    )
    args = parser.parse_args()
    if args.validate_caches == "torch-compiled" and not args.torch_compile:
        parser.error("--validate-caches torch-compiled requires --torch-compile")
    if args.validate_caches == "mega" and not args.fused:
        parser.error("--validate-caches mega requires --fused")
    if args.batch_size < 1 or args.benchmark_repeats < 1:
        parser.error("--batch-size and --benchmark-repeats must be positive")
    if any(t < 1 or (t - 1) % 4 for t in args.frames):
        parser.error("--frames must be positive and of the form 4n+1")
    return args


def validate_decoder_caches(eager, decode_chunk, latent, *, atol, rtol, label, allow_bf16_neighbors=False):
    """Compare pre-unpatchify, unclamped chunks from the same eager history."""
    if eager.use_tiling and (
        latent.shape[-1] > eager.tile_latent_min_width or latent.shape[-2] > eager.tile_latent_min_height
    ):
        print("  rejected: tiling is enabled.", flush=True)
        return False
    x = eager.post_quant_conv(latent)
    iterations = x.shape[2]
    return validate_cache_chunks(
        (x[:, :, i : i + 1] for i in range(iterations)),
        eager.new_cache(),
        lambda chunk, cache, index: eager.decode_chunk(chunk, cache, index == 0),
        lambda chunk, cache, index: decode_chunk(chunk, cache, index == 0),
        iterations=iterations,
        atol=atol,
        rtol=rtol,
        label=label,
        allow_bf16_neighbors=allow_bf16_neighbors,
    )


def load_vae(args: argparse.Namespace) -> AutoencoderKLWan:
    """Load shared channels-last weights without loading a checkpoint by default."""
    dtype = DTYPES[args.dtype]
    if args.model:
        vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    else:
        torch.manual_seed(args.seed)
        vae = AutoencoderKLWan(
            base_dim=160,
            decoder_base_dim=256,
            z_dim=48,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            latents_mean=[0.0] * 48,
            latents_std=[1.0] * 48,
            is_residual=True,
            in_channels=12,
            out_channels=12,
            patch_size=2,
            scale_factor_temporal=4,
            scale_factor_spatial=16,
        ).to(dtype=dtype)
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    # Checkpoint loading already applies dtype while respecting FP32 modules.
    return vae.to(device="cuda").eval()


def median_timings(paths: dict[str, Callable], repeats: int) -> dict[str, float]:
    """Alternate execution order and measure complete decodes without CUDA Graphs."""
    samples = {name: [] for name in paths}
    peaks = {name: 0 for name in paths}
    for i in range(repeats):
        order = list(paths) if i % 2 == 0 else list(reversed(paths))
        for name in order:
            live_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            output = paths[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            peaks[name] = max(peaks[name], torch.cuda.max_memory_allocated() - live_bytes)
            del output
    print(
        "  Peak additional allocated GiB: " + " | ".join(f"{name}={size / 2**30:.3f}" for name, size in peaks.items()),
        flush=True,
    )
    return {name: statistics.median(values) for name, values in samples.items()}


def warmup_and_validate(
    paths: dict[str, Callable],
    expected_shape: tuple[int, ...],
    *,
    atol: float,
    rtol: float,
    strict_reference=None,
    allow_tf32=False,
    accuracy_diagnostics=False,
) -> list[str]:
    """Validate one candidate at a time, retaining only text for grouped reports."""
    reports = {}
    failures = []
    expected = None
    for name in list(paths):
        print(f"  Warmup/validation: {name}", flush=True)
        try:
            if name in ("Mega", "Compiled"):
                with report_compilation(name, cute=name == "Mega"):
                    actual = paths[name]()
            else:
                actual = paths[name]()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            if name != "Compiled":
                raise
            print(f"  Compiled ERROR: {type(error).__name__}: {error}", flush=True)
            failures.append("Compiled ERROR")
            del paths[name]
            continue
        if not report_strict_accuracy(name, actual, strict_reference, operation="decode"):
            failures.append(f"{name} TF32 accuracy FAIL")
        if name == "Diffusers":
            if tuple(actual.shape) != expected_shape:
                raise ValueError(
                    f"Checkpoint output shape {tuple(actual.shape)} does not match requested {expected_shape}"
                )
            expected = actual
        else:
            if expected is None:
                raise ValueError("Diffusers must run before candidate validation")
            lines = []
            label = {"Mega": "Mega", "Torch": "Torch eager", "Compiled": "Torch compiled"}[name]
            matches = compare_model_outputs(
                f"{label} vs Diffusers",
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                frame_label="output frame",
                emit=lines.append,
                allow_tf32=allow_tf32,
                accuracy_diagnostics=accuracy_diagnostics,
                operation="decode",
            )
            reports[name] = lines
            if not matches:
                failures.append(f"{name} FAIL")
        del actual
    for name in ("Mega", "Torch", "Compiled"):
        for line in reports.get(name, ()):
            print(line, flush=True)
    return failures


@torch.inference_mode()
def run_case(
    args: argparse.Namespace,
    vae: AutoencoderKLWan,
    frames: int,
    height: int,
    width: int,
    models: dict[str, torch.nn.Module],
) -> Result:
    """Validate and warm up all variants, optionally profile, then benchmark."""
    spatial = vae.spatial_compression_ratio
    if height % spatial or width % spatial:
        raise ValueError(f"Resolution must be divisible by model spatial compression {spatial}")
    shape = (args.batch_size, 3, frames, height, width)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    latent = torch.randn(
        args.batch_size,
        vae.config.z_dim,
        1 + (frames - 1) // 4,
        height // spatial,
        width // spatial,
        generator=generator,
        device="cuda",
        dtype=DTYPES[args.dtype],
    )
    strict_reference = strict_accuracy_reference(args, lambda: vae.decode(latent, return_dict=False)[0])
    eager = models["Torch"]
    mega, compiled = models.get("Mega"), models.get("Compiled")
    paths = {
        "Diffusers": lambda: vae.decode(latent, return_dict=False)[0],
        "Torch": lambda: eager(latent),
    }
    if mega is not None:
        paths["Mega"] = lambda: mega(latent)
    if compiled is not None:
        paths["Compiled"] = lambda: compiled(latent)
    atol, rtol = TOLERANCES[args.dtype]
    failures = warmup_and_validate(
        paths,
        shape,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
        allow_tf32=args.dtype == "fp32" and args.allow_tf32,
        accuracy_diagnostics=args.accuracy_diagnostics,
    )
    del strict_reference
    if args.validate_caches is not None:
        is_mega = args.validate_caches == "mega"
        label = "Mega" if is_mega else "Compiled"
        if is_mega or "Compiled" in paths:
            matches = validate_decoder_caches(
                eager,
                mega.decode_chunk if is_mega else compiled.chunk_decoder,
                latent,
                atol=atol,
                rtol=rtol,
                label=label,
                allow_bf16_neighbors=not is_mega,
            )
            if not matches:
                print(f"  FAIL {label} chunk outputs/caches", flush=True)
                failures.append(f"{label} cache FAIL")
    # Eager Torch remains a validation reference, not a timed baseline.
    del paths["Torch"]
    if args.profile:
        print("  Profiling: " + " -> ".join(paths), flush=True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for name, call in paths.items():
                with torch.cuda.nvtx.range(f"decode/{shape}/{name}"):
                    output = call()
                    del output
        finally:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
    timings = median_timings(paths, args.benchmark_repeats)
    result = Result(shape, timings, validation_status(failures, args))
    print(f"  Validation: {result.status}", flush=True)
    return result


def print_summary(results: list[Result], args: argparse.Namespace) -> None:
    """Use the same latency/speedup columns as the encoder."""
    rows = [
        BenchmarkResult(
            BenchmarkCase(result.shape[3], result.shape[4], result.shape[2], result.shape[0]),
            result.timings,
            result.status,
        )
        for result in results
    ]
    print_result_table(rows, args.benchmark_repeats, operation="decode", dtype=args.dtype)


def main() -> None:
    """Run a sweep with shared models, releasing per-case activations."""
    args = parse_args()
    print_startup_header(
        args,
        operation="decoder",
        device_name=torch.cuda.get_device_name(),
        batch_label=f"fixed N={args.batch_size}",
    )
    configure_arithmetic(args)
    vae = load_vae(args)
    variants = (["Mega"] if args.fused else []) + ["Torch eager"]
    if args.torch_compile:
        variants.append("Torch compiled")
    print(f"Validation: {', '.join(variants)} vs channels-last Diffusers.", flush=True)
    if not (args.dtype == "fp32" and args.allow_tf32):
        atol, rtol = TOLERANCES[args.dtype]
        print(f"Tolerances: atol={atol}, rtol={rtol}.", flush=True)
    if args.validate_caches is None:
        print("Cache validation: disabled (enable with --validate-caches mega or torch-compiled).", flush=True)
    else:
        print(f"Cache validation: {args.validate_caches} vs eager (expensive, outside timings).", flush=True)
        if args.validate_caches == "torch-compiled" and args.dtype == "bf16":
            print("Compiled caches also allow adjacent finite BF16 values; output tolerances are unchanged.")
    cases = [
        BenchmarkCase(height, width, frames, args.batch_size)
        for height, width in args.resolutions
        for frames in args.frames
    ]
    with torch.inference_mode():
        models = {"Torch": TorchWanVaeDecoder(vae).eval()}
        if args.fused:
            from mega_wan_vae import MegaWanVaeDecoder

            models["Mega"] = MegaWanVaeDecoder(vae).eval()
        if args.torch_compile:
            models["Compiled"] = TorchCompileChunkWanVaeDecoder(vae).eval()
    results = []
    # Preserve the per-case graph budget across the sweep: chunk shapes and
    # temporal-cache layouts can each introduce specializations.
    graph_limit = torch._dynamo.config.recompile_limit * max(1, len(cases))
    try:
        with torch._dynamo.config.patch(recompile_limit=graph_limit):
            for index, case in enumerate(cases, 1):
                print_case_header(index, len(cases), case)
                oom = False
                try:
                    results.append(run_case(args, vae, case.frames, case.height, case.width, models))
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    print("  CUDA OOM: retaining requested batch and continuing sweep.", flush=True)
                    results.append(Result((case.batch_size, 3, case.frames, case.height, case.width), {}, "OOM"))
                finally:
                    vae.clear_cache()
                    if oom:
                        torch.compiler.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        print_summary(results, args)
    if any(not validation_succeeded(result.status) for result in results):
        raise SystemExit("Sweep contains validation, compilation, or OOM failures.")


if __name__ == "__main__":
    main()
