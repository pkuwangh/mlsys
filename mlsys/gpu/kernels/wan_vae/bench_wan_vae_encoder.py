#!/usr/bin/env python3

"""Benchmark Wan 2.1/2.2 VAE encoding on synthetic video input.

The PyTorch/Diffusers baseline records shapes, timings, and numerical references.
Validate and benchmark MegaWanVaeEncoder in BF16 or FP32 with explicit TF32 permission. Both architectures
use fused normalization/SiLU/padding/cache preparation and custom convolutions
where supported. Both versions integrate supported post-ops into convolution
epilogues; Wan 2.1 retains cuDNN and separate preparation for C384.
Other convolutions and remaining operations stay in Torch/cuDNN. Intermediate
Mega implementations live in git history, not runtime configuration options.
Diffusers, compiled Torch and Mega are timed by default; eager Torch runs only
for validation. Use --profile for an extra forward per timed variant in CUDA profiler ranges.
Validation compares their final outputs against Diffusers. Exhaustive
chunk/cache diagnostics require --validate-caches mega or torch-compiled.
The default sweep is HxW=480x832/720x1280 and T=17/93/201.
Without --batch-size, the reference batch scales with free GPU memory, up to
N=16 at 17 frames and 720x1280, then scales by spatial size and frame count.
Automatic batches halve and retry on OOM; the table reports the actual batch.
FP32 automatic batches are divided by 2, then by another 4 with TF32 disabled.
Estimates below MIN_AUTO_BATCH_SIZE are skipped before rounding.
An explicit --batch-size uses that exact batch for every case, without scaling.
Each case reports progress; one final table collects timings and validation.
Torch and Diffusers baselines preserve channels-last storage. One channels-last
Diffusers model is the common numerical and timing reference for every variant.
Use --profile-mega-only to capture just the Mega encode with Nsight;
reference/baseline execution and validation still run outside the capture.

Command to run:
TMPDIR="$PWD/tmp" nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -- python bench_wan_vae_encoder.py --profile \
       --model ../../../../checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers/ \
       --resolutions 480x832 --frames 17

For a multi-case Nsight capture, use --capture-range-end=repeat instead of stop.
Omit ``--model`` to instantiate the Wan 2.2 architecture with deterministic
random weights. That mode measures performance and numerical agreement, not
caption or reconstruction quality.
"""

import argparse
import gc
import math
import statistics
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
from bench_utils import (
    MIN_AUTO_BATCH_SIZE,
    BenchmarkCase,
    BenchmarkResult,
    add_arithmetic_arguments,
    auto_batch_divisor,
    compare_model_outputs,
    configure_arithmetic,
    print_case_header,
    print_startup_header,
    print_summary,
    report_compilation,
    report_strict_accuracy,
    strict_accuracy_reference,
    validate_cache_chunks,
    validation_status,
    validation_succeeded,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from mega_wan_vae import MegaWanVaeEncoder
from mega_wan_vae._utils import (
    FeatCache,
    validate_frame_count,
)
from mega_wan_vae._utils import (
    patchify as raw_patchify,
)
from wan_vae_reference import TorchCompileChunkWanVaeEncoder, TorchWanVaeEncoder

# Common end-to-end BF16 agreement gate; standalone kernels use tighter checks.
COMPARE_ATOL = 0.25
COMPARE_RTOL = 1e-3
TOLERANCES = {"bf16": (COMPARE_ATOL, COMPARE_RTOL), "fp32": (1e-4, 1e-4)}
REFERENCE_FRAMES = 17
REFERENCE_HEIGHT = 720
REFERENCE_WIDTH = 1280
SYNTHETIC_WAN22_CONFIG = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "latents_mean": [0.0] * 48,
    "latents_std": [1.0] * 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}


def scaled_batch_size(
    base_batch_size: int, frames: int, height: int, width: int, *, dtype: str = "bf16", allow_tf32: bool = False
) -> int | None:
    """Choose a spatial batch, then scale it down for videos longer than 17 frames.

    Spatial scaling is capped at twice the memory-scaled 720p batch.
    This is not a peak-VRAM guarantee.
    """
    spatial_batch = min(
        2 * base_batch_size,
        base_batch_size * REFERENCE_HEIGHT * REFERENCE_WIDTH // (height * width),
    )
    scaled = spatial_batch * REFERENCE_FRAMES / max(REFERENCE_FRAMES, frames) / auto_batch_divisor(dtype, allow_tf32)
    if dtype == "fp32" and scaled < MIN_AUTO_BATCH_SIZE:
        return None
    return max(1, int(scaled))


def auto_reference_batch(free_bytes: int) -> int:
    """Scale the original N16 workload to free memory; OOM retries handle peaks."""
    return max(1, min(16, free_bytes * 16 // (256 * 2**30)))


def resolution(value: str) -> tuple[int, int]:
    """Parse a positive HEIGHTxWIDTH command-line value."""
    try:
        height, width = (int(part) for part in value.lower().split("x"))
        if height <= 0 or width <= 0:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected positive HEIGHTxWIDTH") from error
    return height, width


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Wan VAE encode on deterministic synthetic video.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_arithmetic_arguments(parser)
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Local Wan 2.1 or Wan 2.2 Diffusers model directory. When omitted, instantiate "
            "the production architecture with deterministic random weights."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp32"),
        help="Torch dtype for VAE weights and input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Fixed batch for every case, without scaling or OOM retries. When omitted, "
            "scale by free GPU memory and pixel-frame count, then halve on OOM."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[17, 93, 201],
        help="Frame counts to sweep (each must be 4n+1).",
    )
    parser.add_argument(
        "--resolutions",
        type=resolution,
        nargs="+",
        default=[(480, 832), (720, 1280)],
        metavar="HEIGHTxWIDTH",
        help="Spatial resolutions to sweep.",
    )
    parser.add_argument("--height", type=int, help="Single-resolution height; requires --width.")
    parser.add_argument("--width", type=int, help="Single-resolution width; requires --height.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic video generation.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the compiled Torch baseline; Mega may still use torch.compile internally.",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run an extra forward per variant in CUDA profiler ranges."
    )
    parser.add_argument(
        "--validate-caches",
        choices=("mega", "torch-compiled"),
        help="Opt into expensive chunk/cache validation against eager for the selected implementation (default: disabled).",
    )
    parser.add_argument(
        "--profile-mega-only",
        "--profile-custom-only",
        action="store_true",
        help=("Capture only Mega; keep baseline runs and validation outside the CUDA profiler range."),
    )
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Full-encode timing samples.")
    args = parser.parse_args()
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be positive")
    if args.validate_caches == "torch-compiled" and not args.torch_compile:
        parser.error("--validate-caches torch-compiled requires --torch-compile")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be supplied together")
    if args.height is not None:
        if args.height <= 0 or args.width <= 0:
            parser.error("--height and --width must be positive")
        args.resolutions = [(args.height, args.width)]
    for frames in args.frames:
        if frames < 1 or (frames - 1) % 4:
            parser.error("--frames values must be positive and of the form 4n+1")
    return args


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


@contextmanager
def nvtx_range(device: torch.device, name: str) -> Iterator[None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def cuda_profiler_range(device: torch.device, *, enabled: bool = True) -> Iterator[None]:
    """Bound a CUDA capture, or leave an enclosing capture untouched if disabled."""
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


def validate_chunk_caches(
    torch_encoder: TorchWanVaeEncoder,
    encode_chunk: Callable[[torch.Tensor, FeatCache], tuple[torch.Tensor, FeatCache]],
    video: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
    cache_nthwc: bool = False,
    allow_bf16_neighbors: bool = False,
) -> bool:
    """Compare each chunk from the same eager history; full encodes check accumulated drift.

    Wan 2.2 Mega stores all tensor cache entries in NTHWC; other encoders use
    NCTHW. Layout conversions here are views of channels-last storage.
    """
    _, _, num_frame, height, width = video.shape
    x = video
    if torch_encoder.patch_size is not None:
        x = raw_patchify(x, patch_size=int(torch_encoder.patch_size))

    if torch_encoder.use_tiling and (
        width > torch_encoder.tile_sample_min_width or height > torch_encoder.tile_sample_min_height
    ):
        print("  rejected: tiling is enabled.")
        return False

    iterations = 1 + (num_frame - 1) // 4
    chunks = (x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i] for i in range(iterations))
    return validate_cache_chunks(
        chunks,
        torch_encoder.new_cache(),
        lambda chunk, cache, index: torch_encoder.encode_chunk(chunk, cache),
        lambda chunk, cache, index: encode_chunk(chunk, cache),
        iterations=iterations,
        atol=atol,
        rtol=rtol,
        label=label,
        cache_nthwc=cache_nthwc,
        allow_bf16_neighbors=allow_bf16_neighbors,
    )


def sample_video_channels(vae: AutoencoderKLWan) -> int:
    in_channels = int(getattr(vae.config, "in_channels", 3))
    patch_size = getattr(vae.config, "patch_size", None)
    if patch_size is not None:
        patch_area = int(patch_size) * int(patch_size)
        if in_channels % patch_area == 0:
            return in_channels // patch_area
    return in_channels


def make_synthetic_video(
    *,
    batch_size: int,
    channels: int,
    frames: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate deterministic input a frame at a time, outside measured regions.

    Avoid full-video FP32 intermediates. All variants consume the same resident
    CUDA tensor, using the batch size selected for this case.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.empty(batch_size, channels, frames, height, width, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    for index, t in enumerate(torch.linspace(0.0, 1.0, frames)):
        base = [
            torch.sin(2.0 * math.pi * (x + t)).expand(1, 1, height, width),
            torch.cos(2.0 * math.pi * (y - t)).expand(1, 1, height, width),
            torch.sin(math.pi * (x + y + 0.5 * t)).expand(1, 1, height, width),
        ]
        while len(base) < channels:
            base.append(torch.randn(1, 1, height, width, generator=generator))
        frame = torch.cat(base[:channels], dim=1)
        noise = torch.randn(batch_size, channels, height, width, generator=generator)
        frame = (frame + 0.02 * noise).clamp_(-1.0, 1.0)
        video[:, :, index].copy_(frame)
    return video


def load_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> AutoencoderKLWan:
    if args.model is None:
        torch.manual_seed(args.seed)
        vae = AutoencoderKLWan(**SYNTHETIC_WAN22_CONFIG).to(dtype=dtype)
    else:
        vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    # Prepare the single reference in place; do not retain an alternate layout.
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    # Checkpoint loading already applies dtype while respecting FP32 modules.
    return vae.to(device=device).eval()


def median_timings(paths: dict[str, Callable[[], torch.Tensor]], repeats: int) -> dict[str, float]:
    """Alternate variant order; time full encodes, including ordinary launches."""
    samples: dict[str, list[float]] = {name: [] for name in paths}
    peaks = {name: 0 for name in paths}
    for repeat in range(repeats):
        order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
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


@torch.inference_mode()
def run_case(
    args: argparse.Namespace,
    case: BenchmarkCase,
    vae: AutoencoderKLWan,
    device: torch.device,
    models: dict[str, torch.nn.Module],
) -> BenchmarkResult:
    """Validate all variants, optionally profile, then time Diffusers/Compiled/Mega."""
    video = make_synthetic_video(
        batch_size=case.batch_size,
        channels=sample_video_channels(vae),
        frames=case.frames,
        height=case.height,
        width=case.width,
        seed=args.seed,
        device=device,
        dtype=resolve_dtype(args.dtype),
    )
    strict_reference = strict_accuracy_reference(args, lambda: vae.encode(video, return_dict=False)[0].parameters)
    eager, mega = models["Torch"], models["Mega"]
    compiled = models.get("Compiled")
    paths = {
        "Diffusers": lambda: vae.encode(video, return_dict=False)[0].parameters,
        "Torch": lambda: eager(video),
        "Mega": lambda: mega(video),
    }
    outputs = {}
    status = []
    for name, fn in paths.items():
        print(f"  Warmup/validation: {name}", flush=True)
        if name == "Mega":
            with report_compilation(name, cute=True):
                outputs[name] = fn()
        else:
            outputs[name] = fn()
    if compiled is not None:
        print("  Warmup/validation: Compiled", flush=True)
        try:
            with report_compilation("Compiled"):
                outputs["Compiled"] = compiled(video)
            paths["Compiled"] = lambda: compiled(video)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - retain other results, report and fail after summary.
            print(f"  Compiled ERROR: {type(error).__name__}: {error}", flush=True)
            status.append("Compiled ERROR")

    tolerances = TOLERANCES[args.dtype]
    for name, output in outputs.items():
        if not report_strict_accuracy(name, output, strict_reference, operation="encode"):
            status.append(f"{name} TF32 accuracy FAIL")
    del strict_reference
    for name, label, reference, atol, rtol in (
        (
            "Mega",
            "Mega vs Diffusers",
            outputs["Diffusers"],
            tolerances[0],
            tolerances[1],
        ),
        (
            "Torch",
            "Torch eager vs Diffusers",
            outputs["Diffusers"],
            tolerances[0],
            tolerances[1],
        ),
        (
            "Compiled",
            "Torch compiled vs Diffusers",
            outputs["Diffusers"],
            tolerances[0],
            tolerances[1],
        ),
    ):
        if name not in outputs:
            continue
        # Posterior mode is a slice of parameters; checking all parameters covers it.
        matches = compare_model_outputs(
            label,
            outputs[name],
            reference,
            atol=atol,
            rtol=rtol,
            allow_tf32=args.dtype == "fp32" and args.allow_tf32,
            accuracy_diagnostics=args.accuracy_diagnostics,
            operation="encode",
        )
        if not matches:
            status.append(f"{name} FAIL")
    del outputs, reference
    if args.validate_caches is not None:
        is_mega = args.validate_caches == "mega"
        label = "Mega" if is_mega else "Compiled"
        # Compilation errors are already recorded above; do not invoke an invalid baseline.
        if is_mega or "Compiled" in paths:
            chunks_match = validate_chunk_caches(
                eager,
                mega.encode_chunk if is_mega else compiled.chunk_encoder,
                video,
                atol=tolerances[0],
                rtol=tolerances[1],
                label=label,
                cache_nthwc=is_mega and bool(vae.config.is_residual),
                allow_bf16_neighbors=not is_mega,
            )
            if not chunks_match:
                print(f"  FAIL {label} chunk outputs/caches", flush=True)
                status.append(f"{label} cache FAIL")

    # Eager Torch remains a validation reference, not a timed baseline.
    del paths["Torch"]
    # Multi-case Nsight recording needs --capture-range-end=repeat, not stop.
    if args.profile or args.profile_mega_only:
        names = ["Mega"] if args.profile_mega_only else list(paths)
        print("  Profiling: " + " -> ".join(names), flush=True)
        with cuda_profiler_range(device):
            for name in names:
                with nvtx_range(device, f"{case.label}/{name}"):
                    output = paths[name]()
                    del output
    result = BenchmarkResult(case, median_timings(paths, args.benchmark_repeats))
    result.status = validation_status(status, args)
    print(f"  Validation: {result.status}", flush=True)
    return result


def run_case_with_retry(
    args: argparse.Namespace,
    case: BenchmarkCase,
    vae: AutoencoderKLWan,
    device: torch.device,
    models: dict[str, torch.nn.Module],
) -> BenchmarkResult:
    """Retry automatic batches after OOM, retaining validation for every variant."""
    while True:
        result = None
        oom = False
        try:
            result = run_case(args, case, vae, device, models)
        except torch.cuda.OutOfMemoryError:
            oom = True
        finally:
            # Release per-case activations after the exception traceback drops.
            # Keep compiled graphs across successful cases; discard failed
            # compiler state only when recovering from OOM.
            vae.clear_cache()
            if oom:
                torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
        if result is not None:
            return result
        if args.batch_size is not None or case.batch_size == 1:
            print("  CUDA OOM: requested workload does not fit; continuing sweep.", flush=True)
            return BenchmarkResult(case, {}, "OOM")
        smaller = max(1, case.batch_size // 2)
        print(f"  CUDA OOM: auto batch {case.batch_size} -> {smaller}; retrying all variants.", flush=True)
        case = replace(case, batch_size=smaller)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reference_batch = auto_reference_batch(free_bytes)
    cases = []
    for height, width in args.resolutions:
        for frames in args.frames:
            batch = args.batch_size
            if batch is None:
                batch = scaled_batch_size(
                    reference_batch, frames, height, width, dtype=args.dtype, allow_tf32=args.allow_tf32
                )
            if batch is not None:
                cases.append(BenchmarkCase(height, width, frames, batch))
    if not cases:
        print(f"No cases meet the automatic batch cutoff of {MIN_AUTO_BATCH_SIZE:g}.", flush=True)
        return
    effective_reference = reference_batch / auto_batch_divisor(args.dtype, args.allow_tf32)
    batch_label = (
        f"auto batch (reference N={effective_reference:g}, T=17, HxW=720x1280, free={free_bytes / 2**30:.1f} GiB)"
        if args.batch_size is None
        else f"fixed N={args.batch_size}"
    )
    print_startup_header(
        args,
        operation="encoder",
        device_name=torch.cuda.get_device_name(device),
        batch_label=batch_label,
    )
    configure_arithmetic(args)
    vae = load_vae(args, device, resolve_dtype(args.dtype))
    if vae.use_tiling:
        raise ValueError("This benchmark requires spatial tiling to be disabled.")
    for case in cases:
        validate_frame_count(case.frames)
        for factor in (
            getattr(vae.config, "patch_size", None),
            getattr(
                vae.config,
                "scale_factor_spatial",
                getattr(vae, "spatial_compression_ratio", None),
            ),
        ):
            if factor and (case.height % int(factor) or case.width % int(factor)):
                raise ValueError(f"{case.label}: resolution must be divisible by {factor}")
    variants = "Mega, Torch eager and compiled" if args.torch_compile else "Mega and Torch eager"
    print(f"Validation: {variants} vs channels-last Diffusers.", flush=True)
    if not (args.dtype == "fp32" and args.allow_tf32):
        atol, rtol = TOLERANCES[args.dtype]
        print(f"Tolerances: atol={atol}, rtol={rtol}.", flush=True)
    if args.validate_caches is None:
        print("Cache validation: disabled (enable with --validate-caches mega or torch-compiled).", flush=True)
    else:
        print(f"Cache validation: {args.validate_caches} vs eager (expensive, outside timings).", flush=True)
        if args.validate_caches == "torch-compiled" and args.dtype == "bf16":
            print("Compiled caches also allow adjacent finite BF16 values; output tolerances are unchanged.")
    with torch.inference_mode():
        models = {"Torch": TorchWanVaeEncoder(vae).eval(), "Mega": MegaWanVaeEncoder(vae).eval()}
        if args.torch_compile:
            models["Compiled"] = TorchCompileChunkWanVaeEncoder(vae).eval()
    results = []
    # Preserve the per-case graph budget across the sweep: chunk shapes and
    # temporal-cache layouts can each introduce specializations.
    graph_limit = torch._dynamo.config.recompile_limit * max(1, len(cases))
    try:
        with torch._dynamo.config.patch(recompile_limit=graph_limit):
            for index, case in enumerate(cases, 1):
                print_case_header(index, len(cases), case)
                results.append(run_case_with_retry(args, case, vae, device, models))
    finally:
        print_summary(results, args.benchmark_repeats, dtype=args.dtype)
    if any(not validation_succeeded(result.status) for result in results):
        raise SystemExit("Sweep contains failed validation, compilation, or OOM cases.")


if __name__ == "__main__":
    main()
