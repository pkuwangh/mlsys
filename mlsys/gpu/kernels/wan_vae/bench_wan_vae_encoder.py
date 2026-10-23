#!/usr/bin/env python3

"""Benchmark Wan 2.1/2.2 VAE encoding on synthetic video input.

The PyTorch/Diffusers baseline records shapes, timings, and numerical references.
Always validate and benchmark MegaWanVaeEncoder in BF16. Both architectures
use fused normalization/SiLU/padding/cache preparation and custom convolutions
where supported. Wan 2.2 also integrates supported post-ops into convolution
epilogues; Wan 2.1 uses separate preparation kernels and retains cuDNN for C384.
Other convolutions and remaining operations stay in Torch/cuDNN. Intermediate
Mega implementations live in git history, not runtime configuration options.
All four variants, including torch.compile, run and are profiled by default.
The default sweep is HxW=480x832/720x1280 and T=17/93/201.
Without --batch-size, the reference batch scales with free GPU memory, up to
N=16 at 17 frames and 720x1280, then scales by spatial size and frame count.
Automatic batches halve and retry on OOM; the table reports the actual batch.
An explicit --batch-size uses that exact batch for every case, without scaling.
Each case reports progress; one final table collects timings and validation.
Torch and Diffusers baselines preserve channels-last storage. One channels-last
Diffusers model is the common numerical and timing reference for every variant.
Use --profile-mega-only to capture just the Mega encode with Nsight;
reference/baseline execution and validation still run outside the capture.

Command to run:
TMPDIR="$PWD/tmp" nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -- python bench_wan_vae_encoder.py \
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
from dataclasses import dataclass, replace

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from mega_wan_vae import MegaWanVaeEncoder
from mega_wan_vae._utils import (
    FeatCache,
    validate_frame_count,
)
from mega_wan_vae._utils import (
    canonicalize_cache as raw_canonicalize_cache,
)
from mega_wan_vae._utils import (
    patchify as raw_patchify,
)
from wan_vae_reference import TorchCompileChunkWanVaeEncoder, TorchWanVaeEncoder

# Common end-to-end BF16 agreement gate; standalone kernels use tighter checks.
COMPARE_ATOL = 0.25
COMPARE_RTOL = 1e-3
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


def scaled_batch_size(base_batch_size: int, frames: int, height: int, width: int) -> int:
    """Choose a spatial batch, then scale it down for videos longer than 17 frames.

    Spatial scaling is capped at twice the memory-scaled 720p batch.
    This is not a peak-VRAM guarantee.
    """
    spatial_batch = min(
        2 * base_batch_size,
        base_batch_size * REFERENCE_HEIGHT * REFERENCE_WIDTH // (height * width),
    )
    return max(1, spatial_batch * REFERENCE_FRAMES // max(REFERENCE_FRAMES, frames))


def auto_reference_batch(free_bytes: int) -> int:
    """Scale the original N16 workload to free memory; OOM retries handle peaks."""
    return max(1, min(16, free_bytes * 16 // (256 * 2**30)))


@dataclass
class BenchmarkResult:
    """Median full-encode CUDA-event times and validation for one workload."""

    case: BenchmarkCase
    timings: dict[str, float]
    status: str = "ALL PASS"


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
        choices=["fp32", "fp16", "bf16"],
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
        help="Include torch.compile in validation, benchmarking, and profiling.",
    )
    parser.add_argument(
        "--profile-mega-only",
        "--profile-custom-only",
        action="store_true",
        help=("Capture only Mega; keep baseline runs and validation outside the CUDA profiler range."),
    )
    parser.add_argument("--benchmark-repeats", type=int, default=4, help="benchmark iters")
    args = parser.parse_args()
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be positive")
    if args.dtype != "bf16":
        parser.error("Mega requires --dtype bf16")
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
    if name == "fp16":
        return torch.float16
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


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> bool:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms_abs = float(torch.sqrt(torch.mean(diff * diff)).item())
    allclose = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(
        f"  {'PASS' if allclose else 'FAIL'} {name}: atol={atol:g} rtol={rtol:g} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}",
        flush=True,
    )
    if not allclose:
        failed = ~torch.isclose(actual, expected, atol=atol, rtol=rtol)
        count = int(failed.sum().item())
        print(
            f"    outside tolerance: {count}/{failed.numel()} ({100 * count / failed.numel():.4f}%)",
            flush=True,
        )
        if actual.ndim == 5:
            # Final posterior parameters are NCTHW; no extra encode is needed.
            per_frame = failed.sum(dim=(0, 1, 3, 4))
            failing_frames = per_frame.nonzero().flatten()
            first, last = (int(failing_frames[i].item()) for i in (0, -1))
            worst = int(per_frame.argmax().item())
            for t in dict.fromkeys((first, worst, last)):
                print(
                    f"    latent frame {t}/{actual.shape[2] - 1}: "
                    f"failed={int(per_frame[t].item())}/{failed[:, :, t].numel()} "
                    f"max_abs={float(diff[:, :, t].max().item()):.6e} "
                    f"rms_abs={float(diff[:, :, t].square().mean().sqrt().item()):.6e}",
                    flush=True,
                )
    return bool(allclose)


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
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, max_elements: int = 2**20
) -> tuple[int, int]:
    """Count rejected values and BF16 neighbors accepted beyond atol/rtol."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Cache shape and dtype must match")
    failed = torch.zeros((), dtype=torch.int64, device=actual.device)
    adjacent = torch.zeros_like(failed)
    for a, b in tensor_blocks(actual, expected, max_elements=max_elements):
        b = b.to(a.device)
        finite = torch.isfinite(a) & torch.isfinite(b)
        outside = ~(torch.isclose(a, b, atol=atol, rtol=rtol) & finite)
        failed += outside.sum()
        if actual.dtype == torch.bfloat16:
            # nextafter handles unequal spacing at powers of two. All masks and
            # CPU-reference transfers are bounded, even for multi-GiB caches.
            adjacent += (outside & finite & (torch.nextafter(b, a) == a)).sum()
    return int((failed - adjacent).item()), int(adjacent.item())


def debug_compiled_chunk_mismatch(
    torch_encoder: TorchWanVaeEncoder,
    compiled_torch_encoder: TorchCompileChunkWanVaeEncoder,
    video: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    _, _, num_frame, height, width = video.shape
    x = video
    if torch_encoder.patch_size is not None:
        x = raw_patchify(x, patch_size=int(torch_encoder.patch_size))

    if torch_encoder.use_tiling and (
        width > torch_encoder.tile_sample_min_width or height > torch_encoder.tile_sample_min_height
    ):
        print("  rejected: tiling is enabled.")
        return False

    feat_cache = torch_encoder.new_cache()
    iterations = 1 + (num_frame - 1) // 4
    for i in range(iterations):
        if i == 0:
            chunk = x[:, :, :1, :, :]
        else:
            chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]

        cache_in = tuple(
            entry.to(chunk.device) if isinstance(entry, torch.Tensor) else entry
            for entry in raw_canonicalize_cache(feat_cache)
        )
        eager_out, eager_cache = torch_encoder.encode_chunk(chunk, cache_in)
        # Keep the reference on the host during compiled execution. Retaining
        # both GPU cache sets adds tens of GiB beyond a normal encode's peak.
        eager_out = eager_out.cpu()
        eager_cache = tuple(entry.cpu() if isinstance(entry, torch.Tensor) else entry for entry in eager_cache)
        feat_cache = eager_cache
        compiled_out, compiled_cache = compiled_torch_encoder.chunk_encoder(chunk, cache_in)

        if not torch.allclose(compiled_out, eager_out.to(compiled_out.device), atol=atol, rtol=rtol):
            max_abs, mean_abs, rms_abs = tensor_error(compiled_out, eager_out)
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} output mismatch "
                f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}"
            )
            return False

        if len(compiled_cache) != len(eager_cache):
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} "
                "cache length mismatch "
                f"compiled={len(compiled_cache)} eager={len(eager_cache)}"
            )
            return False

        for cache_idx, (actual, expected) in enumerate(zip(compiled_cache, eager_cache)):
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    print(
                        f"  rejected: chunk[{i}] cache[{cache_idx}] shape/dtype mismatch "
                        f"compiled={tuple(actual.shape)}/{actual.dtype} "
                        f"eager={tuple(expected.shape)}/{expected.dtype}"
                    )
                    return False
                failed, adjacent = cache_tolerance_counts(actual, expected, atol=atol, rtol=rtol)
                if failed or adjacent:
                    max_abs, mean_abs, rms_abs = tensor_error(actual, expected)
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
                    f"compiled={type(actual).__name__} eager={type(expected).__name__}"
                )
                return False

        # Loop variables otherwise keep the previous compiled cache alive while
        # Python evaluates the next encode's right-hand side.
        actual = expected = None
        del eager_cache, eager_out, compiled_cache, compiled_out, cache_in

    return True


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
        # print(f"  Timing round {repeat + 1}/{repeats}: {' -> '.join(order)}", flush=True)
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
            # print(f"    {name}: {samples[name][-1]:.3f} ms", flush=True)
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
) -> BenchmarkResult:
    """Warm, validate, capture once, then measure all variants for one video."""
    video = make_synthetic_video(
        batch_size=case.batch_size,
        channels=sample_video_channels(vae),
        frames=case.frames,
        height=case.height,
        width=case.width,
        seed=args.seed,
        device=device,
        dtype=torch.bfloat16,
    )
    eager = TorchWanVaeEncoder(vae).eval()
    mega = MegaWanVaeEncoder(vae).eval()
    compiled = None
    paths = {
        "Diffusers": lambda: vae.encode(video, return_dict=False)[0].parameters,
        "Torch": lambda: eager(video),
        "Mega": lambda: mega(video),
    }
    outputs = {}
    status = []
    for name, fn in paths.items():
        print(f"  Warmup/validation: {name}", flush=True)
        outputs[name] = fn()
    if args.torch_compile:
        print("  Compiling and warming up Torch (setup time not measured)", flush=True)
        try:
            compiled = TorchCompileChunkWanVaeEncoder(vae).eval()
            outputs["Compiled"] = compiled(video)
            paths["Compiled"] = lambda: compiled(video)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - retain other results, report and fail after summary.
            print(f"  Compiled ERROR: {type(error).__name__}: {error}", flush=True)
            status.append("Compiled ERROR")

    for name, label, reference, atol, rtol in (
        (
            "Mega",
            "Mega vs Diffusers",
            outputs["Diffusers"],
            COMPARE_ATOL,
            COMPARE_RTOL,
        ),
        (
            "Torch",
            "Torch eager vs Diffusers",
            outputs["Diffusers"],
            COMPARE_ATOL,
            COMPARE_RTOL,
        ),
        (
            "Compiled",
            "Torch compiled vs Diffusers",
            outputs["Diffusers"],
            COMPARE_ATOL,
            COMPARE_RTOL,
        ),
    ):
        if name not in outputs:
            continue
        # Posterior mode is a slice of parameters; checking all parameters covers it.
        matches = compare_tensors(label, outputs[name], reference, atol=atol, rtol=rtol)
        if not matches:
            status.append(f"{name} FAIL")
    validate_chunks = compiled is not None and "Compiled" in outputs
    del outputs, reference
    if validate_chunks:
        # Validate temporal state too; final posterior agreement is insufficient.
        chunks_match = debug_compiled_chunk_mismatch(eager, compiled, video, atol=COMPARE_ATOL, rtol=COMPARE_RTOL)
        if not chunks_match:
            print("  FAIL Torch compiled chunk outputs/caches", flush=True)
            status.append("Compiled cache FAIL")

    # A single-case command retains the existing Nsight capture semantics.
    # Multi-case Nsight recording needs --capture-range-end=repeat, not stop.
    with cuda_profiler_range(device):
        names = ["Mega"] if args.profile_mega_only else list(paths)
        for name in names:
            with nvtx_range(device, f"{case.label}/{name}"):
                output = paths[name]()
                del output
    result = BenchmarkResult(case, median_timings(paths, args.benchmark_repeats))
    result.status = "; ".join(status) if status else "ALL PASS"
    print(f"  Validation: {result.status}", flush=True)
    return result


def print_summary(results: list[BenchmarkResult], repeats: int) -> None:
    """Print all measured variants; failed or unavailable runs are never hidden."""
    headers = [
        "Resolution",
        "Frames",
        "Batch",
        "Mega (ms)",
        "Diffusers (ms)",
        "Torch (ms)",
        "Compiled (ms)",
        "vs Torch",
        "vs Compiled",
        "Mega frames/s",
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
                *[
                    f"{times[name]:.3f}" if name in times else "-"
                    for name in ("Mega", "Diffusers", "Torch", "Compiled")
                ],
                *[
                    f"{times[name] / mega_ms:.3f}x" if mega_ms and name in times else "-"
                    for name in ("Torch", "Compiled")
                ],
                f"{result.case.batch_size * result.case.frames * 1000 / mega_ms:.1f}" if mega_ms else "-",
                result.status,
            ]
        )
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print(
        f"\nSummary: BF16, resolution=HxW, actual batch shown, median of {repeats} full-encode CUDA-event samples.",
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


def run_case_with_retry(
    args: argparse.Namespace, case: BenchmarkCase, vae: AutoencoderKLWan, device: torch.device
) -> BenchmarkResult:
    """Retry automatic batches after OOM, retaining validation for every variant."""
    while True:
        result = None
        try:
            result = run_case(args, case, vae, device)
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            # Release model caches and graphs after the exception handler drops
            # its traceback, including tensors in the failed run_case frame.
            vae.clear_cache()
            torch._dynamo.reset()
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
    batch_label = (
        f"auto batch (reference N={reference_batch}, T=17, HxW=720x1280, free={free_bytes / 2**30:.1f} GiB)"
        if args.batch_size is None
        else f"fixed N={args.batch_size}"
    )
    print(
        f"Wan VAE | {torch.cuda.get_device_name(device)} | {batch_label} | BF16 | repeats={args.benchmark_repeats}",
        flush=True,
    )
    if args.model is None:
        print(
            "Instantiating WAN 2.2 architecture with deterministic random weights",
            flush=True,
        )
    else:
        print(f"Loading local checkpoint {args.model}", flush=True)
    vae = load_vae(args, device, resolve_dtype(args.dtype))
    if vae.use_tiling:
        raise ValueError("This benchmark requires spatial tiling to be disabled.")
    cases = [
        BenchmarkCase(
            height,
            width,
            frames,
            args.batch_size
            if args.batch_size is not None
            else scaled_batch_size(reference_batch, frames, height, width),
        )
        for height, width in args.resolutions
        for frames in args.frames
    ]
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
    print(
        f"Validation: {variants} vs channels-last Diffusers.\nTolerances: atol={COMPARE_ATOL}, rtol={COMPARE_RTOL}.",
        flush=True,
    )
    if args.torch_compile:
        print("Compiled intermediate caches also allow adjacent finite BF16 values; output tolerances are unchanged.")
    results = []
    try:
        for index, case in enumerate(cases, 1):
            print(f"\n[{index}/{len(cases)}] {case.label}", flush=True)
            results.append(run_case_with_retry(args, case, vae, device))
    finally:
        print_summary(results, args.benchmark_repeats)
    if any(result.status != "ALL PASS" for result in results):
        raise SystemExit("Sweep contains failed validation, compilation, or OOM cases.")


if __name__ == "__main__":
    main()
