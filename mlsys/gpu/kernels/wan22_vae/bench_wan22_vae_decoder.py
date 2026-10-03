#!/usr/bin/env python3

"""Benchmark Wan2.2 decoding: Diffusers, eager Torch, and compiled Torch.

Shapes describe the output video, not the input latent. Latents are seeded
Gaussian tensors; omitting --model uses the production architecture with random
weights. This measures speed and numerical agreement, not reconstruction quality.
All variants use the same channels-last weights and unnormalized latent input.
Full-decode CUDA-event timings include chunk launches, unpatchify, and clamping;
input generation, compilation, warmup, and validation are excluded.

    python bench_wan22_vae_decoder.py --resolutions 480x832 --frames 17
    python bench_wan22_vae_decoder.py --model /path/to/Wan2.2-TI2V-5B-Diffusers

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        python bench_wan22_vae_decoder.py --resolutions 480x832 --frames 17

Use --capture-range-end=repeat to profile multiple cases. Mega is not included.
"""

import argparse
import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from wan_vae_reference import TorchCompileChunkWanVaeDecoder, TorchWanVaeDecoder

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TOLERANCES = {"bf16": (0.05, 0.02), "fp16": (0.01, 0.01), "fp32": (1e-4, 1e-4)}


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", help="Local Diffusers model directory; omit for synthetic weights."
    )
    parser.add_argument(
        "--dtype", choices=DTYPES, default="bf16", help="Weights and latent dtype."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Fixed batch for every case."
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[197],
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Synthetic weights and latent seed."
    )
    parser.add_argument(
        "--benchmark-repeats", type=int, default=5, help="Full-decode timing samples."
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include compiled Torch in validation, timing, and profiling.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.benchmark_repeats < 1:
        parser.error("--batch-size and --benchmark-repeats must be positive")
    if any(t < 1 or (t - 1) % 4 for t in args.frames):
        parser.error("--frames must be positive and of the form 4n+1")
    if any(h % 16 or w % 16 for h, w in args.resolutions):
        parser.error("--resolutions must be divisible by 16")
    return args


def load_vae(args: argparse.Namespace) -> AutoencoderKLWan:
    """Load shared channels-last weights without loading a checkpoint by default."""
    dtype = DTYPES[args.dtype]
    if args.model:
        vae = AutoencoderKLWan.from_pretrained(
            args.model, subfolder="vae", torch_dtype=dtype
        )
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
        )
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    return vae.to(device="cuda", dtype=dtype).eval()


def median_timings(paths: dict[str, Callable], repeats: int) -> dict[str, float]:
    """Alternate execution order and measure complete decodes without CUDA Graphs."""
    samples = {name: [] for name in paths}
    for i in range(repeats):
        order = list(paths) if i % 2 == 0 else list(reversed(paths))
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            output = paths[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del output
    return {name: statistics.median(values) for name, values in samples.items()}


@torch.inference_mode()
def run_case(
    args: argparse.Namespace,
    vae: AutoencoderKLWan,
    frames: int,
    height: int,
    width: int,
) -> Result:
    """Validate all variants, capture one warmed decode each, then benchmark."""
    shape = (args.batch_size, 3, frames, height, width)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    latent = torch.randn(
        args.batch_size,
        vae.config.z_dim,
        1 + (frames - 1) // 4,
        height // 16,
        width // 16,
        generator=generator,
        device="cuda",
        dtype=DTYPES[args.dtype],
    )
    eager = TorchWanVaeDecoder(vae).eval()
    paths = {
        "Diffusers": lambda: vae.decode(latent, return_dict=False)[0],
        "Torch": lambda: eager(latent),
    }
    if args.torch_compile:
        compiled = TorchCompileChunkWanVaeDecoder(vae).eval()
        paths["Compiled"] = lambda: compiled(latent)
    print(f"\nOutput={shape} latent={tuple(latent.shape)}", flush=True)
    expected = paths["Diffusers"]()
    if tuple(expected.shape) != shape:
        raise ValueError(
            f"Checkpoint output shape {tuple(expected.shape)} does not match requested {shape}"
        )
    failures = []
    atol, rtol = TOLERANCES[args.dtype]
    for name in list(paths):
        print(f"  Warmup/validation: {name}", flush=True)
        try:
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
        matches = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        diff = (actual.float() - expected.float()).abs()
        print(
            f"  {'PASS' if matches else 'FAIL'} {name} vs Diffusers: "
            f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}",
            flush=True,
        )
        if not matches:
            failures.append(f"{name} FAIL")
        del actual, diff
        for _ in range(2):
            output = paths[name]()
            del output
    del expected
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
    print(
        "  " + " | ".join(f"{name}={ms:.3f} ms" for name, ms in timings.items()),
        flush=True,
    )
    return Result(shape, timings, "; ".join(failures) if failures else "ALL PASS")


def print_summary(results: list[Result], args: argparse.Namespace) -> None:
    """Report latency and compiled speedup without hiding failed cases."""
    print(
        f"\nSummary: {args.dtype}, median of {args.benchmark_repeats} full-decode samples."
    )
    print(
        "Output N,C,T,H,W | Diffusers (ms) | Torch (ms) | Compiled (ms) | Torch/Compiled | Validation"
    )
    for result in results:
        times = result.timings
        columns = [
            f"{times[name]:.3f}" if name in times else "-"
            for name in ("Diffusers", "Torch", "Compiled")
        ]
        speedup = (
            f"{times['Torch'] / times['Compiled']:.3f}x" if "Compiled" in times else "-"
        )
        print(
            " | ".join([str(result.shape), *columns, speedup, result.status]),
            flush=True,
        )


def main() -> None:
    """Run a sweep, releasing compiled variants and allocations between cases."""
    args = parse_args()
    print(f"Wan2.2 decoder | {torch.cuda.get_device_name()} | {args.dtype}", flush=True)
    print(f"Weights: {args.model or 'synthetic production architecture'}", flush=True)
    print(
        f"Validation tolerances: atol={TOLERANCES[args.dtype][0]}, rtol={TOLERANCES[args.dtype][1]}"
    )
    vae = load_vae(args)
    results = []
    try:
        for height, width in args.resolutions:
            for frames in args.frames:
                try:
                    results.append(run_case(args, vae, frames, height, width))
                except torch.cuda.OutOfMemoryError:
                    print(
                        "  CUDA OOM: retaining requested batch and continuing sweep.",
                        flush=True,
                    )
                    results.append(
                        Result((args.batch_size, 3, frames, height, width), {}, "OOM")
                    )
                finally:
                    torch.compiler.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        print_summary(results, args)
    if any(result.status != "ALL PASS" for result in results):
        raise SystemExit("Sweep contains validation, compilation, or OOM failures.")


if __name__ == "__main__":
    main()
