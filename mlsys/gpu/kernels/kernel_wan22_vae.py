#!/usr/bin/env python3

"""Reference runner for Wan2.2 VAE encode on synthetic video input.

This is intentionally a small PyTorch/Diffusers baseline. Use it to capture
input/output shapes, timing, and numerical reference tensors before replacing a
piece with a custom kernel.

Command to run:
TMPDIR="$PWD/tmp" nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -- python kernel_wan22_vae.py \
       --model ../../../../checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers/
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanAttentionBlock as DiffusersWanAttentionBlock,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResample as DiffusersWanResample,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResidualBlock as DiffusersWanResidualBlock,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResidualDownBlock as DiffusersWanResidualDownBlock,
)

CACHE_T = 2
COMPARE_ATOL = 1e-2
COMPARE_RTOL = 1e-2
COMPILE_COMPARE_ATOL = 0.2
COMPILE_COMPARE_RTOL = 1e-3
T = TypeVar("T")
CacheEntry = torch.Tensor | str | None
FeatCache = tuple[CacheEntry, ...]


@dataclass(frozen=True)
class TensorStats:
    shape: tuple[int, ...]
    dtype: str
    device: str
    minimum: float
    maximum: float
    mean: float
    std: float


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 VAE encode validation on deterministic synthetic video.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to Wan2.2 Diffusers model directory.")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"], help="Torch dtype for VAE weights and input.")
    parser.add_argument("--batch-size", type=int, default=32, help="Synthetic input video batch size.")
    parser.add_argument("--frames", type=int, default=17, help="Synthetic input video frames.")
    parser.add_argument("--height", type=int, default=640, help="Synthetic input video height.")
    parser.add_argument("--width", type=int, default=480, help="Synthetic input video width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic video generation.")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable the torch.compile encoder validation and benchmark path.",
    )
    return parser.parse_args()


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
def cuda_profiler_range(device: torch.device) -> Iterator[None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


def timed(name: str, device: torch.device, fn: Callable[[], T], *, emit_nvtx: bool = False) -> tuple[T, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    range_context = nvtx_range(device, name) if emit_nvtx else nullcontext()
    with range_context:
        start = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def stats(tensor: torch.Tensor) -> TensorStats:
    value = tensor.detach().float()
    return TensorStats(
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        minimum=float(value.min().item()),
        maximum=float(value.max().item()),
        mean=float(value.mean().item()),
        std=float(value.std().item()),
    )


def print_stats(name: str, tensor: torch.Tensor) -> None:
    s = stats(tensor)
    print(
        f"{name}: shape={s.shape} dtype={s.dtype} device={s.device} "
        f"min={s.minimum:.6f} max={s.maximum:.6f} mean={s.mean:.6f} std={s.std:.6f}"
    )


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> bool:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms_abs = float(torch.sqrt(torch.mean(diff * diff)).item())
    allclose = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(
        f"{name}: allclose={allclose} atol={atol:g} rtol={rtol:g} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}"
    )
    return bool(allclose)


def print_performance(timings: dict[str, float]) -> None:
    print("performance:")
    for name, elapsed_ms in timings.items():
        print(f"  {name}: {elapsed_ms:.3f} ms")


def raw_patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")

    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size}).")

    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)


def validate_frame_count(frames: int) -> None:
    if frames != 1 and (frames - 1) % 4 != 0:
        raise ValueError(f"Input video frames must be 1 or 4n+1 for Wan VAE temporal chunking, got {frames}.")


def raw_causal_conv3d(conv: torch.nn.Conv3d, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
    padding = list(conv._padding)
    if cache_x is not None and padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return F.conv3d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def raw_update_cache(feat_cache: FeatCache, idx: int, value: CacheEntry) -> FeatCache:
    next_cache = list(feat_cache)
    next_cache[idx] = value
    return tuple(next_cache)


def raw_contiguous_clone(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor.clone()
    return tensor.contiguous()


def raw_canonicalize_cache(feat_cache: FeatCache) -> FeatCache:
    return tuple(entry.contiguous() if isinstance(entry, torch.Tensor) else entry for entry in feat_cache)


def raw_cached_causal_conv3d(
    conv: torch.nn.Module,
    x: torch.Tensor,
    feat_cache: FeatCache | None,
    feat_idx: int,
) -> tuple[torch.Tensor, FeatCache | None, int]:
    if feat_cache is None:
        return conv(x), feat_cache, feat_idx

    idx = feat_idx
    cache_x = raw_contiguous_clone(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
        prev = feat_cache[idx]
        if isinstance(prev, str):
            raise TypeError(f"Unexpected string cache entry for causal conv: {prev!r}")
        cache_x = torch.cat([prev[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

    x = conv(x, feat_cache[idx])
    return x, raw_update_cache(feat_cache, idx, cache_x), feat_idx + 1


def raw_rms_norm(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
        token in str(x.dtype) for token in ("float4_", "float8_")
    )
    normalized = F.normalize(
        x.float() if needs_fp32_normalize else x,
        dim=(1 if norm.channel_first else -1),
    ).to(x.dtype)
    return normalized * norm.scale * norm.gamma + norm.bias


def raw_conv2d(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def raw_spatial_resample(resample: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    spatial = resample.resample
    if isinstance(spatial, torch.nn.Identity):
        return x

    if resample.mode.startswith("downsample"):
        conv = spatial[1]
        return raw_conv2d(conv, F.pad(x, spatial[0].padding))

    if resample.mode.startswith("upsample"):
        conv = spatial[1]
        x = F.interpolate(x.float(), scale_factor=spatial[0].scale_factor, mode=spatial[0].mode).type_as(x)
        return raw_conv2d(conv, x)

    raise NotImplementedError(f"Unsupported resample mode: {resample.mode}")


def raw_avg_down3d(avg: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    pad_t = (avg.factor_t - x.shape[2] % avg.factor_t) % avg.factor_t
    x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames // avg.factor_t,
        avg.factor_t,
        height // avg.factor_s,
        avg.factor_s,
        width // avg.factor_s,
        avg.factor_s,
    )
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(batch_size, channels * avg.factor, frames // avg.factor_t, height // avg.factor_s, width // avg.factor_s)
    x = x.view(
        batch_size,
        avg.out_channels,
        avg.group_size,
        frames // avg.factor_t,
        height // avg.factor_s,
        width // avg.factor_s,
    )
    return x.mean(dim=2)


class RawWanCausalConv3d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        return raw_causal_conv3d(self.conv, x, cache_x)


class RawWanRMSNorm(torch.nn.Module):
    def __init__(self, norm: torch.nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return raw_rms_norm(self.norm, x)


class RawWanResample(torch.nn.Module):
    def __init__(self, resample: DiffusersWanResample) -> None:
        super().__init__()
        self.mode = resample.mode
        self.resample = resample.resample
        self.time_conv = RawWanCausalConv3d(resample.time_conv) if hasattr(resample, "time_conv") else None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        batch_size, channels, frames, height, width = x.shape

        if self.mode == "upsample3d" and feat_cache is not None:
            if self.time_conv is None:
                raise RuntimeError("upsample3d is missing time_conv.")
            idx = feat_idx
            entry = feat_cache[idx]
            if entry is None:
                feat_cache = raw_update_cache(feat_cache, idx, "Rep")
                feat_idx += 1
            else:
                is_rep = isinstance(entry, str) and entry == "Rep"
                cache_x = raw_contiguous_clone(x[:, :, -CACHE_T:, :, :])
                if cache_x.shape[2] < CACHE_T and isinstance(entry, torch.Tensor):
                    cache_x = torch.cat([entry[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < CACHE_T and is_rep:
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)

                conv_cache = None if is_rep else entry
                x = self.time_conv(x, conv_cache)
                feat_cache = raw_update_cache(feat_cache, idx, cache_x)
                feat_idx += 1

                x = x.reshape(batch_size, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(batch_size, channels, frames * 2, height, width)

        frames = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = raw_spatial_resample(self, x)
        x = x.view(batch_size, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d" and feat_cache is not None:
            if self.time_conv is None:
                raise RuntimeError("downsample3d is missing time_conv.")
            idx = feat_idx
            entry = feat_cache[idx]
            if entry is None:
                feat_cache = raw_update_cache(feat_cache, idx, raw_contiguous_clone(x))
                feat_idx += 1
            else:
                if not isinstance(entry, torch.Tensor):
                    raise TypeError(f"Unexpected cache entry for downsample3d: {entry!r}")
                cache_x = raw_contiguous_clone(x[:, :, -1:, :, :])
                x = self.time_conv(torch.cat([entry[:, :, -1:, :, :], x], 2))
                feat_cache = raw_update_cache(feat_cache, idx, cache_x)
                feat_idx += 1

        return x, feat_cache, feat_idx


class RawWanResidualBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanResidualBlock) -> None:
        super().__init__()
        self.norm1 = RawWanRMSNorm(block.norm1)
        self.conv1 = RawWanCausalConv3d(block.conv1)
        self.norm2 = RawWanRMSNorm(block.norm2)
        self.dropout = block.dropout
        self.conv2 = RawWanCausalConv3d(block.conv2)
        self.conv_shortcut = None if isinstance(block.conv_shortcut, torch.nn.Identity) else RawWanCausalConv3d(block.conv_shortcut)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)

        x = self.norm1(x)
        x = F.silu(x)
        x, feat_cache, feat_idx = raw_cached_causal_conv3d(self.conv1, x, feat_cache, feat_idx)

        x = self.norm2(x)
        x = F.silu(x)
        x = F.dropout(x, p=self.dropout.p, training=self.dropout.training)
        x, feat_cache, feat_idx = raw_cached_causal_conv3d(self.conv2, x, feat_cache, feat_idx)
        return x + residual, feat_cache, feat_idx


class RawWanAttentionBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanAttentionBlock) -> None:
        super().__init__()
        self.norm = RawWanRMSNorm(block.norm)
        self.to_qkv = block.to_qkv
        self.proj = block.proj

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        identity = x
        batch_size, channels, frames, height, width = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.norm(x)

        qkv = raw_conv2d(self.to_qkv, x)
        qkv = qkv.reshape(batch_size * frames, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * frames, channels, height, width)
        x = raw_conv2d(self.proj, x)
        x = x.view(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity, feat_cache, feat_idx


class RawWanMidBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(RawWanResidualBlock(resnet) for resnet in block.resnets)
        self.attentions = torch.nn.ModuleList(RawWanAttentionBlock(attention) for attention in block.attentions)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        x, feat_cache, feat_idx = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            x, feat_cache, feat_idx = attention(x, feat_cache=feat_cache, feat_idx=feat_idx)
            x, feat_cache, feat_idx = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x, feat_cache, feat_idx


class RawWanResidualDownBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanResidualDownBlock) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(RawWanResidualBlock(resnet) for resnet in block.resnets)
        self.downsampler = RawWanResample(block.downsampler) if block.downsampler is not None else None
        self.avg_shortcut = block.avg_shortcut

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        shortcut = x.clone()
        for resnet in self.resnets:
            x, feat_cache, feat_idx = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x, feat_cache, feat_idx = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + raw_avg_down3d(self.avg_shortcut, shortcut), feat_cache, feat_idx


def raw_wan_encoder_layer(layer: torch.nn.Module) -> torch.nn.Module:
    if isinstance(layer, DiffusersWanResidualDownBlock):
        return RawWanResidualDownBlock(layer)
    if isinstance(layer, DiffusersWanResidualBlock):
        return RawWanResidualBlock(layer)
    if isinstance(layer, DiffusersWanAttentionBlock):
        return RawWanAttentionBlock(layer)
    if isinstance(layer, DiffusersWanResample):
        return RawWanResample(layer)
    raise NotImplementedError(f"Unsupported encoder layer: {layer.__class__.__name__}")


class RawWanEncoder3d(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.conv_in = RawWanCausalConv3d(encoder.conv_in)
        self.down_blocks = torch.nn.ModuleList(raw_wan_encoder_layer(layer) for layer in encoder.down_blocks)
        self.mid_block = RawWanMidBlock(encoder.mid_block)
        self.norm_out = RawWanRMSNorm(encoder.norm_out)
        self.conv_out = RawWanCausalConv3d(encoder.conv_out)
        self.cache_size = raw_causal_conv3d_count(encoder)

    def forward(self, x: torch.Tensor, feat_cache: FeatCache) -> tuple[torch.Tensor, FeatCache]:
        x = x.contiguous()
        feat_cache = raw_canonicalize_cache(feat_cache)
        feat_idx = 0

        x, feat_cache, feat_idx = raw_cached_causal_conv3d(self.conv_in, x, feat_cache, feat_idx)
        assert feat_cache is not None
        for layer in self.down_blocks:
            x, feat_cache, feat_idx = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x, feat_cache, feat_idx = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        assert feat_cache is not None
        x = self.norm_out(x)
        x = F.silu(x)
        x, feat_cache, _ = raw_cached_causal_conv3d(self.conv_out, x, feat_cache, feat_idx)
        assert feat_cache is not None
        return x, feat_cache


def raw_causal_conv3d_count(module: torch.nn.Module) -> int:
    return sum(1 for child in module.modules() if child.__class__.__name__ == "WanCausalConv3d")


class TorchWanVaeEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.encoder = RawWanEncoder3d(vae.encoder)
        self.quant_conv = RawWanCausalConv3d(vae.quant_conv)
        self.patch_size = getattr(vae.config, "patch_size", None)
        self.use_tiling = vae.use_tiling
        self.tile_sample_min_width = vae.tile_sample_min_width
        self.tile_sample_min_height = vae.tile_sample_min_height

    def new_cache(self) -> FeatCache:
        return tuple([None] * self.encoder.cache_size)

    def encode_chunk(self, chunk: torch.Tensor, feat_cache: FeatCache) -> tuple[torch.Tensor, FeatCache]:
        return self.encoder(chunk, feat_cache)

    def encode_temporal_chunks(
        self,
        video: torch.Tensor,
        chunk_encoder: Callable[[torch.Tensor, FeatCache], tuple[torch.Tensor, FeatCache]] | None = None,
    ) -> torch.Tensor:
        _, _, num_frame, height, width = video.shape
        validate_frame_count(num_frame)
        if self.patch_size is not None:
            video = raw_patchify(video, patch_size=int(self.patch_size))

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            raise NotImplementedError("raw PyTorch encode does not implement VAE tiling yet.")

        if chunk_encoder is None:
            chunk_encoder = self.encode_chunk

        feat_cache = self.new_cache()
        outs: list[torch.Tensor] = []
        iterations = 1 + (num_frame - 1) // 4
        for i in range(iterations):
            if i == 0:
                chunk = video[:, :, :1, :, :]
            else:
                chunk = video[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            chunk_out, feat_cache = chunk_encoder(chunk, feat_cache)
            outs.append(chunk_out)

        out = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
        return self.quant_conv(out)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encode_temporal_chunks(video)


class TorchCompileChunkWanVaeEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.raw_encoder = TorchWanVaeEncoder(vae).eval()
        self.chunk_encoder = torch.compile(self.raw_encoder.encoder, fullgraph=True, dynamic=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.raw_encoder.encode_temporal_chunks(video, self.chunk_encoder)


def posterior_mode(parameters: torch.Tensor) -> torch.Tensor:
    return torch.chunk(parameters, 2, dim=1)[0]


def cache_state_name(feat_cache: FeatCache) -> str:
    first_tensor = next((entry for entry in feat_cache if isinstance(entry, torch.Tensor)), None)
    if first_tensor is None:
        return "cache_t=0"
    return f"cache_t={first_tensor.shape[2]}"


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.sqrt(torch.mean(diff * diff)).item()),
    )


def debug_compiled_chunk_mismatch(
    torch_encoder: TorchWanVaeEncoder,
    compiled_torch_encoder: TorchCompileChunkWanVaeEncoder,
    video: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    print("torch_compile_production_gate:")
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

        cache_in = raw_canonicalize_cache(feat_cache)
        eager_out, eager_cache = torch_encoder.encode_chunk(chunk, cache_in)
        compiled_out, compiled_cache = compiled_torch_encoder.chunk_encoder(chunk, cache_in)

        if not torch.allclose(compiled_out, eager_out, atol=atol, rtol=rtol):
            max_abs, mean_abs, rms_abs = tensor_error(compiled_out, eager_out)
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} output mismatch "
                f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}"
            )
            return False

        if len(compiled_cache) != len(eager_cache):
            print(
                f"  rejected: chunk[{i}] {cache_state_name(cache_in)} cache length mismatch "
                f"compiled={len(compiled_cache)} eager={len(eager_cache)}"
            )
            return False

        for cache_idx, (actual, expected) in enumerate(zip(compiled_cache, eager_cache)):
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                    max_abs, mean_abs, rms_abs = tensor_error(actual, expected)
                    print(
                        f"  rejected: chunk[{i}] {cache_state_name(cache_in)} cache[{cache_idx}] mismatch "
                        f"shape={tuple(actual.shape)} max_abs={max_abs:.6e} "
                        f"mean_abs={mean_abs:.6e} rms_abs={rms_abs:.6e}"
                    )
                    return False
            elif actual != expected:
                print(
                    f"  rejected: chunk[{i}] {cache_state_name(cache_in)} cache[{cache_idx}] mismatch "
                    f"compiled={type(actual).__name__} eager={type(expected).__name__}"
                )
                return False

        feat_cache = eager_cache

    print("  accepted: compiled chunk outputs and returned caches match eager.")
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
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(0.0, 1.0, frames, dtype=torch.float32).view(1, 1, frames, 1, 1)
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32).view(1, 1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, 1, width)

    target_shape = (1, 1, frames, height, width)
    base_channels = [
        torch.sin(2.0 * math.pi * (x + t)).expand(target_shape),
        torch.cos(2.0 * math.pi * (y - t)).expand(target_shape),
        torch.sin(math.pi * (x + y + 0.5 * t)).expand(target_shape),
    ]
    while len(base_channels) < channels:
        base_channels.append(torch.randn(1, 1, frames, height, width, generator=generator))

    video = torch.cat(base_channels[:channels], dim=1)
    video = video.expand(batch_size, channels, frames, height, width).contiguous()
    noise = 0.02 * torch.randn(video.shape, generator=generator, dtype=video.dtype)
    return torch.clamp(video + noise, -1.0, 1.0)


def load_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> AutoencoderKLWan:
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    return vae.to(device=device).eval()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)

    print(f"loading VAE: model={args.model!r} device={device} dtype={dtype}")
    vae = load_vae(args, device, dtype)
    patch_size = getattr(vae.config, "patch_size", None)
    spatial = getattr(vae.config, "scale_factor_spatial", getattr(vae, "spatial_compression_ratio", None))
    temporal = getattr(vae.config, "scale_factor_temporal", None)
    sample_channels = sample_video_channels(vae)
    print(
        "vae config: "
        f"z_dim={getattr(vae.config, 'z_dim', None)} "
        f"sample_channels={sample_channels} "
        f"patch_size={patch_size} spatial_scale={spatial} temporal_scale={temporal} "
        f"tiling={vae.use_tiling}"
    )

    if patch_size is not None and (args.height % int(patch_size) != 0 or args.width % int(patch_size) != 0):
        raise ValueError(f"height and width must be divisible by VAE patch_size={patch_size}.")
    if spatial is not None and (args.height % int(spatial) != 0 or args.width % int(spatial) != 0):
        raise ValueError(f"height and width must be divisible by VAE spatial_scale={spatial}.")
    if vae.use_tiling:
        raise ValueError("The raw PyTorch and torch.compile validation paths do not implement VAE tiling yet.")
    validate_frame_count(args.frames)

    video = make_synthetic_video(
        batch_size=args.batch_size,
        channels=sample_channels,
        frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
    ).to(device=device, dtype=dtype)
    torch_encoder = TorchWanVaeEncoder(vae).eval()

    with torch.inference_mode():
        print_stats("input_video", video)

        compiled_error: Exception | None = None
        compiled_torch_encoder = None
        if args.torch_compile:
            try:
                compiled_torch_encoder = TorchCompileChunkWanVaeEncoder(vae).eval()
            except Exception as error:  # noqa: BLE001 - report compile failure after printing benchmark timings.
                compiled_error = error
                print(f"torch_compile_create_error: {type(error).__name__}: {error}")
        else:
            print("torch_compile: disabled (use --torch-compile to enable)")

        print("warmup:")
        _, warmup_reference_ms = timed(
            "warmup_reference_encode",
            device,
            lambda: vae.encode(video, return_dict=False)[0],
        )
        print(f"  reference_encode: {warmup_reference_ms:.3f} ms")

        _, warmup_torch_ms = timed("warmup_torch_encode", device, lambda: torch_encoder(video))
        print(f"  torch_encode: {warmup_torch_ms:.3f} ms")

        if args.torch_compile:
            if compiled_torch_encoder is not None:
                try:
                    _, warmup_compile_ms = timed(
                        "warmup_torch_compile_encode",
                        device,
                        lambda: compiled_torch_encoder(video),
                    )
                    print(f"  torch_compile_encode: {warmup_compile_ms:.3f} ms")
                except Exception as error:  # noqa: BLE001 - report compile failure after printing benchmark timings.
                    compiled_error = error
                    print(f"  torch_compile_encode_error: {type(error).__name__}: {error}")
            else:
                print("  torch_compile_encode: skipped because torch.compile creation failed.")

        benchmark_timings: dict[str, float] = {}
        reference_posterior = None
        torch_params = None
        compiled_params = None
        with cuda_profiler_range(device):
            reference_posterior, benchmark_timings["benchmark_reference_encode"] = timed(
                "benchmark_reference_encode",
                device,
                lambda: vae.encode(video, return_dict=False)[0],
                emit_nvtx=True,
            )

            torch_params, benchmark_timings["benchmark_torch_encode"] = timed(
                "benchmark_torch_encode",
                device,
                lambda: torch_encoder(video),
                emit_nvtx=True,
            )

            if compiled_torch_encoder is not None and compiled_error is None:
                try:
                    compiled_params, benchmark_timings["benchmark_torch_compile_encode"] = timed(
                        "benchmark_torch_compile_encode",
                        device,
                        lambda: compiled_torch_encoder(video),
                        emit_nvtx=True,
                    )
                except Exception as error:  # noqa: BLE001 - report compile failure after printing benchmark timings.
                    compiled_error = error
                    print(f"benchmark_torch_compile_encode_error: {type(error).__name__}: {error}")

        assert reference_posterior is not None
        assert torch_params is not None
        reference_params = reference_posterior.parameters
        reference_latents = reference_posterior.mode()
        torch_latents = posterior_mode(torch_params)
        print_stats("reference_latents", reference_latents)
        print_stats("torch_latents", torch_latents)
        if compiled_params is not None:
            compiled_latents = posterior_mode(compiled_params)
            print_stats("torch_compile_latents", compiled_latents)
        else:
            compiled_latents = None

        print("validation:")
        print(f"  raw_vs_reference: fatal atol={COMPARE_ATOL:g} rtol={COMPARE_RTOL:g}")
        if args.torch_compile:
            print(
                f"  torch_compile_vs_reference: production_gate atol={COMPILE_COMPARE_ATOL:g} "
                f"rtol={COMPILE_COMPARE_RTOL:g}; rejected compiled chunks fall back to eager"
            )
        raw_matches = [
            compare_tensors(
                "torch_encoded_params",
                torch_params,
                reference_params,
                atol=COMPARE_ATOL,
                rtol=COMPARE_RTOL,
            ),
            compare_tensors(
                "torch_latents_mode",
                torch_latents,
                reference_latents,
                atol=COMPARE_ATOL,
                rtol=COMPARE_RTOL,
            ),
        ]
        compiled_matches: list[bool] = []
        if args.torch_compile:
            if compiled_error is None:
                assert compiled_params is not None
                assert compiled_latents is not None
                compiled_matches.extend(
                    [
                        compare_tensors(
                            "torch_compile_encoded_params",
                            compiled_params,
                            reference_params,
                            atol=COMPILE_COMPARE_ATOL,
                            rtol=COMPILE_COMPARE_RTOL,
                        ),
                        compare_tensors(
                            "torch_compile_latents_mode",
                            compiled_latents,
                            reference_latents,
                            atol=COMPILE_COMPARE_ATOL,
                            rtol=COMPILE_COMPARE_RTOL,
                        ),
                    ]
                )
                compiled_chunks_match = debug_compiled_chunk_mismatch(
                    torch_encoder,
                    compiled_torch_encoder,
                    video,
                    atol=COMPILE_COMPARE_ATOL,
                    rtol=COMPILE_COMPARE_RTOL,
                )
                if all(compiled_matches) and compiled_chunks_match:
                    print("torch_compile_status: accepted")
                else:
                    print(
                        "torch_compile_status: rejected_expected "
                        "(production-style behavior would fall back to eager for this chunk shape)"
                    )
            else:
                print(
                    "torch_compile_status: unavailable "
                    "(production-style behavior would fall back to eager because compilation failed)"
                )

        print_performance(benchmark_timings)
        if compiled_error is not None:
            print(f"torch_compile_error: {type(compiled_error).__name__}: {compiled_error}")
        if not all(raw_matches):
            raise AssertionError("Raw PyTorch Wan2.2 VAE validation comparisons failed.")


if __name__ == "__main__":
    main()
