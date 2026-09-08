#!/usr/bin/env python3
"""Standalone eager-PyTorch reference for Wan2.2 VAE encoding.

This file intentionally has no imaginaire4 imports. It is meant to be copied
outside the repo and used as a correctness/perf reference for the eager encoder.

Example:
    python wan2pt2_vae_encode_reference.py \
        --checkpoint /path/to/Wan2.2_VAE.pth \
        --height 480 --width 832 --frames 17 \
        --encode-exact-durations 17 \
        --iters 20
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_T: int = 2


def _contiguous_clone(t: torch.Tensor) -> torch.Tensor:  # t: [...], returns [...]
    if t.is_contiguous():
        return t.clone()  # [...]
    return t.contiguous()  # [...]


def _update_cache_and_apply(
    x: torch.Tensor,  # [B,C,T,H,W]
    layer: "CausalConv3d",
    feat_cache: list[torch.Tensor | None],
    feat_idx: list[int],
) -> torch.Tensor:  # returns [B,C_out,T_out,H_out,W_out]
    idx = feat_idx[0]
    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])  # [B,C,<=2,H,W]
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat(
            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
            dim=2,
        )  # [B,C,2,H,W]
    x = layer(x, feat_cache[idx])  # [B,C_out,T_out,H_out,W_out]
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x  # [B,C_out,T_out,H_out,W_out]


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(
        self,
        x: torch.Tensor,  # [B,C,T,H,W]
        cache_x: torch.Tensor | None = None,  # [B,C,<=2,H,W] or None
    ) -> torch.Tensor:  # returns [B,C_out,T_out,H_out,W_out]
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)  # [B,C,<=2,H,W]
            x = torch.cat([cache_x, x], dim=2)  # [B,C,T+cache_T,H,W]
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)  # [B,C,T_pad,H_pad,W_pad]
        return super().forward(x)  # [B,C_out,T_out,H_out,W_out]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,...], returns [B,C,...]
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias  # [B,C,...]


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # [B,C,T,H,W]
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:  # returns [B,C,T_out,H_out,W_out]
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = torch.zeros(b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype)  # [B,C,2,H,W]
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])  # [B,C,<=2,H,W]
                    if cache_x.shape[2] < 2:
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x],
                            dim=2,
                        )  # [B,C,2,H,W]
                    x = self.time_conv(x, feat_cache[idx])  # [B,2*C,T,H,W]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)  # [B,2,C,T,H,W]
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)  # [B,C,T,2,H,W]
                    x = x.reshape(b, c, t * 2, h, w)  # [B,C,2*T,H,W]

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, x.shape[1], x.shape[3], x.shape[4])  # [B*T,C,H,W]
        x = self.resample(x)  # [B*T,C,H_out,W_out]
        _, c2, h2, w2 = x.shape
        x = x.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H_out,W_out]

        if self.mode == "downsample3d":
            if feat_cache is None:
                x = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H,W]
                x = self.time_conv(x)  # [B,C,T_out,H,W]
            else:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    if x.shape[2] == 1:
                        feat_cache[idx] = _contiguous_clone(x)  # [B,C,1,H,W]
                    else:
                        cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H,W]
                        x_in = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H,W]
                        x = self.time_conv(x_in)  # [B,C,T_out,H,W]
                        feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H,W]
                    x_cat = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)  # [B,C,T+1,H,W]
                    if x_cat.shape[2] < 3:
                        x_cat = F.pad(x_cat, (0, 0, 0, 0, 3 - x_cat.shape[2], 0))  # [B,C,3,H,W]
                    x = self.time_conv(x_cat)  # [B,C,T_out,H,W]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x  # [B,C,T_out,H_out,W_out]


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # [B,C,T,H,W]
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:  # returns [B,C_out,T,H,W]
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)  # [B,C_out,T,H,W]
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)  # [B,C_out,T,H,W]
            else:
                x = layer(x)  # [B,C_out,T,H,W]
        return x + h  # [B,C_out,T,H,W]


class AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,T,H,W], returns [B,C,T,H,W]
        identity = x  # [B,C,T,H,W]
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  # [B*T,C,H,W]
        x = self.norm(x)  # [B*T,C,H,W]
        qkv = self.to_qkv(x).reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2).contiguous()  # [B*T,1,H*W,3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # [B*T,1,H*W,C]
        x = F.scaled_dot_product_attention(q, k, v)  # [B*T,1,H*W,C]
        x = x.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w)  # [B*T,C,H,W]
        x = self.proj(x)  # [B*T,C,H,W]
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
        return x + identity  # [B,C,T,H,W]


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:  # x: [B,C,H,W] or [B,C,T,H,W]
    if patch_size == 1:
        return x  # [...]
    if patch_size != 2:
        raise ValueError("This reference only implements the patch_size=2 path used by Wan2.2 VAE.")
    if x.dim() == 4:
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)  # [B,C,H/2,2,W/2,2]
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()  # [B,C,2,2,H/2,W/2]
        return x.view(b, c * 4, h // 2, w // 2)  # [B,4*C,H/2,W/2]
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        x = x.view(b, c, t, h // 2, 2, w // 2, 2)  # [B,C,T,H/2,2,W/2,2]
        x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()  # [B,C,2,2,T,H/2,W/2]
        return x.view(b, c * 4, t, h // 2, w // 2)  # [B,4*C,T,H/2,W/2]
    raise ValueError(f"Invalid input shape: {tuple(x.shape)}")


class AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,T,H,W], returns [B,C_out,T_out,H_out,W_out]
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))  # [B,C,T_pad,H,W]
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )  # [B,C,T/ft,ft,H/fs,fs,W/fs,fs]
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B,C,ft,fs,fs,T/ft,H/fs,W/fs]
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )  # [B,C*factor,T/ft,H/fs,W/fs]
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )  # [B,C_out,group,T/ft,H/fs,W/fs]
        return x.mean(dim=2)  # [B,C_out,T/ft,H/fs,W/fs]


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples: list[nn.Module] = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(
        self,
        x: torch.Tensor,  # [B,C,T,H,W]
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:  # returns [B,C_out,T_out,H_out,W_out]
        if feat_idx is None:
            feat_idx = [0]
        x_shortcut = self.avg_shortcut(x)  # [B,C_out,T_out,H_out,W_out]
        for module in self.downsamples:
            if feat_cache is not None:
                x = module(x, feat_cache, feat_idx)  # [B,C_out,T_out,H_out,W_out]
            else:
                x = module(x)  # [B,C_out,T_out,H_out,W_out]
        return x + x_shortcut  # [B,C_out,T_out,H_out,W_out]


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 160,
        z_dim: int = 96,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temporal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temporal_downsample[i] if i < len(temporal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], dropout),
        )
        self.head = nn.Sequential(RMSNorm(dims[-1], images=False), nn.SiLU(), CausalConv3d(dims[-1], z_dim, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,  # [B,12,T,H/2,W/2]
        feat_cache: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:  # returns [B,z_dim,T/4,H/16,W/16]
        feat_idx = [0]
        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)  # [B,C,T,H,W]
        else:
            x = self.conv1(x)  # [B,C,T,H,W]
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)  # [B,C,T,H,W]
            else:
                x = layer(x)  # [B,C,T,H,W]
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)  # [B,C,T,H,W]
            else:
                x = layer(x)  # [B,C,T,H,W]
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)  # [B,z_dim,T,H,W]
            else:
                x = layer(x)  # [B,z_dim,T,H,W]
        return x  # [B,z_dim,T/4,H/16,W/16]


def count_conv3d(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


class WanVAEEncoderOnly(nn.Module):
    def __init__(
        self,
        z_dim: int = 48,
        temporal_window: int | Mapping[str, int] = 24,
        encode_exact_durations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.temporal_window = temporal_window
        self._encode_exact_durations = set(encode_exact_durations or [])
        self.encoder = Encoder3d(z_dim=z_dim * 2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self._enc_conv_num = count_conv3d(self.encoder)

    def _new_enc_cache(self) -> list[torch.Tensor | None]:
        return [None] * self._enc_conv_num

    def _normalize_latent(
        self,
        z: torch.Tensor,  # [B,48,T,H,W]
        scale: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:  # returns [B,48,T,H,W]
        s0 = scale[0].view(1, self.z_dim, 1, 1, 1)  # [1,48,1,1,1]
        s1 = scale[1].view(1, self.z_dim, 1, 1, 1)  # [1,48,1,1,1]
        return (z - s0) * s1  # [B,48,T,H,W]

    def _encode_chunk_impl(
        self,
        x_chunk: torch.Tensor,  # [B,12,T,H/2,W/2]
        feat_cache: list[torch.Tensor | None],
        scale: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:  # returns ([B,48,T/4,H/16,W/16], cache)
        feat_cache = list(feat_cache)
        x_chunk = x_chunk.contiguous()  # [B,12,T,H/2,W/2]
        feat_cache = [c.contiguous() if c is not None else None for c in feat_cache]
        out = self.encoder(x_chunk, feat_cache=feat_cache)  # [B,96,T/4,H/16,W/16]
        mu, _log_var = self.conv1(out).chunk(2, dim=1)  # [B,48,T/4,H/16,W/16]
        return self._normalize_latent(mu, scale), feat_cache

    def encode(
        self,
        x: torch.Tensor,  # [B,3,T,H,W]
        scale: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:  # returns [B,48,T_latent,H/16,W/16]
        t_in, h, w = x.shape[2], x.shape[3], x.shape[4]
        temporal_window = self.temporal_window
        if isinstance(temporal_window, Mapping):
            raise ValueError("Standalone reference expects --temporal-window as an int.")
        if not (t_in == 1 or (t_in - 1) % 4 == 0):
            raise ValueError(f"Input temporal length must be 1 or 4n+1, got {t_in}.")
        latent_t = 1 + (t_in - 1) // 4
        t = t_in
        should_pad = t_in not in self._encode_exact_durations
        if should_pad:
            t = 1 + ((t_in - 1 + temporal_window - 1) // temporal_window) * temporal_window
            x = F.pad(x, (0, 0, 0, 0, 0, t - t_in))  # [B,3,T_pad,H,W]
        enc_cache = self._new_enc_cache()
        x = patchify(x, patch_size=2)  # [B,12,T,H/2,W/2]
        outs: list[torch.Tensor] = []
        out, enc_cache = self._encode_chunk_impl(x[:, :, :1], enc_cache, scale)  # [B,48,1,H/16,W/16]
        outs.append(out)
        for start in range(1, t, temporal_window):
            x_chunk = x[:, :, start : start + temporal_window]  # [B,12,T_chunk,H/2,W/2]
            out, enc_cache = self._encode_chunk_impl(x_chunk, enc_cache, scale)  # [B,48,T_latent_chunk,H/16,W/16]
            outs.append(out)
        final_out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]  # [B,48,T_latent_pad,H/16,W/16]
        if should_pad:
            final_out = final_out[:, :, :latent_t]  # [B,48,T_latent,H/16,W/16]
        return final_out  # [B,48,T_latent,H/16,W/16]


WAN22_MEAN: list[float] = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]


WAN22_STD: list[float] = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


class WanVAEEncodeReference(nn.Module):
    def __init__(
        self,
        checkpoint: str | None,
        device: torch.device,
        dtype: torch.dtype,
        temporal_window: int,
        encode_exact_durations: list[int] | None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        mean = torch.tensor(WAN22_MEAN, dtype=dtype, device=device)  # [48]
        std = torch.tensor(WAN22_STD, dtype=dtype, device=device)  # [48]
        self.scale = (mean, 1.0 / std)
        self.model = WanVAEEncoderOnly(
            temporal_window=temporal_window,
            encode_exact_durations=encode_exact_durations,
        )
        if checkpoint is not None:
            load_encoder_checkpoint(self.model, checkpoint)
        self.model = self.model.eval().requires_grad_(False).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:  # videos: [B,3,T,H,W], returns [B,48,T_latent,H/16,W/16]
        in_dtype = videos.dtype
        videos = videos.to(dtype=self.dtype)  # [B,3,T,H,W]
        latent = self.model.encode(videos, self.scale)  # [B,48,T_latent,H/16,W/16]
        return latent.to(dtype=in_dtype)  # [B,48,T_latent,H/16,W/16]


def _torch_load(path: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unwrap_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict-like state_dict, got {type(obj).__name__}")
    state: object = obj
    for key in ("state_dict", "model_state_dict", "module"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint payload did not contain a state_dict, got {type(state).__name__}")
    return {str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _best_prefix_stripped_state_dict(
    state: dict[str, torch.Tensor],
    required_keys: set[str],
) -> dict[str, torch.Tensor]:
    prefixes = ("", "model.", "model.model.", "module.", "vae.", "first_stage_model.")
    best: dict[str, torch.Tensor] = {}
    best_hits = -1
    for prefix in prefixes:
        converted: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if prefix and not key.startswith(prefix):
                continue
            converted_key = key[len(prefix) :] if prefix else key
            converted[converted_key] = value
        hits = len(set(converted) & required_keys)
        if hits > best_hits:
            best = converted
            best_hits = hits
    return best


def load_encoder_checkpoint(model: WanVAEEncoderOnly, checkpoint: str) -> None:
    state = _unwrap_state_dict(_torch_load(checkpoint))
    required = set(model.state_dict())
    converted = _best_prefix_stripped_state_dict(state, required)
    filtered = {key: value for key, value in converted.items() if key in required}
    missing = sorted(required - set(filtered))
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Checkpoint is missing {len(missing)} encoder keys, first missing: {preview}")
    unexpected = len(converted) - len(filtered)
    model.load_state_dict(filtered, strict=True)
    print(f"Loaded encoder weights from {checkpoint} ({len(filtered)} tensors, ignored {unexpected} non-encoder tensors).")


def _parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Local Wan2.2_VAE.pth path. If omitted, uses random weights.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--temporal-window", type=int, default=24)
    parser.add_argument("--encode-exact-durations", type=int, nargs="*", default=[17])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--channels-last-3d", action="store_true", help="Create input in channels_last_3d memory format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    model = WanVAEEncodeReference(
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
        temporal_window=args.temporal_window,
        encode_exact_durations=args.encode_exact_durations,
    )
    x = torch.rand(args.batch_size, 3, args.frames, args.height, args.width, device=device, dtype=dtype)  # [B,3,T,H,W]
    x = x.mul(2.0).sub(1.0)  # [B,3,T,H,W]
    if args.channels_last_3d:
        x = x.contiguous(memory_format=torch.channels_last_3d)  # [B,3,T,H,W]

    with torch.inference_mode():
        for _ in range(args.warmup):
            y = model.encode(x)  # [B,48,T_latent,H/16,W/16]
        torch.cuda.synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times_ms: list[float] = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y = model.encode(x)  # [B,48,T_latent,H/16,W/16]
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))

    mean_ms, std_ms = _mean_std(times_ms)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    print(f"input_shape={tuple(x.shape)} dtype={x.dtype} channels_last_3d={x.is_contiguous(memory_format=torch.channels_last_3d)}")
    print(f"latent_shape={tuple(y.shape)} dtype={y.dtype}")
    print(f"encode_ms={mean_ms:.3f} std_ms={std_ms:.3f} throughput_fps={args.frames / (mean_ms / 1000.0):.2f}")
    print(f"peak_allocated_mb={peak_mb:.1f}")


if __name__ == "__main__":
    main()
