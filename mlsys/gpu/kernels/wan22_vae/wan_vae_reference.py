"""Eager and torch.compile Wan VAEs for verification and benchmarking only.

Production code never imports this module. The reference keeps its own Torch
layers and temporal orchestration; only small host-side utilities are shared.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    unpatchify,
)
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
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResidualUpBlock as DiffusersWanResidualUpBlock,
)
from mega_wan_vae._utils import (
    FeatCache,
    validate_frame_count,
    with_channels_last_weights,
)
from mega_wan_vae._utils import (
    canonicalize_cache as raw_canonicalize_cache,
)
from mega_wan_vae._utils import (
    patchify as raw_patchify,
)
from mega_wan_vae._utils import (
    update_cache as raw_update_cache,
)

CACHE_T = 2


def raw_causal_conv3d(
    conv: torch.nn.Conv3d, x: torch.Tensor, cache_x: torch.Tensor | None = None
) -> torch.Tensor:
    padding = list(conv._padding)
    if cache_x is not None and padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return F.conv3d(
        x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
    )


def clone_channels_last_cache(tensor: torch.Tensor) -> torch.Tensor:
    """Own a dense temporal cache without changing channels-last storage."""
    return tensor.clone(memory_format=torch.channels_last_3d)


def raw_cached_causal_conv3d(
    conv: torch.nn.Module,
    x: torch.Tensor,
    feat_cache: FeatCache | None,
    feat_idx: int,
) -> tuple[torch.Tensor, FeatCache | None, int]:
    if feat_cache is None:
        return conv(x), feat_cache, feat_idx

    idx = feat_idx
    cache_x = clone_channels_last_cache(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
        prev = feat_cache[idx]
        if isinstance(prev, str):
            raise TypeError(f"Unexpected string cache entry for causal conv: {prev!r}")
        cache_x = torch.cat(
            [prev[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
        )

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
    return F.conv2d(
        x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
    )


def raw_spatial_resample(resample: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    spatial = resample.resample
    if isinstance(spatial, torch.nn.Identity):
        return x

    if resample.mode.startswith("downsample"):
        conv = spatial[1]
        return raw_conv2d(conv, F.pad(x, spatial[0].padding))

    if resample.mode.startswith("upsample"):
        conv = spatial[1]
        x = F.interpolate(
            x.float(), scale_factor=spatial[0].scale_factor, mode=spatial[0].mode
        ).type_as(x)
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
    x = x.view(
        batch_size,
        channels * avg.factor,
        frames // avg.factor_t,
        height // avg.factor_s,
        width // avg.factor_s,
    )
    x = x.view(
        batch_size,
        avg.out_channels,
        avg.group_size,
        frames // avg.factor_t,
        height // avg.factor_s,
        width // avg.factor_s,
    )
    return x.mean(dim=2)


def raw_dup_up3d(
    up: torch.nn.Module, x: torch.Tensor, first_chunk: bool = False
) -> torch.Tensor:
    """Expand the decoder shortcut directly into channels-last storage."""
    batch, _, frames, height, width = x.shape
    x = x.permute(0, 2, 3, 4, 1).repeat_interleave(up.repeats, dim=-1)
    x = x.reshape(
        batch,
        frames,
        height,
        width,
        up.out_channels,
        up.factor_t,
        up.factor_s,
        up.factor_s,
    )
    # Keep this expansion NTHWC: the NCTHW intermediate in DupUp3D can
    # miscompile with incorrect input strides when fused with the next RMSNorm.
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(
        batch,
        frames * up.factor_t,
        height * up.factor_s,
        width * up.factor_s,
        up.out_channels,
    ).permute(0, 4, 1, 2, 3)
    if first_chunk:
        x = x[:, :, up.factor_t - 1 :]
    return x


class RawWanCausalConv3d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv3d) -> None:
        super().__init__()
        self.conv = with_channels_last_weights(conv)

    def forward(
        self, x: torch.Tensor, cache_x: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        self.resample = with_channels_last_weights(resample.resample)
        self.time_conv = (
            RawWanCausalConv3d(resample.time_conv)
            if hasattr(resample, "time_conv")
            else None
        )

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
                cache_x = clone_channels_last_cache(x[:, :, -CACHE_T:, :, :])
                if cache_x.shape[2] < CACHE_T and isinstance(entry, torch.Tensor):
                    cache_x = torch.cat(
                        [
                            entry[:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if cache_x.shape[2] < CACHE_T and is_rep:
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )

                conv_cache = None if is_rep else entry
                x = self.time_conv(x, conv_cache)
                feat_cache = raw_update_cache(feat_cache, idx, cache_x)
                feat_idx += 1

                x = x.reshape(batch_size, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(batch_size, channels, frames * 2, height, width)

        frames = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames, channels, height, width
        )
        x = raw_spatial_resample(self, x)
        x = x.view(batch_size, frames, x.size(1), x.size(2), x.size(3)).permute(
            0, 2, 1, 3, 4
        )

        if self.mode == "downsample3d" and feat_cache is not None:
            if self.time_conv is None:
                raise RuntimeError("downsample3d is missing time_conv.")
            idx = feat_idx
            entry = feat_cache[idx]
            if entry is None:
                feat_cache = raw_update_cache(
                    feat_cache, idx, clone_channels_last_cache(x)
                )
                feat_idx += 1
            else:
                if not isinstance(entry, torch.Tensor):
                    raise TypeError(
                        f"Unexpected cache entry for downsample3d: {entry!r}"
                    )
                cache_x = clone_channels_last_cache(x[:, :, -1:, :, :])
                x = self.time_conv(torch.cat([entry[:, :, -1:, :, :], x], 2))
                feat_cache = raw_update_cache(feat_cache, idx, cache_x)
                feat_idx += 1

        return x, feat_cache, feat_idx


class RawWanResidualBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanResidualBlock) -> None:
        super().__init__()
        self.norm1 = RawWanRMSNorm(block.norm1)
        self.activation1 = torch.nn.SiLU()
        self.conv1 = RawWanCausalConv3d(block.conv1)
        self.norm2 = RawWanRMSNorm(block.norm2)
        self.activation2 = torch.nn.SiLU()
        self.dropout = block.dropout
        self.conv2 = RawWanCausalConv3d(block.conv2)
        self.conv_shortcut = (
            None
            if isinstance(block.conv_shortcut, torch.nn.Identity)
            else RawWanCausalConv3d(block.conv_shortcut)
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x, feat_cache, feat_idx = raw_cached_causal_conv3d(
            self.conv1, x, feat_cache, feat_idx
        )

        x = self.norm2(x)
        x = self.activation2(x)
        x = F.dropout(x, p=self.dropout.p, training=self.dropout.training)
        x, feat_cache, feat_idx = raw_cached_causal_conv3d(
            self.conv2, x, feat_cache, feat_idx
        )
        return x + residual, feat_cache, feat_idx


class RawWanAttentionBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanAttentionBlock) -> None:
        super().__init__()
        self.norm = RawWanRMSNorm(block.norm)
        self.to_qkv = with_channels_last_weights(block.to_qkv)
        self.proj = with_channels_last_weights(block.proj)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        identity = x
        batch_size, channels, frames, height, width = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames, channels, height, width
        )
        x = self.norm(x)

        qkv = raw_conv2d(self.to_qkv, x)
        qkv = qkv.reshape(batch_size * frames, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * frames, channels, height, width)
        )
        x = raw_conv2d(self.proj, x)
        x = x.view(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity, feat_cache, feat_idx


class RawWanMidBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            RawWanResidualBlock(resnet) for resnet in block.resnets
        )
        self.attentions = torch.nn.ModuleList(
            RawWanAttentionBlock(attention) for attention in block.attentions
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        x, feat_cache, feat_idx = self.resnets[0](
            x, feat_cache=feat_cache, feat_idx=feat_idx
        )
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            x, feat_cache, feat_idx = attention(
                x, feat_cache=feat_cache, feat_idx=feat_idx
            )
            x, feat_cache, feat_idx = resnet(
                x, feat_cache=feat_cache, feat_idx=feat_idx
            )
        return x, feat_cache, feat_idx


class RawWanResidualDownBlock(torch.nn.Module):
    def __init__(self, block: DiffusersWanResidualDownBlock) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            RawWanResidualBlock(resnet) for resnet in block.resnets
        )
        self.downsampler = (
            RawWanResample(block.downsampler) if block.downsampler is not None else None
        )
        self.avg_shortcut = block.avg_shortcut

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        shortcut = x.clone()
        for resnet in self.resnets:
            x, feat_cache, feat_idx = resnet(
                x, feat_cache=feat_cache, feat_idx=feat_idx
            )
        if self.downsampler is not None:
            x, feat_cache, feat_idx = self.downsampler(
                x, feat_cache=feat_cache, feat_idx=feat_idx
            )
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
        self.down_blocks = torch.nn.ModuleList(
            raw_wan_encoder_layer(layer) for layer in encoder.down_blocks
        )
        self.mid_block = RawWanMidBlock(encoder.mid_block)
        self.norm_out = RawWanRMSNorm(encoder.norm_out)
        self.conv_out = RawWanCausalConv3d(encoder.conv_out)
        self.cache_size = raw_causal_conv3d_count(encoder)

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache
    ) -> tuple[torch.Tensor, FeatCache]:
        # Keep channel-first API dimensions, but channel-adjacent storage through
        # convolutions, normalization, activation, and the temporal cache.
        x = x.contiguous(memory_format=torch.channels_last_3d)
        feat_cache = raw_canonicalize_cache(feat_cache)
        feat_idx = 0

        x, feat_cache, feat_idx = raw_cached_causal_conv3d(
            self.conv_in, x, feat_cache, feat_idx
        )
        assert feat_cache is not None
        for layer in self.down_blocks:
            x, feat_cache, feat_idx = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x, feat_cache, feat_idx = self.mid_block(
            x, feat_cache=feat_cache, feat_idx=feat_idx
        )
        assert feat_cache is not None
        x = self.norm_out(x)
        x = F.silu(x)
        x, feat_cache, _ = raw_cached_causal_conv3d(
            self.conv_out, x, feat_cache, feat_idx
        )
        assert feat_cache is not None
        return x, feat_cache


def raw_causal_conv3d_count(module: torch.nn.Module) -> int:
    return sum(
        1 for child in module.modules() if child.__class__.__name__ == "WanCausalConv3d"
    )


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

    def encode_chunk(
        self, chunk: torch.Tensor, feat_cache: FeatCache
    ) -> tuple[torch.Tensor, FeatCache]:
        return self.encoder(chunk, feat_cache)

    def encode_temporal_chunks(
        self,
        video: torch.Tensor,
        chunk_encoder: Callable[
            [torch.Tensor, FeatCache], tuple[torch.Tensor, FeatCache]
        ]
        | None = None,
    ) -> torch.Tensor:
        _, _, num_frame, height, width = video.shape
        validate_frame_count(num_frame)
        if self.patch_size is not None:
            video = raw_patchify(video, patch_size=int(self.patch_size))

        if self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            raise NotImplementedError(
                "raw PyTorch encode does not implement VAE tiling yet."
            )

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
        # Match eager/custom BF16 rounding between fused operators, rather than
        # silently evaluating those intermediates in FP32.
        self.chunk_encoder = torch.compile(
            self.raw_encoder.encoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.raw_encoder.encode_temporal_chunks(video, self.chunk_encoder)


class RawWanResidualUpBlock(torch.nn.Module):
    """Torch residual stage with spatial/temporal upsampling and its shortcut."""

    def __init__(self, block: DiffusersWanResidualUpBlock) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            RawWanResidualBlock(resnet) for resnet in block.resnets
        )
        self.upsampler = (
            RawWanResample(block.upsampler) if block.upsampler is not None else None
        )
        self.avg_shortcut = block.avg_shortcut

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache,
        feat_idx: int = 0,
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, FeatCache, int]:
        shortcut = x
        for resnet in self.resnets:
            x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
        if self.upsampler is not None:
            x, feat_cache, feat_idx = self.upsampler(x, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x = x + raw_dup_up3d(self.avg_shortcut, shortcut, first_chunk)
        return x, feat_cache, feat_idx


class RawWanDecoder3d(torch.nn.Module):
    """Decode a latent chunk using channels-last Torch layers and explicit caches."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        if not all(
            isinstance(block, DiffusersWanResidualUpBlock)
            for block in decoder.up_blocks
        ):
            raise NotImplementedError(
                "This reference targets the Wan2.2 residual decoder."
            )
        self.conv_in = RawWanCausalConv3d(decoder.conv_in)
        self.mid_block = RawWanMidBlock(decoder.mid_block)
        self.up_blocks = torch.nn.ModuleList(
            RawWanResidualUpBlock(block) for block in decoder.up_blocks
        )
        self.norm_out = RawWanRMSNorm(decoder.norm_out)
        self.activation = decoder.nonlinearity
        self.conv_out = RawWanCausalConv3d(decoder.conv_out)
        self.cache_size = raw_causal_conv3d_count(decoder)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache,
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, FeatCache]:
        x = x.contiguous(memory_format=torch.channels_last_3d)
        feat_cache = raw_canonicalize_cache(feat_cache)
        x, feat_cache, feat_idx = raw_cached_causal_conv3d(
            self.conv_in, x, feat_cache, 0
        )
        x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)
        for block in self.up_blocks:
            x, feat_cache, feat_idx = block(x, feat_cache, feat_idx, first_chunk)
        x = self.activation(self.norm_out(x))
        x, feat_cache, _ = raw_cached_causal_conv3d(
            self.conv_out, x, feat_cache, feat_idx
        )
        return x, feat_cache


class TorchWanVaeDecoder(torch.nn.Module):
    """Decode unnormalized latents into clamped NCTHW video, matching Diffusers."""

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.post_quant_conv = RawWanCausalConv3d(vae.post_quant_conv)
        self.decoder = RawWanDecoder3d(vae.decoder)
        self.patch_size = getattr(vae.config, "patch_size", None)
        self.use_tiling = vae.use_tiling
        self.tile_latent_min_width = (
            vae.tile_sample_min_width // vae.spatial_compression_ratio
        )
        self.tile_latent_min_height = (
            vae.tile_sample_min_height // vae.spatial_compression_ratio
        )

    def new_cache(self) -> FeatCache:
        """Create empty history for an independent latent sequence."""
        return (None,) * self.decoder.cache_size

    def decode_chunk(
        self,
        chunk: torch.Tensor,
        feat_cache: FeatCache,
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, FeatCache]:
        """Decode a post-quant-convolution chunk into still-patchified pixels."""
        return self.decoder(chunk, feat_cache, first_chunk)

    def decode_temporal_chunks(
        self,
        latent: torch.Tensor,
        chunk_decoder: Callable[
            [torch.Tensor, FeatCache, bool], tuple[torch.Tensor, FeatCache]
        ]
        | None = None,
    ) -> torch.Tensor:
        """Decode one latent frame per call, then unpatchify and clamp to [-1, 1]."""
        _, _, frames, height, width = latent.shape
        if frames < 1:
            raise ValueError("Decoding requires at least one latent frame.")
        if self.use_tiling and (
            width > self.tile_latent_min_width or height > self.tile_latent_min_height
        ):
            raise NotImplementedError(
                "Raw PyTorch decode does not implement VAE tiling."
            )
        if chunk_decoder is None:
            chunk_decoder = self.decode_chunk
        x = self.post_quant_conv(latent)
        feat_cache = self.new_cache()
        outputs = []
        for i in range(frames):
            out, feat_cache = chunk_decoder(x[:, :, i : i + 1], feat_cache, i == 0)
            outputs.append(out)
        out = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        if self.patch_size is not None:
            out = unpatchify(out, patch_size=self.patch_size)
        return out.clamp(-1.0, 1.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode with fresh history; latent mean/std scaling belongs to the caller."""
        return self.decode_temporal_chunks(latent)


class TorchCompileChunkWanVaeDecoder(torch.nn.Module):
    """Compile decoder chunks while preserving the eager reference's BF16 casts."""

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.raw_decoder = TorchWanVaeDecoder(vae).eval()
        self.chunk_decoder = torch.compile(
            self.raw_decoder.decoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the same temporal loop with compiled chunk execution."""
        return self.raw_decoder.decode_temporal_chunks(latent, self.chunk_decoder)
