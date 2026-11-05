"""Shared Wan residual and attention adapters with opaque custom operations.

Wan 2.1 fuses C96/C192 convolution epilogues; Wan 2.2 uses the C256 path here.
Shape-specific cuDNN fallbacks use the same preparation operations.
Opaque ops retain fullgraph compilation of surrounding Torch work;
NCTHW/NTHWC boundaries are stride views.
"""

import torch
import torch.nn.functional as F

from .._utils import FeatCache, update_cache
from ..kernels._precision import resolve_precision
from ..kernels.conv import ConvConfig, WanConv3d, prepare, prepare_next, prepare_spatial_residual
from ..kernels.memory import pack_history
from ..kernels.norm import rmsnorm, rmsnorm_silu_conv_prep
from .torch import (
    RawWanResidualBlock,
    RawWanRMSNorm,
    raw_cached_causal_conv3d,
    raw_conv2d,
    raw_dup_up3d,
    spatial_convolution,
    use_first_frame_shortcut,
)


@torch.library.custom_op("wan_vae::rmsnorm", mutates_args=())
def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Keep the eager channel reduction tree across torch.compile boundaries."""
    is_image = x.ndim == 4
    value = x.unsqueeze(2) if is_image else x
    value = value.permute(0, 2, 3, 4, 1).contiguous()
    out = rmsnorm(value, gamma, bias).permute(0, 4, 1, 2, 3)
    return out.squeeze(2) if is_image else out


@_rmsnorm.register_fake
def _rmsnorm_fake(x, gamma, bias):
    return torch.empty_like(x, memory_format=torch.channels_last if x.ndim == 4 else torch.channels_last_3d)


class _FusedRMSNorm(torch.nn.Module):
    """Preserve norm weights and channel-first API with a fixed reduction tree."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        bias = self.norm.bias
        return _rmsnorm(x, self.norm.gamma.reshape(-1), bias.reshape(-1) if isinstance(bias, torch.Tensor) else None)


def compile_preserving_norms(module):
    """Compile chunks with stable normalization reductions and BF16 rounding."""

    # Inductor's reduction tree and approximate division can cross BF16 ties.
    # Keep the remaining attention/head norms opaque, like residual preparation.
    def replace_norms(parent):
        for name, child in parent.named_children():
            if isinstance(child, RawWanRMSNorm):
                setattr(parent, name, _FusedRMSNorm(child.norm))
            else:
                replace_norms(child)

    replace_norms(module)
    return torch.compile(
        module,
        fullgraph=True,
        dynamic=False,
        options={"emulate_precision_casts": True, "eager_numerics.division_rounding": True},
    )


@torch.library.custom_op("wan_decoder::input_conv3d", mutates_args=())
def _input_conv3d(
    x: torch.Tensor, weight: torch.Tensor, previous: torch.Tensor | None, ctas: int, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack strided latent input/history and run the C48->C1024 convolution."""
    packed = pack_history(x, previous, input_ncthw=True)
    cfg = ConvConfig(*packed.padded.shape, co=weight.shape[0], ctas=ctas, precision=precision)
    output = prepare(packed.padded, weight, cfg)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, previous, packed.padded, packed.cache, output):
        if tensor is not None:
            tensor.record_stream(stream)
    return output, packed.cache


@_input_conv3d.register_fake
def _input_conv3d_fake(x, weight, previous, ctas, precision):
    """Describe the raw convolution and unpadded latent-history cache."""
    n, c, t, h, w = x.shape
    history = 0 if previous is None else previous.shape[1]
    return x.new_empty((n, t, h, w, weight.shape[0])), x.new_empty((n, min(2, t + history), h, w, c))


@torch.library.custom_op("wan_decoder::norm_prepare", mutates_args=())
def _norm_prepare(
    x: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse optional bias/residual, RMSNorm, SiLU, padding, and cache writes."""
    result = rmsnorm_silu_conv_prep(x, gamma, previous=previous, input_bias=bias, residual=residual)
    stream = torch.cuda.current_stream(x.device)
    for tensor in (
        x,
        gamma,
        previous,
        bias,
        residual,
        result.padded,
        result.cache,
        result.residual,
    ):
        if tensor is not None:
            tensor.record_stream(stream)
    return (
        result.padded,
        result.cache,
        x.new_empty((0,)) if result.residual is None else result.residual,
    )


@_norm_prepare.register_fake
def _norm_prepare_fake(x, gamma, previous, bias=None, residual=None):
    """Describe normalized preparation and its pre-normalization residual."""
    n, t, h, w, c = x.shape
    history = 0 if previous is None else previous.shape[1]
    return (
        x.new_empty((n, t + 2, h + 2, w + 2, c)),
        x.new_empty((n, min(2, t + history), h, w, c)),
        x.new_empty((0,)) if residual is None else torch.empty_like(x),
    )


@torch.library.custom_op("wan_decoder::conv_residual_norm_prepare", mutates_args=())
def _conv_residual_norm_prepare(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    residual: torch.Tensor,
    ctas: int,
    precision: str,
    save_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse conv2 into normalized consumer input, optionally retaining its skip."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas, precision=precision)
    result = prepare_next(x, weight, bias, gamma, cfg, previous, residual=residual, save_residual=save_residual)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (
        x,
        weight,
        bias,
        gamma,
        previous,
        residual,
        result.padded,
        result.cache,
        result.residual,
    ):
        if tensor is not None:
            tensor.record_stream(stream)
    return result.padded, result.cache, result.residual if result.residual is not None else x.new_empty((0,))


@_conv_residual_norm_prepare.register_fake
def _conv_residual_norm_prepare_fake(x, weight, bias, gamma, previous, residual, ctas, precision, save_residual=True):
    """Describe prepared input and an optional rounded residual sum."""
    padded, cache = _conv_norm_prepare_fake(x, weight, bias, gamma, previous, ctas, precision)
    return padded, cache, torch.empty_like(residual) if save_residual else x.new_empty((0,))


@torch.library.custom_op("wan_decoder::conv_norm_prepare", mutates_args=())
def _conv_norm_prepare(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    ctas: int,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run convolution/bias/norm/SiLU and produce padded input plus cache."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas, precision=precision)
    result = prepare_next(x, weight, bias, gamma, cfg, previous)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, bias, gamma, previous, result.padded, result.cache):
        if tensor is not None:
            tensor.record_stream(stream)
    return result.padded, result.cache


@_conv_norm_prepare.register_fake
def _conv_norm_prepare_fake(x, weight, bias, gamma, previous, ctas, precision):
    """Describe fresh NTHWC output storage without launching CUDA."""
    n, t, h, w, _ = x.shape
    history = 0 if previous is None else previous.shape[1]
    c = weight.shape[0]
    return x.new_empty((n, t, h, w, c)), x.new_empty((n, min(2, t - 2 + history), h - 2, w - 2, c))


@torch.library.custom_op("wan_decoder::conv_raw", mutates_args=())
def _conv_raw(x: torch.Tensor, weight: torch.Tensor, ctas: int, precision: str) -> torch.Tensor:
    """Run a raw NTHWC convolution, leaving bias to compiled Torch."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas, precision=precision)
    result = prepare(x, weight, cfg)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, result):
        tensor.record_stream(stream)
    return result


@_conv_raw.register_fake
def _conv_raw_fake(x, weight, ctas, precision):
    """Describe the raw valid-convolution output."""
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 2, w - 2, weight.shape[0]))


@torch.library.custom_op("wan_vae::conv_spatial_residual", mutates_args=())
def _conv_spatial_residual(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, ctas: int, precision: str
) -> torch.Tensor:
    """Produce bias/residual-summed input with a zero bottom/right border."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas, precision=precision)
    result = prepare_spatial_residual(x, weight, bias, residual, cfg)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, bias, residual, result):
        tensor.record_stream(stream)
    return result


@_conv_spatial_residual.register_fake
def _conv_spatial_residual_fake(x, weight, bias, residual, ctas, precision):
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 1, w - 1, weight.shape[0]))


class _FusedResidualBlock(torch.nn.Module):
    """Fuse normalization/cache preparation around custom or cuDNN convolutions."""

    def __init__(self, block: RawWanResidualBlock, *, custom_conv: bool = False) -> None:
        super().__init__()
        self.shortcut = block.conv_shortcut
        conv1, conv2 = block.conv1.conv, block.conv2.conv
        self.precision = resolve_precision(conv1.weight.dtype)
        norm2 = block.norm2.norm
        # C256 TF32 favors cuDNN plus preparation at the measured decoder shapes.
        # Keep integrated BF16 and Wan 2.1 fusion; see BENCHMARKS.md.
        self.fuse_conv = (conv1.out_channels == 256 and self.precision != "tf32") or (
            custom_conv and conv1.out_channels in (96, 192)
        )
        self.custom_conv = custom_conv or self.fuse_conv
        if any(isinstance(bias, torch.Tensor) or bias != 0 for bias in (norm2.bias, block.norm1.norm.bias)):
            raise ValueError("Expected Wan RMSNorm without affine bias")
        weights = (conv1.weight.detach(), conv2.weight.detach())
        if self.custom_conv:
            weights = tuple(WanConv3d(weight, precision=self.precision).weight for weight in weights)
        for name, value in (
            ("weight1", weights[0]),
            ("weight2", weights[1]),
            ("bias1", conv1.bias.detach()),
            ("bias2", conv2.bias.detach()),
            ("gamma", norm2.gamma.detach().reshape(-1)),
            ("gamma1", block.norm1.norm.gamma.detach().reshape(-1)),
        ):
            self.register_buffer(name, value, persistent=False)
        self.first_frame_shortcut = use_first_frame_shortcut(conv1) and (
            conv1.out_channels in (96, 192, 256) or not self.custom_conv or self.precision != "bf16"
        )
        for name, conv in (("first_weight1", conv1), ("first_weight2", conv2)):
            self.register_buffer(
                name,
                conv.weight[:, :, 2].detach().contiguous(memory_format=torch.channels_last)
                if self.first_frame_shortcut
                else None,
                persistent=False,
            )
        self.ctas = torch.cuda.get_device_properties(conv1.weight.device).multi_processor_count // 2 * 2

    def _convolve(self, x, weight, first_weight=None):
        """Convolve already-padded input using the adapter's selected backend."""
        if first_weight is not None:
            return spatial_convolution(x[:, 2].permute(0, 3, 1, 2), first_weight).permute(0, 2, 3, 1).unsqueeze(1)
        if self.custom_conv:
            return _conv_raw(x, weight, self.ctas, self.precision)
        return F.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache,
        feat_idx: int,
        prepared=None,
        next_gamma=None,
        spatial_output=False,
    ):
        """Preserve the reference's two causal-cache entries and BF16 additions."""
        first1 = self.first_frame_shortcut and x.shape[2] == 1 and feat_cache[feat_idx] is None
        first2 = self.first_frame_shortcut and x.shape[2] == 1 and feat_cache[feat_idx + 1] is None
        residual = x if self.shortcut is None else self.shortcut(x)
        if prepared is None:
            padded, cache1, _ = _norm_prepare(
                x.permute(0, 2, 3, 4, 1).contiguous(),
                self.gamma1,
                _history(feat_cache[feat_idx]),
            )
        else:
            padded, cache1 = prepared
        previous2 = _history(feat_cache[feat_idx + 1])
        if self.fuse_conv and not first1:
            padded2, cache2 = _conv_norm_prepare(
                padded, self.weight1, self.bias1, self.gamma, previous2, self.ctas, self.precision
            )
        else:
            raw = self._convolve(padded, self.weight1, self.first_weight1 if first1 else None)
            padded2, cache2, _ = _norm_prepare(raw, self.gamma, previous2, self.bias1)
        next_prepared = None
        if next_gamma is not None:
            previous = _history(feat_cache[feat_idx + 2])
            skip = residual.permute(0, 2, 3, 4, 1).contiguous()
            if self.fuse_conv and not first2:
                padded3, cache3, result = _conv_residual_norm_prepare(
                    padded2,
                    self.weight2,
                    self.bias2,
                    next_gamma,
                    previous,
                    skip,
                    self.ctas,
                    self.precision,
                )
            else:
                raw = self._convolve(padded2, self.weight2, self.first_weight2 if first2 else None)
                padded3, cache3, result = _norm_prepare(raw, next_gamma, previous, self.bias2, skip)
            next_prepared = (padded3, cache3)
            result = result.permute(0, 4, 1, 2, 3)
        elif spatial_output and first2:
            raw = self._convolve(padded2, self.weight2, self.first_weight2)
            result = F.pad((raw + self.bias2).permute(0, 4, 1, 2, 3) + residual, (0, 1, 0, 1))
        elif spatial_output:
            result = _conv_spatial_residual(
                padded2,
                self.weight2,
                self.bias2,
                residual.permute(0, 2, 3, 4, 1).contiguous(),
                self.ctas,
                self.precision,
            ).permute(0, 4, 1, 2, 3)
        else:
            result = self._convolve(padded2, self.weight2, self.first_weight2 if first2 else None)
            result = (result + self.bias2).permute(0, 4, 1, 2, 3) + residual
        feat_cache = update_cache(feat_cache, feat_idx, cache1.permute(0, 4, 1, 2, 3))
        feat_cache = update_cache(feat_cache, feat_idx + 1, cache2.permute(0, 4, 1, 2, 3))
        return result, feat_cache, feat_idx + 2, next_prepared


def _history(value):
    """View the reference's channels-last NCTHW cache as contiguous NTHWC."""
    return None if value is None else value.permute(0, 2, 3, 4, 1).contiguous()


class _FusedUpBlock(torch.nn.Module):
    """Pass fused conv2 preparation directly to the following conv1."""

    def __init__(self, stage, *, custom_conv: bool = False):
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            _FusedResidualBlock(block, custom_conv=custom_conv) for block in stage.resnets
        )
        self.upsampler = stage.upsampler
        self.avg_shortcut = stage.avg_shortcut

    def forward(self, x, feat_cache, feat_idx=0, first_chunk=False):
        """Retain the reference upsampling and stage-shortcut semantics."""
        shortcut = x
        prepared = None
        for index, block in enumerate(self.resnets):
            next_gamma = self.resnets[index + 1].gamma1 if index + 1 < len(self.resnets) else None
            x, feat_cache, feat_idx, prepared = block(x, feat_cache, feat_idx, prepared, next_gamma)
        if self.upsampler is not None:
            x, feat_cache, feat_idx = self.upsampler(x, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x = x + raw_dup_up3d(self.avg_shortcut, shortcut, first_chunk)
        return x, feat_cache, feat_idx


class _FusedMidBlock(torch.nn.Module):
    """Fold attention projection bias/residual into the following norm/preparation."""

    def __init__(self, block, *, custom_conv=False) -> None:
        super().__init__()
        self.resnets = block.resnets
        self.attentions = block.attentions
        self.custom_conv = custom_conv
        if custom_conv:
            self.first = _FusedResidualBlock(block.resnets[0], custom_conv=True)
            self.following = torch.nn.ModuleList(
                _FusedResidualBlock(resnet, custom_conv=True) for resnet in block.resnets[1:]
            )
        for resnet in self.resnets[1:]:
            if resnet.conv_shortcut is not None or resnet.norm1.norm.bias != 0:
                raise ValueError("Attention fusion requires an identity skip and bias-free RMSNorm")

    def forward(self, x, feat_cache, feat_idx=0):
        """Keep attention and C1024 convolutions in Torch; fuse their boundary ops."""
        if self.custom_conv:
            x, feat_cache, feat_idx, _ = self.first(x, feat_cache, feat_idx)
        else:
            x, feat_cache, feat_idx = self.resnets[0](x, feat_cache, feat_idx)
        for index, (attention, block) in enumerate(zip(self.attentions, self.resnets[1:])):
            if attention is None:
                x, feat_cache, feat_idx = block(x, feat_cache, feat_idx)
                continue
            n, c, t, h, w = x.shape
            skip = x.permute(0, 2, 3, 4, 1).contiguous()
            normalized = attention.norm(x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w))
            qkv = raw_conv2d(attention.to_qkv, normalized)
            qkv = qkv.reshape(n * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
            q, k, v = qkv.chunk(3, dim=-1)
            attended = F.scaled_dot_product_attention(q, k, v)
            attended = attended.squeeze(1).permute(0, 2, 1).reshape(n * t, c, h, w)
            projection = attention.proj
            raw = spatial_convolution(
                attended,
                projection.weight,
                None,
                projection.stride,
                projection.padding,
                projection.dilation,
                projection.groups,
            )
            raw = raw.permute(0, 2, 3, 1).reshape(n, t, h, w, c).contiguous()
            if self.custom_conv:
                x = ((raw + projection.bias) + skip).permute(0, 4, 1, 2, 3)
                x, feat_cache, feat_idx, _ = self.following[index](x, feat_cache, feat_idx)
                continue
            padded, cache, residual = _norm_prepare(
                raw,
                block.norm1.norm.gamma.reshape(-1),
                _history(feat_cache[feat_idx]),
                projection.bias,
                skip,
            )
            conv = block.conv1.conv
            if t == 1 and feat_cache[feat_idx] is None and use_first_frame_shortcut(conv):
                x = spatial_convolution(
                    padded[:, 2].permute(0, 3, 1, 2),
                    conv.weight[:, :, 2].contiguous(memory_format=torch.channels_last),
                    conv.bias,
                ).unsqueeze(2)
            else:
                x = F.conv3d(
                    padded.permute(0, 4, 1, 2, 3),
                    conv.weight,
                    conv.bias,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    conv.groups,
                )
            feat_cache = update_cache(feat_cache, feat_idx, cache.permute(0, 4, 1, 2, 3))
            feat_idx += 1
            x = block.dropout(block.activation2(block.norm2(x)))
            x, feat_cache, feat_idx = raw_cached_causal_conv3d(block.conv2, x, feat_cache, feat_idx)
            x = x + residual.permute(0, 4, 1, 2, 3)
        return x, feat_cache, feat_idx
