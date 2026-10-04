"""Wan VAE decoding with custom input, convolution, and post-processing kernels.

Reuse the Torch reference's temporal loop and unsupported layers. Custom kernels
cover latent input packing/convolution, attention post-processing, and C256
convolution fusion. C512 stages use cuDNN convolutions with custom standalone
post-processing. Opaque ops retain fullgraph compilation of surrounding Torch
work; NCTHW/NTHWC boundaries are stride views.
"""

import torch
import torch.nn.functional as F
from wan_vae_reference import (
    RawWanResidualBlock,
    TorchWanVaeDecoder,
    raw_cached_causal_conv3d,
    raw_canonicalize_cache,
    raw_conv2d,
    raw_dup_up3d,
)

from ._utils import FeatCache, update_cache
from .kernels.conv import ConvConfig, WanConv3d, prepare, prepare_next
from .kernels.memory import pack_history
from .kernels.norm import rmsnorm_silu_conv_prep


@torch.library.custom_op("wan_decoder::input_conv3d", mutates_args=())
def _input_conv3d(
    x: torch.Tensor, weight: torch.Tensor, previous: torch.Tensor | None, ctas: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack strided latent input/history and run the C48->C1024 convolution."""
    packed = pack_history(x, previous, input_ncthw=True)
    cfg = ConvConfig(*packed.padded.shape, co=weight.shape[0], ctas=ctas)
    output = prepare(packed.padded, weight, cfg)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, previous, packed.padded, packed.cache, output):
        if tensor is not None:
            tensor.record_stream(stream)
    return output, packed.cache


@_input_conv3d.register_fake
def _input_conv3d_fake(x, weight, previous, ctas):
    """Describe the raw convolution and unpadded latent-history cache."""
    n, c, t, h, w = x.shape
    history = 0 if previous is None else previous.shape[1]
    return x.new_empty((n, t, h, w, weight.shape[0])), x.new_empty(
        (n, min(2, t + history), h, w, c)
    )


@torch.library.custom_op("wan_decoder::norm_prepare", mutates_args=())
def _norm_prepare(
    x: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse optional bias/residual, RMSNorm, SiLU, padding, and cache writes."""
    result = rmsnorm_silu_conv_prep(
        x, gamma, previous=previous, input_bias=bias, residual=residual
    )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse conv2 and its residual into the following block's normalized input."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas)
    result = prepare_next(x, weight, bias, gamma, cfg, previous, residual=residual)()
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
    return result.padded, result.cache, result.residual


@_conv_residual_norm_prepare.register_fake
def _conv_residual_norm_prepare_fake(x, weight, bias, gamma, previous, residual, ctas):
    """Describe prepared next-block input and the rounded residual sum."""
    padded, cache = _conv_norm_prepare_fake(x, weight, bias, gamma, previous, ctas)
    return padded, cache, torch.empty_like(residual)


@torch.library.custom_op("wan_decoder::conv_norm_prepare", mutates_args=())
def _conv_norm_prepare(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    ctas: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run convolution/bias/norm/SiLU and produce padded input plus cache."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas)
    result = prepare_next(x, weight, bias, gamma, cfg, previous)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, bias, gamma, previous, result.padded, result.cache):
        if tensor is not None:
            tensor.record_stream(stream)
    return result.padded, result.cache


@_conv_norm_prepare.register_fake
def _conv_norm_prepare_fake(x, weight, bias, gamma, previous, ctas):
    """Describe fresh NTHWC output storage without launching CUDA."""
    n, t, h, w, _ = x.shape
    history = 0 if previous is None else previous.shape[1]
    c = weight.shape[0]
    return x.new_empty((n, t, h, w, c)), x.new_empty(
        (n, min(2, t - 2 + history), h - 2, w - 2, c)
    )


@torch.library.custom_op("wan_decoder::conv_raw", mutates_args=())
def _conv_raw(x: torch.Tensor, weight: torch.Tensor, ctas: int) -> torch.Tensor:
    """Run a raw NTHWC convolution, leaving bias to compiled Torch."""
    cfg = ConvConfig(*x.shape, co=weight.shape[0], ctas=ctas)
    result = prepare(x, weight, cfg)()
    stream = torch.cuda.current_stream(x.device)
    for tensor in (x, weight, result):
        tensor.record_stream(stream)
    return result


@_conv_raw.register_fake
def _conv_raw_fake(x, weight, ctas):
    """Describe the raw valid-convolution output."""
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 2, w - 2, weight.shape[0]))


class _FusedResidualBlock(torch.nn.Module):
    """Fuse C256 convolutions and keep C512 post-processing separate from cuDNN."""

    def __init__(self, block: RawWanResidualBlock) -> None:
        super().__init__()
        self.shortcut = block.conv_shortcut
        conv1, conv2 = block.conv1.conv, block.conv2.conv
        norm2 = block.norm2.norm
        self.fuse_conv = conv1.out_channels == 256
        if norm2.bias != 0 or block.norm1.norm.bias != 0:
            raise ValueError("Expected Wan RMSNorm without affine bias")
        weights = (conv1.weight.detach(), conv2.weight.detach())
        if self.fuse_conv:
            weights = tuple(
                weight.permute(0, 2, 3, 4, 1).contiguous() for weight in weights
            )
        for name, value in (
            ("weight1", weights[0]),
            ("weight2", weights[1]),
            ("bias1", conv1.bias.detach()),
            ("bias2", conv2.bias.detach()),
            ("gamma", norm2.gamma.detach().reshape(-1)),
            ("gamma1", block.norm1.norm.gamma.detach().reshape(-1)),
        ):
            self.register_buffer(name, value, persistent=False)
        self.ctas = (
            torch.cuda.get_device_properties(conv1.weight.device).multi_processor_count
            // 2
            * 2
        )

    def _convolve(self, x, weight):
        """Use custom C256 or cuDNN C512 convolution on already-padded input."""
        if self.fuse_conv:
            return _conv_raw(x, weight, self.ctas)
        return F.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache,
        feat_idx: int,
        prepared=None,
        next_gamma=None,
    ):
        """Preserve the reference's two causal-cache entries and BF16 additions."""
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
        if self.fuse_conv:
            padded2, cache2 = _conv_norm_prepare(
                padded, self.weight1, self.bias1, self.gamma, previous2, self.ctas
            )
        else:
            raw = self._convolve(padded, self.weight1)
            padded2, cache2, _ = _norm_prepare(raw, self.gamma, previous2, self.bias1)
        next_prepared = None
        if next_gamma is not None:
            previous = _history(feat_cache[feat_idx + 2])
            skip = residual.permute(0, 2, 3, 4, 1).contiguous()
            if self.fuse_conv:
                padded3, cache3, result = _conv_residual_norm_prepare(
                    padded2,
                    self.weight2,
                    self.bias2,
                    next_gamma,
                    previous,
                    skip,
                    self.ctas,
                )
            else:
                raw = self._convolve(padded2, self.weight2)
                padded3, cache3, result = _norm_prepare(
                    raw, next_gamma, previous, self.bias2, skip
                )
            next_prepared = (padded3, cache3)
            result = result.permute(0, 4, 1, 2, 3)
        else:
            result = self._convolve(padded2, self.weight2)
            result = (result + self.bias2).permute(0, 4, 1, 2, 3) + residual
        feat_cache = update_cache(feat_cache, feat_idx, cache1.permute(0, 4, 1, 2, 3))
        feat_cache = update_cache(
            feat_cache, feat_idx + 1, cache2.permute(0, 4, 1, 2, 3)
        )
        return result, feat_cache, feat_idx + 2, next_prepared


def _history(value):
    """View the reference's channels-last NCTHW cache as contiguous NTHWC."""
    return None if value is None else value.permute(0, 2, 3, 4, 1).contiguous()


class _FusedUpBlock(torch.nn.Module):
    """Pass fused conv2 preparation directly to the following conv1."""

    def __init__(self, stage):
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            _FusedResidualBlock(block) for block in stage.resnets
        )
        self.upsampler = stage.upsampler
        self.avg_shortcut = stage.avg_shortcut

    def forward(self, x, feat_cache, feat_idx=0, first_chunk=False):
        """Retain the reference upsampling and stage-shortcut semantics."""
        shortcut = x
        prepared = None
        for index, block in enumerate(self.resnets):
            next_gamma = (
                self.resnets[index + 1].gamma1
                if index + 1 < len(self.resnets)
                else None
            )
            x, feat_cache, feat_idx, prepared = block(
                x, feat_cache, feat_idx, prepared, next_gamma
            )
        if self.upsampler is not None:
            x, feat_cache, feat_idx = self.upsampler(x, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x = x + raw_dup_up3d(self.avg_shortcut, shortcut, first_chunk)
        return x, feat_cache, feat_idx


class _FusedMidBlock(torch.nn.Module):
    """Fold attention projection bias/residual into the following norm/preparation."""

    def __init__(self, block) -> None:
        super().__init__()
        self.resnets = block.resnets
        self.attentions = block.attentions
        for resnet in self.resnets[1:]:
            if resnet.conv_shortcut is not None or resnet.norm1.norm.bias != 0:
                raise ValueError(
                    "Attention fusion requires an identity skip and bias-free RMSNorm"
                )

    def forward(self, x, feat_cache, feat_idx=0):
        """Keep attention and C1024 convolutions in Torch; fuse their boundary ops."""
        x, feat_cache, feat_idx = self.resnets[0](x, feat_cache, feat_idx)
        for attention, block in zip(self.attentions, self.resnets[1:]):
            if attention is None:
                x, feat_cache, feat_idx = block(x, feat_cache, feat_idx)
                continue
            n, c, t, h, w = x.shape
            skip = x.permute(0, 2, 3, 4, 1).contiguous()
            normalized = attention.norm(
                x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
            )
            qkv = raw_conv2d(attention.to_qkv, normalized)
            qkv = qkv.reshape(n * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
            q, k, v = qkv.chunk(3, dim=-1)
            attended = F.scaled_dot_product_attention(q, k, v)
            attended = attended.squeeze(1).permute(0, 2, 1).reshape(n * t, c, h, w)
            projection = attention.proj
            raw = F.conv2d(
                attended,
                projection.weight,
                None,
                projection.stride,
                projection.padding,
                projection.dilation,
                projection.groups,
            )
            raw = raw.permute(0, 2, 3, 1).reshape(n, t, h, w, c).contiguous()
            padded, cache, residual = _norm_prepare(
                raw,
                block.norm1.norm.gamma.reshape(-1),
                _history(feat_cache[feat_idx]),
                projection.bias,
                skip,
            )
            conv = block.conv1.conv
            x = F.conv3d(
                padded.permute(0, 4, 1, 2, 3),
                conv.weight,
                conv.bias,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
            feat_cache = update_cache(
                feat_cache, feat_idx, cache.permute(0, 4, 1, 2, 3)
            )
            feat_idx += 1
            x = block.dropout(block.activation2(block.norm2(x)))
            x, feat_cache, feat_idx = raw_cached_causal_conv3d(
                block.conv2, x, feat_cache, feat_idx
            )
            x = x + residual.permute(0, 4, 1, 2, 3)
        return x, feat_cache, feat_idx


class _FusedDecoder(torch.nn.Module):
    """Replace latent input packing/convolution and attention post-processing."""

    def __init__(self, decoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.input_conv = WanConv3d(decoder.conv_in.conv.weight)
        self.decoder.mid_block = _FusedMidBlock(decoder.mid_block)
        self.cache_size = decoder.cache_size

    def forward(self, x, feat_cache, first_chunk=False):
        """Retain the decoder's cache ordering and unchanged upsampling/output head."""
        feat_cache = raw_canonicalize_cache(feat_cache)
        raw, cache = _input_conv3d(
            x, self.input_conv.weight, _history(feat_cache[0]), self.input_conv.ctas
        )
        x = (raw + self.decoder.conv_in.conv.bias).permute(0, 4, 1, 2, 3)
        feat_cache = update_cache(feat_cache, 0, cache.permute(0, 4, 1, 2, 3))
        x, feat_cache, feat_idx = self.decoder.mid_block(x, feat_cache, 1)
        for block in self.decoder.up_blocks:
            x, feat_cache, feat_idx = block(x, feat_cache, feat_idx, first_chunk)
        x = self.decoder.activation(self.decoder.norm_out(x))
        x, feat_cache, _ = raw_cached_causal_conv3d(
            self.decoder.conv_out, x, feat_cache, feat_idx
        )
        return x, feat_cache


class MegaWanVaeDecoder(torch.nn.Module):
    """Compile a decoder with custom input, attention-boundary, and residual blocks."""

    def __init__(self, vae) -> None:
        super().__init__()
        if vae.dtype != torch.bfloat16:
            raise ValueError("Decoder custom kernels require BF16")
        self.raw_decoder = TorchWanVaeDecoder(vae).eval()
        self.fused_blocks = 0
        for index, stage in enumerate(self.raw_decoder.decoder.up_blocks):
            if stage.resnets[0].conv1.conv.out_channels in (256, 512):
                self.raw_decoder.decoder.up_blocks[index] = _FusedUpBlock(stage)
                self.fused_blocks += len(stage.resnets)
        if not self.fused_blocks:
            raise ValueError("No supported decoder blocks found")
        self.raw_decoder.decoder = _FusedDecoder(self.raw_decoder.decoder)
        self.chunk_decoder = torch.compile(
            self.raw_decoder.decoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode with fresh history using the reference chunk loop."""
        return self.raw_decoder.decode_temporal_chunks(latent, self.chunk_decoder)
