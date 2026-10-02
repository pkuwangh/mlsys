"""MegaWanVaeEncoder and its private NTHWC layer adapters.

The public model owns the full temporal encode path. Kernel implementations
are private to kernels/; nothing here imports the benchmark runner.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    WanAttentionBlock,
    WanCausalConv3d,
    WanResample,
    WanResidualBlock,
    WanResidualDownBlock,
)

from ._utils import (
    FeatCache,
    canonicalize_cache,
    patchify,
    update_cache,
    validate_frame_count,
    with_channels_last_weights,
)
from .kernels._utils import PreparedConvInput
from .kernels.conv import WanConv3d, WanInputConv3d
from .kernels.memory import pack_history
from .kernels.norm import rmsnorm as primitive_norm
from .kernels.norm import rmsnorm_silu_conv_prep as primitive_conv_prep
from .kernels.residual import bias_residual as primitive_bias_residual
from .kernels.residual import stage_add as primitive_stage_add


class _ConvWeights(torch.nn.Module):
    """Own channels-last convolution weights without an eager forward path.

    Keep the conv submodule name stable for existing encoder state dictionaries.
    Custom consumers choose their kernel and handle padding, bias, and history.
    """

    def __init__(self, conv: torch.nn.Conv3d) -> None:
        super().__init__()
        self.conv = with_channels_last_weights(conv)


class _ConvPrep(torch.nn.Module):
    """Wan weights for fused NTHWC bias/norm/SiLU/padding/cache preparation."""

    def __init__(self, norm: torch.nn.Module) -> None:
        super().__init__()
        if not norm.channel_first or norm.scale != math.sqrt(norm.gamma.numel()):
            raise ValueError("Expected Wan channel-first RMSNorm with sqrt(C) scale")
        self.norm = norm

    def affine_bias(self) -> torch.Tensor | None:
        """Expose Wan's optional channel bias as a contiguous vector."""
        bias = self.norm.bias
        if isinstance(bias, torch.Tensor):
            bias = bias.reshape(-1)
        elif bias == 0:
            bias = None
        else:
            raise ValueError("RMSNorm bias must be a tensor or zero")
        return bias

    def forward(
        self,
        x: torch.Tensor,
        previous: torch.Tensor | None,
        contiguous_reduction: bool,
        input_bias: torch.Tensor | None,
        residual: torch.Tensor | None = None,
        *,
        residual_bias: torch.Tensor | None = None,
        save_input: bool = False,
    ) -> PreparedConvInput:
        """Consume NTHWC activations/history and an optional preceding-conv bias."""
        return primitive_conv_prep(
            x,
            self.norm.gamma.reshape(-1),
            self.affine_bias(),
            previous,
            input_bias=input_bias,
            residual=residual,
            residual_bias=residual_bias,
            save_input=save_input,
            contiguous_reduction=contiguous_reduction,
        )


@dataclass(frozen=True)
class ConvOutput:
    """NTHWC convolution result with bias deferred to its consumer."""

    value: torch.Tensor
    bias: torch.Tensor | None


@dataclass(frozen=True)
class ResidualConvOutput:
    """Pending BF16 conv2 bias/add, with all activation tensors in NTHWC.

    With B denoting BF16 rounding, the next preparation consumes
    B(B(value + bias) + B(residual + residual_bias)); absent biases are skipped.
    At a Torch boundary, materialize the same expression in one kernel.
    """

    value: torch.Tensor
    bias: torch.Tensor | None
    residual: torch.Tensor
    residual_bias: torch.Tensor | None = None

    def materialize(self) -> torch.Tensor:
        return primitive_bias_residual(
            self.value, self.residual, self.bias, residual_bias=self.residual_bias
        )


@dataclass(frozen=True)
class PreparedResidualOutput:
    """Next block's prepared input, plus the producer skip for stage ownership.

    prepared.residual is the producer's fully summed output and becomes the
    next block's skip. residual is the producer's input skip, retained separately
    because the outer down-stage also needs its original shortcut input.
    """

    prepared: PreparedConvInput
    residual: torch.Tensor


@dataclass(frozen=True)
class SpatialResidualOutput:
    """Summed and bottom/right-padded input for a spatial downsample."""

    padded: torch.Tensor
    residual: torch.Tensor


class _PreparedResidualBlock(torch.nn.Module):
    """Fused preparation with primitive residual convolutions.

    Input, output operands, and both cache slots are explicitly NTHWC.
    Adjacent C160/C320 blocks pass prepared activations, history and the summed skip
    directly from conv2. C640 blocks defer conv2 bias/residual to the next norm1
    preparation, which also writes the summed input for its skip connection.
    Other cache slots stay in Torch.
    The primitive convolution consumes NTHWC directly. Shortcut convolutions use
    an NCTHW view for cuDNN without a data copy.
    Normalization follows the channels-last Torch reduction order, including
    fused epilogues. C160/C320 conv1 emits norm2/SiLU/padding/history directly;
    C640 conv1 sites defer bias to norm2 preparation. Before a spatial
    downsample, conv2 fuses bias/residual/bottom-right padding at every width.
    Other conv2 sites defer bias to the following residual epilogue.
    Weight layout conversion happens once at construction, not per convolution.
    Boundary adapters live in the down-stage and middle-block wrappers.
    The input-convolution bias is consumed by the first preparation, which
    also writes the biased input for both skip branches. Projected shortcuts
    defer their bias to the next residual preparation without reassociation.
    """

    def __init__(self, block: WanResidualBlock, contiguous_norm1: bool) -> None:
        super().__init__()
        self.norm1 = _ConvPrep(block.norm1)
        self.conv1 = _ConvWeights(block.conv1)
        self.norm2 = _ConvPrep(block.norm2)
        self.conv2 = _ConvWeights(block.conv2)
        self.conv_shortcut = (
            None
            if isinstance(block.conv_shortcut, torch.nn.Identity)
            else _ConvWeights(block.conv_shortcut)
        )
        self.contiguous_norm1 = contiguous_norm1
        for conv in (self.conv1.conv, self.conv2.conv):
            if tuple(conv._padding) != (1, 1, 1, 1, 2, 0) or tuple(conv.padding) != (
                0,
                0,
                0,
            ):
                raise ValueError(
                    "Prepared residual convolution requires causal 3x3x3 padding"
                )
            if (
                tuple(conv.stride) != (1, 1, 1)
                or tuple(conv.dilation) != (1, 1, 1)
                or conv.groups != 1
            ):
                raise ValueError(
                    "Primitive residual convolution requires unit stride/dilation "
                    "and groups=1"
                )
        self.primitive_convs = torch.nn.ModuleDict(
            {
                name: WanConv3d(getattr(self, name).conv.weight)
                for name in ("conv1", "conv2")
            }
        )
        if (
            self.primitive_convs["conv1"].out_channels in (160, 320)
            and self.norm2.affine_bias() is not None
        ):
            raise ValueError("Fused convolution requires bias-free RMSNorm")
        if self.conv_shortcut is not None:
            self.register_buffer(
                "shortcut_weight",
                self.conv_shortcut.conv.weight.detach().contiguous(
                    memory_format=torch.channels_last_3d
                ),
                persistent=False,
            )

    def prepare(
        self,
        x: torch.Tensor,
        norm: _ConvPrep,
        feat_cache: FeatCache | None,
        feat_idx: int,
        contiguous_reduction: bool,
        input_bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        *,
        residual_bias: torch.Tensor | None = None,
        save_input: bool = False,
    ) -> PreparedConvInput:
        """Prepare one convolution without materializing pending bias/residual."""
        previous = None if feat_cache is None else feat_cache[feat_idx]
        if isinstance(previous, str):
            raise TypeError("Residual convolution cache must be a tensor or None")
        return norm(
            x,
            previous,
            contiguous_reduction,
            input_bias,
            residual,
            residual_bias=residual_bias,
            save_input=save_input,
        )

    def convolve(
        self,
        prepared: PreparedConvInput,
        wrapper: _ConvWeights,
        feat_cache: FeatCache | None,
        feat_idx: int,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        """Consume NTHWC preparation and retain its independently owned cache."""
        conv = wrapper.conv
        name = "conv1" if wrapper is self.conv1 else "conv2"
        if (
            tuple(conv.stride) != (1, 1, 1)
            or tuple(conv.dilation) != (1, 1, 1)
            or conv.groups != 1
        ):
            raise ValueError(
                "Primitive residual convolution requires unit stride/dilation "
                "and groups=1"
            )
        out = self.primitive_convs[name](prepared.padded)
        if feat_cache is not None:
            feat_cache = update_cache(feat_cache, feat_idx, prepared.cache)
            feat_idx += 1
        return out, feat_cache, feat_idx

    def forward(
        self,
        x: torch.Tensor | ConvOutput | ResidualConvOutput | PreparedResidualOutput,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
        *,
        next_norm: _ConvPrep | None = None,
        pad_output: bool = False,
    ) -> tuple[
        ResidualConvOutput | PreparedResidualOutput | SpatialResidualOutput,
        FeatCache | None,
        int,
    ]:
        if self.training:
            raise ValueError(
                "Prepared residual convolution is inference-only; call eval()"
            )
        residual_bias = None
        if isinstance(x, PreparedResidualOutput):
            if self.conv_shortcut is not None:
                raise ValueError("Fused prepared input requires an identity skip")
            prepared = x.prepared
            residual = prepared.residual
            if residual is None:
                raise ValueError(
                    "Fused residual preparation must preserve the skip input"
                )
        elif isinstance(x, ResidualConvOutput):
            if self.conv_shortcut is not None:
                raise ValueError("Deferred residual input requires an identity skip")
            prepared = self.prepare(
                x.value,
                self.norm1,
                feat_cache,
                feat_idx,
                self.contiguous_norm1,
                x.bias,
                x.residual,
                residual_bias=x.residual_bias,
            )
            residual = prepared.residual
            assert residual is not None
        elif isinstance(x, ConvOutput):
            if self.conv_shortcut is not None:
                raise ValueError("Deferred encoder input requires an identity skip")
            prepared = self.prepare(
                x.value,
                self.norm1,
                feat_cache,
                feat_idx,
                self.contiguous_norm1,
                x.bias,
                save_input=True,
            )
            residual = prepared.residual
            assert residual is not None
        else:
            prepared = self.prepare(
                x, self.norm1, feat_cache, feat_idx, self.contiguous_norm1
            )
            residual = x
            if self.conv_shortcut is not None:
                conv = self.conv_shortcut.conv
                residual = (
                    F.conv3d(
                        x.permute(0, 4, 1, 2, 3),
                        self.shortcut_weight,
                        None,
                        conv.stride,
                        conv.padding,
                        conv.dilation,
                        conv.groups,
                    )
                    .permute(0, 2, 3, 4, 1)
                    .contiguous()
                )
                residual_bias = conv.bias
        conv1 = self.primitive_convs["conv1"]
        if conv1.out_channels in (160, 320):
            previous = None if feat_cache is None else feat_cache[feat_idx + 1]
            if isinstance(previous, str):
                raise TypeError("Residual convolution cache must be a tensor or None")
            next_prepared = conv1.forward_prepared(
                prepared.padded,
                self.conv1.conv.bias,
                self.norm2.norm.gamma.reshape(-1),
                previous,
            )
            if feat_cache is not None:
                feat_cache = update_cache(feat_cache, feat_idx, prepared.cache)
                feat_idx += 1
            prepared = next_prepared
        else:
            x, feat_cache, feat_idx = self.convolve(
                prepared, self.conv1, feat_cache, feat_idx
            )
            prepared = self.prepare(
                x,
                self.norm2,
                feat_cache,
                feat_idx,
                True,
                self.conv1.conv.bias,
            )
        if next_norm is not None:
            if pad_output:
                raise ValueError(
                    "A residual block cannot prepare two distinct consumers"
                )
            if next_norm.affine_bias() is not None:
                raise ValueError(
                    "Fused inter-block preparation requires bias-free RMSNorm"
                )
            previous = None if feat_cache is None else feat_cache[feat_idx + 1]
            if isinstance(previous, str):
                raise TypeError("Residual convolution cache must be a tensor or None")
            next_prepared = self.primitive_convs["conv2"].forward_prepared(
                prepared.padded,
                self.conv2.conv.bias,
                next_norm.norm.gamma.reshape(-1),
                previous,
                residual=residual,
                residual_bias=residual_bias,
            )
            if feat_cache is not None:
                feat_cache = update_cache(feat_cache, feat_idx, prepared.cache)
                feat_idx += 1
            return PreparedResidualOutput(next_prepared, residual), feat_cache, feat_idx
        if pad_output:
            padded = self.primitive_convs["conv2"].forward_spatial_residual(
                prepared.padded, self.conv2.conv.bias, residual, residual_bias
            )
            if feat_cache is not None:
                feat_cache = update_cache(feat_cache, feat_idx, prepared.cache)
                feat_idx += 1
            return SpatialResidualOutput(padded, residual), feat_cache, feat_idx
        x, feat_cache, feat_idx = self.convolve(
            prepared, self.conv2, feat_cache, feat_idx
        )
        return (
            ResidualConvOutput(x, self.conv2.conv.bias, residual, residual_bias),
            feat_cache,
            feat_idx,
        )


class _Downsample(torch.nn.Module):
    """Fuse bias/residual/pad before cuDNN spatial convolution in NTHWC.

    Temporal history and bias are assembled directly in NTHWC. The final
    convolution bias is consumed by the stage's grouped shortcut/add kernel.
    """

    def __init__(self, resample: WanResample) -> None:
        super().__init__()
        if resample.mode not in ("downsample2d", "downsample3d"):
            raise ValueError("Custom spatial preparation requires downsampling")
        if tuple(resample.resample[0].padding) != (0, 1, 0, 1):
            raise ValueError("Expected one-sided bottom/right spatial padding")
        self.mode = resample.mode
        self.conv = with_channels_last_weights(resample.resample[1])
        self.time_conv = (
            _ConvWeights(resample.time_conv) if hasattr(resample, "time_conv") else None
        )
        self.register_buffer(
            "weight",
            self.conv.weight.detach().contiguous(memory_format=torch.channels_last),
            persistent=False,
        )
        if self.time_conv is not None:
            self.register_buffer(
                "time_weight",
                self.time_conv.conv.weight.detach().contiguous(
                    memory_format=torch.channels_last_3d
                ),
                persistent=False,
            )

    def forward(
        self,
        pending: ResidualConvOutput | SpatialResidualOutput,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[ConvOutput, FeatCache | None, int]:
        if isinstance(pending, SpatialResidualOutput):
            padded = pending.padded
        else:
            padded = primitive_bias_residual(
                pending.value,
                pending.residual,
                pending.bias,
                pad_spatial=True,
                residual_bias=pending.residual_bias,
            )
        n, frames, height, width, channels = padded.shape
        conv = self.conv
        x = F.conv2d(
            padded.view(n * frames, height, width, channels).permute(0, 3, 1, 2),
            self.weight,
            None,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        x = x.permute(0, 2, 3, 1).reshape(n, frames, x.shape[2], x.shape[3], x.shape[1])
        output_bias = conv.bias
        if self.mode == "downsample3d" and feat_cache is not None:
            entry = feat_cache[feat_idx]
            if entry is not None and not isinstance(entry, torch.Tensor):
                raise TypeError("Temporal downsample history must be a tensor")
            packed = pack_history(
                x,
                entry,
                conv.bias,
                padding=(0 if entry is None else 1, 0, 0),
                cache_frames=1,
            )
            x = packed.padded
            output_bias = None
            if entry is not None:
                conv = self.time_conv.conv
                x = (
                    F.conv3d(
                        x.permute(0, 4, 1, 2, 3),
                        self.time_weight,
                        None,
                        conv.stride,
                        conv.padding,
                        conv.dilation,
                        conv.groups,
                    )
                    .permute(0, 2, 3, 4, 1)
                    .contiguous()
                )
                output_bias = conv.bias
            feat_cache = update_cache(feat_cache, feat_idx, packed.cache)
            feat_idx += 1
        return ConvOutput(x, output_bias), feat_cache, feat_idx


class _ResidualDownBlock(torch.nn.Module):
    """Keep the residual chain NTHWC; adapt once at unchanged Torch boundaries."""

    def __init__(self, block: WanResidualDownBlock) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            _PreparedResidualBlock(resnet, contiguous_norm1=True)
            for resnet in block.resnets
        )
        self.downsampler = (
            None if block.downsampler is None else _Downsample(block.downsampler)
        )
        self.avg_shortcut = block.avg_shortcut

    def forward(
        self,
        x: torch.Tensor | ConvOutput,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[torch.Tensor, FeatCache | None, int]:
        # No operation in this stage mutates the shortcut input.
        shortcut = x
        deferred_input = isinstance(x, ConvOutput)
        for index, resnet in enumerate(self.resnets):
            next_norm = None
            if index + 1 < len(self.resnets):
                following = self.resnets[index + 1]
                if resnet.primitive_convs["conv2"].out_channels in (160, 320):
                    if (
                        not following.contiguous_norm1
                        or following.conv_shortcut is not None
                    ):
                        raise ValueError(
                            "Inter-block fusion requires channels-last norm "
                            "and identity skip"
                        )
                    next_norm = following.norm1
            x, feat_cache, feat_idx = resnet(
                x,
                feat_cache,
                feat_idx,
                next_norm=next_norm,
                pad_output=index + 1 == len(self.resnets)
                and self.downsampler is not None,
            )
            if deferred_input and index == 0:
                # The first preparation writes the biased input for both skips.
                shortcut = x.residual
        if self.downsampler is not None:
            x, feat_cache, feat_idx = self.downsampler(x, feat_cache, feat_idx)
        else:
            if x.residual_bias is not None:
                raise ValueError("Final stage block must have an identity shortcut")
            x = primitive_stage_add(
                x.value,
                shortcut,
                self.avg_shortcut.factor_t,
                self.avg_shortcut.factor_s,
                x.bias,
                x.residual,
            )
            return x, feat_cache, feat_idx
        return (
            primitive_stage_add(
                x.value,
                shortcut,
                self.avg_shortcut.factor_t,
                self.avg_shortcut.factor_s,
                x.bias,
            ),
            feat_cache,
            feat_idx,
        )


class _MidBlock(torch.nn.Module):
    """NTHWC residual blocks and attention with deferred projection epilogue."""

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList(
            _PreparedResidualBlock(resnet, contiguous_norm1=True)
            for resnet in block.resnets
        )
        self.attentions = torch.nn.ModuleList(
            _AttentionBlock(attention) for attention in block.attentions
        )

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: int = 0
    ) -> tuple[ResidualConvOutput, FeatCache | None, int]:
        for index, resnet in enumerate(self.resnets):
            if index:
                x, feat_cache, feat_idx = self.attentions[index - 1](
                    x, feat_cache, feat_idx
                )
            out, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
            if index + 1 < len(self.resnets):
                x = out.materialize()
        return out, feat_cache, feat_idx


class _AttentionNorm(torch.nn.Module):
    """Fused attention normalization, with no SiLU and reference reduction order."""

    def __init__(self, norm: torch.nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.norm.bias
        bias = bias.reshape(-1) if isinstance(bias, torch.Tensor) else None
        return primitive_norm(
            x,
            self.norm.gamma.reshape(-1),
            bias,
        )


class _AttentionBlock(torch.nn.Module):
    """Keep attention NTHWC and fuse its final additions into the next prep.

    QKV and SDPA math are unchanged. Projection produces raw BF16 p; the
    consumer computes B(B(p + projection_bias) + identity), then normalizes
    and saves that rounded sum as its residual. B denotes BF16 rounding.
    Channels-last NCHW views are only arguments to cuDNN, never tensor copies.
    """

    def __init__(self, block: WanAttentionBlock) -> None:
        super().__init__()
        self.norm = _AttentionNorm(block.norm)
        self.to_qkv = with_channels_last_weights(block.to_qkv)
        self.proj = with_channels_last_weights(block.proj)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: int = 0,
    ) -> tuple[ResidualConvOutput, FeatCache | None, int]:
        identity = x
        n, t, h, w, c = x.shape
        normalized = self.norm(x).view(n * t, h, w, c).permute(0, 3, 1, 2)
        conv = self.to_qkv
        qkv = F.conv2d(
            normalized,
            conv.weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        qkv = qkv.reshape(n * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.squeeze(1).permute(0, 2, 1).reshape(n * t, c, h, w)
        conv = self.proj
        projected = F.conv2d(
            attended,
            conv.weight,
            None,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        ).permute(0, 2, 3, 1)
        if not projected.is_contiguous():
            raise ValueError("Attention projection must produce channels-last storage")
        return (
            ResidualConvOutput(projected.view(n, t, h, w, c), conv.bias, identity),
            feat_cache,
            feat_idx,
        )


class _ZeroPadConv(_ConvWeights):
    """Avoid F.pad's full tensor clone for convolutions with no explicit padding."""

    def forward(
        self, x: torch.Tensor, cache_x: torch.Tensor | None = None
    ) -> torch.Tensor:
        if cache_x is not None or any(self.conv._padding):
            raise ValueError(
                "Zero-pad adapter cannot consume history or nonzero padding"
            )
        conv = self.conv
        return F.conv3d(
            x,
            conv.weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )


class _Encoder3d(torch.nn.Module):
    """NTHWC custom stages and fused pending residual/output-head preparation."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        if not all(
            isinstance(block, WanResidualDownBlock) for block in encoder.down_blocks
        ):
            raise TypeError("Custom encoder requires Wan residual down stages")
        self.conv_in = _ConvWeights(encoder.conv_in)
        self.down_blocks = torch.nn.ModuleList(
            _ResidualDownBlock(block) for block in encoder.down_blocks
        )
        self.mid_block = _MidBlock(encoder.mid_block)
        self.norm_out = _ConvPrep(encoder.norm_out)
        self.conv_out = _ConvWeights(encoder.conv_out)
        self.cache_size = sum(
            isinstance(layer, WanCausalConv3d) for layer in encoder.modules()
        )
        if tuple(self.conv_out.conv._padding) != (1, 1, 1, 1, 2, 0):
            raise ValueError("Custom output head requires causal 3x3x3 padding")
        self.register_buffer(
            "head_weight",
            self.conv_out.conv.weight.detach().contiguous(
                memory_format=torch.channels_last_3d
            ),
            persistent=False,
        )
        conv = self.conv_in.conv
        if (
            tuple(conv._padding) != (1, 1, 1, 1, 2, 0)
            or tuple(conv.padding) != (0, 0, 0)
            or tuple(conv.stride) != (1, 1, 1)
            or tuple(conv.dilation) != (1, 1, 1)
            or conv.groups != 1
        ):
            raise ValueError(
                "Primitive input convolution requires causal unit-stride 3x3x3"
            )
        self.primitive_input_conv = WanInputConv3d(conv.weight)

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache
    ) -> tuple[torch.Tensor, FeatCache]:
        feat_cache = canonicalize_cache(feat_cache, torch.contiguous_format)
        packed = pack_history(x, feat_cache[0], input_ncthw=True)
        conv = self.conv_in.conv
        x = self.primitive_input_conv(packed.padded)
        feat_cache = update_cache(feat_cache, 0, packed.cache)
        idx = 1
        x = ConvOutput(x, conv.bias)
        for layer in self.down_blocks:
            x, feat_cache, idx = layer(x, feat_cache, idx)
        pending, feat_cache, idx = self.mid_block(x, feat_cache, idx)
        prepared = self.norm_out(
            pending.value,
            feat_cache[idx],
            True,
            pending.bias,
            pending.residual,
            residual_bias=pending.residual_bias,
        )
        conv = self.conv_out.conv
        x = F.conv3d(
            prepared.padded.permute(0, 4, 1, 2, 3),
            self.head_weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        feat_cache = update_cache(feat_cache, idx, prepared.cache)
        return x, feat_cache


class MegaWanVaeEncoder(torch.nn.Module):
    """Inference-only Wan encoder constructed from a loaded AutoencoderKLWan.

    Call eval() and run under torch.inference_mode() with BF16 CUDA weights and
    video [N,C,T,H,W], where T is 1 or 4n+1. Forward returns posterior parameters
    in NCTHW, not sampled latents. Inside the model, activations remain NTHWC.

    Residual and output-head cache slots are NTHWC. Convolution weights may be
    repacked without modifying the reference; attention matmuls remain in Torch.
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.encoder = _Encoder3d(vae.encoder)
        self.quant_conv = _ZeroPadConv(vae.quant_conv)
        self.patch_size = getattr(vae.config, "patch_size", None)
        self.use_tiling = vae.use_tiling
        self.tile_sample_min_width = vae.tile_sample_min_width
        self.tile_sample_min_height = vae.tile_sample_min_height

    def new_cache(self) -> FeatCache:
        """Create empty history for an independent video stream."""
        return (None,) * self.encoder.cache_size

    def encode_chunk(
        self, chunk: torch.Tensor, feat_cache: FeatCache
    ) -> tuple[torch.Tensor, FeatCache]:
        """Encode one already-patchified NCTHW chunk and return updated history."""
        return self.encoder(chunk, feat_cache)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode NCTHW video into posterior parameters, with fresh temporal history.

        Spatial tiling is unsupported. This is an inference encoder, not the
        Diffusers encode() posterior wrapper or a replacement for its decoder.
        """
        _, _, num_frame, height, width = video.shape
        validate_frame_count(num_frame)
        if self.patch_size is not None:
            video = patchify(video, patch_size=int(self.patch_size))
        if self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            raise NotImplementedError(
                "MegaWanVaeEncoder does not implement spatial tiling."
            )

        feat_cache = self.new_cache()
        outs: list[torch.Tensor] = []
        for i in range(1 + (num_frame - 1) // 4):
            chunk = (
                video[:, :, :1] if i == 0 else video[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            )
            out, feat_cache = self.encode_chunk(chunk, feat_cache)
            outs.append(out)
        out = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
        return self.quant_conv(out)
