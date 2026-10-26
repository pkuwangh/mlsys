"""BTC waveform decoder with replaceable Conv1D and SnakeBeta blocks.

Only the convolutional tail is replaced. The codec transformer and its two
ConvNeXt upsamplers remain unchanged. Transposed convolutions and the final
one-channel convolution use channels-last cuDNN.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_qwen3_tts.vocoder import ChannelsLastConv1d

from .kernels.conv1d import conv1d
from .kernels.snake_beta import SnakeBeta


class Activation(nn.Module):
    """SnakeBeta on BTC tensors without a layout copy."""

    def __init__(self, reference, custom):
        super().__init__()
        self.operation = SnakeBeta(reference.alpha, reference.beta) if custom else reference

    def forward(self, x):
        return self.operation(x.transpose(1, 2)).transpose(1, 2)


class Convolution(nn.Module):
    """Prepack immutable weights and optionally fuse a following activation or residual."""

    def __init__(self, reference, custom):
        super().__init__()
        conv = reference.conv
        self.custom = custom and conv.out_channels % 32 == 0 and conv.in_channels % 32 == 0
        self.dilation = conv.dilation[0]
        if self.custom:
            self.register_buffer("weight", conv.weight.detach().permute(0, 2, 1).contiguous())
            self.register_buffer("bias", conv.bias.detach())
        else:
            self.baseline = ChannelsLastConv1d(conv)
            if custom:
                self.baseline = torch.compile(
                    self.baseline, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True}
                )

    def forward(self, x, *, activation=None, residual=None):
        if self.custom:
            return conv1d(x, self.weight, self.bias, dilation=self.dilation, activation=activation, residual=residual)
        x = self.baseline(x)
        if residual is not None:
            x = x + residual
        if activation is not None:
            x = activation(x.transpose(1, 2)).transpose(1, 2)
        return x


class Upsample(nn.Module):
    """Channels-last cuDNN transposed convolution followed by causal cropping."""

    def __init__(self, reference, custom):
        super().__init__()
        conv = reference.conv
        self.register_buffer("weight", conv.weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last))
        self.register_buffer("bias", conv.bias.detach())
        self.rate = conv.stride[0]
        self.crop = reference.crop
        self.operation = (
            torch.compile(self._operation, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if custom
            else self._operation
        )

    def _operation(self, x):
        result = F.conv_transpose2d(x.transpose(1, 2).unsqueeze(2), self.weight, self.bias, stride=(1, self.rate))
        return result.squeeze(2).transpose(1, 2)[:, : -self.crop].contiguous()

    def forward(self, x):
        return self.operation(x)


class ResidualUnit(nn.Module):
    """Fuse K7+bias+Snake and K1+bias+residual when custom convolution is enabled."""

    def __init__(self, reference, custom):
        super().__init__()
        self.act1 = Activation(reference.act1, custom)
        self.act2 = SnakeBeta(reference.act2.alpha, reference.act2.beta) if custom else reference.act2
        self.conv1 = Convolution(reference.conv1, custom)
        self.conv2 = Convolution(reference.conv2, custom)

    def forward(self, x):
        hidden = self.conv1(self.act1(x), activation=self.act2)
        return self.conv2(hidden, residual=x)


class DecoderBlock(nn.Module):
    """One upsampling stage and its three residual units."""

    def __init__(self, reference, custom):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Activation(reference.block[0], custom),
                Upsample(reference.block[1], custom),
                *[ResidualUnit(layer, custom) for layer in reference.block[2:]],
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MegaVocoderTail(nn.Module):
    """Keep intermediate tensors in BTC; convert only at the tail's boundaries."""

    def __init__(self, reference, *, custom=True):
        super().__init__()
        layers = reference.decoder
        self.layers = nn.ModuleList(
            [
                Convolution(layers[0], custom),
                *[DecoderBlock(layer, custom) for layer in layers[1:-2]],
                Activation(layers[-2], custom),
                Convolution(layers[-1], custom),
            ]
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2).clamp(-1, 1).contiguous()
