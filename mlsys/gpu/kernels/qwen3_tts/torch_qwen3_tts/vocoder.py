# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qwen3-TTS modeling_qwen3_tts_tokenizer_v2.py.

"""Torch reference for the convolutional waveform decoder after the codec transformer.

Inputs are BCT features after the two ConvNeXt upsamplers. Parameter names and
BF16 operation boundaries match Qwen3-TTS's decoder ModuleList.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SnakeBeta(nn.Module):
    """Learned periodic activation with upstream intermediate rounding."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).square()


class CausalConv1d(nn.Module):
    """Left-zero-padded, stride-one convolution."""

    def __init__(self, ci, co, kernel, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(ci, co, kernel, dilation=dilation)
        self.padding = (kernel - 1) * dilation

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0))).contiguous()


class CausalConvTranspose1d(nn.Module):
    """Upsample then crop the right-hand noncausal tail."""

    def __init__(self, ci, co, rate):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ci, co, 2 * rate, stride=rate)
        self.crop = rate

    def forward(self, x):
        return self.conv(x)[..., : -self.crop].contiguous()


class ChannelsLastConv1d(nn.Module):
    """cuDNN baseline on BTC storage, folding dilation into independent spatial lanes."""

    def __init__(self, conv):
        super().__init__()
        self.register_buffer("weight", conv.weight.detach().unsqueeze(-1).contiguous(memory_format=torch.channels_last))
        self.register_buffer("bias", conv.bias.detach() if conv.bias is not None else None)
        self.dilation = conv.dilation[0]
        self.padding = (conv.kernel_size[0] - 1) * self.dilation

    def prepare_input(self, x):
        """Left-pad and fold dilation without materializing an im2col matrix."""
        padded = F.pad(x, (0, 0, self.padding, -(x.shape[1] + self.padding) % self.dilation))
        return padded.reshape(x.shape[0], -1, self.dilation, x.shape[2]).permute(0, 3, 1, 2)

    def forward(self, x):
        result = F.conv2d(self.prepare_input(x), self.weight, self.bias)
        result = result.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.weight.shape[0])
        return result[:, : x.shape[1]].contiguous()


class ResidualUnit(nn.Module):
    """Snake, dilated K7 convolution, Snake, K1 convolution, and residual."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = CausalConv1d(channels, channels, 7, dilation)
        self.act2 = SnakeBeta(channels)
        self.conv2 = CausalConv1d(channels, channels, 1)

    def forward(self, x):
        return self.conv2(self.act2(self.conv1(self.act1(x)))) + x


class DecoderBlock(nn.Module):
    """One learned upsample followed by three dilated residual units."""

    def __init__(self, ci, co, rate):
        super().__init__()
        self.block = nn.ModuleList(
            [SnakeBeta(ci), CausalConvTranspose1d(ci, co, rate)]
            + [ResidualUnit(co, dilation) for dilation in (1, 3, 9)]
        )

    def forward(self, x):
        for block in self.block:
            x = block(x)
        return x


class TorchVocoderTail(nn.Module):
    """Convolutional tail only; quantizer, transformer, and ConvNeXt are upstream."""

    def __init__(self, latent_dim=1024, decoder_dim=1536, rates=(8, 5, 4, 3)):
        super().__init__()
        layers = [CausalConv1d(latent_dim, decoder_dim, 7)]
        for index, rate in enumerate(rates):
            ci = decoder_dim // 2**index
            layers.append(DecoderBlock(ci, ci // 2, rate))
        channels = decoder_dim // 2 ** len(rates)
        layers.extend([SnakeBeta(channels), CausalConv1d(channels, 1, 7)])
        self.decoder = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x.clamp(-1, 1)

    @classmethod
    def from_checkpoint(cls, model, *, device="cuda", dtype=torch.bfloat16):
        """Load only waveform-tail weights from a local speech tokenizer checkpoint."""
        from safetensors.torch import load_file

        path = Path(model).expanduser()
        if (path / "speech_tokenizer").is_dir():
            path /= "speech_tokenizer"
        config = json.loads((path / "config.json").read_text())["decoder_config"]
        with torch.device("meta"):
            result = cls(config["latent_dim"], config["decoder_dim"], config["upsample_rates"])
        state = load_file(str(path / "model.safetensors"))
        prefix = "decoder."
        weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith("decoder.decoder.")}
        result.load_state_dict(weights, assign=True)
        return result.to(device=device, dtype=dtype).eval()


class CodecDecoder(nn.Module):
    """Reuse the upstream codec prefix with a replaceable waveform tail."""

    def __init__(self, upstream, tail):
        super().__init__()
        self.quantizer = upstream.quantizer
        self.pre_conv = upstream.pre_conv
        self.pre_transformer = upstream.pre_transformer
        self.upsample = upstream.upsample
        self.tail = tail

    def forward(self, codes):
        x = self.pre_conv(self.quantizer.decode(codes)).transpose(1, 2)
        x = self.pre_transformer(inputs_embeds=x).last_hidden_state.transpose(1, 2)
        for blocks in self.upsample:
            for block in blocks:
                x = block(x)
        return self.tail(x)


def load_upstream_decoder(model):
    """Load the optional qwen-tts full codec decoder for architectural validation."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from safetensors.torch import load_file

    path = Path(model).expanduser()
    if (path / "speech_tokenizer").is_dir():
        path /= "speech_tokenizer"
    config = Qwen3TTSTokenizerV2DecoderConfig(**json.loads((path / "config.json").read_text())["decoder_config"])
    config._attn_implementation = "sdpa"
    decoder = Qwen3TTSTokenizerV2Decoder(config)
    state = load_file(str(path / "model.safetensors"))
    decoder.load_state_dict({k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")})
    return decoder.cuda().bfloat16().eval()
