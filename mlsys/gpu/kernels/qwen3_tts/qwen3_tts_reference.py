"""Torch Qwen3-TTS talker backbone and codec logits, with upstream BF16 rounding.

Inputs are already prepared embeddings. Prompt construction, code prediction,
sampling, and waveform decoding are outside this benchmark. Text/audio positions
use the same coordinate in all three RoPE axes, as in ordinary TTS decoding.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TalkerConfig:
    """Default dimensions of Qwen3-TTS-12Hz-0.6B."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @classmethod
    def from_checkpoint(cls, path: str) -> "TalkerConfig":
        data = json.loads((Path(path) / "config.json").read_text())["talker_config"]
        return cls(**{field.name: data[field.name] for field in fields(cls)})


class RMSNorm(nn.Module):
    """Normalize in FP32, round, then multiply by the BF16 weight."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class TorchPostAttention(nn.Module):
    """Output projection, two residual adds, RMSNorm, and SwiGLU MLP."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        d, f = config.hidden_size, config.intermediate_size
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, d, bias=False)
        self.norm = RMSNorm(d, config.rms_norm_eps)
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, attention, residual):
        return post_attention(
            attention,
            residual,
            self.o_proj.weight,
            self.norm.weight,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.norm.eps,
        )


def post_attention(attention, residual, wo, gamma, gate, up, down, eps):
    """Upstream BF16 boundaries, with explicit weights for rotating-weight timing."""
    residual = F.linear(attention, wo) + residual
    rf = residual.float()
    normalized = (rf * torch.rsqrt(rf.square().mean(-1, keepdim=True) + eps)).to(residual.dtype) * gamma
    activated = F.silu(F.linear(normalized, gate)) * F.linear(normalized, up)
    return F.linear(activated, down) + residual


class TalkerAttention(nn.Module):
    """Q/K-normalized GQA with a fixed-capacity KV cache and Torch SDPA."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        d, h = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(d, config.num_attention_heads * h, bias=False)
        self.k_proj = nn.Linear(d, config.num_key_value_heads * h, bias=False)
        self.v_proj = nn.Linear(d, config.num_key_value_heads * h, bias=False)
        self.q_norm = RMSNorm(h, config.rms_norm_eps)
        self.k_norm = RMSNorm(h, config.rms_norm_eps)

    def forward(self, x, cache, positions, cos, sin, mask):
        b, t, _ = x.shape
        h = self.config.head_dim
        q = self.q_norm(self.q_proj(x).view(b, t, -1, h)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, h)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, h).transpose(1, 2)
        qrot = torch.cat((-q[..., h // 2 :], q[..., : h // 2]), dim=-1)
        krot = torch.cat((-k[..., h // 2 :], k[..., : h // 2]), dim=-1)
        q = q * cos + qrot * sin
        k = k * cos + krot * sin
        cache[0].index_copy_(2, positions, k)
        cache[1].index_copy_(2, positions, v)
        out = F.scaled_dot_product_attention(q, cache[0], cache[1], attn_mask=mask, enable_gqa=True)
        return out.transpose(1, 2).reshape(b, t, -1)


class TalkerLayer(nn.Module):
    """One upstream-style layer with a replaceable post-attention module."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TalkerAttention(config)
        self.post_attention = TorchPostAttention(config)

    def forward(self, x, cache, positions, cos, sin, mask):
        attention = self.self_attn(self.input_layernorm(x), cache, positions, cos, sin, mask)
        return self.post_attention(attention, x)


class TorchQwen3TTSTalker(nn.Module):
    """All talker layers, final norm, and codec head; no generation framework."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TalkerLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self, batch_size: int, capacity: int):
        parameter = self.codec_head.weight
        c = self.config
        return torch.zeros(
            c.num_hidden_layers,
            2,
            batch_size,
            c.num_key_value_heads,
            capacity,
            c.head_dim,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def forward(self, embeddings, cache, positions):
        # Compute frequencies in FP32 even when model weights were cast to BF16.
        inv = 1.0 / self.config.rope_theta ** (
            torch.arange(0, self.config.head_dim, 2, device=embeddings.device).float() / self.config.head_dim
        )
        angles = positions.float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(embeddings.dtype)[None, None, :, :]
        sin = angles.sin().to(embeddings.dtype)[None, None, :, :]
        mask = torch.arange(cache.shape[-2], device=cache.device)[None, :] <= positions[:, None]
        x = embeddings
        for index, layer in enumerate(self.layers):
            x = layer(x, cache[index], positions, cos, sin, mask)
        return self.codec_head(self.norm(x))

    @torch.no_grad()
    def load_checkpoint(self, path: str):
        """Read only backbone/head tensors from local safetensors shards."""
        from safetensors import safe_open

        expected = self.state_dict()
        loaded = set()
        for shard in sorted(Path(path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict or iterable.
                    if key.startswith(("talker.model.layers.", "talker.model.norm.")):
                        name = key.removeprefix("talker.model.")
                        name = name.replace(".self_attn.o_proj.", ".post_attention.o_proj.")
                        name = name.replace(".post_attention_layernorm.", ".post_attention.norm.")
                        name = name.replace(".mlp.", ".post_attention.")
                    elif key.startswith("talker.codec_head."):
                        name = key.removeprefix("talker.")
                    else:
                        continue
                    if name in expected:
                        expected[name].copy_(handle.get_tensor(key))
                        loaded.add(name)
        missing = expected.keys() - loaded
        if missing:
            raise ValueError(f"Missing talker weights: {sorted(missing)}")
