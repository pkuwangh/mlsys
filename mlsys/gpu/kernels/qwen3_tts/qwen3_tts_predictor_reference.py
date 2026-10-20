"""Upstream-style code predictor: one audio frame's 15 additional codec codes."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from qwen3_tts_reference import RMSNorm, TalkerConfig, TalkerLayer
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictorConfig(TalkerConfig):
    """Default predictor dimensions of Qwen3-TTS-12Hz-0.6B."""

    num_hidden_layers: int = 5
    vocab_size: int = 2048
    num_code_groups: int = 16
    talker_hidden_size: int = 1024
    talker_vocab_size: int = 3072

    @classmethod
    def from_checkpoint(cls, path: str) -> "PredictorConfig":
        talker = json.loads((Path(path) / "config.json").read_text())["talker_config"]
        data = dict(talker["code_predictor_config"])
        data.update(talker_hidden_size=talker["hidden_size"], talker_vocab_size=talker["vocab_size"])
        return cls(**{field.name: data[field.name] for field in fields(cls)})


class TorchQwen3TTSPredictor(nn.Module):
    """Five transformer layers with group-specific codec embeddings and output heads."""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TalkerLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.first_embedding = nn.Embedding(config.talker_vocab_size, config.talker_hidden_size)
        self.codec_embedding = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.talker_hidden_size) for _ in range(config.num_code_groups - 1)
        )
        self.lm_head = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)
        )
        self.small_to_mtp_projection = (
            nn.Linear(config.talker_hidden_size, config.hidden_size)
            if config.talker_hidden_size != config.hidden_size
            else nn.Identity()
        )
        self.register_buffer("positions", torch.arange(config.num_code_groups), persistent=False)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.register_buffer("attention_masks", None, persistent=False)

    def make_cache(self, batch_size):
        """Prepare position tables and allocate a frame-local cache, before graph capture."""
        c = self.config
        # Compute frequencies on the final device in FP32 before rounding cos/sin.
        inv = 1.0 / c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=self.positions.device).float() / c.head_dim)
        angles = self.positions.float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self.rope_cos = angles.cos().to(self.norm.weight.dtype)[None, None, :, :]
        self.rope_sin = angles.sin().to(self.norm.weight.dtype)[None, None, :, :]
        self.attention_masks = self.norm.weight.new_zeros(c.num_code_groups, c.num_code_groups)
        self.attention_masks.masked_fill_(self.positions[None, :] > self.positions[:, None], float("-inf"))
        return self.norm.weight.new_zeros(
            c.num_hidden_layers, 2, batch_size, c.num_key_value_heads, c.num_code_groups, c.head_dim
        )

    def forward(self, embeddings, cache, positions, head_weight, cos, sin, mask):
        """Run a prefix or single-code step, returning the last token's logits."""
        x = self.small_to_mtp_projection(embeddings)
        for index, layer in enumerate(self.layers):
            x = layer(x, cache[index], positions, cos, sin, mask)
        return F.linear(self.norm(x[:, -1:]), head_weight)

    def generate_frame(self, hidden, first_code, cache, *, forced_codes=None, return_logits=False, step=None):
        """Generate greedily, or replay fixed codes to compare logits without sampling divergence."""
        step = self if step is None else step
        first = self.first_embedding(first_code)
        embeddings = torch.cat((hidden, first), dim=1)
        codes, feedback, logits = [first_code], [first], []
        for group, head in enumerate(self.lm_head):
            start, end = (0, 2) if group == 0 else (group + 1, group + 2)
            prediction = step(
                embeddings,
                cache,
                self.positions[start:end],
                head.weight,
                self.rope_cos[:, :, start:end],
                self.rope_sin[:, :, start:end],
                self.attention_masks[start:end],
            )
            code = prediction.argmax(-1) if forced_codes is None else forced_codes[:, group : group + 1]
            embedding = self.codec_embedding[group](code)
            codes.append(code)
            feedback.append(embedding)
            if return_logits:
                logits.append(prediction)
            embeddings = embedding
        # Match upstream's single reduction, not sequential BF16 additions.
        return (
            torch.cat(codes, dim=1),
            torch.cat(feedback, dim=1).sum(1, keepdim=True),
            torch.cat(logits, dim=1) if return_logits else None,
        )

    @torch.no_grad()
    def load_checkpoint(self, path: str):
        """Read predictor weights and the talker's first-code embedding from safetensors."""
        from safetensors import safe_open

        expected, loaded = self.state_dict(), set()
        for shard in sorted(Path(path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict or iterable.
                    if key == "talker.model.codec_embedding.weight":
                        name = "first_embedding.weight"
                    elif key.startswith("talker.code_predictor."):
                        name = key.removeprefix("talker.code_predictor.").removeprefix("model.")
                        name = name.replace(".self_attn.o_proj.", ".post_attention.o_proj.")
                        name = name.replace(".post_attention_layernorm.", ".post_attention.norm.")
                        name = name.replace(".mlp.", ".post_attention.")
                    else:
                        continue
                    if name in expected:
                        expected[name].copy_(handle.get_tensor(key))
                        loaded.add(name)
        missing = expected.keys() - loaded
        if missing:
            raise ValueError(f"Missing predictor weights: {sorted(missing)}")
