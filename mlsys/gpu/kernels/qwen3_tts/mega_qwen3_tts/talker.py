"""Talker with fused decode preparation and post-attention blocks."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class _FusedTalkerLayer(nn.Module):
    """Keep SDPA between the two fused blocks; leave multi-token prefill unchanged."""

    def __init__(self, layer):
        super().__init__()
        from .kernels.post_attention import FusedPostAttention
        from .kernels.pre_attention import FusedPreAttention

        self.pre_attention = FusedPreAttention(layer.input_layernorm, layer.self_attn)
        self.post_attention = FusedPostAttention(layer.post_attention)

    def forward(self, x, cache, positions, cos, sin, mask):
        if x.shape[1] == 1:
            q = self.pre_attention(x, cache, positions, cos, sin)
            attention = F.scaled_dot_product_attention(q, cache[0], cache[1], attn_mask=mask, enable_gqa=True)
            attention = attention.transpose(1, 2).reshape(x.shape[0], 1, -1)
        else:
            attention = self.pre_attention.attention(self.pre_attention.norm(x), cache, positions, cos, sin, mask)
        return self.post_attention(attention, x)


class MegaQwen3TTSTalker(nn.Module):
    """Replace supported decode blocks without changing cache ownership or layout."""

    def __init__(self, reference, *, fused: bool = True, fused_qkv: bool = True):
        super().__init__()
        self.model = deepcopy(reference)
        if fused:
            from .kernels.post_attention import FusedPostAttention

            for index, layer in enumerate(self.model.layers):
                if fused_qkv:
                    self.model.layers[index] = _FusedTalkerLayer(layer)
                else:
                    layer.post_attention = FusedPostAttention(layer.post_attention)

    def make_cache(self, batch_size, capacity):
        return self.model.make_cache(batch_size, capacity)

    def forward(self, embeddings, cache, positions):
        if embeddings.shape[1] != 1:
            return self.prefill(embeddings, cache, positions)
        return self.model(embeddings, cache, positions)

    @torch.compiler.disable
    def prefill(self, embeddings, cache, positions):
        """Keep the prefix identical to eager; only decode is optimized here."""
        return self.model(embeddings, cache, positions)
