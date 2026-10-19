"""Talker composition with a replaceable full post-attention block."""

from copy import deepcopy

import torch
from torch import nn


class MegaQwen3TTSTalker(nn.Module):
    """Start with Torch blocks; replace only the supported decode sublayer."""

    def __init__(self, reference, *, fused: bool = True):
        super().__init__()
        self.model = deepcopy(reference)
        if fused:
            from .kernels.post_attention import FusedPostAttention

            for layer in self.model.layers:
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
