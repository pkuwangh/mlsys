"""Code predictor using the talker's fused decode blocks and Torch attention."""

from copy import deepcopy

from torch import nn

from .talker import _FusedTalkerLayer


class MegaQwen3TTSPredictor(nn.Module):
    """Reuse both fused blocks; the two-token prefix stays on the Torch path."""

    def __init__(self, reference, *, fused_selection=True):
        super().__init__()
        self.fused_selection = fused_selection
        self.model = deepcopy(reference)
        for index, layer in enumerate(self.model.layers):
            self.model.layers[index] = _FusedTalkerLayer(layer, fused_mask=False)

    def make_cache(self, batch_size):
        return self.model.make_cache(batch_size)

    def forward(self, embeddings, cache, positions, head_weight, cos, sin, mask):
        return self.model(embeddings, cache, positions, head_weight, cos, sin, mask)

    def generate_frame(self, hidden, first_code, cache, **kwargs):
        return self.model.generate_frame(
            hidden,
            first_code,
            cache,
            step=self,
            select=self.select_embedding if self.fused_selection else None,
            **kwargs,
        )

    @staticmethod
    def select_embedding(logits, table):
        """Select a greedy code and its next-step embedding in one launch."""
        from .kernels.argmax_embedding import argmax_embedding

        codes, embeddings, _ = argmax_embedding(logits.reshape(-1, logits.shape[-1]), table)
        return codes[:, None], embeddings[:, None, :]
