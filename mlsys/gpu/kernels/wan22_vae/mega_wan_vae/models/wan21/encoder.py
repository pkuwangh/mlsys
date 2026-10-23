"""Wan 2.1 encoder stages without outer shortcuts or pixel patchification."""

import torch

from ...layers.fused import _FusedMidBlock, _FusedResidualBlock
from ...layers.torch import RawWanResidualBlock, TorchWanVaeEncoder
from ._common import _CUSTOM_CONV_CHANNELS, _compile


class _EncoderResidual(_FusedResidualBlock):
    def forward(self, x, feat_cache, feat_idx=0):
        x, feat_cache, feat_idx, _ = super().forward(x, feat_cache, feat_idx)
        return x, feat_cache, feat_idx


class Wan21Encoder(torch.nn.Module):
    """Encode video into posterior parameters with fresh temporal history."""

    def __init__(self, vae):
        super().__init__()
        self.runtime = TorchWanVaeEncoder(vae).eval()
        encoder = self.runtime.encoder
        for index, block in enumerate(encoder.down_blocks):
            if isinstance(block, RawWanResidualBlock):
                encoder.down_blocks[index] = _EncoderResidual(
                    block, custom_conv=block.conv1.conv.out_channels in _CUSTOM_CONV_CHANNELS
                )
        encoder.mid_block = _FusedMidBlock(encoder.mid_block)
        self.chunk_encoder = _compile(encoder)

    def new_cache(self):
        return self.runtime.new_cache()

    def encode_chunk(self, chunk, feat_cache):
        return self.chunk_encoder(chunk, feat_cache)

    def forward(self, video):
        return self.runtime.encode_temporal_chunks(video, self.chunk_encoder)
