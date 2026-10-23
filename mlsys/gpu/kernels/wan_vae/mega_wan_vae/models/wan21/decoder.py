"""Wan 2.1 decoder stages without outer shortcuts or pixel patchification."""

import torch

from ...layers.fused import _DecoderWithPreparedHead, _FusedMidBlock, _FusedUpBlock
from ...layers.torch import TorchWanVaeDecoder
from ._common import _CUSTOM_CONV_CHANNELS, _compile


class Wan21Decoder(torch.nn.Module):
    """Decode latent frames, preserving the first-frame upsample schedule."""

    def __init__(self, vae):
        super().__init__()
        self.runtime = TorchWanVaeDecoder(vae).eval()
        decoder = self.runtime.decoder
        decoder.mid_block = _FusedMidBlock(decoder.mid_block)
        decoder.up_blocks = torch.nn.ModuleList(
            _FusedUpBlock(stage, custom_conv=stage.resnets[0].conv1.conv.out_channels in _CUSTOM_CONV_CHANNELS)
            for stage in decoder.up_blocks
        )
        self.fused_blocks = sum(len(stage.resnets) for stage in decoder.up_blocks)
        self.chunk_decoder = _compile(_DecoderWithPreparedHead(decoder))

    def forward(self, latent):
        return self.runtime.decode_temporal_chunks(latent, self.chunk_decoder)
