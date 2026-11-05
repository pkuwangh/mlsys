"""Wan 2.1 decoder stages without outer shortcuts or pixel patchification."""

import torch

from ...layers.fused import _FusedMidBlock, _FusedUpBlock, compile_preserving_norms
from ...layers.torch import TorchWanVaeDecoder
from ._common import _custom_channels


class Wan21Decoder(torch.nn.Module):
    """Decode latent frames, preserving the first-frame upsample schedule."""

    def __init__(self, vae):
        super().__init__()
        self.runtime = TorchWanVaeDecoder(vae).eval()
        decoder = self.runtime.decoder
        decoder.mid_block = _FusedMidBlock(decoder.mid_block, custom_conv=384 in _custom_channels(vae.dtype))
        decoder.up_blocks = torch.nn.ModuleList(
            _FusedUpBlock(stage, custom_conv=stage.resnets[0].conv1.conv.out_channels in _custom_channels(vae.dtype))
            for stage in decoder.up_blocks
        )
        self.fused_blocks = sum(len(stage.resnets) for stage in decoder.up_blocks)
        self.chunk_decoder = compile_preserving_norms(decoder)

    def forward(self, latent):
        return self.runtime.decode_temporal_chunks(latent, self.chunk_decoder)
