"""Wan 2.2 decoder with shape-specialized convolution fusion."""

import torch

from ..._utils import update_cache
from ...kernels.conv import WanConv3d
from ...layers.fused import _FusedMidBlock, _FusedUpBlock, _history, _input_conv3d
from ...layers.torch import TorchWanVaeDecoder, raw_cached_causal_conv3d, raw_canonicalize_cache


class _FusedDecoder(torch.nn.Module):
    """Replace latent input packing/convolution and attention post-processing."""

    def __init__(self, decoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.input_conv = WanConv3d(decoder.conv_in.conv.weight)
        self.decoder.mid_block = _FusedMidBlock(decoder.mid_block)
        self.cache_size = decoder.cache_size

    def forward(self, x, feat_cache, first_chunk=False):
        """Retain the decoder's cache ordering and unchanged upsampling/output head."""
        feat_cache = raw_canonicalize_cache(feat_cache)
        raw, cache = _input_conv3d(
            x, self.input_conv.weight, _history(feat_cache[0]), self.input_conv.ctas, self.input_conv.precision
        )
        x = (raw + self.decoder.conv_in.conv.bias).permute(0, 4, 1, 2, 3)
        feat_cache = update_cache(feat_cache, 0, cache.permute(0, 4, 1, 2, 3))
        x, feat_cache, feat_idx = self.decoder.mid_block(x, feat_cache, 1)
        for block in self.decoder.up_blocks:
            x, feat_cache, feat_idx = block(x, feat_cache, feat_idx, first_chunk)
        x = self.decoder.activation(self.decoder.norm_out(x))
        x, feat_cache, _ = raw_cached_causal_conv3d(self.decoder.conv_out, x, feat_cache, feat_idx)
        return x, feat_cache


class Wan22Decoder(torch.nn.Module):
    """Compile a decoder with custom input, attention-boundary, and residual blocks."""

    def __init__(self, vae) -> None:
        super().__init__()
        if vae.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Decoder custom kernels require BF16 or FP32")
        self.raw_decoder = TorchWanVaeDecoder(vae).eval()
        self.fused_blocks = 0
        for index, stage in enumerate(self.raw_decoder.decoder.up_blocks):
            if stage.resnets[0].conv1.conv.out_channels in (256, 512):
                self.raw_decoder.decoder.up_blocks[index] = _FusedUpBlock(stage)
                self.fused_blocks += len(stage.resnets)
        if not self.fused_blocks:
            raise ValueError("No supported decoder blocks found")
        self.raw_decoder.decoder = _FusedDecoder(self.raw_decoder.decoder)
        self.chunk_decoder = torch.compile(
            self.raw_decoder.decoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode with fresh history using the reference chunk loop."""
        return self.raw_decoder.decode_temporal_chunks(latent, self.chunk_decoder)
