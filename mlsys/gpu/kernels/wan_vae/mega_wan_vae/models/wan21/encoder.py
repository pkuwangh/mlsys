"""Wan 2.1 encoder stages without outer shortcuts or pixel patchification."""

import torch

from ...layers.fused import _FusedMidBlock, _FusedResidualBlock
from ...layers.torch import RawWanResample, RawWanResidualBlock, TorchWanVaeEncoder
from ._common import _CUSTOM_CONV_CHANNELS, _compile


class _EncoderResidual(_FusedResidualBlock):
    def forward(self, x, feat_cache, feat_idx=0):
        x, feat_cache, feat_idx, _ = super().forward(x, feat_cache, feat_idx)
        return x, feat_cache, feat_idx


class _EncoderResidualStage(torch.nn.Module):
    """Prepare adjacent residual inputs and fuse the final spatial border."""

    def __init__(self, blocks, downsampler=None):
        super().__init__()
        self.resnets = torch.nn.ModuleList(_FusedResidualBlock(block, custom_conv=True) for block in blocks)
        self.downsampler = downsampler
        if downsampler is not None:
            if downsampler.mode not in ("downsample2d", "downsample3d") or tuple(downsampler.resample[0].padding) != (
                0,
                1,
                0,
                1,
            ):
                raise ValueError("Prepared downsample requires bottom/right spatial padding")
            # Prepared weights may be borrowed from the reference VAE. Own the
            # sequence before replacing padding; keep its convolution shared.
            downsampler.resample = torch.nn.Sequential(
                torch.nn.ZeroPad2d(0), *list(downsampler.resample.children())[1:]
            )

    def forward(self, x, feat_cache, feat_idx=0):
        prepared = None
        for index, block in enumerate(self.resnets):
            next_gamma = self.resnets[index + 1].gamma1 if index + 1 < len(self.resnets) else None
            x, feat_cache, feat_idx, prepared = block(
                x,
                feat_cache,
                feat_idx,
                prepared,
                next_gamma,
                spatial_output=next_gamma is None and self.downsampler is not None,
            )
        if self.downsampler is not None:
            x, feat_cache, feat_idx = self.downsampler(x, feat_cache, feat_idx)
        return x, feat_cache, feat_idx


class Wan21Encoder(torch.nn.Module):
    """Encode video into posterior parameters with fresh temporal history."""

    def __init__(self, vae):
        super().__init__()
        self.runtime = TorchWanVaeEncoder(vae).eval()
        encoder = self.runtime.encoder
        stages = []
        residuals = []
        for block in encoder.down_blocks:
            if isinstance(block, RawWanResidualBlock) and block.conv1.conv.out_channels in _CUSTOM_CONV_CHANNELS:
                residuals.append(block)
                continue
            if residuals:
                consume_downsample = isinstance(block, RawWanResample) and block.mode.startswith("downsample")
                stages.append(_EncoderResidualStage(residuals, block if consume_downsample else None))
                residuals = []
                if consume_downsample:
                    continue
            stages.append(_EncoderResidual(block) if isinstance(block, RawWanResidualBlock) else block)
        if residuals:
            stages.append(_EncoderResidualStage(residuals))
        encoder.down_blocks = torch.nn.ModuleList(stages)
        encoder.mid_block = _FusedMidBlock(encoder.mid_block)
        self.chunk_encoder = _compile(encoder)

    def new_cache(self):
        return self.runtime.new_cache()

    def encode_chunk(self, chunk, feat_cache):
        return self.chunk_encoder(chunk, feat_cache)

    def forward(self, video):
        return self.runtime.encode_temporal_chunks(video, self.chunk_encoder)
