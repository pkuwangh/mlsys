"""Public Wan VAE decoder, dispatching from the checkpoint architecture."""

import torch


class MegaWanVaeDecoder(torch.nn.Module):
    """BF16 inference decoder for Wan 2.1 and Wan 2.2 checkpoints.

    Accept unnormalized NCTHW latents and return pixels clamped to [-1, 1].
    Latent mean/std scaling belongs to the caller. Each forward starts with
    fresh temporal history; spatial tiling is not implemented. Construct from
    a loaded AutoencoderKLWan, then use eval() and torch.inference_mode().
    """

    def __init__(self, vae):
        super().__init__()
        if vae.dtype != torch.bfloat16:
            raise ValueError("Custom Wan VAE kernels require BF16")
        if vae.config.is_residual:
            from .models.wan22 import Wan22Decoder

            self.impl = Wan22Decoder(vae)
        else:
            from .models.wan21 import Wan21Decoder

            self.impl = Wan21Decoder(vae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)

    @property
    def fused_blocks(self):
        return self.impl.fused_blocks
