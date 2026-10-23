"""Public Wan VAE encoder, dispatching from the checkpoint architecture."""

import torch


class MegaWanVaeEncoder(torch.nn.Module):
    """BF16 inference encoder for Wan 2.1 and Wan 2.2 checkpoints.

    Accept NCTHW pixels with T=4n+1 and return NCTHW posterior parameters
    (mean and log variance), without sampling or latent mean/std scaling.
    Each forward starts with fresh temporal history. Spatial tiling is not
    implemented. Construct from a loaded AutoencoderKLWan, then use eval()
    and torch.inference_mode().
    """

    def __init__(self, vae):
        super().__init__()
        if vae.dtype != torch.bfloat16:
            raise ValueError("Custom Wan VAE kernels require BF16")
        if vae.config.is_residual:
            from .models.wan22 import Wan22Encoder

            self.impl = Wan22Encoder(vae)
        else:
            from .models.wan21 import Wan21Encoder

            self.impl = Wan21Encoder(vae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)

    def new_cache(self):
        return self.impl.new_cache()

    def encode_chunk(self, chunk, feat_cache):
        return self.impl.encode_chunk(chunk, feat_cache)
