"""Public Wan VAE encoder, dispatching from the checkpoint architecture."""

import torch

from ._policy import check_model_policy, model_tf32_policy


class MegaWanVaeEncoder(torch.nn.Module):
    """BF16 or FP32 inference encoder for Wan 2.1 and Wan 2.2 checkpoints.

    Accept NCTHW pixels with T=4n+1 and return NCTHW posterior parameters
    (mean and log variance), without sampling or latent mean/std scaling.
    Each forward starts with fresh temporal history. Spatial tiling is not
    implemented. Construct from a loaded AutoencoderKLWan, then use eval()
    and torch.inference_mode().
    FP32 uses strict library convolution when cuDNN TF32 is disabled, otherwise
    ordinary TF32 custom convolution. Set the framework policy before construction
    and keep it stable during calls and CUDA graph replay. An explicit allow_tf32
    argument checks that the framework has the requested policy; it never mutates it.
    """

    def __init__(self, vae, *, allow_tf32: bool | None = None):
        super().__init__()
        if vae.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Custom Wan VAE kernels require BF16 or FP32")
        self.allow_tf32 = model_tf32_policy(vae.dtype, allow_tf32)
        if vae.config.is_residual:
            from .models.wan22 import Wan22Encoder

            self.impl = Wan22Encoder(vae)
        else:
            from .models.wan21 import Wan21Encoder

            self.impl = Wan21Encoder(vae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_model_policy(self.allow_tf32)
        return self.impl(x)

    def new_cache(self):
        return self.impl.new_cache()

    def encode_chunk(self, chunk, feat_cache):
        check_model_policy(self.allow_tf32)
        return self.impl.encode_chunk(chunk, feat_cache)
