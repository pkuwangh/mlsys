"""Eager/compiled Torch benchmark baselines; Diffusers is the independent oracle."""

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from mega_wan_vae.layers.torch import TorchWanVaeDecoder, TorchWanVaeEncoder


class TorchCompileChunkWanVaeEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.raw_encoder = TorchWanVaeEncoder(vae).eval()
        # Match eager/custom BF16 rounding between fused operators, rather than
        # silently evaluating those intermediates in FP32.
        self.chunk_encoder = torch.compile(
            self.raw_encoder.encoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.raw_encoder.encode_temporal_chunks(video, self.chunk_encoder)


class TorchCompileChunkWanVaeDecoder(torch.nn.Module):
    """Compile decoder chunks while preserving the eager reference's BF16 casts."""

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.raw_decoder = TorchWanVaeDecoder(vae).eval()
        self.chunk_decoder = torch.compile(
            self.raw_decoder.decoder,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the same temporal loop with compiled chunk execution."""
        return self.raw_decoder.decode_temporal_chunks(latent, self.chunk_decoder)
