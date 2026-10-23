"""Check posterior parameters, decoded pixels and fresh-cache repeatability."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import AutoencoderKLWan
from mega_wan_vae import MegaWanVaeDecoder, MegaWanVaeEncoder


def check(label, actual, expected, atol, rtol):
    error = (actual.float() - expected.float()).abs()
    print(f"{label}: max_abs={error.max().item():.6g} mean_abs={error.mean().item():.6g}", flush=True)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    reference_padding = {
        name: tuple(module.padding) for name, module in vae.named_modules() if isinstance(module, torch.nn.ZeroPad2d)
    }
    encoder = MegaWanVaeEncoder(vae).eval()
    decoder = MegaWanVaeDecoder(vae).eval()
    assert reference_padding == {
        name: tuple(module.padding) for name, module in vae.named_modules() if isinstance(module, torch.nn.ZeroPad2d)
    }, "Constructing Mega must not change the reference VAE's padding"
    spatial = vae.spatial_compression_ratio
    for n, t, h, w in (
        (1, 1, 8 * spatial, 10 * spatial),
        (1, 9, 9 * spatial, 11 * spatial),
        (2, 5, 8 * spatial, 10 * spatial),
    ):
        video = torch.rand(n, 3, t, h, w, device="cuda", dtype=torch.bfloat16) * 2 - 1
        expected = vae.encode(video, return_dict=False)[0].parameters
        actual = encoder(video)
        check(f"Encode {tuple(video.shape)}", actual, expected, 0.25, 1e-3)
        # A different input in between must not contaminate the next invocation.
        encoder(-video)
        check("Encode fresh-cache repeat", encoder(video), actual, 0, 0)
        latent = expected[:, : vae.config.z_dim].contiguous()
        expected_video = vae.decode(latent, return_dict=False)[0]
        actual_video = decoder(latent)
        check(f"Decode {tuple(latent.shape)}", actual_video, expected_video, 0.05, 0.02)
        decoder(-latent)
        check("Decode fresh-cache repeat", decoder(latent), actual_video, 0, 0)
    print("Checkpoint encode/decode and cache isolation: PASS", flush=True)


if __name__ == "__main__":
    main()
