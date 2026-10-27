"""A small waveform tail exercises the custom block wiring without a checkpoint."""

import pytest
import torch
from mega_qwen3_tts.vocoder import MegaVocoderTail
from torch_qwen3_tts.vocoder import TorchVocoderTail

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)),
    reason="Hopper or datacenter Blackwell GPU required",
)


@torch.inference_mode()
def test_vocoder_tail():
    """Exercise upsampling, fused residual units and the final one-channel fallback."""
    torch.manual_seed(42)
    reference = TorchVocoderTail(latent_dim=32, decoder_dim=128, rates=(2, 2)).cuda().bfloat16().eval()
    custom = MegaVocoderTail(reference)
    baseline = MegaVocoderTail(reference, custom=False)
    for length in (7, 17):
        x = torch.randn(2, 32, length, device="cuda", dtype=torch.bfloat16)
        expected = reference(x)
        torch.testing.assert_close(baseline(x), expected, atol=0.03125, rtol=0.02)
        torch.testing.assert_close(custom(x), expected, atol=0.03125, rtol=0.02)
