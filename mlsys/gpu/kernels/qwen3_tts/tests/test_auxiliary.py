"""Run the standalone auxiliary-kernel numerical checks under pytest."""

import pytest
import torch
from mega_qwen3_tts.kernels import argmax_embedding, embedding_update, sampling, snake_beta
from mega_qwen3_tts.predictor import MegaQwen3TTSPredictor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)),
    reason="Hopper or datacenter Blackwell GPU required",
)


@pytest.mark.parametrize(
    "kernel", [argmax_embedding, embedding_update, sampling, snake_beta], ids=lambda k: k.__name__.split(".")[-1]
)
def test_validation(kernel):
    kernel.validate()


@torch.inference_mode()
def test_predictor_selection():
    """Check the predictor's code/embedding shapes and values on identical logits."""
    torch.manual_seed(42)
    logits = torch.randn(7, 1, 2048, device="cuda", dtype=torch.bfloat16)
    table = torch.randn(2048, 2048, device="cuda", dtype=logits.dtype)
    codes, embeddings = MegaQwen3TTSPredictor.select_embedding(logits, table)
    expected = logits.argmax(-1)
    torch.testing.assert_close(codes, expected, atol=0, rtol=0)
    torch.testing.assert_close(embeddings, table[expected], atol=0, rtol=0)
