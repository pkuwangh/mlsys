"""Causal convolution boundaries, fused BF16 operations, and changed-input graph replay."""

import pytest
import torch
from mega_qwen3_tts.kernels.conv1d import _compile, conv1d
from mega_qwen3_tts.kernels.snake_beta import SnakeBeta
from torch.nn import functional as F
from torch_qwen3_tts.vocoder import ChannelsLastConv1d
from torch_qwen3_tts.vocoder import SnakeBeta as TorchSnakeBeta

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)),
    reason="Hopper or datacenter Blackwell GPU required",
)


@torch.inference_mode()
def test_bias_rounding():
    """cuDNN's BF16 convolution output is rounded before the separate bias add."""
    # This pins Torch's current separate bias add. If an upgrade fuses it,
    # revisit the reference contract rather than loosening this exact check.
    x = torch.zeros(1, 1, 32, device="cuda", dtype=torch.bfloat16)
    x[..., :2] = 1
    weight = torch.zeros(32, 1, 32, device="cuda", dtype=x.dtype)
    weight[:, 0, 0] = 1
    weight[:, 0, 1] = 1 / 256
    bias = torch.full((32,), 1 / 256, device=x.device, dtype=x.dtype)

    def reference(x, weight, bias):
        return F.conv1d(x.transpose(1, 2), weight.permute(0, 2, 1), bias).transpose(1, 2)

    expected = torch.ones_like(x)
    torch.testing.assert_close(reference(x, weight, bias), expected, atol=0, rtol=0)
    conv = torch.nn.Conv1d(32, 32, 1, device=x.device, dtype=x.dtype)
    conv.weight.copy_(weight.permute(0, 2, 1))
    conv.bias.copy_(bias)
    torch.testing.assert_close(ChannelsLastConv1d(conv)(x), expected, atol=0, rtol=0)
    compiled = torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True})
    torch.testing.assert_close(compiled(x, weight, bias), expected, atol=0, rtol=0)
    torch.testing.assert_close(conv1d(x, weight, bias), expected, atol=0, rtol=0)
    single_round = reference(x.float(), weight.float(), bias.float()).to(x.dtype)
    assert torch.all(single_round == 1 + 1 / 128)


@torch.inference_mode()
def test_dynamic_batch_and_length():
    """Reuse full/partial-tile binaries across lengths, batch sizes and launch grids."""
    torch.manual_seed(42)
    weight = torch.randn(96, 7, 96, device="cuda", dtype=torch.bfloat16) / (7 * 96) ** 0.5
    _compile.cache_clear()
    for batch, length in ((1, 1), (2, 65), (3, 257), (1, 33), (2, 128), (1, 384)):
        x = torch.randn(batch, length, 96, device="cuda", dtype=weight.dtype)
        actual = conv1d(x, weight, dilation=3, tile_n=96, stages=4)
        expected = F.conv1d(
            F.pad(x.transpose(1, 2).contiguous(), (18, 0)),
            weight.permute(0, 2, 1).contiguous(),
            dilation=3,
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
    assert _compile.cache_info().misses == 2


@pytest.mark.parametrize(
    "batch,length,ci,co,kernel,dilation,tile_n,stages",
    [
        (1, 1, 96, 96, 7, 9, None, None),
        (2, 257, 96, 96, 7, 3, None, None),
        (2, 129, 192, 192, 1, 1, 96, None),
        (2, 259, 384, 384, 1, 1, None, None),
        (1, 127, 384, 384, 7, 1, 192, 4),
        (1, 33, 768, 768, 7, 9, None, None),
        (2, 17, 1024, 1536, 7, 1, None, None),
        (1, 65537, 192, 192, 1, 1, None, None),
        # One K loop exactly fills the operand ring, across persistent tile turns.
        (1, 8193, 384, 384, 1, 1, 64, 6),
    ],
)
@torch.inference_mode()
def test_conv1d_replay(batch, length, ci, co, kernel, dilation, tile_n, stages):
    """Replay new activations/weights; check causal tails and exact fused post-ops."""
    torch.manual_seed(42)
    x = torch.randn(batch, length, ci, device="cuda", dtype=torch.bfloat16)
    weight = x.new_empty(co, kernel, ci).normal_(std=(kernel * ci) ** -0.5)
    bias = None if length == 1 else x.new_empty(co).normal_()
    residual = x.new_empty(batch, length, co).normal_()
    reference_activation = TorchSnakeBeta(co).cuda().bfloat16()
    reference_activation.alpha.normal_(std=0.25)
    reference_activation.beta.normal_(std=0.25)
    activation = SnakeBeta(reference_activation.alpha, reference_activation.beta)

    def run():
        options = {"dilation": dilation, "tile_n": tile_n, "stages": stages}
        raw = conv1d(x, weight, bias, **options)
        fused = conv1d(x, weight, bias, residual=residual, activation=activation, **options)
        return raw, fused

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw, fused = run()
    for _ in range(3):
        x.normal_()
        weight.normal_(std=(kernel * ci) ** -0.5)
        residual.normal_()
        graph.replay()
        expected = F.conv1d(
            F.pad(x.transpose(1, 2).contiguous(), ((kernel - 1) * dilation, 0)),
            weight.permute(0, 2, 1).contiguous(),
            bias,
            dilation=dilation,
        ).transpose(1, 2)
        torch.testing.assert_close(raw, expected, atol=0.03125, rtol=0.02)
        expected_fused = reference_activation((raw + residual).transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(fused, expected_fused, atol=0, rtol=0)
