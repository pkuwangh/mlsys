"""Experimental TF32x3 packing, convolution and fused outputs against strict FP32."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mega_wan_vae.kernels import conv, norm, residual
from mega_wan_vae.kernels._compile import compile_stats
from mega_wan_vae.kernels._precision import pack_components


@torch.inference_mode()
def check_components():
    # Tie-to-even and finite-overflow cases have explicit expected high parts.
    values = [1.0, 1.0 + 2**-11, 1.0 + 3 * 2**-11, -1.0 - 2**-11]
    values += [0.0, -0.0, torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]
    x = torch.tensor(values * 2, device="cuda")
    packed = torch.empty(32, device="cuda")
    pack_components(x, packed)
    torch.testing.assert_close(packed[:4], torch.tensor([1, 1, 1 + 2**-9, -1], device="cuda"), atol=0, rtol=0)
    assert torch.isfinite(packed).all()
    reconstructed = packed[:16].double() + packed[16:].double()
    torch.testing.assert_close(reconstructed, x.double(), atol=0, rtol=2**-21)
    assert not (packed.view(torch.int32) & 8191).any()
    print("Component rounding/ties/finite overflow PASS", flush=True)


def reference(x, weight):
    return F.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1).contiguous()


@torch.inference_mode()
def check_convolutions():
    pairs = ((16, 160),) + conv._ENCODER_CHANNEL_PAIRS + conv._DECODER_CHANNEL_PAIRS + conv._WAN21_CHANNEL_PAIRS
    for ci, co in pairs:
        weight = torch.randn(co, ci, 3, 3, 3, device="cuda") * 0.1
        module = conv.WanConv3d(weight, precision="tf32x3").eval()
        module.ctas = 4
        previous_count = None
        for n, t, h, w in ((1, 3, 5, 7), (2, 5, 7, 9)):
            x = torch.randn(n, t, h, w, ci, device="cuda") * 0.1
            actual = module(x)
            expected = reference(x, weight)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            count = compile_stats().compiled
            if previous_count is not None:
                assert count == previous_count, "Runtime geometry caused a new compilation"
            previous_count = count
        print(f"TF32x3 {ci}->{co}: raw and geometry reuse PASS", flush=True)
        if ci != co or co not in (96, 160, 192, 256, 320):
            continue
        bias, gamma, skip_bias = (torch.randn(co, device="cuda") * 0.1 for _ in range(3))
        skip = torch.randn_like(expected) * 0.1
        config = conv.ConvConfig(*x.shape, co=co, ctas=4, precision="tf32x3")
        for history in (0, 1, 2):
            previous = torch.randn(n, history, h - 2, w - 2, co, device="cuda") if history else None
            for use_residual in (False, True):
                options = {"residual": skip, "residual_bias": skip_bias} if use_residual else {}
                fused = conv.prepare_next(x, module.weight, bias, gamma, config, previous, **options)
                actual = fused()
                target = norm.torch_reference(expected, gamma, previous=previous, input_bias=bias, **options)
                for field in ("padded", "cache", "residual"):
                    value, wanted = getattr(actual, field), getattr(target, field)
                    if wanted is not None:
                        torch.testing.assert_close(value, wanted, atol=1e-4, rtol=1e-4)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = fused()
                x.normal_(0, 0.1)
                graph.replay()
                expected = reference(x, weight)
                target = norm.torch_reference(expected, gamma, previous=previous, input_bias=bias, **options)
                torch.testing.assert_close(actual.padded, target.padded, atol=1e-4, rtol=1e-4)
                del graph
        actual = conv.prepare_spatial_residual(x, module.weight, bias, skip, config, skip_bias)()
        target = residual.torch_bias_residual(expected, skip, bias, residual_bias=skip_bias, pad_spatial=True)
        torch.testing.assert_close(actual, target, atol=1e-4, rtol=1e-4)
        print(f"TF32x3 C{co}: fused/cache/graph/spatial residual PASS", flush=True)


if __name__ == "__main__":
    torch.manual_seed(101)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    check_components()
    check_convolutions()
