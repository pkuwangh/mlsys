"""Small sanitizer workload: persistent convolution, fused outputs, history and graph replay.

Run under compute-sanitizer --tool memcheck; add --check-tensor-ops yes to
exercise TMA barrier instrumentation. No cuDNN reference kernels are launched.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mega_wan_vae.kernels import conv, norm


@torch.inference_mode()
def check(precision):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    for channels in (96, 320):
        weight = torch.full((channels, channels, 3, 3, 3), 0.25, device="cuda", dtype=dtype)
        module = conv.WanConv3d(weight, precision=precision).eval()
        module.ctas = 4
        x = torch.full((2, 7, 11, 13, channels), 0.125, device="cuda", dtype=dtype)
        cfg = conv.ConvConfig(*x.shape, co=channels, ctas=4, precision=precision)
        # Exactly representable products/sums isolate indexing and synchronization.
        expected = torch.full(cfg.output_shape, channels * 27 * 0.03125, device="cuda", dtype=dtype)
        torch.testing.assert_close(module(x), expected, atol=0, rtol=0)
        bias = torch.full((channels,), 0.25, device="cuda", dtype=dtype)
        gamma = torch.ones_like(bias)
        residual = torch.full_like(expected, 0.5)
        for frames in (0, 2):
            previous = torch.full((2, frames, 9, 11, channels), 0.75, device="cuda", dtype=dtype) if frames else None
            actual = conv.prepare_next(
                x, module.weight, bias, gamma, cfg, previous, residual=residual, residual_bias=bias
            )()
            reference = norm.torch_reference(
                expected, gamma, previous=previous, input_bias=bias, residual=residual, residual_bias=bias
            )
            for name in ("padded", "cache", "residual"):
                torch.testing.assert_close(getattr(actual, name), getattr(reference, name), atol=2e-6, rtol=2e-6)
        graph = torch.cuda.CUDAGraph()
        raw = conv.prepare(x, module.weight, cfg)
        raw()
        with torch.cuda.graph(graph):
            raw()
        x.fill_(0.25)
        graph.replay()
        torch.testing.assert_close(raw.output, expected * 2, atol=0, rtol=0)
        torch.cuda.synchronize()
        print(f"{precision} C{channels}: persistent raw/fused/cache/graph memory checks PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("bf16", "tf32"), default="tf32")
    check(parser.parse_args().precision)
