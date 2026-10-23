"""Binding, inference, output ownership and stream contracts across policies."""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mega_wan_vae.kernels import conv


def reject(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid binding accepted")


def main():
    torch.manual_seed(103)
    for precision in ("bf16", "ieee", "tf32", "tf32x3"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        x = torch.randn(1, 3, 5, 7, 96, dtype=dtype, device="cuda", requires_grad=True)
        weight = torch.randn(96, 96, 3, 3, 3, dtype=dtype, device="cuda") * 0.01
        with torch.no_grad():
            module = conv.WanConv3d(weight, precision=precision).eval()
        module.ctas = 4
        cfg = conv.ConvConfig(*x.shape, co=96, ctas=4, precision=precision)
        reject(partial(conv.prepare, x, module.weight, cfg))
        with torch.no_grad():
            bound = conv.prepare(x, module.weight, cfg)
        first = bound()
        assert not first.requires_grad, "Bound inference accidentally enabled autograd"
        snapshot = first.clone()
        second = bound()
        assert (first.data_ptr() == second.data_ptr()) is (precision != "ieee")
        torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
        with torch.inference_mode():
            first = module(x)
            second = module(x)
            assert first.data_ptr() != second.data_ptr(), "Module calls must return fresh output"
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            bias = torch.zeros(96, device="cuda", dtype=dtype)
            gamma = torch.ones_like(bias)
            for previous in (
                torch.zeros(1, 3, 3, 5, 96, device="cuda", dtype=dtype),
                torch.zeros(1, 1, 3, 5, 96, device="cpu", dtype=dtype),
                torch.zeros(1, 1, 3, 5, 96, device="cuda", dtype=torch.float64),
                torch.zeros(1, 1, 3, 5, 192, device="cuda", dtype=dtype)[..., ::2],
            ):
                reject(partial(conv.prepare_next, x, module.weight, bias, gamma, cfg, previous))
            reject(partial(conv.prepare, x.cpu(), module.weight, cfg))
            reject(partial(conv.prepare, x, module.weight.double(), cfg))
            reject(partial(conv.prepare_next, x, module.weight, bias.double(), gamma, cfg))
            storage = torch.empty(x.numel() + 1, device="cuda", dtype=dtype)
            reject(partial(conv.prepare, storage[1:].view(x.shape), module.weight, cfg))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = module(x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(actual, first, atol=0, rtol=0)
        print(f"{precision}: inference, validation, ownership and nondefault stream PASS", flush=True)


if __name__ == "__main__":
    main()
