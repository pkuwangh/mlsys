"""Exercise optional residual output and spatial padding, including graph replay."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mega_wan_vae.kernels import conv
from mega_wan_vae.kernels.norm import rmsnorm_silu_conv_prep as norm


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["prepared", "spatial"], default="prepared")
    parser.add_argument("--channels", type=int, nargs="+", default=[96, 160, 192, 256, 320])
    parser.add_argument("--wide-index", action="store_true", help="Exercise 64-bit indexing with small allocations")
    args = parser.parse_args()
    if args.wide_index:
        with patch.object(conv, "_needs_wide_index", return_value=True):
            check(args)
    else:
        check(args)


def check(args):
    torch.manual_seed(84)
    for c in args.channels:
        executables = {}
        for n, t, h, w in ((1, 1, 3, 5), (2, 4, 11, 13)):
            torch.compiler.reset()
            cfg = conv.ConvConfig(n=n, t=t + 2, h=h + 2, w=w + 2, ci=c, co=c, ctas=4)
            x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
            weights = conv.WanConv3d(torch.randn(c, c, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1).weight
            bias = torch.randn(c, device="cuda", dtype=x.dtype) * 0.1
            gamma = torch.randn_like(bias)
            residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype)
            raw = conv.prepare(x, weights, cfg)
            assert executables.setdefault("raw", raw.compiled) is raw.compiled
            for history in (0, 1, 2) if args.operation == "prepared" else (0,):
                previous = torch.randn(n, history, h, w, c, device="cuda", dtype=x.dtype) if history else None
                for with_bias in (False, True):
                    rb = bias if with_bias else None
                    if args.operation == "prepared":
                        launch = conv.prepare_next(
                            x,
                            weights,
                            bias,
                            gamma,
                            cfg,
                            previous,
                            residual=residual,
                            residual_bias=rb,
                            save_residual=False,
                        )
                    else:
                        launch = conv.prepare_spatial_residual(x, weights, bias, residual, cfg, rb)
                    # N/T/H/W and history must not create new compilation keys.
                    assert executables.setdefault(with_bias, launch.compiled) is launch.compiled
                    for _ in range(3):
                        launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch()
                    x.normal_(0, 0.1)
                    residual.normal_()
                    if previous is not None:
                        previous.normal_()
                    if args.operation == "prepared":
                        launch.output.padded.fill_(float("nan"))
                        launch.output.cache.fill_(float("nan"))
                    else:
                        launch.output.fill_(float("nan"))
                    graph.replay()
                    if args.operation == "prepared":
                        expected = norm(
                            raw(), gamma, previous=previous, input_bias=bias, residual=residual, residual_bias=rb
                        )
                        assert launch.output.residual is None
                        torch.testing.assert_close(launch.output.padded, expected.padded, atol=0, rtol=0)
                        torch.testing.assert_close(launch.output.cache, expected.cache, atol=0, rtol=0)
                    else:
                        expected = torch.zeros_like(launch.output)
                        skip = residual if rb is None else residual + rb
                        expected[:, :, :-1, :-1] = (raw() + bias) + skip
                        torch.testing.assert_close(launch.output, expected, atol=0, rtol=0)
                    del graph, launch, expected
                    print(
                        f"{args.operation}: C{c} N{n} T{t} {h}x{w} history={history} residual_bias={with_bias}: bitwise PASS",
                        flush=True,
                    )
        print(f"C{c}: compiled executables reused across shapes and histories", flush=True)


if __name__ == "__main__":
    main()
