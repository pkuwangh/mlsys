"""Independent arithmetic/layout checks for the public convolution policies.

Run on Blackwell with the repository GPU runner. The TF32 oracle rounds operands
with integer Torch operations, then performs TF32-disabled cuDNN convolution.
It does not call the production component packer to construct expected values.
"""

import argparse
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mega_wan_vae.kernels import conv, memory, norm, residual
from mega_wan_vae.kernels._compile import compile_stats
from mega_wan_vae.kernels._precision import pack_components
from mega_wan_vae.kernels._utils import round_tf32_reference as round_tf32
from mega_wan_vae.layers.fused import _conv_raw


def reference(x, weight, policy):
    if policy == "tf32":
        x, weight = round_tf32(x), round_tf32(weight)
    return F.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1).contiguous()


def close(actual, expected, policy):
    atol, rtol = (2e-3, 2e-2) if policy == "bf16" else (1e-4, 1e-4)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def check_spatial(actual, module, x, expected_raw, skip, bias, residual_bias, policy):
    if policy == "bf16":
        # Validate convolution separately, then require exact BF16 post-op rounding.
        # A cuDNN/custom raw rounding difference can be magnified by cancellation
        # in the residual sum; the spatial epilogue must preserve its own raw result.
        raw = module(x)
        close(raw, expected_raw, policy)
        expected = residual.torch_bias_residual(raw, skip, bias, residual_bias=residual_bias, pad_spatial=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        expected = residual.torch_bias_residual(expected_raw, skip, bias, residual_bias=residual_bias, pad_spatial=True)
        close(actual, expected, policy)


def must_reject(fn):
    try:
        fn()
    except (TypeError, ValueError):
        return
    raise AssertionError("Invalid operands or arithmetic policy were accepted")


@torch.inference_mode()
def check_rounding():
    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + 2**-11,
        1.0 + 3 * 2**-11,
        float("inf"),
        -float("inf"),
        float("nan"),
        torch.finfo(torch.float32).max,
        -torch.finfo(torch.float32).max,
        2**-140,
        -(2**-140),
        2**-126,
        3.0,
        -3.0,
    ]
    x = torch.tensor(values, device="cuda")
    actual = torch.empty_like(x)
    pack_components(x, actual, compensated=False)
    torch.testing.assert_close(actual, round_tf32(x), atol=0, rtol=0, equal_nan=True)
    assert torch.equal(actual.view(torch.int32), round_tf32(x).view(torch.int32))
    print("TF32 independent rounding, ties, signed zero, overflow and nonfinite packing PASS", flush=True)


@torch.inference_mode()
def check_mode(policy, *, raw_only=False):
    dtype = torch.bfloat16 if policy == "bf16" else torch.float32
    pairs = conv._ENCODER_CHANNEL_PAIRS + conv._DECODER_CHANNEL_PAIRS + conv._WAN21_CHANNEL_PAIRS
    if policy != "bf16":
        pairs = ((16, 160),) + pairs
    for ci, co in pairs:
        weight = torch.randn(co, ci, 3, 3, 3, device="cuda", dtype=dtype) * 0.1
        module = conv.WanConv3d(weight, precision=policy).eval()
        module.ctas = 4
        for wide in (False, True):
            with patch.object(conv, "_needs_wide_index", return_value=wide):
                count = None
                for shape in ((1, 3, 5, 7, ci), (2, 5, 7, 9, ci)):
                    x = torch.randn(shape, device="cuda", dtype=dtype) * 0.1
                    expected = reference(x, weight, policy)
                    close(module(x), expected, policy)
                    if count is not None:
                        assert compile_stats().compiled == count, "Geometry recompiled a convolution"
                    count = compile_stats().compiled
        cfg = conv.ConvConfig(*x.shape, co=co, ctas=4, precision=policy)
        bound_raw = conv.prepare(x, module.weight, cfg)
        bound_raw()
        raw_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(raw_graph):
            actual_raw = bound_raw()
        x.normal_(0, 0.1)
        raw_graph.replay()
        expected = reference(x, weight, policy)
        close(actual_raw, expected, policy)
        del raw_graph
        # Custom-op dispatch must retain packed-weight policy when the global flag changes.
        torch.backends.cudnn.allow_tf32 = True
        try:
            close(_conv_raw(x, module.weight, 4, policy), expected, policy)
        finally:
            torch.backends.cudnn.allow_tf32 = False
        if not raw_only:
            bias, gamma, rb = (torch.randn(co, device="cuda", dtype=dtype) * 0.1 for _ in range(3))
            skip = torch.randn_like(expected) * 0.1
            spatial = conv.prepare_spatial_residual(x, module.weight, bias, skip, cfg, rb)
            spatial()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual_spatial = spatial()
            x.normal_(0, 0.1)
            skip.normal_(0, 0.1)
            graph.replay()
            expected = reference(x, weight, policy)
            check_spatial(actual_spatial, module, x, expected, skip, bias, rb, policy)
            del graph
        if raw_only or co not in (96, 160, 192, 256, 320):
            print(
                f"{policy} {ci}->{co}: raw, both index widths, reuse, dispatch, graph"
                f"{'/spatial' if not raw_only else ''} PASS",
                flush=True,
            )
            continue
        for history in (0, 1, 2):
            previous = torch.randn(2, history, 5, 7, co, device="cuda", dtype=dtype) if history else None
            for use_residual in (False, True):
                for save_residual in (False, True):
                    options = {"residual": skip, "residual_bias": rb} if use_residual else {}
                    bound = conv.prepare_next(
                        x, module.weight, bias, gamma, cfg, previous, save_residual=save_residual, **options
                    )
                    actual = bound()
                    target = norm.torch_reference(expected, gamma, previous=previous, input_bias=bias, **options)
                    close(actual.padded, target.padded, policy)
                    close(actual.cache, target.cache, policy)
                    if use_residual and save_residual:
                        close(actual.residual, target.residual, policy)
                    else:
                        assert actual.residual is None
            # Replay must observe updated source and history, not stale packed activations.
            options = {"residual": skip, "residual_bias": rb}
            bound = conv.prepare_next(x, module.weight, bias, gamma, cfg, previous, save_residual=True, **options)
            bound()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = bound()
            x.normal_(0, 0.1)
            if previous is not None:
                previous.normal_(0, 0.1)
            graph.replay()
            expected = reference(x, weight, policy)
            target = norm.torch_reference(expected, gamma, previous=previous, input_bias=bias, **options)
            for field in ("padded", "cache", "residual"):
                close(getattr(actual, field), getattr(target, field), policy)
            del graph
        spatial = conv.prepare_spatial_residual(x, module.weight, bias, skip, cfg, rb)()
        check_spatial(spatial, module, x, expected, skip, bias, rb, policy)
        bad_history = torch.empty((2, 3, 5, 7, co), device="cuda", dtype=dtype)
        must_reject(partial(conv.prepare_next, x, module.weight, bias, gamma, cfg, bad_history))
        must_reject(partial(conv.prepare, x, module.weight.double(), cfg))
        if policy != "bf16":
            must_reject(partial(conv.prepare, x, module.weight, replace(cfg, precision="bf16")))
        print(f"{policy} {ci}->{co}: raw/prepared/residual/cache/spatial/graph/contracts PASS", flush=True)


@torch.inference_mode()
def check_input_convolution(policy):
    """C12 packing and native-library input must preserve causal history and policy."""
    dtype = torch.bfloat16 if policy == "bf16" else torch.float32
    weight = torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=dtype) * 0.02
    module = conv.WanInputConv3d(weight, precision=policy).eval()
    assert module.input_channels == (12 if policy == "ieee" else 16)
    for frames, history in ((1, 0), (1, 1), (4, 2)):
        source = torch.randn(2, 12, frames + 2, 7, 9, device="cuda", dtype=dtype) * 0.1
        x = source[:, :, 1 : frames + 1]
        previous = torch.randn(2, history, 7, 9, 12, device="cuda", dtype=dtype) * 0.1 if history else None

        def run(x=x, previous=previous):
            packed = memory.pack_history(x, previous, input_ncthw=True, pad_input_channels=module.input_channels == 16)
            return module(packed.padded), packed.cache

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, cache = run()
        x.normal_(0, 0.1)
        if previous is not None:
            previous.normal_(0, 0.1)
        graph.replay()
        joined = x.permute(0, 2, 3, 4, 1)
        if previous is not None:
            joined = torch.cat((previous, joined), dim=1)
        padded = F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0))
        # Strict library policy remains pinned even if the framework permits TF32.
        with torch.backends.cudnn.flags(allow_tf32=True):
            policy_output, _ = run()
        expected = reference(padded, weight, policy)
        close(actual, expected, policy)
        close(policy_output, expected, policy)
        torch.testing.assert_close(cache, joined[:, -2:], atol=0, rtol=0)
    print(f"{policy}: C12 input, native channel count, history, policy and graph replay PASS", flush=True)


def check_compile_rejection():
    cfg = conv.ConvConfig(precision="ieee")
    must_reject(lambda: conv.compile_conv(cfg))
    must_reject(lambda: conv.compile_prepared(cfg))
    must_reject(lambda: conv.compile_spatial_residual(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes", nargs="*", default=["ieee", "tf32", "bf16"])
    parser.add_argument("--wide-index", action="store_true", help="Force wide indexing for post-ops and graph tests")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(101)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    check_compile_rejection()
    check_rounding()
    with patch.object(conv, "_needs_wide_index", return_value=True) if args.wide_index else nullcontext():
        for mode in args.modes:
            check_mode(mode, raw_only=args.raw_only)
            check_input_convolution(mode)
