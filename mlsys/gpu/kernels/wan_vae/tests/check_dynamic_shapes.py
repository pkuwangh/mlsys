"""Check runtime geometry reuse, primitive values/cache writes and CUDA graph replay."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mega_wan_vae.kernels import conv, memory, norm, residual
from mega_wan_vae.kernels._compile import compile_family_stats, compile_stats


@torch.inference_mode()
def check_precompile_reuse():
    for dtype in (torch.bfloat16, torch.float32):
        norm.compile(dtype)
        memory.compile(dtype)
        residual.compile(dtype)
        before = compile_stats().compiled
        x = torch.randn(2, 1, 4, 6, 640, device="cuda", dtype=dtype)
        bias = torch.randn(640, device="cuda", dtype=dtype)
        norm.rmsnorm(x, bias)
        norm.rmsnorm_silu_conv_prep(x, bias, input_bias=bias, residual=x, save_input=True)
        memory.pack_history(x, bias=bias, padding=(0, 0, 0))
        for pad in (False, True):
            residual.bias_residual(x, x, bias, pad_spatial=pad)
        residual.stage_add(x, x, 1, 1, bias, x)
        for channels, vector in ((12, False), (48, False), (48, True)):
            source = torch.randn(2, channels, 4, 6, 8, device="cuda", dtype=dtype)
            if vector:
                source = source.contiguous(memory_format=torch.channels_last_3d)
            memory.pack_history(source, input_ncthw=True)
        torch.cuda.synchronize()
        assert compile_stats().compiled == before, "Precompiled variants missed the runtime executable cache"
        print(f"{dtype}: precompile -> runtime reuse PASS", flush=True)


@torch.inference_mode()
def check_primitives():
    torch.manual_seed(51)
    for c in (96, 160, 192, 256, 320):
        for wide in (False, True):
            before = None
            for n, t, h, w in ((1, 1, 3, 5), (2, 4, 8, 10), (3, 2, 7, 9)):
                x = torch.randn(n, t, h, w, c, device="cuda", dtype=torch.bfloat16)
                gamma, bias = (torch.randn(c, device="cuda", dtype=x.dtype) for _ in range(2))
                skip = torch.randn_like(x)
                # Force the same width contract at small sizes, including the
                # wide fallback without multi-gigabyte test allocations.
                compile_norm = norm.compile_conv_prep
                compile_pack = memory.compile_pack
                compile_residual = residual.compile_bias_residual

                def norm_variant(*args, wide=wide, compile_norm=compile_norm, **kwargs):
                    kwargs["wide_index"] = wide
                    return compile_norm(*args, **kwargs)

                def pack_variant(config, wide=wide, compile_pack=compile_pack, **kwargs):
                    from dataclasses import replace

                    return compile_pack(replace(config, wide_index=wide), **kwargs)

                def residual_variant(*args, wide=wide, compile_residual=compile_residual, **kwargs):
                    kwargs["wide_index"] = wide
                    return compile_residual(*args, **kwargs)

                with (
                    patch.object(norm, "compile_conv_prep", norm_variant),
                    patch.object(memory, "compile_pack", pack_variant),
                    patch.object(residual, "compile_bias_residual", residual_variant),
                ):
                    for history in (0, 1, 2):
                        previous = torch.randn(n, history, h, w, c, device="cuda", dtype=x.dtype) if history else None
                        kwargs = {"input_bias": bias, "residual": skip, "residual_bias": bias}
                        actual = norm.rmsnorm_silu_conv_prep(x, gamma, previous=previous, **kwargs)
                        expected = norm.torch_reference(x, gamma, previous=previous, **kwargs)
                        torch.testing.assert_close(actual.padded, expected.padded, atol=1e-3, rtol=1e-2)
                        torch.testing.assert_close(actual.cache, expected.cache, atol=1e-3, rtol=1e-2)
                        torch.testing.assert_close(actual.residual, expected.residual, atol=0, rtol=0)
                        packed = memory.pack_history(x, previous, bias)
                        joined = x + bias if previous is None else torch.cat((previous, x + bias), dim=1)
                        torch.testing.assert_close(
                            packed.padded, F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0)), atol=0, rtol=0
                        )
                        torch.testing.assert_close(packed.cache, joined[:, -2:], atol=0, rtol=0)
                    for pad in (False, True):
                        actual = residual.bias_residual(x, skip, bias, residual_bias=bias, pad_spatial=pad)
                        expected = residual.torch_bias_residual(x, skip, bias, residual_bias=bias, pad_spatial=pad)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    # Capture each geometry after compilation, then change data.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = norm.rmsnorm_silu_conv_prep(x, gamma, **kwargs)
                    x.normal_()
                    graph.replay()
                    expected = norm.torch_reference(x, gamma, **kwargs)
                    torch.testing.assert_close(captured.padded, expected.padded, atol=1e-3, rtol=1e-2)
                    torch.testing.assert_close(captured.cache, expected.cache, atol=1e-3, rtol=1e-2)
                    del graph, captured
                count = compile_stats().compiled
                if before is not None:
                    assert count == before, (c, wide, (n, t, h, w), before, count)
                before = count
            print(f"C{c} wide={wide}: values/cache/graph/reuse PASS", flush=True)


@torch.inference_mode()
def check_stage():
    from dataclasses import replace

    torch.manual_seed(52)
    compile_stage = residual.compile_stage
    for wide in (False, True):

        def variant(config, wide=wide, **kwargs):
            return compile_stage(replace(config, wide_index=wide), **kwargs)

        with patch.object(residual, "compile_stage", variant):
            for ci, co, ft, fs in (
                (160, 160, 1, 2),
                (160, 320, 2, 2),
                (320, 640, 2, 2),
                (640, 640, 1, 1),
                (12, 24, 2, 2),
            ):
                before = None
                for n, t, h, w in ((1, 1, 6, 8), (2, 4, 10, 12), (3, 3, 8, 6)):
                    skip = torch.randn(n, t, h, w, ci, device="cuda", dtype=torch.bfloat16)
                    x = torch.randn(n, (t + ft - 1) // ft, h // fs, w // fs, co, device="cuda", dtype=skip.dtype)
                    bias = torch.randn(co, device="cuda", dtype=x.dtype)
                    r = torch.randn_like(x)
                    actual = residual.stage_add(x, skip, ft, fs, bias, r)
                    expected = residual.torch_stage_add(x, skip, ft, fs, bias, r)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    count = compile_stats().compiled
                    if before is not None:
                        assert before == count
                    before = count
        print(f"Stage shortcut wide={wide}: values/reuse PASS", flush=True)


@torch.inference_mode()
def check_input():
    torch.manual_seed(53)
    weight = torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    packed_weight = conv.pack_weight(weight)
    executable = None
    before = None
    for n, t, h, w, video_t in ((1, 1, 5, 7, 17), (2, 4, 8, 10, 93), (1, 3, 11, 9, 201)):
        source = torch.randn(n, 12, video_t, h, w, device="cuda", dtype=weight.dtype) * 0.1
        for history in (0, 1, 2):
            x = source[:, :, 2 : 2 + t]
            previous = torch.randn(n, history, h, w, 12, device="cuda", dtype=x.dtype) * 0.1 if history else None
            packed = memory.pack_history(x, previous, input_ncthw=True)
            joined = (
                x.permute(0, 2, 3, 4, 1) if previous is None else torch.cat((previous, x.permute(0, 2, 3, 4, 1)), dim=1)
            )
            expected = F.pad(joined, (0, 4, 1, 1, 1, 1, 2 - history, 0))
            torch.testing.assert_close(packed.padded, expected, atol=0, rtol=0)
            torch.testing.assert_close(packed.cache, joined[:, -2:], atol=0, rtol=0)
            cfg = conv.InputConvConfig(n, t + 2, h + 2, w + 2, ctas=4)
            launch = conv.prepare_input(packed.padded, packed_weight, cfg)
            if executable is not None:
                assert launch.compiled is executable
            executable = launch.compiled
            actual = launch()
            reference = (
                F.conv3d(expected[..., :12].permute(0, 4, 1, 2, 3).float(), weight.float())
                .to(x.dtype)
                .permute(0, 2, 3, 4, 1)
            )
            torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-2)
        count = compile_stats().compiled
        if before is not None:
            assert before == count, (before, count)
        before = count
        print(f"C12 N{n} T{t} {h}x{w} sourceT{video_t}: values/cache/reuse PASS", flush=True)


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = False
    check_precompile_reuse()
    check_primitives()
    check_stage()
    check_input()
    for name, stats in sorted(compile_family_stats().items()):
        print(name, stats, flush=True)
