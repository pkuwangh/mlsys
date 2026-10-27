"""Validate FP32 pointwise storage, cache/layout contracts and dtype separation.

Run on a CUDA device with the same DSL environment as check_dynamic_shapes.py.
BF16 arithmetic and geometry regressions remain covered by that suite.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mega_wan_vae.kernels import memory, norm, residual
from mega_wan_vae.kernels._compile import compile_stats
from mega_wan_vae.kernels._precision import pack_components


@torch.inference_mode()
def check_norm():
    for channels in (96, 160, 192, 256, 320, 384, 512, 640, 1024):
        # Offset views exercise storage alignment independently of dtype.
        x = torch.randn(2 * 3 * 3 * 5 * channels + 1, device="cuda")[1:].view(2, 3, 3, 5, channels)
        gamma = torch.randn(channels + 1, device="cuda")[1:]
        bias = torch.randn_like(gamma)
        for reduction in (True, False):
            for history in (0, 1, 2):
                previous = torch.randn(2, history, 3, 5, channels, device="cuda") if history else None
                for options in (
                    {},
                    {"input_bias": bias, "save_input": True},
                    {"input_bias": bias, "residual": x, "residual_bias": bias},
                ):
                    actual = norm.rmsnorm_silu_conv_prep(
                        x, gamma, previous=previous, contiguous_reduction=reduction, **options
                    )
                    expected = norm.torch_reference(x, gamma, previous=previous, **options)
                    for field in ("padded", "cache", "residual"):
                        value, reference = getattr(actual, field), getattr(expected, field)
                        if reference is None:
                            assert value is None
                        else:
                            assert value.dtype == torch.float32
                            torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-6)
        expected = F.normalize(x, dim=-1) * channels**0.5 * gamma + bias
        torch.testing.assert_close(norm.rmsnorm(x, gamma, bias), expected, atol=2e-6, rtol=2e-6)
        print(f"FP32 C{channels}: norm/preparation/history PASS", flush=True)


@torch.inference_mode()
def check_stage_and_residual():
    for ci, co, ft, fs in ((12, 24, 2, 2), (160, 160, 1, 1), (160, 320, 1, 2), (320, 640, 2, 2), (640, 640, 1, 1)):
        skip = torch.randn(2, 3, 8, 6, ci, device="cuda")
        x = torch.randn(2, (3 + ft - 1) // ft, 8 // fs, 6 // fs, co, device="cuda")
        bias = torch.randn(co, device="cuda")
        # Exact accumulation-order oracle plus independent Torch pooling.
        residual._verify_stage_ordered(x, skip, ft, fs, bias, x)
        torch.testing.assert_close(
            residual.stage_add(x, skip, ft, fs, bias, x),
            residual.torch_stage_add(x, skip, ft, fs, bias, x),
            atol=2e-6,
            rtol=2e-6,
        )
        for padded in (False, True):
            torch.testing.assert_close(
                residual.bias_residual(x, x, bias, residual_bias=bias, pad_spatial=padded),
                residual.torch_bias_residual(x, x, bias, residual_bias=bias, pad_spatial=padded),
                atol=0,
                rtol=0,
            )
    print("FP32 stage/residual arithmetic PASS", flush=True)


@torch.inference_mode()
def check_pack_graph_and_reuse():
    for dtype in (torch.bfloat16, torch.float32):
        for pad_channels in (False, True):
            count = None
            for n, t, h, w in ((1, 3, 3, 5), (2, 4, 5, 7)):
                x = torch.randn(n, 12, t + 2, h, w, device="cuda", dtype=dtype)[:, :, 1 : t + 1]
                previous = torch.randn(n, 2, h, w, 12, device="cuda", dtype=dtype)
                memory.pack_history(x, previous, input_ncthw=True, pad_input_channels=pad_channels)
                current = compile_stats().compiled
                if count is not None:
                    assert current == count, "Geometry-only change compiled a new packing kernel"
                count = current
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = memory.pack_history(x, previous, input_ncthw=True, pad_input_channels=pad_channels)
                x.normal_()
                previous.normal_()
                graph.replay()
                joined = torch.cat((previous, x.permute(0, 2, 3, 4, 1)), dim=1)
                torch.testing.assert_close(
                    actual.padded, F.pad(joined, (0, 4 if pad_channels else 0, 1, 1, 1, 1)), atol=0, rtol=0
                )
                torch.testing.assert_close(actual.cache, joined[:, -2:], atol=0, rtol=0)
                # Bias takes the generic packing path; preserve the same channel contract.
                bias = torch.randn(12, device="cuda", dtype=dtype)
                actual = memory.pack_history(x, previous, bias, input_ncthw=True, pad_input_channels=pad_channels)
                joined = torch.cat((previous, x.permute(0, 2, 3, 4, 1) + bias), dim=1)
                torch.testing.assert_close(
                    actual.padded, F.pad(joined, (0, 4 if pad_channels else 0, 1, 1, 1, 1)), atol=0, rtol=0
                )
                torch.testing.assert_close(actual.cache, joined[:, -2:], atol=0, rtol=0)
                count = compile_stats().compiled
        print(f"{dtype}: strided packing, graph replay and geometry reuse PASS", flush=True)


@torch.inference_mode()
def check_dtype_rejection():
    x = torch.randn(1, 3, 3, 5, 96, device="cuda")
    gamma = torch.ones(96, device="cuda", dtype=torch.bfloat16)
    for operation in (
        lambda: norm.rmsnorm(x, gamma),
        lambda: norm.rmsnorm_silu_conv_prep(x, gamma),
        lambda: memory.pack_history(x, bias=gamma),
        lambda: residual.bias_residual(x, x.bfloat16()),
        lambda: residual.stage_add(x, x.bfloat16(), 1, 1),
        lambda: memory.pack_history(x.half()),
    ):
        try:
            operation()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Unsupported or mixed storage dtypes were accepted")
    print("Mixed/unsupported storage rejection PASS", flush=True)


@torch.inference_mode()
def check_large_residuals():
    """Exercise launch counts and bounds on either side of 2**31 elements.

    Requires about 20 GiB of free VRAM. Check every output in batch-sized
    slices so validation does not need another full-size reference tensor.
    """
    for dtype in (torch.bfloat16, torch.float32):
        for batch in (14, 16):
            x = torch.empty((batch, 4, 480, 832, 96), device="cuda", dtype=dtype)
            channel = torch.arange(96, device="cuda", dtype=dtype) / 16
            bias = torch.full_like(channel, 0.5)
            for n in range(batch):
                x[n] = channel + n
            for padded in (False, True):
                out = residual.bias_residual(x, x, bias, residual_bias=bias, pad_spatial=padded)
                for n in range(batch):
                    expected = ((channel + n) + bias) * 2
                    assert (out[n, :, :480, :832] == expected).all().item()
                    if padded:
                        assert (out[n, :, 480] == 0).all().item()
                        assert (out[n, :, :, 832] == 0).all().item()
                del out
            # Both vector and scalar grouped-shortcut implementations.
            for channels in (96, 12):
                view = x.view(batch, 4, 480, -1, channels)
                out = residual.stage_add(view, view, 1, 1)
                for n in range(batch):
                    assert (out[n] == view[n] * 2).all().item()
                del out, view
            print(f"{dtype} N={batch}: large residual/padding/stage PASS", flush=True)
            del x


@torch.inference_mode()
def check_large_components():
    """Check every packed value around 2**31 elements, including a partial block."""
    for elements in (2**31 - 1024, 2**31 + 16):
        # A TF32 midpoint rounds to even; the compensated low component is exact.
        x = torch.full((elements // 16, 16), 1 + 2**-11, device="cuda")
        for compensated in (False, True):
            out = torch.full((x.shape[0], 32 if compensated else 16), float("nan"), device="cuda")
            pack_components(x, out, compensated=compensated)
            for rows in out.split(1 << 20):
                assert (rows[:, :16] == 1).all().item()
                if compensated:
                    assert (rows[:, 16:] == 2**-11).all().item()
            del out, rows
            print(f"{elements} elements, compensated={compensated}: TF32 packing PASS", flush=True)
        del x


@torch.inference_mode()
def check_large_history():
    """Check history, C12 input and initial-bias packing above 2**31 elements."""
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.empty((16, 4, 480, 832, 96), device="cuda", dtype=dtype)
        for n in range(16):
            x[n].fill_(n + 0.25)
        for channels in (96, 12):
            view = x.view(16, 4, 480, -1, channels)
            actual = memory.pack_history(
                view.permute(0, 4, 1, 2, 3) if channels == 12 else view,
                input_ncthw=channels == 12,
            )
            assert actual.padded.numel() > 2**31
            for n in range(16):
                out = actual.padded[n]
                assert (out[2:, 1:-1, 1:-1, :channels] == n + 0.25).all().item()
                assert (out[:2] == 0).all().item()
                assert (out[:, 0] == 0).all().item() and (out[:, -1] == 0).all().item()
                assert (out[:, :, 0] == 0).all().item() and (out[:, :, -1] == 0).all().item()
                if channels == 12:
                    assert (out[..., 12:] == 0).all().item()
                assert (actual.cache[n] == n + 0.25).all().item()
            del actual, out, view
            print(f"{dtype} C{channels}: large history packing PASS", flush=True)
        bias = torch.full((96,), 0.5, device="cuda", dtype=dtype)
        actual = memory.pack_history(x.view(16, 1, 1920, 832, 96), bias=bias, padding=(0, 0, 0))
        assert actual.padded.data_ptr() == actual.cache.data_ptr()
        for n in range(16):
            assert (actual.padded[n] == n + 0.75).all().item()
            assert (x[n] == n + 0.25).all().item()
        del actual, x
        print(f"{dtype}: large initial bias packing PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", help="Also check >2**31 elements (about 40 GiB VRAM).")
    args = parser.parse_args()
    torch.manual_seed(931)
    check_dtype_rejection()
    check_norm()
    check_stage_and_residual()
    check_pack_graph_and_reuse()
    if args.large:
        check_large_residuals()
        check_large_components()
        check_large_history()
