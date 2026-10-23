"""CPU index-width boundaries; optional GPU large-offset arithmetic without large allocations."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mega_wan_vae.kernels._conv_config import ConvConfig, LaunchPolicy, needs_wide_index, specialization


def check_boundaries():
    for channels in (96, 160, 192, 256, 320, 384, 640):
        # Minimal logical output: one pixel. The padded-vector bound reaches
        # INT32_MAX first, at N*27*(C/8), without allocating any tensors.
        cutoff = (2**31 + 27 * (channels // 8) - 1) // (27 * (channels // 8))
        small = ConvConfig(n=cutoff - 1, t=3, h=3, w=3, ci=channels, co=channels)
        large = replace(small, n=cutoff)
        assert not needs_wide_index(small)
        assert needs_wide_index(large)
        assert specialization(small) == specialization(large)
        # A >4 GiB BF16 allocation still uses narrow coordinates when all
        # vector indices fit; byte/element offsets must remain 64-bit.
        assert (cutoff - 1) * 27 * channels * 2 > 2**32
    # Spatial, temporal and batch extents all participate in the bound.
    for field in ("n", "t", "h", "w"):
        assert needs_wide_index(replace(ConvConfig(ci=192, co=192), **{field: 2**31}))
    for channels in (96, 160, 192, 256, 320):
        variant = specialization(ConvConfig(ci=channels, co=channels))
        for residual in (False, True):
            policy = LaunchPolicy.for_conv(variant, fuse_norm=True, has_residual=residual)
            expected = 704 if channels == 320 else 640 if residual or channels in (192, 256) else 384
            assert policy.threads == expected
        assert LaunchPolicy.for_conv(variant).threads == 192
        assert LaunchPolicy.for_conv(variant, spatial_output=True, has_residual=True).threads == 320
    assert LaunchPolicy.for_input().threads == 288
    c192 = LaunchPolicy.for_conv(specialization(ConvConfig(ci=192, co=192)), fuse_norm=True)
    for changes in ({"helper_registers": 72}, {"auxiliary_warps": 1}, {"epilogue_registers": 105}):
        try:
            replace(c192, **changes)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid launch policy: {changes}")
    print("Index-width boundaries and launch-policy invariants: PASS")


def check_device_offsets():
    import torch
    from cuda.bindings import driver
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from mega_wan_vae.kernels.conv import _flat_nthwc_offset

    @cute.kernel
    def offsets(values: cute.Tensor, result: cute.Tensor):
        i, _, _ = cute.arch.thread_idx()
        if i < values.shape[0]:
            result[i] = _flat_nthwc_offset(
                values[i, 0],
                values[i, 1],
                values[i, 2],
                values[i, 3],
                values[i, 4],
                values[i, 5],
                values[i, 6],
                values[i, 7],
                values[i, 8],
            )

    @cute.jit
    def launch(values: cute.Tensor, result: cute.Tensor, stream: driver.CUstream):
        offsets(values, result).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    for dtype in (torch.int32, torch.int64):
        coordinates = [
            (100_000, 2, 721, 1279, 191, 3, 722, 1282, 192),
            (0, 2, 719, 1279, 319, 3, 722, 1282, 320),
            (2**31 - 1, 1, 1, 1, 7, 3, 3, 3, 8),
        ]
        if dtype == torch.int64:
            coordinates.append((2**32 + 1, 1, 1, 1, 7, 3, 3, 3, 8))
        expected = [
            (((n * tsize + t) * hsize + h) * wsize + w) * channels + c
            for n, t, h, w, c, tsize, hsize, wsize, channels in coordinates
        ]
        values = torch.tensor(coordinates, dtype=dtype, device="cuda")
        result = torch.empty(len(coordinates), dtype=torch.int64, device="cuda")
        operands = from_dlpack(values), from_dlpack(result)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        executable = cute.compile(launch, *operands, stream, options="--enable-tvm-ffi")
        executable(values, result, stream)
        assert result.cpu().tolist() == expected
    print("Production 64-bit offset arithmetic with narrow/wide coordinates: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    check_boundaries()
    if args.device == "cuda":
        check_device_offsets()
