"""Gather or scatter rows of multiple BF16 state buffers in one CUDA launch."""

import math
import sys
from functools import cache
from pathlib import Path

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary


@cute.kernel
def state_copy_kernel(
    storage: tuple[cute.Tensor, ...],
    rows: tuple[cute.Tensor, ...],
    indices: cute.Tensor,
    widths: cutlass.Constexpr,
    scatter: cutlass.Constexpr,
):
    tile, row, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    slot = indices[row]
    owns_slot = True
    if cutlass.const_expr(scatter):
        # Padded graph rows may share a scratch slot. Compare prior rows in
        # parallel while retaining first-row ownership of each destination.
        for first in cutlass.range(0, row, 32):
            previous = first + thread % 32
            duplicate = False
            if previous < row:
                duplicate = indices[previous] == slot
            else:
                pass
            owns_slot = owns_slot & ~cute.arch.vote_any_sync(duplicate)
    else:
        pass
    first_tile = 0
    for buffer_index in cutlass.range_constexpr(len(widths)):
        width = widths[buffer_index]
        tiles = (width + 1023) // 1024
        if owns_slot and tile >= first_tile and tile < first_tile + tiles:
            offset = (tile - first_tile) * 1024 + thread * 8
            if offset < width:
                storage_pointer = storage[buffer_index].iterator.raw_ptr() + slot * width + offset
                row_pointer = rows[buffer_index].iterator.raw_ptr() + cutlass.Int64(row) * width + offset
                if cutlass.const_expr(scatter):
                    storage_pointer.store(row_pointer.load(count=8, alignment=16), alignment=16)
                else:
                    row_pointer.store(storage_pointer.load(count=8, alignment=16), alignment=16)
            else:
                pass
        else:
            pass
        first_tile += tiles


@cute.jit
def state_copy_host(storage, rows, indices, stream, widths: cutlass.Constexpr, scatter: cutlass.Constexpr):
    tiles = 0
    for buffer_index in cutlass.range_constexpr(len(widths)):
        tiles += (widths[buffer_index] + 1023) // 1024
    state_copy_kernel(storage, rows, indices, widths, scatter).launch(
        grid=(tiles, indices.shape[0], 1), block=(128, 1, 1), stream=stream
    )


@cache
def compile_kernel(widths: tuple[int, ...], scatter: bool, architecture: str):
    slots, batch = cutlass.sym_int64(), cutlass.sym_int64()
    storage = tuple(
        make_fake_compact_tensor(cutlass.BFloat16, (slots, width), stride_order=(1, 0), assumed_align=16)
        for width in widths
    )
    rows = tuple(
        make_fake_compact_tensor(cutlass.BFloat16, (batch, width), stride_order=(1, 0), assumed_align=16)
        for width in widths
    )
    indices = make_fake_compact_tensor(cutlass.Int64, (batch,), stride_order=(0,))
    return cute.compile(
        state_copy_host,
        storage,
        rows,
        indices,
        make_fake_stream(),
        widths,
        scatter,
        options=f"--enable-tvm-ffi --gpu-arch={architecture}",
    )


def copy_state_rows(
    storage: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    *,
    scatter: bool,
) -> None:
    """Copy BF16 rows; scatter uses the first occurrence of each destination index."""
    widths = tuple(math.prod(buffer.shape[1:]) for buffer in storage)
    assert len(storage) == len(rows) > 0
    assert indices.dtype == torch.int64 and indices.ndim == 1 and indices.is_contiguous()
    for source, target, width in zip(storage, rows, widths):
        assert width > 0 and width % 8 == 0
        assert source.dtype == target.dtype == torch.bfloat16
        assert source.device == target.device == indices.device
        assert source.is_contiguous() and target.is_contiguous()
        assert target.shape == (indices.numel(), *source.shape[1:])
    compile_kernel(widths, scatter, device_arch(indices.device))(
        tuple(buffer.view(buffer.shape[0], -1) for buffer in storage),
        tuple(buffer.view(buffer.shape[0], -1) for buffer in rows),
        indices,
        driver.CUstream(torch.cuda.current_stream(indices.device).cuda_stream),
    )


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    results = []
    for batch in (1, 8):
        indices = torch.arange(batch, device="cuda", dtype=torch.int64) * 2
        storage = tuple(torch.randn(32, width, device="cuda", dtype=torch.bfloat16) for width in (96, 41472, 72704))
        rows = tuple(torch.empty(batch, buffer.shape[1], device="cuda", dtype=buffer.dtype) for buffer in storage)
        for scatter in (False, True):

            def reference(storage=storage, rows=rows, indices=indices, scatter=scatter):
                for source, target in zip(storage, rows):
                    if scatter:
                        source.index_copy_(0, indices, target)
                    else:
                        torch.index_select(source, 0, indices, out=target)

            def custom(storage=storage, rows=rows, indices=indices, scatter=scatter):
                copy_state_rows(storage, rows, indices, scatter=scatter)

            compiled = torch.compile(reference, fullgraph=True)
            for trial in range(3):
                indices.copy_((torch.arange(batch, device="cuda") * 2 + trial) % 32)
                if scatter:
                    expected = tuple(buffer.clone() for buffer in storage)
                    for buffer, row in zip(expected, rows):
                        row.normal_()
                        buffer.index_copy_(0, indices, row)
                else:
                    expected = tuple(buffer.index_select(0, indices) for buffer in storage)
                custom()
                for actual, target in zip(storage if scatter else rows, expected):
                    torch.testing.assert_close(actual, target, atol=0, rtol=0)
            results.append(
                (
                    (batch, "scatter" if scatter else "gather"),
                    measure({"Torch": reference, "Compiled": compiled, "Fused": custom}),
                )
            )
    print_kernel_summary(results, columns=["B", "Operation"], workload="multi-buffer row copies; exact validation PASS")


if __name__ == "__main__":
    main()
