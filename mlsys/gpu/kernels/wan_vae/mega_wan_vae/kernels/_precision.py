# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Operand packing for TF32 convolution with FP32 storage.

Ordinary TF32 rounds each operand to nearest/even with a 10-bit fraction and
keeps FP32 storage. Its finite overflow rounds to infinity. Single TF32 products
are accumulated in FP32; periodic SIMT promotion limits long-reduction error.

Experimental TF32x3 packs each group of 16 channels as high/low components:
high = RN_TF32(x), low = RN_TF32(x - high). Convolution evaluates high*high,
high*low and low*high, omitting low*low. Both MMA accumulation and periodic
SIMT summation use FP32. This is an approximate finite-input arithmetic mode,
not strict IEEE FP32 multiplication or a bitwise substitute for cuDNN.

Extreme exponents, subnormal products and nonfinite operands are outside the
validated VAE contract. Packing preserves nonfinite high components and avoids
overflowing a finite high component, but the three-product expansion does not
promise IEEE infinity/NaN propagation or gradual underflow in Tensor Cores.
"""

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__:
    from ._compile import compile_cache, compile_kernel
else:
    from _compile import compile_cache, compile_kernel


@cute.jit
def _round_tf32(value, finite_components: cutlass.Constexpr = True):
    """Round binary32 significands to nearest/even TF32, keeping FP32 storage."""
    bits = value.bitcast(cutlass.Int32)
    rounded = (bits + 4095 + ((bits >> 13) & 1)) & -8192
    finite = (bits & 0x7F800000) != 0x7F800000
    overflow = (rounded & 0x7F800000) == 0x7F800000
    # A finite value rounded to infinity has a finite high/low decomposition
    # if the high component instead rounds toward zero.
    if cutlass.const_expr(finite_components):
        rounded = cutlass.vector.where(finite & overflow, bits & -8192, rounded)
    rounded = cutlass.vector.where(finite, rounded, bits)
    return rounded.bitcast(cutlass.Float32)


@cute.kernel
def _pack_kernel(x: cute.Tensor, out: cute.Tensor, numel: cutlass.Int64, compensated: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    offset = (cutlass.Int64(block) * 256 + tid) * 4
    if offset < numel:
        value = (x.iterator.raw_ptr() + offset).load(count=4, alignment=16)
        high = _round_tf32(value, compensated)
        if cutlass.const_expr(compensated):
            low = _round_tf32(value - high)
            target = (offset // 16) * 32 + offset % 16
            (out.iterator.raw_ptr() + target).store(high, alignment=16)
            (out.iterator.raw_ptr() + target + 16).store(low, alignment=16)
        else:
            (out.iterator.raw_ptr() + offset).store(high, alignment=16)


@cute.jit
def _pack_host(
    x: cute.Tensor, out: cute.Tensor, numel: cutlass.Int64, stream: driver.CUstream, compensated: cutlass.Constexpr
):
    # Pass the count explicitly: tensor shape access can narrow symbolic Int64
    # extents before launch arithmetic, despite the fake tensor's declaration.
    _pack_kernel(x, out, numel, compensated).launch(
        grid=((numel + 1023) // 1024, 1, 1), block=(256, 1, 1), stream=stream
    )


@compile_cache()
def _compile_components(compensated: bool):
    tensor = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int64(divisibility=16),), assumed_align=16)
    output = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int64(divisibility=16),), assumed_align=16)
    return compile_kernel(
        _pack_host, tensor, output, cutlass.Int64(1), make_fake_stream(), compensated, options="--enable-tvm-ffi"
    )


def pack_components(x: torch.Tensor, out: torch.Tensor, *, compensated: bool = True) -> None:
    """Pack into caller-owned storage on the current stream, without a CPU sync."""
    if x.ndim < 1 or x.shape[-1] == 0 or x.shape[-1] % 16:
        raise ValueError("Component packing requires channel rows divisible by 16")
    if out.shape != (*x.shape[:-1], x.shape[-1] * (2 if compensated else 1)):
        raise ValueError("Output shape must match the selected TF32 packing policy")
    for tensor in (x, out):
        if (
            tensor.dtype != torch.float32
            or not tensor.is_cuda
            or tensor.device != x.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            raise ValueError("Component tensors must be aligned contiguous FP32 on one CUDA device")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("Component packing is inference-only")
    if x.numel():
        with torch.cuda.device(x.device):
            _compile_components(compensated)(
                x.view(-1),
                out.view(-1),
                x.numel(),
                driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
            )


def resolve_precision(dtype: torch.dtype, precision: str | None = None) -> str:
    """Resolve storage plus an explicit policy, or the current cuDNN permission.

    Resolve defaults once, before packing weights; carry the returned policy
    through dispatch. TF32x3 is experimental and never selected by allow_tf32.
    """
    if dtype == torch.bfloat16:
        if precision not in (None, "bf16"):
            raise ValueError("BF16 weights require bf16 arithmetic")
        return "bf16"
    if dtype != torch.float32:
        raise ValueError("Convolution requires BF16 or FP32 storage")
    if precision is None:
        return "tf32" if torch.backends.cudnn.allow_tf32 else "ieee"
    if precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("FP32 weights require ieee, tf32 or tf32x3 arithmetic")
    return precision
