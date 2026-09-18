# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pipelined BF16 Wan convolution using the public CUTLASS DSL runtime.

For contiguous NTHWC input X and OTRSC filters W, this computes

    Y[n,t,h,w,o] = BF16(sum_{dt,dh,dw,c}
        FP32(X[n,t+dt,h+dh,w+dw,c]) * FP32(W[o,dt,dh,dw,c]))

Input already includes causal/spatial padding. The filter is 3x3x3;
stride and dilation are one, groups is one. The N tile is fixed at 160.
``prepare`` returns this raw convolution. ``prepare_next`` additionally applies
the C160/C320 postprocessing below, with B denoting a separate BF16 rounding:

    v_c = B(FP32(Y_c) + FP32(conv_bias_c))
    d = max(sqrt(sum_c(FP32(v_c)**2)), 1e-12)
    u_c = B(FP32(v_c) / d)
    a_c = B(FP32(B(FP32(u_c) * sqrt(Co))) * FP32(gamma_c))
    out_c = B(FP32(a_c) / (1 + exp(-FP32(a_c))))

TMA performs implicit im2col directly
into staged shared memory; no global im2col or tensor transpose is allocated.
The persistent device pipeline has independent producer, MMA, and epilogue
warps and two TMEM accumulator stages. Fused C320->C320 uses six operand stages
and six SMEM output-store stages. ``prepare_next`` stores activated output
directly into the next convolution's padded NTHWC input and emits its temporal
cache in the same launch. With a residual branch it replaces v_c above by
B(FP32(v_c) + FP32(B(FP32(residual_c) + FP32(residual_bias_c)))), omitting
the inner addition when residual_bias is absent. This summed value is also
saved as the next block's skip input. No raw convolution output is materialized.
``prepare_spatial_residual`` applies that bias/residual sum without normalization
and writes a zero bottom row/right column for spatial downsampling. This
pointwise path supports independent output-channel tiles. Normalization keeps
all channels of a row in one CTA group: one N160 tile for C160, two sequential
N160 tiles for C320. C320 requires 2CTA MMA; both widths support residual addition.

C12 input uses a 1CTA gather producer and seven K64 tiles over C16-padded
activations and [160,448] weights. General convolutions use TMA im2col with
27*ceil(Ci/64) reduction tiles. Both use two TMEM accumulators; epilogue warps
drain them into 64B-swizzled SMEM for TMA stores. Halo/cache writers are
disjoint from the epilogue's interior writers. A padded peer CTA must still
participate in every paired-MMA handshake, including the final M tail.

The private descriptor bridge below owns aligned CUDA driver tensor maps and
their backing tensors. Descriptors travel as GridConstant arguments through
the 4.7.1 default executor, not TVM FFI. Fake descriptors permit GPU-free
compilation. Bound launches retain outputs and descriptor owners for graph
replay; module forwards allocate fresh outputs and record their CUDA stream.
The deprecated scheduler is intentionally the installed 4.7.1 wheel API.
"""

import argparse
from dataclasses import dataclass
from collections.abc import Callable
from functools import lru_cache
import math
import os
from pathlib import Path

import torch

os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(Path(__file__).resolve().parent))

import cutlass
import ctypes
from collections.abc import Sequence
from cutlass._mlir import ir
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver
from cutlass.utils import PersistentTileSchedulerParams, StaticPersistentTileScheduler

if __package__:
    from ._utils import PreparedConvInput, measure, print_benchmark_table
else:
    from _utils import PreparedConvInput, measure, print_benchmark_table

_DESCRIPTOR_BYTES = 128
_DESCRIPTOR_ALIGNMENT = 64
_DATA_TYPES = {
    cutlass.Float16: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    cutlass.BFloat16: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    cutlass.Float32: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
}
_SWIZZLE_BYTES = {
    cuda.TensorMapSwizzle.none: 0,
    cuda.TensorMapSwizzle.s32b: 32,
    cuda.TensorMapSwizzle.s64b: 64,
    cuda.TensorMapSwizzle.s128b: 128,
}


class _HostTensorMap:
    """Owned, aligned descriptor and JIT argument; use the builders below.

    At the host ABI, !cuda.tensor_map lowers to a pointer. Consequently
    __c_pointers__ returns the address of a pointer slot, NOT the descriptor
    address. The generated launch shim copies all 128 bytes into a by-value,
    64-byte-aligned grid-constant kernel parameter.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        box_dims: tuple[int, ...],
        swizzle: cuda.TensorMapSwizzle,
        owner: object | None = None,
    ) -> None:
        self.dtype = dtype
        self.box_dims = box_dims
        self.swizzle = swizzle
        self._owner = owner
        self._storage = ctypes.create_string_buffer(
            _DESCRIPTOR_BYTES + _DESCRIPTOR_ALIGNMENT - 1
        )
        self._address = (ctypes.addressof(self._storage) + 63) & ~63
        self._pointer = ctypes.c_void_p(self._address)
        self._encoded = False

    @property
    def address(self) -> int:
        """Host descriptor address, suitable for a CUDA encoder output pointer."""
        return self._address

    def __repr__(self) -> str:
        # Exclude address, owner and fake/real state from compilation identity.
        return (
            f"_HostTensorMap({self.dtype.__name__},"
            f"box_dims={self.box_dims},swizzle={self.swizzle.name})"
        )

    def __c_pointers__(self) -> list[int]:
        """Marshal one retained pointer slot for the default JIT executor."""
        if not self._encoded:
            return []
        return [ctypes.addressof(self._pointer)]

    def __get_mlir_types__(self) -> list[ir.Type]:
        """Expose the same argument type for fake and real descriptors."""
        return [ir.Type.parse("!cuda.tensor_map")]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> cuda.TensorMap:
        """Reconstruct the public device handle, retaining static box metadata."""
        if len(values) != 1:
            raise ValueError("A tensor map requires exactly one MLIR value")
        return cuda.TensorMap(
            values[0], dtype=self.dtype, box_dims=self.box_dims, swizzle=self.swizzle
        )


def _contiguous_tma_layout(
    shape: Sequence[int], dtype: type[cutlass.Numeric]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convert a C-contiguous shape to (TMA dimensions, byte strides).

    NDHWC becomes C,W,H,D,N; KTRSC becomes C,S,R,T,K. Noncontiguous tensors
    must instead supply their actual dimensions and byte strides explicitly.
    """
    if dtype not in _DATA_TYPES:
        raise ValueError("Only Float16, BFloat16 and Float32 are supported")
    dims = _integers(shape, "shape", 1, 2**32)
    if not 1 <= len(dims) <= 5:
        raise ValueError("Tensor rank must be between 1 and 5")
    dims = dims[::-1]
    stride = dtype.width // 8
    strides = []
    for extent in dims[:-1]:
        stride *= extent
        strides.append(stride)
    return dims, tuple(strides)


def _integers(
    values: Sequence[int], name: str, lower: int, upper: int
) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(x) is not int or not lower <= x <= upper for x in result):
        raise ValueError(f"{name} must contain integers in [{lower}, {upper}]")
    return result


def _validate_layout(
    global_address: int | None,
    dtype: type[cutlass.Numeric],
    global_dims: Sequence[int],
    global_strides: Sequence[int],
    fake: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if dtype not in _DATA_TYPES:
        raise ValueError("Only Float16, BFloat16 and Float32 are supported")
    dims = _integers(global_dims, "global_dims", 1, 2**32)
    strides = _integers(global_strides, "global_strides", 16, 2**40 - 1)
    if not 1 <= len(dims) <= 5 or len(strides) != len(dims) - 1:
        raise ValueError("Expected rank 1..5 and rank-1 global byte strides")
    if any(s % 16 for s in strides):
        raise ValueError("Global byte strides must be multiples of 16")
    if not (fake and global_address is None):
        if (
            type(global_address) is not int
            or not 0 < global_address < 2**64
            or global_address % 16
        ):
            raise ValueError("global_address must be a nonzero 16-byte-aligned pointer")
    return dims, strides


def _validate_swizzle(
    dtype: type[cutlass.Numeric], inner: int, swizzle: cuda.TensorMapSwizzle
) -> cuda.TensorMapSwizzle:
    swizzle = cuda.TensorMapSwizzle(swizzle)
    if swizzle not in _SWIZZLE_BYTES:
        raise ValueError("Only none/32B/64B/128B ordinary swizzles are supported")
    inner_bytes = inner * dtype.width // 8
    if inner_bytes % 16:
        raise ValueError("The inner box size must be a multiple of 16 bytes")
    if _SWIZZLE_BYTES[swizzle] and inner_bytes > _SWIZZLE_BYTES[swizzle]:
        raise ValueError("The inner box byte size exceeds the swizzle size")
    return swizzle


@lru_cache(maxsize=1)
def _driver_library() -> ctypes.CDLL:
    """Load only when encoding a real descriptor; never initialize CUDA."""
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError(
            "Tensor-map encoding requires the NVIDIA libcuda.so.1 driver"
        ) from exc
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    prefix = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        u64p,
        u64p,
    ]
    suffix = [u32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.cuTensorMapEncodeTiled.argtypes = prefix + [u32p] + suffix
    library.cuTensorMapEncodeTiled.restype = ctypes.c_int
    library.cuTensorMapEncodeIm2col.argtypes = (
        prefix + [i32p, i32p, ctypes.c_uint32, ctypes.c_uint32] + suffix
    )
    library.cuTensorMapEncodeIm2col.restype = ctypes.c_int
    return library


def _check_encoding(
    result: int, name: str, descriptor: _HostTensorMap
) -> _HostTensorMap:
    if result != 0:
        raise RuntimeError(f"{name} failed with CUresult {result}")
    descriptor._encoded = True
    return descriptor


def _create_tensor_map_im2col(
    global_address: int | None,
    dtype: type[cutlass.Numeric],
    global_dims: Sequence[int],
    global_strides: Sequence[int],
    *,
    lower_corner: Sequence[int],
    upper_corner: Sequence[int],
    channels_per_pixel: int,
    pixels_per_column: int,
    traversal_strides: Sequence[int] | None = None,
    swizzle: cuda.TensorMapSwizzle = cuda.TensorMapSwizzle.s128b,
    owner: object | None = None,
    fake: bool = False,
) -> _HostTensorMap:
    """Encode an activation IM2COL descriptor, or its compile-only counterpart.

    For 3D fprop, spatial order is W,H,D; lower=-pad_lower and
    upper=pad_upper-(filter-1)*dilation. Traversal is (1,sw,sh,sd,1), NOT
    dilation. Defaults to unit traversal. Kernel anchors and filter offsets
    must use the same convolution geometry. Zero corners/unit traversal also
    describe an NZPQK IM2COL output store. Metadata box_dims=(channels,pixels)
    describes the shared-memory tile, not the rank of the global tensor.
    """
    dims, strides = _validate_layout(
        global_address, dtype, global_dims, global_strides, fake
    )
    rank = len(dims)
    if rank not in (3, 4, 5):
        raise ValueError("IM2COL requires tensor rank 3, 4 or 5")
    limit = {3: 32768, 4: 128, 5: 16}[rank]
    lower = _integers(lower_corner, "lower_corner", -limit, limit - 1)
    upper = _integers(upper_corner, "upper_corner", -limit, limit - 1)
    if len(lower) != rank - 2 or len(upper) != rank - 2:
        raise ValueError("IM2COL requires rank-2 spatial corners")
    if any(dims[i + 1] + upper[i] - lower[i] <= 0 for i in range(rank - 2)):
        raise ValueError("IM2COL spatial bounding box must have positive extent")
    _integers((channels_per_pixel,), "channels_per_pixel", 1, 256)
    _integers((pixels_per_column,), "pixels_per_column", 1, 1024)
    traversal = _integers(
        (1,) * rank if traversal_strides is None else traversal_strides,
        "traversal_strides",
        1,
        8,
    )
    if len(traversal) != rank:
        raise ValueError("Expected rank traversal strides")
    swizzle = _validate_swizzle(dtype, channels_per_pixel, swizzle)
    descriptor = _HostTensorMap(
        dtype, (channels_per_pixel, pixels_per_column), swizzle, owner
    )
    if fake:
        return descriptor
    result = _driver_library().cuTensorMapEncodeIm2col(
        descriptor.address,
        int(_DATA_TYPES[dtype]),
        rank,
        global_address,
        (ctypes.c_uint64 * rank)(*dims),
        (ctypes.c_uint64 * (rank - 1))(*strides),
        (ctypes.c_int32 * (rank - 2))(*lower),
        (ctypes.c_int32 * (rank - 2))(*upper),
        channels_per_pixel,
        pixels_per_column,
        (ctypes.c_uint32 * rank)(*traversal),
        int(driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE),
        int(swizzle),
        int(driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE),
        int(driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
    )
    return _check_encoding(result, "cuTensorMapEncodeIm2col", descriptor)


def _create_tensor_map_tiled(
    global_address: int | None,
    dtype: type[cutlass.Numeric],
    global_dims: Sequence[int],
    global_strides: Sequence[int],
    box_dims: Sequence[int],
    *,
    swizzle: cuda.TensorMapSwizzle = cuda.TensorMapSwizzle.s128b,
    owner: object | None = None,
    fake: bool = False,
) -> _HostTensorMap:
    """Encode unit-traversal tiled weights; all arrays are in TMA order.

    For KTRSC weights use dims=(C,S,R,T,K), box=(k_tile,1,1,1,n_per_cta).
    ``fake=True`` accepts global_address=None and performs no driver calls.
    """
    dims, strides = _validate_layout(
        global_address, dtype, global_dims, global_strides, fake
    )
    rank = len(dims)
    box = _integers(box_dims, "box_dims", 1, 256)
    if len(box) != rank:
        raise ValueError("Expected rank box dimensions")
    swizzle = _validate_swizzle(dtype, box[0], swizzle)
    descriptor = _HostTensorMap(dtype, box, swizzle, owner)
    if fake:
        return descriptor
    result = _driver_library().cuTensorMapEncodeTiled(
        descriptor.address,
        int(_DATA_TYPES[dtype]),
        rank,
        global_address,
        (ctypes.c_uint64 * rank)(*dims),
        (ctypes.c_uint64 * (rank - 1))(*strides),
        (ctypes.c_uint32 * rank)(*box),
        (ctypes.c_uint32 * rank)(*((1,) * rank)),
        int(driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE),
        int(swizzle),
        int(driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE),
        int(driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
    )
    return _check_encoding(result, "cuTensorMapEncodeTiled", descriptor)


@cute.kernel
def kernel(
    # Leading specialization label remains visible before DSL name truncation.
    operation: cutlass.Constexpr[str],
    # Persistent tile scheduler parameters
    tile_sched_params: PersistentTileSchedulerParams,
    # Constexpr knobs
    mma_tiler: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_shape_mnk: cutlass.Constexpr[tuple[int, int, int]],
    # Host-computed: T*R*S*ceil_div(Ci, 64), including each channel tail.
    k_tile_cnt: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    use_2cta_instrs: cutlass.Constexpr[bool],
    # Conv geometry consumed by the im2col TMA producer / epilogue:
    # zpq      = output spatial dims (Z, P, Q) used to decompose linear M -> (n,z,p,q)
    # trs      = filter spatial dims (T, R, S) used to decompose K-tile index
    # lower_pad_dhw / stride_dhw / dilation_dhw = standard conv params
    zpq: cutlass.Constexpr[tuple[int, int, int]],
    # Host-built TMA descriptors, consumed by the device-side
    # prims.cp_async_bulk_tensor_* / prims.prefetch_tensormap calls.
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    conv_bias: cute.Tensor = None,
    gamma: cute.Tensor = None,
    packed_x: cute.Tensor = None,
    packed_shape: cutlass.Constexpr = None,
    prep_padded: cute.Tensor = None,
    prep_cache: cute.Tensor = None,
    prep_previous: cute.Tensor = None,
    PREP_SHAPE: cutlass.Constexpr = None,
    residual: cute.Tensor = None,
    residual_bias: cute.Tensor = None,
    prep_residual: cute.Tensor = None,
    SPATIAL_SHAPE: cutlass.Constexpr = None,
) -> None:
    """Run TMA -> BF16 tcgen05 MMA -> BF16 TMA store; see module ABI contract.

    All geometry and stage arguments are constexpr. Unsupported configurations
    fail at specialization; descriptor shapes and k_tile_cnt remain the host's
    responsibility because Ci and descriptor metadata are not kernel arguments.
    """
    FUSE_NORM: cutlass.Constexpr = PREP_SHAPE is not None
    ab_dtype = cutlass.BFloat16
    c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    num_acc_stage = 2
    trs = (3, 3, 3)
    lower_pad_dhw = (0, 0, 0)
    stride_dhw = (1, 1, 1)
    dilation_dhw = (1, 1, 1)
    if cutlass.const_expr(mma_tiler[0] != (256 if use_2cta_instrs else 128)):
        raise ValueError("Convolution core requires 128 rows per CTA")
    if cutlass.const_expr(mma_tiler[1] != 160):
        raise ValueError("Wan convolution uses tile N=160")
    if cutlass.const_expr(mma_tiler[2] != 64):
        raise ValueError("Convolution core requires tile K=64")
    if cutlass.const_expr(mma_inst_shape_mnk != (mma_tiler[0], mma_tiler[1], 16)):
        raise ValueError("MMA instruction shape must be (tileM, tileN, 16)")
    if cutlass.const_expr(num_ab_stage < 1 or num_c_stage < 1 or num_c_stage > 8):
        raise ValueError("AB stages must be positive; C stages must be in [1, 8]")
    if cutlass.const_expr(k_tile_cnt <= 0):
        raise ValueError("Convolution core requires a positive K tile count")
    if cutlass.const_expr(packed_shape is not None):
        if cutlass.const_expr(use_2cta_instrs or FUSE_NORM or k_tile_cnt != 7):
            raise ValueError("Packed C12 gather requires raw 1CTA and seven K tiles")
        if cutlass.const_expr(
            packed_x is None
            or trs != (3, 3, 3)
            or lower_pad_dhw != (0, 0, 0)
            or stride_dhw != (1, 1, 1)
            or dilation_dhw != (1, 1, 1)
        ):
            raise ValueError(
                "Packed gather requires C12 input and unpadded unit-stride 3x3x3"
            )
        if cutlass.const_expr(zpq != tuple(d - 2 for d in packed_shape[1:])):
            raise ValueError(
                "Packed gather output geometry must match the padded input"
            )
    if cutlass.const_expr(FUSE_NORM):
        if cutlass.const_expr(mma_tiler[1] != 160):
            raise ValueError("Fused normalization requires tileN=160")
        if cutlass.const_expr(conv_bias is None or gamma is None):
            raise ValueError("Fused normalization requires conv_bias and gamma")
        if cutlass.const_expr(
            conv_bias.element_type != cutlass.BFloat16
            or gamma.element_type != cutlass.BFloat16
        ):
            raise TypeError("Fused conv_bias and gamma must be BF16")
        if cutlass.const_expr(
            conv_bias.shape not in ((160,), (320,)) or gamma.shape != conv_bias.shape
        ):
            raise ValueError(
                "Fused conv_bias and gamma must have shape (160,) or (320,)"
            )
    if cutlass.const_expr(PREP_SHAPE is not None):
        if cutlass.const_expr(not FUSE_NORM or packed_shape is not None):
            raise ValueError("Next-convolution preparation requires normalization")
        if cutlass.const_expr(
            prep_padded is None or prep_cache is None or prep_previous is None
        ):
            raise ValueError("Preparation requires padded, cache and history tensors")
    if cutlass.const_expr(residual is not None):
        if cutlass.const_expr(
            SPATIAL_SHAPE is None and (PREP_SHAPE is None or prep_residual is None)
        ):
            raise ValueError(
                "Residual fusion requires preparation and a saved-sum output"
            )
    if cutlass.const_expr(SPATIAL_SHAPE is not None):
        if cutlass.const_expr(
            FUSE_NORM
            or PREP_SHAPE is not None
            or residual is None
            or conv_bias is None
            or prep_padded is None
        ):
            raise ValueError(
                "Spatial residual output requires bias/residual "
                "and output without norm/history"
            )

    norm_channels: cutlass.Constexpr = gamma.shape[0] if FUSE_NORM else 160
    paired_norm: cutlass.Constexpr = FUSE_NORM and norm_channels == 320
    if cutlass.const_expr(paired_norm and not use_2cta_instrs):
        raise ValueError("C320 normalization requires 2CTA MMA")

    # Warp / thread / cluster identity
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    # CTA-group geometry, threaded through every tcgen05 call site below.
    # _atom_thr = MMA-atom CTA count (2 for the 2-CTA tcgen05 group, else 1).
    # Cluster ranks are M-fast (grid.x = cluster_m), so a 2-CTA group G spans
    # ranks {2G, 2G+1} with its leader at the even rank 2G. _cta_group is the
    # matching nvvm enum.
    _atom_thr = 2 if cutlass.const_expr(use_2cta_instrs) else 1
    _cta_group = (
        prims.CTAGroup.CTA_2
        if cutlass.const_expr(use_2cta_instrs)
        else prims.CTAGroup.CTA_1
    )
    if cutlass.const_expr(use_2cta_instrs):
        # 2-CTA group leader = even cluster rank (covers multi-group clusters,
        # e.g. cluster_m > 2, where group 1's leader sits at rank 2).
        is_leader_cta = (cta_rank_in_cluster % _atom_thr) == 0
        # tcgen05_commit multicast mask: 3 << rank covers the issuing group's
        # own pair {2G, 2G+1}. Only the leader (even rank) issues the commit,
        # so this is the multi-group-safe form (hard-coded 3 would deadlock
        # groups G>0). See prims.tcgen05_commit docstring.
        _commit_mask = cutlass.Int32(3) << cta_rank_in_cluster
    else:
        # 1-CTA path: every CTA owns its own accumulator => its own leader.
        # constexpr True so the leader-gated scf.if branches elide entirely.
        is_leader_cta = True
        # None selects the CTA_1 no-multicast commit path (arrive on own mbar).
        _commit_mask = None

    # Wide TMEM loads let four lanes share each fused-normalization row.
    epilogue_warp_ids = tuple(
        range((16 if residual is not None or paired_norm else 8) if FUSE_NORM else 4)
    )
    mma_warp_id = len(epilogue_warp_ids)
    tma_warp_id = mma_warp_id + 1

    # Prefetch the three TMA descriptors from the MMA warp; the descriptor cache
    # is shared across the cluster.
    if warp_idx == mma_warp_id:
        if cutlass.const_expr(packed_shape is None):
            prims.prefetch_tensormap(tma_a_desc.get_ptr())
        prims.prefetch_tensormap(tma_b_desc.get_ptr())
        prims.prefetch_tensormap(tma_c_desc.get_ptr())

    # ===== Shared memory: data tiles + mbarriers + cluster init =====
    #
    # Every SMEM object is a flat cutlass.Array(space=smem), laid out in
    # declaration order. Swizzle is expressed by the descriptors
    # (cuda.TensorMapSwizzle.* on the host, Tcgen05SmemDesc layout on device),
    # not by the SMEM tensor type.
    #
    # Layout: mbarriers first (Int64 each):
    #   - ab_full / ab_empty (one per AB pipeline stage)
    #   - acc_full / acc_empty (one per accumulator stage)
    #   - tmem_dealloc_mbar (single, for the tcgen05 dealloc handshake)
    # then the TMEM holding slot (Int32, where tcgen05_alloc writes the TMEM
    # ptr), then A/B as Int8 byte arrays and C as a BF16 array.
    # cutlass.Array offsets an element/byte view with
    # ``.subview(n)`` and hands out a raw Pointer with ``.data_ptr(n)`` (the
    # latter is masked for bit-24 CTA_2 routing).
    ab_full_mbar_ptr = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem
    )
    ab_empty_mbar_ptr = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem
    )
    acc_full_mbar_ptr = cutlass.Array(
        cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem
    )
    acc_empty_mbar_ptr = cutlass.Array(
        cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem
    )
    tmem_dealloc_mbar_ptr = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # Per-CTA per-stage SMEM byte sizes; host descriptors must agree.
    _m_per_cta = mma_tiler[0] // _atom_thr
    _n_per_cta = mma_tiler[1] // _atom_thr
    _k_tile = cute.size(mma_tiler, mode=[2])
    _a_stage_bytes = _m_per_cta * _k_tile * (ab_dtype.width // 8)
    _b_stage_bytes = _n_per_cta * _k_tile * (ab_dtype.width // 8)
    # Both epilogue mappings publish the same 128-row, 32-channel BF16 tile.
    _epi_subtile_n = 32
    _c_tile_rows = 128
    _c_stage_bytes = _c_tile_rows * _epi_subtile_n * (c_dtype.width // 8)

    # A/B byte arrays and the BF16 C array are all 1024B aligned.
    sA = cutlass.Array(
        cutlass.Int8,
        _a_stage_bytes * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    sB = cutlass.Array(
        cutlass.Int8,
        _b_stage_bytes * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    sC = cutlass.Array(
        c_dtype,
        (_c_stage_bytes // (c_dtype.width // 8)) * num_c_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    if cutlass.const_expr(paired_norm):
        norm_params = cutlass.Array(
            cutlass.BFloat16, 640, space=cutlass.AddressSpace.smem, alignment=16
        )
        if tidx < 160:
            norm_params.store(
                (conv_bias.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                tidx * 2,
            )
            norm_params.store(
                (gamma.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                320 + tidx * 2,
            )
        # Publish ordinary SMEM loads independently of the async TMA barriers.
        prims.barrier_cta_sync(
            3,
            thread_count=32
            * (len(epilogue_warp_ids) + 2 + (4 if PREP_SHAPE is not None else 0)),
        )

    # mbarrier_init: one warp / one lane initializes every barrier.
    # Each epilogue warp in each cooperating CTA releases the accumulator.
    num_acc_empty_arrives = len(epilogue_warp_ids) * (2 if use_2cta_instrs else 1)
    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
            for i in cutlass.range_constexpr(num_acc_stage):
                prims.mbarrier_init(
                    acc_empty_mbar_ptr.subview(i), num_acc_empty_arrives
                )
                prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            for i in cutlass.range_constexpr(num_ab_stage):
                prims.mbarrier_init(
                    ab_full_mbar_ptr.subview(i),
                    128 if packed_shape is not None else 1,
                )
                prims.mbarrier_init(ab_empty_mbar_ptr.subview(i), 1)

    # Cluster sync sandwich: fence_mbarrier_init publishes the mbarrier_init
    # writes to the cluster; barrier_cluster_arrive_relaxed signals this CTA is
    # done initializing; barrier_cluster_wait (later, after tile geometry is
    # set up) blocks until every CTA in the cluster arrives.
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()

    # Initial pipeline phase bits — parity flips on stage wrap-around.
    ab_empty_phase_bit = 1
    ab_full_phase_bit = 0
    acc_empty_phase_bit = 1
    acc_full_phase_bit = 0

    # Pre-build the tcgen05 instruction descriptor (constexpr) so MMA / epilogue
    # warps can reference it without recomputing. mma_inst_shape_mnk encodes the
    # per-instruction (M, N, K) shape, e.g. (256, 32, 16) for BF16/2cta.
    idesc = prims.Tcgen05InstrDesc.build(
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        c_dtype=acc_dtype,
        m_dim=mma_inst_shape_mnk[0],
        n_dim=mma_inst_shape_mnk[1],
    )
    # BF16 uses the F16 instruction kind; idesc selects BF16 operands.
    mma_kind = prims.Tcgen05MMAKind.F16

    # Persistent tile scheduler.
    tile_sched = StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    # tcgen05 TMEM allocation knob (always 512 cols on Blackwell).
    num_tmem_cols = 512

    # Cluster wait — gates SMEM/mbarrier visibility before any warp loads/stores.
    prims.barrier_cluster_wait()

    # ===== Tile geometry (visible to all warps) =====
    #
    # The A/B loads and the C store are raw prims.cp_async_bulk_tensor_* calls
    # that take a descriptor pointer plus hand-computed coords; the epilogue
    # computes its im2col store coords (k_off, q, p, z, n) by hand. The MMA is
    # driven by prims.Tcgen05SmemDesc (A/B) + prims.make_tmem_ptr (acc). So no
    # partition tensors or MMA fragment objects are materialized here — only the
    # per-CTA tile scalars below.

    # Per-CTA tile sizes (used by raw TMA producer + tcgen05 desc strides).
    mma_tiler_per_cta_m = mma_tiler[0] // (2 if use_2cta_instrs else 1)
    n_per_cta = mma_tiler[1] // (2 if use_2cta_instrs else 1)
    k_tile_size = cute.size(mma_tiler, mode=[2])

    # Bytes-per-K-tile expectation. For the 2-CTA cluster the leader writes
    # expect_tx for both CTAs' loads (the peer slot is reached via the
    # mbarrier bit-24 clear inside the wrapper).
    a_bytes_per_stage = mma_tiler_per_cta_m * k_tile_size * (ab_dtype.width // 8)
    b_bytes_per_stage = n_per_cta * k_tile_size * (ab_dtype.width // 8)
    num_tma_copy_bytes = a_bytes_per_stage + b_bytes_per_stage
    if cutlass.const_expr(use_2cta_instrs):
        num_tma_copy_bytes = num_tma_copy_bytes * 2

    # ===== TMA producer warp =====
    # Per-CTA unicast (no multicast_mask) prims.cp_async_bulk_tensor loads.
    # Im2col coords are decomposed from the linear M index:
    # m_off_cta -> (n, z, p, q); spatial anchors are then q*str_w-pad_w (etc).
    # The filter coord (s, r, t) and C-element offset are loop-carried and
    # advanced by a colexicographic carry (add + compare + conditional reset),
    # so the loop body never divides the linear k by T*R*S.
    producer_warp_end = (
        9 if cutlass.const_expr(packed_shape is not None) else tma_warp_id + 1
    )
    auxiliary_threads = 64 if FUSE_NORM and not paired_norm else 128
    if warp_idx >= producer_warp_end:
        if cutlass.const_expr(SPATIAL_SHAPE is not None):
            batches, channels = SPATIAL_SHAPE
            z, h, w = zpq
            border_rows = h + w + 1
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (
                cutlass.Int64(by) + cutlass.Int64(gy) * bz
            )
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            while vector_idx < batches * z * border_rows * (channels // 8):
                channel = vector_idx % (channels // 8) * 8
                border_row = vector_idx // (channels // 8)
                nt = border_row // border_rows
                border = border_row % border_rows
                ph = cutlass.Int64(h)
                pw = border
                if border >= w + 1:
                    ph = border - w - 1
                    pw = cutlass.Int64(w)
                offset = ((nt * (h + 1) + ph) * (w + 1) + pw) * channels + channel
                zeros = cutlass.vector.full((8,), 0, cutlass.BFloat16)
                (prep_padded.iterator.raw_ptr() + offset).store(zeros, alignment=16)
                vector_idx += total_ctas * auxiliary_threads
        if cutlass.const_expr(PREP_SHAPE is not None):
            # Auxiliary rows and the TMA interior are disjoint. Enumerate only
            # the two history planes and current-frame spatial borders, not
            # the full output. Eight BF16 elements per lane give aligned,
            # coalesced stores across each 160-channel row.
            Z_out, P_out, Q_out = zpq
            batches, previous_frames = PREP_SHAPE
            cache_frames = min(2, Z_out + previous_frames)
            padded_h = P_out + 2
            padded_w = Q_out + 2
            history_rows = 2 * padded_h * padded_w
            border_rows = 2 * padded_w + 2 * P_out
            auxiliary_rows = history_rows + Z_out * border_rows
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            # The persistent scheduler places clusters along grid Z, not X.
            # Row/vector coordinates fit in 32 bits for encoder shapes. Widen
            # only element offsets, which can exceed 2 Gi elements after padding.
            index_type: cutlass.Constexpr = (
                cutlass.Int32
                if batches * auxiliary_rows * (norm_channels // 8) < 2**31
                else cutlass.Int64
            )
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (
                cutlass.Int64(by) + cutlass.Int64(gy) * bz
            )
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            while vector_idx < batches * auxiliary_rows * (norm_channels // 8):
                coordinate_idx = index_type(vector_idx)
                channel = coordinate_idx % (norm_channels // 8) * 8
                auxiliary_row = coordinate_idx // (norm_channels // 8)
                batch = auxiliary_row // auxiliary_rows
                local_row = auxiliary_row % auxiliary_rows
                pt = index_type(0)
                ph = index_type(0)
                pw = index_type(0)
                if local_row < history_rows:
                    pt = local_row // (padded_h * padded_w)
                    ph = local_row // padded_w % padded_h
                    pw = local_row % padded_w
                else:
                    border = (local_row - history_rows) % border_rows
                    pt = (local_row - history_rows) // border_rows + 2
                    if border < 2 * padded_w:
                        ph = border // padded_w * (P_out + 1)
                        pw = border % padded_w
                    else:
                        ph = (border - 2 * padded_w) // 2 + 1
                        pw = (border - 2 * padded_w) % 2 * (Q_out + 1)
                values = cutlass.vector.full((8,), 0, cutlass.BFloat16)
                valid_spatial = ph > 0 and ph <= P_out and pw > 0 and pw <= Q_out
                if cutlass.const_expr(previous_frames > 0):
                    if pt < 2 and pt >= 2 - previous_frames and valid_spatial:
                        source = (
                            (
                                (
                                    cutlass.Int64(batch) * previous_frames
                                    + pt
                                    - 2
                                    + previous_frames
                                )
                                * P_out
                                + ph
                                - 1
                            )
                            * Q_out
                            + pw
                            - 1
                        ) * norm_channels + channel
                        values = (prep_previous.iterator.raw_ptr() + source).load(
                            count=8, alignment=16
                        )
                        if pt - 2 >= Z_out - cache_frames:
                            cache_offset = (
                                (
                                    (
                                        cutlass.Int64(batch) * cache_frames
                                        + pt
                                        - 2
                                        - Z_out
                                        + cache_frames
                                    )
                                    * P_out
                                    + ph
                                    - 1
                                )
                                * Q_out
                                + pw
                                - 1
                            ) * norm_channels + channel
                            (prep_cache.iterator.raw_ptr() + cache_offset).store(
                                values, alignment=16
                            )
                destination = (
                    ((cutlass.Int64(batch) * (Z_out + 2) + pt) * padded_h + ph)
                    * padded_w
                    + pw
                ) * norm_channels + channel
                (prep_padded.iterator.raw_ptr() + destination).store(
                    values, alignment=16
                )
                vector_idx += total_ctas * auxiliary_threads
    elif warp_idx >= tma_warp_id:
        if cutlass.const_expr(packed_shape is not None):
            # Four producer warps gather flat TRSC in aligned eight-BF16 vectors.
            # Each writer arrives asynchronously after its copies finish. Writer
            # zero expects packed-weight TMA bytes, so full means A AND B are ready.
            ab_stage_idx = 0
            producer_tid = tidx - 160
            batch_count, input_t, input_h, input_w = packed_shape
            Z_out, P_out, Q_out = zpq
            total_rows = batch_count * Z_out * P_out * Q_out
            while work_tile.is_valid_tile:
                m_start = work_tile.tile_idx[0] * 128
                n_start = work_tile.tile_idx[1] * mma_tiler[1]
                for k in cutlass.range(7, unroll=1):
                    full = ab_full_mbar_ptr.subview(ab_stage_idx)
                    empty = ab_empty_mbar_ptr.subview(ab_stage_idx)
                    while not prims.mbarrier_try_wait_parity(
                        empty, ab_empty_phase_bit, time_limit=10000000
                    ):
                        pass
                    if producer_tid == 0:
                        prims.mbarrier_expect_tx(full, b_bytes_per_stage)
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sB.subview(ab_stage_idx * b_bytes_per_stage),
                            tma_b_desc.get_ptr(),
                            (k * 64, n_start),
                            full,
                            [],
                            group=prims.CTAGroup.CTA_1,
                        )
                    local_k = producer_tid % 8 * 8
                    kk = k * 64 + local_k
                    channel = kk % 16
                    tap = kk // 16
                    dw = tap % 3
                    dh = tap // 3 % 3
                    dt = tap // 9
                    a_ptr = cutlass.inttoptr(
                        sA.data_ptr().toint(),
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    )
                    for chunk in cutlass.range_constexpr(8):
                        local_row = producer_tid // 8 + chunk * 16
                        row = m_start + local_row
                        q = row % Q_out
                        p = row // Q_out % P_out
                        z = row // (Q_out * P_out) % Z_out
                        batch = row // (Q_out * P_out * Z_out)
                        offset = cutlass.Int64(0)
                        source_bytes = cutlass.Int32(0)
                        if row < total_rows and tap < 27:
                            offset = cutlass.Int64(
                                (
                                    ((batch * input_t + z + dt) * input_h + p + dh)
                                    * input_w
                                    + q
                                    + dw
                                )
                                * 16
                                + channel
                            )
                            source_bytes = cutlass.Int32(16)
                        swizzled = local_row * 64 + (local_k ^ (local_row % 8 * 8))
                        prims.cp_async_shared_global(
                            a_ptr + ab_stage_idx * 8192 + swizzled,
                            packed_x.iterator.raw_ptr() + offset,
                            16,
                            "ca",
                            cp_size=source_bytes,
                        )
                    # Each lane contributes one arrival AFTER its asynchronous
                    # copies complete. Their bytes are not part of the TMA count.
                    prims.cp_async_mbarrier_arrive(full, noinc=True)
                    ab_stage_idx += 1
                    if ab_stage_idx == num_ab_stage:
                        ab_stage_idx = 0
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        else:
            ab_stage_idx = 0
            Z_out, P_out, Q_out = zpq
            T_filt, R_filt, S_filt = trs
            pad_d, pad_h, pad_w = lower_pad_dhw
            str_d, str_h, str_w = stride_dhw
            dil_d, dil_h, dil_w = dilation_dhw
            while work_tile.is_valid_tile:
                for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                        channel_half if paired_norm else cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    # M index for this CTA's portion of the M-tile.
                    m_off_cta = (
                        mma_tile_coord_mnl[0] * mma_tiler[0]
                        + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0)
                        * mma_tiler_per_cta_m
                    )
                    # Output (n, z, p, q) from linear M (col-major Q-fastest).
                    n_idx = m_off_cta // (Q_out * P_out * Z_out)
                    rem = m_off_cta % (Q_out * P_out * Z_out)
                    z_idx = rem // (Q_out * P_out)
                    rem = rem % (Q_out * P_out)
                    p_idx = rem // Q_out
                    q_idx = rem % Q_out
                    # Spatial anchors for im2col TMA: idx*stride - pad_lower.
                    w_anchor = q_idx * str_w - pad_w
                    h_anchor = p_idx * str_h - pad_h
                    d_anchor = z_idx * str_d - pad_d
                    # B side: per-CTA N offset (KTRSC tiled descriptor consumes this).
                    n_off_cta = (
                        mma_tile_coord_mnl[1] * mma_tiler[1]
                        + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0)
                        * n_per_cta
                    )

                    # Filter coord (s, r, t) + C-chunk, carried across K-tiles and
                    # advanced colexicographically (s fastest). Reset to origin per
                    # work-tile since each tile sweeps GEMM-K from 0. The carry below
                    # replaces k // (T*R*S) style division.
                    s_idx = 0
                    r_idx = 0
                    t_idx = 0
                    c_chunk_idx = 0
                    for k in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        mbar_full = ab_full_mbar_ptr.subview(ab_stage_idx)
                        mbar_empty = ab_empty_mbar_ptr.subview(ab_stage_idx)

                        # Acquire after the MMA consumer releases this stage.
                        # try_wait_parity issues a single non-blocking attempt that may
                        # hardware-suspend up to time_limit then return False;
                        # a blocking wait must retry in a loop.
                        while not prims.mbarrier_try_wait_parity(
                            mbar_empty, ab_empty_phase_bit, time_limit=10000000
                        ):
                            pass

                        # Filter/C offsets from the carried colex coord (no division).
                        c_off = c_chunk_idx * k_tile_size
                        # Im2col offsets fold dilation in.
                        s_off = s_idx * dil_w
                        r_off = r_idx * dil_h
                        t_off = t_idx * dil_d

                        # Per-stage SMEM slices as cutlass.Array byte offsets: sA/sB are
                        # flat Int8 arrays, stage stride = bytes-per-stage.
                        sA_stage = sA.subview(ab_stage_idx * a_bytes_per_stage)
                        sB_stage = sB.subview(ab_stage_idx * b_bytes_per_stage)

                        if prims.elect_sync():
                            # Group leader sets the byte expectation. In 2cta the
                            # per-group leader (even rank) counts both pair members'
                            # loads (num_tma_copy_bytes doubled, bit-24 routing folds
                            # the peer's complete_tx onto the leader mbar). In 1cta
                            # is_leader_cta is constexpr-True, so every CTA counts its
                            # own (undoubled) bytes on its own mbar.
                            if is_leader_cta:
                                prims.mbarrier_arrive_expect_tx(
                                    mbar_full, num_tma_copy_bytes
                                )
                            # Per-CTA unicast load; _cta_group selects the 1-/2-CTA
                            # tcgen05 group (identical coords/box on both paths).
                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                sA_stage,
                                tma_a_desc.get_ptr(),
                                (c_off, w_anchor, h_anchor, d_anchor, n_idx),
                                mbar_full,
                                [s_off, r_off, t_off],
                                mode=prims.TMALoadMode.IM2COL,
                                group=_cta_group,
                            )
                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                sB_stage,
                                tma_b_desc.get_ptr(),
                                (c_off, s_idx, r_idx, t_idx, n_off_cta),
                                mbar_full,
                                [],
                                group=_cta_group,
                            )

                        ab_stage_idx += 1
                        if ab_stage_idx == num_ab_stage:
                            ab_stage_idx = 0
                            ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                        # Colex carry of (s, r, t, c_chunk): bump s, ripple right on
                        # wrap. Nested single-compare ifs (no boolean-and on traced
                        # predicates) advance the coord over (S, R, T, *).
                        s_idx += 1
                        if s_idx == S_filt:
                            s_idx = 0
                            r_idx += 1
                            if r_idx == R_filt:
                                r_idx = 0
                                t_idx += 1
                                if t_idx == T_filt:
                                    t_idx = 0
                                    c_chunk_idx += 1

                # Advance to next persistent work-tile (static schedule).
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

    # ---- MMA consumer warp ----
    elif warp_idx == mma_warp_id:
        # MMA and all epilogue warps participate in this barrier so the MMA warp
        # can pick up the tmem_ptr written by the allocator (epilogue warp 0).
        # This subset has 160 raw or 288 fused threads, excluding producers.
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)

        # AB / acc pipeline consumer state — tracks the current MMA stage.
        ab_stage_idx = 0
        acc_stage_idx = 0

        # tcgen05 SMEM descriptors for stage 0. tcgen05_mma encodes SMEM
        # addresses in 16-byte units, so per-stage / per-K-block increments
        # are bytes >> 4. (leading=16, stride=1024, layout=2) is the 128B-swizzle
        # K-major layout, matching the cuda.TensorMapSwizzle.s128b on the A/B
        # descriptors. sA/sB are flat cutlass.Arrays, so the SMEM base address is
        # the array itself.
        desc_a_base = prims.Tcgen05SmemDesc.build(
            sA,
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=2,
        )
        desc_b_base = prims.Tcgen05SmemDesc.build(
            sB,
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=2,
        )

        # Per-stage descriptor delta (one full AB stage in SMEM, in 16B units).
        # a_bytes_per_stage / b_bytes_per_stage are already per-stage values,
        # so no division by num_ab_stage here.
        sA_increment_per_stage = a_bytes_per_stage >> 4
        sB_increment_per_stage = b_bytes_per_stage >> 4

        desc_a_cur = desc_a_base
        desc_b_cur = desc_b_base

        # Per-K-block descriptor delta inside one AB stage.
        inc_bytes_per_iter = mma_inst_shape_mnk[2] * ab_dtype.width // 8
        increment = inc_bytes_per_iter >> 4
        # Four K=16 instructions consume each K=64 AB stage.
        num_k_blocks = cute.size(mma_tiler, mode=[2]) // mma_inst_shape_mnk[2]

        # Persistent loop: outer = work tiles (acc stages); inner = K-tiles.
        while work_tile.is_valid_tile:
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                current_acc_stage = acc_stage_idx
                acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
                acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
                current_empty_phase_bit = acc_empty_phase_bit

                acc_stage_idx += 1
                if acc_stage_idx == num_acc_stage:
                    acc_stage_idx = 0
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                if is_leader_cta:
                    # Wait until the previous result has been consumed (epilogue
                    # released this acc stage). No elect_sync — every thread is fine.
                    # try_wait_parity is one non-blocking attempt; loop until it
                    # reports the phase advanced.
                    while not prims.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr_stage,
                        current_empty_phase_bit,
                        time_limit=10000000,
                    ):
                        pass
                    # Order the epilogue's completed TMEM reads before reuse.
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

                    # Per-stage TMEM pointer. A TMEM Pointer holds a packed i32
                    # address token (col in bits [0:16), row in [16:32)); adding an
                    # int advances the raw column id directly, with no element/byte
                    # scaling. Each acc stage owns a full mma_tiler[1]-wide column
                    # band, so the per-stage stride is exactly mma_tiler[1] columns.
                    tmem_ptr_for_mma = (
                        tmem_ptr.data_ptr() + current_acc_stage * mma_tiler[1]
                    )
                    tmem_ptr_curr = cutlass.Array(
                        tmem_ptr_for_mma, dtype=cutlass.Int32, addrspace=6
                    )

                    # scale_d=False (overwrite) only on the very first MMA of this
                    # C-tile's K-loop; True (accumulate) for every subsequent MMA.
                    # We use a runtime expression on (k_idx, kb_idx) instead of a
                    # mutable Python local — the local would freeze at trace time
                    # and re-trigger overwrite on every dynamic outer iteration of
                    # the partially-unrolled K-tile loop.
                    for k_idx in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_full_mbar_ptr_stage = ab_full_mbar_ptr.subview(ab_stage_idx)
                        ab_empty_mbar_ptr_stage = ab_empty_mbar_ptr.subview(
                            ab_stage_idx
                        )

                        # Wait for producer (TMA warp) to fill this AB stage.
                        # try_wait_parity is one non-blocking attempt; loop until the
                        # producer's arrive advances the phase.
                        while not prims.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr_stage,
                            ab_full_phase_bit,
                            time_limit=10000000,
                        ):
                            pass
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

                        # Issue all K-blocks for this AB stage.
                        desc_a_cur_ = desc_a_cur
                        desc_b_cur_ = desc_b_cur
                        for kb_idx in cutlass.range_constexpr(num_k_blocks):
                            scale_d_now = (k_idx > 0) | (kb_idx > 0)
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    mma_kind,
                                    _cta_group,
                                    tmem_ptr_curr,
                                    desc_a_cur_,
                                    desc_b_cur_,
                                    idesc,
                                    scale_d_now,
                                )
                            desc_a_cur_ = desc_a_cur_ + increment
                            desc_b_cur_ = desc_b_cur_ + increment

                        # Advance AB stage state.
                        ab_stage_idx += 1
                        desc_a_cur = desc_a_cur + sA_increment_per_stage
                        desc_b_cur = desc_b_cur + sB_increment_per_stage
                        if ab_stage_idx == num_ab_stage:
                            ab_stage_idx = 0
                            desc_a_cur = desc_a_base
                            desc_b_cur = desc_b_base
                            ab_full_phase_bit = ab_full_phase_bit ^ 1

                        if prims.elect_sync():
                            # Release this AB stage to the TMA producer. In 2cta
                            # _commit_mask = 3 << rank broadcasts to the issuing
                            # group's pair; in 1cta it is None (arrive on own mbar).
                            prims.tcgen05_commit(
                                ab_empty_mbar_ptr_stage,
                                multicast_mask=_commit_mask,
                                group=_cta_group,
                            )

                    if prims.elect_sync():
                        # K-tile loop done — signal accumulator full to epilogue
                        # warps (both CTAs of the group in 2cta; own CTA in 1cta).
                        prims.tcgen05_commit(
                            acc_full_mbar_ptr_stage,
                            multicast_mask=_commit_mask,
                            group=_cta_group,
                        )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Producer tail: drain the remaining acc stages before exit so the
        # kernel doesn't race against epilogue warps still signalling
        # acc_empty after the MMA warp has died.
        tail_stage = acc_stage_idx
        tail_phase = acc_empty_phase_bit
        if is_leader_cta:
            for _ in cutlass.range_constexpr(num_acc_stage - 1):
                tail_stage = tail_stage + 1
                if tail_stage == num_acc_stage:
                    tail_stage = 0
                    tail_phase = tail_phase ^ 1
            if prims.elect_sync():
                while not prims.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10000000,
                ):
                    pass

    # ---- Epilogue warps: raw 0-3, fused 0-7 ----
    elif warp_idx < mma_warp_id:
        # Per-CTA M tile size — 2cta cluster halves mma_tiler[0] across the pair.
        mma_tiler_per_cta_m = (
            mma_tiler[0] // 2 if cutlass.const_expr(use_2cta_instrs) else mma_tiler[0]
        )
        # Raw warps own 32 rows; fused warps own 16 channel-cooperative rows.
        subtile_n = _epi_subtile_n
        subtile_cnt = mma_tiler[1] // subtile_n

        # Sync ids — tmem_bar_id=1 already used by the MMA consumer warp.
        # epilog_sync_bar_id=2 is separate and covers all epilogue warps.
        threads_in_epilogue = 32 * len(epilogue_warp_ids)
        epilog_sync_bar_id = 2
        allocator_warp_id = epilogue_warp_ids[0]
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))

        # 128-bit (16-byte) SMEM store width — store_swizzled vectorizes per lane.
        vsize = 128 // c_dtype.width
        # Keep the same elected lane through every store and the final drain.
        store_issuer = prims.elect_sync()

        # C-side im2col store coords are computed by hand. The im2col STORE op
        # has no im2col-offset operand (unlike the A LOAD), and the C descriptor
        # uses zero corners, so output spatial coords are bare pixels (no pad
        # subtraction, no stride multiply). Decompose the linear per-CTA M index
        # into (n, z, p, q) per work-tile in the loop.
        Z_out, P_out, Q_out = zpq

        if cutlass.const_expr(PREP_SHAPE is not None):
            batches, previous_frames = PREP_SHAPE
            cache_frames = min(2, Z_out + previous_frames)

        # Allocator warp (epi 0) reserves 512 TMEM cols and stashes the pointer
        # in tmem_ptr_i32. Every CTA allocates from its own 512-col bank;
        # _cta_group arranges peer-side state for the 2-CTA group (no-op for
        # CTA_1). This op is warp-collective (NOT elect-safe) and must stay
        # outside any rank-divergent branch, so it is unconditional here.
        if warp_idx == allocator_warp_id:
            prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group=_cta_group)

        # The MMA consumer warp also waits with all epilogue warps, so it
        # picks up the same tmem_ptr right after the allocator publishes it.
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)
        tmem_raw_addr = tmem_ptr_i32.load()

        # Persistent loop: outer over work-tiles (one per acc stage), inner over
        # epilogue subtiles within each tile.
        acc_stage_idx = 0
        epi_stage_idx = 0
        while work_tile.is_valid_tile:
            if cutlass.const_expr(paired_norm):
                # Both N160 halves belong to this M task. Keep BF16 N0 values
                # and contiguous-channel FP32 partials live until N1 arrives.
                retained = cutlass.Array(
                    cutlass.Uint32, 40, space=cutlass.AddressSpace.rmem
                )
                partials = cutlass.Array(
                    cutlass.Float32, 32, space=cutlass.AddressSpace.rmem
                )
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                current_acc_stage = acc_stage_idx
                acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
                acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
                current_full_phase_bit = acc_full_phase_bit

                acc_stage_idx += 1
                if acc_stage_idx == num_acc_stage:
                    acc_stage_idx = 0
                    acc_full_phase_bit = acc_full_phase_bit ^ 1

                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                    channel_half if paired_norm else cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Wait for MMA warp to commit acc_full for this stage.
                # try_wait_parity is one non-blocking attempt; loop until the MMA
                # warp's commit advances the phase.
                while not prims.mbarrier_try_wait_parity(
                    acc_full_mbar_ptr_stage, current_full_phase_bit, time_limit=10000000
                ):
                    pass
                # The MMA completion handshake precedes this warp's TMEM loads.
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

                # Per-stage TMEM column origin. Lower 16 bits of tmem_raw_addr is
                # the column id; upper 16 bits is the row id (always 0 for warp 0).
                base_col_id = (tmem_raw_addr & 0xFFFF) + (
                    current_acc_stage * mma_tiler[1]
                )

                # Per-tile output (im2col store) spatial coords. The 2CTA cluster
                # splits M (the NZPQ pixel axis) across the pair, so the spatial
                # coords carry cta_rank. The K-out channel axis (GEMM N) is NOT
                # split: each CTA's tcgen05 accumulator holds its own M-row band
                # over the full N columns.
                m_off_cta = (
                    mma_tile_coord_mnl[0] * mma_tiler[0]
                    + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0)
                    * mma_tiler_per_cta_m
                )
                # Bare output pixels (col-major Q-fastest). No pad/stride: the C
                # descriptor uses zero corners, so output has no halo concept.
                n_out = m_off_cta // (Q_out * P_out * Z_out)
                rem_m = m_off_cta % (Q_out * P_out * Z_out)
                z_out = rem_m // (Q_out * P_out)
                rem_m = rem_m % (Q_out * P_out)
                p_out = rem_m // Q_out
                q_out = rem_m % Q_out
                # K-out channel base for this N-tile (no cta_rank, see above).
                k_off_base = mma_tile_coord_mnl[1] * mma_tiler[1]

                if cutlass.const_expr(FUSE_NORM):
                    # Residual loads need more latency hiding: sixteen warps own
                    # one row/lane, versus eight warps and two rows for norm-only.
                    rows_per_lane: cutlass.Constexpr = (
                        1 if residual is not None or paired_norm else 2
                    )
                    lane = tidx % 32
                    lane_col = lane % 4
                    # Warp rank modulo four fixes the accessible 32-row TMEM band.
                    # The second warpgroup drains its upper 16 rows.
                    row_base = (warp_idx % 4) * 32 + ((warp_idx // 4) % 2) * 16
                    drain_half = warp_idx // 8
                    if cutlass.const_expr(not paired_norm):
                        retained = cutlass.Array(
                            cutlass.Uint32,
                            20 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                    if cutlass.const_expr(residual is not None):
                        # Drain before global residual traffic. Reuse the same packed
                        # words for raw values, then overwrite them with summed values.
                        for drain_subtile in cutlass.range_constexpr(5):
                            drain_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (
                                base_col_id + drain_subtile * 32
                            )
                            drain_ptr = cutlass.inttoptr(drain_addr, 6, cutlass.Float32)
                            drain_values = prims.tcgen05_ld("16x256b", drain_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            drain_selected = cutlass.Array(
                                cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem
                            )
                            if drain_half == 0:
                                for j in cutlass.range_constexpr(8):
                                    drain_selected[j] = drain_values[
                                        (j // 2) * 4 + j % 2
                                    ].to(cutlass.BFloat16)
                            else:
                                for j in cutlass.range_constexpr(8):
                                    drain_selected[j] = drain_values[
                                        (j // 2) * 4 + j % 2 + 2
                                    ].to(cutlass.BFloat16)
                            retained.store(
                                drain_selected.load(0, 8).bitcast(cutlass.Uint32),
                                (channel_half * 5 + drain_subtile) * 4,
                            )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            drain_leader = (
                                cta_rank_in_cluster // _atom_thr
                            ) * _atom_thr
                            prims.mbarrier_arrive(
                                prims.mapa(acc_empty_mbar_ptr_stage, drain_leader),
                                count=1,
                                scope=prims.MemScope.CLUSTER,
                            )
                    if cutlass.const_expr(not paired_norm):
                        partials = cutlass.Array(
                            cutlass.Float32,
                            32 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                    if cutlass.const_expr(channel_half == 0):
                        for part in cutlass.range_constexpr(32 * rows_per_lane):
                            partials[part] = cutlass.Float32(0.0)
                    for norm_subtile in cutlass.range_constexpr(5):
                        rounded = cutlass.Array(
                            cutlass.BFloat16,
                            8 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                        if cutlass.const_expr(residual is None):
                            norm_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (
                                base_col_id + norm_subtile * 32
                            )
                            norm_ptr = cutlass.inttoptr(norm_addr, 6, cutlass.Float32)
                            norm_rmem = prims.tcgen05_ld("16x256b", norm_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            if cutlass.const_expr(paired_norm):
                                if drain_half == 0:
                                    for j in cutlass.range_constexpr(8):
                                        rounded[j] = norm_rmem[(j // 2) * 4 + j % 2].to(
                                            cutlass.BFloat16
                                        )
                                else:
                                    for j in cutlass.range_constexpr(8):
                                        rounded[j] = norm_rmem[
                                            (j // 2) * 4 + j % 2 + 2
                                        ].to(cutlass.BFloat16)
                            else:
                                rounded.store(norm_rmem.to(cutlass.BFloat16), 0)
                        else:
                            rounded.store(
                                retained.load(
                                    (channel_half * 5 + norm_subtile) * 4, 4
                                ).bitcast(cutlass.BFloat16),
                                0,
                            )
                        math_width = 1 if paired_norm else 2
                        for j in cutlass.range_constexpr(
                            8 * rows_per_lane // math_width
                        ):
                            scalar = j * math_width
                            channel = (
                                channel_half * 160
                                + norm_subtile * 32
                                + (scalar // (2 * rows_per_lane)) * 8
                                + lane_col * 2
                                + scalar % 2
                            )
                            if cutlass.const_expr(paired_norm):
                                bias_pair = (norm_params.data_ptr() + channel).load(
                                    count=math_width, alignment=2 * math_width
                                )
                            else:
                                bias_pair = (
                                    conv_bias.iterator.raw_ptr() + channel
                                ).load(count=2, alignment=4)
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            rounded.store(
                                (value + bias_pair.to(cutlass.Float32)).to(
                                    cutlass.BFloat16
                                ),
                                scalar,
                            )
                        if cutlass.const_expr(residual is not None):
                            for pair in cutlass.range_constexpr(4):
                                skip_row = row_base + lane // 4 + drain_half * 8
                                skip_pixel = cutlass.Int64(m_off_cta) + skip_row
                                if skip_pixel < PREP_SHAPE[0] * Z_out * P_out * Q_out:
                                    channel = (
                                        channel_half * 160
                                        + norm_subtile * 32
                                        + pair * 8
                                        + lane_col * 2
                                    )
                                    offset = skip_pixel * norm_channels + channel
                                    skip = (residual.iterator.raw_ptr() + offset).load(
                                        count=2, alignment=4
                                    )
                                    if cutlass.const_expr(residual_bias is not None):
                                        skip_bias = (
                                            residual_bias.iterator.raw_ptr() + channel
                                        ).load(count=2, alignment=4)
                                        skip = (
                                            skip.to(cutlass.Float32)
                                            + skip_bias.to(cutlass.Float32)
                                        ).to(cutlass.BFloat16)
                                    summed_pair = (
                                        rounded.load(pair * 2, 2).to(cutlass.Float32)
                                        + skip.to(cutlass.Float32)
                                    ).to(cutlass.BFloat16)
                                    rounded.store(summed_pair, pair * 2)
                                    (prep_residual.iterator.raw_ptr() + offset).store(
                                        summed_pair, alignment=4
                                    )
                        for j in cutlass.range_constexpr(
                            8 * rows_per_lane // math_width
                        ):
                            scalar = j * math_width
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            # Torch accumulates c, c+128, c+256 independently
                            # for each of four adjacent channels per virtual lane.
                            # Physical lanes retain two channels from each group
                            # of eight; neighboring pairs complete a virtual lane.
                            channel_group = (
                                (
                                    channel_half * 160
                                    + norm_subtile * 32
                                    + (scalar // (2 * rows_per_lane)) * 8
                                )
                                // 8
                                % 16
                            )
                            part = (
                                ((scalar // 2) % rows_per_lane) * 32
                                + channel_group * 2
                                + scalar % 2
                            )
                            partials.store(
                                partials.load(part, math_width) + value * value, part
                            )
                        retained.store(
                            rounded.load(0, 8 * rows_per_lane).bitcast(cutlass.Uint32),
                            (channel_half * 5 + norm_subtile) * 4 * rows_per_lane,
                        )

                    # All accumulators are now register-owned. Release TMEM before
                    # normalization and stores, allowing MMA to reuse this stage.
                    if cutlass.const_expr(residual is None):
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            leader_cta_rank = (
                                cta_rank_in_cluster // _atom_thr
                            ) * _atom_thr
                            mbar_cluster_ptr = prims.mapa(
                                acc_empty_mbar_ptr_stage, leader_cta_rank
                            )
                            prims.mbarrier_arrive(
                                mbar_cluster_ptr, count=1, scope=prims.MemScope.CLUSTER
                            )

                    # Reconstruct Torch's contiguous-channel warp tree without
                    # changing the four-physical-lanes-per-row TMEM mapping.
                    denominators = cutlass.Array(
                        cutlass.Float32, rows_per_lane, space=cutlass.AddressSpace.rmem
                    )
                    for row_half in cutlass.range_constexpr(rows_per_lane):
                        virtual_lanes = cutlass.Array(
                            cutlass.Float32, 16, space=cutlass.AddressSpace.rmem
                        )
                        for q in cutlass.range_constexpr(16):
                            lo = partials[row_half * 32 + q * 2]
                            hi = partials[row_half * 32 + q * 2 + 1]
                            # Even physical lanes produce ((p0+p1)+p2)+p3.
                            # Odd-lane values are not consumed by the final sum.
                            virtual_lanes[q] = (
                                (lo + hi) + cute.arch.shuffle_sync(lo, lane ^ 1)
                            ) + cute.arch.shuffle_sync(hi, lane ^ 1)
                        for offset in cutlass.range_constexpr(4):
                            step = 8 >> offset
                            for q in cutlass.range_constexpr(step):
                                virtual_lanes[q] = (
                                    virtual_lanes[q] + virtual_lanes[q + step]
                                )
                        total = cute.arch.shuffle_sync(
                            virtual_lanes[0], (lane // 4) * 4
                        ) + cute.arch.shuffle_sync(
                            virtual_lanes[0], (lane // 4) * 4 + 2
                        )
                        denominator = cute.math.sqrt(total, fastmath=False)
                        if denominator < 1e-12:
                            denominator = cutlass.Float32(1e-12)
                        denominators[row_half] = denominator

                if cutlass.const_expr(not paired_norm or channel_half == 1):
                    # Subtile loop on the N axis.
                    for subtile_idx in cutlass.range(
                        subtile_cnt * (2 if paired_norm else 1), unroll_full=FUSE_NORM
                    ):
                        # Rotate through C SMEM stages so the previous TMA store can
                        # drain in parallel with the next t2r/r2s.
                        epi_stage_idx = (epi_stage_idx + 1) % num_c_stage

                        if cutlass.const_expr(not FUSE_NORM):
                            tmem_ctm = prims.make_tmem_ptr_from_warp_row_col(
                                tmem_raw_addr,
                                warp_idx,
                                base_col_id + subtile_idx * subtile_n,
                                cutlass.Float32,
                            )
                            t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ctm, num=32)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

                        if cutlass.const_expr(FUSE_NORM):
                            retained_values = retained.load(
                                subtile_idx * 4 * rows_per_lane, 4 * rows_per_lane
                            ).bitcast(cutlass.BFloat16)
                            fused_rmem = cutlass.Array(
                                c_dtype,
                                8 * rows_per_lane,
                                space=cutlass.AddressSpace.rmem,
                                alignment=4,
                            )
                            math_width = 1 if paired_norm else 2
                            for j in cutlass.range_constexpr(
                                8 * rows_per_lane // math_width
                            ):
                                scalar = j * math_width
                                col = (
                                    subtile_idx * 32
                                    + (scalar // (2 * rows_per_lane)) * 8
                                    + lane_col * 2
                                    + scalar % 2
                                )
                                value = retained_values[
                                    scalar : scalar + math_width
                                ].to(cutlass.Float32)
                                normalized = (
                                    value / denominators[(scalar // 2) % rows_per_lane]
                                ).to(cutlass.BFloat16)
                                scaled = (
                                    normalized.to(cutlass.Float32)
                                    * (norm_channels**0.5)
                                ).to(cutlass.BFloat16)
                                if cutlass.const_expr(paired_norm):
                                    gamma_pair = (
                                        norm_params.data_ptr() + 320 + col
                                    ).load(count=math_width, alignment=2 * math_width)
                                else:
                                    gamma_pair = (gamma.iterator.raw_ptr() + col).load(
                                        count=2, alignment=4
                                    )
                                affine = (
                                    scaled.to(cutlass.Float32)
                                    * gamma_pair.to(cutlass.Float32)
                                ).to(cutlass.BFloat16)
                                value = affine.to(cutlass.Float32)
                                activated = (
                                    value
                                    / (1.0 + cute.math.exp(-value, fastmath=False))
                                ).to(cutlass.BFloat16)
                                fused_rmem.store(activated, scalar)

                        # Raw lanes store four 16B vectors. Fused lanes store eight
                        # adjacent BF16 pairs, distributed across four-lane row groups.
                        smem_tile_base = sC.subview(
                            epi_stage_idx * (_c_tile_rows * _epi_subtile_n)
                        )
                        if cutlass.const_expr(FUSE_NORM):
                            for pair in cutlass.range_constexpr(4 * rows_per_lane):
                                row = (
                                    row_base
                                    + lane // 4
                                    + (pair % rows_per_lane + drain_half) * 8
                                )
                                col = (pair // rows_per_lane) * 8 + lane_col * 2
                                vec_io = fused_rmem[pair * 2 : 2]
                                smem_thr_ptr = smem_tile_base.subview(
                                    row * subtile_n + col
                                )
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=4,
                                    swizzle=cutlass.Swizzle(2, 4, 3),
                                )
                                if cutlass.const_expr(PREP_SHAPE is not None):
                                    pixel = cutlass.Int64(m_off_cta) + row
                                    batch = pixel // (Z_out * P_out * Q_out)
                                    time = pixel // (P_out * Q_out) % Z_out
                                    spatial = pixel % (P_out * Q_out)
                                    if batch < batches and time >= Z_out - cache_frames:
                                        cache_offset = (
                                            (
                                                (
                                                    batch * cache_frames
                                                    + time
                                                    - Z_out
                                                    + cache_frames
                                                )
                                                * P_out
                                                * Q_out
                                                + spatial
                                            )
                                            * norm_channels
                                            + subtile_idx * 32
                                            + col
                                        )
                                        (
                                            prep_cache.iterator.raw_ptr() + cache_offset
                                        ).store(vec_io, alignment=4)
                        else:
                            for j in cutlass.range_constexpr(32 // vsize):
                                vec_io = t2r_rmem[j * vsize : j * vsize + vsize].to(
                                    c_dtype
                                )
                                if cutlass.const_expr(SPATIAL_SHAPE is not None):
                                    spatial_pixel = cutlass.Int64(m_off_cta) + tidx
                                    spatial_channel = (
                                        k_off_base + subtile_idx * subtile_n + j * vsize
                                    )
                                    if (
                                        spatial_pixel
                                        < SPATIAL_SHAPE[0] * Z_out * P_out * Q_out
                                    ):
                                        bias_vec = (
                                            conv_bias.iterator.raw_ptr()
                                            + spatial_channel
                                        ).load(count=vsize, alignment=16)
                                        vec_io = (
                                            vec_io.to(cutlass.Float32)
                                            + bias_vec.to(cutlass.Float32)
                                        ).to(c_dtype)
                                        spatial_skip = (
                                            residual.iterator.raw_ptr()
                                            + spatial_pixel * SPATIAL_SHAPE[1]
                                            + spatial_channel
                                        ).load(count=vsize, alignment=16)
                                        if cutlass.const_expr(
                                            residual_bias is not None
                                        ):
                                            spatial_skip_bias = (
                                                residual_bias.iterator.raw_ptr()
                                                + spatial_channel
                                            ).load(count=vsize, alignment=16)
                                            spatial_skip = (
                                                spatial_skip.to(cutlass.Float32)
                                                + spatial_skip_bias.to(cutlass.Float32)
                                            ).to(c_dtype)
                                        vec_io = (
                                            vec_io.to(cutlass.Float32)
                                            + spatial_skip.to(cutlass.Float32)
                                        ).to(c_dtype)
                                smem_thr_ptr = smem_tile_base.subview(
                                    tidx * subtile_n + j * vsize
                                )
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=16,
                                    swizzle=cutlass.Swizzle(2, 4, 3),
                                )

                        # Make swizzled SMEM stores visible to the TMA proxy.
                        cute.arch.fence_view_async_shared()
                        prims.barrier_cta_sync(
                            epilog_sync_bar_id, thread_count=threads_in_epilogue
                        )

                        # One issuer owns the TMA store, commit, and wait sequence.
                        if warp_idx == epilogue_warp_ids[0]:
                            if store_issuer:
                                k_off = (
                                    0 if paired_norm else k_off_base
                                ) + subtile_idx * subtile_n
                                prims.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_desc.get_ptr(),
                                    smem_tile_base,
                                    (k_off, q_out, p_out, z_out, n_out),
                                    mode=prims.TMAStoreMode.IM2COL,
                                )
                                prims.cp_async_bulk_commit_group()
                                # Bound outstanding reads before the next SMEM reuse.
                                prims.cp_async_bulk_wait_group(
                                    num_c_stage - 1, read=True
                                )

                        prims.barrier_cta_sync(
                            epilog_sync_bar_id, thread_count=threads_in_epilogue
                        )

                # All lanes must order completed TMEM loads before releasing it.
                if cutlass.const_expr(not FUSE_NORM):
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

                # Signal acc_empty back to the group leader's mbarrier. In 2cta the
                # mbar lives in the even-rank leader's SMEM and arrive_count was set
                # to (4 epi warps * 2 CTAs); the per-group leader rank is
                # (rank // _atom_thr) * _atom_thr (degrades to 0 for the single
                # group, 2 for group 1, etc.). In 1cta _atom_thr=1 so this is the
                # CTA's own rank (count 4, own mbar).
                if cutlass.const_expr(not FUSE_NORM) and prims.elect_sync():
                    leader_cta_rank = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                    mbar_cluster_ptr = prims.mapa(
                        acc_empty_mbar_ptr_stage, leader_cta_rank
                    )
                    prims.mbarrier_arrive(
                        mbar_cluster_ptr,
                        count=1,
                        scope=prims.MemScope.CLUSTER,
                    )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # The issuing lane drains global writes, not only SMEM reads.
        if warp_idx == epilogue_warp_ids[0]:
            if store_issuer:
                prims.cp_async_bulk_wait_group(0, read=False)

        # All epilogue readers must finish before the allocator frees TMEM.
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        # tcgen05 dealloc. In 2cta both CTAs of the group must agree the TMEM
        # is no longer in use, so we route an arrival through mapa to the pair
        # partner's tmem_dealloc_mbar (partner = rank ^ 1, correct within each
        # consecutive {2G, 2G+1} pair) and wait on our own before freeing. In
        # 1cta each CTA owns its TMEM independently, so the handshake is elided
        # and we relinquish + dealloc directly.
        if warp_idx == allocator_warp_id:
            prims.tcgen05_relinquish_alloc_permit(group=_cta_group)

            if cutlass.const_expr(use_2cta_instrs):
                peer_cta_rank = cute.arch.make_warp_uniform(cta_rank_in_cluster ^ 1)
                peer_mbar = prims.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                prims.mbarrier_arrive(
                    peer_mbar,
                    count=1,
                    scope=prims.MemScope.CLUSTER,
                )

                # Wait until peer also signalled, then physically free the TMEM.
                # try_wait_parity is one non-blocking attempt; loop until the
                # peer's arrive advances the phase.
                while not prims.mbarrier_try_wait_parity(
                    tmem_dealloc_mbar_ptr, 0, time_limit=10000000
                ):
                    pass

            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=_cta_group)


@dataclass(frozen=True)
class ConvConfig:
    """Padded NTHWC dimensions for the fixed 3x3x3, N160, 2CTA Wan kernel."""

    n: int = 1
    t: int = 4
    h: int = 8
    w: int = 8
    ci: int = 160
    co: int = 160
    ctas: int = 148

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.n, self.t, self.h, self.w, self.ci

    @property
    def packed_weight_shape(self) -> tuple[int, ...]:
        """OTRSC filters with each input-channel row padded to K64."""
        return self.co, 3, 3, 3, (self.ci + 63) // 64 * 64

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.n, self.t - 2, self.h - 2, self.w - 2, self.co

    def validate(self) -> None:
        """Reject geometry outside the encoder's production convolution family."""
        if (self.ci, self.co) not in (
            (160, 160),
            (160, 320),
            (320, 320),
            (320, 640),
            (640, 640),
        ):
            raise ValueError("Unsupported Wan residual-convolution channel pair")
        if min(*self.output_shape, self.ctas) <= 0 or self.ctas % 2:
            raise ValueError("Expected nonempty output and complete 2CTA groups")


class _Launch:
    def __init__(
        self,
        config: ConvConfig,
        previous_frames: int | None = None,
        has_residual: bool = False,
        has_residual_bias: bool = False,
        spatial_output: bool = False,
    ) -> None:
        self.config = config
        self.fuse_norm = previous_frames is not None
        self.previous_frames = previous_frames
        self.has_residual = has_residual
        self.has_residual_bias = has_residual_bias
        self.spatial_output = spatial_output

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"WanConv{'NormSilu' if self.fuse_norm else 'Raw'}_"
            f"{cfg.n}x{cfg.t}x{cfg.h}x{cfg.w}x{cfg.ci}_{cfg.co}_"
            f"2cta_n160_k3_ctas{cfg.ctas}_prev{self.previous_frames}"
            f"_res{self.has_residual}_rb{self.has_residual_bias}"
            f"_spatial{self.spatial_output}"
        )

    @cute.jit
    def __call__(
        self,
        a: _HostTensorMap,
        b: _HostTensorMap,
        c: _HostTensorMap,
        stream: driver.CUstream,
        conv_bias: cute.Tensor = None,
        gamma: cute.Tensor = None,
        prep_padded: cute.Tensor = None,
        prep_cache: cute.Tensor = None,
        prep_previous: cute.Tensor = None,
        residual: cute.Tensor = None,
        residual_bias: cute.Tensor = None,
        prep_residual: cute.Tensor = None,
    ) -> None:
        cfg = self.config
        groups = 2
        tiles = (
            cute.ceil_div(math.prod(cfg.output_shape[:-1]), 128),
            1 if self.fuse_norm else cute.ceil_div(cfg.co, 160),
            1,
        )
        scheduler = PersistentTileSchedulerParams(tiles, (groups, 1, 1))
        grid = StaticPersistentTileScheduler.get_grid_shape(
            scheduler, cfg.ctas // groups
        )
        ab_stages, c_stages = 8, 2
        if cutlass.const_expr(self.fuse_norm and cfg.co == 320 and cfg.ci == 320):
            # The longer K135 convolution benefits from a deeper output-store
            # queue; C160->C320 keeps its measured faster eight/two schedule.
            ab_stages, c_stages = 6, 6
        # Each spatial filter position has its own independently zero-filled
        # channel tail. Flattening TRS*C before ceil-div truncates C=160 work.
        reduction_tiles = 3 * 3 * 3 * ((cfg.ci + 63) // 64)
        kernel(
            "conv_bias_residual_pad"
            if self.spatial_output
            else "conv_residual_norm_prepare"
            if self.has_residual
            else "conv_norm_silu_prepare"
            if self.previous_frames is not None
            else "conv_raw",
            scheduler,
            (128 * groups, 160, 64),
            (128 * groups, 160, 16),
            reduction_tiles,
            ab_stages,
            c_stages,
            True,
            cfg.output_shape[1:4],
            a,
            b,
            c,
            conv_bias,
            gamma,
            prep_padded=prep_padded,
            prep_cache=prep_cache,
            prep_previous=prep_previous,
            PREP_SHAPE=(cfg.n, self.previous_frames)
            if self.previous_frames is not None
            else None,
            residual=residual,
            residual_bias=residual_bias,
            prep_residual=prep_residual,
            SPATIAL_SHAPE=(cfg.n, cfg.co) if self.spatial_output else None,
        ).launch(
            grid=grid,
            block=(
                (704 if cfg.co == 320 else (640 if self.has_residual else 384))
                if self.previous_frames is not None
                else (320 if self.spatial_output else 192),
                1,
                1,
            ),
            min_blocks_per_mp=1
            if self.previous_frames is not None or self.spatial_output
            else 0,
            cluster=(groups, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


def _descriptors(
    cfg: ConvConfig,
    x: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    *,
    prepared_output: bool = False,
    spatial_output: bool = False,
) -> tuple[_HostTensorMap, _HostTensorMap, _HostTensorMap]:
    fake = x is None
    dtype = cutlass.BFloat16
    a_dims, a_strides = _contiguous_tma_layout(cfg.input_shape, dtype)
    b_dims, b_strides = _contiguous_tma_layout(
        cfg.packed_weight_shape if fake else tuple(weight.shape), dtype
    )
    c_dims, c_strides = _contiguous_tma_layout(cfg.output_shape, dtype)
    if prepared_output or spatial_output:
        # Logical T/H/W omit the halo, while physical pitches include it.
        # An IM2COL store can therefore walk 128 logical pixels across rows,
        # frames and batches without ever touching the separately owned halo.
        _, t, h, w, c = cfg.output_shape
        pad_hw = 2 if prepared_output else 1
        pad_t = 2 if prepared_output else 0
        c_strides = (
            c * 2,
            (w + pad_hw) * c * 2,
            (h + pad_hw) * (w + pad_hw) * c * 2,
            (t + pad_t) * (h + pad_hw) * (w + pad_hw) * c * 2,
        )
    a = _create_tensor_map_im2col(
        None if fake else x.data_ptr(),
        dtype,
        a_dims,
        a_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(-2, -2, -2),
        channels_per_pixel=64,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s128b,
        owner=x,
        fake=fake,
    )
    b = _create_tensor_map_tiled(
        None if fake else weight.data_ptr(),
        dtype,
        b_dims,
        b_strides,
        box_dims=(64, 1, 1, 1, 80),
        swizzle=cuda.TensorMapSwizzle.s128b,
        owner=weight,
        fake=fake,
    )
    c = _create_tensor_map_im2col(
        None if fake else output.data_ptr(),
        dtype,
        c_dims,
        c_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s64b,
        owner=output,
        fake=fake,
    )
    return a, b, c


@lru_cache(maxsize=32)
def compile_conv(config: ConvConfig = ConvConfig()) -> Callable:
    """Compile without a GPU/context; artifact dumps require CUTE_DSL_KEEP."""
    config.validate()
    return cute.compile(
        _Launch(config),
        *_descriptors(config),
        driver.CUstream(0),
    )


@lru_cache(maxsize=32)
def compile_prepared(
    config: ConvConfig = ConvConfig(),
    previous_frames: int = 0,
    has_residual: bool = False,
    has_residual_bias: bool = False,
) -> Callable:
    """GPU-free compilation of convolution and next-input/cache production."""
    config.validate()
    if config.co not in (160, 320):
        raise ValueError(
            "The prepared epilogue requires Co in (160, 320) and tile_n=160"
        )
    if previous_frames not in (0, 1, 2):
        raise ValueError("History must contain zero, one or two frames")
    if has_residual_bias and not has_residual:
        raise ValueError("Residual bias requires a residual input")
    parameter = make_fake_compact_tensor(
        cutlass.BFloat16, (config.co,), assumed_align=16
    )
    tensors = tuple(
        make_fake_compact_tensor(
            cutlass.BFloat16, (math.prod(shape),), assumed_align=16
        )
        for shape in _prepared_shapes(config, previous_frames)
    )
    return cute.compile(
        _Launch(
            config,
            previous_frames=previous_frames,
            has_residual=has_residual,
            has_residual_bias=has_residual_bias,
        ),
        *_descriptors(config, prepared_output=True),
        driver.CUstream(0),
        parameter,
        parameter,
        *tensors,
        make_fake_compact_tensor(
            cutlass.BFloat16, (math.prod(config.output_shape),), assumed_align=16
        )
        if has_residual
        else None,
        parameter if has_residual_bias else None,
        make_fake_compact_tensor(
            cutlass.BFloat16, (math.prod(config.output_shape),), assumed_align=16
        )
        if has_residual
        else None,
        options="--ptxas-options=--fmad=false",
    )


@lru_cache(maxsize=32)
def compile_spatial_residual(
    config: ConvConfig = ConvConfig(),
    has_residual_bias: bool = False,
) -> Callable:
    """Compile conv/bias/residual plus bottom/right padding for spatial downsample."""
    config.validate()
    if config.co % 160:
        raise ValueError("Spatial residual fusion requires whole output-channel tiles")
    n, t, h, w, c = config.output_shape
    parameter = make_fake_compact_tensor(cutlass.BFloat16, (c,), assumed_align=16)

    def flat(shape: tuple[int, ...]) -> cute.Tensor:
        return make_fake_compact_tensor(
            cutlass.BFloat16, (math.prod(shape),), assumed_align=16
        )

    return cute.compile(
        _Launch(
            config,
            has_residual=True,
            has_residual_bias=has_residual_bias,
            spatial_output=True,
        ),
        *_descriptors(config, spatial_output=True),
        driver.CUstream(0),
        parameter,
        None,
        flat((n, t, h + 1, w + 1, c)),
        None,
        None,
        flat(config.output_shape),
        parameter if has_residual_bias else None,
        None,
        options="--ptxas-options=--fmad=false",
    )


def _prepared_shapes(
    config: ConvConfig,
    previous_frames: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Padded, new-cache and history argument shapes (one dummy row if absent)."""
    n, t, h, w, c = config.output_shape
    return (
        (n, t + 2, h + 2, w + 2, c),
        (n, min(2, t + previous_frames), h, w, c),
        (n, previous_frames, h, w, c) if previous_frames else (c,),
    )


class PreparedConv:
    """Bound buffers/descriptors and reusable output for graph-safe launches.

    The object retains all tensor allocations. Do not resize or replace their
    storage while this object or a captured graph is in use. Launches follow
    the current Torch CUDA stream; host descriptor creation is outside capture.
    Retain this object until asynchronous work completes, including captured
    graph replay. Calls and output consumption must be serialized: output is
    reused, and concurrent streams are not independently buffered. Autograd is
    unsupported; grad-enabled operands are rejected before DLPack detachment.
    Weights may use logical OTRSC storage or zero-filled K64-padded channel
    rows. Descriptors use their actual shape and retain that storage, without
    hidden packing, so graph replay still sees in-place operand updates.
    """

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        config: ConvConfig,
        conv_bias: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        *,
        prepare_next_input: bool = False,
        previous: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
        spatial_output: bool = False,
    ) -> None:
        config.validate()
        if torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad
            for tensor in (
                x,
                weight,
                conv_bias,
                gamma,
                previous,
                residual,
                residual_bias,
            )
        ):
            raise ValueError(
                "PreparedConv supports inference only; use no_grad or inference_mode"
            )
        weight_shape = tuple(weight.shape)
        if weight_shape != config.packed_weight_shape:
            raise ValueError("Weight must have logical OTRSC or K64-padded OTRSC shape")
        for label, tensor, shape in (
            ("input", x, config.input_shape),
            ("weight", weight, weight_shape),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(f"{label} must be BF16 with shape {shape}")
            if (
                not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{label} must be CUDA, contiguous, and 16-byte aligned"
                )
        if x.device != weight.device:
            raise ValueError("Input and weight must be on the same device")
        if spatial_output and (
            prepare_next_input
            or gamma is not None
            or residual is None
            or conv_bias is None
        ):
            raise ValueError(
                "Spatial residual fusion requires bias/residual "
                "without normalization or temporal preparation"
            )
        if not spatial_output and (conv_bias is None) != (gamma is None):
            raise ValueError("The fused epilogue requires both bias and gamma")
        if conv_bias is not None and not (prepare_next_input or spatial_output):
            raise ValueError("Bias fusion requires prepared or spatial output")
        if prepare_next_input and conv_bias is None:
            raise ValueError("Next-input preparation requires bias and gamma")
        if previous is not None and not prepare_next_input:
            raise ValueError("History is only consumed by next-input preparation")
        if residual is not None and not (prepare_next_input or spatial_output):
            raise ValueError("Residual fusion requires next-input preparation")
        if residual_bias is not None and residual is None:
            raise ValueError("Residual bias requires a residual input")
        for label, tensor, shape in (
            ("residual", residual, config.output_shape),
            ("residual bias", residual_bias, (config.co,)),
        ):
            if tensor is not None and (
                tuple(tensor.shape) != shape
                or tensor.dtype != x.dtype
                or tensor.device != x.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{label} must be aligned contiguous BF16 "
                    f"on the input device, shape {shape}"
                )
        self.residual_owners = (residual, residual_bias)
        self.parameters = ()
        self.parameter_owners = (conv_bias, gamma)
        if conv_bias is not None:
            if not spatial_output and (config.co not in (160, 320)):
                raise ValueError(
                    "The fused epilogue requires Co in (160, 320) and tile_n=160"
                )
            for parameter in (conv_bias, gamma):
                if parameter is None:
                    continue
                if (
                    parameter.shape != (config.co,)
                    or parameter.dtype != x.dtype
                    or parameter.device != x.device
                    or not parameter.is_contiguous()
                    or parameter.data_ptr() % 16
                ):
                    raise ValueError(
                        f"Bias/gamma must be aligned CUDA BF16 vectors of {config.co}"
                    )
            self.parameters = tuple(
                # Checkpoint Parameters keep requires_grad inside inference
                # mode, but DLPack only accepts explicitly detached views.
                from_dlpack(parameter.detach(), assumed_align=16)
                if parameter is not None
                else None
                for parameter in (conv_bias, gamma)
            )
        self.device = x.device
        self.previous = previous
        if prepare_next_input:
            if previous is not None and (
                previous.ndim != 5 or previous.shape[1] not in (1, 2)
            ):
                raise ValueError("History must be NTHWC with one or two frames")
            previous_frames = 0 if previous is None else previous.shape[1]
            padded_shape, cache_shape, previous_shape = _prepared_shapes(
                config, previous_frames
            )
            if previous is not None and (
                tuple(previous.shape) != previous_shape
                or previous.device != x.device
                or previous.dtype != x.dtype
                or not previous.is_contiguous()
                or previous.data_ptr() % 16
            ):
                raise ValueError(
                    "History must be aligned contiguous BF16 NTHWC on the input device"
                )
            padded = torch.empty(padded_shape, device=x.device, dtype=x.dtype)
            cache = torch.empty(cache_shape, device=x.device, dtype=x.dtype)
            summed = (
                torch.empty(config.output_shape, device=x.device, dtype=x.dtype)
                if residual is not None
                else None
            )
            self.output = PreparedConvInput(padded, cache, summed)
            self.descriptors = _descriptors(
                config,
                x,
                weight,
                padded[:, 2:, 1:-1, 1:-1, :],
                prepared_output=True,
            )
            self.parameters += tuple(
                from_dlpack(tensor.detach().view(-1), assumed_align=16)
                for tensor in (
                    padded,
                    cache,
                    conv_bias if previous is None else previous,
                )
            )
            self.parameters += tuple(
                from_dlpack(tensor.detach().view(-1), assumed_align=16)
                if tensor is not None
                else None
                for tensor in (residual, residual_bias, summed)
            )
            self.compiled = compile_prepared(
                config, previous_frames, residual is not None, residual_bias is not None
            )
        elif spatial_output:
            n, t, h, w, c = config.output_shape
            self.output = torch.empty(
                (n, t, h + 1, w + 1, c), device=x.device, dtype=x.dtype
            )
            self.descriptors = _descriptors(
                config, x, weight, self.output, spatial_output=True
            )
            self.parameters += tuple(
                from_dlpack(tensor.detach().view(-1), assumed_align=16)
                if tensor is not None
                else None
                for tensor in (self.output, None, None, residual, residual_bias, None)
            )
            self.compiled = compile_spatial_residual(config, residual_bias is not None)
        else:
            self.output = torch.empty(
                config.output_shape, device=x.device, dtype=x.dtype
            )
            self.descriptors = _descriptors(config, x, weight, self.output)
            self.compiled = compile_conv(config)

    def __call__(self) -> torch.Tensor | PreparedConvInput:
        """Enqueue convolution and return the reused output allocation."""
        with torch.cuda.device(self.device):
            stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            self.compiled(*self.descriptors, stream, *self.parameters)
        return self.output


def prepare(x: torch.Tensor, weight: torch.Tensor, config: ConvConfig) -> PreparedConv:
    """Validate and bind contiguous NTHWC/OTRSC tensors before benchmarking."""
    return PreparedConv(x, weight, config)


def prepare_spatial_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    residual: torch.Tensor,
    config: ConvConfig,
    residual_bias: torch.Tensor | None = None,
) -> PreparedConv:
    """Bind B(B(conv+bias)+B(residual+residual_bias)) and bottom/right padding.

    B denotes separate BF16 rounding, with FP32 arithmetic between casts.
    conv is the already BF16-rounded convolution; absent residual_bias skips
    its addition. Output is a contiguous NTHWC Tensor with one zero bottom row
    and right column, no temporal padding and no normalization.
    """
    return PreparedConv(
        x,
        weight,
        config,
        conv_bias,
        residual=residual,
        residual_bias=residual_bias,
        spatial_output=True,
    )


def prepare_next(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    gamma: torch.Tensor,
    config: ConvConfig,
    previous: torch.Tensor | None = None,
    *,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> PreparedConv:
    """Bind C160/C320 conv/bias/norm/SiLU with causal padding and history outputs.

    Let v = B(conv(x) + bias), where B denotes separate BF16 rounding. With
    a residual, replace v by B(v + B(residual + residual_bias)) and save v
    in output.residual; absent residual_bias skips its addition. Then let
    y = SiLU(WanRMSNorm(v)) and s = concat(previous, y) along time.
    Emit padded[:,2:,...] = y with one zero spatial border,
    padded[:,2-P:2,1:-1,1:-1,:] = previous, and cache = s[:,-min(2,T+P):].
    Missing history is zero padding, not part of the cache. Outputs own fresh
    storage; no input or history is mutated. Retain the bound launch during
    asynchronous execution/graph replay, as for ``prepare``.
    """
    return PreparedConv(
        x,
        weight,
        config,
        conv_bias,
        gamma,
        prepare_next_input=True,
        previous=previous,
        residual=residual,
        residual_bias=residual_bias,
    )


class WanConv3d(torch.nn.Module):
    """Inference-only residual convolution on a padded NTHWC activation.

    Weights enter in Torch OITHW order and are stored once as contiguous OTHWI
    with zero-filled input-channel tails rounded up to K64. This aligns every
    filter row to 128 bytes without changing the K16 accumulation sequence.
    Forward allocates a fresh output, so successive temporal chunks cannot
    overwrite a previous result. Descriptors are built for the current buffers;
    only shape-specialized code is cached. Raw forward defers bias; the prepared
    and spatial-residual methods fuse bias and the corresponding postprocessing.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if (
            weight.ndim != 5
            or tuple(weight.shape[2:]) != (3, 3, 3)
            or (weight.shape[1], weight.shape[0])
            not in ((160, 160), (160, 320), (320, 320), (320, 640), (640, 640))
        ):
            raise ValueError("WanConv3d requires a supported Wan residual 3x3x3 weight")
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        # All residual channel counts divide 160 exactly, avoiding N-tail MMA.
        self.register_buffer(
            "weight",
            torch.nn.functional.pad(
                weight.detach().permute(0, 2, 3, 4, 1).contiguous(),
                (0, (-self.in_channels) % 64),
            ),
            persistent=False,
        )
        multiprocessors = torch.cuda.get_device_properties(
            weight.device
        ).multi_processor_count
        self.ctas = multiprocessors // 2 * 2

    def _config(self, x: torch.Tensor) -> ConvConfig:
        """Validate inference input and derive its shape-specialized launch."""
        if self.training or (torch.is_grad_enabled() and x.requires_grad):
            raise ValueError("WanConv3d supports inference only")
        n, t, h, w, ci = x.shape
        if ci != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {ci}")
        return ConvConfig(
            n=n,
            t=t,
            h=h,
            w=w,
            ci=ci,
            co=self.out_channels,
            ctas=self.ctas,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Launch the raw convolution without retaining activations."""
        config = self._config(x)
        with torch.cuda.device(x.device):
            bound = prepare(x, self.weight, config)
            out = bound()
            stream = torch.cuda.current_stream()
            # A bound launch is temporary here; protect tensor storage on a
            # non-allocation stream until the enqueued GPU work has completed.
            x.record_stream(stream)
            self.weight.record_stream(stream)
            out.record_stream(stream)
        return out

    def forward_prepared(
        self,
        x: torch.Tensor,
        conv_bias: torch.Tensor,
        gamma: torch.Tensor,
        previous: torch.Tensor | None,
        *,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
    ) -> PreparedConvInput:
        """Fuse normalization and produce the next convolution's inputs."""
        config = self._config(x)
        with torch.cuda.device(x.device):
            bound = prepare_next(
                x,
                self.weight,
                conv_bias,
                gamma,
                config,
                previous,
                residual=residual,
                residual_bias=residual_bias,
            )
            out = bound()
            stream = torch.cuda.current_stream()
            for tensor in (
                x,
                self.weight,
                conv_bias,
                gamma,
                previous,
                residual,
                residual_bias,
                out.residual,
                out.padded,
                out.cache,
            ):
                if tensor is not None:
                    tensor.record_stream(stream)
        return out

    def forward_spatial_residual(
        self,
        x: torch.Tensor,
        conv_bias: torch.Tensor,
        residual: torch.Tensor,
        residual_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce the following spatial downsample's padded NHWC input directly."""
        config = self._config(x)
        with torch.cuda.device(x.device):
            bound = prepare_spatial_residual(
                x, self.weight, conv_bias, residual, config, residual_bias
            )
            out = bound()
            stream = torch.cuda.current_stream()
            for tensor in (x, self.weight, conv_bias, residual, residual_bias, out):
                if tensor is not None:
                    tensor.record_stream(stream)
        return out


@dataclass(frozen=True)
class InputConvConfig:
    """Prepared input dimensions; the convolution itself applies no padding."""

    n: int = 1
    t: int = 3
    h: int = 7
    w: int = 9
    ctas: int = 148

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.n, self.t, self.h, self.w, 16

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.n, self.t - 2, self.h - 2, self.w - 2, 160

    def validate(self) -> None:
        """Reject empty outputs and nonpositive launch sizes before compilation."""
        if min(*self.output_shape, self.ctas) <= 0:
            raise ValueError("Input needs positive batch, T/H/W >= 3 and positive CTAs")
        if math.prod(self.input_shape) >= 2**31:
            raise ValueError("Packed gather requires input element offsets below 2**31")


def _input_descriptors(
    cfg: InputConvConfig,
    weight: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> tuple[_HostTensorMap, _HostTensorMap]:
    fake = weight is None
    dims, strides = _contiguous_tma_layout((160, 448), cutlass.BFloat16)
    b = _create_tensor_map_tiled(
        None if fake else weight.data_ptr(),
        cutlass.BFloat16,
        dims,
        strides,
        box_dims=(64, 160),
        swizzle=cuda.TensorMapSwizzle.s128b,
        owner=weight,
        fake=fake,
    )
    dims, strides = _contiguous_tma_layout(cfg.output_shape, cutlass.BFloat16)
    c = _create_tensor_map_im2col(
        None if fake else output.data_ptr(),
        cutlass.BFloat16,
        dims,
        strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s64b,
        owner=output,
        fake=fake,
    )
    return b, c


class _InputLaunch:
    def __init__(self, cfg: InputConvConfig) -> None:
        self.cfg = cfg

    def __repr__(self) -> str:
        return f"WanInputConv_{self.cfg}"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        b: _HostTensorMap,
        c: _HostTensorMap,
        stream: driver.CUstream,
    ) -> None:
        cfg = self.cfg
        tiles = ((math.prod(cfg.output_shape[:-1]) + 127) // 128, 1, 1)
        scheduler = PersistentTileSchedulerParams(tiles, (1, 1, 1))
        grid = StaticPersistentTileScheduler.get_grid_shape(scheduler, cfg.ctas)
        kernel(
            "conv_input_c12",
            scheduler,
            (128, 160, 64),
            (128, 160, 16),
            7,
            5,
            2,
            False,
            cfg.output_shape[1:4],
            b,
            b,
            c,
            packed_x=x,
            packed_shape=(cfg.n, cfg.t, cfg.h, cfg.w),
        ).launch(
            grid=grid,
            block=(288, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


@lru_cache(maxsize=32)
def compile_input(cfg: InputConvConfig = InputConvConfig()) -> Callable:
    """Compile with fake descriptors and no CUDA tensors or context."""
    cfg.validate()
    x = make_fake_compact_tensor(
        cutlass.BFloat16, (math.prod(cfg.input_shape),), assumed_align=16
    )
    return cute.compile(
        _InputLaunch(cfg), x, *_input_descriptors(cfg), driver.CUstream(0)
    )


def pack_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack constant OITHW weights once into zero-tailed O(TRSC) rows."""
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (160, 12, 3, 3, 3):
        raise ValueError("Expected BF16 input-convolution weights [160,12,3,3,3]")
    packed = torch.zeros((160, 28, 16), dtype=weight.dtype, device=weight.device)
    packed[:, :27, :12].copy_(
        weight.detach().permute(0, 2, 3, 4, 1).reshape(160, 27, 12)
    )
    packed = packed.reshape(160, 448)
    return packed


class PreparedInputConv:
    """Owning launch binding. Retain it for asynchronous work and graph replay.

    Each launch reuses output; concurrent calls on different streams are not
    supported. The module wrapper creates a fresh binding/output per forward.
    """

    def __init__(
        self, x: torch.Tensor, weight: torch.Tensor, cfg: InputConvConfig
    ) -> None:
        cfg.validate()
        for tensor, shape in ((x, cfg.input_shape), (weight, (160, 448))):
            if (
                not tensor.is_cuda
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                or tuple(tensor.shape) != shape
                or tensor.device != x.device
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "Expected aligned contiguous BF16 CUDA operands matching config"
                )
        self.x = x
        self.weight = weight
        self.output = torch.empty(cfg.output_shape, dtype=x.dtype, device=x.device)
        with torch.cuda.device(x.device):
            self.descriptors = _input_descriptors(cfg, weight, self.output)
            self.x_dsl = from_dlpack(x.reshape(-1), assumed_align=16)
            self.compiled = compile_input(cfg)

    def __call__(self) -> torch.Tensor:
        with torch.cuda.device(self.x.device):
            self.compiled(
                self.x_dsl,
                *self.descriptors,
                driver.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        return self.output


def prepare_input(
    x: torch.Tensor, weight: torch.Tensor, cfg: InputConvConfig
) -> PreparedInputConv:
    """Bind prepacked weights and the already padded activation."""
    return PreparedInputConv(x, weight, cfg)


class WanInputConv3d(torch.nn.Module):
    """Inference-only C12->C160 module with prepacked constant weights."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", pack_weight(weight), persistent=False)
        self.ctas = torch.cuda.get_device_properties(
            weight.device
        ).multi_processor_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or (torch.is_grad_enabled() and x.requires_grad):
            raise ValueError("WanInputConv3d supports inference only")
        cfg = InputConvConfig(*x.shape[:4], ctas=self.ctas)
        bound = prepare_input(x, self.weight, cfg)
        result = bound()
        stream = torch.cuda.current_stream(x.device)
        for tensor in (x, self.weight, result):
            tensor.record_stream(stream)
        return result


def compile() -> None:
    """Compile raw, normalized, residual-prepared, spatial, and input kernels."""
    compile_input()
    for ci, co in ((160, 160), (160, 320), (320, 320), (320, 640), (640, 640)):
        cfg = ConvConfig(ci=ci, co=co)
        compile_conv(cfg)
        if co <= 320:
            compile_prepared(
                cfg, previous_frames=2, has_residual=True, has_residual_bias=co == 320
            )
        if ci == co:
            compile_spatial_residual(cfg)


@torch.inference_mode()
def verify() -> None:
    """Check small cases without timing them, including history and partial CTAs.

    Raw convolution uses an FP32 Torch oracle (TF32 off). Fused preparation
    must match separate primitives exactly, including padding, cache, and skips.
    Performance is measured separately at production shapes.
    """
    # These are the actual separate kernels, not duplicated reference kernels.
    if __package__:
        from .norm import rmsnorm_silu_conv_prep
        from .residual import torch_bias_residual
    else:
        from norm import rmsnorm_silu_conv_prep
        from residual import torch_bias_residual

    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for ci, co in ((160, 160), (160, 320), (320, 320), (320, 640), (640, 640)):
        # These reduced batches are correctness checks, never performance results.
        frames, height, width = (
            (4, 320, 240) if co == 160 else (2, 160, 120) if co == 320 else (1, 80, 60)
        )
        cfg = ConvConfig(
            n=1,
            t=frames + 2,
            h=height + 2,
            w=width + 2,
            ci=ci,
            co=co,
            ctas=sm_count // 2 * 2,
        )
        x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
        torch_weight = (
            torch.randn(co, ci, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1
        ).contiguous(memory_format=torch.channels_last_3d)
        module = WanConv3d(torch_weight).eval()
        raw = prepare(x, module.weight, cfg)
        torch_x = x.permute(0, 4, 1, 2, 3)

        fp32 = torch.nn.functional.conv3d(
            torch_x.float(), torch_weight.float()
        ).permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(raw().float(), fp32, atol=0.02, rtol=0.02)
        bias = torch.randn(co, device="cuda", dtype=x.dtype) * 0.1
        gamma = torch.randn_like(bias)
        residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype)
        if co <= 320:
            for history, with_residual in ((0, False), (1, False), (2, True)):
                previous = (
                    torch.randn(
                        cfg.n,
                        history,
                        cfg.h - 2,
                        cfg.w - 2,
                        co,
                        device="cuda",
                        dtype=x.dtype,
                    )
                    if history
                    else None
                )
                kwargs = dict(
                    residual=residual if with_residual else None,
                    residual_bias=bias if with_residual and co == 320 else None,
                )
                fused = prepare_next(
                    x, module.weight, bias, gamma, cfg, previous, **kwargs
                )

                def separate() -> PreparedConvInput:
                    return rmsnorm_silu_conv_prep(
                        raw(), gamma, previous=previous, input_bias=bias, **kwargs
                    )

                actual, expected = fused(), separate()
                torch.testing.assert_close(
                    actual.padded, expected.padded, atol=0, rtol=0
                )
                torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
                if with_residual:
                    torch.testing.assert_close(
                        actual.residual, expected.residual, atol=0, rtol=0
                    )
        if ci == co:
            fused_spatial = prepare_spatial_residual(
                x, module.weight, bias, residual, cfg
            )
            expected = torch_bias_residual(raw(), residual, bias, pad_spatial=True)
            torch.testing.assert_close(fused_spatial(), expected, atol=0, rtol=0)
        # A 35-pixel output leaves the entire second CTA inactive for stores.
        # That peer must still complete every paired-MMA handshake.
        tail_cfg = ConvConfig(t=3, h=7, w=9, ci=ci, co=co, ctas=cfg.ctas)
        tail_x = torch.randn(tail_cfg.input_shape, device="cuda", dtype=x.dtype) * 0.1
        tail_raw = prepare(tail_x, module.weight, tail_cfg)
        tail_ref = torch.nn.functional.conv3d(
            tail_x.permute(0, 4, 1, 2, 3).float(), torch_weight.float()
        ).permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(tail_raw().float(), tail_ref, atol=0.02, rtol=0.02)
        if co <= 320:
            history = torch.randn(1, 2, 5, 7, co, device="cuda", dtype=x.dtype)
            for with_residual in (False, True):
                skip = (
                    torch.randn(tail_cfg.output_shape, device="cuda", dtype=x.dtype)
                    if with_residual
                    else None
                )
                tail_fused = prepare_next(
                    tail_x, module.weight, bias, gamma, tail_cfg, history, residual=skip
                )
                actual = tail_fused()
                expected = rmsnorm_silu_conv_prep(
                    tail_raw(), gamma, previous=history, input_bias=bias, residual=skip
                )
                torch.testing.assert_close(
                    actual.padded, expected.padded, atol=0, rtol=0
                )
                torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
                if with_residual:
                    torch.testing.assert_close(
                        actual.residual, expected.residual, atol=0, rtol=0
                    )
        print(f"C{ci}->{co}: correctness PASS (including partial CTA pair)", flush=True)
    cfg = InputConvConfig(n=1, t=6, h=322, w=242, ctas=sm_count)
    x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    x[..., 12:] = 0
    weight = (
        torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1
    ).contiguous(memory_format=torch.channels_last_3d)
    raw = prepare_input(x, pack_weight(weight), cfg)
    torch_x = x[..., :12].contiguous().permute(0, 4, 1, 2, 3)

    def reference() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    torch.testing.assert_close(raw(), reference(), atol=0.02, rtol=0.02)
    print("Convolution family: PASS (raw, fused, history, residual, input)", flush=True)


@torch.inference_mode()
def _benchmark_conv(cfg: ConvConfig, history: int, spatial: bool) -> None:
    """Time one production callsite; release its large buffers before the next."""
    if __package__:
        from .norm import rmsnorm_silu_conv_prep, torch_reference
        from .residual import bias_residual, torch_bias_residual
    else:
        from norm import rmsnorm_silu_conv_prep, torch_reference
        from residual import bias_residual, torch_bias_residual

    print(
        f"\nC{cfg.ci}->C{cfg.co} | output NTHWC={cfg.output_shape} | history={history}",
        flush=True,
    )
    x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = (
        torch.randn(cfg.co, cfg.ci, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1
    ).contiguous(memory_format=torch.channels_last_3d)
    module = WanConv3d(weight).eval()
    raw = prepare(x, module.weight, cfg)
    torch_x = x.permute(0, 4, 1, 2, 3)

    def torch_conv() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    torch.testing.assert_close(raw(), torch_conv(), atol=0.02, rtol=0.02)
    torch_results = [measure("conv", torch_conv, raw)]
    fusion_results = []
    bias = torch.randn(cfg.co, device="cuda", dtype=x.dtype) * 0.1
    if cfg.co <= 320:
        gamma = torch.randn_like(bias)
        previous = (
            torch.randn(
                cfg.n,
                history,
                cfg.h - 2,
                cfg.w - 2,
                cfg.co,
                device="cuda",
                dtype=x.dtype,
            )
            if history
            else None
        )
        # The channel-transition conv uses norm-only fusion in the model.
        for with_residual in (False, True) if cfg.ci == cfg.co else (False,):
            residual = (
                torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype)
                if with_residual
                else None
            )
            kwargs = dict(
                residual=residual,
                residual_bias=bias if with_residual and cfg.co == 320 else None,
            )
            fused = prepare_next(x, module.weight, bias, gamma, cfg, previous, **kwargs)

            def separate() -> PreparedConvInput:
                return rmsnorm_silu_conv_prep(
                    raw(), gamma, previous=previous, input_bias=bias, **kwargs
                )

            def reference() -> PreparedConvInput:
                return torch_reference(
                    torch_conv(), gamma, previous=previous, input_bias=bias, **kwargs
                )

            actual, expected = fused(), separate()
            torch.testing.assert_close(actual.padded, expected.padded, atol=0, rtol=0)
            torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
            if with_residual:
                torch.testing.assert_close(
                    actual.residual, expected.residual, atol=0, rtol=0
                )
            del actual, expected
            operation = (
                "conv + residual + norm/SiLU/prep"
                if with_residual
                else "conv + norm/SiLU/prep"
            )
            torch_results.append(measure(operation, reference, fused))
            fusion_results.append(measure(operation, separate, fused))
        del fused, previous, residual

    if spatial:
        residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype)
        fused_spatial = prepare_spatial_residual(x, module.weight, bias, residual, cfg)
        torch.testing.assert_close(
            fused_spatial(),
            torch_bias_residual(raw(), residual, bias, pad_spatial=True),
            atol=0,
            rtol=0,
        )
        operation = "conv + residual + spatial pad"
        torch_results.append(
            measure(
                operation,
                lambda: torch_bias_residual(
                    torch_conv(), residual, bias, pad_spatial=True
                ),
                fused_spatial,
            )
        )
        fusion_results.append(
            measure(
                operation,
                lambda: bias_residual(raw(), residual, bias, pad_spatial=True),
                fused_spatial,
            )
        )

    print("Against Torch eager/cuDNN:", flush=True)
    print_benchmark_table(torch_results)
    if fusion_results:
        print(
            "Fusion only (custom=fused; reference=our conv + separate processing):",
            flush=True,
        )
        print_benchmark_table(fusion_results, reference_name="Separate custom")


@torch.inference_mode()
def _benchmark_input(cfg: InputConvConfig) -> None:
    """Time the production C12 input convolution with prepacked operands."""
    print(f"\nC12->C160 input | output NTHWC={cfg.output_shape}", flush=True)
    x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    x[..., 12:] = 0
    weight = (
        torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1
    ).contiguous(memory_format=torch.channels_last_3d)
    raw = prepare_input(x, pack_weight(weight), cfg)
    torch_x = x[..., :12].contiguous().permute(0, 4, 1, 2, 3)

    def reference() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    torch.testing.assert_close(raw(), reference(), atol=0.02, rtol=0.02)
    print_benchmark_table([measure("input conv", reference, raw)])


def benchmark_production() -> None:
    """Measure the batch-32 encoder's initial and steady-state chunk shapes.

    C160/C320 stages retain four warm frames; the temporal downsamplers reduce
    C640 at 80x60 to two frames and C640 at 40x30 to one. All initial chunks have
    one frame. Shape reduction belongs only in verify(), never in these timings.
    """
    torch.manual_seed(44)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        "\nProduction benchmark: batch=32, BF16, prepared channels-last operands."
        "\nMedian GPU ms; 5 alternating samples x 20 CUDA-graph replays."
        "\nCompilation, weight packing, and input preparation are excluded."
        "\nTorch reference is eager, not torch.compile; speedup=reference/custom.",
        flush=True,
    )
    for ci, co, frames, height, width in (
        (160, 160, 4, 320, 240),
        (160, 320, 4, 160, 120),
        (320, 320, 4, 160, 120),
        (320, 640, 2, 80, 60),
        (640, 640, 2, 80, 60),
        (640, 640, 1, 40, 30),
    ):
        for current_frames in sorted({1, frames}):
            cfg = ConvConfig(
                n=32,
                t=current_frames + 2,
                h=height + 2,
                w=width + 2,
                ci=ci,
                co=co,
                ctas=sm_count // 2 * 2,
            )
            _benchmark_conv(
                cfg,
                history=0 if current_frames == 1 else 2,
                spatial=ci == co and height >= 80,
            )
    for frames in (1, 4):
        _benchmark_input(
            InputConvConfig(n=32, t=frames + 2, h=322, w=242, ctas=sm_count)
        )
    print("\nProduction benchmark: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-compile", action="store_true")
    args = parser.parse_args()
    if args.only_compile:
        compile()
        print("Convolution family compile: PASS", flush=True)
    else:
        print("Correctness checks (small cases, untimed):", flush=True)
        verify()
        benchmark_production()
