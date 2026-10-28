# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false, reportMissingImports=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportPossiblyUnboundVariable=false, reportReturnType=false

"""Pipelined Wan convolution using the public CUTLASS DSL runtime.

BF16 preserves the rounding contract below. FP32 supports strict library
convolution (ieee) and ordinary single-product TF32 convolution (tf32).
Experimental tf32x3 explicitly selects high/low components and three products.
Both Tensor Core FP32 policies use short MMA reductions with periodic SIMT
FP32 sums to bound accumulation error. TF32 uses K32 logical channels;
TF32x3 uses K16 in K32 component storage. See _precision.py for packing.

For contiguous NTHWC input X and OTRSC filters W, this computes

    Y[n,t,h,w,o] = BF16(sum_{dt,dh,dw,c}
        FP32(X[n,t+dt,h+dh,w+dw,c]) * FP32(W[o,dt,dh,dw,c]))

Input already includes causal/spatial padding. The filter is 3x3x3;
stride and dilation are one, groups is one. Wan 2.1 convolutions use N96
tiles for C96 outputs and N192 tiles for C192/C384 outputs. Wan 2.2 encoder
convolutions use N160 tiles. Decoder convolutions use N256 tiles for 512->256,
256->256, and 48->1024.
C96/C192/C256 support fused normalization and next-input preparation.
``prepare`` returns this raw convolution. ``prepare_next`` additionally applies
the C96/C160/C192/C256/C320 postprocessing below, with B denoting a separate BF16 rounding:

    v_c = B(FP32(Y_c) + FP32(conv_bias_c))
    d = max(sqrt(sum_c(FP32(v_c)**2)), 1e-12)
    u_c = B(FP32(v_c) / d)
    a_c = B(FP32(B(FP32(u_c) * sqrt(Co))) * FP32(gamma_c))
    out_c = B(FP32(a_c) / (1 + exp(-FP32(a_c))))

TMA performs implicit im2col directly
into staged shared memory; no global im2col or tensor transpose is allocated.
The persistent device pipeline has independent producer, MMA, and epilogue
warps with persistent M-major scheduling. BF16 uses two TMEM accumulator
stages; FP32 uses one stage with partial and running-sum bands.
Fused C320->C320 uses six operand stages
and six SMEM output-store stages. ``prepare_next`` stores activated output
directly into the next convolution's padded NTHWC input and emits current
cache frames in the same launch. Auxiliary warps copy existing history and
write zero padding concurrently with convolution. Their output regions are
disjoint from the epilogue's current-frame and cache writes.
With a residual branch it replaces v_c above by
B(FP32(v_c) + FP32(B(FP32(residual_c) + FP32(residual_bias_c)))), omitting
the inner addition when residual_bias is absent. This summed value is also
saved as the next block's skip input unless ``save_residual=False``. No raw
convolution output is materialized.
``prepare_spatial_residual`` applies that bias/residual sum without normalization
and writes a zero bottom row/right column for spatial downsampling. This
pointwise path supports independent output-channel tiles. Normalization keeps
all channels of a row in one CTA group: one N160 tile for C160, two sequential
N160 tiles for C320. C320 requires 2CTA MMA; both widths support residual addition.
C96/C192/C256 each use one full-channel tile with the same rounding and
input/cache semantics. C96 preserves Torch's scalar-lane reduction order;
the wider channels use its vectorized reduction order.

C12 BF16 input uses a 1CTA gather producer and seven K64 tiles over C16-padded
activations and [160,448] weights. General convolutions use TMA im2col with
27*ceil(Ci/64) reduction tiles for BF16. FP32 C12 input uses the general TMA
path over zero-padded C16 input for TF32, with 27*ceil(Ci/32) tiles;
strict FP32 uses cuDNN on native C12.
BF16 paths use two TMEM accumulators; epilogue warps
drain them into 64B-swizzled SMEM for TMA stores. Halo/cache writers are
disjoint from the epilogue's interior writers. A padded peer CTA must still
participate in every paired-MMA handshake, including the final M tail.

The CUTLASS host wrapper builds public tiled and im2col tensor maps directly
from the bound tensor arguments. Descriptors travel as GridConstant arguments;
module forwards retain their operands, allocate fresh outputs, and record the
active CUDA stream.

Running this file benchmarks both Wan versions. Use --model wan21 or
--model wan22 to select one; --verify-only and --only-compile use the same
selection for BF16 checks. Benchmarks accept --dtype bf16/fp32 and
--allow-tf32/--no-allow-tf32. Total timings include per-launch activation
conversion; external padding, weight packing, and compilation are excluded.
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import cuda
from cutlass.experimental import primitives as prims

if __package__:
    from ._compile import compile_cache, compile_kernel
    from ._conv_config import (
        _DECODER_CHANNEL_PAIRS,
        _ENCODER_CHANNEL_PAIRS,
        _WAN21_CHANNEL_PAIRS,
        ConvConfig,
        ConvSpecialization,
        LaunchPolicy,
        specialization,
    )
    from ._conv_config import needs_wide_index as _needs_wide_index
    from ._precision import pack_components, resolve_precision
    from ._scheduler import TileSchedule
    from ._utils import PreparedConvInput, measure, measure_group, print_benchmark_table, round_tf32_reference
else:
    from _compile import compile_cache, compile_kernel
    from _conv_config import (
        _DECODER_CHANNEL_PAIRS,
        _ENCODER_CHANNEL_PAIRS,
        _WAN21_CHANNEL_PAIRS,
        ConvConfig,
        ConvSpecialization,
        LaunchPolicy,
        specialization,
    )
    from _conv_config import needs_wide_index as _needs_wide_index
    from _precision import pack_components, resolve_precision
    from _scheduler import TileSchedule
    from _utils import PreparedConvInput, measure, measure_group, print_benchmark_table, round_tf32_reference


def _contiguous_tma_layout(
    shape: tuple[int, ...], dtype: type[cutlass.Numeric]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convert a C-contiguous shape to TMA dimensions and 16-byte strides.

    NDHWC becomes C,W,H,D,N; KTRSC becomes C,S,R,T,K.
    """
    dims = shape[::-1]
    stride_bytes = cutlass.Int64(dtype.width // 8)
    strides = []
    for extent in dims[:-1]:
        stride_bytes *= extent
        strides.append(stride_bytes // 16)
    return dims, tuple(strides)


@cute.jit
def _flat_nthwc_offset(n, t, h, w, c, frames, height, width, channels):
    """Promote before multiplication, even on the 32-bit coordinate fast path."""
    return (((cutlass.Int64(n) * frames + t) * height + h) * width + w) * channels + c


@cute.jit
def _zero_prepared_padding(
    padded,
    vector_idx,
    vector_stride,
    zpq,
    shape,
    channels: cutlass.Constexpr,
    *,
    padded_plane_div,
    padded_width_div,
    auxiliary_rows_div,
    border_rows_div,
    wide_index: cutlass.Constexpr,
):
    """Zero missing history and spatial borders, leaving existing history intact."""
    z, h, w = zpq
    batches, previous_frames = shape
    padded_h, padded_w = h + 2, w + 2
    history_rows = (2 - previous_frames) * padded_h * padded_w
    border_rows = 2 * padded_w + 2 * h
    auxiliary_rows = history_rows + (z + previous_frames) * border_rows
    index_type: cutlass.Constexpr = cutlass.Int64 if wide_index else cutlass.Int32
    while vector_idx < batches * auxiliary_rows * (channels // 8):
        coordinate_idx = index_type(vector_idx)
        channel = coordinate_idx % (channels // 8) * 8
        auxiliary_row = coordinate_idx // (channels // 8)
        batch = auxiliary_row // auxiliary_rows_div
        local_row = auxiliary_row % auxiliary_rows_div
        pt, ph, pw = index_type(0), index_type(0), index_type(0)
        if local_row < history_rows:
            pt = local_row // padded_plane_div
            ph = (local_row % padded_plane_div) // padded_width_div
            pw = local_row % padded_width_div
        else:
            border = (local_row - history_rows) % border_rows_div
            pt = (local_row - history_rows) // border_rows_div + 2 - previous_frames
            if border < 2 * padded_w:
                ph = border // padded_width_div * (h + 1)
                pw = border % padded_width_div
            else:
                ph = (border - 2 * padded_w) // 2 + 1
                pw = (border - 2 * padded_w) % 2 * (w + 1)
        destination = _flat_nthwc_offset(batch, pt, ph, pw, channel, z + 2, padded_h, padded_w, channels)
        (padded.iterator.raw_ptr() + destination).store(cutlass.vector.full((8,), 0, padded.element_type), alignment=16)
        vector_idx += vector_stride


@cute.jit
def _copy_prepared_history(
    previous,
    padded,
    cache,
    vector_idx,
    vector_stride,
    zpq,
    shape,
    channels: cutlass.Constexpr,
    *,
    plane_div,
    width_div,
    history_div,
    wide_index: cutlass.Constexpr,
):
    """Copy old frames in auxiliary warps while the MMA/epilogue run.

    History occupies temporal slots before the epilogue's current output. Only
    a one-frame output retains an old frame in the new cache, also disjoint
    from current-cache writes. Each thread copies aligned 16-byte vectors.
    """
    z, h, w = zpq
    batches, previous_frames = shape
    cache_frames = min(2, z + previous_frames)
    old_cache_frames = max(0, cache_frames - z)
    vectors = cutlass.Int64(batches) * previous_frames * h * w * (channels // 8)
    index_type: cutlass.Constexpr = cutlass.Int64 if wide_index else cutlass.Int32
    while vector_idx < vectors:
        coordinate_idx = index_type(vector_idx)
        channel = coordinate_idx % (channels // 8) * 8
        pixel = coordinate_idx // (channels // 8)
        pw = pixel % width_div
        ph = (pixel % plane_div) // width_div
        pt = (pixel // plane_div) % history_div
        batch = (pixel // plane_div) // history_div
        values = (previous.iterator.raw_ptr() + vector_idx * 8).load(count=8, alignment=16)
        padded_offset = _flat_nthwc_offset(
            batch, 2 - previous_frames + pt, ph + 1, pw + 1, channel, z + 2, h + 2, w + 2, channels
        )
        (padded.iterator.raw_ptr() + padded_offset).store(values, alignment=16)
        if old_cache_frames > 0:  # noqa: SIM102 - avoid accessing history when no old frame is retained
            if pt >= previous_frames - old_cache_frames:
                cache_t = pt - (previous_frames - old_cache_frames)
                cache_offset = _flat_nthwc_offset(batch, cache_t, ph, pw, channel, cache_frames, h, w, channels)
                (cache.iterator.raw_ptr() + cache_offset).store(values, alignment=16)
        vector_idx += vector_stride


@cute.kernel
def kernel(
    # Leading specialization label remains visible before DSL name truncation.
    operation: cutlass.Constexpr[str],
    # Persistent tile scheduler parameters
    schedule: cutlass.Constexpr[TileSchedule],
    # Constexpr knobs
    mma_tiler: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_shape_mnk: cutlass.Constexpr[tuple[int, int, int]],
    # Host-computed: T*R*S*ceil_div(Ci, 64), including each channel tail.
    k_tile_cnt: cutlass.Constexpr[int],
    policy: cutlass.Constexpr[LaunchPolicy],
    use_2cta_instrs: cutlass.Constexpr[bool],
    # Conv geometry consumed by the im2col TMA producer / epilogue:
    # zpq      = output spatial dims (Z, P, Q) used to decompose linear M -> (n,z,p,q)
    # trs      = filter spatial dims (T, R, S) used to decompose K-tile index
    # lower_pad_dhw / stride_dhw / dilation_dhw = standard conv params
    zpq,
    # Host-built TMA descriptors, consumed by the device-side
    # prims.cp_async_bulk_tensor_* / prims.prefetch_tensormap calls.
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    storage: cutlass.Constexpr = cutlass.BFloat16,
    conv_bias: cute.Tensor = None,
    gamma: cute.Tensor = None,
    packed_x: cute.Tensor = None,
    packed_shape=None,
    prep_padded: cute.Tensor = None,
    prep_cache: cute.Tensor = None,
    PREP_SHAPE=None,
    residual: cute.Tensor = None,
    residual_bias: cute.Tensor = None,
    prep_residual: cute.Tensor = None,
    SPATIAL_SHAPE=None,
    prep_history: cute.Tensor = None,
    runtime_clusters_m=None,
    volume_div=None,
    plane_div=None,
    width_div=None,
    padded_plane_div=None,
    padded_width_div=None,
    auxiliary_rows_div=None,
    border_rows_div=None,
    history_div=None,
    spatial_border_div=None,
    wide_index: cutlass.Constexpr = False,
    compensated: cutlass.Constexpr = False,
) -> None:
    """Run TMA -> tcgen05 MMA -> dtype-aware TMA store; see module contract.

    Tile sizes, channel layout and pipeline stages are specialized. N/T/H/W
    and history length are runtime values for the TMA path. The host prepares
    tensor maps and fast divisors without reading or synchronizing GPU data.
    """
    num_ab_stage: cutlass.Constexpr = policy.ab_stages
    num_c_stage: cutlass.Constexpr = policy.c_stages
    FUSE_NORM: cutlass.Constexpr = PREP_SHAPE is not None
    if cutlass.const_expr(volume_div is None):
        volume_div, plane_div, width_div = zpq[0] * zpq[1] * zpq[2], zpq[1] * zpq[2], zpq[2]
    ab_dtype = storage
    c_dtype = storage
    fp32: cutlass.Constexpr = storage == cutlass.Float32
    bands: cutlass.Constexpr = 2 if fp32 else 1
    retained_type: cutlass.Constexpr = cutlass.Float32 if fp32 else cutlass.Uint32
    retention_scale: cutlass.Constexpr = 2 if fp32 else 1
    acc_dtype = cutlass.Float32
    num_acc_stage = 1 if fp32 else 2
    # Bound Tensor Core summation error while retaining full-channel tiles.
    # Promotion reuses the epilogue warps and the same two TMEM bands.
    promotion_tiles: cutlass.Constexpr = 64
    promotion_count: cutlass.Constexpr = (k_tile_cnt + promotion_tiles - 1) // promotion_tiles if fp32 else 1
    trs = (3, 3, 3)
    lower_pad_dhw = (0, 0, 0)
    stride_dhw = (1, 1, 1)
    dilation_dhw = (1, 1, 1)
    if cutlass.const_expr(mma_tiler[0] != (256 if use_2cta_instrs else 128)):
        raise ValueError("Convolution core requires 128 rows per CTA")
    if cutlass.const_expr(mma_tiler[1] not in (96, 160, 192, 256)):
        raise ValueError("Wan convolution uses tile N=96, N=160, N=192 or N=256")
    if cutlass.const_expr(mma_tiler[2] != (32 if fp32 else 64)):
        raise ValueError("Convolution core requires BF16 K64 or FP32-component K32")
    if cutlass.const_expr(mma_inst_shape_mnk != (mma_tiler[0], mma_tiler[1], 8 if fp32 else 16)):
        raise ValueError("MMA instruction K must be 16 for BF16 or 8 for TF32")
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
            raise ValueError("Packed gather requires C12 input and unpadded unit-stride 3x3x3")
    if cutlass.const_expr(FUSE_NORM):
        if cutlass.const_expr(conv_bias is None or gamma is None):
            raise ValueError("Fused normalization requires conv_bias and gamma")
        if cutlass.const_expr(conv_bias.element_type != c_dtype or gamma.element_type != c_dtype):
            raise TypeError("Fused conv_bias and gamma must match the convolution storage dtype")
        if cutlass.const_expr(
            conv_bias.shape not in ((96,), (160,), (192,), (256,), (320,)) or gamma.shape != conv_bias.shape
        ):
            raise ValueError("Fused conv_bias and gamma require 96, 160, 192, 256, or 320 channels")
        if cutlass.const_expr(mma_tiler[1] != (160 if gamma.shape[0] == 320 else gamma.shape[0])):
            raise ValueError("Normalization tile width must match the channel family")
    if cutlass.const_expr(PREP_SHAPE is not None):
        if cutlass.const_expr(not FUSE_NORM or packed_shape is not None):
            raise ValueError("Next-convolution preparation requires normalization")
        if cutlass.const_expr(prep_padded is None or prep_cache is None):
            raise ValueError("Preparation requires padded and cache tensors")
    if cutlass.const_expr(residual is not None):  # noqa: SIM102 - preserve DSL specialization guard
        if cutlass.const_expr(SPATIAL_SHAPE is None and PREP_SHAPE is None):
            raise ValueError("Residual fusion requires prepared or spatial output")
    if cutlass.const_expr(
        SPATIAL_SHAPE is not None
        and (FUSE_NORM or PREP_SHAPE is not None or residual is None or conv_bias is None or prep_padded is None)
    ):
        raise ValueError("Spatial residual output requires bias/residual and output without norm/history")

    norm_channels: cutlass.Constexpr = gamma.shape[0] if FUSE_NORM else 160
    paired_norm: cutlass.Constexpr = FUSE_NORM and norm_channels == 2 * mma_tiler[1]
    if cutlass.const_expr(paired_norm and not use_2cta_instrs):
        raise ValueError("Two-tile normalization requires 2CTA MMA")

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
    _cta_group = prims.CTAGroup.CTA_2 if cutlass.const_expr(use_2cta_instrs) else prims.CTAGroup.CTA_1
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

    epilogue_warp_ids = tuple(range(policy.epilogue_warps))
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
    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
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
            c_dtype,
            2 * norm_channels,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        if tidx < norm_channels // 2:
            norm_params.store(
                (conv_bias.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                tidx * 2,
            )
            norm_params.store(
                (gamma.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                norm_channels + tidx * 2,
            )
        # Publish ordinary SMEM loads independently of the async TMA barriers.
        prims.barrier_cta_sync(
            3,
            thread_count=policy.threads,
        )

    # mbarrier_init: one warp / one lane initializes every barrier.
    # Each epilogue warp in each cooperating CTA releases the accumulator.
    num_acc_empty_arrives = len(epilogue_warp_ids) * (2 if use_2cta_instrs else 1)
    if warp_idx == 0 and prims.elect_sync():
        prims.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
        for i in cutlass.range_constexpr(num_acc_stage):
            prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_acc_empty_arrives)
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
    mma_kind = prims.Tcgen05MMAKind.TF32 if fp32 else prims.Tcgen05MMAKind.F16

    # Persistent tile scheduler.
    work_index = cute.arch.block_idx()[2]
    work_stride = cute.arch.grid_dim()[2]
    tile_idx, valid_tile = schedule.get_work(work_index, runtime_clusters_m, wide_index)

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
    producer_warp_end = policy.auxiliary_start_warp
    auxiliary_threads = 32 * policy.auxiliary_warps
    if cutlass.const_expr(policy.helper_registers):  # noqa: SIM102 - static selection before warp-role branch
        if warp_idx >= mma_warp_id:
            prims.setmaxregister(policy.helper_registers, "decrease")
    if warp_idx >= producer_warp_end:
        if cutlass.const_expr(SPATIAL_SHAPE is not None):
            batches, channels = SPATIAL_SHAPE
            z, h, w = zpq
            border_rows = h + w + 1
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (cutlass.Int64(by) + cutlass.Int64(gy) * bz)
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            while vector_idx < batches * z * border_rows * (channels // 8):
                channel = vector_idx % (channels // 8) * 8
                border_row = (cutlass.Int64 if wide_index else cutlass.Int32)(vector_idx // (channels // 8))
                nt = border_row // spatial_border_div
                border = border_row % spatial_border_div
                ph = (cutlass.Int64 if wide_index else cutlass.Int32)(h)
                pw = border
                if border >= w + 1:
                    ph = border - w - 1
                    pw = (cutlass.Int64 if wide_index else cutlass.Int32)(w)
                offset = _flat_nthwc_offset(nt, 0, ph, pw, channel, 1, h + 1, w + 1, channels)
                zeros = cutlass.vector.full((8,), 0, c_dtype)
                (prep_padded.iterator.raw_ptr() + offset).store(zeros, alignment=16)
                vector_idx += total_ctas * auxiliary_threads
        if cutlass.const_expr(PREP_SHAPE is not None):
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            # The persistent scheduler places clusters along grid Z, not X.
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (cutlass.Int64(by) + cutlass.Int64(gy) * bz)
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            _zero_prepared_padding(
                prep_padded,
                vector_idx,
                total_ctas * auxiliary_threads,
                zpq,
                PREP_SHAPE,
                norm_channels,
                padded_plane_div=padded_plane_div,
                padded_width_div=padded_width_div,
                auxiliary_rows_div=auxiliary_rows_div,
                border_rows_div=border_rows_div,
                wide_index=wide_index,
            )
            if cutlass.const_expr(prep_history is not None):
                _copy_prepared_history(
                    prep_history,
                    prep_padded,
                    prep_cache,
                    vector_idx,
                    total_ctas * auxiliary_threads,
                    zpq,
                    PREP_SHAPE,
                    norm_channels,
                    plane_div=plane_div,
                    width_div=width_div,
                    history_div=history_div,
                    wide_index=wide_index,
                )
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
            while valid_tile:
                m_start = tile_idx[0] * 128
                n_start = tile_idx[1] * mma_tiler[1]
                for k in cutlass.range(7, unroll=1):
                    full = ab_full_mbar_ptr.subview(ab_stage_idx)
                    empty = ab_empty_mbar_ptr.subview(ab_stage_idx)
                    while not prims.mbarrier_try_wait_parity(empty, ab_empty_phase_bit, time_limit=10000000):
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
                        c_dtype,
                    )
                    for chunk in cutlass.range_constexpr(8):
                        local_row = producer_tid // 8 + chunk * 16
                        row = m_start + local_row
                        batch = row // volume_div
                        remainder = row % volume_div
                        z = remainder // plane_div
                        remainder = remainder % plane_div
                        p = remainder // width_div
                        q = remainder % width_div
                        offset = cutlass.Int64(0)
                        source_bytes = cutlass.Int32(0)
                        if row < total_rows and tap < 27:
                            offset = cutlass.Int64(
                                (((batch * input_t + z + dt) * input_h + p + dh) * input_w + q + dw) * 16 + channel
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
                work_index += work_stride
                tile_idx, valid_tile = schedule.get_work(work_index, runtime_clusters_m, wide_index)
        else:
            ab_stage_idx = 0
            Z_out, P_out, Q_out = zpq
            T_filt, R_filt, S_filt = trs
            pad_d, pad_h, pad_w = lower_pad_dhw
            str_d, str_h, str_w = stride_dhw
            dil_d, dil_h, dil_w = dilation_dhw
            while valid_tile:
                for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                    cur_tile_coord = tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                        channel_half if paired_norm else cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    # M index for this CTA's portion of the M-tile.
                    m_off_cta = (
                        mma_tile_coord_mnl[0] * mma_tiler[0]
                        + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * mma_tiler_per_cta_m
                    )
                    # Output (n, z, p, q) from linear M (col-major Q-fastest).
                    n_idx = m_off_cta // volume_div
                    rem = m_off_cta % volume_div
                    z_idx = rem // plane_div
                    rem = rem % plane_div
                    p_idx = rem // width_div
                    q_idx = rem % width_div
                    # Spatial anchors for im2col TMA: idx*stride - pad_lower.
                    w_anchor = q_idx * str_w - pad_w
                    h_anchor = p_idx * str_h - pad_h
                    d_anchor = z_idx * str_d - pad_d
                    # B side: per-CTA N offset (KTRSC tiled descriptor consumes this).
                    n_off_cta = (
                        mma_tile_coord_mnl[1] * mma_tiler[1]
                        + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * n_per_cta
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
                        while not prims.mbarrier_try_wait_parity(mbar_empty, ab_empty_phase_bit, time_limit=10000000):
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
                                prims.mbarrier_arrive_expect_tx(mbar_full, num_tma_copy_bytes)
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

                work_index += work_stride
                tile_idx, valid_tile = schedule.get_work(work_index, runtime_clusters_m, wide_index)

    # ---- MMA consumer warp ----
    elif warp_idx == mma_warp_id:
        # MMA and all epilogue warps participate in this barrier so the MMA warp
        # can pick up the tmem_ptr written by the allocator (epilogue warp 0).
        # Producers, scheduler, and padding warps do not participate.
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
        while valid_tile:
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                current_acc_stage = acc_stage_idx
                acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
                acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
                current_empty_phase_bit = acc_empty_phase_bit

                acc_stage_idx += 1
                if acc_stage_idx == num_acc_stage:
                    acc_stage_idx = 0
                    acc_empty_phase_bit = acc_empty_phase_bit ^ (promotion_count % 2)

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
                    tmem_ptr_for_mma = tmem_ptr.data_ptr() + current_acc_stage * mma_tiler[1] * bands
                    tmem_ptr_curr = cutlass.Array(tmem_ptr_for_mma, dtype=cutlass.Int32, addrspace=6)

                    # BF16 overwrites only at the first MMA of an output tile.
                    # FP32 also restarts after each promoted partial sum.
                    # We use a runtime expression on (k_idx, kb_idx) instead of a
                    # mutable Python local — the local would freeze at trace time
                    # and re-trigger overwrite on every dynamic outer iteration of
                    # the partially-unrolled K-tile loop.
                    for k_idx in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_full_mbar_ptr_stage = ab_full_mbar_ptr.subview(ab_stage_idx)
                        ab_empty_mbar_ptr_stage = ab_empty_mbar_ptr.subview(ab_stage_idx)

                        # Wait for producer (TMA warp) to fill this AB stage.
                        # try_wait_parity is one non-blocking attempt; loop until the
                        # producer's arrive advances the phase.
                        while not prims.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr_stage,
                            ab_full_phase_bit,
                            time_limit=10000000,
                        ):
                            pass
                        if cutlass.const_expr(packed_shape is not None):
                            cute.arch.fence_view_async_shared()
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

                        # Issue all K-blocks for this AB stage.
                        desc_a_cur_ = desc_a_cur
                        desc_b_cur_ = desc_b_cur
                        for kb_idx in cutlass.range_constexpr(num_k_blocks // (2 if compensated else 1)):
                            scale_d_now = ((k_idx % promotion_tiles if fp32 else k_idx) > 0) | (kb_idx > 0)
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
                                if cutlass.const_expr(compensated):
                                    prims.tcgen05_mma(
                                        mma_kind,
                                        _cta_group,
                                        tmem_ptr_curr,
                                        desc_a_cur_,
                                        desc_b_cur_ + 4,
                                        idesc,
                                        True,
                                    )
                                    prims.tcgen05_mma(
                                        mma_kind, _cta_group, tmem_ptr_curr, desc_a_cur_ + 4, desc_b_cur_, idesc, True
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

                        if cutlass.const_expr(fp32):  # noqa: SIM102 - exclude promotion from BF16 code
                            if (k_idx + 1) % promotion_tiles == 0 or k_idx + 1 == k_tile_cnt:
                                if prims.elect_sync():
                                    prims.tcgen05_commit(
                                        acc_full_mbar_ptr_stage, multicast_mask=_commit_mask, group=_cta_group
                                    )
                                if k_idx + 1 < k_tile_cnt:
                                    current_empty_phase_bit = current_empty_phase_bit ^ 1
                                    while not prims.mbarrier_try_wait_parity(
                                        acc_empty_mbar_ptr_stage, current_empty_phase_bit, time_limit=10000000
                                    ):
                                        pass
                                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if cutlass.const_expr(not fp32):  # noqa: SIM102 - preserve compile-time selection
                        if prims.elect_sync():
                            # K-tile loop done — signal accumulator full to epilogue
                            # warps (both CTAs of the group in 2cta; own CTA in 1cta).
                            prims.tcgen05_commit(
                                acc_full_mbar_ptr_stage,
                                multicast_mask=_commit_mask,
                                group=_cta_group,
                            )

            work_index += work_stride
            tile_idx, valid_tile = schedule.get_work(work_index, runtime_clusters_m, wide_index)

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
        if cutlass.const_expr(policy.epilogue_registers):
            prims.setmaxregister(policy.epilogue_registers, "increase")
        # Per-CTA M tile size — 2cta cluster halves mma_tiler[0] across the pair.
        mma_tiler_per_cta_m = mma_tiler[0] // 2 if cutlass.const_expr(use_2cta_instrs) else mma_tiler[0]
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
        while valid_tile:
            if cutlass.const_expr(paired_norm):
                # Both channel halves belong to this M task. Keep BF16 N0 values
                # and contiguous-channel FP32 partials live until N1 arrives.
                retained = cutlass.Array(
                    retained_type, subtile_cnt * 8 * retention_scale, space=cutlass.AddressSpace.rmem
                )
                partials = cutlass.Array(cutlass.Float32, 32, space=cutlass.AddressSpace.rmem)
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                current_acc_stage = acc_stage_idx
                acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
                acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
                current_full_phase_bit = acc_full_phase_bit

                acc_stage_idx += 1
                if acc_stage_idx == num_acc_stage:
                    acc_stage_idx = 0
                    acc_full_phase_bit = acc_full_phase_bit ^ (promotion_count % 2)

                cur_tile_coord = tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                    channel_half if paired_norm else cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                base_col_id = (tmem_raw_addr & 0xFFFF) + current_acc_stage * mma_tiler[1] * bands
                # BF16 performs one ordinary accumulator wait. FP32 repeatedly
                # sums short MMA reductions with SIMT round-to-nearest adds.
                # Only four warps own the 128 TMEM rows; every epilogue warp
                # joins the barrier before releasing the partial band to MMA.
                for promotion in cutlass.range(promotion_count, unroll=1):
                    while not prims.mbarrier_try_wait_parity(
                        acc_full_mbar_ptr_stage, current_full_phase_bit, time_limit=10000000
                    ):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if cutlass.const_expr(fp32):
                        if warp_idx < 4:
                            for part in cutlass.range_constexpr(mma_tiler[1] // 32):
                                partial_ptr = prims.make_tmem_ptr_from_warp_row_col(
                                    tmem_raw_addr, warp_idx, base_col_id + part * 32, cutlass.Float32
                                )
                                sum_ptr = prims.make_tmem_ptr_from_warp_row_col(
                                    tmem_raw_addr, warp_idx, base_col_id + mma_tiler[1] + part * 32, cutlass.Float32
                                )
                                promotion_value = prims.tcgen05_ld("32x32b", partial_ptr, num=32)
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                if promotion > 0:
                                    promotion_previous = prims.tcgen05_ld("32x32b", sum_ptr, num=32)
                                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                    promotion_value = promotion_value + promotion_previous
                                prims.tcgen05_st("32x32b", sum_ptr, promotion_value)
                            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if promotion + 1 < promotion_count:
                            if prims.elect_sync():
                                promote_empty = prims.mapa(acc_empty_mbar_ptr_stage, cta_rank_in_cluster & -2)
                                prims.mbarrier_arrive(promote_empty, count=1, scope=prims.MemScope.CLUSTER)
                            current_full_phase_bit = current_full_phase_bit ^ 1
                if cutlass.const_expr(fp32):
                    base_col_id = base_col_id + mma_tiler[1]

                # Per-tile output (im2col store) spatial coords. The 2CTA cluster
                # splits M (the NZPQ pixel axis) across the pair, so the spatial
                # coords carry cta_rank. The K-out channel axis (GEMM N) is NOT
                # split: each CTA's tcgen05 accumulator holds its own M-row band
                # over the full N columns.
                m_off_cta = (
                    mma_tile_coord_mnl[0] * mma_tiler[0]
                    + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * mma_tiler_per_cta_m
                )
                # Bare output pixels (col-major Q-fastest). No pad/stride: the C
                # descriptor uses zero corners, so output has no halo concept.
                n_out = m_off_cta // volume_div
                rem_m = m_off_cta % volume_div
                z_out = rem_m // plane_div
                rem_m = rem_m % plane_div
                p_out = rem_m // width_div
                q_out = rem_m % width_div
                # K-out channel base for this N-tile (no cta_rank, see above).
                k_off_base = mma_tile_coord_mnl[1] * mma_tiler[1]

                if cutlass.const_expr(FUSE_NORM):
                    # Sixteen epilogue warps own one row/lane; eight own two.
                    rows_per_lane: cutlass.Constexpr = 16 // len(epilogue_warp_ids)
                    lane = tidx % 32
                    lane_col = lane % 4
                    # Warp rank modulo four fixes the accessible 32-row TMEM band.
                    # The second warpgroup drains its upper 16 rows.
                    row_base = (warp_idx % 4) * 32 + ((warp_idx // 4) % 2) * 16
                    drain_half = warp_idx // 8
                    if cutlass.const_expr(not paired_norm):
                        retained = cutlass.Array(
                            retained_type,
                            subtile_cnt * 4 * rows_per_lane * retention_scale,
                            space=cutlass.AddressSpace.rmem,
                        )
                    if cutlass.const_expr(residual is not None):
                        # Drain before global residual traffic. Reuse the same packed
                        # words for raw values, then overwrite them with summed values.
                        for drain_subtile in cutlass.range_constexpr(subtile_cnt):
                            drain_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (base_col_id + drain_subtile * 32)
                            drain_ptr = cutlass.inttoptr(drain_addr, 6, cutlass.Float32)
                            drain_values = prims.tcgen05_ld("16x256b", drain_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            drain_selected = cutlass.Array(c_dtype, 8, space=cutlass.AddressSpace.rmem)
                            for j in cutlass.range_constexpr(8):
                                index = (j // 2) * 4 + j % 2
                                selected = drain_values[index] if drain_half == 0 else drain_values[index + 2]
                                drain_selected[j] = selected.to(c_dtype)
                            retained.store(
                                drain_selected.load(0, 8).bitcast(retained_type),
                                (channel_half * subtile_cnt + drain_subtile) * 4 * retention_scale,
                            )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            drain_leader = (cta_rank_in_cluster // _atom_thr) * _atom_thr
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
                    for norm_subtile in cutlass.range_constexpr(subtile_cnt):
                        rounded = cutlass.Array(
                            c_dtype,
                            8 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                        if cutlass.const_expr(residual is None):
                            norm_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (base_col_id + norm_subtile * 32)
                            norm_ptr = cutlass.inttoptr(norm_addr, 6, cutlass.Float32)
                            norm_rmem = prims.tcgen05_ld("16x256b", norm_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            if cutlass.const_expr(rows_per_lane == 1):
                                for j in cutlass.range_constexpr(8):
                                    index = (j // 2) * 4 + j % 2
                                    selected = norm_rmem[index] if drain_half == 0 else norm_rmem[index + 2]
                                    rounded[j] = selected.to(c_dtype)
                            else:
                                rounded.store(norm_rmem.to(c_dtype), 0)
                        else:
                            rounded.store(
                                retained.load(
                                    (channel_half * subtile_cnt + norm_subtile) * 4 * retention_scale,
                                    4 * retention_scale,
                                ).bitcast(c_dtype),
                                0,
                            )
                        math_width = 1 if norm_channels == 320 else 2
                        for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                            scalar = j * math_width
                            channel = (
                                channel_half * mma_tiler[1]
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
                                bias_pair = (conv_bias.iterator.raw_ptr() + channel).load(count=2, alignment=4)
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            rounded.store(
                                (value + bias_pair.to(cutlass.Float32)).to(c_dtype),
                                scalar,
                            )
                        if cutlass.const_expr(residual is not None):
                            # N96 benefits from full-sector residual transactions.
                            # Exchange adjacent-lane BF16 pairs to move four values
                            # per lane; preserve the TMEM/reduction lane mapping.
                            # Wider tiles retain the cheaper pairwise instruction path.
                            if cutlass.const_expr(mma_tiler[1] == 96 and not fp32):
                                skip_row = row_base + lane // 4 + drain_half * 8
                                skip_pixel = cutlass.Int64(m_off_cta) + skip_row
                                if skip_pixel < PREP_SHAPE[0] * Z_out * P_out * Q_out:
                                    for group in cutlass.range_constexpr(2):
                                        channel = (
                                            channel_half * mma_tiler[1]
                                            + norm_subtile * 32
                                            + group * 16
                                            + (lane_col % 2) * 8
                                            + (lane_col // 2) * 4
                                        )
                                        offset = skip_pixel * norm_channels + channel
                                        skip_quad = (residual.iterator.raw_ptr() + offset).load(count=4, alignment=8)
                                        if cutlass.const_expr(residual_bias is not None):
                                            bias_quad = (residual_bias.iterator.raw_ptr() + channel).load(
                                                count=4, alignment=8
                                            )
                                            skip_quad = (
                                                skip_quad.to(cutlass.Float32) + bias_quad.to(cutlass.Float32)
                                            ).to(c_dtype)
                                        # A two-lane mask keeps tail-row predicates safe.
                                        words = skip_quad.bitcast(cutlass.Uint32)
                                        neighbor0 = cute.arch.shuffle_sync(
                                            words[0], lane ^ 1, mask=cutlass.Uint32(3) << ((lane // 2) * 2)
                                        )
                                        neighbor1 = cute.arch.shuffle_sync(
                                            words[1], lane ^ 1, mask=cutlass.Uint32(3) << ((lane // 2) * 2)
                                        )
                                        pairs = cutlass.Array(cutlass.Uint32, 2, space=cutlass.AddressSpace.rmem)
                                        if lane_col % 2 == 0:
                                            pairs[0] = words[0]
                                            pairs[1] = neighbor0
                                        else:
                                            pairs[0] = neighbor1
                                            pairs[1] = words[1]
                                        for part in cutlass.range_constexpr(2):
                                            skip = pairs.load(part, 1).bitcast(c_dtype)
                                            summed_pair = (
                                                rounded.load(group * 4 + part * 2, 2).to(cutlass.Float32)
                                                + skip.to(cutlass.Float32)
                                            ).to(c_dtype)
                                            rounded.store(summed_pair, group * 4 + part * 2)
                                        residual_lo = rounded.load(group * 4, 2).bitcast(cutlass.Uint32)[0]
                                        residual_hi = rounded.load(group * 4 + 2, 2).bitcast(cutlass.Uint32)[0]
                                        residual_neighbor_lo = cute.arch.shuffle_sync(
                                            residual_lo, lane ^ 1, mask=cutlass.Uint32(3) << ((lane // 2) * 2)
                                        )
                                        residual_neighbor_hi = cute.arch.shuffle_sync(
                                            residual_hi, lane ^ 1, mask=cutlass.Uint32(3) << ((lane // 2) * 2)
                                        )
                                        if lane_col % 2 == 0:
                                            pairs[0] = residual_lo
                                            pairs[1] = residual_neighbor_lo
                                        else:
                                            pairs[0] = residual_neighbor_hi
                                            pairs[1] = residual_hi
                                        if cutlass.const_expr(prep_residual is not None):
                                            (prep_residual.iterator.raw_ptr() + offset).store(
                                                pairs.load(0, 2).bitcast(c_dtype), alignment=8
                                            )
                            else:
                                for pair in cutlass.range_constexpr(4):
                                    skip_row = row_base + lane // 4 + drain_half * 8
                                    skip_pixel = cutlass.Int64(m_off_cta) + skip_row
                                    if skip_pixel < PREP_SHAPE[0] * Z_out * P_out * Q_out:
                                        channel = (
                                            channel_half * mma_tiler[1] + norm_subtile * 32 + pair * 8 + lane_col * 2
                                        )
                                        offset = skip_pixel * norm_channels + channel
                                        skip = (residual.iterator.raw_ptr() + offset).load(count=2, alignment=4)
                                        if cutlass.const_expr(residual_bias is not None):
                                            skip_bias = (residual_bias.iterator.raw_ptr() + channel).load(
                                                count=2, alignment=4
                                            )
                                            skip = (skip.to(cutlass.Float32) + skip_bias.to(cutlass.Float32)).to(
                                                c_dtype
                                            )
                                        summed_pair = (
                                            rounded.load(pair * 2, 2).to(cutlass.Float32) + skip.to(cutlass.Float32)
                                        ).to(c_dtype)
                                        rounded.store(summed_pair, pair * 2)
                                        if cutlass.const_expr(prep_residual is not None):
                                            (prep_residual.iterator.raw_ptr() + offset).store(summed_pair, alignment=4)
                        for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                            scalar = j * math_width
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            # Torch accumulates c, c+128, c+256 independently
                            # for each of four adjacent channels per virtual lane.
                            # Physical lanes retain two channels from each group
                            # of eight; neighboring pairs complete a virtual lane.
                            if cutlass.const_expr(norm_channels == 96):
                                # Scalar Torch lanes accumulate c, c+32, c+64.
                                # Each physical lane owns two channels per eight.
                                channel_group = scalar // (2 * rows_per_lane)
                            else:
                                channel_group = (
                                    (
                                        channel_half * mma_tiler[1]
                                        + norm_subtile * 32
                                        + (scalar // (2 * rows_per_lane)) * 8
                                    )
                                    // 8
                                    % 16
                                )
                            part = ((scalar // 2) % rows_per_lane) * 32 + channel_group * 2 + scalar % 2
                            partials.store(partials.load(part, math_width) + value * value, part)
                        retained.store(
                            rounded.load(0, 8 * rows_per_lane).bitcast(retained_type),
                            (channel_half * subtile_cnt + norm_subtile) * 4 * rows_per_lane * retention_scale,
                        )

                    # All accumulators are now register-owned. Release TMEM before
                    # normalization and stores, allowing MMA to reuse this stage.
                    if cutlass.const_expr(residual is None):
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            leader_cta_rank = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                            mbar_cluster_ptr = prims.mapa(acc_empty_mbar_ptr_stage, leader_cta_rank)
                            prims.mbarrier_arrive(mbar_cluster_ptr, count=1, scope=prims.MemScope.CLUSTER)

                    # Reconstruct Torch's contiguous-channel warp tree without
                    # changing the four-physical-lanes-per-row TMEM mapping.
                    denominators = cutlass.Array(cutlass.Float32, rows_per_lane, space=cutlass.AddressSpace.rmem)
                    for row_half in cutlass.range_constexpr(rows_per_lane):
                        if cutlass.const_expr(norm_channels == 96):
                            sums = cutlass.Array(cutlass.Float32, 2, space=cutlass.AddressSpace.rmem)
                            for part in cutlass.range_constexpr(2):
                                base = row_half * 32 + part
                                # Virtual shuffle-down offsets 16 and 8 are
                                # local; offsets 4 and 2 cross physical lanes.
                                channel_sum = (partials[base] + partials[base + 4]) + (
                                    partials[base + 2] + partials[base + 6]
                                )
                                channel_sum = channel_sum + cute.arch.shuffle_sync(channel_sum, lane ^ 2)
                                channel_sum = channel_sum + cute.arch.shuffle_sync(channel_sum, lane ^ 1)
                                sums[part] = channel_sum
                            total = cute.arch.shuffle_sync(sums[0] + sums[1], (lane // 4) * 4)
                        else:
                            virtual_lanes = cutlass.Array(cutlass.Float32, 16, space=cutlass.AddressSpace.rmem)
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
                                    virtual_lanes[q] = virtual_lanes[q] + virtual_lanes[q + step]
                            total = cute.arch.shuffle_sync(virtual_lanes[0], (lane // 4) * 4) + cute.arch.shuffle_sync(
                                virtual_lanes[0], (lane // 4) * 4 + 2
                            )
                        denominator = cute.math.sqrt(total, fastmath=False)
                        if denominator < 1e-12:
                            denominator = cutlass.Float32(1e-12)
                        denominators[row_half] = denominator

                if cutlass.const_expr(not paired_norm or channel_half == 1):
                    # Subtile loop on the N axis.
                    for subtile_idx in cutlass.range(subtile_cnt * (2 if paired_norm else 1), unroll_full=FUSE_NORM):
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
                                subtile_idx * 4 * rows_per_lane * retention_scale, 4 * rows_per_lane * retention_scale
                            ).bitcast(c_dtype)
                            fused_rmem = cutlass.Array(
                                c_dtype,
                                8 * rows_per_lane,
                                space=cutlass.AddressSpace.rmem,
                                alignment=4,
                            )
                            math_width = 1 if norm_channels == 320 else 2
                            for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                                scalar = j * math_width
                                col = subtile_idx * 32 + (scalar // (2 * rows_per_lane)) * 8 + lane_col * 2 + scalar % 2
                                value = retained_values[scalar : scalar + math_width].to(cutlass.Float32)
                                normalized = (value / denominators[(scalar // 2) % rows_per_lane]).to(c_dtype)
                                scaled = (normalized.to(cutlass.Float32) * (norm_channels**0.5)).to(c_dtype)
                                if cutlass.const_expr(paired_norm):
                                    gamma_pair = (norm_params.data_ptr() + norm_channels + col).load(
                                        count=math_width, alignment=2 * math_width
                                    )
                                else:
                                    gamma_pair = (gamma.iterator.raw_ptr() + col).load(count=2, alignment=4)
                                affine = (scaled.to(cutlass.Float32) * gamma_pair.to(cutlass.Float32)).to(c_dtype)
                                value = affine.to(cutlass.Float32)
                                activated = (value / (1.0 + cute.math.exp(-value, fastmath=False))).to(c_dtype)
                                fused_rmem.store(activated, scalar)

                        # Raw lanes store four 16B vectors. Fused lanes store eight
                        # adjacent BF16 pairs, distributed across four-lane row groups.
                        smem_tile_base = sC.subview(epi_stage_idx * (_c_tile_rows * _epi_subtile_n))
                        if cutlass.const_expr(FUSE_NORM):
                            for pair in cutlass.range_constexpr(4 * rows_per_lane):
                                row = row_base + lane // 4 + (pair % rows_per_lane + drain_half) * 8
                                col = (pair // rows_per_lane) * 8 + lane_col * 2
                                vec_io = fused_rmem[pair * 2 : 2]
                                smem_thr_ptr = smem_tile_base.subview(row * subtile_n + col)
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=4,
                                    swizzle=cutlass.Swizzle(3 if fp32 else 2, 4, 3),
                                )
                                if cutlass.const_expr(PREP_SHAPE is not None):
                                    pixel = (cutlass.Int64 if wide_index else cutlass.Int32)(m_off_cta) + row
                                    batch = pixel // volume_div
                                    time = (pixel % volume_div) // plane_div
                                    spatial = pixel % plane_div
                                    if batch < batches and time >= Z_out - cache_frames:
                                        cache_offset = (
                                            (
                                                (cutlass.Int64(batch) * cache_frames + time - Z_out + cache_frames)
                                                * P_out
                                                * Q_out
                                                + spatial
                                            )
                                            * norm_channels
                                            + subtile_idx * 32
                                            + col
                                        )
                                        (prep_cache.iterator.raw_ptr() + cache_offset).store(vec_io, alignment=4)
                        else:
                            for j in cutlass.range_constexpr(32 // vsize):
                                vec_io = t2r_rmem[j * vsize : j * vsize + vsize].to(c_dtype)
                                if cutlass.const_expr(SPATIAL_SHAPE is not None):
                                    spatial_pixel = cutlass.Int64(m_off_cta) + tidx
                                    spatial_channel = k_off_base + subtile_idx * subtile_n + j * vsize
                                    if spatial_pixel < SPATIAL_SHAPE[0] * Z_out * P_out * Q_out:
                                        bias_vec = (conv_bias.iterator.raw_ptr() + spatial_channel).load(
                                            count=vsize, alignment=16
                                        )
                                        vec_io = (vec_io.to(cutlass.Float32) + bias_vec.to(cutlass.Float32)).to(c_dtype)
                                        spatial_skip = (
                                            residual.iterator.raw_ptr()
                                            + spatial_pixel * SPATIAL_SHAPE[1]
                                            + spatial_channel
                                        ).load(count=vsize, alignment=16)
                                        if cutlass.const_expr(residual_bias is not None):
                                            spatial_skip_bias = (
                                                residual_bias.iterator.raw_ptr() + spatial_channel
                                            ).load(count=vsize, alignment=16)
                                            spatial_skip = (
                                                spatial_skip.to(cutlass.Float32) + spatial_skip_bias.to(cutlass.Float32)
                                            ).to(c_dtype)
                                        vec_io = (vec_io.to(cutlass.Float32) + spatial_skip.to(cutlass.Float32)).to(
                                            c_dtype
                                        )
                                smem_thr_ptr = smem_tile_base.subview(tidx * subtile_n + j * vsize)
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=16,
                                    swizzle=cutlass.Swizzle(3 if fp32 else 2, 4, 3),
                                )

                        # Make swizzled SMEM stores visible to the TMA proxy.
                        cute.arch.fence_view_async_shared()
                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

                        # One issuer owns the TMA store, commit, and wait sequence.
                        if warp_idx == epilogue_warp_ids[0] and store_issuer:
                            k_off = (0 if paired_norm else k_off_base) + subtile_idx * subtile_n
                            prims.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_desc.get_ptr(),
                                smem_tile_base,
                                (k_off, q_out, p_out, z_out, n_out),
                                mode=prims.TMAStoreMode.IM2COL,
                            )
                            prims.cp_async_bulk_commit_group()
                            # Bound outstanding reads before the next SMEM reuse.
                            prims.cp_async_bulk_wait_group(num_c_stage - 1, read=True)

                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

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
                    mbar_cluster_ptr = prims.mapa(acc_empty_mbar_ptr_stage, leader_cta_rank)
                    prims.mbarrier_arrive(
                        mbar_cluster_ptr,
                        count=1,
                        scope=prims.MemScope.CLUSTER,
                    )

            work_index += work_stride
            tile_idx, valid_tile = schedule.get_work(work_index, runtime_clusters_m, wide_index)

        # The issuing lane drains global writes, not only SMEM reads.
        if warp_idx == epilogue_warp_ids[0] and store_issuer:
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
                while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10000000):
                    pass

            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=_cta_group)


_DEFAULT_CONV_CONFIG = ConvConfig()


def _coordinate_divisor(value, wide_index: bool):
    """Build a fast 32-bit divisor on the host, or preserve a wide divisor."""
    return value if wide_index else cute.fast_divmod_create_divisor(value)


class _Launch:
    def __init__(
        self,
        config: ConvSpecialization,
        fuse_norm: bool = False,
        has_residual: bool = False,
        has_residual_bias: bool = False,
        spatial_output: bool = False,
        wide_index: bool = False,
    ) -> None:
        self.config = config
        self.fuse_norm = fuse_norm
        self.policy = LaunchPolicy.for_conv(
            config, fuse_norm=fuse_norm, has_residual=has_residual, spatial_output=spatial_output
        )
        self.has_residual = has_residual
        self.has_residual_bias = has_residual_bias
        self.spatial_output = spatial_output
        self.wide_index = wide_index

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"WanConv{'NormSilu' if self.fuse_norm else 'Raw'}_"
            f"{cfg.ci}_{cfg.co}_"
            f"2cta_n{cfg.tile_n}_k3_ctas{cfg.ctas}_norm{self.fuse_norm}"
            f"_res{self.has_residual}_rb{self.has_residual_bias}"
            f"_spatial{self.spatial_output}_wide{self.wide_index}_{cfg.precision}"
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: driver.CUstream,
        conv_bias: cute.Tensor = None,
        gamma: cute.Tensor = None,
        prep_padded: cute.Tensor = None,
        prep_cache: cute.Tensor = None,
        residual: cute.Tensor = None,
        residual_bias: cute.Tensor = None,
        prep_residual: cute.Tensor = None,
        prep_history: cute.Tensor = None,
        *,
        geometry,
        history: cutlass.Int32,
    ) -> None:
        cfg = ConvConfig(
            *geometry, ci=self.config.ci, co=self.config.co, ctas=self.config.ctas, precision=self.config.precision
        )
        tma_a_desc, tma_b_desc, tma_c_desc = _make_tensor_maps(
            cfg,
            a,
            b,
            c,
            prepared_output=self.fuse_norm,
            spatial_output=self.spatial_output,
        )
        groups = 2
        scheduler = TileSchedule(
            tiles_n=1 if self.fuse_norm else (cfg.co + cfg.tile_n - 1) // cfg.tile_n,
            cluster_m=groups,
        )
        clusters_m = (math.prod(cfg.output_shape[:-1]) + 255) // 256
        grid = (groups, 1, cutlass.min(clusters_m * scheduler.tiles_n, cfg.ctas // groups))
        policy = self.policy
        # Each spatial filter position has its own independently zero-filled
        # channel tail. Flattening TRS*C before ceil-div truncates C=160 work.
        logical_k = 16 if cfg.precision == "tf32x3" else 32 if cfg.precision == "tf32" else 64
        reduction_tiles = 27 * ((cfg.ci + logical_k - 1) // logical_k)
        z, h, w = cfg.output_shape[1:4]
        hp, wp = h + 2, w + 2
        auxiliary_rows = (2 - history) * hp * wp + (z + history) * (2 * wp + 2 * h)
        volume_div = _coordinate_divisor(z * h * w, self.wide_index)
        plane_div = _coordinate_divisor(h * w, self.wide_index)
        width_div = _coordinate_divisor(w, self.wide_index)
        padded_plane_div = _coordinate_divisor(hp * wp, self.wide_index)
        padded_width_div = _coordinate_divisor(wp, self.wide_index)
        auxiliary_rows_div = _coordinate_divisor(auxiliary_rows, self.wide_index)
        border_rows_div = _coordinate_divisor(2 * wp + 2 * h, self.wide_index)
        history_div = _coordinate_divisor(max(history, 1), self.wide_index)
        spatial_border_div = _coordinate_divisor(h + w + 1, self.wide_index)
        cluster_divisor = _coordinate_divisor(clusters_m, self.wide_index)
        kernel(
            "conv_bias_residual_pad"
            if self.spatial_output
            else "conv_residual_norm_prepare"
            if self.has_residual
            else "conv_norm_silu_prepare"
            if self.fuse_norm
            else "conv_raw",
            scheduler,
            (128 * groups, cfg.tile_n, 32 if cfg.precision != "bf16" else 64),
            (128 * groups, cfg.tile_n, 8 if cfg.precision != "bf16" else 16),
            reduction_tiles,
            policy,
            True,
            cfg.output_shape[1:4],
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            a.element_type,
            conv_bias,
            gamma,
            prep_padded=prep_padded,
            prep_cache=prep_cache,
            PREP_SHAPE=(cfg.n, history) if self.fuse_norm else None,
            residual=residual,
            residual_bias=residual_bias,
            prep_residual=prep_residual,
            SPATIAL_SHAPE=(cfg.n, cfg.co) if self.spatial_output else None,
            prep_history=prep_history,
            runtime_clusters_m=cluster_divisor,
            volume_div=volume_div,
            plane_div=plane_div,
            width_div=width_div,
            padded_plane_div=padded_plane_div,
            padded_width_div=padded_width_div,
            auxiliary_rows_div=auxiliary_rows_div,
            border_rows_div=border_rows_div,
            history_div=history_div,
            spatial_border_div=spatial_border_div,
            wide_index=self.wide_index,
            compensated=cfg.precision == "tf32x3",
        ).launch(
            grid=grid,
            block=(policy.threads, 1, 1),
            min_blocks_per_mp=1 if self.fuse_norm or self.spatial_output else 0,
            cluster=(groups, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


def _make_tensor_maps(
    cfg: ConvConfig,
    x: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    *,
    prepared_output: bool = False,
    spatial_output: bool = False,
) -> tuple[cuda.TensorMap, cuda.TensorMap, cuda.TensorMap]:
    """Build public tensor maps inside the JIT host wrapper."""
    dtype = x.element_type
    component_channels = cfg.ci * 2 if cfg.precision == "tf32x3" else cfg.ci
    a_dims, a_strides = _contiguous_tma_layout((*cfg.input_shape[:-1], component_channels), dtype)
    b_dims, b_strides = _contiguous_tma_layout(cfg.packed_weight_shape, dtype)
    c_dims, c_strides = _contiguous_tma_layout(cfg.output_shape, dtype)
    output_offset = 0
    if prepared_output or spatial_output:
        # Logical T/H/W omit the halo, while physical pitches include it.
        # An IM2COL store can therefore walk 128 logical pixels across rows,
        # frames and batches without ever touching the separately owned halo.
        _, t, h, w, c = cfg.output_shape
        pad_hw = 2 if prepared_output else 1
        pad_t = 2 if prepared_output else 0
        c_strides_bytes = (
            cutlass.Int64(c) * dtype.width // 8,
            cutlass.Int64(w + pad_hw) * c * dtype.width // 8,
            cutlass.Int64(h + pad_hw) * (w + pad_hw) * c * dtype.width // 8,
            cutlass.Int64(t + pad_t) * (h + pad_hw) * (w + pad_hw) * c * dtype.width // 8,
        )
        c_strides = tuple(stride // 16 for stride in c_strides_bytes)
        if prepared_output:
            output_offset = (cutlass.Int64(2) * (h + 2) * (w + 2) + (w + 2) + 1) * c

    tma_a_desc = cuda.create_tensor_map_im2col(
        global_address=x.iterator.toint(),
        dtype=dtype,
        global_dims=a_dims,
        global_strides=a_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(-2, -2, -2),
        channels_per_pixel=32 if cfg.precision != "bf16" else 64,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_b_desc = cuda.create_tensor_map_tiled(
        global_address=weight.iterator.toint(),
        dtype=dtype,
        global_dims=b_dims,
        global_strides=b_strides,
        box_dims=(32 if cfg.precision != "bf16" else 64, 1, 1, 1, cfg.tile_n // 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_c_desc = cuda.create_tensor_map_im2col(
        global_address=(output.iterator + output_offset).toint(),
        dtype=dtype,
        global_dims=c_dims,
        global_strides=c_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s128b if cfg.precision != "bf16" else cuda.TensorMapSwizzle.s64b,
    )
    return tma_a_desc, tma_b_desc, tma_c_desc


def _fake_flat(precision="bf16") -> cute.Tensor:
    """Symbolic contiguous storage with the kernel's 16-byte alignment."""
    return make_fake_compact_tensor(
        cutlass.Float32 if precision != "bf16" else cutlass.BFloat16,
        (cute.sym_int64(divisibility=8),),
        assumed_align=16,
    )


def _fake_conv_operands(config: ConvSpecialization) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    weight = make_fake_compact_tensor(
        cutlass.Float32 if config.precision != "bf16" else cutlass.BFloat16,
        (math.prod(config.packed_weight_shape),),
        assumed_align=16,
    )
    return _fake_flat(config.precision), weight, _fake_flat(config.precision)


def _fake_geometry(wide_index: bool) -> tuple:
    integer = cutlass.Int64 if wide_index else cutlass.Int32
    return tuple(integer(value) for value in (1, 3, 3, 3))


@compile_cache()
def _compile_conv(config: ConvSpecialization, wide_index: bool = False) -> Callable:
    """Compile one channel/tile/index-width variant, independent of N/T/H/W."""
    return compile_kernel(
        _Launch(config, wide_index=wide_index),
        *_fake_conv_operands(config),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        geometry=_fake_geometry(wide_index),
        history=cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


@compile_cache()
def _compile_prepared(
    config: ConvSpecialization,
    has_residual: bool = False,
    has_residual_bias: bool = False,
    save_residual: bool = True,
    wide_index: bool = False,
) -> Callable:
    """Compile fused next-input/cache production with runtime history length."""
    parameter = make_fake_compact_tensor(
        cutlass.Float32 if config.precision != "bf16" else cutlass.BFloat16, (config.co,), assumed_align=16
    )
    return compile_kernel(
        _Launch(
            config,
            fuse_norm=True,
            has_residual=has_residual,
            has_residual_bias=has_residual_bias,
            wide_index=wide_index,
        ),
        *_fake_conv_operands(config),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        parameter,
        parameter,
        _fake_flat(config.precision),
        _fake_flat(config.precision),
        _fake_flat(config.precision) if has_residual else None,
        parameter if has_residual_bias else None,
        _fake_flat(config.precision) if has_residual and save_residual else None,
        _fake_flat(config.precision),
        geometry=_fake_geometry(wide_index),
        history=cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


@compile_cache()
def _compile_spatial_residual(
    config: ConvSpecialization,
    has_residual_bias: bool = False,
    wide_index: bool = False,
) -> Callable:
    """Compile conv/bias/residual plus bottom/right padding for spatial downsample."""
    parameter = make_fake_compact_tensor(
        cutlass.Float32 if config.precision != "bf16" else cutlass.BFloat16, (config.co,), assumed_align=16
    )
    return compile_kernel(
        _Launch(
            config,
            has_residual=True,
            has_residual_bias=has_residual_bias,
            spatial_output=True,
            wide_index=wide_index,
        ),
        *_fake_conv_operands(config),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        parameter,
        None,
        _fake_flat(config.precision),
        None,
        _fake_flat(config.precision),
        parameter if has_residual_bias else None,
        None,
        geometry=_fake_geometry(wide_index),
        history=cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


def _prepared_shapes(
    config: ConvConfig,
    previous_frames: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Padded, new-cache and history shapes."""
    n, t, h, w, c = config.output_shape
    return (
        (n, t + 2, h + 2, w + 2, c),
        (n, min(2, t + previous_frames), h, w, c),
        (n, previous_frames, h, w, c),
    )


def compile_conv(config: ConvConfig = _DEFAULT_CONV_CONFIG) -> Callable:
    """Return a shared executable; launches must supply geometry and history."""
    _require_tensor_core_policy(config)
    return _compile_conv(specialization(config), _needs_wide_index(config))


def _require_tensor_core_policy(config: ConvConfig) -> None:
    if config.precision == "ieee":
        raise ValueError("Strict FP32 uses library convolution; bind with prepare/prepare_next instead")


def compile_prepared(
    config: ConvConfig = _DEFAULT_CONV_CONFIG,
    previous_frames: int = 0,
    has_residual: bool = False,
    has_residual_bias: bool = False,
    save_residual: bool = True,
) -> Callable:
    """History 0/1/2 shares one executable for a given fusion configuration."""
    _require_tensor_core_policy(config)
    variant = specialization(config)
    if config.co not in (96, 160, 192, 256, 320):
        raise ValueError("The prepared epilogue requires Co in (96, 160, 192, 256, 320)")
    if previous_frames not in (0, 1, 2):
        raise ValueError("History must contain zero, one or two frames")
    if has_residual_bias and not has_residual:
        raise ValueError("Residual bias requires a residual input")
    return _compile_prepared(variant, has_residual, has_residual_bias, save_residual, _needs_wide_index(config))


def compile_spatial_residual(
    config: ConvConfig = _DEFAULT_CONV_CONFIG,
    has_residual_bias: bool = False,
) -> Callable:
    _require_tensor_core_policy(config)
    variant = specialization(config)
    if config.co % config.tile_n:
        raise ValueError("Spatial residual fusion requires whole output-channel tiles")
    return _compile_spatial_residual(variant, has_residual_bias, _needs_wide_index(config))


class PreparedConv:
    """Bound operands for stream-ordered, graph-capturable inference.

    The object retains bound operands and custom-kernel workspaces. Do not resize or replace their
    storage while this object or a captured graph is in use. Launches follow
    the current Torch CUDA stream. Retain this object until asynchronous work
    completes, including captured graph replay. Calls and output consumption
    must be serialized for custom kernels: output is reused, and concurrent streams are not
    independently buffered. Autograd is unsupported; grad-enabled operands are
    rejected when the launch is bound.
    Weights use the packed shape/layout for the selected arithmetic mode:
    K64-padded OTRSC for BF16, K32-padded rounded values for TF32, interleaved
    high/low K16 components for experimental TF32x3, and ordinary OTRSC for IEEE.
    Packed weight tails must be zero; use the matching weight packer.
    TF32/TF32x3 activations are packed on every launch, including graph replay.
    The IEEE library path allocates fresh outputs on each eager call. During
    CUDA graph replay its captured allocations are reused by the graph.
    Use the call return value for outputs across both backends.
    Outputs, caches, and residuals always use ordinary storage in the input dtype.
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
        save_residual: bool = True,
    ) -> None:
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("Convolution requires BF16 or FP32 storage")
        config = replace(
            config,
            precision=resolve_precision(x.dtype, config.precision),
        )
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
            raise ValueError("PreparedConv supports inference only; use no_grad or inference_mode")
        weight_shape = tuple(weight.shape)
        if weight_shape != config.packed_weight_shape:
            raise ValueError(f"Weight must use the packed {config.precision} shape {config.packed_weight_shape}")
        for label, tensor, shape in (
            ("input", x, config.input_shape),
            ("weight", weight, weight_shape),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != x.dtype:
                raise ValueError(f"{label} must have {x.dtype} storage and shape {shape}")
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.data_ptr() % 16:
                raise ValueError(f"{label} must be CUDA, contiguous, and 16-byte aligned")
        if x.device != weight.device:
            raise ValueError("Input and weight must be on the same device")
        if spatial_output and (prepare_next_input or gamma is not None or residual is None or conv_bias is None):
            raise ValueError(
                "Spatial residual fusion requires bias/residual without normalization or temporal preparation"
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
                raise ValueError(f"{label} must be aligned, contiguous, and match input dtype/device, shape {shape}")
        self.parameters = ()
        if conv_bias is not None:
            if not spatial_output and (config.co not in (96, 160, 192, 256, 320)):
                raise ValueError("The fused epilogue requires Co in (96, 160, 192, 256, 320)")
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
                    raise ValueError(f"Bias/gamma must be aligned CUDA {x.dtype} vectors of {config.co}")
            self.parameters = tuple(
                parameter.detach() if parameter is not None else None for parameter in (conv_bias, gamma)
            )
        previous_frames = 0
        if prepare_next_input:
            if previous is not None and (previous.ndim != 5 or previous.shape[1] not in (1, 2)):
                raise ValueError("History must be NTHWC with one or two frames")
            previous_frames = 0 if previous is None else previous.shape[1]
            padded_shape, cache_shape, previous_shape = _prepared_shapes(config, previous_frames)
            if previous is not None and (
                tuple(previous.shape) != previous_shape
                or previous.device != x.device
                or previous.dtype != x.dtype
                or not previous.is_contiguous()
                or previous.data_ptr() % 16
            ):
                raise ValueError("History must be aligned contiguous NTHWC with the input dtype/device")
        self.device = x.device
        self.library = config.precision == "ieee"
        if self.library:
            self.library_args = tuple(
                tensor.detach() if tensor is not None else None
                for tensor in (x, weight, conv_bias, gamma, previous, residual, residual_bias)
            )
            self.benchmark = torch.backends.cudnn.benchmark
            self.deterministic = torch.backends.cudnn.deterministic or torch.are_deterministic_algorithms_enabled()
            self.prepare_next_input = prepare_next_input
            self.spatial_output = spatial_output
            self.save_residual = save_residual
            return
        self.compensated = config.precision == "tf32x3"
        self.source = x.detach() if config.precision in ("tf32", "tf32x3") else None
        if self.source is not None:
            x = torch.empty(
                (*x.shape[:-1], x.shape[-1] * (2 if self.compensated else 1)), device=x.device, dtype=x.dtype
            )
        self.component_input = x if self.source is not None else None
        self.device = x.device
        if prepare_next_input:
            padded = torch.empty(padded_shape, device=x.device, dtype=x.dtype)
            cache = torch.empty(cache_shape, device=x.device, dtype=x.dtype)
            summed = (
                torch.empty(config.output_shape, device=x.device, dtype=x.dtype)
                if residual is not None and save_residual
                else None
            )
            self.output = PreparedConvInput(padded, cache, summed)
            output_storage = padded
            self.parameters += tuple(tensor.detach().view(-1) for tensor in (padded, cache))
            # A real pointer keeps history presence out of the compile key.
            # history=0 makes the copy loop empty, so this alias is never read.
            self.parameters += tuple(
                tensor.detach().view(-1) if tensor is not None else None
                for tensor in (residual, residual_bias, summed, previous if previous is not None else padded)
            )
            self.compiled = compile_prepared(
                config, previous_frames, residual is not None, residual_bias is not None, save_residual
            )
        elif spatial_output:
            n, t, h, w, c = config.output_shape
            self.output = torch.empty((n, t, h + 1, w + 1, c), device=x.device, dtype=x.dtype)
            output_storage = self.output
            self.parameters += tuple(
                tensor.detach().view(-1) if tensor is not None else None
                for tensor in (self.output, None, residual, residual_bias, None)
            )
            self.compiled = compile_spatial_residual(config, residual_bias is not None)
        else:
            self.output = torch.empty(config.output_shape, device=x.device, dtype=x.dtype)
            output_storage = self.output
            self.compiled = compile_conv(config)
        self.operands = tuple(tensor.detach().view(-1) for tensor in (x, weight, output_storage))
        self.geometry = (config.n, config.t, config.h, config.w)
        self.history = 0 if previous is None else previous.shape[1]

    def __call__(self) -> torch.Tensor | PreparedConvInput:
        """Enqueue convolution/post-ops; only custom kernels reuse bound outputs."""
        if self.library:
            return self._library_forward()
        with torch.cuda.device(self.device):
            stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            if self.source is not None:
                pack_components(self.source, self.component_input, compensated=self.compensated)
            self.compiled(*self.operands, stream, *self.parameters, geometry=self.geometry, history=self.history)
        return self.output

    def _library_forward(self) -> torch.Tensor | PreparedConvInput:
        """Strict FP32 convolution followed by a separate fused post-op kernel."""
        if __package__:
            from .norm import rmsnorm_silu_conv_prep
            from .residual import bias_residual
        else:
            from norm import rmsnorm_silu_conv_prep
            from residual import bias_residual
        x, weight, bias, gamma, previous, residual, residual_bias = self.library_args
        with torch.cuda.device(self.device):
            output = (
                torch.ops.aten.cudnn_convolution.default(
                    x.permute(0, 4, 1, 2, 3),
                    weight.permute(0, 4, 1, 2, 3),
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    1,
                    self.benchmark,
                    self.deterministic,
                    False,
                )
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )
        if self.prepare_next_input:
            result = rmsnorm_silu_conv_prep(
                output,
                gamma,
                previous=previous,
                input_bias=bias,
                residual=residual,
                residual_bias=residual_bias,
            )
            if not self.save_residual:
                return PreparedConvInput(result.padded, result.cache, None)
            return result
        if self.spatial_output:
            return bias_residual(output, residual, bias, residual_bias=residual_bias, pad_spatial=True)
        return output


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

    B denotes rounding to the input storage dtype, with FP32 arithmetic between casts.
    conv is the convolution rounded to the input storage dtype; absent residual_bias skips
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
    save_residual: bool = True,
) -> PreparedConv:
    """Bind C96/C160/C192/C256/C320 conv/bias/norm/SiLU with causal padding and history outputs.

    Let v = B(conv(x) + bias), where B denotes rounding to the input storage dtype. With
    a residual, replace v by B(v + B(residual + residual_bias)) and save v
    in output.residual when save_residual=True; otherwise omit that allocation
    and store. Absent residual_bias skips its addition. Then let
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
        save_residual=save_residual,
    )


class WanConv3d(torch.nn.Module):
    """Inference-only residual convolution on a padded NTHWC activation.

    Weights enter in Torch OITHW order and are packed once as OTRSC: K64
    channel tails for BF16, rounded K32 values for TF32, high/low K16 components
    for experimental TF32x3, or ordinary OTRSC for strict FP32 library convolution.
    An omitted policy resolves from dtype and cuDNN's TF32 permission once at
    construction. Explicit policies are independent of subsequent backend flags.
    Forward allocates a fresh output, so successive temporal chunks cannot
    overwrite a previous result. Descriptors are built for the current buffers;
    only channel/pipeline/index-width variants are cached. Raw forward defers bias; the prepared
    and spatial-residual methods fuse bias and the corresponding postprocessing.
    """

    def __init__(self, weight: torch.Tensor, *, precision: str | None = None) -> None:
        super().__init__()
        if weight.dtype not in (torch.bfloat16, torch.float32) or not weight.is_cuda:
            raise ValueError("WanConv3d requires BF16 or FP32 CUDA weights")
        precision = resolve_precision(weight.dtype, precision)
        if (
            weight.ndim != 5
            or tuple(weight.shape[2:]) != (3, 3, 3)
            or (weight.shape[1], weight.shape[0])
            not in _ENCODER_CHANNEL_PAIRS
            + _DECODER_CHANNEL_PAIRS
            + _WAN21_CHANNEL_PAIRS
            + (((16, 160),) if weight.dtype == torch.float32 else ())
            + (((12, 160),) if precision == "ieee" else ())
        ):
            raise ValueError("WanConv3d requires a supported Wan residual 3x3x3 weight")
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        self.precision = precision
        # Supported output channels are multiples of their selected N tile.
        logical_k = {"bf16": 64, "ieee": 1, "tf32": 32, "tf32x3": 16}[self.precision]
        packed = torch.nn.functional.pad(
            weight.detach().permute(0, 2, 3, 4, 1).contiguous(), (0, (-self.in_channels) % logical_k)
        )
        if self.precision in ("tf32", "tf32x3"):
            compensated = self.precision == "tf32x3"
            components = torch.empty(
                (*packed.shape[:-1], packed.shape[-1] * (2 if compensated else 1)),
                device=weight.device,
                dtype=weight.dtype,
            )
            pack_components(packed, components, compensated=compensated)
            packed = components
        self.register_buffer("weight", packed, persistent=False)
        multiprocessors = torch.cuda.get_device_properties(weight.device).multi_processor_count
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
            precision=self.precision,
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
            bound = prepare_spatial_residual(x, self.weight, conv_bias, residual, config, residual_bias)
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


def _make_input_tensor_maps(
    cfg: InputConvConfig,
    weight: cute.Tensor,
    output: cute.Tensor,
) -> tuple[cuda.TensorMap, cuda.TensorMap]:
    """Build input-convolution tensor maps inside the JIT host wrapper."""
    dims, strides = _contiguous_tma_layout((160, 448), cutlass.BFloat16)
    tma_b_desc = cuda.create_tensor_map_tiled(
        global_address=weight.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=dims,
        global_strides=strides,
        box_dims=(64, 160),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    dims, strides = _contiguous_tma_layout(cfg.output_shape, cutlass.BFloat16)
    tma_c_desc = cuda.create_tensor_map_im2col(
        global_address=output.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=dims,
        global_strides=strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s64b,
    )
    return tma_b_desc, tma_c_desc


class _InputLaunch:
    def __init__(self, ctas: int) -> None:
        self.ctas = ctas

    def __repr__(self) -> str:
        return f"WanInputConv_ctas{self.ctas}"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        geometry,
        stream: driver.CUstream,
    ) -> None:
        cfg = InputConvConfig(*geometry, ctas=self.ctas)
        tma_b_desc, tma_c_desc = _make_input_tensor_maps(cfg, weight, output)
        scheduler = TileSchedule(tiles_n=1, cluster_m=1)
        clusters_m = (math.prod(cfg.output_shape[:-1]) + 127) // 128
        policy = LaunchPolicy.for_input()
        z, h, w = cfg.output_shape[1:4]
        kernel(
            "conv_input_c12",
            scheduler,
            (128, 160, 64),
            (128, 160, 16),
            7,
            policy,
            False,
            (z, h, w),
            tma_b_desc,
            tma_b_desc,
            tma_c_desc,
            packed_x=x,
            packed_shape=geometry,
            runtime_clusters_m=cute.fast_divmod_create_divisor(clusters_m),
            volume_div=cute.fast_divmod_create_divisor(z * h * w),
            plane_div=cute.fast_divmod_create_divisor(h * w),
            width_div=cute.fast_divmod_create_divisor(w),
        ).launch(
            grid=(1, 1, cutlass.min(clusters_m, self.ctas)),
            block=(policy.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


@compile_cache()
def compile_input(ctas: int = 148) -> Callable:
    """Compile the C12 gather pipeline; launch geometry is bounded by its binding."""
    if ctas <= 0:
        raise ValueError("CTA count must be positive")
    activation = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int64(),), assumed_align=16)
    weight = make_fake_compact_tensor(cutlass.BFloat16, (160 * 448,), assumed_align=16)
    output = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int64(),), assumed_align=16)
    return compile_kernel(
        _InputLaunch(ctas),
        activation,
        weight,
        output,
        tuple(cutlass.Int32(1) for _ in range(4)),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def pack_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack constant OITHW weights once into zero-tailed O(TRSC) rows."""
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (160, 12, 3, 3, 3):
        raise ValueError("Expected BF16 input-convolution weights [160,12,3,3,3]")
    packed = torch.zeros((160, 28, 16), dtype=weight.dtype, device=weight.device)
    packed[:, :27, :12].copy_(weight.detach().permute(0, 2, 3, 4, 1).reshape(160, 27, 12))
    packed = packed.reshape(160, 448)
    return packed


class PreparedInputConv:
    """Owning launch binding. Retain it for asynchronous work and graph replay.

    Each launch reuses output; concurrent calls on different streams are not
    supported. The module wrapper creates a fresh binding/output per forward.
    C16 input tails must be finite zeros, as produced by pack_history.
    Weights must be zero-tailed pack_weight output.
    """

    def __init__(self, x: torch.Tensor, weight: torch.Tensor, cfg: InputConvConfig) -> None:
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
                raise ValueError("Expected aligned contiguous BF16 CUDA operands matching config")
        self.x = x
        self.weight = weight
        self.output = torch.empty(cfg.output_shape, dtype=x.dtype, device=x.device)
        with torch.cuda.device(x.device):
            self.operands = tuple(tensor.detach().view(-1) for tensor in (x, weight, self.output))
            self.compiled = compile_input(cfg.ctas)
            self.geometry = (cfg.n, cfg.t, cfg.h, cfg.w)

    def __call__(self) -> torch.Tensor:
        with torch.cuda.device(self.x.device):
            self.compiled(
                *self.operands,
                self.geometry,
                driver.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        return self.output


def prepare_input(x: torch.Tensor, weight: torch.Tensor, cfg: InputConvConfig) -> PreparedInputConv:
    """Bind prepacked weights and the already padded activation."""
    return PreparedInputConv(x, weight, cfg)


class WanInputConv3d(torch.nn.Module):
    """Inference-only C12->C160 convolution with policy-specific input packing.

    Strict FP32 consumes C12 directly through cuDNN. Tensor Core paths consume
    C16 packed input with finite zeros in channels 12:16 (use pack_history).
    ``input_channels`` tells the caller which packing to use.
    """

    def __init__(self, weight: torch.Tensor, *, precision: str | None = None) -> None:
        super().__init__()
        if tuple(weight.shape) != (160, 12, 3, 3, 3):
            raise ValueError("Expected input-convolution weights [160,12,3,3,3]")
        self.general_conv = None
        precision = resolve_precision(weight.dtype, precision)
        self.input_channels = 12 if precision == "ieee" else 16
        if precision == "ieee":
            self.general_conv = WanConv3d(weight, precision=precision)
            return
        if weight.dtype == torch.float32:
            padded = torch.zeros((160, 16, 3, 3, 3), device=weight.device, dtype=weight.dtype)
            padded[:, :12].copy_(weight)
            self.general_conv = WanConv3d(padded, precision=precision)
            return
        self.register_buffer("weight", pack_weight(weight), persistent=False)
        self.ctas = torch.cuda.get_device_properties(weight.device).multi_processor_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or (torch.is_grad_enabled() and x.requires_grad):
            raise ValueError("WanInputConv3d supports inference only")
        if self.general_conv is not None:
            return self.general_conv(x)
        cfg = InputConvConfig(*x.shape[:4], ctas=self.ctas)
        bound = prepare_input(x, self.weight, cfg)
        result = bound()
        stream = torch.cuda.current_stream(x.device)
        for tensor in (x, self.weight, result):
            tensor.record_stream(stream)
        return result


def _compile_wan22() -> None:
    """Compile raw, normalized, residual-prepared, spatial, and input kernels."""
    compile_input()
    for ci, co in _ENCODER_CHANNEL_PAIRS:
        cfg = ConvConfig(ci=ci, co=co)
        compile_conv(cfg)
        if co <= 320:
            compile_prepared(cfg, previous_frames=2, has_residual=True, has_residual_bias=co == 320)
        if ci == co:
            compile_spatial_residual(cfg)
    for ci, co in _DECODER_CHANNEL_PAIRS:
        cfg = ConvConfig(ci=ci, co=co)
        compile_conv(cfg)
        if co != 1024:
            compile_prepared(cfg, previous_frames=2, has_residual=True)


@torch.inference_mode()
def _verify_decoder() -> None:
    """Check raw and fused decodes, including persistent partial CTA pairs."""
    if __package__:
        from .norm import torch_reference as norm_reference
    else:
        from norm import torch_reference as norm_reference
    torch.manual_seed(44)
    torch.backends.cudnn.allow_tf32 = False
    for ci, co in _DECODER_CHANNEL_PAIRS:
        weight = torch.randn(co, ci, 3, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.1
        module = WanConv3d(weight).eval()
        module.ctas = 4  # Force static-scheduler reuse on small tensors.
        shapes = ((2, 6, 11, 13), (1, 3, 7, 9))
        for n, t, h, w in shapes:
            # Keep separate reference-preparation specializations shape-local.
            torch.compiler.reset()
            cfg = ConvConfig(n=n, t=t, h=h, w=w, ci=ci, co=co, ctas=module.ctas)
            x = torch.randn(cfg.input_shape, device="cuda", dtype=weight.dtype) * 0.1
            bound = prepare(x, module.weight, cfg)

            def reference(x=x, weight=weight) -> torch.Tensor:
                return torch.nn.functional.conv3d(x.permute(0, 4, 1, 2, 3).float(), weight.float()).permute(
                    0, 2, 3, 4, 1
                )

            expected = reference()
            torch.testing.assert_close(module(x).float(), expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(bound().float(), expected, atol=0.02, rtol=0.02)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bound()
            for _ in range(3):
                x.normal_(0, 0.1)
                bound.output.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(bound.output.float(), reference(), atol=0.02, rtol=0.02)
            if co == 1024:
                continue  # The latent input convolution has no integrated normalization.
            bias = torch.randn(co, device="cuda", dtype=x.dtype)
            gamma = torch.randn_like(bias)
            for history, has_residual in (
                (0, False),
                (1, False),
                (2, False),
                (1, True),
                (2, True),
            ):
                previous = torch.randn(n, history, h - 2, w - 2, co, device="cuda", dtype=x.dtype) if history else None
                residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype) if has_residual else None
                actual = prepare_next(x, module.weight, bias, gamma, cfg, previous, residual=residual)()
                expected_prep = norm_reference(
                    bound(),
                    gamma,
                    previous=previous,
                    input_bias=bias,
                    residual=residual,
                )
                torch.testing.assert_close(actual.padded, expected_prep.padded, atol=0, rtol=0)
                torch.testing.assert_close(actual.cache, expected_prep.cache, atol=0, rtol=0)
                if has_residual:
                    torch.testing.assert_close(actual.residual, expected_prep.residual, atol=0, rtol=0)
        print(
            f"Decoder C{ci}->{co}: correctness PASS (including partial CTA pair and graph replay)",
            flush=True,
        )


@torch.inference_mode()
def _verify_wan22() -> None:
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
    for ci, co in _ENCODER_CHANNEL_PAIRS:
        # Keep separate reference-preparation specializations family-local.
        torch.compiler.reset()
        # These reduced batches are correctness checks, never performance results.
        frames, height, width = (4, 320, 240) if co == 160 else (2, 160, 120) if co == 320 else (1, 80, 60)
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
        torch_weight = (torch.randn(co, ci, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1).contiguous(
            memory_format=torch.channels_last_3d
        )
        module = WanConv3d(torch_weight).eval()
        raw = prepare(x, module.weight, cfg)
        torch_x = x.permute(0, 4, 1, 2, 3)

        fp32 = torch.nn.functional.conv3d(torch_x.float(), torch_weight.float()).permute(0, 2, 3, 4, 1)
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
                kwargs = {
                    "residual": residual if with_residual else None,
                    "residual_bias": bias if with_residual and co == 320 else None,
                }
                fused = prepare_next(x, module.weight, bias, gamma, cfg, previous, **kwargs)

                def separate(
                    raw=raw,
                    gamma=gamma,
                    previous=previous,
                    bias=bias,
                    kwargs=kwargs,
                ) -> PreparedConvInput:
                    return rmsnorm_silu_conv_prep(raw(), gamma, previous=previous, input_bias=bias, **kwargs)

                actual, expected = fused(), separate()
                torch.testing.assert_close(actual.padded, expected.padded, atol=0, rtol=0)
                torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
                if with_residual:
                    torch.testing.assert_close(actual.residual, expected.residual, atol=0, rtol=0)
        if ci == co:
            fused_spatial = prepare_spatial_residual(x, module.weight, bias, residual, cfg)
            expected = torch_bias_residual(raw(), residual, bias, pad_spatial=True)
            torch.testing.assert_close(fused_spatial(), expected, atol=0, rtol=0)
        # A 35-pixel output leaves the entire second CTA inactive for stores.
        # That peer must still complete every paired-MMA handshake.
        tail_cfg = ConvConfig(t=3, h=7, w=9, ci=ci, co=co, ctas=cfg.ctas)
        tail_x = torch.randn(tail_cfg.input_shape, device="cuda", dtype=x.dtype) * 0.1
        tail_raw = prepare(tail_x, module.weight, tail_cfg)
        tail_ref = torch.nn.functional.conv3d(tail_x.permute(0, 4, 1, 2, 3).float(), torch_weight.float()).permute(
            0, 2, 3, 4, 1
        )
        torch.testing.assert_close(tail_raw().float(), tail_ref, atol=0.02, rtol=0.02)
        if co <= 320:
            for history_frames in (0, 1, 2):
                history = (
                    torch.randn(1, history_frames, 5, 7, co, device="cuda", dtype=x.dtype) if history_frames else None
                )
                for with_residual in (False, True):
                    skip = torch.randn(tail_cfg.output_shape, device="cuda", dtype=x.dtype) if with_residual else None
                    tail_fused = prepare_next(
                        tail_x,
                        module.weight,
                        bias,
                        gamma,
                        tail_cfg,
                        history,
                        residual=skip,
                    )
                    actual = tail_fused()
                    if history_frames == 2:
                        for _ in range(3):
                            tail_fused()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            tail_fused()
                        history.normal_()
                        actual.padded.fill_(float("nan"))
                        actual.cache.fill_(float("nan"))
                        graph.replay()
                    expected = rmsnorm_silu_conv_prep(
                        tail_raw(),
                        gamma,
                        previous=history,
                        input_bias=bias,
                        residual=skip,
                    )
                    torch.testing.assert_close(actual.padded, expected.padded, atol=0, rtol=0)
                    torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
                    if with_residual:
                        torch.testing.assert_close(actual.residual, expected.residual, atol=0, rtol=0)
        print(f"C{ci}->{co}: correctness PASS (including partial CTA pair)", flush=True)
    cfg = InputConvConfig(n=1, t=6, h=322, w=242, ctas=sm_count)
    x = torch.randn(cfg.input_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    x[..., 12:] = 0
    weight = (torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1).contiguous(
        memory_format=torch.channels_last_3d
    )
    raw = prepare_input(x, pack_weight(weight), cfg)
    torch_x = x[..., :12].contiguous().permute(0, 4, 1, 2, 3)

    def reference() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    torch.testing.assert_close(raw(), reference(), atol=0.02, rtol=0.02)
    # Exercise the C12 producer/MMA ordering across persistent reuse and replay.
    for _ in range(3):
        raw()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw()
    for _ in range(3):
        x.normal_(0, 0.1)
        x[..., 12:] = 0
        torch_x.copy_(x[..., :12].permute(0, 4, 1, 2, 3))
        raw.output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(raw.output, reference(), atol=0.02, rtol=0.02)
    _verify_decoder()
    print(
        "Convolution family: PASS (raw, fused, history, residual, input, decoder)",
        flush=True,
    )


def _print_fusion_costs(results: list[tuple[str, tuple[float, float | None, float, float]]]) -> None:
    """Compare kernel and complete-operation costs against raw/separate paths."""
    headers = (
        "Operation",
        "Raw ms",
        "Kernel ms",
        "Total ms",
        "Separate ms",
        "Total/raw",
        "Extra ms",
        "Separate/total",
    )
    rows = [
        (
            label,
            f"{raw:.4f}",
            f"{kernel_ms:.4f}" if kernel_ms is not None else "-",
            f"{total:.4f}",
            f"{separate:.4f}",
            f"{total / raw:.3f}x",
            f"{total - raw:+.4f}",
            f"{separate / total:.3f}x",
        )
        for label, (raw, kernel_ms, total, separate) in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for index, row in enumerate([headers, *rows]):
        print(
            " | ".join(
                value.ljust(width) if i == 0 else value.rjust(width)
                for i, (value, width) in enumerate(zip(row, widths))
            ),
            flush=True,
        )
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


def _benchmark_convolution_reference(x: torch.Tensor, weight: torch.Tensor, precision: str) -> torch.Tensor:
    """Untimed oracle; TF32 uses independent operand rounding and strict convolution."""
    if precision == "tf32":
        x, weight = round_tf32_reference(x), round_tf32_reference(weight)
    with torch.backends.cudnn.flags(allow_tf32=False):
        return torch.nn.functional.conv3d(x, weight).permute(0, 2, 3, 4, 1)


def _check_benchmark_convolution(actual: torch.Tensor, expected: torch.Tensor, precision: str) -> None:
    """Fail the benchmark on mismatch against the selected arithmetic oracle."""
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        raise AssertionError("Nonfinite convolution benchmark output")
    atol, rtol = (0.02, 0.02) if precision == "bf16" else (1e-4, 1e-4)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.inference_mode()
def _benchmark_conv(cfg: ConvConfig, history: int, spatial: bool, *, residual_bias: bool | None = None) -> None:
    """Time one production callsite; release its large buffers before the next."""
    torch.compiler.reset()
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
    print(
        f"Logical convolution work={2 * math.prod(cfg.output_shape) * 27 * cfg.ci / 1e9:.3f} GFLOP | "
        f"Policy={cfg.precision} | packed Ci={cfg.packed_weight_shape[-1]} | tile N={cfg.tile_n}",
        flush=True,
    )
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
    fusion_tolerance = 0.0 if cfg.precision == "bf16" else 1e-4
    x = torch.randn(cfg.input_shape, device="cuda", dtype=dtype) * 0.1
    weight = (torch.randn(cfg.co, cfg.ci, 3, 3, 3, device="cuda", dtype=x.dtype) * 0.1).contiguous(
        memory_format=torch.channels_last_3d
    )
    module = WanConv3d(weight, precision=cfg.precision).eval()
    raw = prepare(x, module.weight, cfg)
    torch_x = x.permute(0, 4, 1, 2, 3)

    def torch_conv() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    _check_benchmark_convolution(raw(), _benchmark_convolution_reference(torch_x, weight, cfg.precision), cfg.precision)
    torch_results = [measure("conv", torch_conv, raw)]
    fusion_results = []

    def fusion_costs(label, fused, separate):
        if fused.library:
            raw_ms, total_ms, separate_ms = measure_group((raw, fused, separate), samples=21)
            fusion_results.append((label, (raw_ms, None, total_ms, separate_ms)))
            return

        def kernel_only():
            # Warmup has packed the fixed activation. This diagnostic excludes
            # TF32 conversion; raw, total and separate timings include it.
            stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
            fused.compiled(*fused.operands, stream, *fused.parameters, geometry=fused.geometry, history=fused.history)
            return fused.output

        fusion_results.append((label, measure_group((raw, kernel_only, fused, separate))))

    bias = torch.randn(cfg.co, device="cuda", dtype=x.dtype) * 0.1
    if cfg.co in (96, 160, 192, 256, 320):
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
            residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype) if with_residual else None
            kwargs = {
                "residual": residual,
                "residual_bias": bias
                if with_residual and (cfg.co == 320 if residual_bias is None else residual_bias)
                else None,
            }
            fused = prepare_next(x, module.weight, bias, gamma, cfg, previous, **kwargs)

            def separate(kwargs=kwargs) -> PreparedConvInput:
                return rmsnorm_silu_conv_prep(raw(), gamma, previous=previous, input_bias=bias, **kwargs)

            def reference(kwargs=kwargs) -> PreparedConvInput:
                return torch_reference(torch_conv(), gamma, previous=previous, input_bias=bias, **kwargs)

            actual, expected = fused(), separate()
            torch.testing.assert_close(actual.padded, expected.padded, atol=fusion_tolerance, rtol=fusion_tolerance)
            torch.testing.assert_close(actual.cache, expected.cache, atol=fusion_tolerance, rtol=fusion_tolerance)
            if with_residual:
                torch.testing.assert_close(
                    actual.residual, expected.residual, atol=fusion_tolerance, rtol=fusion_tolerance
                )
            del actual, expected
            operation = "conv + residual + norm/SiLU/prep" if with_residual else "conv + norm/SiLU/prep"
            torch_results.append(measure(operation, reference, fused))
            fusion_costs(operation, fused, separate)
        del fused, previous, residual

    if spatial:
        residual = torch.randn(cfg.output_shape, device="cuda", dtype=x.dtype)
        fused_spatial = prepare_spatial_residual(x, module.weight, bias, residual, cfg)
        torch.testing.assert_close(
            fused_spatial(),
            torch_bias_residual(raw(), residual, bias, pad_spatial=True),
            atol=fusion_tolerance,
            rtol=fusion_tolerance,
        )
        operation = "conv + residual + spatial pad"
        torch_results.append(
            measure(
                operation,
                lambda: torch_bias_residual(torch_conv(), residual, bias, pad_spatial=True),
                fused_spatial,
            )
        )
        fusion_costs(operation, fused_spatial, lambda: bias_residual(raw(), residual, bias, pad_spatial=True))

    print("Against Torch eager/cuDNN:", flush=True)
    print_benchmark_table(torch_results)
    if fusion_results:
        print(
            "Fusion cost (same operands, rotating CUDA-graph timing order):"
            "\nKernel=direct fused launch, excluding activation conversion; -=no fused kernel."
            "\nRaw/Total/Separate include per-call activation conversion; fused paths include history copying."
            "\nSeparate=raw convolution + custom postprocessing; IEEE Total also uses separate library/post-ops."
            "\nTotal/raw is a cost ratio (lower is better); Extra=Total-Raw.",
            flush=True,
        )
        _print_fusion_costs(fusion_results)


@torch.inference_mode()
def benchmark_matched(model: str, shapes: list[tuple[int, ...]], histories: list[int], precision: str = "bf16") -> None:
    """Hold output N/T/H/W and epilogue options fixed while varying channel width."""
    channels = ((96, 192) if model in ("all", "wan21") else ()) + ((160, 256, 320) if model in ("all", "wan22") else ())
    ctas = torch.cuda.get_device_properties(0).multi_processor_count // 2 * 2
    print(
        f"Matched geometry benchmark: {precision}, Ci=Co, 3x3x3, stride=1, channels-last."
        "\nOnly channel width varies within each geometry/history group."
        "\nBias + norm/SiLU/prep, with/without residual; residual bias disabled for every width."
        "\nTotal timings include activation conversion and history copying; external padding, weight packing and compilation excluded."
        f"\nGPU: {torch.cuda.get_device_name(0)} | CTAs={ctas} | cuDNN autotuning={torch.backends.cudnn.benchmark}",
        flush=True,
    )
    for n, frames, h, w in shapes:
        for history in histories:
            for c in sorted(channels):
                torch.manual_seed(44)
                cfg = ConvConfig(n=n, t=frames + 2, h=h + 2, w=w + 2, ci=c, co=c, ctas=ctas, precision=precision)
                _benchmark_conv(cfg, history, spatial=False, residual_bias=False)
    print("\nMatched geometry benchmark: convolution and fusion checks PASS", flush=True)


@torch.inference_mode()
def _benchmark_input(cfg: InputConvConfig, precision: str = "bf16") -> None:
    """Time the production C12 input convolution with prepacked operands."""
    print(f"\nC12->C160 input | output NTHWC={cfg.output_shape}", flush=True)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    weight = (torch.randn(160, 12, 3, 3, 3, device="cuda", dtype=dtype) * 0.1).contiguous(
        memory_format=torch.channels_last_3d
    )
    module = WanInputConv3d(weight, precision=precision).eval()
    x = torch.randn((*cfg.input_shape[:-1], module.input_channels), device="cuda", dtype=dtype) * 0.1
    x[..., 12:] = 0
    if module.general_conv is None:
        raw = prepare_input(x, module.weight, cfg)
    else:
        general_cfg = ConvConfig(cfg.n, cfg.t, cfg.h, cfg.w, module.input_channels, 160, cfg.ctas // 2 * 2, precision)
        raw = prepare(x, module.general_conv.weight, general_cfg)
    torch_x = x[..., :12].contiguous().permute(0, 4, 1, 2, 3)

    def reference() -> torch.Tensor:
        return torch.nn.functional.conv3d(torch_x, weight).permute(0, 2, 3, 4, 1)

    _check_benchmark_convolution(raw(), _benchmark_convolution_reference(torch_x, weight, precision), precision)
    print_benchmark_table([measure("input conv", reference, raw)])


def _benchmark_wan22(precision: str = "bf16") -> None:
    """Measure production encoder and decoder convolution shapes.

    The batch-32 encoder's C160/C320 stages retain four warm frames; downsampling reduces
    C640 at 80x60 to two frames and C640 at 40x30 to one. All initial chunks have
    one frame. Decoder convolutions and C256 fusions use batch 1 and four output frames.
    Shape reduction belongs only in verify(), never in these timings.
    """
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print("\nWan 2.2 encoder: batch=32", flush=True)
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
                precision=precision,
            )
            _benchmark_conv(
                cfg,
                history=0 if current_frames == 1 else 2,
                spatial=ci == co and height >= 80,
            )
    for frames in (1, 4):
        _benchmark_input(InputConvConfig(n=32, t=frames + 2, h=322, w=242, ctas=sm_count), precision)
    print("\nWan 2.2 decoder: batch=1, four output frames", flush=True)
    for ci, co, height, width in (
        (512, 256, 360, 640),
        (256, 256, 360, 640),
    ):
        cfg = ConvConfig(n=1, t=6, h=height + 2, w=width + 2, ci=ci, co=co, ctas=sm_count // 2 * 2, precision=precision)
        _benchmark_conv(cfg, history=0, spatial=False)
    print("\nWan 2.2 decoder latent input: batch=1, one latent frame", flush=True)
    _benchmark_conv(
        ConvConfig(n=1, t=3, h=47, w=82, ci=48, co=1024, ctas=sm_count // 2 * 2, precision=precision),
        history=2,
        spatial=False,
    )


@torch.inference_mode()
def _verify_wan21() -> None:
    """Check all Wan 2.1 pairs with batch/spatial tails and temporal chunks."""
    if __package__:
        from .norm import rmsnorm_silu_conv_prep
    else:
        from norm import rmsnorm_silu_conv_prep

    torch.manual_seed(42)
    for ci, co in _WAN21_CHANNEL_PAIRS:
        weight = torch.randn(co, ci, 3, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.02
        conv = WanConv3d(weight).eval()
        for n, t, h, w in ((1, 1, 3, 5), (2, 3, 5, 7)):
            x = torch.randn(n, t + 2, h + 2, w + 2, ci, device="cuda", dtype=weight.dtype)
            expected = torch.nn.functional.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)
            torch.testing.assert_close(conv(x), expected, atol=0.03125, rtol=0.02)
        print(f"Wan 2.1 C{ci}->C{co}: correctness PASS (batch/spatial tails, cold/warm geometry)", flush=True)
        if co not in (96, 192):
            continue
        bias = torch.randn(co, device="cuda", dtype=weight.dtype) * 0.1
        gamma = torch.randn_like(bias)
        for n, frames, h, w in ((1, 1, 3, 5), (2, 4, 11, 13)):
            torch.compiler.reset()
            # Two persistent CTA pairs exercise reuse and an inactive tail peer.
            cfg = ConvConfig(n=n, t=frames + 2, h=h + 2, w=w + 2, ci=ci, co=co, ctas=4)
            x = torch.randn(cfg.input_shape, device="cuda", dtype=weight.dtype) * 0.1
            raw = prepare(x, conv.weight, cfg)
            for history in (0, 1, 2):
                previous = torch.randn(n, history, h, w, co, device="cuda", dtype=weight.dtype) if history else None
                for with_residual in (False, True):
                    residual = (
                        torch.randn(cfg.output_shape, device="cuda", dtype=weight.dtype) if with_residual else None
                    )
                    kwargs = {"residual": residual, "residual_bias": bias if with_residual and history == 2 else None}
                    fused = prepare_next(x, conv.weight, bias, gamma, cfg, previous, **kwargs)
                    actual = fused()
                    if history == 2:
                        for _ in range(3):
                            fused()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            fused()
                        x.normal_(0, 0.1)
                        previous.normal_()
                        actual.padded.fill_(float("nan"))
                        actual.cache.fill_(float("nan"))
                        if actual.residual is not None:
                            actual.residual.fill_(float("nan"))
                        graph.replay()
                    expected = rmsnorm_silu_conv_prep(raw(), gamma, previous=previous, input_bias=bias, **kwargs)
                    torch.testing.assert_close(actual.padded, expected.padded, atol=0, rtol=0)
                    torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
                    if with_residual:
                        torch.testing.assert_close(actual.residual, expected.residual, atol=0, rtol=0)
                    print(
                        f"  fused N={n} T={frames} history={history} residual={with_residual}: bitwise PASS", flush=True
                    )


def _benchmark_wan21(precision: str = "bf16") -> None:
    """Benchmark supported 3x3x3 callsites for 480x832 video, batch 1.

    Encoder warm chunks start with four video frames; decoder chunks start
    with one latent frame. Include the single-frame first chunk at each stage.
    Wider pairs are kernel experiments: the model retains cuDNN for C384.
    """
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    stages = {
        "encoder": (
            (96, 96, 4, 480, 832),
            (96, 192, 4, 240, 416),
            (192, 192, 4, 240, 416),
            (192, 384, 2, 120, 208),
            (384, 384, 2, 120, 208),
            (384, 384, 1, 60, 104),
        ),
        "decoder": (
            (16, 384, 1, 60, 104),
            (384, 384, 1, 60, 104),
            (192, 384, 2, 120, 208),
            (384, 384, 2, 120, 208),
            (192, 192, 4, 240, 416),
            (96, 96, 4, 480, 832),
        ),
    }
    for direction, cases in stages.items():
        print(f"\nWan 2.1 {direction}: batch=1, video resolution=480x832", flush=True)
        for ci, co, frames, height, width in cases:
            for current_frames in sorted({1, frames}):
                cfg = ConvConfig(
                    n=1,
                    t=current_frames + 2,
                    h=height + 2,
                    w=width + 2,
                    ci=ci,
                    co=co,
                    ctas=sm_count // 2 * 2,
                    precision=precision,
                )
                _benchmark_conv(cfg, history=0 if current_frames == 1 else 2, spatial=False)


def compile(model: str = "all") -> None:
    """Compile convolution specializations for the selected Wan architecture."""
    if model in ("all", "wan21"):
        for ci, co in _WAN21_CHANNEL_PAIRS:
            cfg = ConvConfig(ci=ci, co=co)
            compile_conv(cfg)
            if co in (96, 192):
                compile_prepared(cfg, previous_frames=0)
                compile_prepared(cfg, previous_frames=2, has_residual=True, has_residual_bias=True)
    if model in ("all", "wan22"):
        _compile_wan22()


def verify(model: str = "all") -> None:
    """Run untimed convolution correctness checks for the selected architecture."""
    if model in ("all", "wan21"):
        _verify_wan21()
    if model in ("all", "wan22"):
        _verify_wan22()


def benchmark_production(model: str = "all", precision: str = "bf16") -> None:
    """Measure the selected architectures with prepared channels-last operands."""
    torch.manual_seed(44)
    print(
        f"\nProduction benchmark: {precision}, prepared channels-last operands."
        "\nMedian GPU ms from alternating CUDA-graph samples."
        "\nPer-call activation conversion included; compilation, weight packing, and external padding excluded."
        "\nTorch reference is eager/cuDNN, not torch.compile; speedup=reference/custom."
        f"\ncuDNN autotuning: {torch.backends.cudnn.benchmark}.",
        flush=True,
    )
    if model in ("all", "wan21"):
        _benchmark_wan21(precision)
    if model in ("all", "wan22"):
        _benchmark_wan22(precision)
    print("\nProduction benchmark: convolution and fusion checks PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("all", "wan21", "wan22"), default="all", help="Wan architecture to exercise (default: all)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-compile", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="Benchmark input/weight dtype")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit TF32 convolution for FP32 kernels and cuDNN references; matmul stays strict",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Run correctness checks without production benchmarks",
    )
    mode.add_argument(
        "--matched-shape",
        nargs=4,
        type=int,
        action="append",
        metavar=("N", "T", "H", "W"),
        help="Benchmark identical output geometry across supported fused channel widths; repeat for a sweep",
    )
    parser.add_argument(
        "--history",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        help="History lengths for --matched-shape (default: 0 2)",
    )
    args = parser.parse_args()
    if args.history is not None and args.matched_shape is None:
        parser.error("--history requires --matched-shape")
    if args.matched_shape is not None and any(min(shape) < 1 for shape in args.matched_shape):
        parser.error("--matched-shape dimensions must be positive")
    if (args.only_compile or args.verify_only) and (args.dtype != "bf16" or args.allow_tf32):
        parser.error("--only-compile/--verify-only cover BF16; use tests/check_arithmetic_modes.py for FP32 checks")
    precision = "bf16" if args.dtype == "bf16" else "tf32" if args.allow_tf32 else "ieee"
    torch.backends.cudnn.allow_tf32 = args.dtype == "fp32" and args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(f"Convolution policy: {precision}; matching cuDNN permission; FP32 matmul stays strict", flush=True)
    if args.only_compile:
        compile(args.model)
        print("Convolution family compile: PASS", flush=True)
    elif args.verify_only:
        print("Correctness checks (small cases, untimed):", flush=True)
        verify(args.model)
    elif args.matched_shape is not None:
        benchmark_matched(
            args.model, args.matched_shape, args.history if args.history is not None else [0, 2], precision
        )
    else:
        benchmark_production(args.model, precision)
