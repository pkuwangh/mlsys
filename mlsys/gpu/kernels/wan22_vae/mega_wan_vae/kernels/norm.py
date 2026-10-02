# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportMissingImports=false
# pyright: reportOperatorIssue=false

"""Fuse Wan convolution bias, RMSNorm/SiLU, causal padding, and cache updates.

All activation and cache tensors are contiguous BF16 [N,T,H,W,C]. The canonical
C160/320/640 path partitions one launch into current-frame normalization CTAs
and vectorized history/padding CTAs. Post-attention reduction keeps one group per
padded output row. Each owner also writes its part of the next cache; history
is already activated and is never normalized again. There are no temporary
activation/concatenation tensors, shared memory, or inter-warp dependencies.
The attention entry point returns affine normalization without SiLU or padding.
Uses only the installed public CUTLASS DSL 4.7.1 runtime.
"""

import argparse
from collections.abc import Callable
from functools import cache, partial

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__:
    from ._utils import PreparedConvInput, measure, print_benchmark_table
else:
    from _utils import PreparedConvInput, measure, print_benchmark_table

THREADS = 128

CACHE_FRAMES = 2
EPILOGUE_VECTOR = 8
AUX_VECTORS_PER_THREAD = 4


@cute.jit
def rmsnorm_silu_row(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    row: cutlass.Int64,
    lane: cutlass.Int32,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    CONTIGUOUS_REDUCTION: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    ROW_LANES: cutlass.Constexpr[int],
    APPLY_SILU: cutlass.Constexpr[bool] = True,
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool] = False,
    SAVE_INPUT: cutlass.Constexpr[bool] = False,
) -> cutlass.Array:
    """Fuse Wan RMSNorm, channel affine, and SiLU for NTHWC rows.

    For each (n,t,h,w), reducing only c=0..C-1:
        b_c = bf16(fp32(x_c) + fp32(input_bias_c))  [or x_c if absent]
        r_c = bf16(fp32(residual_c) + fp32(residual_bias_c))  [or residual_c]
        b_c = bf16(fp32(b_c) + fp32(r_c))  [if residual present]
        residual_out_c = b_c  [if HAS_RESIDUAL or SAVE_INPUT, before norm]
        v_c = fp32(b_c)
        d = max(sqrt(sum_c(v_c * v_c)), 1e-12)
        u_c = bf16(v_c / d)
        s_c = bf16(fp32(u_c) * sqrt(C))
        a_c = bf16(fp32(s_c) * fp32(gamma_c))
        z_c = bf16(fp32(a_c) + fp32(bias_c))  [or a_c when bias absent]
        out_c = bf16(fp32(z_c) / (1 + exp(-fp32(z_c))))
        out_c = z_c instead when APPLY_SILU=False (attention normalization)

    This follows F.normalize then scale/affine, NOT rsqrt(mean(x*x)+eps).
    All math is FP32 between the explicit BF16 cast points. One lane group owns
    a row and retains its inputs in registers. All group lanes participate in the
    reduction, including lanes whose last channel slot is out of bounds.
    For C=160/320/640, use 16 interleaved sums, merge four sums per group,
    then merge the four groups. This reproduces the eager Torch channel
    reduction order for the target's large NCTHW tensors (output vector=4,
    block height=4, four accumulators per thread), while loading NTHWC.
    CONTIGUOUS_REDUCTION instead matches Torch's contiguous-channel reduction:
    four adjacent values per lane, four accumulators, then a warp sum.
    Both variants read and write the same NTHWC layout.
    ROW_LANES=16 packs two independent rows per warp for the interleaved
    reduction. Explicit half-warp masks keep history/padding divergence legal;
    reduction source lanes and the final broadcast stay inside the row group.
    """
    VECTOR_REDUCE = CONTIGUOUS_REDUCTION and CHANNELS >= 128 and CHANNELS % 4 == 0
    GROUPS = 16 if not VECTOR_REDUCE and CHANNELS in (160, 320, 640) else 32
    SLOTS = (
        (CHANNELS + 127) // 128 * 4
        if VECTOR_REDUCE
        else (CHANNELS + GROUPS - 1) // GROUPS
    )
    lane_base = cutlass.Int32(0)
    shuffle_mask = cutlass.Uint32(0xFFFFFFFF)
    if ROW_LANES == 16:
        tid, _, _ = cute.arch.thread_idx()
        lane_base = tid % 32 // 16 * 16
        shuffle_mask = cutlass.Uint32(0xFFFF) << lane_base
    result = cutlass.Array(cutlass.BFloat16, SLOTS, space=cutlass.AddressSpace.rmem)
    for slot in cutlass.range_constexpr(SLOTS):
        result[slot] = cutlass.BFloat16(0.0)
    if row < x.shape[0]:
        values = cutlass.Array(
            cutlass.Float32,
            SLOTS,
            space=cutlass.AddressSpace.rmem,
        )
        partials = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.rmem)
        for part in cutlass.range_constexpr(4):
            partials[part] = cutlass.Float32(0.0)
        total = cutlass.Float32(0.0)
        for slot in cutlass.range_constexpr(SLOTS):
            col = lane + slot * GROUPS
            if VECTOR_REDUCE:
                col = lane * 4 + (slot // 4) * 128 + slot % 4
            value = cutlass.Float32(0.0)
            if lane < GROUPS and col < CHANNELS:
                value = cutlass.Float32(x[row, col])
                if HAS_INPUT_BIAS:
                    value = cutlass.Float32(
                        cutlass.BFloat16(value + cutlass.Float32(input_bias[col]))
                    )
                if HAS_RESIDUAL:
                    skip = cutlass.Float32(residual[row, col])
                    if HAS_RESIDUAL_BIAS:
                        skip = cutlass.Float32(
                            cutlass.BFloat16(skip + cutlass.Float32(residual_bias[col]))
                        )
                    value = cutlass.Float32(cutlass.BFloat16(value + skip))
                if HAS_RESIDUAL or SAVE_INPUT:
                    residual_out[row, col] = cutlass.BFloat16(value)
            values[slot] = value
            if VECTOR_REDUCE:
                partials[slot % 4] = partials[slot % 4] + value * value
            else:
                total = total + value * value
        if VECTOR_REDUCE:
            total = partials[0]
            for part in cutlass.range_constexpr(1, 4):
                total = total + partials[part]
            for offset in [16, 8, 4, 2, 1]:
                total = total + cute.arch.shuffle_sync_down(total, offset)
        elif GROUPS == 16:
            partial = total
            for offset in [4, 8, 12]:
                total = total + cute.arch.shuffle_sync(
                    partial,
                    lane_base + (lane + offset) % ROW_LANES,
                    mask=shuffle_mask,
                )
            for offset in [2, 1]:
                total = total + cute.arch.shuffle_sync_down(
                    total, offset, mask=shuffle_mask
                )
        else:
            for offset in [16, 8, 4, 2, 1]:
                total = total + cute.arch.shuffle_sync_down(total, offset)
        denominator = cute.math.sqrt(
            cute.arch.shuffle_sync(total, lane_base, mask=shuffle_mask), fastmath=False
        )
        if denominator < 1e-12:
            denominator = cutlass.Float32(1e-12)
        for slot in cutlass.range_constexpr(SLOTS):
            col = lane + slot * GROUPS
            if VECTOR_REDUCE:
                col = lane * 4 + (slot // 4) * 128 + slot % 4
            if lane < GROUPS and col < CHANNELS:
                normalized = cutlass.BFloat16(values[slot] / denominator)
                scaled = cutlass.BFloat16(cutlass.Float32(normalized) * (CHANNELS**0.5))
                affine = cutlass.BFloat16(
                    cutlass.Float32(scaled) * cutlass.Float32(gamma[col])
                )
                if HAS_BIAS:
                    affine = cutlass.BFloat16(
                        cutlass.Float32(affine) + cutlass.Float32(bias[col])
                    )
                value = cutlass.Float32(affine)
                result[slot] = affine
                if APPLY_SILU:
                    result[slot] = cutlass.BFloat16(
                        value / (1.0 + cute.math.exp(-value, fastmath=False))
                    )
    return result


@cute.kernel
def rmsnorm_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
) -> None:
    """Store the register-resident result of rmsnorm_silu_row as NTHWC."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    row = cutlass.Int64(bid) * (THREADS // 32) + tid // 32
    lane = tid % 32
    if row < x.shape[0]:
        values = rmsnorm_silu_row(
            x,
            gamma,
            bias,
            gamma,
            row,
            lane,
            CHANNELS,
            HAS_BIAS,
            True,
            False,
            x,
            gamma,
            out,
            False,
            32,
            False,
        )
        SLOTS = (CHANNELS + 127) // 128 * 4
        for slot in cutlass.range_constexpr(SLOTS):
            col = lane * 4 + (slot // 4) * 128 + slot % 4
            if col < CHANNELS:
                out[row, col] = values[slot]


@cute.jit
def rmsnorm_host(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
) -> None:
    """Launch four independent row warps per CTA on the caller's stream."""
    rows_per_block = THREADS // 32
    rmsnorm_kernel(x, gamma, bias, out, CHANNELS, HAS_BIAS).launch(
        grid=((x.shape[0] + rows_per_block - 1) // rows_per_block, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@cache
def compile_rmsnorm(
    channels: int = 160,
    has_bias: bool = False,
) -> Callable:
    """Cache code by channel count, bias mode, and reference reduction order."""
    if channels not in (160, 320, 640):
        raise ValueError("RMSNorm SiLU supports C160, C320, and C640")
    tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (cute.sym_int64(), channels),
        stride_order=(1, 0),
        assumed_align=2,
    )
    weight = make_fake_compact_tensor(cutlass.BFloat16, (channels,), assumed_align=2)
    # Keep the BF16 gamma multiply and bias addition separately rounded.
    return cute.compile(
        rmsnorm_host,
        tensor,
        weight,
        weight,
        tensor,
        make_fake_stream(),
        channels,
        has_bias,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


def rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run inference-only Wan attention RMSNorm-affine on contiguous BF16 NTHWC.

    Gamma and optional bias must be contiguous BF16 vectors of length C on
    the input device. Reject unsupported inputs instead of copying/falling back.
    Attention uses the contiguous-channel reference reduction and no activation.
    """
    validate_rmsnorm_inputs(x, gamma, bias)
    channels = x.shape[-1]
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    with torch.cuda.device(x.device):
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        compile_rmsnorm(channels, bias is not None)(
            x.view(-1, channels),
            gamma,
            gamma if bias is None else bias,
            out.view(-1, channels),
            stream,
        )
    return out


def validate_rmsnorm_inputs(
    x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None
) -> None:
    """Validate the shared inference-only BF16 NTHWC and affine contract."""
    if x.ndim != 5 or not x.is_contiguous():
        raise ValueError("RMSNorm SiLU requires contiguous [N,T,H,W,C] input")
    channels = x.shape[-1]
    if channels not in (160, 320, 640):
        raise ValueError("RMSNorm SiLU supports C160, C320, and C640")
    tensors = (x, gamma) if bias is None else (x, gamma, bias)
    for tensor in tensors:
        if tensor.device.type != "cuda" or tensor.device != x.device:
            raise ValueError("RMSNorm SiLU tensors must share a CUDA device")
        if tensor.dtype != torch.bfloat16:
            raise TypeError("RMSNorm SiLU requires torch.bfloat16")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError(
                "RMSNorm SiLU is inference-only; disable gradient tracking"
            )
    for weight in tensors[1:]:
        if weight.shape != (channels,) or not weight.is_contiguous():
            raise ValueError("RMSNorm SiLU affine weights must be contiguous C-vectors")


@cute.jit
def _conv_prep_row(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    row: cutlass.Int64,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Int64,
    cache_frames: cutlass.Int64,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    CONTIGUOUS_REDUCTION: cutlass.Constexpr[bool],
    PAD_T: cutlass.Constexpr[int],
    PAD_H: cutlass.Constexpr[int],
    PAD_W: cutlass.Constexpr[int],
) -> None:
    """Compute padded = pad(concat(previous, SiLU(WanRMSNorm(x))), ...).

    For each current-frame row, with BF16 cast B and FP32 math between casts:
        v_c = B(float(x_c) + float(input_bias_c))  [or x_c if absent]
        r_c = B(float(residual_c) + float(residual_bias_c))  [or residual_c]
        v_c = B(float(v_c) + float(r_c))  [if residual present]
        residual_out_c = v_c  [if HAS_RESIDUAL or SAVE_INPUT, before norm]
        d = max(sqrt(sum_c(float(v_c)**2)), 1e-12)
        u_c = B(float(v_c) / d)
        a_c = B(float(B(float(u_c) * sqrt(C))) * float(gamma_c))
        z_c = B(float(a_c) + float(bias_c))  [or a_c if bias absent]
        y_c = B(float(z_c) / (1 + exp(-float(z_c))))
    Let s = concat(previous, y) along time and L = previous_T + T:
        padded[n, t + PAD_T - previous_T, h + PAD_H, w + PAD_W, c]
            = s[n,t,h,w,c], for 0 <= t < L; zero elsewhere.
        cache = s[:, max(0,L-2):, :, :, :].
    History is already activated and is never normalized again. First-chunk
    missing history is zero padding, NOT part of the cache. Every output/cache
    element has exactly one row-group/lane owner. Shuffles use row-group masks.
    """
    tid, _, _ = cute.arch.thread_idx()
    ROW_LANES = 16 if not CONTIGUOUS_REDUCTION and CHANNELS in (160, 320, 640) else 32
    lane = tid % ROW_LANES
    if row < padded.shape[0]:
        padded_h = height + 2 * PAD_H
        padded_w = width + 2 * PAD_W
        w = row % padded_w - PAD_W
        h = (row // padded_w) % padded_h - PAD_H
        t = (row // (padded_w * padded_h)) % (frames + PAD_T) - PAD_T
        n = row // (padded_w * padded_h * (frames + PAD_T))
        valid_spatial = h >= 0 and h < height and w >= 0 and w < width
        VECTOR_REDUCE = CONTIGUOUS_REDUCTION and CHANNELS >= 128 and CHANNELS % 4 == 0
        GROUPS = 16 if not VECTOR_REDUCE and CHANNELS in (160, 320, 640) else 32
        SLOTS = (
            (CHANNELS + 127) // 128 * 4
            if VECTOR_REDUCE
            else (CHANNELS + GROUPS - 1) // GROUPS
        )
        values = cutlass.Array(cutlass.BFloat16, SLOTS, space=cutlass.AddressSpace.rmem)
        for slot in cutlass.range_constexpr(SLOTS):
            values[slot] = cutlass.BFloat16(0.0)
        if valid_spatial:
            if t >= 0:
                input_row = ((n * frames + t) * height + h) * width + w
                values = rmsnorm_silu_row(
                    x,
                    gamma,
                    bias,
                    input_bias,
                    input_row,
                    lane,
                    CHANNELS,
                    HAS_BIAS,
                    CONTIGUOUS_REDUCTION,
                    HAS_INPUT_BIAS,
                    residual,
                    residual_bias,
                    residual_out,
                    HAS_RESIDUAL,
                    ROW_LANES,
                    HAS_RESIDUAL_BIAS=HAS_RESIDUAL_BIAS,
                    SAVE_INPUT=SAVE_INPUT,
                )
            elif t >= -previous_frames:
                previous_row = (
                    (n * previous_frames + t + previous_frames) * height + h
                ) * width + w
                for slot in cutlass.range_constexpr(SLOTS):
                    col = lane + slot * GROUPS
                    if VECTOR_REDUCE:
                        col = lane * 4 + (slot // 4) * 128 + slot % 4
                    if lane < GROUPS and col < CHANNELS:
                        values[slot] = previous[previous_row, col]
        for slot in cutlass.range_constexpr(SLOTS):
            col = lane + slot * GROUPS
            if VECTOR_REDUCE:
                col = lane * 4 + (slot // 4) * 128 + slot % 4
            if lane < GROUPS and col < CHANNELS:
                padded[row, col] = values[slot]
                if valid_spatial and t >= frames - cache_frames:
                    cache_row = (
                        (n * cache_frames + t - frames + cache_frames) * height + h
                    ) * width + w
                    cache[cache_row, col] = values[slot]


@cute.kernel
def rmsnorm_silu_conv_prep_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Int64,
    cache_frames: cutlass.Int64,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    CONTIGUOUS_REDUCTION: cutlass.Constexpr[bool],
    PAD_T: cutlass.Constexpr[int],
    PAD_H: cutlass.Constexpr[int],
    PAD_W: cutlass.Constexpr[int],
) -> None:
    """Give each padded row one warp group; history/cache writers are disjoint."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    ROW_LANES = 16 if not CONTIGUOUS_REDUCTION and CHANNELS in (160, 320, 640) else 32
    row = cutlass.Int64(bid) * (THREADS // ROW_LANES) + tid // ROW_LANES
    _conv_prep_row(
        x,
        gamma,
        bias,
        input_bias,
        residual,
        residual_bias,
        residual_out,
        previous,
        padded,
        cache,
        row,
        frames,
        height,
        width,
        previous_frames,
        cache_frames,
        CHANNELS,
        HAS_BIAS,
        HAS_INPUT_BIAS,
        HAS_RESIDUAL,
        HAS_RESIDUAL_BIAS,
        SAVE_INPUT,
        CONTIGUOUS_REDUCTION,
        PAD_T,
        PAD_H,
        PAD_W,
    )


@cute.jit
def _conv_prep_aux_vector(
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    vector_idx: cutlass.Int64,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Int64,
    cache_frames: cutlass.Int64,
    CHANNELS: cutlass.Constexpr[int],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Own one history/halo vector; current-frame interiors are excluded.

    Enumerate two full history planes followed by each current frame's border.
    History contributes to cache only when current frames alone do not fill it.
    Narrow only bounded within-batch coordinates; all global offsets stay Int64.
    """
    padded_h = height + 2
    padded_w = width + 2
    history_rows = 2 * padded_h * padded_w
    border_rows = 2 * padded_w + 2 * height
    auxiliary_rows = history_rows + frames * border_rows
    vectors_per_row = CHANNELS // EPILOGUE_VECTOR
    vectors_per_batch = auxiliary_rows * vectors_per_row
    batches = padded.shape[0] // ((frames + 2) * padded_h * padded_w)
    if vector_idx < batches * vectors_per_batch:
        batch = vector_idx // vectors_per_batch
        index_type: cutlass.Constexpr = (
            cutlass.Int32 if vectors_per_batch < 2**31 else cutlass.Int64
        )
        local_vector = index_type(vector_idx % vectors_per_batch)
        channel = local_vector % vectors_per_row * EPILOGUE_VECTOR
        local_row = local_vector // vectors_per_row
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
                ph = border // padded_w * (height + 1)
                pw = border % padded_w
            else:
                ph = (border - 2 * padded_w) // 2 + 1
                pw = (border - 2 * padded_w) % 2 * (width + 1)
        values = cutlass.vector.full((EPILOGUE_VECTOR,), 0, cutlass.BFloat16)
        valid_spatial = ph > 0 and ph <= height and pw > 0 and pw <= width
        if pt < 2 and pt >= 2 - previous_frames and valid_spatial:
            source = (
                ((batch * previous_frames + pt - 2 + previous_frames) * height + ph - 1)
                * width
                + pw
                - 1
            ) * CHANNELS + channel
            values = (previous.iterator.raw_ptr() + source).load(
                count=EPILOGUE_VECTOR, alignment=ALIGNMENT
            )
            if pt - 2 >= frames - cache_frames:
                cache_offset = (
                    (
                        (batch * cache_frames + pt - 2 - frames + cache_frames) * height
                        + ph
                        - 1
                    )
                    * width
                    + pw
                    - 1
                ) * CHANNELS + channel
                (cache.iterator.raw_ptr() + cache_offset).store(
                    values, alignment=ALIGNMENT
                )
        destination = (
            ((batch * (frames + 2) + pt) * padded_h + ph) * padded_w + pw
        ) * CHANNELS + channel
        (padded.iterator.raw_ptr() + destination).store(values, alignment=ALIGNMENT)


@cute.jit
def _prep_current_row_pair(
    x: cute.Tensor,
    gamma: cute.Tensor,
    input_bias: cute.Tensor,
    residual_out: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    input_row: cutlass.Int64,
    padded_row: cutlass.Int64,
    cache_row: cutlass.Int64,
    write_cache: cutlass.Boolean,
    CHANNELS: cutlass.Constexpr[int],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Compute non-residual C160/C320 prep with two adjacent channels per lane.

    B denotes BF16 rounding; omit the addition when input_bias is absent.
    Math and every BF16 rounding follow rmsnorm_silu_row. Lane l retains
    B(x[16*s+2*l:16*s+2*l+2] + input_bias) packed in one register and sums
    the two channel parities independently. Each parity merges original sums
    at offsets 2,4,6; lane zero forms (G0+G2)+(G1+G3). This is the original
    16-sum FP32 tree, not a reassociated eight-sum reduction. Four independent
    rows share a warp; masks and broadcasts remain within each eight-lane row.
    """
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 8
    lane_base = tid % 32 // 8 * 8
    mask = cutlass.Uint32(0xFF) << lane_base
    retained = cutlass.Array(
        cutlass.Uint32, CHANNELS // 16, space=cutlass.AddressSpace.rmem
    )
    partials = cutlass.Array(cutlass.Float32, 2, space=cutlass.AddressSpace.rmem)
    for parity in cutlass.range_constexpr(2):
        partials[parity] = cutlass.Float32(0.0)
    for slot in cutlass.range_constexpr(CHANNELS // 16):
        col = slot * 16 + lane * 2
        offset = input_row * CHANNELS + col
        pair = (x.iterator.raw_ptr() + offset).load(count=2, alignment=ALIGNMENT)
        if HAS_INPUT_BIAS:
            bias_pair = (input_bias.iterator.raw_ptr() + col).load(
                count=2, alignment=ALIGNMENT
            )
            pair = (pair.to(cutlass.Float32) + bias_pair.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        if SAVE_INPUT:
            (residual_out.iterator.raw_ptr() + offset).store(pair, alignment=ALIGNMENT)
        retained.store(pair.bitcast(cutlass.Uint32), slot)
        values = pair.to(cutlass.Float32)
        for parity in cutlass.range_constexpr(2):
            partials[parity] = partials[parity] + values[parity] * values[parity]
    groups = cutlass.Array(cutlass.Float32, 2, space=cutlass.AddressSpace.rmem)
    for parity in cutlass.range_constexpr(2):
        partial = partials[parity]
        total = partial
        for offset in (2, 4, 6):
            total = total + cute.arch.shuffle_sync(
                partial, lane_base + (lane + offset) % 8, mask=mask
            )
        groups[parity] = total
    even = groups[0] + cute.arch.shuffle_sync(groups[0], lane_base + 1, mask=mask)
    odd = groups[1] + cute.arch.shuffle_sync(groups[1], lane_base + 1, mask=mask)
    total = cute.arch.shuffle_sync(even + odd, lane_base, mask=mask)
    denominator = cute.math.sqrt(total, fastmath=False)
    if denominator < 1e-12:
        denominator = cutlass.Float32(1e-12)
    for slot in cutlass.range_constexpr(CHANNELS // 16):
        col = slot * 16 + lane * 2
        values = retained.load(slot, 1).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        normalized = (values / denominator).to(cutlass.BFloat16)
        scaled = (normalized.to(cutlass.Float32) * (CHANNELS**0.5)).to(cutlass.BFloat16)
        gamma_pair = (gamma.iterator.raw_ptr() + col).load(count=2, alignment=ALIGNMENT)
        affine = (scaled.to(cutlass.Float32) * gamma_pair.to(cutlass.Float32)).to(
            cutlass.BFloat16
        )
        activated = affine.to(cutlass.Float32)
        result = (activated / (1.0 + cute.math.exp(-activated, fastmath=False))).to(
            cutlass.BFloat16
        )
        (padded.iterator.raw_ptr() + padded_row * CHANNELS + col).store(
            result, alignment=ALIGNMENT
        )
        if write_cache:
            (cache.iterator.raw_ptr() + cache_row * CHANNELS + col).store(
                result, alignment=ALIGNMENT
            )


@cute.jit
def _prep_contiguous_row(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    input_row: cutlass.Int64,
    padded_row: cutlass.Int64,
    cache_row: cutlass.Int64,
    write_cache: cutlass.Boolean,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Vectorize channels while preserving Torch's 32-lane reduction tree.

    C160/C320 use eight physical lanes to represent 32 virtual lanes;
    C640 uses a full warp to bound per-thread retained values. Each lane owns
    four adjacent channels per group. Channels separated by 128 accumulate
    before each virtual lane's four partials are summed. For eight lanes,
    ((group0 + group2) + (group1 + group3)) implements warp offsets 16 and 8;
    physical shuffles finish the tree. This is the same FP32 tree as
    rmsnorm_silu_row(CONTIGUOUS_REDUCTION=True), with the same BF16 cast points.
    Values stay packed in registers between load/reduction and activation/store.
    Canonical row strides and column offsets preserve the checked 8B alignment.
    """
    tid, _, _ = cute.arch.thread_idx()
    LANES = 32 if CHANNELS == 640 else 8
    VIRTUAL_GROUPS = 32 // LANES
    lane = tid % LANES
    lane_base = tid % 32 // LANES * LANES
    mask = cutlass.Uint32((1 << LANES) - 1) << lane_base
    retained = cutlass.Array(
        cutlass.BFloat16, CHANNELS // LANES, space=cutlass.AddressSpace.rmem
    )
    partials = cutlass.Array(
        cutlass.Float32, VIRTUAL_GROUPS * 4, space=cutlass.AddressSpace.rmem
    )
    for group in cutlass.range_constexpr(CHANNELS // (LANES * 4)):
        col = lane * 4 + group * LANES * 4
        offset = input_row * CHANNELS + col
        value = (x.iterator.raw_ptr() + offset).load(count=4, alignment=ALIGNMENT)
        if HAS_INPUT_BIAS:
            b = (input_bias.iterator.raw_ptr() + col).load(count=4, alignment=ALIGNMENT)
            value = (value.to(cutlass.Float32) + b.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        if HAS_RESIDUAL:
            skip = (residual.iterator.raw_ptr() + offset).load(
                count=4, alignment=ALIGNMENT
            )
            if HAS_RESIDUAL_BIAS:
                b = (residual_bias.iterator.raw_ptr() + col).load(
                    count=4, alignment=ALIGNMENT
                )
                skip = (skip.to(cutlass.Float32) + b.to(cutlass.Float32)).to(
                    cutlass.BFloat16
                )
            value = (value.to(cutlass.Float32) + skip.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        if HAS_RESIDUAL or SAVE_INPUT:
            (residual_out.iterator.raw_ptr() + offset).store(value, alignment=ALIGNMENT)
        retained.store(value, group * 4)
        fp = value.to(cutlass.Float32)
        squares = fp * fp
        if cutlass.const_expr(group >= VIRTUAL_GROUPS):
            squares = partials.load((group % VIRTUAL_GROUPS) * 4, 4) + squares
        partials.store(squares, (group % VIRTUAL_GROUPS) * 4)
    groups = cutlass.Array(
        cutlass.Float32, VIRTUAL_GROUPS, space=cutlass.AddressSpace.rmem
    )
    for group in cutlass.range_constexpr(VIRTUAL_GROUPS):
        total = partials[group * 4]
        for part in cutlass.range_constexpr(1, 4):
            total = total + partials[group * 4 + part]
        groups[group] = total
    if cutlass.const_expr(LANES == 8):
        total = (groups[0] + groups[2]) + (groups[1] + groups[3])
    else:
        total = groups[0]
    for offset in (16, 8, 4, 2, 1) if LANES == 32 else (4, 2, 1):
        total = total + cute.arch.shuffle_sync_down(total, offset, mask=mask)
    denominator = cute.math.sqrt(
        cute.arch.shuffle_sync(total, lane_base, mask=mask), fastmath=False
    )
    if denominator < 1e-12:
        denominator = cutlass.Float32(1e-12)
    for group in cutlass.range_constexpr(CHANNELS // (LANES * 4)):
        col = lane * 4 + group * LANES * 4
        value = retained.load(group * 4, 4).to(cutlass.Float32)
        normalized = (value / denominator).to(cutlass.BFloat16)
        scaled = (normalized.to(cutlass.Float32) * (CHANNELS**0.5)).to(cutlass.BFloat16)
        g = (gamma.iterator.raw_ptr() + col).load(count=4, alignment=ALIGNMENT)
        affine = (scaled.to(cutlass.Float32) * g.to(cutlass.Float32)).to(
            cutlass.BFloat16
        )
        if HAS_BIAS:
            b = (bias.iterator.raw_ptr() + col).load(count=4, alignment=ALIGNMENT)
            affine = (affine.to(cutlass.Float32) + b.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        value = affine.to(cutlass.Float32)
        result = (value / (1.0 + cute.math.exp(-value, fastmath=False))).to(
            cutlass.BFloat16
        )
        (padded.iterator.raw_ptr() + padded_row * CHANNELS + col).store(
            result, alignment=ALIGNMENT
        )
        if write_cache:
            (cache.iterator.raw_ptr() + cache_row * CHANNELS + col).store(
                result, alignment=ALIGNMENT
            )


@cute.kernel
def rmsnorm_silu_conv_prep_partitioned_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Int64,
    cache_frames: cutlass.Int64,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    CONTIGUOUS_REDUCTION: cutlass.Constexpr[bool],
) -> None:
    """Partition one nonpersistent launch into normalization and auxiliary CTAs.

    Canonical C160/320/640 with padding (2,1,1) enters here. Channels-last
    normalization uses vector memory operations, with four rows per warp for
    C160/C320 and one row per warp for C640 to bound register pressure.
    Legacy arithmetic keeps
    its eight/sixteen-lane mapping for explicit standalone comparisons only.
    Auxiliary CTAs own disjoint vectors, including the history part of cache;
    no cross-CTA communication or additional kernel launch is needed.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    PAIRED = (
        not CONTIGUOUS_REDUCTION
        and CHANNELS in (160, 320)
        and not HAS_RESIDUAL
        and not HAS_BIAS
    )
    VECTOR_CONTIGUOUS = CONTIGUOUS_REDUCTION
    ROW_LANES = (
        (32 if CHANNELS == 640 else 8) if VECTOR_CONTIGUOUS else (8 if PAIRED else 16)
    )
    SLOTS = (CHANNELS + 127) // 128 * 4 if CONTIGUOUS_REDUCTION else CHANNELS // 16
    current_blocks = (x.shape[0] + THREADS // ROW_LANES - 1) // (THREADS // ROW_LANES)
    if bid < current_blocks:
        input_row = cutlass.Int64(bid) * (THREADS // ROW_LANES) + tid // ROW_LANES
        if input_row < x.shape[0]:
            lane = tid % ROW_LANES
            values = cutlass.Array(
                cutlass.BFloat16, SLOTS, space=cutlass.AddressSpace.rmem
            )
            # Decode output coordinates after normalization: hoisting them
            # across normalization increases live registers on residual paths.
            if cutlass.const_expr(not PAIRED and not VECTOR_CONTIGUOUS):
                values = rmsnorm_silu_row(
                    x,
                    gamma,
                    bias,
                    input_bias,
                    input_row,
                    lane,
                    CHANNELS,
                    HAS_BIAS,
                    CONTIGUOUS_REDUCTION,
                    HAS_INPUT_BIAS,
                    residual,
                    residual_bias,
                    residual_out,
                    HAS_RESIDUAL,
                    ROW_LANES,
                    HAS_RESIDUAL_BIAS=HAS_RESIDUAL_BIAS,
                    SAVE_INPUT=SAVE_INPUT,
                )
            rows_per_batch = frames * height * width
            index_type: cutlass.Constexpr = (
                cutlass.Int32 if rows_per_batch < 2**31 else cutlass.Int64
            )
            batch = input_row // rows_per_batch
            local_row = index_type(input_row % rows_per_batch)
            t = local_row // (height * width)
            h = local_row // width % height
            w = local_row % width
            padded_row = (
                ((batch * (frames + 2) + t + 2) * (height + 2) + h + 1) * (width + 2)
                + w
                + 1
            )
            cache_row = (
                (batch * cache_frames + t - frames + cache_frames) * height + h
            ) * width + w
            if cutlass.const_expr(VECTOR_CONTIGUOUS):
                addresses = (
                    x.iterator.toint()
                    | gamma.iterator.toint()
                    | padded.iterator.toint()
                    | cache.iterator.toint()
                )
                if HAS_BIAS:
                    addresses = addresses | bias.iterator.toint()
                if HAS_INPUT_BIAS:
                    addresses = addresses | input_bias.iterator.toint()
                if HAS_RESIDUAL:
                    addresses = addresses | residual.iterator.toint()
                if HAS_RESIDUAL_BIAS:
                    addresses = addresses | residual_bias.iterator.toint()
                if HAS_RESIDUAL or SAVE_INPUT:
                    addresses = addresses | residual_out.iterator.toint()
                if (addresses & 7) == 0:
                    _prep_contiguous_row(
                        x,
                        gamma,
                        bias,
                        input_bias,
                        residual,
                        residual_bias,
                        residual_out,
                        padded,
                        cache,
                        input_row,
                        padded_row,
                        cache_row,
                        t >= frames - cache_frames,
                        CHANNELS,
                        HAS_BIAS,
                        HAS_INPUT_BIAS,
                        HAS_RESIDUAL,
                        HAS_RESIDUAL_BIAS,
                        SAVE_INPUT,
                        8,
                    )
                else:
                    _prep_contiguous_row(
                        x,
                        gamma,
                        bias,
                        input_bias,
                        residual,
                        residual_bias,
                        residual_out,
                        padded,
                        cache,
                        input_row,
                        padded_row,
                        cache_row,
                        t >= frames - cache_frames,
                        CHANNELS,
                        HAS_BIAS,
                        HAS_INPUT_BIAS,
                        HAS_RESIDUAL,
                        HAS_RESIDUAL_BIAS,
                        SAVE_INPUT,
                        2,
                    )
            elif cutlass.const_expr(PAIRED):
                addresses = (
                    x.iterator.toint()
                    | gamma.iterator.toint()
                    | padded.iterator.toint()
                    | cache.iterator.toint()
                )
                if HAS_INPUT_BIAS:
                    addresses = addresses | input_bias.iterator.toint()
                if SAVE_INPUT:
                    addresses = addresses | residual_out.iterator.toint()
                # Preserve the 2B ABI for offset views; only aligned pointers
                # use 32-bit pair loads/stores. Optional dummies are not read.
                if (addresses & 3) == 0:
                    _prep_current_row_pair(
                        x,
                        gamma,
                        input_bias,
                        residual_out,
                        padded,
                        cache,
                        input_row,
                        padded_row,
                        cache_row,
                        t >= frames - cache_frames,
                        CHANNELS,
                        HAS_INPUT_BIAS,
                        SAVE_INPUT,
                        4,
                    )
                else:
                    _prep_current_row_pair(
                        x,
                        gamma,
                        input_bias,
                        residual_out,
                        padded,
                        cache,
                        input_row,
                        padded_row,
                        cache_row,
                        t >= frames - cache_frames,
                        CHANNELS,
                        HAS_INPUT_BIAS,
                        SAVE_INPUT,
                        2,
                    )
            else:
                for slot in cutlass.range_constexpr(SLOTS):
                    col = lane + slot * 16
                    if CONTIGUOUS_REDUCTION:
                        col = lane * 4 + (slot // 4) * 128 + slot % 4
                    if col < CHANNELS:
                        padded[padded_row, col] = values[slot]
                        if t >= frames - cache_frames:
                            cache[cache_row, col] = values[slot]
    else:
        vector_idx = (
            cutlass.Int64(bid) - current_blocks
        ) * THREADS * AUX_VECTORS_PER_THREAD + tid
        # The public ABI promises only 2B alignment, including offset history
        # views and direct compiled-call outputs. Use wide operations only when
        # every auxiliary base is aligned; all vector offsets are multiples of 8.
        aligned = (
            (
                previous.iterator.toint()
                | padded.iterator.toint()
                | cache.iterator.toint()
            )
            & 15
        ) == 0
        for step in cutlass.range(AUX_VECTORS_PER_THREAD, unroll=1):
            if aligned:
                _conv_prep_aux_vector(
                    previous,
                    padded,
                    cache,
                    vector_idx + step * THREADS,
                    frames,
                    height,
                    width,
                    previous_frames,
                    cache_frames,
                    CHANNELS,
                    16,
                )
            else:
                _conv_prep_aux_vector(
                    previous,
                    padded,
                    cache,
                    vector_idx + step * THREADS,
                    frames,
                    height,
                    width,
                    previous_frames,
                    cache_frames,
                    CHANNELS,
                    2,
                )


@cute.jit
def conv_prep_host(
    x: cute.Tensor,
    gamma: cute.Tensor,
    bias: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Int64,
    cache_frames: cutlass.Int64,
    stream: cuda.CUstream,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    SAVE_INPUT: cutlass.Constexpr[bool],
    CONTIGUOUS_REDUCTION: cutlass.Constexpr[bool],
    PAD_T: cutlass.Constexpr[int],
    PAD_H: cutlass.Constexpr[int],
    PAD_W: cutlass.Constexpr[int],
) -> None:
    """Use partitioned canonical CTAs, or the unchanged general row scheduler."""
    if cutlass.const_expr(
        CHANNELS in (160, 320, 640) and (PAD_T, PAD_H, PAD_W) == (2, 1, 1)
    ):
        row_lanes = (
            (32 if CHANNELS == 640 else 8)
            if CONTIGUOUS_REDUCTION
            else (
                8
                if CHANNELS in (160, 320) and not HAS_RESIDUAL and not HAS_BIAS
                else 16
            )
        )
        current_blocks = (x.shape[0] + THREADS // row_lanes - 1) // (
            THREADS // row_lanes
        )
        padded_h = height + 2
        padded_w = width + 2
        auxiliary_rows = 2 * padded_h * padded_w + frames * (2 * padded_w + 2 * height)
        batches = x.shape[0] // (frames * height * width)
        vectors = batches * auxiliary_rows * (CHANNELS // EPILOGUE_VECTOR)
        vectors_per_cta = THREADS * AUX_VECTORS_PER_THREAD
        auxiliary_blocks = (vectors + vectors_per_cta - 1) // vectors_per_cta
        rmsnorm_silu_conv_prep_partitioned_kernel(
            x,
            gamma,
            bias,
            input_bias,
            residual,
            residual_bias,
            residual_out,
            previous,
            padded,
            cache,
            frames,
            height,
            width,
            previous_frames,
            cache_frames,
            CHANNELS,
            HAS_BIAS,
            HAS_INPUT_BIAS,
            HAS_RESIDUAL,
            HAS_RESIDUAL_BIAS,
            SAVE_INPUT,
            CONTIGUOUS_REDUCTION,
        ).launch(
            grid=(current_blocks + auxiliary_blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )
        return
    row_lanes = 16 if not CONTIGUOUS_REDUCTION and CHANNELS in (160, 320, 640) else 32
    rows_per_block = THREADS // row_lanes
    grid = (padded.shape[0] + rows_per_block - 1) // rows_per_block
    rmsnorm_silu_conv_prep_kernel(
        x,
        gamma,
        bias,
        input_bias,
        residual,
        residual_bias,
        residual_out,
        previous,
        padded,
        cache,
        frames,
        height,
        width,
        previous_frames,
        cache_frames,
        CHANNELS,
        HAS_BIAS,
        HAS_INPUT_BIAS,
        HAS_RESIDUAL,
        HAS_RESIDUAL_BIAS,
        SAVE_INPUT,
        CONTIGUOUS_REDUCTION,
        PAD_T,
        PAD_H,
        PAD_W,
    ).launch(
        grid=(grid, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@cache
def compile_conv_prep(
    channels: int = 160,
    has_bias: bool = False,
    contiguous_reduction: bool = True,
    has_input_bias: bool = False,
    has_residual: bool = False,
    shape: tuple[int, int, int] = (1, 3, 5),
    *,
    has_residual_bias: bool = False,
    save_input: bool = False,
) -> Callable:
    """Specialize immutable metadata without GPU allocations or device queries."""
    if has_residual_bias and not has_residual:
        raise ValueError("residual_bias requires residual")
    padding = (2, 1, 1)
    # Distinct symbols: current, history, padded, and cache row counts differ.
    tensors = [
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int64(), channels),
            stride_order=(1, 0),
            assumed_align=2,
        )
        for _ in range(4)
    ]
    weight = make_fake_compact_tensor(cutlass.BFloat16, (channels,), assumed_align=2)
    return cute.compile(
        conv_prep_host,
        tensors[0],
        weight,
        weight,
        weight,
        tensors[0],
        weight,
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        *shape,
        0,
        1,
        make_fake_stream(),
        channels,
        has_bias,
        has_input_bias,
        has_residual,
        has_residual_bias,
        save_input,
        contiguous_reduction,
        *padding,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


def rmsnorm_silu_conv_prep(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    previous: torch.Tensor | None = None,
    *,
    input_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    save_input: bool = False,
    contiguous_reduction: bool = True,
) -> PreparedConvInput:
    """Return fused padded input and next two-frame cache, both BF16 NTHWC.

    Padding is two leading temporal planes and one pixel on each spatial side.
    input_bias is the preceding convolution's optional C-vector, added and
    rounded to BF16 before normalization. It is never added to history/padding.
    When residual is present, add it after bias with a second BF16 rounding,
    and return that sum separately for the following block's skip connection.
    residual_bias optionally biases the raw residual with its own BF16 rounding
    before the skip addition; it requires residual. save_input also returns the
    pre-normalization biased input when residual is absent. Neither option
    changes normalization arithmetic or allocates a separate biased branch.
    Previous history contains one or two already-activated frames. No inputs
    are mutated, including cache and affine weights. Empty batches are allowed;
    time/spatial dimensions must be positive. Reject invalid layouts, devices,
    dtypes, history lengths, padding, and autograd instead of falling back.
    """
    padding = (2, 1, 1)
    if type(save_input) is not bool:
        raise TypeError("save_input must be a bool")
    if residual_bias is not None and residual is None:
        raise ValueError("residual_bias requires residual")
    validate_rmsnorm_inputs(x, gamma, bias)
    if input_bias is not None:
        validate_rmsnorm_inputs(x, gamma, input_bias)
    if residual_bias is not None:
        validate_rmsnorm_inputs(x, gamma, residual_bias)
    if residual is not None:
        validate_rmsnorm_inputs(residual, gamma, bias)
        if residual.shape != x.shape or residual.device != x.device:
            raise ValueError("Residual must match input shape and CUDA device")
    n, frames, height, width, channels = x.shape
    if min(frames, height, width) <= 0:
        raise ValueError("Convolution preparation requires positive T,H,W")
    previous_frames = 0
    if previous is not None:
        validate_rmsnorm_inputs(previous, gamma, bias)
        if (previous.shape[0], *previous.shape[2:]) != (n, height, width, channels):
            raise ValueError("Previous cache must match input N,H,W,C")
        if previous.device != x.device:
            raise ValueError("Previous cache must share the input CUDA device")
        previous_frames = previous.shape[1]
        if not 1 <= previous_frames <= min(CACHE_FRAMES, padding[0]):
            raise ValueError(
                "Previous cache length must be 1..min(2, temporal padding)"
            )
    cache_frames = min(CACHE_FRAMES, frames + previous_frames)
    padded = torch.empty(
        (
            n,
            frames + padding[0],
            height + 2 * padding[1],
            width + 2 * padding[2],
            channels,
        ),
        device=x.device,
        dtype=x.dtype,
    )
    cache = torch.empty(
        (n, cache_frames, height, width, channels), device=x.device, dtype=x.dtype
    )
    residual_out = torch.empty_like(x) if save_input or residual is not None else None
    if n:
        with torch.cuda.device(x.device):
            stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
            compile_conv_prep(
                channels,
                bias is not None,
                contiguous_reduction,
                input_bias is not None,
                residual is not None,
                (frames, height, width),
                has_residual_bias=residual_bias is not None,
                save_input=save_input,
            )(
                x.view(-1, channels),
                gamma,
                gamma if bias is None else bias,
                gamma if input_bias is None else input_bias,
                (x if residual is None else residual).view(-1, channels),
                gamma if residual_bias is None else residual_bias,
                (x if residual_out is None else residual_out).view(-1, channels),
                (x if previous is None else previous).view(-1, channels),
                padded.view(-1, channels),
                cache.view(-1, channels),
                previous_frames,
                cache_frames,
                stream,
            )
    return PreparedConvInput(padded, cache, residual_out)


def torch_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    previous: torch.Tensor | None = None,
    *,
    input_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    save_input: bool = False,
) -> PreparedConvInput:
    """Torch math reference, including every BF16 rounding and cache output.

    The encoder uses this contiguous-channel reduction order. An explicit
    legacy interleaved-reduction comparison can differ near BF16 rounding ties.
    """
    value = x if input_bias is None else x + input_bias
    if residual is not None:
        value = value + (
            residual if residual_bias is None else residual + residual_bias
        )
    affine = torch.nn.functional.normalize(value.float(), dim=-1).to(x.dtype)
    affine = affine * x.shape[-1] ** 0.5 * gamma
    if bias is not None:
        affine = affine + bias
    y = torch.nn.functional.silu(affine)
    joined = y if previous is None else torch.cat((previous, y), dim=1)
    history = 0 if previous is None else previous.shape[1]
    return PreparedConvInput(
        torch.nn.functional.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0)),
        joined[:, -2:].contiguous(),
        value if residual is not None or save_input else None,
    )


def compile() -> None:
    """Compile the live widths, post-attention reduction, and residual modes."""
    compile_rmsnorm(640)
    for channels in (160, 320, 640):
        for residual in (False, True):
            compile_conv_prep(
                channels,
                has_input_bias=True,
                has_residual=residual,
                has_residual_bias=residual and channels != 160,
                save_input=True,
            )
    compile_conv_prep(
        640,
        contiguous_reduction=True,
        has_input_bias=True,
        has_residual=True,
        save_input=True,
    )


@torch.inference_mode()
def verify() -> None:
    """Check first/cached chunks on small, untimed inputs."""
    torch.manual_seed(42)
    for channels in (160, 320, 640):
        gamma = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn_like(gamma)
        for frames, history, has_residual in (
            (1, 0, False),
            (1, 1, True),
            (4, 2, True),
        ):
            x = torch.randn(2, frames, 3, 5, channels, device="cuda", dtype=gamma.dtype)
            previous = (
                torch.randn(2, history, 3, 5, channels, device="cuda", dtype=x.dtype)
                if history
                else None
            )
            kwargs = {
                "input_bias": bias,
                "residual": torch.randn_like(x) if has_residual else None,
                "residual_bias": bias if has_residual and channels != 160 else None,
                "save_input": True,
            }
            actual = rmsnorm_silu_conv_prep(x, gamma, previous=previous, **kwargs)
            expected = torch_reference(x, gamma, previous=previous, **kwargs)
            torch.testing.assert_close(
                actual.padded, expected.padded, atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                actual.cache, expected.cache, atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                actual.residual, expected.residual, atol=0, rtol=0
            )
    x = torch.randn(1, 4, 40, 30, 640, device="cuda", dtype=torch.bfloat16)
    expected = (
        torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype) * 640**0.5 * gamma
    )
    torch.testing.assert_close(rmsnorm(x, gamma), expected, atol=1e-3, rtol=1e-2)
    # Vector accesses must also accept 2B-aligned offset views and a
    # partial final CTA. Exercise every optional affine/residual input here.
    for channels, frames, history in (
        (c, t, p) for c in (160, 320, 640) for t, p in ((1, 0), (1, 1), (4, 2))
    ):
        shape = (1, frames, 3, 5, channels)
        x = torch.randn(
            frames * 3 * 5 * channels + 1, device="cuda", dtype=torch.bfloat16
        )[1:].view(shape)
        gamma = torch.randn(channels + 1, device="cuda", dtype=x.dtype)[1:]
        bias = torch.randn(channels + 1, device="cuda", dtype=x.dtype)[1:]
        previous = (
            torch.randn(history * 3 * 5 * channels + 1, device="cuda", dtype=x.dtype)[
                1:
            ].view(1, history, 3, 5, channels)
            if history
            else None
        )
        residual = torch.randn(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view(
            shape
        )
        for zero in (False, True):
            if zero:
                x.zero_()
                residual.zero_()
                bias.zero_()
            kwargs = {
                "bias": bias,
                "input_bias": bias,
                "residual": residual,
                "residual_bias": bias,
            }
            actual = rmsnorm_silu_conv_prep(x, gamma, previous=previous, **kwargs)
            expected = torch_reference(x, gamma, previous=previous, **kwargs)
            torch.testing.assert_close(
                actual.padded, expected.padded, atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                actual.cache, expected.cache, atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                actual.residual, expected.residual, atol=0, rtol=0
            )
    print(
        "Normalization/preparation: PASS (Torch values, padding, cache, saved skip)",
        flush=True,
    )


@torch.inference_mode()
def benchmark_production() -> None:
    """Time batch-32 preparation and attention normalization with eager Torch.

    Use the 480x832 encode shapes, including both standalone C160 call sites.
    Include preparations used for the separate side of conv-fusion
    comparisons. Cold chunks have one frame and no history; steady-state chunks
    have four, two, or one frame depending on temporal-downsample depth.
    """
    torch.manual_seed(42)
    print(
        "\nProduction benchmark: batch=32, BF16, NTHWC; Torch reference is eager."
        "\nMedian GPU ms; 5 alternating samples x 20 CUDA-graph replays."
        "\nCompilation and input generation excluded; speedup=reference/custom.",
        flush=True,
    )
    for c, frames, h, w, save_input in (
        (160, 4, 240, 416, True),
        (160, 4, 120, 208, False),
        (320, 4, 120, 208, True),
        (640, 2, 60, 104, True),
        (640, 1, 30, 52, True),
    ):
        for t, p in ((1, 0), (frames, 2)):
            shape = (32, t, h, w, c)
            print(
                f"\nInput NTHWC={shape} | history={p} | reduction=channels-last",
                flush=True,
            )
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(c, device="cuda", dtype=x.dtype)
            bias = torch.randn_like(gamma)
            previous = (
                torch.randn(32, p, h, w, c, device="cuda", dtype=x.dtype) if p else None
            )
            results = []
            for has_residual in (False, True) if save_input else (False,):
                kwargs = {
                    "input_bias": bias if save_input else None,
                    "residual": torch.randn_like(x) if has_residual else None,
                    "residual_bias": bias
                    if has_residual and (c == 320 or frames == 2)
                    else None,
                    "save_input": save_input,
                }

                reference = partial(
                    torch_reference, x, gamma, previous=previous, **kwargs
                )
                custom = partial(
                    rmsnorm_silu_conv_prep,
                    x,
                    gamma,
                    previous=previous,
                    **kwargs,
                )

                actual = custom()
                expected = reference()
                torch.testing.assert_close(
                    actual.padded, expected.padded, atol=2e-3, rtol=2e-2
                )
                torch.testing.assert_close(
                    actual.cache, expected.cache, atol=2e-3, rtol=2e-2
                )
                torch.testing.assert_close(
                    actual.residual, expected.residual, atol=0, rtol=0
                )
                del actual, expected
                operation = (
                    "bias + RMSNorm/SiLU/prep" if save_input else "RMSNorm/SiLU/prep"
                )
                if has_residual:
                    operation = "bias + residual + RMSNorm/SiLU/prep"
                    if kwargs["residual_bias"] is not None:
                        operation += " + skip bias"
                if save_input or has_residual:
                    operation += " + saved skip"
                results.append(measure(operation, reference, custom))
            print_benchmark_table(results)

    x = torch.randn(32, 1, 30, 52, 640, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(640, device="cuda", dtype=x.dtype)

    def attention_reference() -> torch.Tensor:
        return (
            torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype)
            * 640**0.5
            * gamma
        )

    expected = attention_reference()
    torch.testing.assert_close(rmsnorm(x, gamma), expected, atol=1e-3, rtol=1e-2)
    print(
        f"\nAttention input NTHWC={tuple(x.shape)} (same shape in every chunk)",
        flush=True,
    )
    print_benchmark_table(
        [measure("attention RMSNorm", attention_reference, lambda: rmsnorm(x, gamma))]
    )
    print("Production normalization/preparation correctness: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-compile", action="store_true")
    args = parser.parse_args()
    if args.only_compile:
        compile()
        print("Normalization/preparation compile: PASS", flush=True)
    else:
        print("Correctness checks (small cases, untimed):", flush=True)
        verify()
        benchmark_production()
