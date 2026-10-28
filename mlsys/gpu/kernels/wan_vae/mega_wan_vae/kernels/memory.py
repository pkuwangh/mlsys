# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
# pyright: reportMissingImports=false, reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false

"""Direct causal padding/history/cache assembly for BF16 or FP32 Wan activations.

Output and history are NTHWC. The input may be NTHWC or the initial NCTHW
encoder tensor; the latter conversion is fused into the same pass. The generic
path assigns eight adjacent channels per thread for NTHWC tensors; the initial
input path gathers a pixel's channels into register vectors. Initial downsampling uses a
vector bias-add and shares the output/cache buffer. No shared memory, barriers,
input mutation, or intermediate concatenation is needed.
NCTHW inputs may be temporal views with noncompact batch/channel strides;
packing reads those strides directly, without an intermediate contiguous copy.
The initial C12 NCTHW input can be padded to C16 for aligned convolution gathers;
its cache remains C12 and every extra output channel is zero.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__:
    from ._compile import compile_cache, compile_kernel
    from ._dtype import storage_type
    from ._geometry import SpatialDivisors
    from ._utils import measure, print_benchmark_table
else:
    from _compile import compile_cache, compile_kernel
    from _dtype import storage_type
    from _geometry import SpatialDivisors
    from _utils import measure, print_benchmark_table


@dataclass(frozen=True)
class PackedHistory:
    """NTHWC convolution input and history; consumers must not mutate either.

    The initial unpadded one-frame downsample aliases output and cache. Neither
    aliases the caller's input; subsequent calls never mutate saved history.
    """

    padded: torch.Tensor
    cache: torch.Tensor


@cute.kernel
def input_pack_kernel(
    x: cute.Tensor,
    previous: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    out_numel: cutlass.Int64,
    T,
    H,
    W,
    P,
    CT,
    STRIDES,
    divisors,
    OUTPUT_CHANNELS: cutlass.Constexpr[int],
) -> None:
    """Pack C12 input: out=pad(concat(previous, NTHWC(x))), cache=last_CT.

    All other output elements are zero. Spatially adjacent lanes load the same
    channel, coalescing NCTHW reads; register gathering yields C12 or C16 output.
    C12 cache pixels are 24 bytes apart in BF16 and 48 bytes in FP32;
    their vector stores promise only 8-byte alignment. One thread owns each
    output pixel and its optional cache pixel.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    pixel = cutlass.Int64(bid) * 256 + tid
    if pixel < out_numel // OUTPUT_CHANNELS:
        n, t, h, w = divisors.coordinates(pixel, promote_batch=False)
        t -= 2
        h -= 1
        w -= 1
        values = cutlass.Array(x.element_type, 16, space=cutlass.AddressSpace.rmem)
        values.store(cutlass.vector.full((16,), 0, x.element_type), 0)
        if h >= 0 and h < H and w >= 0 and w < W:
            if t >= 0:
                sn, sc, st, sh, sw = STRIDES
                for c in cutlass.range_constexpr(12):
                    values[c] = x[n * sn + c * sc + t * st + h * sh + w * sw]
            elif t >= -P:
                offset = cutlass.Int64(((n * P + t + P) * H + h) * W + w) * 12
                for part in cutlass.range_constexpr(3):
                    ptr = previous.iterator.raw_ptr() + offset + part * 4
                    if (previous.iterator.toint() & 7) == 0:
                        values.store(ptr.load(count=4, alignment=8), part * 4)
                    else:
                        values.store(ptr.load(count=4, alignment=2), part * 4)
            if t >= T - CT:
                cache_offset = cutlass.Int64(((n * CT + t - T + CT) * H + h) * W + w) * 12
                for part in cutlass.range_constexpr(3):
                    (cache.iterator.raw_ptr() + cache_offset + part * 4).store(values.load(part * 4, 4), alignment=8)
        if cutlass.const_expr(OUTPUT_CHANNELS == 16):
            for part in cutlass.range_constexpr(2):
                (out.iterator.raw_ptr() + pixel * 16 + part * 8).store(values.load(part * 8, 8), alignment=16)
        else:
            for part in cutlass.range_constexpr(3):
                (out.iterator.raw_ptr() + pixel * 12 + part * 4).store(values.load(part * 4, 4), alignment=8)


@cute.kernel
def initial_bias_kernel(
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    out_numel: cutlass.Int64,
    C: cutlass.Constexpr[int],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """out[i]=BF16(FP32(x[i])+FP32(bias[i%C])); output also owns the T1 cache.

    Eight adjacent elements per thread; C is divisible by eight. Offset views
    use the conservative two-byte alignment specialization.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    offset = (cutlass.Int64(bid) * 256 + tid) * 8
    if offset < out_numel:
        value = (x.iterator.raw_ptr() + offset).load(count=8, alignment=ALIGNMENT)
        channel_bias = (bias.iterator.raw_ptr() + offset % C).load(count=8, alignment=ALIGNMENT)
        result = (value.to(cutlass.Float32) + channel_bias.to(cutlass.Float32)).to(x.element_type)
        (out.iterator.raw_ptr() + offset).store(result, alignment=16)


@cute.jit
def initial_bias_host(
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    out_numel: cutlass.Int64,
    stream: cuda.CUstream,
    channels: cutlass.Constexpr[int],
    alignment: cutlass.Constexpr[int],
) -> None:
    initial_bias_kernel(x, bias, out, out_numel, channels, alignment).launch(
        grid=((out_numel // 8 + 255) // 256, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@compile_cache()
def compile_initial_bias(channels: int, alignment: int, storage_dtype: torch.dtype = torch.bfloat16) -> Callable:
    """Initial unpadded one-frame bias addition, with shared output/cache storage."""
    x = make_fake_compact_tensor(storage_type(storage_dtype), (cute.sym_int64(),), assumed_align=2)
    bias = make_fake_compact_tensor(storage_type(storage_dtype), (channels,), assumed_align=2)
    out = make_fake_compact_tensor(storage_type(storage_dtype), (cute.sym_int64(divisibility=8),), assumed_align=16)
    return compile_kernel(
        initial_bias_host,
        x,
        bias,
        out,
        cutlass.Int64(1),
        make_fake_stream(),
        channels,
        alignment,
        options="--enable-tvm-ffi",
    )


@cute.kernel
def causal_pack_kernel(
    x: cute.Tensor,
    previous: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    out_numel: cutlass.Int64,
    T,
    H,
    W,
    C: cutlass.Constexpr[int],
    P,
    PT: cutlass.Constexpr[int],
    PH: cutlass.Constexpr[int],
    PW: cutlass.Constexpr[int],
    CT,
    NCTHW: cutlass.Constexpr[bool],
    HAS_BIAS: cutlass.Constexpr[bool],
    INPUT_STRIDES,
    divisors,
    VECTOR: cutlass.Constexpr[int] = 1,
    ALIGNMENT: cutlass.Constexpr[int] = 2,
    PAD_INPUT_CHANNELS: cutlass.Constexpr[bool] = True,
) -> None:
    """Compute out=pad(concat(previous, B(x+bias))), cache=last_CT_frames.

    B is BF16 rounding. Bias applies only to current input, not saved history.
    Missing history and spatial padding are zero; missing history does not
    enter the cache. P<=PT and CT<=P+T ensure every cache element is written
    by exactly one in-bounds output owner.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    OC: cutlass.Constexpr = 16 if NCTHW and C == 12 and PAD_INPUT_CHANNELS else C
    i = (cutlass.Int64(bid) * 256 + tid) * VECTOR
    if i < out_numel:
        c = i % OC
        n, t, h, w = divisors.coordinates(i // OC, promote_batch=False)
        t -= PT
        h -= PH
        w -= PW
        value = cutlass.vector.full((VECTOR,), 0, x.element_type)
        spatial = h >= 0 and h < H and w >= 0 and w < W
        if spatial and c < C:
            if t >= 0:
                offset = cutlass.Int64(((n * T + t) * H + h) * W + w) * C + c
                if NCTHW:
                    sn, sc, st, sh, sw = INPUT_STRIDES
                    offset = n * sn + c * sc + t * st + h * sh + w * sw
                value = (x.iterator.raw_ptr() + offset).load(count=VECTOR, alignment=ALIGNMENT)
                if HAS_BIAS:
                    channel_bias = (bias.iterator.raw_ptr() + c).load(count=VECTOR, alignment=ALIGNMENT)
                    value = (value.to(cutlass.Float32) + channel_bias.to(cutlass.Float32)).to(x.element_type)
            elif t >= -P:
                offset = cutlass.Int64(((n * P + t + P) * H + h) * W + w) * C + c
                value = (previous.iterator.raw_ptr() + offset).load(count=VECTOR, alignment=ALIGNMENT)
        (out.iterator.raw_ptr() + i).store(value, alignment=ALIGNMENT)
        if spatial and c < C and t >= T - CT:
            offset = cutlass.Int64(((n * CT + t - T + CT) * H + h) * W + w) * C + c
            (cache.iterator.raw_ptr() + offset).store(value, alignment=ALIGNMENT)


@dataclass(frozen=True)
class PackConfig:
    """Capabilities that change generated code; no tensor extents or strides."""

    channels: int
    padding: tuple[int, int, int]
    input_ncthw: bool
    has_bias: bool
    vector: int
    wide_index: bool = False
    pad_input_channels: bool = True


@cute.jit
def causal_pack_host(
    x: cute.Tensor,
    previous: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    out_numel: cutlass.Int64,
    geometry,
    input_strides,
    stream: cuda.CUstream,
    config: cutlass.Constexpr,
) -> None:
    t, h, w, p, ct = geometry
    pt, ph, pw = config.padding
    c = config.channels
    divisors = SpatialDivisors(t + pt, h + 2 * ph, w + 2 * pw, wide=config.wide_index)
    if cutlass.const_expr(config.input_ncthw and c == 12 and not config.has_bias and config.padding == (2, 1, 1)):
        output_channels = 16 if config.pad_input_channels else 12
        input_pack_kernel(
            x, previous, out, cache, out_numel, t, h, w, p, ct, input_strides, divisors, output_channels
        ).launch(
            grid=((out_numel // output_channels + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
        return
    vector = config.vector
    addresses = x.iterator.toint() | out.iterator.toint() | cache.iterator.toint()
    if p > 0:
        addresses = addresses | previous.iterator.toint()
    if cutlass.const_expr(config.has_bias):
        addresses = addresses | bias.iterator.toint()
    if vector >= 8 and (addresses & 15) == 0:
        causal_pack_kernel(
            x,
            previous,
            bias,
            out,
            cache,
            out_numel,
            t,
            h,
            w,
            c,
            p,
            pt,
            ph,
            pw,
            ct,
            config.input_ncthw,
            config.has_bias,
            input_strides,
            divisors,
            VECTOR=vector,
            ALIGNMENT=16,
            PAD_INPUT_CHANNELS=config.pad_input_channels,
        ).launch(
            grid=((out_numel + 256 * vector - 1) // (256 * vector), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    else:
        causal_pack_kernel(
            x,
            previous,
            bias,
            out,
            cache,
            out_numel,
            t,
            h,
            w,
            c,
            p,
            pt,
            ph,
            pw,
            ct,
            config.input_ncthw,
            config.has_bias,
            input_strides,
            divisors,
            VECTOR=vector,
            ALIGNMENT=2,
            PAD_INPUT_CHANNELS=config.pad_input_channels,
        ).launch(
            grid=((out_numel + 256 * vector - 1) // (256 * vector), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@compile_cache()
def compile_pack(config: PackConfig, storage_dtype: torch.dtype = torch.bfloat16) -> Callable:
    """Reuse packing code across extents, chunk lengths, and physical strides."""
    tensors = [
        make_fake_compact_tensor(storage_type(storage_dtype), (cute.sym_int64(),), assumed_align=2) for _ in range(4)
    ]
    bias = make_fake_compact_tensor(storage_type(storage_dtype), (config.channels,), assumed_align=2)
    return compile_kernel(
        causal_pack_host,
        tensors[0],
        tensors[1],
        bias,
        tensors[2],
        tensors[3],
        cutlass.Int64(1),
        tuple((cutlass.Int64 if config.wide_index else cutlass.Int32)(1) for _ in range(5)),
        tuple(cutlass.Int64(1) for _ in range(5)),
        make_fake_stream(),
        config,
        options="--enable-tvm-ffi",
    )


def pack_history(
    x: torch.Tensor,
    previous: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    padding: tuple[int, int, int] = (2, 1, 1),
    cache_frames: int = 2,
    input_ncthw: bool = False,
    pad_input_channels: bool = True,
) -> PackedHistory:
    """Fuse conversion/bias, padding, and cache writes.

    Initial NCTHW C12 input pads to C16 by default for Tensor Core gathers.
    Set pad_input_channels=False for a library convolution that consumes C12.

    NCTHW input accepts nonnegative-stride views (including temporal slices).
    History, bias, and NTHWC input remain contiguous. The flattened input view
    spans storage gaps but the kernel reads only coordinates in the input view.
    """
    if type(pad_input_channels) is not bool:
        raise TypeError("pad_input_channels must be a bool")
    if x.ndim != 5:
        raise ValueError("Expected a five-dimensional activation")
    n, t, h, w, c = x.permute(0, 2, 3, 4, 1).shape if input_ncthw else x.shape
    pt, ph, pw = padding
    p = 0 if previous is None else previous.shape[1]
    if min(n, t, h, w, c) <= 0 or min(padding) < 0 or p > pt or cache_frames < 1:
        raise ValueError("Invalid activation, padding, or cache geometry")
    if previous is not None and previous.shape != (n, p, h, w, c):
        raise ValueError("History must be NTHWC with matching batch/spatial/channels")
    if bias is not None and bias.shape != (c,):
        raise ValueError("Bias must be a C-vector")
    for value in (x, previous, bias):
        if value is None:
            continue
        if (
            value.device != x.device
            or value.device.type != "cuda"
            or value.dtype != x.dtype
            or value.dtype not in (torch.bfloat16, torch.float32)
            or (not value.is_contiguous() and not (value is x and input_ncthw))
        ):
            raise ValueError(
                "Pack inputs must share BF16 or FP32 dtype on one CUDA device; only NCTHW input may be strided"
            )
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError("History packing is inference-only")
    ct = min(cache_frames, p + t)
    out_channels = 16 if input_ncthw and c == 12 and pad_input_channels else c
    out = torch.empty(
        (n, t + pt, h + 2 * ph, w + 2 * pw, out_channels),
        device=x.device,
        dtype=x.dtype,
    )
    initial_bias = not input_ncthw and t == 1 and p == 0 and padding == (0, 0, 0) and bias is not None and c % 8 == 0
    cache = out if initial_bias else torch.empty((n, ct, h, w, c), device=x.device, dtype=x.dtype)
    input_strides = tuple(x.stride()) if input_ncthw else None
    if input_strides is not None and min(input_strides) < 0:
        raise ValueError("NCTHW input strides must be nonnegative")
    input_span = 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    flat_input = x.as_strided((input_span,), (1,))
    with torch.cuda.device(x.device):
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        if initial_bias:
            alignment = 16 if x.data_ptr() % 16 == 0 and bias.data_ptr() % 16 == 0 else 2
            compile_initial_bias(c, alignment, storage_dtype=x.dtype)(
                flat_input, bias, out.view(-1), out.numel(), stream
            )
            return PackedHistory(out, cache)
        channel_contiguous = not input_ncthw or (
            input_strides[1] == 1 and all(stride % 8 == 0 for axis, stride in enumerate(input_strides) if axis != 1)
        )
        config = PackConfig(
            c,
            padding,
            input_ncthw,
            bias is not None,
            8 if channel_contiguous and c % 8 == 0 else 1,
            n * (t + pt) * (h + 2 * ph) * (w + 2 * pw) >= 2**31,
            pad_input_channels,
        )
        compile_pack(config, storage_dtype=x.dtype)(
            flat_input,
            flat_input if previous is None else previous.view(-1),
            cache.view(-1)[:c] if bias is None else bias,
            out.view(-1),
            cache.view(-1),
            out.numel(),
            (t, h, w, p, ct),
            input_strides or (0, 0, 0, 0, 0),
            stream,
        )
    return PackedHistory(out, cache)


def compile(storage_dtype: torch.dtype = torch.bfloat16) -> None:
    """Precompile common encoder packing variants using the runtime cache keys."""
    for channels in (320, 640):
        for alignment in (2, 16):
            compile_initial_bias(channels, alignment, storage_dtype=storage_dtype)
    for channels, ncthw, padding, vector in (
        (12, True, (2, 1, 1), 1),
        (48, True, (2, 1, 1), 1),
        (48, True, (2, 1, 1), 8),
        (320, False, (1, 0, 0), 8),
    ):
        compile_pack(PackConfig(channels, padding, ncthw, not ncthw, vector), storage_dtype=storage_dtype)


@torch.inference_mode()
def verify() -> None:
    """Direct Torch comparison for causal zeros, history, bias, and both layouts."""
    torch.manual_seed(43)
    for c, ncthw, pad, history in (
        (12, True, (2, 1, 1), 2),
        (48, True, (2, 1, 1), 2),
        (320, False, (1, 0, 0), 1),
    ):
        for t, p in [(t, p) for t in (1, 4) for p in range(history + 1)]:
            x = torch.randn(2, t, 8, 6, c, device="cuda", dtype=torch.bfloat16)
            previous = (
                torch.randn(2 * p * 8 * 6 * c + 1, device="cuda", dtype=x.dtype)[1:].view(2, p, 8, 6, c) if p else None
            )
            bias = None if ncthw else torch.randn(c, device="cuda", dtype=x.dtype)
            current = x if bias is None else x + bias
            joined = current if previous is None else torch.cat((previous, current), dim=1)
            pt, ph, pw = pad
            expected = torch.nn.functional.pad(joined.permute(0, 4, 1, 2, 3), (pw, pw, ph, ph, pt - p, 0))
            packed = pack_history(
                x.permute(0, 4, 1, 2, 3).contiguous() if ncthw else x,
                previous,
                bias,
                padding=pad,
                cache_frames=history,
                input_ncthw=ncthw,
            )
            expected = expected.permute(0, 2, 3, 4, 1)
            if ncthw and c == 12:
                expected = torch.nn.functional.pad(expected, (0, 4))
            torch.testing.assert_close(packed.padded, expected, atol=0, rtol=0)
            torch.testing.assert_close(packed.cache, joined[:, -history:], atol=0, rtol=0)
            if ncthw:
                for layout in (torch.contiguous_format, torch.channels_last_3d):
                    storage = torch.full(
                        (2, c, t + 5, 8, 6),
                        float("nan"),
                        device=x.device,
                        dtype=x.dtype,
                    ).contiguous(memory_format=layout)
                    sliced = storage[:, :, 2 : 2 + t]
                    sliced.copy_(x.permute(0, 4, 1, 2, 3))
                    strided = pack_history(
                        sliced,
                        previous,
                        bias,
                        padding=pad,
                        cache_frames=history,
                        input_ncthw=True,
                    )
                    torch.testing.assert_close(strided.padded, expected, atol=0, rtol=0)
                    torch.testing.assert_close(strided.cache, joined[:, -history:], atol=0, rtol=0)
    # Offset/strided views and partial CTAs exercise both initial specializations.
    for c in (320, 640, 7):
        x = torch.randn(3 * 3 * 5 * c + 1, device="cuda", dtype=torch.bfloat16)[1:].view(1, 3, 3, 5, c)
        previous = torch.randn(3 * 5 * c + 1, device="cuda", dtype=x.dtype)[1:].view(1, 1, 3, 5, c)
        bias = torch.randn(c + 1, device="cuda", dtype=x.dtype)[1:]
        packed = pack_history(x, previous, bias, padding=(2, 1, 1))
        joined = torch.cat((previous, x + bias), dim=1)
        torch.testing.assert_close(
            packed.padded,
            torch.nn.functional.pad(joined, (0, 0, 1, 1, 1, 1, 1, 0)),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(packed.cache, joined[:, -2:], atol=0, rtol=0)
    for n, h, w in ((1, 3, 5), (2, 8, 6)):
        storage = torch.randn(n, 12, 7, h, w * 2 + 1, device="cuda", dtype=torch.bfloat16)
        x = storage[:, :, 2:3, :, 1 : 1 + 2 * w : 2]
        packed = pack_history(x, input_ncthw=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = pack_history(x, input_ncthw=True)
        for _ in range(2):
            x.add_(0.25)
            replayed.padded.fill_(float("nan"))
            replayed.cache.fill_(float("nan"))
            graph.replay()
            expected_cache = x.permute(0, 2, 3, 4, 1).contiguous()
            expected = torch.nn.functional.pad(expected_cache, (0, 4, 1, 1, 1, 1, 2, 0))
            torch.testing.assert_close(replayed.padded, expected, atol=0, rtol=0)
            torch.testing.assert_close(replayed.cache, expected_cache, atol=0, rtol=0)
        for c in (320, 640):
            x = torch.randn(n * h * w * c + 1, device="cuda", dtype=torch.bfloat16)[1:].view(n, 1, h, w, c)
            bias = torch.randn(c + 1, device="cuda", dtype=x.dtype)[1:]
            saved_input = x.clone()
            packed = pack_history(x, bias=bias, padding=(0, 0, 0), cache_frames=1)
            assert packed.padded.data_ptr() == packed.cache.data_ptr()
            assert packed.padded.data_ptr() != x.data_ptr()
            torch.testing.assert_close(packed.padded, x + bias, atol=0, rtol=0)
            torch.testing.assert_close(x, saved_input, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = pack_history(x, bias=bias, padding=(0, 0, 0), cache_frames=1)
            for _ in range(2):
                x.add_(0.25)
                bias.mul_(0.5)
                replayed.padded.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(replayed.padded, x + bias, atol=0, rtol=0)
                assert replayed.padded.data_ptr() == replayed.cache.data_ptr()
    print(
        "Causal pack/bias/cache vs Torch: PASS (including initial paths and graph replay)",
        flush=True,
    )


@torch.inference_mode()
def benchmark_production() -> None:
    """Time encoder input/downsample packing and batch-1 decoder latent packing.

    Initial input chunks are strided views into a patchified 17-frame NCTHW
    480x832 video, as in the encoder. Downsample inputs are already NTHWC:
    C320 at 60x104 before temporal reduction, C640 at 30x52 before the second.
    Initial downsample chunks have no history or temporal padding.
    Decoder input is a one-frame view into channels-last C48 latent storage.
    """
    torch.manual_seed(43)
    print(
        "\nProduction benchmark: encoder batch=32, decoder batch=1, BF16; Torch reference is eager."
        "\nMedian GPU ms from alternating CUDA-graph samples."
        "\nCompilation and input generation are excluded; speedup=reference/custom."
        "\nOperation labels give current N,T,H,W,C and previous-history frames."
        "\nInput pack includes NCTHW->NTHWC, padding and cache (C12 also pads to C16)."
        "\nDownsample pack includes bias, history and cache.",
        flush=True,
    )
    results = []
    for n, c, ncthw, t, h, w, p in (
        (32, 12, True, 1, 240, 416, 0),
        (32, 12, True, 4, 240, 416, 1),
        (32, 12, True, 4, 240, 416, 2),
        (32, 320, False, 1, 60, 104, 0),
        (32, 320, False, 4, 60, 104, 1),
        (32, 640, False, 1, 30, 52, 0),
        (32, 640, False, 2, 30, 52, 1),
        (1, 48, True, 1, 45, 80, 0),
        (1, 48, True, 1, 45, 80, 1),
        (1, 48, True, 1, 45, 80, 2),
    ):
        if ncthw:
            video = torch.randn(n, c, 5 if c == 48 else 17, h, w, device="cuda", dtype=torch.bfloat16)
            if c == 48:
                video = video.contiguous(memory_format=torch.channels_last_3d)
            start = p if c == 48 else 0 if p == 0 else 1 if p == 1 else 5
            x = video[:, :, start : start + t]
        else:
            x = torch.randn(n, t, h, w, c, device="cuda", dtype=torch.bfloat16)
        pad = (2, 1, 1) if ncthw else (p, 0, 0)
        cache_frames = 2 if ncthw else 1
        previous = torch.randn(n, p, h, w, c, device="cuda", dtype=x.dtype) if p else None
        bias = None if ncthw else torch.randn(c, device="cuda", dtype=x.dtype)

        def reference(
            x=x,
            ncthw=ncthw,
            bias=bias,
            previous=previous,
            pad=pad,
            c=c,
            p=p,
            cache_frames=cache_frames,
        ) -> PackedHistory:
            current = x.permute(0, 2, 3, 4, 1) if ncthw else x + bias
            joined = current if previous is None else torch.cat((previous, current), dim=1)
            pt, ph, pw = pad
            torch_padding = (0, 4 if c == 12 else 0, pw, pw, ph, ph, pt - p, 0)
            padded = torch.nn.functional.pad(joined, torch_padding) if any(torch_padding) else joined
            return PackedHistory(padded, joined[:, -cache_frames:].contiguous())

        def custom(
            x=x,
            previous=previous,
            bias=bias,
            pad=pad,
            cache_frames=cache_frames,
            ncthw=ncthw,
        ) -> PackedHistory:
            return pack_history(
                x,
                previous,
                bias,
                padding=pad,
                cache_frames=cache_frames,
                input_ncthw=ncthw,
            )

        actual, expected = custom(), reference()
        torch.testing.assert_close(actual.padded, expected.padded, atol=0, rtol=0)
        torch.testing.assert_close(actual.cache, expected.cache, atol=0, rtol=0)
        del actual, expected
        operation = "Latent pack" if c == 48 else "Input pack" if ncthw else "Downsample pack"
        results.append(measure(f"{operation} ({n},{t},{h},{w},{c}) history={p}", reference, custom))
    print_benchmark_table(results)
    print("Production pack/cache correctness: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-compile", action="store_true")
    args = parser.parse_args()
    if args.only_compile:
        compile()
        print("Causal pack compile: PASS", flush=True)
    else:
        print("Correctness checks (small cases, untimed):", flush=True)
        verify()
        benchmark_production()
