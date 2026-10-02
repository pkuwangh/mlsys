# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
# pyright: reportMissingImports=false, reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false

"""Direct causal padding/history/cache assembly for BF16 Wan activations.

Output and history are NTHWC. The input may be NTHWC or the initial NCTHW
encoder tensor; the latter conversion is fused into the same pass. The generic
path assigns eight adjacent channels per thread for NTHWC tensors; the initial
input path gathers a pixel's channels into register vectors. Initial downsampling uses a
vector bias-add and shares the output/cache buffer. No shared memory, barriers,
input mutation, or intermediate concatenation is needed.
NCTHW inputs may be temporal views with noncompact batch/channel strides;
packing reads those strides directly, without an intermediate contiguous copy.
The initial C12 NCTHW input is padded to C16 for aligned convolution gathers;
its cache remains C12 and every extra output channel is zero.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__:
    from ._utils import measure, print_benchmark_table
else:
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
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    P: cutlass.Constexpr[int],
    CT: cutlass.Constexpr[int],
    STRIDES: cutlass.Constexpr,
) -> None:
    """Pack C12 input: out=pad(concat(previous, NTHWC(x))), cache=last_CT.

    All other output elements are zero. Spatially adjacent lanes load the same
    channel, coalescing NCTHW reads; register gathering yields C16 output vectors.
    Cache pixels are 24 bytes apart, so their vector stores promise only 8-byte
    alignment. One thread owns each output pixel and its optional cache pixel.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    pixel = cutlass.Int64(bid) * 256 + tid
    if pixel < out.shape[0] // 16:
        w = pixel % (W + 2) - 1
        h = pixel // (W + 2) % (H + 2) - 1
        t = pixel // ((W + 2) * (H + 2)) % (T + 2) - 2
        n = pixel // ((T + 2) * (W + 2) * (H + 2))
        values = cutlass.Array(cutlass.BFloat16, 16, space=cutlass.AddressSpace.rmem)
        values.store(cutlass.vector.full((16,), 0, cutlass.BFloat16), 0)
        if h >= 0 and h < H and w >= 0 and w < W:
            if t >= 0:
                sn, sc, st, sh, sw = STRIDES
                for c in cutlass.range_constexpr(12):
                    values[c] = x[n * sn + c * sc + t * st + h * sh + w * sw]
            elif t >= -P:
                offset = (((n * P + t + P) * H + h) * W + w) * 12
                for part in cutlass.range_constexpr(3):
                    ptr = previous.iterator.raw_ptr() + offset + part * 4
                    if (previous.iterator.toint() & 7) == 0:
                        values.store(ptr.load(count=4, alignment=8), part * 4)
                    else:
                        values.store(ptr.load(count=4, alignment=2), part * 4)
            if t >= T - CT:
                cache_offset = (((n * CT + t - T + CT) * H + h) * W + w) * 12
                for part in cutlass.range_constexpr(3):
                    (cache.iterator.raw_ptr() + cache_offset + part * 4).store(
                        values.load(part * 4, 4), alignment=8
                    )
        for part in cutlass.range_constexpr(2):
            (out.iterator.raw_ptr() + pixel * 16 + part * 8).store(
                values.load(part * 8, 8), alignment=16
            )


@cute.kernel
def initial_bias_kernel(
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
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
    if offset < out.shape[0]:
        value = (x.iterator.raw_ptr() + offset).load(count=8, alignment=ALIGNMENT)
        channel_bias = (bias.iterator.raw_ptr() + offset % C).load(
            count=8, alignment=ALIGNMENT
        )
        result = (value.to(cutlass.Float32) + channel_bias.to(cutlass.Float32)).to(
            cutlass.BFloat16
        )
        (out.iterator.raw_ptr() + offset).store(result, alignment=16)


@cute.jit
def initial_pack_host(
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    stream: cuda.CUstream,
    SPEC: cutlass.Constexpr,
) -> None:
    ncthw, h, w, c, strides, alignment = SPEC
    if cutlass.const_expr(ncthw):
        input_pack_kernel(x, x, out, cache, 1, h, w, 0, 1, strides).launch(
            grid=((out.shape[0] // 16 + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    else:
        initial_bias_kernel(x, bias, out, c, alignment).launch(
            grid=((out.shape[0] // 8 + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@cache
def compile_initial_pack(spec: tuple) -> Callable:
    """Compile initial-chunk specializations without allocating device tensors."""
    x = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int64(),), assumed_align=2)
    bias = make_fake_compact_tensor(cutlass.BFloat16, (spec[3],), assumed_align=2)
    out = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int64(divisibility=8),), assumed_align=16
    )
    cache = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int64(),), assumed_align=8
    )
    return cute.compile(
        initial_pack_host,
        x,
        bias,
        out,
        cache,
        make_fake_stream(),
        spec,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


@cute.kernel
def causal_pack_kernel(
    x: cute.Tensor,
    previous: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    C: cutlass.Constexpr[int],
    P: cutlass.Constexpr[int],
    PT: cutlass.Constexpr[int],
    PH: cutlass.Constexpr[int],
    PW: cutlass.Constexpr[int],
    CT: cutlass.Constexpr[int],
    NCTHW: cutlass.Constexpr[bool],
    HAS_BIAS: cutlass.Constexpr[bool],
    INPUT_STRIDES: cutlass.Constexpr = None,
    VECTOR: cutlass.Constexpr[int] = 1,
    ALIGNMENT: cutlass.Constexpr[int] = 2,
) -> None:
    """Compute out=pad(concat(previous, B(x+bias))), cache=last_CT_frames.

    B is BF16 rounding. Bias applies only to current input, not saved history.
    Missing history and spatial padding are zero; missing history does not
    enter the cache. P<=PT and CT<=P+T ensure every cache element is written
    by exactly one in-bounds output owner.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    OC: cutlass.Constexpr = 16 if NCTHW and C == 12 else C
    i = (cutlass.Int64(bid) * 256 + tid) * VECTOR
    if i < out.shape[0]:
        c = i % OC
        w = i // OC % (W + 2 * PW) - PW
        h = i // (OC * (W + 2 * PW)) % (H + 2 * PH) - PH
        t = i // (OC * (W + 2 * PW) * (H + 2 * PH)) % (T + PT) - PT
        n = i // (OC * (W + 2 * PW) * (H + 2 * PH) * (T + PT))
        value = cutlass.vector.full((VECTOR,), 0, cutlass.BFloat16)
        spatial = h >= 0 and h < H and w >= 0 and w < W
        if spatial and c < C:
            if t >= 0:
                offset = (((n * T + t) * H + h) * W + w) * C + c
                if NCTHW:
                    offset = (((n * C + c) * T + t) * H + h) * W + w
                    if cutlass.const_expr(INPUT_STRIDES is not None):
                        sn, sc, st, sh, sw = INPUT_STRIDES
                        offset = n * sn + c * sc + t * st + h * sh + w * sw
                value = (x.iterator.raw_ptr() + offset).load(
                    count=VECTOR, alignment=ALIGNMENT
                )
                if HAS_BIAS:
                    channel_bias = (bias.iterator.raw_ptr() + c).load(
                        count=VECTOR, alignment=ALIGNMENT
                    )
                    value = (
                        value.to(cutlass.Float32) + channel_bias.to(cutlass.Float32)
                    ).to(cutlass.BFloat16)
            elif t >= -P:
                offset = (((n * P + t + P) * H + h) * W + w) * C + c
                value = (previous.iterator.raw_ptr() + offset).load(
                    count=VECTOR, alignment=ALIGNMENT
                )
        (out.iterator.raw_ptr() + i).store(value, alignment=ALIGNMENT)
        if spatial and c < C and t >= T - CT:
            offset = (((n * CT + t - T + CT) * H + h) * W + w) * C + c
            (cache.iterator.raw_ptr() + offset).store(value, alignment=ALIGNMENT)


@cute.jit
def causal_pack_host(
    x: cute.Tensor,
    previous: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    cache: cute.Tensor,
    stream: cuda.CUstream,
    SPEC: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(
        SPEC[9] and SPEC[3] == 12 and not SPEC[10] and SPEC[5:8] == (2, 1, 1)
    ):
        t, h, w, c, p = SPEC[:5]
        ct = SPEC[8]
        strides = SPEC[11] if len(SPEC) > 11 else None
        if cutlass.const_expr(strides is None):
            strides = (c * t * h * w, t * h * w, h * w, w, 1)
        input_pack_kernel(x, previous, out, cache, t, h, w, p, ct, strides).launch(
            grid=((out.shape[0] // 16 + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
        return
    vector = 8 if not SPEC[9] and SPEC[3] % 8 == 0 else 1
    addresses = x.iterator.toint() | out.iterator.toint() | cache.iterator.toint()
    if cutlass.const_expr(SPEC[4] > 0):
        addresses = addresses | previous.iterator.toint()
    if cutlass.const_expr(SPEC[10]):
        addresses = addresses | bias.iterator.toint()
    if vector == 8 and (addresses & 15) == 0:
        causal_pack_kernel(
            x, previous, bias, out, cache, *SPEC, VECTOR=vector, ALIGNMENT=16
        ).launch(
            grid=((out.shape[0] + 256 * vector - 1) // (256 * vector), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    else:
        causal_pack_kernel(
            x, previous, bias, out, cache, *SPEC, VECTOR=vector, ALIGNMENT=2
        ).launch(
            grid=((out.shape[0] + 256 * vector - 1) // (256 * vector), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@cache
def compile_pack(spec: tuple) -> Callable:
    """Compile only immutable geometry/layout metadata; never cache tensors."""
    tensors = [
        make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int64(),), assumed_align=2)
        for _ in range(4)
    ]
    bias = make_fake_compact_tensor(cutlass.BFloat16, (spec[3],), assumed_align=2)
    return cute.compile(
        causal_pack_host,
        tensors[0],
        tensors[1],
        bias,
        tensors[2],
        tensors[3],
        make_fake_stream(),
        spec,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


def pack_history(
    x: torch.Tensor,
    previous: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    padding: tuple[int, int, int] = (2, 1, 1),
    cache_frames: int = 2,
    input_ncthw: bool = False,
) -> PackedHistory:
    """Fuse conversion/bias, padding, and cache writes; initial C12 pads to C16.

    NCTHW input accepts nonnegative-stride views (including temporal slices).
    History, bias, and NTHWC input remain contiguous. The flattened input view
    spans storage gaps but the kernel reads only coordinates in the input view.
    """
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
            or value.dtype != torch.bfloat16
            or (not value.is_contiguous() and not (value is x and input_ncthw))
        ):
            raise ValueError(
                "Pack inputs must be BF16 on one CUDA device; "
                "only NCTHW input may be strided"
            )
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError("History packing is inference-only")
    ct = min(cache_frames, p + t)
    out_channels = 16 if input_ncthw and c == 12 else c
    out = torch.empty(
        (n, t + pt, h + 2 * ph, w + 2 * pw, out_channels),
        device=x.device,
        dtype=x.dtype,
    )
    initial_bias = (
        not input_ncthw
        and t == 1
        and p == 0
        and padding == (0, 0, 0)
        and bias is not None
        and c % 8 == 0
    )
    cache = (
        out
        if initial_bias
        else torch.empty((n, ct, h, w, c), device=x.device, dtype=x.dtype)
    )
    input_strides = tuple(x.stride()) if input_ncthw else None
    if input_strides is not None and min(input_strides) < 0:
        raise ValueError("NCTHW input strides must be nonnegative")
    input_span = 1 + sum(
        (size - 1) * stride for size, stride in zip(x.shape, x.stride())
    )
    flat_input = x.as_strided((input_span,), (1,))
    with torch.cuda.device(x.device):
        initial_input = (
            input_ncthw
            and t == 1
            and c == 12
            and p == 0
            and padding == (2, 1, 1)
            and bias is None
        )
        if initial_input or initial_bias:
            alignment = (
                16
                if initial_bias and x.data_ptr() % 16 == 0 and bias.data_ptr() % 16 == 0
                else 2
            )
            compile_initial_pack((initial_input, h, w, c, input_strides, alignment))(
                flat_input,
                cache.view(-1)[:c] if bias is None else bias,
                out.view(-1),
                cache.view(-1),
                cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
            )
            return PackedHistory(out, cache)
        compile_pack(
            (
                t,
                h,
                w,
                c,
                p,
                pt,
                ph,
                pw,
                ct,
                input_ncthw,
                bias is not None,
                input_strides,
            )
        )(
            flat_input,
            flat_input if previous is None else previous.view(-1),
            cache.view(-1)[:c] if bias is None else bias,
            out.view(-1),
            cache.view(-1),
            cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        )
    return PackedHistory(out, cache)


def compile() -> None:
    compile_initial_pack(
        (True, 8, 6, 12, (12 * 17 * 8 * 6, 17 * 8 * 6, 8 * 6, 6, 1), 2)
    )
    for channels in (320, 640):
        for alignment in (2, 16):
            compile_initial_pack((False, 8, 6, channels, None, alignment))
    for c, ncthw, pad, history in (
        (12, True, (2, 1, 1), 2),
        (320, False, (1, 0, 0), 1),
    ):
        for t, p in ((1, 0), (4, history)):
            compile_pack((t, 8, 6, c, p, *pad, min(history, p + t), ncthw, not ncthw))
            if ncthw:
                strides = (c * 17 * 8 * 6, 17 * 8 * 6, 8 * 6, 6, 1)
                compile_pack(
                    (t, 8, 6, c, p, *pad, min(history, p + t), True, False, strides)
                )


@torch.inference_mode()
def verify() -> None:
    """Direct Torch comparison for causal zeros, history, bias, and both layouts."""
    torch.manual_seed(43)
    for c, ncthw, pad, history in (
        (12, True, (2, 1, 1), 2),
        (320, False, (1, 0, 0), 1),
    ):
        for t, p in [(t, p) for t in (1, 4) for p in range(history + 1)]:
            x = torch.randn(2, t, 8, 6, c, device="cuda", dtype=torch.bfloat16)
            previous = (
                torch.randn(2 * p * 8 * 6 * c + 1, device="cuda", dtype=x.dtype)[
                    1:
                ].view(2, p, 8, 6, c)
                if p
                else None
            )
            bias = None if ncthw else torch.randn(c, device="cuda", dtype=x.dtype)
            current = x if bias is None else x + bias
            joined = (
                current if previous is None else torch.cat((previous, current), dim=1)
            )
            pt, ph, pw = pad
            expected = torch.nn.functional.pad(
                joined.permute(0, 4, 1, 2, 3), (pw, pw, ph, ph, pt - p, 0)
            )
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
            torch.testing.assert_close(
                packed.cache, joined[:, -history:], atol=0, rtol=0
            )
            if ncthw:
                storage = torch.full(
                    (2, c, t + 5, 8, 6), float("nan"), device=x.device, dtype=x.dtype
                )
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
                torch.testing.assert_close(
                    strided.cache, joined[:, -history:], atol=0, rtol=0
                )
    # Offset/strided views and partial CTAs exercise both initial specializations.
    for c in (320, 640, 7):
        x = torch.randn(3 * 3 * 5 * c + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view(1, 3, 3, 5, c)
        previous = torch.randn(3 * 5 * c + 1, device="cuda", dtype=x.dtype)[1:].view(
            1, 1, 3, 5, c
        )
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
        storage = torch.randn(
            n, 12, 7, h, w * 2 + 1, device="cuda", dtype=torch.bfloat16
        )
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
            x = torch.randn(n * h * w * c + 1, device="cuda", dtype=torch.bfloat16)[
                1:
            ].view(n, 1, h, w, c)
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
        "Causal pack/bias/cache vs Torch: PASS "
        "(including initial paths and graph replay)",
        flush=True,
    )


@torch.inference_mode()
def benchmark_production() -> None:
    """Time batch-32 input packing and both temporal-downsample callsites.

    Initial input chunks are strided views into a patchified 17-frame NCTHW
    480x832 video, as in the encoder. Downsample inputs are already NTHWC:
    C320 at 60x104 before temporal reduction, C640 at 30x52 before the second.
    Initial downsample chunks have no history or temporal padding.
    """
    torch.manual_seed(43)
    print(
        "\nProduction benchmark: batch=32, BF16; Torch reference is eager."
        "\nMedian GPU ms; 5 alternating samples x 20 CUDA-graph replays."
        "\nCompilation and input generation are excluded; speedup=reference/custom."
        "\nOperation labels give current T,H,W,C and previous-history frames."
        "\nInput pack includes NCTHW->NTHWC, C12->C16, padding and cache."
        "\nDownsample pack includes bias, history and cache.",
        flush=True,
    )
    results = []
    for c, ncthw, t, h, w, p in (
        (12, True, 1, 240, 416, 0),
        (12, True, 4, 240, 416, 1),
        (12, True, 4, 240, 416, 2),
        (320, False, 1, 60, 104, 0),
        (320, False, 4, 60, 104, 1),
        (640, False, 1, 30, 52, 0),
        (640, False, 2, 30, 52, 1),
    ):
        if ncthw:
            video = torch.randn(32, c, 17, h, w, device="cuda", dtype=torch.bfloat16)
            start = 0 if p == 0 else 1 if p == 1 else 5
            x = video[:, :, start : start + t]
        else:
            x = torch.randn(32, t, h, w, c, device="cuda", dtype=torch.bfloat16)
        pad = (2, 1, 1) if ncthw else (p, 0, 0)
        cache_frames = 2 if ncthw else 1
        previous = (
            torch.randn(32, p, h, w, c, device="cuda", dtype=x.dtype) if p else None
        )
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
            joined = (
                current if previous is None else torch.cat((previous, current), dim=1)
            )
            pt, ph, pw = pad
            torch_padding = (0, 16 - c if ncthw else 0, pw, pw, ph, ph, pt - p, 0)
            padded = (
                torch.nn.functional.pad(joined, torch_padding)
                if any(torch_padding)
                else joined
            )
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
        operation = "Input pack" if ncthw else "Downsample pack"
        results.append(
            measure(f"{operation} ({t},{h},{w},{c}) history={p}", reference, custom)
        )
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
