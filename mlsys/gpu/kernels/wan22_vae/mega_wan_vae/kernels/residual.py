# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportMissingImports=false
# pyright: reportOperatorIssue=false

"""Wan bias/residual additions and grouped stage shortcuts.

All activations are contiguous BF16 NTHWC. Bias, each residual branch, and
shortcut means are rounded separately before the final addition. Spatial
residual preparation writes the bottom/right zero border in the same pass.
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
    from ._utils import measure, print_benchmark_table
else:
    from _utils import measure, print_benchmark_table

THREADS = 128
EPILOGUE_VECTOR = 8


@cute.kernel
def bias_residual_kernel(
    x: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    out: cute.Tensor,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    HEIGHT: cutlass.Constexpr[int],
    WIDTH: cutlass.Constexpr[int],
    PAD_SPATIAL: cutlass.Constexpr[bool],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Fuse channel bias and residual addition over contiguous NTHWC storage.

    For element i, c = i % C, with FP32 additions and explicit BF16 rounding:
        biased_i = bf16(fp32(x_i) + fp32(bias_c))  [or x_i if absent]
        skip_i = bf16(fp32(residual_i) + fp32(residual_bias_c))  [or residual_i]
        out_i = bf16(fp32(biased_i) + fp32(skip_i))
    Do not reassociate the additions: the intermediate BF16 rounding is part
    of the Torch contract. Each thread owns eight adjacent elements; C must
    be a multiple of eight so neither bias loads nor the tail cross a row.
    Aligned inputs use 16-byte loads/stores; offset views retain a two-byte
    alignment specialization. Every address advances by eight BF16 elements
    because C is divisible by eight. No inputs are mutated.
    With PAD_SPATIAL, append one zero row and column on the bottom/right:
        out[n,t,h,w,c] = B(B(x[n,t,h,w,c] + bias[c]) + skip[n,t,h,w,c])
            for h < H and w < W; zero for h == H or w == W.
    Flattening N,T gives the NHWC input for the stride-2 spatial convolution.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    offset = (cutlass.Int64(bid) * THREADS + tid) * EPILOGUE_VECTOR
    if offset < out.shape[0]:
        input_offset = offset
        valid = cutlass.Boolean(True)
        if PAD_SPATIAL:
            row = offset // CHANNELS
            w = row % (WIDTH + 1)
            h = row // (WIDTH + 1) % (HEIGHT + 1)
            nt = row // ((WIDTH + 1) * (HEIGHT + 1))
            input_offset = (
                (nt * HEIGHT + h) * WIDTH + w
            ) * CHANNELS + offset % CHANNELS
            valid = h < HEIGHT and w < WIDTH
        if valid:
            value = (x.iterator.raw_ptr() + input_offset).load(
                count=EPILOGUE_VECTOR, alignment=ALIGNMENT
            )
            if HAS_BIAS:
                channel_bias = (bias.iterator.raw_ptr() + offset % CHANNELS).load(
                    count=EPILOGUE_VECTOR, alignment=ALIGNMENT
                )
                value = (
                    value.to(cutlass.Float32) + channel_bias.to(cutlass.Float32)
                ).to(cutlass.BFloat16)
            skip = (residual.iterator.raw_ptr() + input_offset).load(
                count=EPILOGUE_VECTOR, alignment=ALIGNMENT
            )
            if HAS_RESIDUAL_BIAS:
                skip_bias = (residual_bias.iterator.raw_ptr() + offset % CHANNELS).load(
                    count=EPILOGUE_VECTOR, alignment=ALIGNMENT
                )
                skip = (skip.to(cutlass.Float32) + skip_bias.to(cutlass.Float32)).to(
                    cutlass.BFloat16
                )
            result = (value.to(cutlass.Float32) + skip.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
            (out.iterator.raw_ptr() + offset).store(result, alignment=ALIGNMENT)
        else:
            zeros = cutlass.vector.full((EPILOGUE_VECTOR,), 0, cutlass.BFloat16)
            (out.iterator.raw_ptr() + offset).store(zeros, alignment=ALIGNMENT)


@cute.jit
def bias_residual_host(
    x: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    CHANNELS: cutlass.Constexpr[int],
    HAS_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
    HEIGHT: cutlass.Constexpr[int],
    WIDTH: cutlass.Constexpr[int],
    PAD_SPATIAL: cutlass.Constexpr[bool],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Launch one CTA per 1024 BF16 output elements on the caller's stream."""
    tile = THREADS * EPILOGUE_VECTOR
    bias_residual_kernel(
        x,
        bias,
        residual,
        residual_bias,
        out,
        CHANNELS,
        HAS_BIAS,
        HAS_RESIDUAL_BIAS,
        HEIGHT,
        WIDTH,
        PAD_SPATIAL,
        ALIGNMENT,
    ).launch(
        grid=((out.shape[0] + tile - 1) // tile, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@cache
def compile_bias_residual(
    channels: int = 160,
    has_bias: bool = True,
    spatial_shape: tuple[int, int] | None = None,
    *,
    has_residual_bias: bool = False,
    alignment: int = 16,
) -> Callable:
    """Cache shape-dynamic epilogue code by channel count and bias presence."""
    if channels <= 0 or channels % EPILOGUE_VECTOR:
        raise ValueError(
            "Bias/residual fusion requires C to be a positive multiple of 8"
        )
    if alignment not in (2, 16):
        raise ValueError("Bias/residual alignment must be 2 or 16 bytes")
    tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (cute.sym_int64(divisibility=EPILOGUE_VECTOR),),
        assumed_align=alignment,
    )
    out = make_fake_compact_tensor(
        cutlass.BFloat16,
        (cute.sym_int64(divisibility=EPILOGUE_VECTOR),),
        assumed_align=alignment,
    )
    weight = make_fake_compact_tensor(
        cutlass.BFloat16, (channels,), assumed_align=alignment
    )
    return cute.compile(
        bias_residual_host,
        tensor,
        weight,
        tensor,
        weight,
        out,
        make_fake_stream(),
        channels,
        has_bias,
        has_residual_bias,
        *(spatial_shape or (1, 1)),
        spatial_shape is not None,
        alignment,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


def bias_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    residual_bias: torch.Tensor | None = None,
    pad_spatial: bool = False,
) -> torch.Tensor:
    """Return BF16 bias-plus-residual, preserving contiguous NTHWC shape.

    Inputs must share shape, device and dtype; bias and residual_bias are optional
    contiguous C-vectors. Each branch is separately rounded to BF16 before their
    sum is rounded to BF16. Inference only, with no mutation; empty inputs do not
    launch. Channels must be a positive multiple of eight for vector accesses.
    Sixteen-byte-aligned inputs use wide accesses; misaligned offset views
    automatically use natural alignment without making a copy.
    pad_spatial appends one zero row/column for Wan's spatial downsampling.
    """
    if x.ndim != 5 or residual.shape != x.shape:
        raise ValueError("Bias/residual fusion requires equal NTHWC input shapes")
    channels = x.shape[-1]
    if channels <= 0 or channels % EPILOGUE_VECTOR:
        raise ValueError(
            "Bias/residual fusion requires C to be a positive multiple of 8"
        )
    for weight in (bias, residual_bias):
        if weight is not None and weight.shape != (channels,):
            raise ValueError("Convolution biases must be C-vectors")
    alignment = 16
    for tensor in (x, residual, bias, residual_bias):
        if tensor is None:
            continue
        if tensor.device.type != "cuda" or tensor.device != x.device:
            raise ValueError("Bias/residual tensors must share a CUDA device")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("Bias/residual tensors must be contiguous BF16")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("Bias/residual fusion is inference-only")
        if tensor.data_ptr() % 16:
            alignment = 2
    n, frames, height, width, _ = x.shape
    if pad_spatial and min(frames, height, width) <= 0:
        raise ValueError("Spatial downsample preparation requires positive T,H,W")
    out = torch.empty(
        (n, frames, height + pad_spatial, width + pad_spatial, channels),
        device=x.device,
        dtype=x.dtype,
    )
    if x.numel():
        with torch.cuda.device(x.device):
            stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
            compile_bias_residual(
                channels,
                bias is not None,
                (height, width) if pad_spatial else None,
                has_residual_bias=residual_bias is not None,
                alignment=alignment,
            )(
                x.view(-1),
                x.view(-1)[:channels] if bias is None else bias,
                residual.view(-1),
                x.view(-1)[:channels] if residual_bias is None else residual_bias,
                out.view(-1),
                stream,
            )
    return out


def torch_bias_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    residual_bias: torch.Tensor | None = None,
    pad_spatial: bool = False,
) -> torch.Tensor:
    """Torch reference with separately rounded BF16 additions."""
    value = (x if bias is None else x + bias) + (
        residual if residual_bias is None else residual + residual_bias
    )
    return torch.nn.functional.pad(value, (0, 0, 0, 1, 0, 1)) if pad_spatial else value


@cute.kernel
def stage_add_kernel(
    x: cute.Tensor,
    skip: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    CI: cutlass.Constexpr[int],
    CO: cutlass.Constexpr[int],
    FT: cutlass.Constexpr[int],
    FS: cutlass.Constexpr[int],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_BIAS: cutlass.Constexpr[bool],
) -> None:
    """Compute y = B(main + B(mean_g(shortcut))), with B = BF16 rounding.

    factor = FT*FS*FS; group = CI*factor/CO.
    For output (n,t,h,w,c), j=c*group+g, ci=j//factor,
    q=j%factor, dt=q//FS**2, dh=(q//FS)%FS, dw=q%FS:
      shortcut_g = skip[n, t*FT+dt-pad_t, h*FS+dh, w*FS+dw, ci]
    Out-of-range time contributes zero; pad_t=(-T)%FT.
    For deferred residual input, main=B(B(x+bias)+residual).
    Mean is accumulated in FP32 and rounded before the final BF16 addition.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bid) * 256 + tid
    OT = (T + FT - 1) // FT
    OH, OW = H // FS, W // FS
    FACTOR = FT * FS * FS
    GROUP = CI * FACTOR // CO
    if i < out.shape[0]:
        c = i % CO
        w = i // CO % OW
        h = i // (CO * OW) % OH
        t = i // (CO * OW * OH) % OT
        n = i // (CO * OW * OH * OT)
        total = cutlass.Float32(0)
        for g in cutlass.range_constexpr(GROUP):
            j = c * GROUP + g
            ci = j // FACTOR
            q = j % FACTOR
            ti = t * FT + q // (FS * FS) - ((FT - T % FT) % FT)
            hi = h * FS + q // FS % FS
            wi = w * FS + q % FS
            if ti >= 0:
                off = (((n * T + ti) * H + hi) * W + wi) * CI + ci
                total += cutlass.Float32(skip[off])
        pooled = cutlass.BFloat16(total / GROUP)
        value = cutlass.Float32(x[i])
        if HAS_BIAS:
            value = cutlass.Float32(cutlass.BFloat16(value + cutlass.Float32(bias[c])))
        if HAS_RESIDUAL:
            value = cutlass.Float32(
                cutlass.BFloat16(value + cutlass.Float32(residual[i]))
            )
        out[i] = cutlass.BFloat16(value + cutlass.Float32(pooled))


@cute.jit
def stage_add_vector(
    x: cute.Tensor,
    skip: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    CI: cutlass.Constexpr[int],
    CO: cutlass.Constexpr[int],
    FT: cutlass.Constexpr[int],
    FS: cutlass.Constexpr[int],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_BIAS: cutlass.Constexpr[bool],
    ALIGNMENT: cutlass.Constexpr[int],
) -> None:
    """Eight outputs per lane, preserving stage_add_kernel's ordered mean.

    When CO=2*CI, even and odd output channels use different temporal groups
    of the same input channel. Load four adjacent input channels per group,
    then interleave their results in registers before the vector output store.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = (cutlass.Int64(bid) * 256 + tid) * 8
    OT = (T + FT - 1) // FT
    OH, OW = H // FS, W // FS
    FACTOR = FT * FS * FS
    RATIO = CO // CI
    GROUP = FACTOR // RATIO
    if i < out.shape[0]:
        c = i % CO
        w = i // CO % OW
        h = i // (CO * OW) % OH
        t = i // (CO * OW * OH) % OT
        n = i // (CO * OW * OH * OT)
        pooled = cutlass.Array(cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem)
        for part in cutlass.range_constexpr(RATIO):
            total = cutlass.vector.full((8 // RATIO,), 0, cutlass.Float32)
            for g in cutlass.range_constexpr(GROUP):
                q = part * GROUP + g
                ti = t * FT + q // (FS * FS) - ((FT - T % FT) % FT)
                hi = h * FS + q // FS % FS
                wi = w * FS + q % FS
                if ti >= 0:
                    off = (((n * T + ti) * H + hi) * W + wi) * CI + c // RATIO
                    values = (skip.iterator.raw_ptr() + off).load(
                        count=8 // RATIO,
                        alignment=ALIGNMENT // RATIO if ALIGNMENT == 16 else 2,
                    )
                    total = total + values.to(cutlass.Float32)
            mean = (total / GROUP).to(cutlass.BFloat16)
            for channel in cutlass.range_constexpr(8 // RATIO):
                pooled[channel * RATIO + part] = mean[channel]
        value = (x.iterator.raw_ptr() + i).load(count=8, alignment=ALIGNMENT)
        if HAS_BIAS:
            b = (bias.iterator.raw_ptr() + c).load(count=8, alignment=ALIGNMENT)
            value = (value.to(cutlass.Float32) + b.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        if HAS_RESIDUAL:
            r = (residual.iterator.raw_ptr() + i).load(count=8, alignment=ALIGNMENT)
            value = (value.to(cutlass.Float32) + r.to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        result = (value.to(cutlass.Float32) + pooled.load(0, 8).to(cutlass.Float32)).to(
            cutlass.BFloat16
        )
        (out.iterator.raw_ptr() + i).store(result, alignment=ALIGNMENT)


@cute.kernel
def stage_add_vector_kernel(
    x: cute.Tensor,
    skip: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    SPEC: cutlass.Constexpr,
) -> None:
    addresses = x.iterator.toint() | skip.iterator.toint() | out.iterator.toint()
    if cutlass.const_expr(SPEC[7]):
        addresses = addresses | residual.iterator.toint()
    if cutlass.const_expr(SPEC[8]):
        addresses = addresses | bias.iterator.toint()
    if (addresses & 15) == 0:
        stage_add_vector(x, skip, bias, residual, out, *SPEC, 16)
    else:
        stage_add_vector(x, skip, bias, residual, out, *SPEC, 2)


@cute.jit
def stage_add_host(
    x: cute.Tensor,
    skip: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    SPEC: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(SPEC[3] % 8 == 0 and SPEC[4] in (SPEC[3], SPEC[3] * 2)):
        stage_add_vector_kernel(x, skip, bias, residual, out, SPEC).launch(
            grid=((out.shape[0] + 2047) // 2048, 1, 1), block=(256, 1, 1), stream=stream
        )
    else:
        stage_add_kernel(x, skip, bias, residual, out, *SPEC).launch(
            grid=((out.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )


@cache
def compile_stage(
    spec: tuple[int, int, int, int, int, int, int, bool, bool],
) -> Callable:
    """Cache code by scalar shape/factor metadata, never by tensor identity."""
    tensor = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int64(),), assumed_align=2
    )
    skip = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int64(),), assumed_align=2
    )
    bias = make_fake_compact_tensor(cutlass.BFloat16, (spec[4],), assumed_align=2)
    return cute.compile(
        stage_add_host,
        tensor,
        skip,
        bias,
        tensor,
        tensor,
        make_fake_stream(),
        spec,
        options="--enable-tvm-ffi --ptxas-options=--fmad=false",
    )


def stage_add(
    x: torch.Tensor,
    skip: torch.Tensor,
    ft: int,
    fs: int,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse the stage's grouped shortcut mean and main-branch addition."""
    if x.ndim != 5 or skip.ndim != 5 or ft not in (1, 2) or fs not in (1, 2):
        raise ValueError("Expected NTHWC tensors and Wan downsample factors")
    n, t, h, w, ci = skip.shape
    co = x.shape[-1]
    if min(t, h, w, ci, co) <= 0 or h % fs or w % fs or ci * ft * fs * fs % co:
        raise ValueError("Invalid stage shortcut geometry")
    if x.shape != (n, (t + ft - 1) // ft, h // fs, w // fs, co):
        raise ValueError("Main branch and reduced shortcut shapes disagree")
    for value in (x, skip, bias, residual):
        if value is None:
            continue
        if (
            value.device != x.device
            or value.device.type != "cuda"
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
        ):
            raise ValueError("Stage tensors must be contiguous BF16 on one CUDA device")
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError("Stage fusion is inference-only")
    if bias is not None and bias.shape != (co,):
        raise ValueError("Bias must be a C-vector")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("Residual shape must match the main branch")
    out = torch.empty_like(x)
    if out.numel():
        with torch.cuda.device(x.device):
            compile_stage(
                (t, h, w, ci, co, ft, fs, residual is not None, bias is not None)
            )(
                x.view(-1),
                skip.view(-1),
                x.view(-1)[:co] if bias is None else bias,
                (x if residual is None else residual).view(-1),
                out.view(-1),
                cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
            )
    return out


def torch_stage_add(
    x: torch.Tensor,
    skip: torch.Tensor,
    ft: int,
    fs: int,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch shortcut grouping in channel/time/height/width order."""
    n, t, h, w, ci = skip.shape
    padded = torch.nn.functional.pad(skip, (0, 0, 0, 0, 0, 0, (-t) % ft, 0))
    groups = padded.view(n, (t + ft - 1) // ft, ft, h // fs, fs, w // fs, fs, ci)
    groups = groups.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(*x.shape, -1)
    value = x if bias is None else x + bias
    if residual is not None:
        value = value + residual
    return value + groups.mean(-1)


def compile() -> None:
    """Compile the encoder's residual widths and four stage shortcuts."""
    for c in (160, 320, 640):
        compile_bias_residual(c)
        compile_bias_residual(c, spatial_shape=(3, 5))
    for ci, co, ft, fs in (
        (160, 160, 1, 2),
        (160, 320, 2, 2),
        (320, 640, 2, 2),
        (640, 640, 1, 1),
    ):
        for t in (1, 4):
            compile_stage((t, 8, 6, ci, co, ft, fs, fs == 1, fs == 1))


@torch.inference_mode()
def verify() -> None:
    """Check BF16 rounding, vector tails, spatial zeros, and causal shortcuts."""
    torch.manual_seed(43)
    for c in (160, 320, 640):
        bias = torch.randn(c, device="cuda", dtype=torch.bfloat16)
        # Offset storage exercises the live natural-alignment specialization.
        for offset in (0, 1):
            x = torch.randn(15 * c + offset, device="cuda", dtype=bias.dtype)[
                offset:
            ].view(1, 1, 3, 5, c)
            skip = -(x + bias)
            torch.testing.assert_close(
                bias_residual(x, skip, bias), torch.zeros_like(x), atol=0, rtol=0
            )
            for pad in (False, True):
                for rb in (None, bias):
                    kwargs = {"residual_bias": rb, "pad_spatial": pad}
                    torch.testing.assert_close(
                        bias_residual(x, skip, bias, **kwargs),
                        torch_bias_residual(x, skip, bias, **kwargs),
                        atol=0,
                        rtol=0,
                    )
    for ci, co, ft, fs in (
        (160, 160, 1, 2),
        (160, 320, 2, 2),
        (320, 640, 2, 2),
        (640, 640, 1, 1),
    ):
        for t in (1, 2, 4):
            skip = torch.randn(2, t, 8, 6, ci, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(
                2,
                (t + ft - 1) // ft,
                8 // fs,
                6 // fs,
                co,
                device="cuda",
                dtype=skip.dtype,
            )
            bias = torch.randn(co, device="cuda", dtype=x.dtype) if fs == 1 else None
            residual = torch.randn_like(x) if fs == 1 else None
            torch.testing.assert_close(
                stage_add(x, skip, ft, fs, bias, residual),
                torch_stage_add(x, skip, ft, fs, bias, residual),
                atol=0,
                rtol=0,
            )
            # All vector inputs accept naturally aligned offset views.
            shifted = []
            for value in (x, skip, bias, residual):
                if value is None:
                    shifted.append(None)
                else:
                    storage = torch.empty(
                        value.numel() + 1, device=value.device, dtype=value.dtype
                    )
                    view = storage[1:].view(value.shape)
                    view.copy_(value)
                    shifted.append(view)
            sx, ss, sb, sr = shifted
            torch.testing.assert_close(
                stage_add(sx, ss, ft, fs, sb, sr),
                torch_stage_add(x, skip, ft, fs, bias, residual),
                atol=0,
                rtol=0,
            )
    print(
        "Residual operations: PASS (Torch values, rounding, padding, shortcuts)",
        flush=True,
    )


@torch.inference_mode()
def _verify_stage_ordered(
    x: torch.Tensor,
    skip: torch.Tensor,
    ft: int,
    fs: int,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> None:
    """Check exact FP32 accumulation order, independently of Torch mean's tree.

    At full size, random inputs can land on BF16 ties where Torch mean's tree
    differs from the kernel's sequential sum. Keep the exact check using Torch
    arithmetic in that order; benchmark the normal torch_stage_add separately.
    """
    n, t, h, w, ci = skip.shape
    padded = torch.nn.functional.pad(skip, (0, 0, 0, 0, 0, 0, (-t) % ft, 0))
    groups = padded.view(n, (t + ft - 1) // ft, ft, h // fs, fs, w // fs, fs, ci)
    groups = groups.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(*x.shape, -1)
    total = torch.zeros_like(x, dtype=torch.float32)
    for index in range(groups.shape[-1]):
        total.add_(groups[..., index].float())
    pooled = (total / groups.shape[-1]).to(x.dtype)
    value = x if bias is None else x + bias
    if residual is not None:
        value = value + residual
    torch.testing.assert_close(
        stage_add(x, skip, ft, fs, bias, residual), value + pooled, atol=0, rtol=0
    )


@torch.inference_mode()
def benchmark_production() -> None:
    """Time 480x832 batch-32 residual boundaries and four stage shortcuts."""
    torch.manual_seed(43)
    print(
        "\nProduction benchmark: batch=32, BF16, NTHWC; Torch reference is eager."
        "\nMedian GPU ms; 5 alternating samples x 20 CUDA-graph replays."
        "\nCompilation and input generation excluded; speedup=reference/custom.",
        flush=True,
    )
    for c, frames, h, w in (
        (160, 4, 240, 416),
        (320, 4, 120, 208),
        (640, 2, 60, 104),
        (640, 1, 30, 52),
    ):
        for t in sorted({1, frames}):
            x = torch.randn(32, t, h, w, c, device="cuda", dtype=torch.bfloat16)
            skip = torch.randn_like(x)
            bias = torch.randn(c, device="cuda", dtype=x.dtype)
            print(f"\nResidual input NTHWC={tuple(x.shape)}", flush=True)
            results = []
            # Spatial padding is the separate reference for fused downsample convs.
            for pad in (False, True) if h >= 60 else (False,):

                reference = partial(
                    torch_bias_residual, x, skip, bias, pad_spatial=pad
                )
                custom = partial(bias_residual, x, skip, bias, pad_spatial=pad)

                torch.testing.assert_close(custom(), reference(), atol=0, rtol=0)
                label = "bias + residual" + (" + spatial pad" if pad else "")
                results.append(measure(label, reference, custom))
            print_benchmark_table(results)

    for index, (ci, co, ft, fs, frames, h, w) in enumerate(
        (
            (160, 160, 1, 2, 4, 240, 416),
            (160, 320, 2, 2, 4, 120, 208),
            (320, 640, 2, 2, 2, 60, 104),
            (640, 640, 1, 1, 1, 30, 52),
        )
    ):
        for t in sorted({1, frames}):
            skip = torch.randn(32, t, h, w, ci, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(
                32,
                (t + ft - 1) // ft,
                h // fs,
                w // fs,
                co,
                device="cuda",
                dtype=skip.dtype,
            )
            # Temporal downsampling defers bias only after the initial chunk.
            has_bias = ft == 1 or t > 1
            bias = torch.randn(co, device="cuda", dtype=x.dtype) if has_bias else None
            residual = torch.randn_like(x) if fs == 1 else None
            print(
                f"\nStage {index} | skip NTHWC={tuple(skip.shape)} | "
                f"output NTHWC={tuple(x.shape)} | factors T={ft}, S={fs}",
                flush=True,
            )

            reference = partial(torch_stage_add, x, skip, ft, fs, bias, residual)
            custom = partial(stage_add, x, skip, ft, fs, bias, residual)

            _verify_stage_ordered(x, skip, ft, fs, bias, residual)
            label = "shortcut mean + add"
            if has_bias:
                label += " + bias"
            if residual is not None:
                label += " + residual"
            print_benchmark_table([measure(label, reference, custom)])
    print("Production residual/shortcut correctness: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-compile", action="store_true")
    args = parser.parse_args()
    if args.only_compile:
        compile()
        print("Residual operations compile: PASS", flush=True)
    else:
        print("Correctness checks (small cases, untimed):", flush=True)
        verify()
        benchmark_production()
