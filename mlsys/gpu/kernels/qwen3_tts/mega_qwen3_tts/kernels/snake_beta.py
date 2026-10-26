"""Fused BF16 SnakeBeta with cached channel coefficients and vectorized transfers.

Preserve eager BF16 rounding after each arithmetic operation, including sine.
Both contiguous [B,C,T] and channels-last [B,T,C] storage are supported.
Large tensors use an exact BF16 sine lookup table shared across modules.
"""

import argparse
import sys
from functools import cache, partial
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary


@cache
def sine_table(device):
    """Share one immutable 128 KiB table across inference modules on a device."""
    return torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).sin()


@cute.kernel
def snake_beta_kernel(
    x,
    alpha,
    inverse_beta,
    sine_table,
    output,
    elements: cutlass.Int64,
    channels: cutlass.Constexpr,
    length: cutlass.Int32,
    channels_last: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    offset = (cutlass.Int64(block) * threads + tid) * vector
    if offset < elements:
        iket.range_push("input_coefficients")
        values = (x.iterator.raw_ptr() + offset).load(count=vector, alignment=vector * 2)
        if cutlass.const_expr(channels_last):
            channel = offset % channels
            a = (alpha.iterator.raw_ptr() + channel).load(count=vector, alignment=vector * 2)
            r = (inverse_beta.iterator.raw_ptr() + channel).load(count=vector, alignment=vector * 2)
        else:
            channel = (offset // length) % channels
            a = alpha[channel]
            r = inverse_beta[channel]
        iket.range_pop()
        iket.range_push("scale_sine")
        scaled = values * a
        if cutlass.const_expr(sine_table is not None):
            offsets = scaled.bitcast(cutlass.Uint16).to(cutlass.Int32)
            sine = sine_table.iterator.raw_ptr().masked_gather(offsets, cutlass.full((vector,), True, cutlass.Boolean))
        else:
            sine = cute.math.sin(scaled.to(cutlass.Float32), fastmath=False).to(cutlass.BFloat16)
        iket.range_pop()
        iket.range_push("square_residual_store")
        square = sine * sine
        product = square * r
        result = values + product
        (output.iterator.raw_ptr() + offset).store(result, alignment=vector * 2)
        iket.range_pop()


@cute.jit
def snake_beta_host(
    x,
    alpha,
    inverse_beta,
    sine_table,
    output,
    stream,
    channels: cutlass.Constexpr,
    length: cutlass.Int32,
    elements: cutlass.Int64,
    channels_last: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    # CUTLASS 4.8 shape extraction narrows to Int32; keep numel a separate Int64 argument.
    blocks = cutlass.Int32((elements + threads * vector - 1) // (threads * vector))
    snake_beta_kernel(
        x, alpha, inverse_beta, sine_table, output, elements, channels, length, channels_last, threads, vector
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)


@cache
def compile_kernel(channels, channels_last, threads, vector, arch, lookup=False):
    elements = cutlass.sym_int64(divisibility=vector)

    def fake(n):
        return make_fake_compact_tensor(cutlass.BFloat16, (n,), stride_order=(0,))

    return cute.compile(
        snake_beta_host,
        fake(elements),
        fake(channels),
        fake(channels),
        fake(65536) if lookup else None,
        fake(elements),
        make_fake_stream(),
        channels,
        cutlass.Int32(0),
        cutlass.Int64(0),
        channels_last,
        threads,
        vector,
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


class SnakeBeta(torch.nn.Module):
    """Inference-only activation with immutable, precomputed BF16 channel coefficients.

    The default launch policy uses four values/thread for small inputs, eight otherwise,
    and the 128 KiB sine table for inputs with at least four million elements.
    Construct on the final device before CUDA graph capture.
    """

    def __init__(self, alpha, beta, epsilon=1e-9, *, threads=128, vector=None, lookup=None):
        super().__init__()
        if alpha.ndim != 1 or beta.shape != alpha.shape or alpha.dtype != torch.bfloat16 or beta.dtype != alpha.dtype:
            raise ValueError("Expected BF16 alpha and beta vectors [C]")
        if not alpha.is_cuda or alpha.device != beta.device:
            raise ValueError("Expected parameters on the same CUDA device")
        if threads not in (64, 128, 256) or vector not in (None, 1, 2, 4, 8):
            raise ValueError("Expected 64/128/256 threads and vector width 1/2/4/8")
        self.register_buffer("alpha", alpha.detach().exp().contiguous())
        self.register_buffer("inverse_beta", (1.0 / (beta.detach().exp() + epsilon)).contiguous())
        self.register_buffer("sine_table", sine_table(alpha.device) if lookup is not False else None)
        self.threads, self.vector, self.lookup = threads, vector, lookup
        self.arch = device_arch(alpha.device)

    @torch.compiler.disable
    def forward(self, x):
        if (
            x.ndim != 3
            or x.shape[1] != self.alpha.numel()
            or x.dtype != torch.bfloat16
            or x.device != self.alpha.device
        ):
            raise ValueError("Expected BF16 input [B,C,T] matching the prepared parameters")
        channels_last = not x.is_contiguous()
        if channels_last and not x.transpose(1, 2).is_contiguous():
            raise ValueError("Expected contiguous BCT or BTC storage")
        physical = x.transpose(1, 2) if channels_last else x
        output = torch.empty_like(physical)
        contiguous_width = x.shape[1] if channels_last else x.shape[2]
        vector = self.vector or (4 if x.numel() <= 65536 else 8)
        while contiguous_width % vector:
            vector //= 2
        table = self.sine_table if (self.lookup if self.lookup is not None else x.numel() >= 4 * 1024**2) else None
        if x.numel():
            compile_kernel(x.shape[1], channels_last, self.threads, vector, self.arch, table is not None)(
                physical.view(-1),
                self.alpha,
                self.inverse_beta,
                table,
                output.view(-1),
                cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
                x.shape[2],
                x.numel(),
            )
        return output.transpose(1, 2) if channels_last else output


def reference(x, alpha, beta, epsilon=1e-9):
    a = alpha[None, :, None].exp()
    r = 1.0 / (beta[None, :, None].exp() + epsilon)
    return x + r * torch.sin(x * a).square()


@torch.inference_mode()
def validate():
    for batch, channels, length in ((1, 8, 33), (3, 96, 256), (1, 8, 65536)):
        alpha = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.25
        beta = torch.randn_like(alpha) * 0.25
        for channels_last in (False, True):
            x = torch.randn(batch, channels, length, device="cuda", dtype=torch.bfloat16)
            if length == 65536:
                bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                x.copy_(bits)
            if channels_last:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            for lookup in (False, True):
                if length == 65536:
                    x.copy_(bits)
                operation = SnakeBeta(alpha, beta, lookup=lookup)
                operation(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = operation(x)
                for replay in range(2):
                    if replay:
                        x.normal_()
                    graph.replay()
                    torch.testing.assert_close(actual, reference(x, alpha, beta), atol=0, rtol=0, equal_nan=True)
            print(f"PASS B={batch} C={channels} T={length} channels_last={channels_last}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    validate()
    results = []
    for shape in ((1, 1024, 64), (4, 512, 512), (4, 128, 4096), (4, 64, 16384), (4, 64, 131072)):
        for channels_last in (False, True):
            torch.compiler.reset()
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            if channels_last:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            alpha = torch.randn(shape[1], device="cuda", dtype=x.dtype) * 0.25
            beta = torch.randn_like(alpha) * 0.25
            compiled = torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True})
            torch.testing.assert_close(compiled(x, alpha, beta), reference(x, alpha, beta), atol=0, rtol=0)
            for threads, vector, lookup in (
                ((128, 4, False), (128, 8, False), (128, 4, True), (128, 8, True), (256, 8, True))
                if args.tune
                else ((128, None, None),)
            ):
                operation = SnakeBeta(alpha, beta, threads=threads, vector=vector, lookup=lookup)
                times = measure(
                    {"Compiled": partial(compiled, x, alpha, beta), "Fused": partial(operation, x)},
                    calls_per_graph=16,
                    profile=args.profile,
                )
                effective_vector = vector or (4 if x.numel() <= 65536 else 8)
                use_lookup = lookup if lookup is not None else x.numel() >= 4 * 1024**2
                results.append(
                    (
                        (
                            shape,
                            "BTC" if channels_last else "BCT",
                            threads,
                            effective_vector,
                            "table" if use_lookup else "sin",
                        ),
                        times,
                    )
                )
    print_kernel_summary(results, columns=["B,C,T", "Layout", "Threads", "Vector", "Sine"], workload="SnakeBeta, BF16")


if __name__ == "__main__":
    main()
