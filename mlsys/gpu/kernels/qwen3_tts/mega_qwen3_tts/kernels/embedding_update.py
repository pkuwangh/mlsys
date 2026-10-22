"""Codec embedding lookup, BF16 accumulation and optional projected lookup.

Run directly for graph-replay correctness and interleaved compiled-Torch timings.
The projection table is precomputed; no projection GEMM is performed here.
Embedding and projected-embedding tiles run in separate CTAs of the same launch.
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
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary


@cute.jit
def gather_update_row(table, gathered, accumulated, row, token, start, width: cutlass.Constexpr):
    """Use 128-bit transfers and BF16 additions with round-to-nearest."""
    iket.range_push("embedding_load")
    values = (table.iterator.raw_ptr() + token * width + start).load(count=8, alignment=16)
    if cutlass.const_expr(accumulated is not None):
        pointer = accumulated.iterator.raw_ptr() + row * width + start
        previous = pointer.load(count=8, alignment=16)
    iket.range_pop()
    iket.range_push("embedding_store_add")
    (gathered.iterator.raw_ptr() + row * width + start).store(values, alignment=16)
    if cutlass.const_expr(accumulated is not None):
        pointer.store(previous + values, alignment=16)
    iket.range_pop()


@cute.kernel
def embedding_update_kernel(
    tokens,
    table,
    accumulated,
    gathered,
    projected_table,
    projected,
    width: cutlass.Constexpr,
    projected_width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    row, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    token = tokens[row]
    embedding_tiles = cute.ceil_div(width, threads * 8)
    # Each CTA owns one independent output tile; no handoff is needed.
    if tile < embedding_tiles:
        start = (tile * threads + tid) * 8
        if start < width:
            gather_update_row(table, gathered, accumulated, row, token, start, width)
    else:
        if cutlass.const_expr(projected_width > 0):
            start = ((tile - embedding_tiles) * threads + tid) * 8
            if start < projected_width:
                gather_update_row(projected_table, projected, None, row, token, start, projected_width)


@cute.jit
def embedding_update_host(
    tokens, table, accumulated, gathered, projected_table, projected, stream, threads: cutlass.Constexpr
):
    width = table.shape[1]
    projected_width = projected_table.shape[1] if projected_table is not None else 0
    embedding_update_kernel(
        tokens, table, accumulated, gathered, projected_table, projected, width, projected_width, threads
    ).launch(
        grid=(tokens.shape[0], cute.ceil_div(width, threads * 8) + cute.ceil_div(projected_width, threads * 8), 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cache
def compile_kernel(batch, vocab, width, projected_width, token_type, arch, threads, token_stride=1):
    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        embedding_update_host,
        make_fake_tensor(token_type, (batch,), stride=(token_stride,), assumed_align=token_type.width // 8),
        fake(cutlass.BFloat16, vocab, width),
        fake(cutlass.BFloat16, batch, width),
        fake(cutlass.BFloat16, batch, width),
        fake(cutlass.BFloat16, vocab, projected_width) if projected_width else None,
        fake(cutlass.BFloat16, batch, projected_width) if projected_width else None,
        make_fake_stream(),
        threads,
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


@torch.compiler.disable
def embedding_update(tokens, table, accumulated, projected_table=None, *, threads=128):
    """Return gathered rows and optional projected rows; update accumulated in place."""
    if tokens.ndim != 1 or table.ndim != 2 or accumulated.shape != (tokens.numel(), table.shape[1]):
        raise ValueError("Expected token IDs [B], table [V,D], and accumulator [B,D]")
    if tokens.dtype not in (torch.int32, torch.int64) or table.dtype != torch.bfloat16:
        raise ValueError("Expected INT32/INT64 IDs and BF16 embeddings")
    if table.shape[1] % 8 or table.shape[1] == 0 or tokens.numel() == 0:
        raise ValueError("Expected B > 0 and positive D divisible by eight")
    tensors = [table, accumulated]
    if projected_table is not None:
        if projected_table.ndim != 2 or projected_table.shape[0] != table.shape[0] or projected_table.shape[1] % 8:
            raise ValueError("Expected projected table [V,DP] with DP divisible by eight")
        tensors.append(projected_table)
    if (
        not tokens.is_cuda
        or tokens.stride(0) < 1
        or any(t.device != tokens.device or t.dtype != table.dtype or not t.is_contiguous() for t in tensors)
    ):
        raise ValueError("Expected contiguous tensors on the same CUDA device, with BF16 tables and accumulator")
    if threads not in (32, 64, 128):
        raise ValueError("Expected 32, 64 or 128 threads")
    batch, width = accumulated.shape
    projected_width = projected_table.shape[1] if projected_table is not None else 0
    gathered = torch.empty_like(accumulated)
    projected = table.new_empty(batch, projected_width) if projected_width else None
    compile_kernel(
        batch,
        table.shape[0],
        width,
        projected_width,
        cutlass.Int32 if tokens.dtype == torch.int32 else cutlass.Int64,
        device_arch(tokens.device),
        threads,
        tokens.stride(0),
    )(
        tokens,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return gathered, projected


def reference(tokens, table, accumulated, projected_table=None):
    gathered = F.embedding(tokens, table)
    accumulated.add_(gathered)
    return gathered, F.embedding(tokens, projected_table) if projected_table is not None else None


@torch.inference_mode()
def validate():
    for batch, width, projected_width in ((1, 1024, 0), (7, 2048, 1024), (7, 1032, 520), (32, 1024, 2048)):
        table = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16)
        projection = torch.randn(2048, projected_width, device="cuda", dtype=table.dtype) if projected_width else None
        tokens = torch.randint(2048, (batch,), device="cuda", dtype=torch.int32 if batch == 7 else torch.int64)
        if batch == 7:
            tokens = tokens[:, None].repeat(1, 16)[:, 3]
        for threads in (32, 64, 128):
            accumulated = table.new_zeros(batch, width)
            embedding_update(tokens, table, accumulated, projection, threads=threads)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = embedding_update(tokens, table, accumulated, projection, threads=threads)
            for _ in range(3):
                tokens.random_(0, 2048)
                accumulated.normal_()
                expected_sum = accumulated.clone()
                expected = reference(tokens, table, expected_sum, projection)
                graph.replay()
                torch.testing.assert_close(accumulated, expected_sum, atol=0, rtol=0)
                for actual, target in zip(outputs, expected):
                    if target is not None:
                        torch.testing.assert_close(actual, target, atol=0, rtol=0)
        print(f"PASS B={batch} D={width} DP={projected_width}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--tune", action="store_true", help="Compare CTA sizes")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    validate()
    results = []
    for batch in args.batch_size:
        for width, projected_width in ((1024, 0), (2048, 1024)):
            torch.compiler.reset()
            compiled = torch.compile(reference, fullgraph=True)
            tokens = torch.randint(2048, (batch,), device="cuda")
            table = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16)
            projection = (
                torch.randn(2048, projected_width, device="cuda", dtype=table.dtype) if projected_width else None
            )
            baseline_sum = table.new_zeros(batch, width)
            custom_sum = torch.zeros_like(baseline_sum)
            for threads in (32, 64, 128) if args.tune else (128,):
                times = measure(
                    {
                        "Compiled": partial(compiled, tokens, table, baseline_sum, projection),
                        "Fused": partial(embedding_update, tokens, table, custom_sum, projection, threads=threads),
                    },
                    calls_per_graph=16,
                    profile=args.profile,
                )
                results.append(((batch, width, projected_width, threads), times))
    print_kernel_summary(results, columns=["Batch", "D", "DP", "Threads"], workload="embedding update, BF16, V=2048")


if __name__ == "__main__":
    main()
