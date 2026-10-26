"""Greedy token selection and embedding lookup in one CUDA launch.

One CTA per row, small vocabularies up to 8192 entries, independent embedding
and optional projected-embedding widths. FP16/BF16/FP32 inputs; ties and NaNs
select the first index, matching Torch argmax. Inference only, Hopper/Blackwell.
Run this file for exact correctness checks and eager/compiled Torch timings.
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
from cutlass.experimental import primitives as p
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary


@cute.jit
def prefer_candidate(value, index, best, best_index):
    """Order NaNs first, then descending values, then ascending indices."""
    is_nan = value != value  # noqa: PLR0124 - IEEE NaN test in device code.
    best_is_number = best == best  # noqa: PLR0124
    return (is_nan & (best_is_number | (index < best_index))) | (
        best_is_number & ((value > best) | ((value == best) & (index < best_index)))
    )


@cute.jit
def warp_argmax(value, index):
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, 1 << shift)
        other_index = cute.arch.shuffle_sync_bfly(index, 1 << shift)
        if prefer_candidate(other, other_index, value, index):
            value, index = other, other_index
    return value, index


@cute.jit
def gather_row(table, output, row, token, width: cutlass.Constexpr, accumulated=None):
    tid, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(width % 8 == 0):
        for block in range(cute.ceil_div(width, 128 * 8)):
            col = block * 128 * 8 + tid * 8
            if col < width:
                values = (table.iterator.raw_ptr() + token * width + col).load(count=8, alignment=16)
                if cutlass.const_expr(accumulated is not None):
                    pointer = accumulated.iterator.raw_ptr() + row * width + col
                    previous = pointer.load(count=8, alignment=16)
                (output.iterator.raw_ptr() + row * width + col).store(values, alignment=16)
                if cutlass.const_expr(accumulated is not None):
                    pointer.store(previous + values, alignment=16)
    else:
        for block in range(cute.ceil_div(width, 128)):
            col = block * 128 + tid
            if col < width:
                output[row, col] = table[token, col]
                if cutlass.const_expr(accumulated is not None):
                    accumulated[row, col] = table.element_type(
                        cutlass.Float32(accumulated[row, col]) + cutlass.Float32(table[token, col])
                    )


@cute.kernel
def argmax_embedding_kernel(
    logits: cute.Tensor,
    table: cute.Tensor,
    projected_table: cute.Tensor | None,
    codes: cute.Tensor,
    embeddings: cute.Tensor,
    projected: cute.Tensor | None,
    accumulated: cute.Tensor | None,
    V: cutlass.Constexpr,
    D: cutlass.Constexpr,
    DP: cutlass.Constexpr,
):
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    partial_values = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    partial_indices = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    winner = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    best, index = cutlass.Float32(float("-inf")), cutlass.Int32(0x7FFFFFFF)
    iket.range_push("logits_load_local_max")
    if cutlass.const_expr(V % 1024 == 0):
        for block in cutlass.range_constexpr(V // 1024):
            start = block * 1024 + tid * 8
            values = (logits.iterator.raw_ptr() + row * V + start).load(count=8, alignment=16)
            for item in cutlass.range_constexpr(8):
                col = start + item
                value = cutlass.Float32(values[item])
                if prefer_candidate(value, col, best, index):
                    best, index = value, col
    else:
        for block in cutlass.range_constexpr(cute.ceil_div(V, 128)):
            col = block * 128 + tid
            if col < V:
                value = cutlass.Float32(logits[row, col])
                if prefer_candidate(value, col, best, index):
                    best, index = value, col
    iket.range_pop()
    iket.range_push("argmax_reduce")
    best, index = warp_argmax(best, index)
    if tid % 32 == 0:
        partial_values[tid // 32], partial_indices[tid // 32] = best, index
    p.barrier_cta_sync()
    if tid < 32:
        best, index = cutlass.Float32(float("-inf")), cutlass.Int32(0x7FFFFFFF)
        if tid < 4:
            best, index = partial_values[tid], partial_indices[tid]
        best, index = warp_argmax(best, index)
        if tid == 0:
            winner[0] = index
            codes[row] = cutlass.Int64(index)
    p.barrier_cta_sync()
    index = winner[0]
    iket.range_pop()
    iket.range_push("embedding_gather_add")
    gather_row(table, embeddings, row, index, D, accumulated)
    if cutlass.const_expr(DP > 0):
        gather_row(projected_table, projected, row, index, DP)
    iket.range_pop()


@cute.jit
def argmax_embedding_host(logits, table, projected_table, codes, embeddings, projected, accumulated, stream):
    argmax_embedding_kernel(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        accumulated,
        logits.shape[1],
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
    ).launch(grid=(logits.shape[0], 1, 1), block=(128, 1, 1), stream=stream)


@cache
def compile_kernel(vocab, width, projected_width, logits_dtype, embedding_dtype, arch, accumulate, code_stride):
    batch = cutlass.sym_int64()

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        argmax_embedding_host,
        fake(logits_dtype, batch, vocab),
        fake(embedding_dtype, vocab, width),
        fake(embedding_dtype, vocab, projected_width) if projected_width else None,
        make_fake_tensor(cutlass.Int64, (batch,), stride=(code_stride,), assumed_align=8),
        fake(embedding_dtype, batch, width),
        fake(embedding_dtype, batch, projected_width) if projected_width else None,
        fake(embedding_dtype, batch, width) if accumulate else None,
        make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


@torch.compiler.disable
def argmax_embedding(logits, table, projected_table=None, *, accumulated=None, out_codes=None):
    """Select and gather, optionally updating the embedding sum and strided output-code slots."""
    types = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    if logits.ndim != 2 or table.ndim != 2 or logits.shape[1] != table.shape[0]:
        raise ValueError("Expected logits [B,V] and embedding table [V,D]")
    batch, vocab = logits.shape
    if batch < 1 or not 1 <= vocab <= 8192 or table.shape[1] < 1:
        raise ValueError("Expected B,D >= 1 and 1 <= V <= 8192")
    if logits.dtype not in types or table.dtype not in types:
        raise ValueError("Expected FP16, BF16, or FP32 tensors")
    tensors = [logits, table]
    if projected_table is not None:
        if (
            projected_table.ndim != 2
            or projected_table.shape[0] != vocab
            or projected_table.shape[1] < 1
            or projected_table.dtype != table.dtype
        ):
            raise ValueError("Expected projected table [V,DP] with the embedding dtype")
        tensors.append(projected_table)
    if accumulated is not None:
        if accumulated.shape != (batch, table.shape[1]) or accumulated.dtype != table.dtype:
            raise ValueError("Expected accumulator [B,D] with the embedding dtype")
        tensors.append(accumulated)
    if any(not t.is_cuda or t.device != logits.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors on the same CUDA device")
    arch = device_arch(logits.device)
    if out_codes is not None and (
        out_codes.shape != (batch,)
        or out_codes.dtype != torch.int64
        or out_codes.device != logits.device
        or out_codes.stride(0) < 1
    ):
        raise ValueError("Expected INT64 output-code slots [B] on the logits device")
    codes = out_codes if out_codes is not None else torch.empty(batch, device=logits.device, dtype=torch.int64)
    embeddings = table.new_empty(batch, table.shape[1])
    projected = projected_table.new_empty(batch, projected_table.shape[1]) if projected_table is not None else None
    compile_kernel(
        vocab,
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
        types[logits.dtype],
        types[table.dtype],
        arch,
        accumulated is not None,
        codes.stride(0),
    )(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        accumulated,
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    return codes, embeddings, projected


def reference(logits, table, projected_table=None, *, accumulated=None, out_codes=None):
    """Torch greedy selection and one or two embedding lookups."""
    codes = logits.argmax(-1)
    gathered = F.embedding(codes, table)
    if accumulated is not None:
        accumulated.add_(gathered)
    if out_codes is not None:
        out_codes.copy_(codes)
        codes = out_codes
    return (
        codes,
        gathered,
        F.embedding(codes, projected_table) if projected_table is not None else None,
    )


@torch.inference_mode()
def validate():
    """Check ties, NaNs, irregular widths, optional projection, and graph replay."""
    cases = (
        (torch.bfloat16, torch.bfloat16, 16, 2048, 1024, 0),
        (torch.float16, torch.float16, 3, 2053, 513, 257),
        (torch.float32, torch.float32, 7, 3072, 256, 1024),
        (torch.float32, torch.bfloat16, 1, 8192, 64, 128),
    )
    for logits_dtype, embedding_dtype, batch, vocab, width, projected_width in cases:
        logits = torch.randn(batch, vocab, device="cuda", dtype=logits_dtype)
        table = torch.randn(vocab, width, device="cuda", dtype=embedding_dtype)
        projection = (
            torch.randn(vocab, projected_width, device="cuda", dtype=embedding_dtype) if projected_width else None
        )
        argmax_embedding(logits, table, projection)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = argmax_embedding(logits, table, projection)
        for mode in ("random", "ties", "nan", "negative_inf"):
            if mode == "ties":
                logits.fill_(0)
                logits[:, 7] = logits[:, vocab - 1] = 10
            elif mode == "nan":
                logits[:, 3] = logits[:, vocab - 2] = float("nan")
            elif mode == "negative_inf":
                logits.fill_(float("-inf"))
            table.normal_()
            if projection is not None:
                projection.normal_()
            graph.replay()
            for actual, expected in zip(output, reference(logits, table, projection)):
                if expected is not None:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        print(f"PASS {logits_dtype}/{embedding_dtype} B={batch} V={vocab} D={width} DP={projected_width}", flush=True)
        accumulated = table.new_zeros(batch, width)
        code_buffer = torch.full((batch, 16), -1, device=logits.device, dtype=torch.int64)
        argmax_embedding(logits, table, projection, accumulated=accumulated, out_codes=code_buffer[:, 3])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = argmax_embedding(logits, table, projection, accumulated=accumulated, out_codes=code_buffer[:, 3])
        for _ in range(3):
            logits.normal_()
            accumulated.normal_()
            expected_sum = accumulated.clone()
            expected = reference(logits, table, projection, accumulated=expected_sum)
            graph.replay()
            torch.testing.assert_close(accumulated, expected_sum, atol=0, rtol=0)
            for actual, target in zip(output, expected):
                if target is not None:
                    torch.testing.assert_close(actual, target, atol=0, rtol=0)
            assert (code_buffer[:, :3] == -1).all() and (code_buffer[:, 4:] == -1).all()
        print("PASS accumulation and strided code writes", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16, 32], help="Benchmark batches")
    args = parser.parse_args()
    torch.manual_seed(42)
    validate()
    results = []
    for batch in args.batch_size:
        logits = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        for embedding_width, width in ((1024, 0), (2048, 1024)):
            torch.compiler.reset()
            compiled = torch.compile(reference, fullgraph=True)
            table = torch.randn(2048, embedding_width, device="cuda", dtype=torch.bfloat16)
            projection = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16) if width else None
            sums = [table.new_zeros(batch, embedding_width) for _ in range(3)]
            codes = [torch.empty(batch, 16, device="cuda", dtype=torch.int64)[:, 3] for _ in range(3)]
            times = measure(
                {
                    "Torch": partial(reference, logits, table, projection, accumulated=sums[0], out_codes=codes[0]),
                    "Compiled": partial(compiled, logits, table, projection, accumulated=sums[1], out_codes=codes[1]),
                    "Fused": partial(
                        argmax_embedding, logits, table, projection, accumulated=sums[2], out_codes=codes[2]
                    ),
                },
                calls_per_graph=16,
            )
            results.append(((batch, embedding_width, width), times))
    print_kernel_summary(results, columns=["Batch", "D", "DP"], workload="argmax + embedding update, BF16, V=2048")


if __name__ == "__main__":
    main()
