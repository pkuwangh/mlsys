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
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as p
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure


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
def gather_row(table, output, row, token, width: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(width % 8 == 0):
        for block in range(cute.ceil_div(width, 128 * 8)):
            col = block * 128 * 8 + tid * 8
            if col < width:
                values = (table.iterator.raw_ptr() + token * width + col).load(count=8, alignment=16)
                (output.iterator.raw_ptr() + row * width + col).store(values, alignment=16)
    else:
        for block in range(cute.ceil_div(width, 128)):
            col = block * 128 + tid
            if col < width:
                output[row, col] = table[token, col]


@cute.kernel
def argmax_embedding_kernel(
    logits: cute.Tensor,
    table: cute.Tensor,
    projected_table: cute.Tensor | None,
    codes: cute.Tensor,
    embeddings: cute.Tensor,
    projected: cute.Tensor | None,
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
    for block in cutlass.range_constexpr(cute.ceil_div(V, 128)):
        col = block * 128 + tid
        if col < V:
            value = cutlass.Float32(logits[row, col])
            if prefer_candidate(value, col, best, index):
                best, index = value, col
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
    gather_row(table, embeddings, row, index, D)
    if cutlass.const_expr(DP > 0):
        gather_row(projected_table, projected, row, index, DP)


@cute.jit
def argmax_embedding_host(logits, table, projected_table, codes, embeddings, projected, stream):
    argmax_embedding_kernel(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        logits.shape[1],
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
    ).launch(grid=(logits.shape[0], 1, 1), block=(128, 1, 1), stream=stream)


@cache
def compile_kernel(batch, vocab, width, projected_width, logits_dtype, embedding_dtype, arch):
    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        argmax_embedding_host,
        fake(logits_dtype, batch, vocab),
        fake(embedding_dtype, vocab, width),
        fake(embedding_dtype, vocab, projected_width) if projected_width else None,
        fake(cutlass.Int64, batch),
        fake(embedding_dtype, batch, width),
        fake(embedding_dtype, batch, projected_width) if projected_width else None,
        make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


@torch.compiler.disable
def argmax_embedding(logits, table, projected_table=None):
    """Return token IDs [B], embeddings [B,D], and optionally projected rows [B,DP]."""
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
    if any(not t.is_cuda or t.device != logits.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors on the same CUDA device")
    arch = device_arch(logits.device)
    codes = torch.empty(batch, device=logits.device, dtype=torch.int64)
    embeddings = table.new_empty(batch, table.shape[1])
    projected = projected_table.new_empty(batch, projected_table.shape[1]) if projected_table is not None else None
    compile_kernel(
        batch,
        vocab,
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
        types[logits.dtype],
        types[table.dtype],
        arch,
    )(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return codes, embeddings, projected


def reference(logits, table, projected_table=None):
    """Torch greedy selection and one or two embedding lookups."""
    codes = logits.argmax(-1)
    return (
        codes,
        F.embedding(codes, table),
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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16, 32], help="Benchmark batches")
    args = parser.parse_args()
    torch.manual_seed(42)
    validate()
    compiled = torch.compile(reference, fullgraph=True)
    print("Batch | Projected width | Torch (us) | Compiled (us) | Fused (us) | Compiled/Fused")
    for batch in args.batch_size:
        logits = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        table = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
        for width in (0, 512):
            projection = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16) if width else None
            times = measure(
                {
                    "Torch": partial(reference, logits, table, projection),
                    "Compiled": partial(compiled, logits, table, projection),
                    "Fused": partial(argmax_embedding, logits, table, projection),
                }
            )
            print(
                f"{batch:5} | {width:15} | {times['Torch']:10.3f} | {times['Compiled']:13.3f} | "
                f"{times['Fused']:10.3f} | {times['Compiled'] / times['Fused']:.3f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
