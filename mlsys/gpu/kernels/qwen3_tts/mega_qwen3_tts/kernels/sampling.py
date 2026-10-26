"""Seeded BF16 top-k/top-p sampling, optionally fused with codec embedding updates.

Warp-local sorting retains only each warp's top-k candidates for the final merge.
Helper warps generate Gumbel noise while warp 0 merges candidates and computes softmax.
The 2048-entry vocabulary uses ordered BF16 keys with token-index tie breaking.
Seeds and positions are GPU inputs, so graph replay does not repeat a fixed draw.
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary
from mega_qwen3_tts.kernels.embedding_update import gather_update_row


@cute.jit
def select(condition, first, second):
    return first if condition else second


@cute.jit
def warp_sort(keys, items: cutlass.Constexpr):
    """Descending bitonic sort of lane-interleaved register keys."""
    lane = cute.arch.lane_idx()
    for stage in cutlass.range_constexpr((32 * items).bit_length() - 1):
        for step in cutlass.range_constexpr(stage + 1):
            distance = 1 << (stage - step)
            if cutlass.const_expr(distance < 32):
                for item in cutlass.range_constexpr(items):
                    value = keys[item]
                    other = cute.arch.shuffle_sync_bfly(value, distance)
                    maximum = ((lane + item * 32) & (2 << stage) == 0) == (lane & distance == 0)
                    keys[item] = select((value > other) == maximum, value, other)
            else:
                for item in cutlass.range_constexpr(items):
                    partner = item ^ (distance // 32)
                    if cutlass.const_expr(item < partner):
                        first, second = keys[item], keys[partner]
                        descending = item * 32 & (2 << stage) == 0
                        keys[item] = select((first > second) == descending, first, second)
                        keys[partner] = select((first > second) == descending, second, first)
    return keys


@cute.jit
def warp_topk(keys, items: cutlass.Constexpr, keep: cutlass.Constexpr, presorted: cutlass.Constexpr = False):
    """Merge sorted runs, discarding the bottom half after each merge."""
    lane = cute.arch.lane_idx()
    if cutlass.const_expr(not presorted):
        for group in cutlass.range_constexpr(items // keep):
            run = cutlass.Array(cutlass.Uint32, keep)
            for item in cutlass.range_constexpr(keep):
                run[item] = keys[group * keep + item]
            run = warp_sort(run, keep)
            for item in cutlass.range_constexpr(keep):
                keys[group * keep + item] = run[item]
    for level in cutlass.range_constexpr((items // keep).bit_length() - 1):
        for group in cutlass.range_constexpr(items // (keep * (2 << level))):
            run = cutlass.Array(cutlass.Uint32, keep)
            for item in cutlass.range_constexpr(keep):
                first = keys[group * 2 * keep + item]
                second = cute.arch.shuffle_sync_bfly(keys[(group * 2 + 2) * keep - item - 1], 31)
                run[item] = select(first > second, first, second)
            # Pairwise maxima form a bitonic run; only its merge remains.
            for step in cutlass.range_constexpr((32 * keep).bit_length() - 1):
                distance = (32 * keep) >> (step + 1)
                if cutlass.const_expr(distance < 32):
                    for item in cutlass.range_constexpr(keep):
                        value = run[item]
                        other = cute.arch.shuffle_sync_bfly(value, distance)
                        run[item] = select((value > other) == (lane & distance == 0), value, other)
                else:
                    for item in cutlass.range_constexpr(keep):
                        partner = item ^ (distance // 32)
                        if cutlass.const_expr(item < partner):
                            first, second = run[item], run[partner]
                            run[item] = select(first > second, first, second)
                            run[partner] = select(first > second, second, first)
            for item in cutlass.range_constexpr(keep):
                keys[group * keep + item] = run[item]
    return keys


@cute.jit
def mix_hash(h, word):
    word *= cutlass.Uint32(0xCC9E2D51)
    word = (word << 15) | (word >> 17)
    word *= cutlass.Uint32(0x1B873593)
    h ^= word
    h = (h << 13) | (h >> 19)
    return h * cutlass.Uint32(5) + cutlass.Uint32(0xE6546B64)


@cute.jit
def gumbel(seed, position, rank):
    """Murmur3 counter-based draw, including the zero/one endpoint behavior."""
    h = mix_hash(cutlass.Uint32(0), cutlass.Uint32(seed))
    h = mix_hash(h, cutlass.Uint32(cutlass.Uint64(seed) >> 32))
    h = mix_hash(h, cutlass.Uint32(position))
    h = mix_hash(h, cutlass.Uint32(rank))
    h ^= cutlass.Uint32(16)
    h ^= h >> 16
    h *= cutlass.Uint32(0x85EBCA6B)
    h ^= h >> 13
    h *= cutlass.Uint32(0xC2B2AE35)
    h ^= h >> 16
    u = cutlass.Float64(h) / cutlass.Float64(4294967295.0)
    log_u = cute.math.log(u)
    log_u = cute.math.max(log_u, cutlass.Float64(-1.7976931348623157e308))
    log_u = cute.math.min(log_u, cutlass.Float64(-2.3283064365386963e-10))
    return -cute.math.log(-log_u)


@cute.jit
def warp_sum(value):
    for step in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << step)
    return value


@cute.jit
def warp_winner(score, rank):
    for step in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(score, 1 << step)
        other_rank = cute.arch.shuffle_sync_bfly(rank, 1 << step)
        take = (other > score) | ((other == score) & (other_rank < rank))
        score, rank = select(take, other, score), select(take, other_rank, rank)
    return rank


@cute.jit
def key_score(key):
    ordered = key >> 16
    bits = ordered ^ select((ordered & 0x8000) != 0, cutlass.Uint32(0x8000), cutlass.Uint32(0xFFFF))
    return cutlass.Float32(cutlass.Uint16(bits).bitcast(cutlass.BFloat16))


@cute.kernel
def sample_kernel(
    logits,
    temperatures,
    top_ks,
    top_ps,
    seeds,
    positions,
    codes,
    table,
    accumulated,
    gathered,
    projected_table,
    projected,
    top_k: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    warps: cutlass.Constexpr,
):
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    candidates = cutlass.Array(cutlass.Uint32, warps * block_k, space=cutlass.AddressSpace.smem, alignment=16)
    winner = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    noise = cutlass.Array(cutlass.Float64, block_k, space=cutlass.AddressSpace.smem, alignment=16)
    keys = cutlass.Array(cutlass.Uint32, 2048 // (warps * 32))
    start = tid * (2048 // (warps * 32))
    iket.range_push("logits_load_keys")
    values = (logits.iterator.raw_ptr() + row * 2048 + start).load(
        count=2048 // (warps * 32), alignment=min(16, 4096 // (warps * 32))
    )
    for item in cutlass.range_constexpr(2048 // (warps * 32)):
        token = start + item
        bits = cutlass.Uint32(values[item].bitcast(cutlass.Uint16))
        bits = select((bits & 0x7FFF) == 0, cutlass.Uint32(0), bits)
        ordered = bits ^ select((bits & 0x8000) != 0, cutlass.Uint32(0xFFFF), cutlass.Uint32(0x8000))
        keys[item] = (ordered << 16) | (cutlass.Uint32(0xFFFF) - cutlass.Uint32(token))
    iket.range_pop()
    iket.range_push("warp_topk")
    keys = warp_topk(keys, 2048 // (warps * 32), block_k // 32)
    iket.range_pop()
    iket.range_push("candidate_handoff")
    for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
        if item * 32 + lane < block_k:
            candidates[warp * block_k + item * 32 + lane] = keys[item]
    p.barrier_cta_sync()
    iket.range_pop()
    merged = cutlass.Array(cutlass.Uint32, cute.ceil_div(warps * block_k, 32))
    probabilities = cutlass.Array(cutlass.Float32, cute.ceil_div(block_k, 32))
    count, total = cutlass.Int32(0), cutlass.Float32(0)
    # Noise depends on candidate rank, not its token ID or probability.
    if warp > 0:
        iket.range_push("gumbel_prepare")
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, (warps - 1) * 32)):
            rank = item * (warps - 1) * 32 + (warp - 1) * 32 + lane
            if rank < top_k:
                noise[rank] = gumbel(seeds[row], positions[row], rank)
        iket.range_pop()
    if warp == 0:
        iket.range_push("merge_topk")
        for item in cutlass.range_constexpr(cute.ceil_div(warps * block_k, 32)):
            merged[item] = cutlass.Uint32(0)
            if item * 32 + lane < warps * block_k:
                merged[item] = candidates[item * 32 + lane]
        merged = warp_topk(merged, warps * block_k // 32, block_k // 32, True)
        iket.range_pop()
        iket.range_push("softmax")
        count = cute.math.min(cutlass.Int32(top_ks[row]), cutlass.Int32(top_k))
        temperature = cute.math.max(temperatures[row], cutlass.Float32(1e-5))
        maximum = cute.arch.shuffle_sync(key_score(merged[0]) / temperature, 0)
        total = cutlass.Float32(0)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            probability = cutlass.Float32(0)
            if item * 32 + lane < count:
                probability = cute.math.exp(key_score(merged[item]) / temperature - maximum)
            probabilities[item] = probability
            total += probability
        total = warp_sum(total)
        iket.range_pop()
    iket.range_push("gumbel_ready")
    p.barrier_cta_sync()
    iket.range_pop()
    if warp == 0:
        iket.range_push("top_p_gumbel")
        before = cutlass.Float32(0)
        best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            rank = item * 32 + lane
            probability = probabilities[item] / total
            cumulative = probability
            for step in cutlass.range_constexpr(5):
                earlier = cute.arch.shuffle_sync_up(cumulative, 1 << step)
                if lane >= (1 << step):
                    cumulative += earlier
            prefix = before + cumulative - probability
            before += cute.arch.shuffle_sync(cumulative, 31)
            cutoff = top_ps[row]
            keep = (rank < count) & ((cutoff <= 0) | (cutoff >= 1) | (prefix < cutoff) | (rank == 0))
            score = cutlass.Float64(float("-inf"))
            if keep:
                score = cutlass.Float64(cute.math.log(probability)) + noise[rank]
            if (score > best) | ((score == best) & (rank < best_rank)):
                best, best_rank = score, rank
        iket.range_pop()
        iket.range_push("sample_select")
        best_rank = warp_winner(best, best_rank)
        token = cutlass.Int32(0)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            candidate = cute.arch.shuffle_sync(merged[item], best_rank % 32)
            if best_rank // 32 == item:
                token = cutlass.Int32(0xFFFF - (candidate & 0xFFFF))
        if lane == 0:
            codes[row] = cutlass.Int64(token)
            winner[0] = token
        iket.range_pop()
    if cutlass.const_expr(table is not None):
        iket.range_push("sample_handoff")
        p.barrier_cta_sync()
        token = winner[0]
        iket.range_pop()
        for tile in range(cute.ceil_div(table.shape[1], warps * 32 * 8)):
            start = (tile * warps * 32 + tid) * 8
            if start < table.shape[1]:
                gather_update_row(table, gathered, accumulated, row, token, start, table.shape[1])
        if cutlass.const_expr(projected_table is not None):
            for tile in range(cute.ceil_div(projected_table.shape[1], warps * 32 * 8)):
                start = (tile * warps * 32 + tid) * 8
                if start < projected_table.shape[1]:
                    gather_update_row(projected_table, projected, None, row, token, start, projected_table.shape[1])


@cute.jit
def sample_host(
    logits,
    temperatures,
    top_ks,
    top_ps,
    seeds,
    positions,
    codes,
    table,
    accumulated,
    gathered,
    projected_table,
    projected,
    stream,
    top_k: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    warps: cutlass.Constexpr,
):
    sample_kernel(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        codes,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        top_k,
        block_k,
        warps,
    ).launch(grid=(logits.shape[0], 1, 1), block=(warps * 32, 1, 1), stream=stream)


@cache
def compile_kernel(top_k, width, projected_width, accumulate, code_stride, warps, arch):
    batch = cutlass.sym_int64()

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        sample_host,
        fake(cutlass.BFloat16, batch, 2048),
        fake(cutlass.Float32, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Float32, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        make_fake_tensor(cutlass.Int64, (batch,), stride=(code_stride,), assumed_align=8),
        fake(cutlass.BFloat16, 2048, width) if width else None,
        fake(cutlass.BFloat16, batch, width) if accumulate else None,
        fake(cutlass.BFloat16, batch, width) if width else None,
        fake(cutlass.BFloat16, 2048, projected_width) if projected_width else None,
        fake(cutlass.BFloat16, batch, projected_width) if projected_width else None,
        make_fake_stream(),
        top_k,
        max(32, 1 << (top_k - 1).bit_length()),
        warps,
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


@cute.kernel
def sorted_sample_kernel(
    logprobs, indices, seeds, positions, output, count: cutlass.Constexpr, warps: cutlass.Constexpr
):
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
    best_token = cutlass.Int64(0)
    iket.range_push("logprobs_gumbel")
    seed, position = seeds[row], positions[row]
    for item in cutlass.range_constexpr(cute.ceil_div(count, warps * 32)):
        rank = item * warps * 32 + tid
        score = cutlass.Float64(float("-inf"))
        token = cutlass.Int64(0)
        if rank < count:
            if cutlass.const_expr(warps == 1):
                token = indices[row, rank]
            score = cutlass.Float64(logprobs[row, rank]) + gumbel(seed, position, rank)
        if (score > best) | ((score == best) & (rank < best_rank)):
            best, best_rank = score, rank
            best_token = token
    iket.range_pop()
    iket.range_push("sample_reduce_store")
    if cutlass.const_expr(warps > 1):
        scores = cutlass.Array(cutlass.Float64, warps, space=cutlass.AddressSpace.smem, alignment=8)
        ranks = cutlass.Array(cutlass.Int32, warps, space=cutlass.AddressSpace.smem, alignment=4)
        selected = warp_winner(best, best_rank)
        selected_score = cute.arch.shuffle_sync(best, selected % 32)
        if lane == 0:
            scores[warp], ranks[warp] = selected_score, selected
        p.barrier_cta_sync()
        if warp == 0:
            best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
            if lane < warps:
                best, best_rank = scores[lane], ranks[lane]
            best_rank = warp_winner(best, best_rank)
            if lane == 0:
                output[row] = indices[row, best_rank]
    else:
        best_rank = warp_winner(best, best_rank)
        best_token = cute.arch.shuffle_sync(best_token, best_rank % 32)
        if lane == 0:
            output[row] = best_token
    iket.range_pop()


@cute.jit
def sorted_sample_host(logprobs, indices, seeds, positions, output, stream):
    count = logprobs.shape[1]
    warps = 1 if count <= 32 else (4 if count <= 128 else (8 if count <= 512 else 32))
    sorted_sample_kernel(logprobs, indices, seeds, positions, output, logprobs.shape[1], warps).launch(
        grid=(logprobs.shape[0], 1, 1), block=(warps * 32, 1, 1), stream=stream
    )


@cache
def compile_sorted(count, arch):
    batch = cutlass.sym_int64()

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        sorted_sample_host,
        fake(cutlass.Float32, batch, count),
        fake(cutlass.Int64, batch, count),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


@torch.compiler.disable
def sample_sorted(logprobs, indices, seeds, positions):
    """Sample prefiltered FP32 log probabilities using their supplied candidate order."""
    if logprobs.ndim != 2 or logprobs.dtype != torch.float32 or not logprobs.is_cuda:
        raise ValueError("Expected CUDA FP32 log probabilities [B,K]")
    batch, count = logprobs.shape
    if batch < 1 or not 1 <= count <= 2048 or indices.shape != logprobs.shape:
        raise ValueError("Expected matching [B,K] probabilities/indices, B > 0 and K <= 2048")
    if (
        not logprobs.is_contiguous()
        or any(
            t.device != logprobs.device or t.dtype != torch.int64 or not t.is_contiguous()
            for t in (indices, seeds, positions)
        )
        or seeds.shape != (batch,)
        or positions.shape != (batch,)
    ):
        raise ValueError("Expected contiguous INT64 indices, seeds and positions on the same CUDA device")
    output = indices.new_empty(batch)
    compile_sorted(count, device_arch(logprobs.device))(
        logprobs,
        indices,
        seeds,
        positions,
        output,
        cuda.CUstream(torch.cuda.current_stream(logprobs.device).cuda_stream),
    )
    return output


@torch.compiler.disable
def sample_logits(
    logits,
    temperatures,
    top_ks,
    top_ps,
    seeds,
    positions,
    *,
    max_top_k=50,
    table=None,
    accumulated=None,
    projected_table=None,
    out_codes=None,
    warps=8,
):
    """Sample BF16 [B,2048] logits; optionally return embeddings and update their sum.

    Per-row top_k must be in [1,max_top_k]. Logits must be finite or -inf, with
    at least one finite value per row. Equal logits use ascending token IDs,
    independent of PyTorch's unspecified topk tie ordering. Temperatures must
    be finite and are clamped to at least 1e-5.
    """
    if logits.ndim != 2 or logits.shape[1] != 2048 or logits.dtype != torch.bfloat16 or not logits.is_cuda:
        raise ValueError("Expected CUDA BF16 logits [B,2048]")
    batch = logits.shape[0]
    if batch == 0 or max_top_k < 1 or max_top_k > 128 or warps not in (4, 8):
        raise ValueError("Expected B > 0, 1 <= max_top_k <= 128, and 4 or 8 warps")
    if not logits.is_contiguous() or any(
        t.shape != (batch,) or t.device != logits.device or t.dtype != dtype or not t.is_contiguous()
        for t, dtype in (
            (temperatures, torch.float32),
            (top_ks, torch.int64),
            (top_ps, torch.float32),
            (seeds, torch.int64),
            (positions, torch.int64),
        )
    ):
        raise ValueError("Expected contiguous logits and per-row FP32 temperature/top_p, INT64 top_k/seed/position")
    width = 0 if table is None else table.shape[1]
    projected_width = 0 if projected_table is None else projected_table.shape[1]
    if (accumulated is not None or projected_table is not None) and table is None:
        raise ValueError("Embedding outputs require an embedding table")
    for tensor, shape in (
        (table, (2048, width)),
        (accumulated, (batch, width)),
        (projected_table, (2048, projected_width)),
    ):
        if tensor is not None and (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or tensor.device != logits.device
            or not tensor.is_contiguous()
            or shape[1] < 8
            or shape[1] % 8
        ):
            raise ValueError("Expected contiguous BF16 tables/accumulator with widths divisible by eight")
    if out_codes is not None and (
        out_codes.shape != (batch,)
        or out_codes.dtype != torch.int64
        or out_codes.device != logits.device
        or out_codes.stride(0) < 1
    ):
        raise ValueError("Expected INT64 output slots [B] on the logits device")
    codes = logits.new_empty(batch, dtype=torch.int64) if out_codes is None else out_codes
    gathered = logits.new_empty(batch, width) if width else None
    projected = logits.new_empty(batch, projected_width) if projected_width else None
    compile_kernel(
        max_top_k,
        width,
        projected_width,
        accumulated is not None,
        codes.stride(0),
        warps,
        device_arch(logits.device),
    )(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        codes,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    return codes, gathered, projected


def torch_gumbel(seeds, positions, ranks):
    """Independent tensor reference for the seeded categorical draw."""
    mask = 0xFFFFFFFF
    h = torch.zeros_like(seeds[:, None] + ranks[None, :])
    for word in (seeds[:, None] & mask, (seeds[:, None] >> 32) & mask, positions[:, None] & mask, ranks[None, :]):
        # The signed multiplier is equivalent modulo 2**32 and avoids an
        # out-of-range int32 literal in Inductor's arange specialization.
        word = (word * -862048943) & mask
        word = ((word << 15) | (word >> 17)) & mask
        word = (word * 0x1B873593) & mask
        h = h ^ word
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    h = h ^ 16
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & mask
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & mask
    h = h ^ (h >> 16)
    log_u = (h.double() / 4294967295.0).log().clamp(min=torch.finfo(torch.float64).min, max=-(2**-32))
    return -(-log_u).log()


def reference(
    logits,
    temperatures,
    top_ks,
    top_ps,
    seeds,
    positions,
    *,
    max_top_k=50,
    table=None,
    accumulated=None,
    projected_table=None,
    out_codes=None,
):
    bits = logits.view(torch.int16).int() & 0xFFFF
    bits = torch.where(bits & 0x7FFF == 0, 0, bits)
    ordered = bits ^ torch.where(bits & 0x8000 != 0, 0xFFFF, 0x8000).int()
    tokens = torch.arange(2048, device=logits.device, dtype=torch.int32)
    keys = ((ordered << 16) | (0xFFFF - tokens)) ^ -2147483648
    indices = keys.topk(max_top_k, dim=-1).indices
    sorted_scores = logits.float().gather(1, indices) / temperatures[:, None].clamp_min(1e-5)
    ranks = torch.arange(max_top_k, device=logits.device)
    sorted_scores = sorted_scores.masked_fill(ranks[None, :] >= top_ks[:, None], -float("inf"))
    probabilities = sorted_scores.softmax(-1)
    remove = (probabilities.cumsum(-1) - probabilities >= top_ps[:, None]) & (ranks[None, :] != 0)
    remove = remove & ((top_ps > 0) & (top_ps < 1))[:, None]
    logprobs = probabilities.log().masked_fill(remove, -float("inf"))
    rank = (logprobs.double() + torch_gumbel(seeds, positions, ranks)).argmax(-1)
    codes = indices.gather(1, rank[:, None]).flatten()
    if out_codes is not None:
        out_codes.copy_(codes)
        codes = out_codes
    gathered = torch.nn.functional.embedding(codes, table) if table is not None else None
    if accumulated is not None:
        accumulated.add_(gathered)
    projected = torch.nn.functional.embedding(codes, projected_table) if projected_table is not None else None
    return codes, gathered, projected


def sorted_reference(logprobs, indices, seeds, positions):
    ranks = torch.arange(logprobs.shape[1], device=logprobs.device)
    selected = (logprobs.double() + torch_gumbel(seeds, positions, ranks)).argmax(-1)
    return indices.gather(1, selected[:, None]).flatten()


def inputs(batch, top_k):
    logits = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    temperatures = torch.linspace(0.2, 1.7, batch, device="cuda")
    top_ks = torch.full((batch,), top_k, device="cuda", dtype=torch.int64)
    top_ps = torch.full((batch,), 0.9, device="cuda")
    seeds = torch.randint(2**48, (batch,), device="cuda")
    positions = torch.arange(batch, device="cuda")
    return logits, temperatures, top_ks, top_ps, seeds, positions


@torch.inference_mode()
def validate():
    for batch, top_k in ((1, 1), (7, 16), (16, 50), (32, 128)):
        tensors = inputs(batch, top_k)
        table = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        projected = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
        accumulated = table.new_zeros(batch, 2048)
        slots = torch.full((batch, 16), -1, device="cuda", dtype=torch.int64)
        for warps in (4, 8):
            options = {
                "max_top_k": top_k,
                "table": table,
                "accumulated": accumulated,
                "projected_table": projected,
                "out_codes": slots[:, 4],
            }
            sample_logits(*tensors, **options, warps=warps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = sample_logits(*tensors, **options, warps=warps)
            for mode in ("random", "ties", "mask", "heterogeneous"):
                tensors[0].normal_()
                tensors[4].random_(-(2**50), 2**50)
                tensors[5].add_(17)
                if mode == "ties":
                    tensors[0].zero_()
                    tensors[0][:, ::2] = -0.0
                elif mode == "mask":
                    tensors[0][:, 100:].fill_(-float("inf"))
                elif mode == "heterogeneous":
                    tensors[2].random_(1, top_k + 1)
                    tensors[3].copy_(torch.linspace(0.01, 1.0, batch, device="cuda"))
                accumulated.normal_()
                target_sum = accumulated.clone()
                expected = reference(*tensors, **dict(options, accumulated=target_sum, out_codes=None))
                graph.replay()
                for output, target in zip(actual, expected):
                    torch.testing.assert_close(output, target, atol=0, rtol=0)
                torch.testing.assert_close(accumulated, target_sum, atol=0, rtol=0)
                assert (slots[:, :4] == -1).all() and (slots[:, 5:] == -1).all()
        print(f"PASS sampling B={batch} K={top_k}, seeded graph replay and embedding updates", flush=True)
    for count in (7, 64, 129, 2048):
        tensors = inputs(16, 50)
        logprobs = torch.randn(16, count, device="cuda").log_softmax(-1)
        indices = torch.randint(2048, (16, count), device="cuda")
        sample_sorted(logprobs, indices, tensors[4], tensors[5])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sample_sorted(logprobs, indices, tensors[4], tensors[5])
        for _ in range(3):
            tensors[4].random_(-(2**50), 2**50)
            tensors[5].add_(19)
            graph.replay()
            torch.testing.assert_close(
                actual, sorted_reference(logprobs, indices, tensors[4], tensors[5]), atol=0, rtol=0
            )
        print(f"PASS prefiltered sampling K={count}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--top-k", type=int, nargs="+", default=[50])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    validate()
    results = []
    for batch in args.batch_size:
        for top_k in args.top_k:
            tensors = inputs(batch, top_k)
            for with_embedding in (False, True):
                torch.compiler.reset()
                compiled = torch.compile(reference, fullgraph=True)
                table = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) if with_embedding else None
                projected = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16) if with_embedding else None
                sums = [table.new_zeros(batch, 2048) if with_embedding else None for _ in range(2)]
                options = {"max_top_k": top_k, "table": table, "projected_table": projected}
                expected = reference(*tensors, **options)
                actual = compiled(*tensors, **options)
                for output, target in zip(actual, expected):
                    if target is not None:
                        torch.testing.assert_close(output, target, atol=0, rtol=0)
                for warps in (4, 8) if args.tune else (8,):
                    times = measure(
                        {
                            "Compiled": partial(compiled, *tensors, accumulated=sums[0], **options),
                            "Fused": partial(sample_logits, *tensors, accumulated=sums[1], warps=warps, **options),
                        },
                        calls_per_graph=16,
                        profile=args.profile,
                    )
                    results.append(((batch, top_k, with_embedding, warps), times))
    sorted_results = []
    for batch in args.batch_size:
        for count in (50, 2048):
            torch.compiler.reset()
            compiled = torch.compile(sorted_reference, fullgraph=True)
            tensors = inputs(batch, 50)
            logprobs = torch.randn(batch, count, device="cuda").log_softmax(-1)
            indices = torch.arange(count, device="cuda").expand(batch, -1).contiguous()
            arguments = (logprobs, indices, tensors[4], tensors[5])
            torch.testing.assert_close(compiled(*arguments), sorted_reference(*arguments), atol=0, rtol=0)
            times = measure(
                {"Compiled": partial(compiled, *arguments), "Fused": partial(sample_sorted, *arguments)},
                calls_per_graph=16,
                profile=args.profile,
            )
            sorted_results.append(((batch, count), times))
    print_kernel_summary(
        results,
        columns=["Batch", "K", "Embedding", "Warps"],
        workload="sampling, BF16, V=2048, optional D=2048/DP=1024 embedding update",
    )
    print_kernel_summary(
        sorted_results, columns=["Batch", "K"], workload="prefiltered sampling, FP32 log-probabilities"
    )


if __name__ == "__main__":
    main()
