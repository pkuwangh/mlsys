"""Blackwell fused QKV preparation."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from .mma import descriptor, fragment_column, fragment_row, load_accumulator, mma, swizzle


@cute.kernel
def pre_attention_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weight_map: cutlass.GridConstant[tensor_map.TensorMap],
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    mask: cute.Tensor | None,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    CAPACITY: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    head, rank = bid // 4, bid % 4
    split = rank % 2
    weights = cutlass.Array(cutlass.BFloat16, 32768, space=cutlass.AddressSpace.smem, alignment=1024)
    normalized = cutlass.Array(cutlass.BFloat16, N * 512, space=cutlass.AddressSpace.smem, alignment=1024)
    partials = cutlass.Array(cutlass.Float32, 4 * N * 64, space=cutlass.AddressSpace.smem, alignment=128)
    local_partial = cutlass.Array(cutlass.Float32, N * 64, space=cutlass.AddressSpace.smem, alignment=128)
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if tid >= 128 and tid < 160:
        p.tcgen05_alloc(tmem_address, 32)
        p.tcgen05_relinquish_alloc_permit()
    if tid == 0:
        iket.range_push("qkv_weight_prefetch")
        p.prefetch_tensormap(weight_map.get_ptr())
        p.mbarrier_init(bars, 1)
        p.mbarrier_init(mma_done, 1)
        if rank == 0:
            p.mbarrier_init(bars.subview(1), 1)
            p.mbarrier_arrive_expect_tx(bars.subview(1), 3 * N * 64 * 4)
        p.fence_mbarrier_init()
        p.mbarrier_arrive_expect_tx(bars, 65536)
        p.cp_async_bulk_tensor_shared_cluster_global(
            weights, weight_map.get_ptr(), [0, 0, split * 8, head * 2 + rank // 2], bars, []
        )
        iket.range_pop()

    iket.range_push("input_norm")
    lane, warp = tid % 32, tid // 32
    for token in cutlass.range_constexpr(N // 16):
        row = warp + token * 16
        values = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
        scales = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
        for group in cutlass.range_constexpr(4):
            col = lane * 8 + group * 256
            val = cutlass.full((8,), 0, cutlass.BFloat16)
            if row < B:
                val = (x.iterator.raw_ptr() + row * 1024 + col).load(count=8, alignment=16)
            scale = (gamma.iterator.raw_ptr() + col).load(count=8, alignment=16)
            for i in cutlass.range_constexpr(8):
                values[i, group], scales[i, group] = val[i], scale[i]
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(4):
            v = values[None, group].load().to_vector().to(cutlass.Float32)
            squares = v * v
            total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                (squares[4] + squares[5]) + (squares[6] + squares[7])
            )
        for shift in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
        inverse = cute.math.rsqrt(total / 1024 + EPS)
        for group in cutlass.range_constexpr(4):
            if group // 2 == split:
                col = lane * 8 + (group % 2) * 256
                v = values[None, group].load().to_vector().to(cutlass.Float32)
                norm_scale = scales[None, group].load().to_vector().to(cutlass.Float32)
                norm_result = ((v * inverse).to(cutlass.BFloat16).to(cutlass.Float32) * norm_scale).to(cutlass.BFloat16)
                normalized.store(norm_result, idx=(col // 64) * N * 64 + swizzle(row, col % 64), alignment=16)
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    # Join the cluster before publishing DSMEM, but overlap its arrival with MMA.
    p.barrier_cluster_arrive_relaxed()
    iket.range_pop()

    acc = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
    acc.fill(0)
    if tid >= 128 and tid < 256:
        iket.range_push("qkv_operand_wait")
        while not p.mbarrier_try_wait_parity(bars, 0):
            pass
        iket.range_pop()
        p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
        iket.range_push("qkv_mma")
        if tid == 128:
            for slab in cutlass.range_constexpr(8):
                da = descriptor(weights, slab * 4096)
                db = descriptor(normalized, slab * N * 64)
                for k in cutlass.range_constexpr(4):
                    mma(tmem_address.load(), da + 2 * k, db + 2 * k, N, slab != 0 or k != 0)
            p.tcgen05_commit(mma_done)
        while not p.mbarrier_try_wait_parity(mma_done, 0):
            pass
        load_accumulator(acc, tmem_address.load(), tid - 128, N)
        iket.range_pop()
    iket.range_push("qkv_partial_publish")
    if tid >= 128 and tid < 256:
        for i in cutlass.range_constexpr(N // 2):
            channel = fragment_row(tid - 128, i)
            row = fragment_column(tid - 128, i)
            if rank == 0:
                partials.store(acc[i], idx=row * 64 + channel)
            else:
                local_partial.store(acc[i], idx=row * 64 + channel)
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    p.barrier_cluster_wait()
    if tid == 0 and rank != 0:
        p.cp_async_bulk_shared_cluster_shared_cta(
            p.mapa(partials.subview(rank * N * 64), 0), local_partial, p.mapa(bars.subview(1), 0), N * 64 * 4
        )
    iket.range_pop()
    if rank == 0:
        iket.range_push("qkv_partial_wait")
        while not p.mbarrier_try_wait_parity(bars.subview(1), 0):
            pass
        iket.range_pop()
        iket.range_push("head_norm_rope_cache")
        position = positions[0]
        for token in cutlass.range_constexpr(N // 16):
            row = warp + token * 16
            channel = lane * 4
            half, local_channel = channel // 64, channel % 64
            first = partials.load(idx=(half * 2 * N + row) * 64 + local_channel, vector_size=4, alignment=16)
            second = partials.load(idx=((half * 2 + 1) * N + row) * 64 + local_channel, vector_size=4, alignment=16)
            projected = (first + second).to(cutlass.BFloat16)
            if head < 24:
                vf = projected.to(cutlass.Float32)
                head_squares = vf * vf
                total = (head_squares[0] + head_squares[1]) + (head_squares[2] + head_squares[3])
                for shift in cutlass.range_constexpr(5):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                inverse = cute.math.rsqrt(total / 128 + EPS)
                head_scale = cutlass.full((4,), 0, cutlass.BFloat16)
                if head < 16:
                    head_scale = (qgamma.iterator.raw_ptr() + channel).load(count=4, alignment=8)
                else:
                    head_scale = (kgamma.iterator.raw_ptr() + channel).load(count=4, alignment=8)
                unit = (vf * inverse).to(cutlass.BFloat16).to(cutlass.Float32)
                scaled = (unit * head_scale.to(cutlass.Float32)).to(cutlass.BFloat16).to(cutlass.Float32)
                rotated = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    opposite = cute.arch.shuffle_sync_bfly(scaled[i], 16)
                    rotated[i] = cutlass.select_(channel < 64, -opposite, opposite)
                cv = (cos.iterator.raw_ptr() + channel).load(count=4, alignment=8).to(cutlass.Float32)
                sv = (sin.iterator.raw_ptr() + channel).load(count=4, alignment=8).to(cutlass.Float32)
                left = (scaled * cv).to(cutlass.BFloat16).to(cutlass.Float32)
                right = (rotated.load().to_vector() * sv).to(cutlass.BFloat16).to(cutlass.Float32)
                result = (left + right).to(cutlass.BFloat16)
                if row < B:
                    if head < 16:
                        (q.iterator.raw_ptr() + (row * 16 + head) * 128 + channel).store(result, alignment=8)
                    else:
                        (kv.iterator.raw_ptr() + ((row * 8 + head - 16) * CAPACITY + position) * 128 + channel).store(
                            result, alignment=8
                        )
            else:
                if row < B:
                    offset = B * 8 * CAPACITY * 128 + ((row * 8 + head - 24) * CAPACITY + position) * 128 + channel
                    (kv.iterator.raw_ptr() + offset).store(projected, alignment=8)
        iket.range_pop()
    if cutlass.const_expr(mask is not None):  # noqa: SIM102 - preserve DSL specialization guard
        # Rank 1 of the first cluster is free while rank 0 finishes its epilogue.
        if bid == 1:
            position = positions[0]
            for col in cutlass.range(tid, CAPACITY, 512):
                mask[0, col] = cutlass.BFloat16(cutlass.select_(col <= position, 0.0, -float("inf")))
    p.barrier_cluster_arrive_relaxed()
    p.barrier_cluster_wait()
    if tid >= 128 and tid < 160:
        p.tcgen05_dealloc(cutlass.inttoptr(tmem_address.load(), 6, cutlass.Int32), 32)


@cute.jit
def pre_attention_host(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weights: cute.Tensor,
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    mask: cute.Tensor | None,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    CAPACITY: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    weight_map = tensor_map.create_tensor_map_tiled_from_view(
        weights, (1, 8, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    pre_attention_kernel(
        x, gamma, weight_map, qgamma, kgamma, cos, sin, positions, kv, q, mask, B, N, CAPACITY, EPS
    ).launch(grid=(128, 1, 1), block=(512, 1, 1), cluster=(4, 1, 1), stream=stream)
