"""Hopper fused post-attention for hidden=2048 and FFN=6144."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from .mma import descriptor, swizzle
from .mma import fragment_column as col64
from .mma import fragment_row as row64
from .mma import mma as mma64
from .post_attention import PAYLOAD_EMPTY, wait_barrier


@cute.jit
def gate_up_index(row, col, N: cutlass.Constexpr):
    """Swizzle channels for fragment stores and contiguous paired SiLU reads."""
    return col * 128 + (row ^ (((col // 2) % 4) * 8))


@cute.jit
def fragment_row(thread, index, N: cutlass.Constexpr):
    """Map the paired M64 accumulators onto 128 output channels."""
    return row64(thread, index % (N // 2)) + (index // (N // 2)) * 64


@cute.jit
def fragment_column(thread, index, N: cutlass.Constexpr):
    """Both M64 accumulators cover the same batch columns."""
    return col64(thread, index % (N // 2))


@cute.jit
def scatter_partial(acc, mailbox, ready, phase: cutlass.Constexpr, tid, rank, N: cutlass.Constexpr):
    """Scatter four K partials; each rank owns 32 output channels."""
    iket.range_push("splitk_store")
    if tid >= 128:
        for i in cutlass.range_constexpr(N):
            row = fragment_row(tid - 128, i, N)
            col = fragment_column(tid - 128, i, N)
            cute.arch.store_async_dsmem(
                cute.make_ptr(cutlass.Float32, mailbox.data_ptr((rank * N + col) * 32 + row % 32).ir_value()),
                acc[i].bitcast(cutlass.Int32),
                cute.make_ptr(cutlass.Int64, ready.data_ptr(phase).ir_value()),
                peer_cta_rank=row // 32,
            )
    iket.range_pop()
    iket.range_push("splitk_receive")
    wait_barrier(ready.subview(phase), 0)
    iket.range_pop()


@cute.jit
def reduce_partial(mailbox, row, col, N: cutlass.Constexpr):
    """Preserve the baseline four-way FP32 accumulation order."""
    value = cutlass.Float32(0)
    for peer in cutlass.range_constexpr(4):
        value += mailbox.load(idx=(peer * N + col) * 32 + row % 32)
    return value


@cute.jit
def project(
    acc,
    weights,
    operands,
    inverse_rms,
    r,
    gamma,
    bar,
    free,
    weight_map,
    attention,
    activation,
    tid,
    rank,
    tile,
    slot,
    MODE: cutlass.Constexpr,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
):
    """Double-buffer weights and operands between producer and MMA warpgroups."""
    K = 256
    stages = (2048 if MODE == "gate_up" else (1536 if MODE == "down" else 512)) // K
    acc.fill(0)
    acc0 = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
    acc1 = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
    acc0.fill(0)
    acc1.fill(0)
    if tid == 0:
        for buffer in cutlass.range_constexpr(2):
            p.mbarrier_init(bar.subview(buffer), 1)
            p.mbarrier_init(free.subview(buffer), 1)
            p.mbarrier_arrive(free.subview(buffer))
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    if cutlass.const_expr(MODE == "output"):
        iket.range_push("output_projection", tile)
    elif cutlass.const_expr(MODE == "gate_up"):
        iket.range_push("gate_up_projection", tile)
    else:
        iket.range_push("down_projection", tile)
    if tid < 128:
        for stage in cutlass.range(stages, unroll=1):
            buffer = stage % 2
            wait_barrier(free.subview(buffer), (stage // 2) & 1)
            kbase = stage * K if MODE == "gate_up" else rank * (1536 if MODE == "down" else 512) + stage * K
            if tid == 0:
                p.mbarrier_expect_tx(bar.subview(buffer), 128 * K * 2)
                p.cp_async_bulk_tensor_shared_cluster_global(
                    weights.subview(buffer * 128 * K),
                    weight_map.get_ptr(),
                    [0, 0, kbase // 64, tile],
                    bar.subview(buffer),
                    [],
                )
            if cutlass.const_expr(MODE == "down"):
                iket.range_push("activation_wait", stage)
                payload = cute.make_rmem_tensor((4, K * N // 1024), cutlass.Int32)
                for j in cutlass.range_constexpr(K * N // 1024):
                    idx = (tid + j * 128) * 8
                    slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                    channel = kbase + slab * 64 + k
                    offset = ((slot * 96 + channel // 64) * N + row) * 32 + channel % 64 // 2
                    values = p.load_ext(
                        activation.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile"
                    )
                    for word in cutlass.range_constexpr(4):
                        payload[word, j] = values[word]
                ready = cutlass.Boolean(False)
                while not ready:
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(K * N // 1024):
                        if cutlass.any_(payload[None, j].load().to_vector() == PAYLOAD_EMPTY):
                            idx = (tid + j * 128) * 8
                            slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                            channel = kbase + slab * 64 + k
                            offset = ((slot * 96 + channel // 64) * N + row) * 32 + channel % 64 // 2
                            values = p.load_ext(
                                activation.iterator.raw_ptr() + offset,
                                dtype=cutlass.Int32,
                                count=4,
                                order="volatile",
                            )
                            for word in cutlass.range_constexpr(4):
                                payload[word, j] = values[word]
                        ready &= cutlass.all_(payload[None, j].load().to_vector() != PAYLOAD_EMPTY)
                iket.range_pop()
            if cutlass.const_expr(MODE == "gate_up"):
                payload = cute.make_rmem_tensor((4, K * N // 1024), cutlass.Int32)
                scales = cute.make_rmem_tensor((4, K * N // 1024), cutlass.Int32)
                for j in cutlass.range_constexpr(K * N // 1024):
                    idx = (tid + j * 128) * 8
                    slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                    channel = kbase + slab * 64 + k
                    offset = (slot * N + row) * 1024 + channel // 2
                    values = p.load_ext(r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile")
                    scale = (gamma.iterator.raw_ptr() + channel).load(count=8, alignment=16).bitcast(cutlass.Int32)
                    for word in cutlass.range_constexpr(4):
                        payload[word, j] = values[word]
                        scales[word, j] = scale[word]
            iket.range_push("operand_copy", stage)
            for j in cutlass.range_constexpr(K * N // 1024):
                idx = (tid + j * 128) * 8
                slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                value = cutlass.full((8,), 0, cutlass.BFloat16)
                if cutlass.const_expr(MODE == "output"):
                    if row < B:
                        value = (attention.iterator.raw_ptr() + row * 2048 + kbase + slab * 64 + k).load(
                            count=8, alignment=16
                        )
                elif cutlass.const_expr(MODE == "gate_up"):
                    words = payload[None, j].load().to_vector()
                    raw = words.bitcast(cutlass.BFloat16).to(cutlass.Float32)
                    scale = scales[None, j].load().to_vector().bitcast(cutlass.BFloat16).to(cutlass.Float32)
                    unit = (raw * inverse_rms.load(idx=row)).to(cutlass.BFloat16).to(cutlass.Float32)
                    value = (unit * scale).to(cutlass.BFloat16)
                else:
                    value = payload[None, j].load().to_vector().bitcast(cutlass.BFloat16)
                operands.store(value, idx=buffer * N * K + slab * N * 64 + swizzle(row, k), alignment=16)
            iket.range_pop()
            p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
            p.barrier_cta_sync(barrier_id=1, thread_count=128)
            if tid == 0:
                p.mbarrier_arrive(bar.subview(buffer))
    else:
        for stage in cutlass.range(stages, unroll=1):
            buffer = stage % 2
            iket.range_push("operand_wait", stage)
            wait_barrier(bar.subview(buffer), (stage // 2) & 1)
            iket.range_pop()
            p.wgmma_fence_aligned()
            iket.range_push("mma", stage)
            da_base = descriptor(weights, buffer * 128 * K)
            db_base = descriptor(operands, buffer * N * K)
            for slab in cutlass.range_constexpr(K // 64):
                for k in cutlass.range_constexpr(4):
                    mma64(acc0, da_base + slab * 1024 + 2 * k, db_base + slab * N * 8 + 2 * k, N)
                    mma64(acc1, da_base + slab * 1024 + 512 + 2 * k, db_base + slab * N * 8 + 2 * k, N)
            p.wgmma_commit_group_sync_aligned()
            p.wgmma_wait_group_sync_aligned(0)
            if stage == stages - 1:
                for i in cutlass.range_constexpr(N // 2):
                    acc[i] = acc0[i]
                    acc[i + N // 2] = acc1[i]
            iket.range_pop()
            if tid == 128:
                p.mbarrier_arrive(free.subview(buffer))
    p.barrier_cta_sync()
    iket.range_pop()


@cute.jit
def compute_inverse_rms(r, inverse_rms, tid, slot, N: cutlass.Constexpr, EPS: cutlass.Constexpr):
    """Poll each residual word and retain one FP32 inverse RMS per token."""
    iket.range_push("residual_norm")
    for token in cutlass.range(N // 8, unroll=1):
        row = tid // 32 + token * 8
        words = cute.make_rmem_tensor((4, 8), cutlass.Int32)
        for group in cutlass.range_constexpr(8):
            offset = (slot * N + row) * 1024 + (tid % 32) * 4 + group * 128
            values = p.load_ext(r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile")
            for i in cutlass.range_constexpr(4):
                words[i, group] = values[i]
        ready = cutlass.Boolean(False)
        while not ready:
            ready = cutlass.Boolean(True)
            for group in cutlass.range_constexpr(8):
                if cutlass.any_(words[None, group].load().to_vector() == PAYLOAD_EMPTY):
                    offset = (slot * N + row) * 1024 + (tid % 32) * 4 + group * 128
                    values = p.load_ext(r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile")
                    for i in cutlass.range_constexpr(4):
                        words[i, group] = values[i]
                ready &= cutlass.all_(words[None, group].load().to_vector() != PAYLOAD_EMPTY)
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(8):
            values = words[None, group].load().to_vector().bitcast(cutlass.BFloat16).to(cutlass.Float32)
            squares = values * values
            total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                (squares[4] + squares[5]) + (squares[6] + squares[7])
            )
        for delta in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
        inv = cute.math.rsqrt(total / 2048 + EPS)
        if tid % 32 == 0:
            inverse_rms.store(inv, idx=row)
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    iket.range_pop()


@cute.kernel
def post_attention_kernel(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cutlass.GridConstant[tensor_map.TensorMap],
    wgu: cutlass.GridConstant[tensor_map.TensorMap],
    wd: cutlass.GridConstant[tensor_map.TensorMap],
    gamma: cute.Tensor,
    r: cute.Tensor,
    activation: cute.Tensor,
    epochs: cute.Tensor,
    output: cute.Tensor,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    """Use one channel wave: 64 CTAs for output/down and all 96 for gate/up."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    cluster, rank = bid // 4, bid % 4
    weights = cutlass.Array(cutlass.BFloat16, 65536, space=cutlass.AddressSpace.smem, alignment=1024)
    inverse_rms = cutlass.Array(cutlass.Float32, N, space=cutlass.AddressSpace.smem, alignment=128)
    operands = cutlass.Array(cutlass.BFloat16, 2 * N * 256, space=cutlass.AddressSpace.smem, alignment=1024)
    partial = cutlass.Array(cutlass.Float32, N * 128, space=cutlass.AddressSpace.smem, alignment=128)
    mailbox = cutlass.Array(cutlass.Float32, N * 128, space=cutlass.AddressSpace.smem, alignment=128)
    bar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    free = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    ready = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    epoch_s = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if tid == 0:
        epoch_s.store(epochs[bid] + 1)
        epochs[bid] = epochs[bid] + 1
        for phase in cutlass.range_constexpr(2):
            p.mbarrier_init(ready.subview(phase), 1)
            p.mbarrier_arrive_expect_tx(ready.subview(phase), 4 * 32 * N * 4)
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    slot = epoch_s.load() & 1
    p.barrier_cluster_arrive_relaxed()
    p.barrier_cluster_wait()
    # Double-buffered payloads: reset only the next serialized invocation's slot.
    tile = bid
    for j in cutlass.range_constexpr(N // 8):
        offset = ((1 - slot) * 96 + tile) * N * 32 + tid + j * 256
        p.store_ext(cutlass.Int32(PAYLOAD_EMPTY).ir_value(), activation.iterator.raw_ptr() + offset, order="volatile")
    acc = cute.make_rmem_tensor((N,), cutlass.Float32)
    if cluster < 16:
        tile = cluster
        project(
            acc,
            weights,
            operands,
            inverse_rms,
            r,
            gamma,
            bar,
            free,
            wo,
            attention,
            activation,
            tid,
            rank,
            tile,
            slot,
            "output",
            B,
            N,
        )
        scatter_partial(acc, mailbox, ready, 0, tid, rank, N)
        for j in cutlass.range_constexpr(N // 8):
            idx = tid + j * 256
            row, col = rank * 32 + idx % 32, idx // 32
            value = cutlass.BFloat16(0)
            if col < B:
                projected = cutlass.BFloat16(reduce_partial(mailbox, row, col, N))
                value = cutlass.BFloat16(cutlass.Float32(projected) + cutlass.Float32(residual[col, tile * 128 + row]))
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            neighbor = cute.arch.shuffle_sync_bfly(bits, 1)
            if row % 2 == 0:
                offset = (slot * N + col) * 1024 + tile * 64 + row // 2
                p.store_ext((bits | (neighbor << 16)).ir_value(), r.iterator.raw_ptr() + offset, order="volatile")
                reset = ((1 - slot) * N + col) * 1024 + tile * 64 + row // 2
                p.store_ext(cutlass.Int32(PAYLOAD_EMPTY).ir_value(), r.iterator.raw_ptr() + reset, order="volatile")
        # Join the four ranks before reusing their split-K mailboxes.
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
    compute_inverse_rms(r, inverse_rms, tid, slot, N, EPS)
    tile = bid
    project(
        acc,
        weights,
        operands,
        inverse_rms,
        r,
        gamma,
        bar,
        free,
        wgu,
        attention,
        activation,
        tid,
        rank,
        tile,
        slot,
        "gate_up",
        B,
        N,
    )
    if tid >= 128:
        for i in cutlass.range_constexpr(N):
            row, col = fragment_row(tid - 128, i, N), fragment_column(tid - 128, i, N)
            partial.store(cutlass.Float32(cutlass.BFloat16(acc[i])), idx=gate_up_index(row, col, N))
    p.barrier_cta_sync()
    iket.range_push("silu_and_gate")
    for j in cutlass.range_constexpr(N // 8):
        idx = tid + j * 256
        channel, batch = 2 * (idx % 32), idx // 32
        packed = cutlass.Int32(0)
        for half in cutlass.range_constexpr(2):
            gate = partial.load(idx=gate_up_index(channel + half, batch, N))
            up = partial.load(idx=gate_up_index(channel + half + 64, batch, N))
            silu = cutlass.BFloat16(gate / (1 + cute.math.exp(-gate)))
            value = cutlass.BFloat16(cutlass.Float32(silu) * up)
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            packed |= bits << (16 * half)
        offset = ((slot * 96 + tile) * N + batch) * 32 + channel // 2
        p.store_ext(packed.ir_value(), activation.iterator.raw_ptr() + offset, order="volatile")
    p.barrier_cta_sync()
    iket.range_pop()
    if cluster < 16:
        tile = cluster
        project(
            acc,
            weights,
            operands,
            inverse_rms,
            r,
            gamma,
            bar,
            free,
            wd,
            attention,
            activation,
            tid,
            rank,
            tile,
            slot,
            "down",
            B,
            N,
        )
        scatter_partial(acc, mailbox, ready, 1, tid, rank, N)
        for j in cutlass.range_constexpr(N // 8):
            idx = tid + j * 256
            row, col = rank * 32 + idx % 32, idx // 32
            value = cutlass.BFloat16(reduce_partial(mailbox, row, col, N))
            if col < B:
                offset = (slot * N + col) * 1024 + tile * 64 + row // 2
                word = p.load_ext(r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, order="volatile")
                skip = cutlass.Uint16((word >> (16 * (row % 2))) & 0xFFFF).bitcast(cutlass.BFloat16)
                output[col, tile * 128 + row] = cutlass.BFloat16(cutlass.Float32(value) + cutlass.Float32(skip))
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()


@cute.jit
def post_attention_host(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cute.Tensor,
    wgu: cute.Tensor,
    wd: cute.Tensor,
    gamma: cute.Tensor,
    r: cute.Tensor,
    activation: cute.Tensor,
    epochs: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    """Build K-stage tensor maps and launch the cooperative wide schedule."""
    wo_map = tensor_map.create_tensor_map_tiled_from_view(
        wo, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wgu_map = tensor_map.create_tensor_map_tiled_from_view(
        wgu, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wd_map = tensor_map.create_tensor_map_tiled_from_view(
        wd, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    post_attention_kernel(
        attention, residual, wo_map, wgu_map, wd_map, gamma, r, activation, epochs, output, B, N, EPS
    ).launch(grid=(96, 1, 1), block=(256, 1, 1), cluster=(4, 1, 1), cooperative=True, stream=stream)
