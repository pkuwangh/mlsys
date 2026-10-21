"""Hopper fused output projection and MLP."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from .mma import descriptor, fragment_column, fragment_row, mma, swizzle

# Two BF16 negative zeros, reserved for an unpublished aligned 32-bit word.
PAYLOAD_EMPTY = -2147450880


@cute.jit
def wait_barrier(barrier, phase):
    while not p.mbarrier_try_wait_parity(barrier, phase):
        pass


@cute.jit
def prefetch_gate_up(weights, bars, weight_map, tile, tid):
    """Prefetch a resident 128-KiB gate/up tile."""
    iket.range_push("gate_up_weight_prefetch")
    if tid == 0:
        p.mbarrier_init(bars, 1)
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    if tid == 0:
        p.mbarrier_expect_tx(bars, 131072)
        p.cp_async_bulk_tensor_shared_cluster_global(weights, weight_map.get_ptr(), [0, 0, 0, tile], bars, [])
        p.mbarrier_arrive(bars)
    iket.range_pop()


@cute.jit
def activation_words(activation, kbase, tid, j: cutlass.Constexpr, slot, N: cutlass.Constexpr):
    """Load eight BF16 values as four independently published 32-bit words."""
    idx = (tid + j * 128) * 8
    slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
    tile = kbase // 32 + slab * 2 + k // 32
    offset = ((slot * 96 + tile) * N + row) * 16 + k % 32 // 2
    return p.load_ext(activation.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile")


@cute.jit
def residual_words(residual, tid, j: cutlass.Constexpr, slot, N: cutlass.Constexpr):
    """Load one of four 8-value groups per lane; each warp owns a full row."""
    row, col = tid // 32 + (j // 4) * 8, (tid % 32) * 8 + (j % 4) * 256
    offset = (slot * N + row) * 512 + col // 2
    return p.load_ext(residual.iterator.raw_ptr() + offset, dtype=cutlass.Int32, count=4, order="volatile")


@cute.jit
def gemm(
    acc,
    weights,
    operands,
    bars,
    weight_map,
    attention,
    normalized,
    activation,
    epochs,
    epoch,
    tid,
    cluster,
    rank,
    tile,
    MODE: cutlass.Constexpr,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
):
    """Consume resident gate/up operands or stage two split-projection operands."""
    if cutlass.const_expr(MODE == "output"):
        iket.range_push("output_projection")
    elif cutlass.const_expr(MODE == "gate_up"):
        iket.range_push("gate_up_projection", tile)
    else:
        iket.range_push("down_projection")
    if cutlass.const_expr(MODE == "gate_up"):
        if tid >= 128:
            p.wgmma_fence_aligned()
            iket.range_push("operand_wait", cutlass.Int32(0))
            wait_barrier(bars, 0)
            iket.range_pop()
            iket.range_push("mma_issue", cutlass.Int32(0))
            da_base = descriptor(weights)
            db_base = descriptor(normalized)
            # Descriptor addresses use 16-byte units; a weight K64 slab is 8 KiB.
            for slab in cutlass.range_constexpr(16):
                da = da_base + slab * 512
                db = db_base + slab * N * 8
                for k in cutlass.range_constexpr(4):
                    mma(acc, da + 2 * k, db + 2 * k, N)
            p.wgmma_commit_group_sync_aligned()
            iket.range_pop()
            iket.range_push("mma_drain", cutlass.Int32(0))
            p.wgmma_wait_group_sync_aligned(0)
            iket.range_pop()
        p.barrier_cta_sync()
    else:
        # Output weights are already in flight from the kernel prologue.
        if cutlass.const_expr(MODE == "down"):
            if tid == 0:
                for stage in cutlass.range_constexpr(2):
                    p.mbarrier_init(bars.subview(stage), 1)
            p.fence_mbarrier_init()
            p.barrier_cta_sync()
        group = 6 if MODE == "down" else 4
        if tid < 128:
            for stage in range(2):
                if cutlass.const_expr(MODE == "output"):
                    kbase = rank * 512 + stage * 256
                else:
                    kbase = stage * 1536 + rank * 384
                if cutlass.const_expr(MODE == "down"):  # noqa: SIM102
                    if tid == 0:
                        p.mbarrier_expect_tx(bars.subview(stage), group * 8192)
                        p.cp_async_bulk_tensor_shared_cluster_global(
                            weights.subview(stage * group * 4096),
                            weight_map.get_ptr(),
                            [0, 0, kbase // 64, cluster],
                            bars.subview(stage),
                            [],
                        )
                p.barrier_cta_sync(barrier_id=1, thread_count=128)
                if cutlass.const_expr(MODE == "down"):
                    iket.range_push("activation_wait", stage)
                    # Issue all loads before testing them. Polling each vector
                    # separately serializes L2 latency. Retain completed words.
                    payload = cute.make_rmem_tensor((4, group * N // 16), cutlass.Int32)
                    for j in cutlass.range_constexpr(group * N // 16):
                        words = activation_words(activation, kbase, tid, j, epoch & 1, N)
                        for word in cutlass.range_constexpr(4):
                            payload[word, j] = words[word]
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(group * N // 16):
                        ready &= cutlass.all_(payload[None, j].load().to_vector() != PAYLOAD_EMPTY)
                    while not ready:
                        for j in cutlass.range_constexpr(group * N // 16):
                            if cutlass.any_(payload[None, j].load().to_vector() == PAYLOAD_EMPTY):
                                words = activation_words(activation, kbase, tid, j, epoch & 1, N)
                                for word in cutlass.range_constexpr(4):
                                    payload[word, j] = words[word]
                        ready = cutlass.Boolean(True)
                        for j in cutlass.range_constexpr(group * N // 16):
                            ready &= cutlass.all_(payload[None, j].load().to_vector() != PAYLOAD_EMPTY)
                    iket.range_pop()
                iket.range_push("operand_copy", stage)
                for j in cutlass.range_constexpr(group * N // 16):
                    idx = (tid + j * 128) * 8
                    slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                    if cutlass.const_expr(MODE == "output"):
                        value = cutlass.full((8,), 0, cutlass.BFloat16)
                        if row < B:
                            value = (attention.iterator.raw_ptr() + row * 2048 + kbase + slab * 64 + k).load(
                                count=8, alignment=16
                            )
                    else:
                        value = payload[None, j].load().to_vector().bitcast(cutlass.BFloat16)
                    operands.store(value, idx=(stage * group + slab) * N * 64 + swizzle(row, k), alignment=16)
                p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
                p.barrier_cta_sync(barrier_id=1, thread_count=128)
                if tid == 0:
                    p.mbarrier_arrive(bars.subview(stage))
                iket.range_pop()
        else:
            p.wgmma_fence_aligned()
            for stage in range(2):
                iket.range_push("operand_wait", stage)
                wait_barrier(bars.subview(stage), 0)
                iket.range_pop()
                iket.range_push("mma_issue", stage)
                da_base = descriptor(weights, stage * group * 4096)
                db_base = descriptor(operands, stage * group * N * 64)
                for slab in cutlass.range_constexpr(group):
                    da = da_base + slab * 512
                    db = db_base + slab * N * 8
                    for k in cutlass.range_constexpr(4):
                        mma(acc, da + 2 * k, db + 2 * k, N)
                p.wgmma_commit_group_sync_aligned()
                iket.range_pop()
                iket.range_push("mma_drain", stage)
                p.wgmma_wait_group_sync_aligned(0)
                iket.range_pop()
                if cutlass.const_expr(MODE == "output"):  # noqa: SIM102
                    if stage == 0 and rank == 3 and tid == 128:
                        # Scheduling hint only: the norm still polls every
                        # residual word. No payload ordering or release fence.
                        p.store_ext(epoch.ir_value(), epochs.iterator.raw_ptr() + 96 + cluster, order="volatile")
        p.barrier_cta_sync()

    iket.range_pop()


@cute.jit
def gate_up_index(row, col, N: cutlass.Constexpr):
    """Keep accumulator pairs adjacent and batch-row epilogue reads conflict-free."""
    rows_per_bank_group = 32 // N
    batch_swizzle = ((row // rows_per_bank_group) * 8) % N
    return row * N + (col ^ batch_swizzle)


@cute.jit
def scatter_partial(acc, partial, ready, phase: cutlass.Constexpr, tid, rank, N: cutlass.Constexpr):
    """Push partials into each owner's mailbox and await this CTA's inputs."""
    iket.range_push("splitk_store", cutlass.Int32(phase))
    if tid >= 128:
        for i in cutlass.range_constexpr(N // 2):
            row = fragment_row(tid - 128, i)
            col = fragment_column(tid - 128, i)
            offset = (rank * N + col) * 16 + row % 16
            cute.arch.store_async_dsmem(
                cute.make_ptr(cutlass.Float32, partial.data_ptr(offset).ir_value()),
                acc[i].bitcast(cutlass.Int32),
                cute.make_ptr(cutlass.Int64, ready.data_ptr(phase).ir_value()),
                peer_cta_rank=row // 16,
            )
    iket.range_pop()
    iket.range_push("splitk_receive", cutlass.Int32(phase))
    wait_barrier(ready.subview(phase), 0)
    iket.range_pop()


@cute.jit
def reduce_partial(partial, row, col, N: cutlass.Constexpr):
    """Sum the four sender slots in the original split-K order."""
    value = cutlass.Float32(0)
    for peer in cutlass.range_constexpr(4):
        value += partial.load(idx=(peer * N + col) * 16 + row % 16)
    return value


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
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    cluster, rank = bid // 4, bid % 4
    # Gate/up needs 128 KiB of weights but no staged operands; alias that unused space.
    weights = cutlass.Array(
        cutlass.BFloat16, max(65536, 49152 + 12 * N * 64), space=cutlass.AddressSpace.smem, alignment=1024
    )
    operands = weights.subview(49152)
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    # Weight loads need neither the epoch nor the DSMEM mailbox. Start both
    # stages before that setup; the epoch CTA barrier also joins this init.
    if bid < 64 and tid == 0:
        iket.range_push("output_weight_prefetch")
        p.prefetch_tensormap(wo.get_ptr())
        for stage in cutlass.range_constexpr(2):
            p.mbarrier_init(bars.subview(stage), 1)
        p.fence_mbarrier_init()
        for stage in cutlass.range_constexpr(2):
            p.mbarrier_expect_tx(bars.subview(stage), 32768)
            p.cp_async_bulk_tensor_shared_cluster_global(
                weights.subview(stage * 16384),
                wo.get_ptr(),
                [0, 0, rank * 8 + stage * 4, cluster],
                bars.subview(stage),
                [],
            )
        iket.range_pop()
    normalized = cutlass.Array(cutlass.BFloat16, N * 1024, space=cutlass.AddressSpace.smem, alignment=1024)
    partial = cutlass.Array(cutlass.Float32, 64 * N, space=cutlass.AddressSpace.smem, alignment=128)
    # A peer may start down while this CTA still reads its gate/up partials.
    # Keep the incoming mailbox separate from the gate/up epilogue.
    mailbox = cutlass.Array(cutlass.Float32, 64 * N, space=cutlass.AddressSpace.smem, alignment=128)
    epoch_s = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    if tid == 0:
        next_epoch = epochs[bid] + 1
        epoch_s.store(next_epoch)
        epochs[bid] = next_epoch
    p.barrier_cta_sync()
    epoch = epoch_s.load()
    ready = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    if cluster < 16:
        if tid == 0:
            for stage in cutlass.range_constexpr(2):
                p.mbarrier_init(ready.subview(stage), 1)
                p.mbarrier_arrive_expect_tx(ready.subview(stage), 4 * 16 * N * 4)
        p.fence_mbarrier_init()
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
        # Reset only the inactive residual slot for the next serialized call.
        for j in cutlass.range_constexpr(N // 16):
            idx = tid + j * 256
            batch, channel = idx // 16, idx % 16
            if channel % 2 == 0:
                offset = ((1 - (epoch & 1)) * N + batch) * 512 + bid * 8 + channel // 2
                p.store_ext(cutlass.Int32(PAYLOAD_EMPTY).ir_value(), r.iterator.raw_ptr() + offset, order="volatile")
    # Clear the inactive slot. The next serialized invocation uses it, so no
    # producer can race its reset and no extra reset kernel is needed.
    tile = bid
    for j in cutlass.range_constexpr(N // 16):
        offset = ((1 - (epoch & 1)) * 96 + tile) * N * 16 + tid + j * 256
        p.store_ext(cutlass.Int32(PAYLOAD_EMPTY).ir_value(), activation.iterator.raw_ptr() + offset, order="volatile")
    acc = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
    acc.fill(0)

    if cluster < 16:
        # Split K=2048 over the four ranks; every rank covers the same 64 outputs.
        gemm(
            acc,
            weights,
            operands,
            bars,
            wo,
            attention,
            normalized,
            activation,
            epochs,
            epoch,
            tid,
            cluster,
            rank,
            0,
            MODE="output",
            B=B,
            N=N,
        )
        scatter_partial(acc, mailbox, ready, 0, tid, rank, N)

        # Each rank owns 16 output channels after the cluster reduction.
        for j in cutlass.range_constexpr(N // 16):
            idx = tid + j * 256
            batch, channel = idx // 16, idx % 16
            row = rank * 16 + channel
            iket.range_push("output_reduce", j)
            projected = cutlass.BFloat16(reduce_partial(mailbox, row, batch, N))
            iket.range_pop()
            iket.range_push("output_residual", j)
            value = cutlass.BFloat16(0)
            if batch < B:
                value = cutlass.BFloat16(
                    cutlass.Float32(projected) + cutlass.Float32(residual[batch, cluster * 64 + row])
                )
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            # Neighboring lanes own adjacent channels; publish each pair in
            # one aligned word so consumers can test that word independently.
            neighbor = cute.arch.shuffle_sync_bfly(bits, 1)
            if channel % 2 == 0:
                packed = bits | (neighbor << 16)
                offset = ((epoch & 1) * N + batch) * 512 + bid * 8 + channel // 2
                p.store_ext(packed.ir_value(), r.iterator.raw_ptr() + offset, order="volatile")
            iket.range_pop()
        iket.range_push("output_cta_sync")
        p.barrier_cta_sync()
        iket.range_pop()

    # Every CTA produces one gate/up tile before the output CTAs start down.
    if cluster >= 16:
        iket.range_push("gate_up_prefetch_wait")
        if tid == 0:
            # Start prefetch once a producer finishes its first MMA
            # stage, rather than competing with its initial loads.
            ptr = epochs.iterator.raw_ptr() + 96 + (bid - 64) % 16
            while p.load_ext(ptr, dtype=cutlass.Int32, order="volatile") != epoch:
                p.nanosleep(32)
        # prefetch_gate_up's initialization barrier also joins this wait.
        iket.range_pop()
        if tid == 0:
            iket.range_push("gate_up_l2_prefetch")
            # These 32 CTAs warm the other 64 tiles for the CTAs still
            # doing output projection. Their own tiles go to SMEM below.
            for tile_offset in cutlass.range_constexpr(2):
                p.cp_async_bulk_tensor_prefetch(wgu.get_ptr(), [0, 0, 0, bid - 64 + tile_offset * 32], [])
            iket.range_pop()
    prefetch_gate_up(weights, bars, wgu, bid, tid)
    norm_inputs = cute.make_rmem_tensor((4, N // 2), cutlass.Int32)
    iket.range_push("norm_prefetch")
    norm_scales = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
    for group in cutlass.range_constexpr(4):
        values = (gamma.iterator.raw_ptr() + (tid % 32) * 8 + group * 256).load(count=8, alignment=16)
        for k in cutlass.range_constexpr(8):
            norm_scales[k, group] = values[k]
    for j in cutlass.range_constexpr(N // 2):
        values = residual_words(r, tid, j, epoch & 1, N)
        for k in cutlass.range_constexpr(4):
            norm_inputs[k, j] = values[k]
    iket.range_pop()
    iket.range_push("normalization_input_wait")
    norm_ready = cutlass.Boolean(True)
    for j in cutlass.range_constexpr(N // 2):
        norm_ready &= cutlass.all_(norm_inputs[None, j].load().to_vector() != PAYLOAD_EMPTY)
    while not norm_ready:
        for j in cutlass.range_constexpr(N // 2):
            if cutlass.any_(norm_inputs[None, j].load().to_vector() == PAYLOAD_EMPTY):
                values = residual_words(r, tid, j, epoch & 1, N)
                for k in cutlass.range_constexpr(4):
                    norm_inputs[k, j] = values[k]
        norm_ready = cutlass.Boolean(True)
        for j in cutlass.range_constexpr(N // 2):
            norm_ready &= cutlass.all_(norm_inputs[None, j].load().to_vector() != PAYLOAD_EMPTY)
    iket.range_pop()
    iket.range_push("norm_local_stats")
    # Retain polled inputs in registers. Each warp reduces its own rows,
    # with no shared partials or cross-warp inverse-RMS broadcast.
    norm_inverse = cute.make_rmem_tensor((N // 8,), cutlass.Float32)
    for token in cutlass.range_constexpr(N // 8):
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(4):
            words = norm_inputs[None, token * 4 + group].load().to_vector()
            values = words.bitcast(cutlass.BFloat16).to(cutlass.Float32)
            squares = values * values
            left = (squares[0] + squares[1]) + (squares[2] + squares[3])
            right = (squares[4] + squares[5]) + (squares[6] + squares[7])
            total += left + right
        for delta in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
        norm_inverse[token] = cute.math.rsqrt(total / 1024 + EPS)
    iket.range_pop()
    iket.range_push("norm_apply")
    for j in cutlass.range_constexpr(N // 2):
        row, col = tid // 32 + (j // 4) * 8, (tid % 32) * 8 + (j % 4) * 256
        values = norm_inputs[None, j].load().to_vector().bitcast(cutlass.BFloat16).to(cutlass.Float32)
        scale = norm_scales[None, j % 4].load().to_vector().to(cutlass.Float32)
        unit = (values * norm_inverse[j // 4]).to(cutlass.BFloat16).to(cutlass.Float32)
        offset = (col // 64) * N * 64 + swizzle(row, col % 64)
        normalized.store((unit * scale).to(cutlass.BFloat16), idx=offset, alignment=16)
    iket.range_pop()
    iket.range_push("norm_publish")
    # The only norm CTA rendezvous: all operand rows must reach SMEM
    # before the consumer warpgroup issues WGMMA.
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    iket.range_pop()
    acc.fill(0)
    gemm(
        acc,
        weights,
        operands,
        bars,
        wgu,
        attention,
        normalized,
        activation,
        epochs,
        epoch,
        tid,
        cluster,
        rank,
        tile,
        MODE="gate_up",
        B=B,
        N=N,
    )
    iket.range_push("gate_up_accumulator_store")
    if tid >= 128:
        for j in cutlass.range_constexpr(N // 2):
            row = fragment_row(tid - 128, j)
            col = fragment_column(tid - 128, j)
            partial.store(cutlass.Float32(cutlass.BFloat16(acc[j])), idx=gate_up_index(row, col, N))
    p.barrier_cta_sync()
    iket.range_pop()
    iket.range_push("silu_and_gate")
    for j in cutlass.range_constexpr(N // 16):
        idx = tid + j * 256
        channel, batch = 2 * (idx // N), idx % N
        # Publish a whole aligned word, not two independent 16-bit
        # stores. Canonicalize -0 so no live word equals the sentinel.
        packed = cutlass.Int32(0)
        for half in cutlass.range_constexpr(2):
            gate = partial.load(idx=gate_up_index(channel + half, batch, N))
            up = partial.load(idx=gate_up_index(channel + half + 32, batch, N))
            silu = cutlass.BFloat16(gate / (1 + cute.math.exp(-gate)))
            value = cutlass.BFloat16(cutlass.Float32(silu) * up)
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            packed |= bits << (16 * half)
        offset = (((epoch & 1) * 96 + tile) * N + batch) * 16 + channel // 2
        p.store_ext(packed.ir_value(), activation.iterator.raw_ptr() + offset, order="volatile")
    iket.range_pop()
    iket.range_push("activation_sync", 1)
    p.barrier_cta_sync()
    iket.range_pop()

    if cluster < 16:
        acc.fill(0)

        # Down waits only for its own activation tiles from the other CTAs.
        gemm(
            acc,
            weights,
            operands,
            bars,
            wd,
            attention,
            normalized,
            activation,
            epochs,
            epoch,
            tid,
            cluster,
            rank,
            0,
            MODE="down",
            B=B,
            N=N,
        )
        scatter_partial(acc, mailbox, ready, 1, tid, rank, N)
        for j in cutlass.range_constexpr(N // 16):
            idx = tid + j * 256
            row, col = rank * 16 + idx % 16, idx // 16
            iket.range_push("down_reduce", j)
            value = cutlass.BFloat16(reduce_partial(mailbox, row, col, N))
            iket.range_pop()
            iket.range_push("down_residual_store", j)
            if col < B:
                offset = ((epoch & 1) * N + col) * 512 + cluster * 32 + row // 2
                word = p.load_ext(r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, order="volatile")
                skip = cutlass.Uint16((word >> (16 * (row % 2))) & 0xFFFF).bitcast(cutlass.BFloat16)
                output[col, cluster * 64 + row] = cutlass.BFloat16(cutlass.Float32(value) + cutlass.Float32(skip))
            iket.range_pop()
        iket.range_push("down_cluster_release")
        # Every mailbox is consumed; only the cluster lifetime rendezvous remains.
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
        iket.range_pop()


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
    wo_map = tensor_map.create_tensor_map_tiled_from_view(wo, (1, 4, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b)
    wgu_map = tensor_map.create_tensor_map_tiled_from_view(
        wgu, (1, 16, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wd_map = tensor_map.create_tensor_map_tiled_from_view(wd, (1, 6, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b)
    post_attention_kernel(
        attention, residual, wo_map, wgu_map, wd_map, gamma, r, activation, epochs, output, B, N, EPS
    ).launch(
        grid=(96, 1, 1),
        block=(256, 1, 1),
        cluster=(4, 1, 1),
        cooperative=True,
        stream=stream,
    )
