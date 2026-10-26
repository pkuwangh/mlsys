"""Hopper causal Conv1D with two alternating WGMMA consumer warpgroups."""

import cutlass
from cutlass import cute
from cutlass.experimental import cuda
from cutlass.experimental import primitives as p

from .mma import fragment_column, fragment_row, mma


@cute.kernel
def conv1d_kernel(
    operation: cutlass.Constexpr,
    input_map: cutlass.GridConstant[cuda.TensorMap],
    weight_map: cutlass.GridConstant[cuda.TensorMap],
    output_map: cutlass.GridConstant[cuda.TensorMap],
    bias: cute.Tensor,
    residual: cute.Tensor,
    alpha: cute.Tensor,
    inverse_beta: cute.Tensor,
    sine_table: cute.Tensor,
    B: cutlass.Int32,
    L: cutlass.Int32,
    CI: cutlass.Constexpr,
    CO: cutlass.Constexpr,
    K: cutlass.Constexpr,
    D: cutlass.Constexpr,
    N: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    BK: cutlass.Constexpr,
    E: cutlass.Constexpr,
    FULL_ROWS: cutlass.Constexpr,
):
    """Warps 0-3/4-7 alternate output tiles; warp 8 stages operands for both."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    group = cute.arch.make_warp_uniform(tid // 128)
    local_tid = tid % 128
    MT = (L + 63) // 64
    NT: cutlass.Constexpr = (CO + N - 1) // N
    TILES = B * MT * NT
    CK: cutlass.Constexpr = (CI + BK - 1) // BK
    KT: cutlass.Constexpr = K * CK
    AB: cutlass.Constexpr = 64 * BK * 2
    BB: cutlass.Constexpr = N * BK * 2
    full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem)
    empty = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem)
    mma_done = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    a = cutlass.Array(cutlass.Int8, AB * STAGES, space=cutlass.AddressSpace.smem, alignment=1024)
    b = cutlass.Array(cutlass.Int8, BB * STAGES, space=cutlass.AddressSpace.smem, alignment=1024)
    output = cutlass.Array(cutlass.BFloat16, 2 * 2 * 64 * E, space=cutlass.AddressSpace.smem, alignment=1024)
    if tid == 0:
        p.prefetch_tensormap(input_map.get_ptr())
        p.prefetch_tensormap(weight_map.get_ptr())
        p.prefetch_tensormap(output_map.get_ptr())
        for stage in cutlass.range_constexpr(STAGES):
            p.mbarrier_init(full.subview(stage), 1)
            p.mbarrier_init(empty.subview(stage), 1)
        for index in cutlass.range_constexpr(2):
            p.mbarrier_init(mma_done.subview(index), 1)
        p.fence_mbarrier_init()
    p.barrier_cta_sync(0, thread_count=288)
    if group == 2:
        if p.elect_sync():
            cursor = cutlass.Int32(0)
            work = block
            while work < TILES:
                batch = work // (MT * NT)
                row = work // NT % MT * 64
                col = work % NT * N
                for k in cutlass.range(KT, unroll=1):
                    stage = cursor % STAGES
                    while not p.mbarrier_try_wait_parity(empty.subview(stage), (cursor // STAGES % 2) ^ 1):
                        pass
                    p.mbarrier_arrive_expect_tx(full.subview(stage), AB + BB)
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        a.subview(stage * AB),
                        input_map.get_ptr(),
                        (k % CK * BK, row + (k // CK - K + 1) * D, batch),
                        full.subview(stage),
                        [],
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        b.subview(stage * BB),
                        weight_map.get_ptr(),
                        (k % CK * BK, k // CK, col),
                        full.subview(stage),
                        [],
                    )
                    cursor += 1
                work += grid
    else:
        cursor = group * KT
        phase = group ^ 1
        store_stage = cutlass.Int32(0)
        work = block + group * grid
        # K-major swizzle atoms contain eight rows of BK BF16 elements.
        desc_bits: cutlass.Constexpr = (1 << 16) | (BK << 32) | ((1 if BK == 64 else 2) << 62)
        da = (cutlass.Int64(a.data_ptr().toint()) >> 4) | desc_bits
        db = (cutlass.Int64(b.data_ptr().toint()) >> 4) | desc_bits
        acc = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
        while work < TILES:
            # Consume tiles in order: a future consumer must not mistake an
            # unfinished earlier ring-buffer generation for its own ready data.
            # The other warpgroup's epilogue runs concurrently with this MMA.
            while not p.mbarrier_try_wait_parity(mma_done.subview(1 - group), phase):
                pass
            acc.fill(0)
            p.wgmma_fence_aligned()
            for k in cutlass.range(KT, unroll=1):
                stage = cursor % STAGES
                while not p.mbarrier_try_wait_parity(full.subview(stage), cursor // STAGES % 2):
                    pass
                for subk in cutlass.range_constexpr(BK // 16):
                    mma(acc, da + stage * (AB // 16) + subk * 2, db + stage * (BB // 16) + subk * 2, N)
                p.wgmma_commit_group_sync_aligned()
                p.wgmma_wait_group_sync_aligned(1)
                if k > 0 and local_tid == 0:
                    p.mbarrier_arrive(empty.subview((cursor - 1) % STAGES))
                cursor += 1
            p.wgmma_wait_group_sync_aligned(0)
            if local_tid == 0:
                p.mbarrier_arrive(empty.subview((cursor - 1) % STAGES))
                p.mbarrier_arrive(mma_done.subview(group))
            batch = work // (MT * NT)
            row_base = work // NT % MT * 64
            col_base = work % NT * N
            for sub in cutlass.range_constexpr(N // E):
                for pair in cutlass.range_constexpr(E // 4):
                    index = sub * (E // 2) + pair * 2
                    row_local = fragment_row(local_tid, index)
                    column = fragment_column(local_tid, index)
                    row, col = row_base + row_local, col_base + column
                    rounded = cutlass.full((2,), 0, cutlass.BFloat16)
                    if (FULL_ROWS or row < L) and col < CO:
                        values = cute.make_rmem_tensor((2,), cutlass.Float32)
                        values[0], values[1] = acc[index], acc[index + 1]
                        rounded = values.load().to_vector().to(cutlass.BFloat16)
                        if cutlass.const_expr(bias is not None):
                            rounded += (bias.iterator.raw_ptr() + col).load(count=2, alignment=4)
                        if cutlass.const_expr(residual is not None):
                            offset = (cutlass.Int64(batch) * L + row) * CO + col
                            rounded += (residual.iterator.raw_ptr() + offset).load(count=2, alignment=4)
                        if cutlass.const_expr(alpha is not None):
                            scale = (alpha.iterator.raw_ptr() + col).load(count=2, alignment=4)
                            inverse = (inverse_beta.iterator.raw_ptr() + col).load(count=2, alignment=4)
                            scaled = rounded * scale
                            sine = sine_table.iterator.raw_ptr().masked_gather(
                                scaled.bitcast(cutlass.Uint16).to(cutlass.Int32),
                                cutlass.full((2,), True, cutlass.Boolean),
                            )
                            square = sine * sine
                            rounded += square * inverse
                    output.subview(
                        group * 128 * E + store_stage * 64 * E + row_local * E + column % E
                    ).data_ptr().store_swizzled(
                        rounded, alignment=4, swizzle=cutlass.Swizzle(3 if E == 64 else 2, 4, 3)
                    )
                cute.arch.fence_view_async_shared()
                p.barrier_cta_sync(group + 1, thread_count=128)
                if local_tid == 0:
                    if cutlass.const_expr(E == 96):
                        coordinates = (0, (col_base + sub * E) // 32, row_base, batch)
                    else:
                        coordinates = (col_base + sub * E, row_base, batch)
                    p.cp_async_bulk_tensor_global_shared_cta(
                        output_map.get_ptr(),
                        output.subview(group * 128 * E + store_stage * 64 * E),
                        coordinates,
                    )
                    p.cp_async_bulk_commit_group()
                    p.cp_async_bulk_wait_group(1, read=True)
                p.barrier_cta_sync(group + 1, thread_count=128)
                store_stage ^= 1
            cursor += KT
            phase ^= 1
            work += 2 * grid
        if local_tid == 0:
            p.cp_async_bulk_wait_group(0, read=False)
    p.barrier_cta_sync(0, thread_count=288)


class Conv1dLaunch:
    """Bind tiled TMA maps for M64 convolution tiles."""

    def __init__(self, weight_shape, dilation, tile_n, stages, tile_k, store_n, full_rows):
        self.weight_shape = weight_shape
        self.full_rows = full_rows
        self.dilation, self.tile_n, self.stages = dilation, tile_n, stages
        self.tile_k, self.store_n = tile_k, store_n

    @cute.jit
    def __call__(self, x, weight, output, bias, residual, alpha, inverse_beta, sine_table, stream, ctas: cutlass.Int32):
        batch, length = cutlass.Int32(x.shape[0]), cutlass.Int32(x.shape[1])
        ci = self.weight_shape[2]
        tile_k, store_n = self.tile_k, self.store_n
        co, kernel, _ = self.weight_shape
        a = cuda.create_tensor_map_tiled(
            global_address=x.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(ci, length, batch),
            global_strides=(ci // 8, length * ci // 8),
            box_dims=(tile_k, 64, 1),
            swizzle=cuda.TensorMapSwizzle.s128b if tile_k == 64 else cuda.TensorMapSwizzle.s64b,
        )
        b = cuda.create_tensor_map_tiled(
            global_address=weight.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(ci, kernel, co),
            global_strides=(ci // 8, kernel * ci // 8),
            box_dims=(tile_k, 1, self.tile_n),
            swizzle=cuda.TensorMapSwizzle.s128b if tile_k == 64 else cuda.TensorMapSwizzle.s64b,
        )
        # Group three 32-channel blocks to store N96 with a 64-byte swizzle.
        c = cuda.create_tensor_map_tiled(
            global_address=output.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(32, co // 32, length, batch) if store_n == 96 else (co, length, batch),
            global_strides=(4, co // 8, length * co // 8) if store_n == 96 else (co // 8, length * co // 8),
            box_dims=(32, 3, 64, 1) if store_n == 96 else (store_n, 64, 1),
            swizzle=cuda.TensorMapSwizzle.s128b if store_n == 64 else cuda.TensorMapSwizzle.s64b,
        )
        conv1d_kernel(
            f"c{ci}_c{co}_k{kernel}_d{self.dilation}_"
            f"{'residual' if residual is not None else 'bias'}_{'snake' if alpha is not None else 'raw'}",
            a,
            b,
            c,
            bias,
            residual,
            alpha,
            inverse_beta,
            sine_table,
            batch,
            length,
            ci,
            co,
            kernel,
            self.dilation,
            self.tile_n,
            self.stages,
            tile_k,
            store_n,
            self.full_rows,
        ).launch(grid=(ctas, 1, 1), block=(288, 1, 1), stream=stream)
