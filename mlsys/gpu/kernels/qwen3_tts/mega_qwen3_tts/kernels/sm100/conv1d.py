"""Persistent BF16 causal Conv1D with tiled TMA and double-buffered accumulators."""

import cutlass
from cutlass import cute
from cutlass.experimental import cuda
from cutlass.experimental import primitives as p


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
    """Warps 0-3 store results, warp 4 issues MMA, warp 5 loads operands."""
    tid, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(tid // 32)
    work, _, _ = cute.arch.block_idx()
    stride, _, _ = cute.arch.grid_dim()
    MT = (L + 127) // 128
    NT: cutlass.Constexpr = (CO + N - 1) // N
    TILES = B * MT * NT
    CK: cutlass.Constexpr = (CI + BK - 1) // BK
    KT: cutlass.Constexpr = K * CK
    AB: cutlass.Constexpr = 128 * BK * 2
    BB: cutlass.Constexpr = N * BK * 2
    TMEM: cutlass.Constexpr = 1 << (2 * N - 1).bit_length()
    LOAD_N: cutlass.Constexpr = 64 if E == 64 else 32
    full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem)
    empty = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    a = cutlass.Array(cutlass.Int8, AB * STAGES, space=cutlass.AddressSpace.smem, alignment=1024)
    b = cutlass.Array(cutlass.Int8, BB * STAGES, space=cutlass.AddressSpace.smem, alignment=1024)
    c = cutlass.Array(cutlass.BFloat16, 2 * 128 * E, space=cutlass.AddressSpace.smem, alignment=1024)
    if tid == 0:
        p.prefetch_tensormap(input_map.get_ptr())
        p.prefetch_tensormap(weight_map.get_ptr())
        p.prefetch_tensormap(output_map.get_ptr())
        for i in cutlass.range_constexpr(STAGES):
            p.mbarrier_init(full.subview(i), 1)
            p.mbarrier_init(empty.subview(i), 1)
        for i in cutlass.range_constexpr(2):
            p.mbarrier_init(acc_full.subview(i), 1)
            p.mbarrier_init(acc_empty.subview(i), 4)
        p.fence_mbarrier_init()
    if warp == 0:
        p.tcgen05_alloc(tmem_address, TMEM)
        p.tcgen05_relinquish_alloc_permit()
    p.barrier_cta_sync(0, thread_count=192)
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    acc_stage = cutlass.Int32(0)
    acc_phase = cutlass.Int32(0)
    if warp == 5:
        if p.elect_sync():
            while work < TILES:
                batch = work // (MT * NT)
                row = work // NT % MT * 128
                col = work % NT * N
                for k in cutlass.range(KT, unroll=1):
                    while not p.mbarrier_try_wait_parity(empty.subview(stage), phase ^ 1):
                        pass
                    channel = k % CK * BK
                    tap = k // CK
                    p.mbarrier_arrive_expect_tx(full.subview(stage), AB + BB)
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        a.subview(stage * AB),
                        input_map.get_ptr(),
                        (channel, row + (tap - K + 1) * D, batch),
                        full.subview(stage),
                        [],
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        b.subview(stage * BB), weight_map.get_ptr(), (channel, tap, col), full.subview(stage), []
                    )
                    stage += 1
                    if stage == STAGES:
                        stage = 0
                        phase ^= 1
                work += stride
    elif warp == 4:
        p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
        desc_a = p.Tcgen05SmemDesc.build(
            a, leading_byte_offset=16, stride_byte_offset=BK * 16, layout=2 if BK == 64 else 4
        )
        desc_b = p.Tcgen05SmemDesc.build(
            b, leading_byte_offset=16, stride_byte_offset=BK * 16, layout=2 if BK == 64 else 4
        )
        instruction = p.Tcgen05InstrDesc.build(
            a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, c_dtype=cutlass.Float32, m_dim=128, n_dim=N
        )
        while work < TILES:
            while not p.mbarrier_try_wait_parity(acc_empty.subview(acc_stage), acc_phase ^ 1):
                pass
            p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
            destination = cutlass.Array(
                p.make_tmem_ptr(tmem_address.load(), cutlass.Float32).data_ptr() + acc_stage * N,
                dtype=cutlass.Int32,
                addrspace=6,
            )
            for k in cutlass.range(KT, unroll=1):
                while not p.mbarrier_try_wait_parity(full.subview(stage), phase):
                    pass
                p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
                for subk in cutlass.range_constexpr(BK // 16):
                    if p.elect_sync():
                        p.tcgen05_mma(
                            p.Tcgen05MMAKind.F16,
                            p.CTAGroup.CTA_1,
                            destination,
                            desc_a + stage * (AB // 16) + subk * 2,
                            desc_b + stage * (BB // 16) + subk * 2,
                            instruction,
                            (k > 0) | (subk > 0),
                        )
                if p.elect_sync():
                    p.tcgen05_commit(empty.subview(stage))
                stage += 1
                if stage == STAGES:
                    stage = 0
                    phase ^= 1
            if p.elect_sync():
                p.tcgen05_commit(acc_full.subview(acc_stage))
            acc_stage ^= 1
            if acc_stage == 0:
                acc_phase ^= 1
            work += stride
    else:
        address = tmem_address.load()
        store_stage = cutlass.Int32(0)
        issuer = p.elect_sync()
        while work < TILES:
            batch = work // (MT * NT)
            row = work // NT % MT * 128 + tid
            col = work % NT * N
            while not p.mbarrier_try_wait_parity(acc_full.subview(acc_stage), acc_phase):
                pass
            p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
            for chunk in cutlass.range_constexpr(N // E):
                for fragment in cutlass.range_constexpr(E // LOAD_N):
                    column = chunk * E + fragment * LOAD_N
                    pointer = p.make_tmem_ptr_from_warp_row_col(
                        address, warp, (address & 65535) + acc_stage * N + column, cutlass.Float32
                    )
                    values = p.tcgen05_ld("32x32b", pointer, num=LOAD_N)
                    p.tcgen05_wait(p.Tcgen05Wait.LOAD)
                    for v in cutlass.range_constexpr(LOAD_N // 8):
                        channel = col + column + v * 8
                        rounded = cutlass.full((8,), 0, cutlass.BFloat16)
                        if (FULL_ROWS or row < L) and channel < CO:
                            result = values[v * 8 : v * 8 + 8]
                            offset = (cutlass.Int64(batch) * L + row) * CO + channel
                            if cutlass.const_expr(bias is not None):
                                result = result.to(cutlass.BFloat16).to(cutlass.Float32) + (
                                    (bias.iterator.raw_ptr() + channel).load(count=8, alignment=16).to(cutlass.Float32)
                                )
                            rounded = result.to(cutlass.BFloat16)
                            if cutlass.const_expr(residual is not None):
                                rounded += (residual.iterator.raw_ptr() + offset).load(count=8, alignment=16)
                            if cutlass.const_expr(alpha is not None):
                                scale = (alpha.iterator.raw_ptr() + channel).load(count=8, alignment=16)
                                inverse = (inverse_beta.iterator.raw_ptr() + channel).load(count=8, alignment=16)
                                scaled = rounded * scale
                                indices = scaled.bitcast(cutlass.Uint16).to(cutlass.Int32)
                                sine = sine_table.iterator.raw_ptr().masked_gather(
                                    indices, cutlass.full((8,), True, cutlass.Boolean)
                                )
                                square = sine * sine
                                rounded += square * inverse
                        c.subview(
                            store_stage * 128 * E + tid * E + fragment * LOAD_N + v * 8
                        ).data_ptr().store_swizzled(
                            rounded, alignment=16, swizzle=cutlass.Swizzle(3 if E == 64 else 2, 4, 3)
                        )
                cute.arch.fence_view_async_shared()
                p.barrier_cta_sync(2, thread_count=128)
                if warp == 0 and issuer:
                    if cutlass.const_expr(E == 96):
                        coordinates = (0, (col + chunk * E) // 32, work // NT % MT * 128, batch)
                    else:
                        coordinates = (col + chunk * E, work // NT % MT * 128, batch)
                    p.cp_async_bulk_tensor_global_shared_cta(
                        output_map.get_ptr(),
                        c.subview(store_stage * 128 * E),
                        coordinates,
                    )
                    p.cp_async_bulk_commit_group()
                    p.cp_async_bulk_wait_group(1, read=True)
                p.barrier_cta_sync(2, thread_count=128)
                store_stage ^= 1
            p.tcgen05_fence(p.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if p.elect_sync():
                p.mbarrier_arrive(acc_empty.subview(acc_stage))
            acc_stage ^= 1
            if acc_stage == 0:
                acc_phase ^= 1
            work += stride
        if warp == 0 and issuer:
            p.cp_async_bulk_wait_group(0, read=False)
    # All producer/consumer work must finish before freeing the shared TMEM allocation.
    p.barrier_cta_sync(0, thread_count=192)
    if warp == 0:
        p.tcgen05_dealloc(p.make_tmem_ptr(tmem_address.load(), cutlass.Float32), TMEM)


class Conv1dLaunch:
    """Bind tiled tensor maps without im2col APIs or driver-side descriptor shims."""

    def __init__(self, weight_shape, dilation, tile_n, stages, tile_k, store_n, full_rows):
        self.weight_shape = weight_shape
        self.full_rows = full_rows
        self.dilation = dilation
        self.tile_n = tile_n
        self.stages = stages
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
            box_dims=(tile_k, 128, 1),
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
            box_dims=(32, 3, 128, 1) if store_n == 96 else (store_n, 128, 1),
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
        ).launch(grid=(ctas, 1, 1), block=(192, 1, 1), stream=stream)
