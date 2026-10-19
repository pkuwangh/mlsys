"""Fused output projection, residual, RMSNorm, SwiGLU, and down projection.

Hopper, BF16, hidden=1024, FFN=3072, attention width=2048, B<=32.
B<=16 reuses 96 CTAs across the three projections. Larger batches dedicate
64 CTAs to output/down and 48 to gate/up. DSMEM reduces split-K partials;
GPU release/acquire flags publish residuals and activated gate/up tiles.

Run this file to validate and benchmark the complete fused operation.
"""

import argparse
import sys
from functools import cache
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._hopper import descriptor, fragment_column, fragment_row, mma, swizzle
from mega_qwen3_tts.kernels._utils import measure, report_error


@cute.jit
def wait_flag(flags, index, epoch):
    while p.load_ext(flags.subview(index), dtype=cutlass.Int32, order="acquire", scope="gpu") != epoch:
        p.nanosleep(32)


@cute.jit
def publish(flags, index, epoch):
    p.store_ext(epoch.ir_value(), flags.subview(index), order="release", scope="gpu")


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
def gemm(
    acc,
    weights,
    operands,
    bars,
    weight_map,
    attention,
    normalized,
    activation,
    flags,
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
    if cutlass.const_expr(MODE == 0):
        iket.range_push("output_projection")
    elif cutlass.const_expr(MODE == 1):
        iket.range_push("gate_up_projection", tile)
    else:
        iket.range_push("down_projection")
    if cutlass.const_expr(MODE == 1):
        if tid >= 128:
            p.wgmma_fence_aligned()
            iket.range_push("operand_wait", cutlass.Int32(0))
            wait_barrier(bars, 0)
            iket.range_pop()
            iket.range_push("mma_issue", cutlass.Int32(0))
            for slab in cutlass.range_constexpr(16):
                da = descriptor(weights, slab * 4096)
                db = descriptor(normalized, slab * N * 64)
                for k in cutlass.range_constexpr(4):
                    mma(acc, da + 2 * k, db + 2 * k, N)
            p.wgmma_commit_group_sync_aligned()
            iket.range_pop()
            iket.range_push("mma_drain", cutlass.Int32(0))
            p.wgmma_wait_group_sync_aligned(0)
            iket.range_pop()
        p.barrier_cta_sync()
    else:
        if tid == 0:
            for stage in cutlass.range_constexpr(2):
                p.mbarrier_init(bars.subview(stage), 1)
        p.fence_mbarrier_init()
        p.barrier_cta_sync()
        group = 6 if MODE == 2 else 4
        if tid < 128:
            for stage in range(2):
                if cutlass.const_expr(MODE == 0):
                    kbase = rank * 512 + stage * 256
                else:
                    kbase = stage * 1536 + rank * 384
                if tid == 0:
                    p.mbarrier_expect_tx(bars.subview(stage), group * 8192)
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        weights.subview(stage * group * 4096),
                        weight_map.get_ptr(),
                        [0, 0, kbase // 64, cluster],
                        bars.subview(stage),
                        [],
                    )
                iket.range_push("activation_wait", stage)
                if cutlass.const_expr(MODE == 2):  # noqa: SIM102 - keep the compile-time branch.
                    if tid < 12:
                        wait_flag(flags, 64 + kbase // 32 + tid, epoch)
                p.barrier_cta_sync(barrier_id=1, thread_count=128)
                iket.range_pop()
                iket.range_push("operand_copy", stage)
                for j in cutlass.range_constexpr(group * N // 16):
                    idx = (tid + j * 128) * 8
                    slab, row, k = idx // (N * 64), (idx // 64) % N, idx % 64
                    if cutlass.const_expr(MODE == 0):
                        value = cutlass.full((8,), 0, cutlass.BFloat16)
                        if row < B:
                            value = (attention.iterator.raw_ptr() + row * 2048 + kbase + slab * 64 + k).load(
                                count=8, alignment=16
                            )
                    else:
                        first = kbase // 32
                        offset = ((first + slab * 2 + k // 32) * N + row) * 32 + k % 32
                        value = (activation.iterator.raw_ptr() + offset).load(count=8, alignment=16)
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
                for slab in cutlass.range_constexpr(group):
                    da = descriptor(weights, (stage * group + slab) * 4096)
                    db = descriptor(operands, (stage * group + slab) * N * 64)
                    for k in cutlass.range_constexpr(4):
                        mma(acc, da + 2 * k, db + 2 * k, N)
                p.wgmma_commit_group_sync_aligned()
                iket.range_pop()
                iket.range_push("mma_drain", stage)
                p.wgmma_wait_group_sync_aligned(0)
                iket.range_pop()
        p.barrier_cta_sync()

    iket.range_pop()


@cute.jit
def partial_index(row, col):
    """Swizzle WGMMA stores and 16-channel DSMEM reduction reads across banks."""
    row_swizzle = ((col // 2) * 8 + (col % 2) * 16) % 64
    return col * 64 + (row ^ row_swizzle)


@cute.jit
def gate_up_index(row, col, N: cutlass.Constexpr):
    """Keep accumulator pairs adjacent and batch-row epilogue reads conflict-free."""
    rows_per_bank_group = 32 // N
    batch_swizzle = ((row // rows_per_bank_group) * 8) % N
    return row * N + (col ^ batch_swizzle)


@cute.jit
def scatter_partial(acc, partial, tid, N: cutlass.Constexpr):
    iket.range_push("splitk_store")
    if tid >= 128:
        for i in cutlass.range_constexpr(N // 2):
            row = fragment_row(tid - 128, i)
            col = fragment_column(tid - 128, i)
            partial.store(acc[i], idx=partial_index(row, col))
    iket.range_pop()
    iket.range_push("splitk_cluster_sync")
    p.barrier_cluster_arrive()
    p.barrier_cluster_wait()
    iket.range_pop()


@cute.jit
def reduce_partial(partial, row, col, N: cutlass.Constexpr):
    value = cutlass.Float32(0)
    for peer in cutlass.range_constexpr(4):
        remote = p.mapa(partial.subview(partial_index(row, col)), peer)
        value += remote.load()
    return value


@cute.kernel
def fused_kernel(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cutlass.GridConstant[tensor_map.TensorMap],
    wgu: cutlass.GridConstant[tensor_map.TensorMap],
    wd: cutlass.GridConstant[tensor_map.TensorMap],
    gamma: cute.Tensor,
    r: cute.Tensor,
    stats: cute.Tensor,
    activation: cute.Tensor,
    flags_tensor: cute.Tensor,
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
    normalized = cutlass.Array(cutlass.BFloat16, N * 1024, space=cutlass.AddressSpace.smem, alignment=1024)
    partial = cutlass.Array(cutlass.Float32, 64 * N, space=cutlass.AddressSpace.smem, alignment=128)
    inverse = cutlass.Array(cutlass.Float32, N, space=cutlass.AddressSpace.smem, alignment=128)
    epoch_s = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    flags = cutlass.make_array_view(flags_tensor)
    if tid == 0:
        epoch_s.store(epochs[bid] + 1)
        epochs[bid] = epochs[bid] + 1
    p.barrier_cta_sync()
    epoch = epoch_s.load()
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
            flags,
            epoch,
            tid,
            cluster,
            rank,
            0,
            0,
            B,
            N,
        )
        scatter_partial(acc, partial, tid, N)

        # Each rank owns 16 output channels after the cluster reduction.
        for j in cutlass.range_constexpr(N // 16):
            idx = tid + j * 256
            batch, channel = idx // 16, idx % 16
            row = rank * 16 + channel
            iket.range_push("output_reduce", j)
            projected = cutlass.BFloat16(reduce_partial(partial, row, batch, N))
            iket.range_pop()
            iket.range_push("output_residual_stats", j)
            value = cutlass.BFloat16(0)
            if batch < B:
                value = cutlass.BFloat16(
                    cutlass.Float32(projected) + cutlass.Float32(residual[batch, cluster * 64 + row])
                )
            r[batch, cluster * 64 + row] = value
            square_sum = cutlass.Float32(value) * cutlass.Float32(value)
            for delta in cutlass.range_constexpr(4):
                square_sum += cute.arch.shuffle_sync_bfly(square_sum, 1 << delta)
            if channel == 0:
                stats[bid, batch] = square_sum
            iket.range_pop()
        iket.range_push("output_cta_sync")
        p.barrier_cta_sync()
        iket.range_pop()
        iket.range_push("output_release")
        if tid == 0:
            publish(flags, bid, epoch)
        iket.range_pop()
        iket.range_push("output_cluster_release")
        p.barrier_cluster_arrive()
        p.barrier_cluster_wait()
        iket.range_pop()

    # At N16 all 96 CTAs produce one tile; at N32 only the 48 dedicated CTAs do so twice.
    if N == 16 or cluster >= 16:
        prefetch_gate_up(weights, bars, wgu, bid if N == 16 else bid - 64, tid)
        iket.range_push("normalization_input_wait")
        if tid < 64:
            wait_flag(flags, tid, epoch)
        p.barrier_cta_sync()
        iket.range_pop()
        # At N16, overlap residual/gamma loads with the statistics reduction.
        if cutlass.const_expr(N == 16):
            norm_inputs = cute.make_rmem_tensor((8, N // 2), cutlass.BFloat16)
            norm_scales = cute.make_rmem_tensor((8, N // 2), cutlass.BFloat16)
            iket.range_push("norm_prefetch")
            for j in cutlass.range_constexpr(N // 2):
                idx = (tid + j * 256) * 8
                col = idx % 1024
                values = (r.iterator.raw_ptr() + idx).load(count=8, alignment=16)
                scales = (gamma.iterator.raw_ptr() + col).load(count=8, alignment=16)
                for k in cutlass.range_constexpr(8):
                    norm_inputs[k, j] = values[k]
                    norm_scales[k, j] = scales[k]
            iket.range_pop()
        iket.range_push("norm_stats_reduce")
        if tid < N * 8:
            batch, part = tid // 8, tid % 8
            total = cutlass.Float32(0)
            for j in cutlass.range_constexpr(8):
                total += stats[j * 8 + part, batch]
            for delta in cutlass.range_constexpr(3):
                total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
            if part == 0:
                inverse.store(cute.math.rsqrt(total / 1024 + EPS), idx=batch)
        p.barrier_cta_sync()
        iket.range_pop()
        iket.range_push("norm_apply")
        for j in cutlass.range_constexpr(N // 2):
            idx = (tid + j * 256) * 8
            row, col = idx // 1024, idx % 1024
            if cutlass.const_expr(N == 16):
                norm_value = norm_inputs[None, j].load().to(cutlass.Float32)
                scale = norm_scales[None, j].load().to(cutlass.Float32)
            else:
                norm_value = (r.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
                scale = (gamma.iterator.raw_ptr() + col).load(count=8, alignment=16).to(cutlass.Float32)
            unit = (norm_value * inverse.load(idx=row)).to(cutlass.BFloat16).to(cutlass.Float32)
            # Gate/up reads this resident WGMMA layout directly.
            offset = (col // 64) * N * 64 + swizzle(row, col % 64)
            normalized.store((unit * scale).to(cutlass.BFloat16), idx=offset, alignment=16)
        p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
        p.barrier_cta_sync()
        iket.range_pop()
        for wave in range(1 if N == 16 else 2):
            tile = bid if N == 16 else wave * 48 + bid - 64
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
                flags,
                epoch,
                tid,
                cluster,
                rank,
                tile,
                1,
                B,
                N,
            )
            if cutlass.const_expr(N == 32):  # noqa: SIM102 - keep the compile-time branch.
                if wave == 0:
                    # The accumulator no longer needs these weights; overlap the next load with SiLU.
                    prefetch_gate_up(weights, bars, wgu, tile + 48, tid)
            iket.range_push("gate_up_accumulator_store")
            if tid >= 128:
                for j in cutlass.range_constexpr(N // 2):
                    row = fragment_row(tid - 128, j)
                    col = fragment_column(tid - 128, j)
                    partial.store(cutlass.Float32(cutlass.BFloat16(acc[j])), idx=gate_up_index(row, col, N))
            p.barrier_cta_sync()
            iket.range_pop()
            iket.range_push("silu_and_gate")
            for j in cutlass.range_constexpr(N // 8):
                idx = tid + j * 256
                channel, batch = idx // N, idx % N
                gate = partial.load(idx=gate_up_index(channel, batch, N))
                up = partial.load(idx=gate_up_index(channel + 32, batch, N))
                silu = cutlass.BFloat16(gate / (1 + cute.math.exp(-gate)))
                activation[tile, batch, channel] = cutlass.BFloat16(cutlass.Float32(silu) * up)
            iket.range_pop()
            iket.range_push("activation_sync", 0)
            p.barrier_cta_sync()
            iket.range_pop()
            iket.range_push("activation_release")
            if tid == 0:
                publish(flags, 64 + tile, epoch)
            iket.range_pop()
            iket.range_push("activation_sync", 1)
            p.barrier_cta_sync()
            iket.range_pop()

    if cluster < 16:
        acc.fill(0)

        # Down waits only for its own activation tiles, overlapping gate/up at N32.
        gemm(
            acc,
            weights,
            operands,
            bars,
            wd,
            attention,
            normalized,
            activation,
            flags,
            epoch,
            tid,
            cluster,
            rank,
            0,
            2,
            B,
            N,
        )
        scatter_partial(acc, partial, tid, N)
        for j in cutlass.range_constexpr(N // 16):
            idx = tid + j * 256
            row, col = rank * 16 + idx % 16, idx // 16
            iket.range_push("down_reduce", j)
            value = cutlass.BFloat16(reduce_partial(partial, row, col, N))
            iket.range_pop()
            iket.range_push("down_residual_store", j)
            if col < B:
                output[col, cluster * 64 + row] = cutlass.BFloat16(
                    cutlass.Float32(value) + cutlass.Float32(r[col, cluster * 64 + row])
                )
            iket.range_pop()
        iket.range_push("down_cluster_release")
        p.barrier_cluster_arrive()
        p.barrier_cluster_wait()
        iket.range_pop()


@cute.jit
def fused_host(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cute.Tensor,
    wgu: cute.Tensor,
    wd: cute.Tensor,
    gamma: cute.Tensor,
    r: cute.Tensor,
    stats: cute.Tensor,
    activation: cute.Tensor,
    flags: cute.Tensor,
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
    fused_kernel(
        attention, residual, wo_map, wgu_map, wd_map, gamma, r, stats, activation, flags, epochs, output, B, N, EPS
    ).launch(
        grid=(96 if N == 16 else 112, 1, 1),
        block=(256, 1, 1),
        cluster=(4, 1, 1),
        cooperative=True,
        stream=stream,
    )


@cache
def compile_kernel(batch, bucket, eps):
    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        fused_host,
        fake(cutlass.BFloat16, batch, 2048),
        fake(cutlass.BFloat16, batch, 1024),
        fake(cutlass.BFloat16, 16, 32, 64, 64),
        fake(cutlass.BFloat16, 96, 16, 64, 64),
        fake(cutlass.BFloat16, 16, 48, 64, 64),
        fake(cutlass.BFloat16, 1024),
        fake(cutlass.BFloat16, bucket, 1024),
        fake(cutlass.Float32, 64, bucket),
        fake(cutlass.BFloat16, 96, bucket, 32),
        fake(cutlass.Int32, 160),
        fake(cutlass.Int32, 96 if bucket == 16 else 112),
        fake(cutlass.BFloat16, batch, 1024),
        make_fake_stream(),
        batch,
        bucket,
        eps,
        options="--enable-tvm-ffi --gpu-arch=sm_90a",
    )


class FusedPostAttention(nn.Module):
    """Inference-only replacement; each instance owns a serialized workspace."""

    def __init__(self, reference):
        super().__init__()
        if tuple(reference.o_proj.weight.shape) != (1024, 2048) or tuple(reference.gate_proj.weight.shape) != (
            3072,
            1024,
        ):
            raise ValueError("The fused block supports the 0.6B talker's 1024/2048/3072 dimensions")
        if reference.o_proj.weight.dtype != torch.bfloat16 or not reference.o_proj.weight.is_cuda:
            raise ValueError("The fused block requires CUDA BF16 weights")
        if torch.cuda.get_device_capability(reference.o_proj.weight.device)[0] != 9:
            raise ValueError("This WGMMA implementation targets Hopper")
        self.reference = reference
        packed = (
            torch.cat(
                (reference.gate_proj.weight.reshape(96, 32, 1024), reference.up_proj.weight.reshape(96, 32, 1024)),
                dim=1,
            )
            .detach()
            .contiguous()
        )
        self.register_buffer("packed_gate_up", packed.reshape(96, 64, 16, 64).permute(0, 2, 1, 3).contiguous())
        self.register_buffer(
            "packed_o", reference.o_proj.weight.detach().reshape(16, 64, 32, 64).permute(0, 2, 1, 3).contiguous()
        )
        self.register_buffer(
            "packed_down", reference.down_proj.weight.detach().reshape(16, 64, 48, 64).permute(0, 2, 1, 3).contiguous()
        )
        self.workspaces = {}

    @torch.compiler.disable
    def forward(self, attention, residual):
        # Prefill stays in Torch; only one-token decoding uses this schedule.
        if attention.ndim == 3 and attention.shape[1] != 1:
            return self.reference(attention, residual)
        batch = attention.shape[0]
        if not 1 <= batch <= 32:
            raise ValueError("Fused decode supports 1 <= B <= 32")
        # N=16 minimum here, so the reduction epilogue has >=256 elements/rank.
        bucket = 16 if batch <= 16 else 32
        if batch not in self.workspaces:
            device = attention.device
            self.workspaces[batch] = (
                torch.empty(bucket, 1024, device=device, dtype=torch.bfloat16),
                torch.empty(64, bucket, device=device, dtype=torch.float32),
                torch.empty(96, bucket, 32, device=device, dtype=torch.bfloat16),
                torch.zeros(160, device=device, dtype=torch.int32),
                torch.zeros(96 if bucket == 16 else 112, device=device, dtype=torch.int32),
            )
        output = torch.empty_like(residual, memory_format=torch.contiguous_format)
        fn = compile_kernel(batch, bucket, self.reference.norm.eps)
        fn(
            attention.reshape(batch, 2048),
            residual.reshape(batch, 1024),
            self.packed_o,
            self.packed_gate_up,
            self.packed_down,
            self.reference.norm.weight,
            *self.workspaces[batch],
            output.reshape(batch, 1024),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return output


@torch.inference_mode()
def trace_once(batch):
    """Launch one fused block without reference GEMMs or benchmark replays."""
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention

    torch.manual_seed(42)
    reference = TorchPostAttention(TalkerConfig()).to(device="cuda", dtype=torch.bfloat16).eval()
    fused = FusedPostAttention(reference)
    attention = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    fused(attention, residual)
    torch.cuda.synchronize()
    print(f"B={batch}: one fused block completed")


@torch.inference_mode()
def benchmark(batch, layers, profile=False):
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention, post_attention

    torch.manual_seed(42)
    references = [
        TorchPostAttention(TalkerConfig()).to(device="cuda", dtype=torch.bfloat16).eval() for _ in range(layers)
    ]
    for reference in references:
        reference.norm.weight.uniform_(0.5, 1.5)
    customs = [FusedPostAttention(reference) for reference in references]
    compiled = torch.compile(post_attention, fullgraph=True, options={"emulate_precision_casts": True})
    weights = [
        (r.o_proj.weight, r.norm.weight, r.gate_proj.weight, r.up_proj.weight, r.down_proj.weight, r.norm.eps)
        for r in references
    ]
    attention = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        attention.normal_()
        residual.normal_()
        expected = torch.stack([reference(attention, residual) for reference in references])
        report_error(
            f"B={batch} custom seed={seed}", torch.stack([custom(attention, residual) for custom in customs]), expected
        )
        report_error(
            f"B={batch} compiled seed={seed}",
            torch.stack([compiled(attention, residual, *w) for w in weights]),
            expected,
        )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        result = customs[0](attention, residual)
    for zero in (True, False):
        if zero:
            attention.zero_()
            residual.zero_()
        else:
            attention.normal_()
            residual.normal_()
        graph.replay()
        report_error(f"B={batch} graph {'zero' if zero else 'new input'}", result, references[0](attention, residual))
    timings = measure(
        {
            "Torch": lambda: [r(attention, residual) for r in references],
            "Compiled": lambda: [compiled(attention, residual, *w) for w in weights],
            "Fused": lambda: [c(attention, residual) for c in customs],
        },
        profile=profile,
    )
    return {name: us / layers for name, us in timings.items()}


def print_summary(results, layers):
    """Print per-block latency and baseline/fused speedups for every batch."""
    headers = ["Batch", "Fused (us)", "Torch (us)", "Compiled (us)", "vs Torch", "vs Compiled"]
    rows = [
        [
            str(batch),
            *[f"{times[name]:.3f}" for name in ("Fused", "Torch", "Compiled")],
            *[f"{times[name] / times['Fused']:.3f}x" for name in ("Torch", "Compiled")],
        ]
        for batch, times in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print(f"\nSummary: BF16, {layers} weight sets, median interleaved CUDA-graph timing per block.")
    print("Speedup=baseline/Fused. Compilation, weight packing and validation excluded.")
    for index, row in enumerate([headers, *rows]):
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16, 32], help="Decode batches")
    parser.add_argument("--layers", type=int, default=1, help="Distinct weight sets to rotate through")
    parser.add_argument("--profile", action="store_true", help="Capture named CUDA-graph replays")
    parser.add_argument("--trace-only", action="store_true", help="Launch only one custom block per batch for IKET")
    args = parser.parse_args()
    if args.layers < 1 or any(not 1 <= b <= 32 for b in args.batch_size):
        parser.error("Expected layers >= 1 and 1 <= batch size <= 32")
    if args.trace_only:
        for batch in args.batch_size:
            trace_once(batch)
    else:
        results = [(batch, benchmark(batch, args.layers, args.profile)) for batch in args.batch_size]
        print_summary(results, args.layers)
