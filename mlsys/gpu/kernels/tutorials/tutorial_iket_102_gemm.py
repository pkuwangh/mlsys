#################################################################################################
#
# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""IKET on CUTLASS's persistent, two-CTA FP16 GEMM (SM100/SM103).

Adapted from NVIDIA CUTLASS fp16_gemm_3.py at e13f5cd4142c43efc37c2f6339923eb4a57afb99:
https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_3.py
The upstream pipeline is retained, with IKET ranges added and a local M-major
persistent loop replacing the deprecated scheduler helper. The host passes an
explicit CUDA stream for graph capture. The driver uses random inputs, an
FP32 reference, and optional untraced timing.

C = A @ B.T: FP16 inputs/output, FP32 accumulation. A persistent CTA pair
computes 256x256x64 tiles: warp 5 loads A/B through six SMEM stages, warp 4
issues MMA, and warps 0-3 drain two TMEM accumulator stages through TMA stores.
This uses CuTe pipeline abstractions; tutorial_iket_101.py shows raw barriers.

Run correctness and uninstrumented performance:
  python tutorial_iket_102_gemm.py --benchmark
Capture one persistent cluster (no benchmark loop):
  run-iket profile --enabled-cluster 0,0,0 --postprocess perfetto -- \
    python tutorial_iket_102_gemm.py

Issue ranges measure instruction dispatch; waits observe readiness/completion.
Open the .pftrace in https://ui.perfetto.dev/. See tutorial_iket.md for roles.
"""

import argparse
import statistics

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass.cute.experimental import iket
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.memory import SmemAllocator, TmemAllocator
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
use_2cta_instrs = True
cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
threads_in_epilogue = 128  # epilogue threads per cta

ab_stages = 6
epi_stages = 2
acc_stages = 2


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
    tile_layout: cute.Layout,
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, cluster_start = cute.arch.block_idx()
    _, _, cluster_stride = cute.arch.grid_dim()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
    is_leader_cta = mma_tile_coord_v == 0

    epilogue_warp_ids = (
        0,
        1,
        2,
        3,
    )
    mma_warp_id = 4
    tma_warp_id = 5

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    num_mcast_participants = cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1

    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)

    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=epilog_sync_bar_id,
        num_threads=threads_in_epilogue,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=tmem_alloc_sync_bar_id,
        num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)),  # 5 warps = 160 threads
    )
    tmem = TmemAllocator(
        storage.tmem_holding_buffer.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])

    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=num_mcast_participants)

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=mainloop_pipeline_producer_group,
        consumer_group=mainloop_pipeline_consumer_group,
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        size=cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
    )

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )
    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=epi_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=epi_smem_layout_staged.inner,
    )

    gA = cute.local_tile(mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None))
    gC = cute.local_tile(mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None))

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )

    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)

    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(gC_epi, 0, 2),
    )

    pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)

    num_k_tiles = cute.size(gA, mode=[3])

    # Each persistent CTA pair visits start, start + gridDim.z, ... in M-major order.
    if warp_idx == tma_warp_id:
        for tile_idx in range(cluster_start, cute.size(tile_layout), cluster_stride):
            tile_m, tile_n = tile_layout.get_flat_coord(tile_idx)
            mma_tile_coord_mnl = (
                cute.arch.make_warp_uniform(tile_m),
                cute.arch.make_warp_uniform(tile_n),
                0,
            )

            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]

            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]

            for k_tile_idx in range(num_k_tiles):
                iket.range_push("tma.wait_empty")
                handle = ab_producer.acquire_and_advance()
                iket.range_pop()
                iket.range_push("tma.issue")

                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_tile_idx)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_tile_idx)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_b,
                )
                iket.range_pop()

        iket.range_push("tma.drain")
        ab_producer.tail()
        iket.range_pop()

    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        for tile_idx in range(cluster_start, cute.size(tile_layout), cluster_stride):
            if is_leader_cta:
                iket.range_push("mma.wait_acc_empty")
                acc_empty = acc_producer.acquire_and_advance()
                iket.range_pop()

                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile_idx in range(num_k_tiles):
                    iket.range_push("mma.wait_ab")
                    handle = ab_consumer.wait_and_advance()
                    iket.range_pop()
                    iket.range_push("mma.issue")

                    num_k_blocks = cute.size(tCrA, mode=[2])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    handle.release()
                    iket.range_pop()

                acc_empty.commit()

        iket.range_push("mma.drain")
        acc_producer.tail()
        iket.range_pop()

    elif warp_idx < mma_warp_id:
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=128,
        )
        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stages,
            producer_group=epilogue_pipeline_producer_group,
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE),
            cutlass.Float32,
        )

        for tile_idx in range(cluster_start, cute.size(tile_layout), cluster_stride):
            tile_m, tile_n = tile_layout.get_flat_coord(tile_idx)
            mma_tile_coord_mnl = (
                cute.arch.make_warp_uniform(tile_m),
                cute.arch.make_warp_uniform(tile_n),
                0,
            )

            iket.range_push("epilogue.wait_acc")
            acc_full = acc_consumer.wait_and_advance()
            iket.range_pop()
            iket.range_push("epilogue.tile")

            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

            tCtAcc_epi = cute.flat_divide(
                tCtAcc[((None, None), 0, 0)],  # why 0,0 ?
                epi_tile,
            )

            mma_tile_coord_mn = cute.slice_(mma_tile_coord_mnl, (None, None, 0))
            tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, *mma_tile_coord_mn)], epi_tile)

            tCgC_tma_cur_tile = tCgC_tma[(None, None, None, *mma_tile_coord_mn)]

            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
                c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
            tCgC_grouped = cute.group_modes(tCgC_tma_cur_tile, 1, cute.rank(tCgC_tma_cur_tile))

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

            for subtile_idx in cutlass.range(subtile_cnt):
                tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
                iket.range_push("epilogue.tmem_to_smem")
                cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)

                c_buffer = subtile_idx % epi_stages
                tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]

                tRS_rC.store(tRS_rAcc.load().to(io_dtype))

                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)

                cute.arch.fence_view_async_shared()
                epilogue_sync_barrier.arrive_and_wait()
                iket.range_pop()
                if warp_idx == epilogue_warp_ids[0]:
                    iket.range_push("epilogue.tma_issue")
                    cute.copy(
                        tma_atom_c,
                        tCsC[(None, c_buffer)],
                        tCgC_grouped[(None, subtile_idx)],
                    )

                    epilogue_pipeline.producer_commit()
                    iket.range_pop()
                    iket.range_push("epilogue.wait_store_slot")
                    epilogue_pipeline.producer_acquire()
                    iket.range_pop()
                epilogue_sync_barrier.arrive_and_wait()

            with cute.arch.elect_one():
                acc_full.release()
            iket.range_pop()

        iket.range_push("epilogue.drain")
        epilogue_pipeline.producer_tail()
        iket.range_pop()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):

    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )

    c_smem_layout_kind = LayoutEnum.from_tensor(c)

    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
    )
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    epi_tile = utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk,
        use_2cta_instrs,
        c_smem_layout_kind,
        io_dtype,
    )

    epi_smem_layout_staged = cutlass.utils.blackwell_helpers.make_smem_layout_epi(
        io_dtype,
        c_smem_layout_kind,
        epi_tile,
        epi_stages,
    )

    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        c,
        epi_smem_layout,
        epi_tile,
    )

    tile_layout = cute.make_layout((c.shape[0] // mma_tiler_mnk[0], c.shape[1] // mma_tiler_mnk[1]))
    grid_shape = (
        cluster_shape_mnk[0],
        cluster_shape_mnk[1],
        min(cute.size(tile_layout), max_active_clusters),
    )

    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c_tma_atom,
        c_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout_kind,
        epi_smem_layout_staged,
        epi_tile,
        cta_layout_vmnk,
        tile_layout,
    ).launch(
        grid=grid_shape,
        block=[192, 1, 1],
        cluster=cluster_shape_mnk,
        stream=stream,
    )


def benchmark(compiled, a, b, out):
    """Median of five alternating rounds, each with ten CUDA-graph replays."""
    reference_out = torch.empty_like(out)
    calls = {
        "CUTLASS": lambda: compiled(a, b, out, cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)),
        "Torch/cuBLAS": lambda: torch.mm(a, b.T, out=reference_out),
    }
    graphs = {}
    for name, call in calls.items():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
    # Validate replay itself: an empty graph must not appear as a fast GEMM.
    out.fill_(float("nan"))
    graphs["CUTLASS"].replay()
    torch.testing.assert_close(out, reference_out, atol=2e-3, rtol=2e-3)
    samples = {name: [] for name in calls}
    for round_id in range(5):
        order = list(calls) if round_id % 2 == 0 else list(reversed(calls))
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 10)
    m, k = a.shape
    n = b.shape[0]
    print("Untraced CUDA-graph timing: median of 5 rounds, 10 replays/round; warm cache")
    for name, times in samples.items():
        ms = statistics.median(times)
        print(
            f"  {name}: {ms:.3f} ms, {2 * m * n * k / ms / 1e9:.1f} TFLOP/s "
            f"(range {min(times):.3f}-{max(times):.3f} ms)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", type=int, nargs=3, default=(4096, 4096, 4096), metavar=("M", "N", "K"))
    parser.add_argument("--benchmark", action="store_true", help="Compare uninstrumented GEMM with Torch/cuBLAS")
    args = parser.parse_args()
    m, n, k = args.mnk
    if any(value <= 0 or value % tile for value, tile in zip((m, n, k), (256, 256, 64))):
        parser.error("M/N/K must be positive multiples of 256/256/64")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This GEMM requires SM100/SM103 Blackwell")
    torch.manual_seed(42)
    # Keep output magnitudes near one as K grows, for a fixed FP16 tolerance.
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    a /= k**0.5
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    descriptors = [make_fake_compact_tensor(cutlass.Float16, t.shape, stride_order=(1, 0)) for t in (a, b, out)]
    max_clusters = utils.HardwareInfo().get_max_active_clusters(2)
    compiled = cute.compile(host_function, *descriptors, max_clusters, make_fake_stream(), options="--enable-tvm-ffi")
    compiled(a, b, out, cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = (a.float() @ b.float().T).half()
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)
    del expected
    print(f"PASS: {torch.cuda.get_device_name()} | M/N/K={m}/{n}/{k} | persistent 2-CTA FP16 GEMM", flush=True)
    if args.benchmark:
        benchmark(compiled, a, b, out)


if __name__ == "__main__":
    main()
