"""Stage A operands in TMEM while waiting for a late B operand.

Two TMEM slots free the two-slot SMEM ring before MMA can start. Compare
against keeping A in SMEM until MMA consumes it. Both compute sum(A_i @ B),
using real TMA loads and BF16 MMA; a controlled delay models B preparation.
Each CTA reads six distinct 64-KiB A tiles.

Command:
  python tutorial_tmem_101_staging.py
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python tutorial_tmem_101_staging.py
"""

import statistics

import cuda.bindings.driver as cuda_driver
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import cuda
from cutlass.experimental import primitives as prims

M, N, K = 128, 8, 256
TILES = 6
SLOTS = 2

TMEM_A_BASE = 32  # Accumulator: columns 0..7; A slots: 32..159 and 160..287.
TMEM_COLS = 512
B_DELAYS_US = (0, 20)
REPLAYS = 5
SAMPLES = 3
MODES = ("smem", "tmem")


@cute.jit
def wait(barrier, phase):
    """Wait for a barrier phase to complete."""
    while not prims.mbarrier_try_wait_parity(barrier, phase):
        pass


@cute.kernel
def staging_kernel(
    a_map: cutlass.GridConstant[cuda.TensorMap],
    b: cute.Tensor,
    out: cute.Tensor,
    stamps: cute.Tensor,
    b_delay_ns: cutlass.Int64,
    STAGE: cutlass.Constexpr[bool],
):
    """Warps load A, prepare B, stage A, and issue MMA, respectively."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tid % 32
    a_smem = cutlass.Array(cutlass.BFloat16, SLOTS * M * K, space=cutlass.AddressSpace.smem, alignment=128)
    b_smem = cutlass.Array(cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128)
    # Full/empty pairs for each ring, followed by B-ready and MMA-done.
    bars = cutlass.Array(cutlass.Int64, 4 * SLOTS + 2, space=cutlass.AddressSpace.smem, alignment=8)
    smem_full, smem_empty = bars, bars.subview(SLOTS)
    tmem_full, tmem_empty = bars.subview(2 * SLOTS), bars.subview(3 * SLOTS)
    b_ready, mma_done = bars.subview(4 * SLOTS), bars.subview(4 * SLOTS + 1)
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if tid == 0:
        for index in cutlass.range_constexpr(4 * SLOTS + 2):
            prims.mbarrier_init(bars.subview(index), 1)
        stamps[bid, TILES, 0] = cute.arch.globaltimer()
    prims.fence_mbarrier_init()
    if warp == 0:
        prims.tcgen05_alloc(tmem_address, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.barrier_cta_sync()
    tmem = tmem_address.load()
    a_desc = prims.Tcgen05SmemDesc.build(
        start_address=a_smem,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    b_desc = prims.Tcgen05SmemDesc.build(
        start_address=b_smem,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    if warp == 0:
        for tile in range(TILES):
            slot = tile % SLOTS
            phase = (tile // SLOTS) % 2
            wait(smem_empty.subview(slot), phase ^ 1)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(smem_full.subview(slot), M * K * 2)
                stamps[bid, tile, 0] = cute.arch.globaltimer()
                prims.cp_async_bulk_tensor_shared_cta_global(
                    a_smem.subview(slot * M * K),
                    a_map.get_ptr(),
                    (
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        bid * TILES + tile,
                    ),
                    smem_full.subview(slot),
                )
            # Observe completion for the timeline; the consumer also waits here.
            wait(smem_full.subview(slot), phase)
            if lane == 0:
                stamps[bid, tile, 1] = cute.arch.globaltimer()

    elif warp == 1:
        start = cute.arch.globaltimer()
        while cute.arch.globaltimer() - start < b_delay_ns:
            prims.nanosleep(100)
        for index in cutlass.range_constexpr(N * K // 32):
            linear = index * 32 + lane
            row, col = linear // K, linear % K
            # Each 64-column block has 128-byte rows with a 128-byte swizzle.
            offset = (col // 64) * N * 64 + row * 64 + col % 64
            offset = offset ^ ((row % 8) * 8)
            b_smem.store(cutlass.BFloat16(cutlass.Float32(b[bid, row, col]) * 2), idx=offset)
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        cute.arch.sync_warp()
        if lane == 0:
            stamps[bid, TILES, 1] = cute.arch.globaltimer()
            prims.mbarrier_arrive(b_ready)

    elif warp == 2:
        if cutlass.const_expr(STAGE):
            for tile in range(TILES):
                slot = tile % SLOTS
                phase = (tile // SLOTS) % 2
                wait(tmem_empty.subview(slot), phase ^ 1)
                wait(smem_full.subview(slot), phase)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    # SMEM layout: [slot][64-column block][row][column], swizzled within each block.
                    # Each copy selects 16 BF16 columns from all 128 rows.
                    for k_start in cutlass.range_constexpr(0, K, 16):
                        smem_slot_bytes = slot * (M * K * 2)
                        smem_block_bytes = (k_start // 64) * (M * 64 * 2)
                        column_bytes = (k_start % 64) * 2
                        # Descriptor addresses use 16-byte units; row strides and swizzling stay unchanged.
                        source_desc = a_desc + (smem_slot_bytes + smem_block_bytes + column_bytes) // 16
                        # TMEM columns hold 4 bytes per row, i.e. two BF16 values.
                        tmem_slot_columns = slot * (K * 2 // 4)
                        tmem_copy_columns = k_start * 2 // 4
                        destination = cutlass.inttoptr(
                            tmem + TMEM_A_BASE + tmem_slot_columns + tmem_copy_columns, 6, cutlass.Int32
                        )
                        prims.tcgen05_cp(
                            prims.Tcgen05CpShape.SHAPE_128X256B,
                            destination,
                            source_desc,
                            group=prims.CTAGroup.CTA_1,
                        )
                    # Copy completion frees SMEM and publishes the TMEM operand.
                    prims.tcgen05_commit(smem_empty.subview(slot))
                    prims.tcgen05_commit(tmem_full.subview(slot))

    elif warp == 3:
        instruction = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            m_dim=M,
            n_dim=N,
        )
        wait(b_ready, 0)
        for tile in range(TILES):
            slot = tile % SLOTS
            phase = (tile // SLOTS) % 2
            if cutlass.const_expr(STAGE):
                wait(tmem_full.subview(slot), phase)
            else:
                wait(smem_full.subview(slot), phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                stamps[bid, tile, 2] = cute.arch.globaltimer()
                for k_start in cutlass.range_constexpr(0, K, 16):
                    column_bytes = (k_start % 64) * 2
                    if cutlass.const_expr(STAGE):
                        tmem_slot_columns = slot * (K * 2 // 4)
                        tmem_copy_columns = k_start * 2 // 4
                        operand_a = cutlass.inttoptr(
                            tmem + TMEM_A_BASE + tmem_slot_columns + tmem_copy_columns, 6, cutlass.Int32
                        )
                    else:
                        smem_slot_bytes = slot * (M * K * 2)
                        smem_block_bytes = (k_start // 64) * (M * 64 * 2)
                        operand_a = a_desc + (smem_slot_bytes + smem_block_bytes + column_bytes) // 16
                    b_block_bytes = (k_start // 64) * (N * 64 * 2)
                    operand_b = b_desc + (b_block_bytes + column_bytes) // 16
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        cutlass.inttoptr(tmem, 6, cutlass.Int32),
                        operand_a,
                        operand_b,
                        instruction,
                        (tile != 0) or (k_start != 0),
                    )
                if cutlass.const_expr(STAGE):
                    prims.tcgen05_commit(tmem_empty.subview(slot))
                else:
                    prims.tcgen05_commit(smem_empty.subview(slot))
        if prims.elect_sync():
            prims.tcgen05_commit(mma_done)

    wait(mma_done, 0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    values = prims.tcgen05_ld("32x32b", cutlass.inttoptr(tmem, 6, cutlass.Float32), num=N)
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    for col in cutlass.range_constexpr(N):
        out[bid, tid, col] = values[col]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync()
    if warp == 0:
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem, 6, cutlass.Int32), TMEM_COLS)


@cute.jit
def staging_host(
    a,
    b,
    out,
    stamps,
    b_delay_ns: cutlass.Int64,
    stream: cuda_driver.CUstream,
    STAGE: cutlass.Constexpr[bool],
):
    """Launch independent CTAs with the same SMEM capacity in both paths."""
    # K64-major storage lets one TMA load cover all four 128-byte-wide slabs.
    a_map = cuda.create_tensor_map_tiled_from_view(a, (1, K // 64, M, 64), swizzle=cuda.TensorMapSwizzle.s128b)
    staging_kernel(a_map, b, out, stamps, b_delay_ns, STAGE).launch(
        grid=(out.shape[0], 1, 1), block=(128, 1, 1), stream=stream
    )


def main():
    """Check both paths, time interleaved replays, and print the prefetch timeline."""
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This demo requires SM100/SM103 Blackwell")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = properties.L2_cache_size
    bytes_per_wave = properties.multi_processor_count * TILES * M * K * 2
    waves = 2 * l2_bytes // bytes_per_wave + 1
    blocks = waves * properties.multi_processor_count
    a = torch.empty((blocks * TILES, K // 64, M, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((blocks, N, K), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((blocks, M, N), dtype=torch.float32, device="cuda")
    stamps = torch.empty((blocks, TILES + 1, 3), dtype=torch.int64, device="cuda")
    descriptors = [
        make_fake_compact_tensor(dtype, tensor.shape, stride_order=tuple(reversed(range(tensor.ndim))))
        for tensor, dtype in (
            (a, cutlass.BFloat16),
            (b, cutlass.BFloat16),
            (out, cutlass.Float32),
            (stamps, cutlass.Int64),
        )
    ]
    compiled = {
        mode: cute.compile(
            staging_host,
            *descriptors,
            cutlass.Int64(0),
            make_fake_stream(),
            mode == "tmem",
            options="--enable-tvm-ffi",
        )
        for mode in MODES
    }
    stream = torch.cuda.Stream()
    graphs = {}
    for delay in B_DELAYS_US:
        for mode in MODES:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                compiled[mode](
                    a,
                    b,
                    out,
                    stamps,
                    delay * 1000,
                    cuda_driver.CUstream(stream.cuda_stream),
                )
            graphs[delay, mode] = graph

    print(
        f"{properties.name} | {properties.multi_processor_count} SMs | {blocks} CTAs ({waves} waves)\n"
        f"{TILES} A tiles/CTA ({M}x{K}, {M * K * 2 // 1024} KiB each) | B={K}x{N} | two SMEM slots, two TMEM slots\n"
        f"A working set={a.numel() * a.element_size() / 2**20:.1f} MiB | L2={l2_bytes / 2**20:.1f} MiB",
        flush=True,
    )
    for seed in (1, 7, 19):
        torch.manual_seed(seed)
        a.copy_(torch.randint(-4, 5, a.shape, device="cuda") / 4)
        b.copy_(torch.randint(-4, 5, b.shape, device="cuda") / 4)
        a_tiles = a.reshape(blocks, TILES, K // 64, M, 64).permute(0, 3, 1, 2, 4)
        expected = a_tiles.float().reshape(blocks, M, TILES, K).sum(dim=2) @ (b.float().transpose(1, 2) * 2)
        for graph in graphs.values():
            out.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print("Both paths: correctness PASS (three inputs, exact FP32 results)", flush=True)

    timings = {key: [] for key in graphs}
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for sample in range(SAMPLES):
        keys = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
        for key in keys:
            start.record()
            for _ in range(REPLAYS):
                graphs[key].replay()
            end.record()
            end.synchronize()
            timings[key].append(start.elapsed_time(end) * 1000 / REPLAYS)

    print("\nB delay us | Mode | Graph us | Mean A tiles loaded before B ready / CTA")
    for (delay, mode), graph in graphs.items():
        graph.replay()
        times = stamps.cpu().tolist()
        loaded = statistics.mean(sum(row[1] < cta[-1][1] for row in cta[:-1]) for cta in times)
        print(f"{delay:10} | {mode:4} | {statistics.median(timings[delay, mode]):8.2f} | {loaded:.2f}/{TILES}")
        if delay:
            origin, b_ready, _ = times[0][-1]
            print(f"  CTA 0 timeline | B ready: {(b_ready - origin) / 1000:.2f} us")
            print("  Tile | TMA issue us | A load done us | Load latency us | MMA issue us")
            for tile, (issue, load, mma) in enumerate(times[0][:-1]):
                print(
                    f"  {tile:4} | {(issue - origin) / 1000:12.2f} | {(load - origin) / 1000:14.2f} | "
                    f"{(load - issue) / 1000:15.2f} | {(mma - origin) / 1000:12.2f}"
                )

    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        for (delay, mode), graph in graphs.items():
            with torch.cuda.nvtx.range(f"{mode}/B_delay_{delay}us"):
                graph.replay()
                torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
