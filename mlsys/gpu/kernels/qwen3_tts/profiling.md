# Post-Attention Fusion Profiling

H200, 2026-10-19. Baseline: `f0a74e6`. See the [README](README.md#performance)
for unprofiled block timings and the full-backbone validation caveat.

## What Was Limiting It

NCU and SASS showed substantial waiting and shared-memory serialization, not
register spills or saturated tensor cores:

- Long-scoreboard samples concentrated around `mbarrier` waits and GPU-scope
  acquire polling. That counter does not mean every stall is a DRAM load.
- The original row-major partial layout conflicted with the DSMEM reduce-scatter
  access pattern. NCU attributed 896 excessive shared wavefronts to each of eight
  remote-load instructions at B=16. Those conflicts disappear with the new layout.
- Gate/up had only 48 compute CTAs, two waves, and a two-slot weight pipeline.
  The producers repeatedly waited for empty buffers; consumers waited for TMA.
- Output/down each used only two K stages, but still carried buffer-reuse
  handshakes from the generic gate/up pipeline.

This supports optimizing internal staging, publication, and epilogues as well
as GEMM. It does **not** establish a fixed 1,000-cycle cost at every CUDA kernel
boundary. The measured barriers here are inside the fused kernel.

## Retained Changes

1. **Batch-dependent CTA roles.** B<=16 reuses 96 CTAs for one gate/up wave;
   B>16 keeps 64 output/down CTAs and 48 gate/up CTAs, preserving overlap.
2. **Resident gate/up weights.** Keep all 128 KiB for one tile in SMEM. N16
   issues four independently ready K256 transfers. N32 uses one full-K transfer,
   prefetching before normalization and overlapping the next transfer with SiLU.
3. **Alias disjoint storage lifetimes.** Gate/up's extra weight space reuses the
   output/down operand buffers; normalized activations have separate storage.
4. **Remove unused handshakes.** Output/down stage buffers are never overwritten
   during their GEMM, so no empty-buffer notifications are needed. Data-readiness
   barriers, proxy fences, cluster ordering, and global release/acquire remain.
5. **Use separate partial layouts.** DSMEM reduction and gate/up epilogues have
   different access patterns and now use different swizzles.

Math, BF16 rounding boundaries, weight packing, and split-K reduction order are
unchanged. The optimized implementation matches the original bit-for-bit in
the tested block and real-checkpoint backbone replays.

## NCU Comparison

Identical collection settings: kernel replay, cache flush before each pass,
full metric set, unlocked clocks. These are single-launch profiled results,
not the CUDA-graph timings used for performance claims.

| Metric | B16 original | B16 optimized | B32 original | B32 optimized |
| --- | ---: | ---: | ---: | ---: |
| Duration (us) | 19.52 | 16.99 | 24.74 | 20.96 |
| DRAM reads (MB) | 23.21 | 23.21 | 23.33 | 23.33 |
| DRAM active cycles (%) | 24.8 | 28.5 | 19.6 | 23.2 |
| SM issue active, elapsed (%) | 3.99 | 5.23 | 4.24 | 5.95 |
| Tensor pipe active, elapsed (%) | 1.91 | 2.19 | 2.97 | 3.58 |
| Registers/thread | 44 | 40 | 60 | 227 |
| Dynamic SMEM/CTA (bytes) | 160836 | 169028 | 222340 | 222340 |
| Local-memory load/store sectors | 0 | 0 | 0 | 0 |

The N32 full-K instruction group increases register allocation substantially.
There are no local loads/stores in SASS and no measured spill traffic. SMEM
already restricts these kernels to one CTA per SM, but the register cost matters
for future fusion. Reducing unrolling slightly regressed B32 (~19.78 vs 19.51 us).

Vector operand loads/stores remain 128-bit. Normalization uses packed BF16
arithmetic; SiLU and final publication still contain scalar operations. The new
layout removes the problematic reduction/epilogue load conflicts, not every
shared-store conflict in the kernel.

## IKET Phase Traces

[IKET](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)
adds per-warp timestamp ranges inside a kernel. Instrumented scratch copies
mark GEMM, operand waits, TMA issue/prefetch, normalization, split-K exchange,
and gate/up epilogues. Production has no IKET dependency or instrumentation.

The traces make the pipeline behavior visible:

- Original B16 gate/up consumes two ~4.2-us GEMM waves per compute warp, with
  exposed operand waits between K stages. The optimized schedule has one wave
  across more CTAs; a single CTA's GEMM is not necessarily faster.
- Original B32 gate/up spends ~4.3 us per GEMM wave. With full-tile prefetch,
  the corresponding instrumented range is ~1.1 us and its operand wait is short.
- Down-projection ranges include waiting for gate/up publication. They must not
  be read as pure down-GEMM compute time.

Those numbers describe instrumented traces, not exact uninstrumented phase
latencies. Instrumentation can change registers and scheduling; nested and
overlapping warp ranges cannot be added to obtain kernel time.

## Experiments Not Retained

Representative unprofiled complete-block timings, in microseconds:

| Experiment | B16 | B32 | Outcome |
| --- | ---: | ---: | --- |
| Reuse 96 CTAs without pipeline/layout changes | 17.62 | 24.05 | Only wins at B16 |
| Reuse 64 CTAs | 23.01 | 28.02 | Too little parallelism |
| Two outstanding WGMMA groups in original pipeline | 19.86 | 25.04 | Slower |
| Interleaved gate/up channels, register shuffle epilogue | 20.08 | ~25.0 | Worse global-store pattern |
| 192 CTAs, two-way gate/up split-K, two CTAs/SM | 17.81 | Not tested | Extra staging/reduction loses to optimized B16 |
| Shared normalization plus cluster copies | ~17.3 | ~21.7 | Extra coordination outweighs saved work |

Relaxed polling, grouping ready flags, explicit paired epilogue stores, delayed
first prefetch, and weaker lifetime-only cluster barriers did not provide a
consistent improvement worth retaining. No experimental schedule switches were
added to the public interface.

## Local Artifacts

All paths below are relative to the repository root, under ignored `tmp/`.
They are measurement artifacts, not dependencies of the implementation.

- Block A/B: `sglang_qwen_variant_ab_20261019_123120_921059.log`.
- Standalone entry point: `sglang_post_attention_20261019_123857_321362.log`.
- Real-checkpoint replay: `sglang_qwen_backbone_ab_20261019_123202_513557.log`.
- Memcheck/racecheck: `sglang_qwen_variant_sanitize_20261019_122741_532578.log`
  and `sglang_qwen_variant_sanitize_20261019_122853_374733.log`.
- NSYS: `nsys_qwen3_tts_0p6b_h200_b{16,32}_optimized_post_attention.1.nsys-rep`.
  Each contains `Compiled`, `Original`, and `Optimized` NVTX ranges, 25 graph
  replays per range, graph-node tracing, and GPU metrics.
- NCU: `qwen_ncu_qwen_post_original_b{16,32}_final_cold.ncu-repz` and
  `qwen_ncu_mega_qwen3_tts.kernels.post_attention_b{16,32}_final_cold.ncu-repz`.
- SASS: `qwen3_tts_h200_b{16,32}_{original,optimized}.sass`.
- IKET: `qwen_iket_qwen_post_{iket,optimized_iket}_b{16,32}_final/` contains
  Perfetto traces, JSON, and HTML viewers.

Tools: Nsight Systems 2026.5.1.97 and Nsight Compute 2026.4.0.6. The latter
handled this compiler's generated code; older NCU builds failed to profile it.

## Remaining Opportunities

The block still spends much more time waiting/staging than issuing tensor-core
work. Next experiments should target the output-projection publication and
normalization dependency, or a different projection tiling that reduces those
costs without adding another expensive reduction. More CTAs alone were not
enough. Full-talker numerical and generation-quality validation remains separate
from these scheduling improvements.
