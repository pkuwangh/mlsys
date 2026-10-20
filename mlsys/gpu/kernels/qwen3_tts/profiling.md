# Post-Attention Fusion Profiling

H200, 2026-10-19. Baseline: `f0a74e6`. See the [README](README.md#performance)
for unprofiled block timings and the full-backbone validation caveat.
The initial comparison below ends at `c03b2bc`; the prefetch follow-up describes
`9465e8b`. The handoff follow-up describes the current synchronization changes.

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

## First-Round Changes

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

## First-Round NCU

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
adds per-warp timestamp ranges inside a kernel. The initial traces used
instrumented scratch copies. The kernel now includes named ranges, which are
removed when compiling without IKET profiling.

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

### Detailed Markers

| Marker | Work included |
| --- | --- |
| `output_projection`, `gate_up_projection`, `down_projection` | Projection source region, with operand-wait and WGMMA subranges |
| `operand_wait`, `mma_issue`, `mma_drain` | Wait for operands, issue WGMMA, wait for completion |
| `activation_wait`, `operand_copy` | Poll activation words into registers, then stage operands in SMEM |
| `splitk_store`, `splitk_receive` | Push partials into owner mailboxes, then wait for transaction completion; **not** the reduction itself |
| `output_reduce`, `down_reduce` | Load and sum the four sender slots in local SMEM |
| `output_residual` | Residual addition, BF16 rounding, packed sentinel-safe publication |
| `output_cta_sync` | CTA rendezvous after residual publication |
| `gate_up_prefetch_wait` | Early consumers wait for the first-MMA progress hint (N16) or one residual word (N32) |
| `normalization_input_wait` | Validate every residual payload word, retrying incomplete vectors |
| `norm_prefetch`, `norm_local_stats` | Load residual/gamma, then compute the full sum and inverse RMS within each warp |
| `norm_apply`, `norm_publish` | Apply RMSNorm into SMEM, then proxy fence and the final CTA rendezvous |
| `gate_up_l2_prefetch` | N16 early consumers prefetch the other 64 gate/up weight tiles into L2 |
| `gate_up_weight_prefetch` | Issue weight TMA; does not wait for its completion |
| `gate_up_accumulator_store`, `silu_and_gate` | Accumulator staging and BF16 SwiGLU |
| `activation_sync` | CTA rendezvous after packed activation publication |
| `down_residual_store`, `down_cluster_release` | Final residual addition/store and DSMEM lifetime rendezvous |

Stage payloads are 0/1 for output/down; `gate_up_projection` carries the tile
index. `splitk_store`/`splitk_receive` payloads are 0 for output, 1 for down;
`activation_sync` retains payload 1 for the post-publication rendezvous. A loader
warp finishing a projection range does not mean the compute warps have finished.
Barrier stalls can appear as gaps between ranges; markers are not additional
hardware completion fences. Keep each event name's payload type consistent:
mixing payload/no-payload uses caused missing and incorrectly paired ranges in
this IKET build.

Older captures contain `normalization_start_wait`, `norm_partials_ready`,
`norm_stats_reduce`, and `norm_inverse_ready`. The warp-per-row norm removes
the latter three; `norm_apply` no longer includes the final CTA rendezvous.

From `mlsys/gpu/kernels/qwen3_tts`, capture one B16 block without reference
GEMMs or timing loops:

```bash
run-iket --output-dir /path/to/trace profile --postprocess all \
  --max-ts-cnt-per-warp 256 --keep -- \
  python mega_qwen3_tts/kernels/post_attention.py --batch-size 16 --trace-only
```

### Prefetch Follow-Up

H200 `viking-prod-260`, 28 rotating weight sets, interleaved CUDA graphs:

| Batch | Compiled Torch | `c03b2bc` | Current |
| --- | ---: | ---: | ---: |
| 16 | 27.11 us | 16.24 us | 14.64 us |
| 32 | 27.81 us | 19.49 us | 19.54 us |

The retained change starts the full gate/up weight TMA **before normalization**,
so that transfer can overlap the input-readiness wait and normalization. B16
also loads residual/gamma into registers before the statistics reduction.
The B32 normalization path is unchanged: applying the same register prefetch
there regressed slightly. One full-K transfer replaces B16's four K256
transfers and removes their intermediate WGMMA drains. CTA roles, arithmetic,
rounding, DSMEM reduction order, and release/acquire synchronization remain intact.

Isolated B16 trials, in microseconds:

| Experiment | Time | Decision |
| --- | ---: | --- |
| Normalization-input prefetch alone | 15.85 | Keep at B<=16 |
| Also start four gate/up transfers before normalization | 14.87 | Improvement |
| One full-tile gate/up transfer plus cleanup | 14.64 | Keep |
| Two-wave warp-specialized gate/up epilogue, with norm prefetch | 15.97 | No advantage over the simpler one-wave schedule |
| Pack BF16 payload and epoch into each activation word | 24.55 | Reject; per-vector polling and extra traffic are expensive |

Removing the post-output cluster barrier or retaining residuals in registers
also regressed B16. A long synchronization range is a useful investigation
target, not proof that removing that barrier will make the complete kernel faster.

The new B16 trace separates the previously ambiguous split-K-to-normalization
gap. For CTA 0 / warp 0, it shows DSMEM reduction, residual/statistics stores,
ready-flag release, and the cluster rendezvous before normalization. Gate/up
weights are ready when compute starts: CTA 0 / warp 4 spends about 0.03 us in
its gate/up operand wait. These are instrumented single-block observations,
not the rotating-weight benchmark timings above.

Memcheck and racecheck pass B=1/8/16/17/32 with zero/new-input graph replays;
results are bitwise identical to the original fused implementation.
NCU reports 60 registers/thread at B16 (previously 40), unchanged 169,028-byte
dynamic SMEM, and zero local-memory load/store traffic. Cold-cache profiled
duration is 16.74 us; it is not the unprofiled graph latency.
SASS retains 128-bit operand loads/stores and has no local load/store instructions.

A same-node real-checkpoint replay also matches `c03b2bc` bit-for-bit across all
32 decode steps and KV caches at B16/B32. Diagnostic full-talker medians:

| Batch | Compiled Torch | `c03b2bc` fused path | Current fused path |
| --- | ---: | ---: | ---: |
| 16 | 1.715 ms | 1.450 ms | 1.393 ms |
| 32 | 2.010 ms | 1.731 ms | 1.724 ms |

Both fused versions and compiled Torch still fail the existing strict
pointwise comparison against eager Torch. The prefetch change does not alter
that numerical behavior; it is not a generation-quality validation.

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

## Handoff Follow-Up

Two changes reduce synchronization cost without removing data-readiness checks:

- **Split-K push mailboxes.** WGMMA threads send FP32 partials directly to their
  owner's SMEM with `st.async`, completing bytes on that owner's transaction
  barrier. The reducer waits for all four sender slots and adds them in the same
  order. This replaces remote loads and the ordered cluster rendezvous. N16
  needs a separate 4-KiB mailbox: a fast peer can send down partials while the
  receiving CTA still uses its gate/up epilogue buffer. N32 already separates
  those roles. The final cluster lifetime rendezvous uses relaxed arrival.
- **Payload readiness for gate/up.** Publish two BF16 values per aligned 32-bit
  volatile store, reserving `0x80008000` as unpublished. Live negative zeros are
  converted to positive zero. Consumers poll four words at a time, check every
  word, and use those register values directly. They do not read a flag and then
  reload the payload. Two slots alternate; each invocation clears the inactive
  slot for the next serialized invocation, without an extra reset kernel.

The first scalar-vector polling experiment was slower: B16 went from 14.60 to
17.50 us. Issuing all stage loads before testing readiness recovered parallel
memory accesses; retry scans only revisit incomplete vectors. The initial
mailbox prototype also failed changed-input validation because it aliased the
gate/up buffer, which is why N16's incoming mailbox is separate.

Final interleaved rotating-weight medians on H200 `viking-prod-260`:

| Batch | Compiled Torch | `9465e8b` | Current | Reduction vs `9465e8b` |
| --- | ---: | ---: | ---: | ---: |
| 16 | 27.14 us | 14.59 us | 13.50 us | 7.5% |
| 32 | 27.76 us | 19.53 us | 18.38 us | 5.9% |

At this stage, the residual/statistics-to-normalization handoff still used a
GPU-scope release flag and acquire polling (replaced in the next follow-up).
A GPU-scope fence is not itself a grid-wide execution barrier; its scope
does not establish a fixed cost equivalent to a kernel boundary.

The B16 SASS listing has one `MEMBAR.ALL.GPU` site, down from nine in `9465e8b`;
these are static instruction sites, not a per-CTA execution count. B32 also has
one remaining site. NCU reports 62/218 registers per thread at B16/B32 and zero
local-memory load/store traffic. Dynamic SMEM is 173,144/222,360 bytes per CTA.
Cold-cache profiled durations are 15.81/20.03 us, separate from graph timings.
Memcheck and racecheck pass B=1/8/16/17/32 with zero/new-input graph replays and
exact output agreement with the original fused implementation.

The new B16 IKET trace replaces `splitk_cluster_sync` with `splitk_receive` and
has no `output_cluster_release` or `activation_release` ranges. In CTA 0, the
local mailbox reduction is about 0.06 us and the final relaxed cluster rendezvous
about 0.22 us. The residual/statistics readiness wait remains visible before
normalization; that handoff is still a useful next target. Instrumented timings
are not the unprofiled performance numbers in the table.

Real-checkpoint replay matches `9465e8b` bit-for-bit in logits and KV caches for
all 32 captured decode steps at B16/B32. Diagnostic full-talker medians:

| Batch | Compiled Torch | `9465e8b` fused path | Current fused path |
| --- | ---: | ---: | ---: |
| 16 | 1.714 ms | 1.400 ms | 1.367 ms |
| 32 | 2.010 ms | 1.730 ms | 1.709 ms |

Both compiled Torch and both fused versions still fail the previously documented
strict pointwise tolerance against eager Torch; this is not a new numerical
regression or an end-to-end TTS quality claim.

## Residual Handoff Follow-Up

The first handoff now uses the same packed BF16 sentinel protocol as gate/up:
output-projection CTAs publish residual pairs, and normalization consumes exactly
the words it polled. Statistics are computed locally with the original reduction
tree, eliminating both the global statistics buffer and the ready flags.

Early consumer CTAs first poll one residual word with one thread. This reduces
polling traffic while output projection runs; it does not establish readiness
of any other word. All payload words are still checked before use. Residuals
also have two alternating slots, with the inactive slot reset inside the kernel.

Polling separate residual and statistics buffers regressed performance. Local
statistics with row-major SMEM scratch, plus the single-thread initial wait,
gave a small improvement without another publication protocol:

| Batch | Compiled Torch | Before | Residual sentinel handoff |
| --- | ---: | ---: | ---: |
| 16 | 27.12 us | 13.48 us | 13.31 us |
| 32 | 27.76 us | 18.27 us | 18.08 us |

Same H200, 28 rotating weight sets, interleaved CUDA graphs, median of three
rounds. Both inactive-slot resets and local statistics are included.

SASS now has **zero `MEMBAR.ALL.GPU` sites** at B16 and B32; CTA-local fences
remain. Registers/thread are 96/218 (previously 62/218), with unchanged dynamic
SMEM and zero local-memory traffic or spill instructions. Cold-cache NCU
durations are 14.82/19.01 us, separate from the graph timings above. Memcheck
and racecheck pass B=1/8/16/17/32, including zero and changed-input graph replay.

Real-checkpoint replay preserves every logit and KV-cache byte over 32 captured
decode steps at B16/B32. Diagnostic full-talker medians are 1.374 -> 1.357 ms
at B16 and 1.718 -> 1.709 ms at B32; the latter is close to measurement variation.
The existing strict pointwise failures against eager Torch are unchanged.

## Early Consumer Polling

B16 CTAs 64-95 were tested with delayed polling and redundant output-projection
work. H200, 28 rotating weight sets, interleaved graphs, five measurement rounds:

| Experiment | Complete block latency |
| --- | ---: |
| Current kernel | 13.30-13.36 us |
| Wait 4 us after gate/up prefetch, before polling | 13.17-13.19 us |
| Wait 4 us before gate/up prefetch and polling | 12.65-12.73 us |
| Redundant output GEMM, no global publication | 13.32 us |

The delay sweep covered 0.5/1/2/3/4/5/6 us after prefetch. The stronger gain
when delaying prefetch too suggests early weight traffic matters, not only
polling. Redundant GEMM does not improve latency once its arithmetic is retained.
The first unused-result prototype measured about 12.75 us, but SASS showed the
compiler removed the arithmetic, leaving staging and two dummy MMA instructions.
The retained-compute version stores one accumulator per compute thread to
volatile scratch SMEM, without split-K reduction or global output publication.
SASS confirms all 32 additional MMA instructions remain: 176 static sites and
49,152 warp-level executions, versus 144 and 45,056 in the baseline.

These remain scratch experiments, not default scheduling changes. Three random
seeds across all 28 weight sets and zero/changed-input graph replay pass the
existing block tolerance. No full-backbone or other-batch claim is made here.

### Why the Norm Range Was Long

In the earlier B16 trace, GPC0/TPC0/vSM12 maps to CTA16. Its old
`norm_stats_reduce` range covered local squared sums, a CTA rendezvous, the
64-partial reduction, and another rendezvous: roughly 1.6-1.8 us, not just the
final sum. The new markers separate these steps. The final reduction itself is
about 0.19-0.22 us in the detailed trace.

Warps leave residual polling at different times, and some local-statistics
ranges appear in multiple pieces. A diagnostic warp sync before local statistics
removes those split ranges and shows about 0.16-0.29 us of local work, but does
not improve unprofiled block latency. This supports treating the long original
range as work plus readiness/reconvergence and CTA waiting, not pure arithmetic.
IKET can show barrier waiting as a gap after the barrier marker, so the next
phase's start also matters.

NCU also confirms four-way bank conflicts on the eight final-statistics SMEM
loads. Swizzling their row layout removes the conflict pattern by construction;
the isolated timing trial improves about 0.15 us, much less than the full long
range. Neither that layout nor the diagnostic warp sync was made the default.

## Norm and Prefetch Scheduling

Baseline for this round: `465a629`. All timings below are unprofiled,
interleaved CUDA graphs on H200, rotating 28 weight sets. The fixed delays
and alternative signals were scratch experiments, not public kernel options.

Final reversed-order A/B: B16 **13.30 -> 12.22 us**, B32 **18.16 -> 16.64 us**,
median of three rounds. Compiled Torch: 27.14 / 27.84 us. The source changes
preserve the block tolerance, not bitwise equivalence to the previous kernel.

With the original norm, B16 scheduling experiments gave:

| Scheduling change | us/block |
| --- | ---: |
| Baseline | 13.35 |
| Delay weight prefetch by 2 / 4 / 6 us | 13.39 / 12.74 / 13.40 |
| Delay polling by 4 us, but leave weight TMA early | 13.29 |
| Prefetch the other gate/up tiles into L2 immediately | 14.11 |
| Delay 4 us, then prefetch own tile plus other tiles into L2 | 12.44 |
| Signal after output GEMM, before split-K, then prefetch own tile | 13.91 |
| Same signal, also prefetch other tiles into L2 | 13.06 |
| Signal after the first output MMA stage, with L2 prefetch | 12.60 |
| Wait for an existing residual word, with L2 prefetch | 12.69 |

The delay sweep suggests that early weight traffic matters, not just polling.
It does not establish a universally optimal delay. The retained N16 schedule
uses actual producer progress: rank 3 of each output cluster publishes an epoch
hint after its first MMA stage. CTAs 64-95 wait on these hints, prefetch the
other 64 gate/up tiles into L2, and load their own tiles into SMEM. The hint
uses a volatile word, not a release/acquire payload protocol. Every residual
word is still independently polled before use. N32 instead waits for an existing
residual word before weight TMA; the early-MMA signal was slower there.

For RMSNorm, reducing four warp sums and computing inverse RMS independently
removed a barrier but regressed B16 from 13.33 to 15.13 us. Neither that version
nor the baseline spilled; the four-warp version generated substantially more
shuffle/inverse-RMS instructions. Computing inverse RMS immediately before
applying it did not recover the loss.

The retained norm assigns a full 1024-channel row to one warp. Each lane keeps
four 8-value groups in registers; eight warps cover two rows each at N16 and
four at N32. The reduction uses warp shuffles, with no shared partials or inverse
broadcast. Only the final SMEM-to-WGMMA publication rendezvous remains. This
alone measured 12.87 / 17.20 us at B16/B32, versus 13.33 / 18.00 us. Upstream
BF16 cast boundaries are unchanged; the FP32 sum order changes.

Preserving the previous summation order with warp gathers restored bitwise
identity in the 32-step checkpoint replay, but cost about 1 us/block at B16
(13.23 us combined). Eight lanes per row reduced that cost to 12.94 us at B16
but regressed B32 to 18.10 us. Neither is the current default.

The faster norm changes the full-model result starting at replay step 14.
Mean relative L2 logit error against eager was 0.913% / 0.927% at B16/B32,
versus 0.952% / 0.923% before; maximum error stayed 2.242% in both cases.
Top-token agreement was 31/32 steps at B16 and 32/32 at B32, versus 32/32 before.
The old/new pointwise comparison fails from step 14 onward. The block still
passes its existing tolerance; do not describe this as bitwise-preserving or
as a validated free-running generation improvement.

Removing the prefetch-wait CTA barrier is safe because the immediately
following TMA initialization barrier already joins the waiting thread. It
left B16 effectively unchanged and saved about 0.2 us at B32. The norm's final
WGMMA publication barrier remains.

Final SASS has 122 / 218 registers per thread at B16/B32, no local-memory
loads/stores, and no `MEMBAR.ALL.GPU`. Vectorized operand loads/stores remain.
Memcheck and racecheck pass B=1/8/16/17/32, including changed-input graph replays.

In the new B16 IKET pair, the median interval from the first `norm_local_stats`
timestamp to `gate_up_projection` starts falls from 2.000 to 0.704 us for CTAs
0-63, and from 2.144 to 1.824 us for CTAs 64-95. This includes inter-marker
waiting, not just arithmetic. Divergent polling can still split ranges; do not
sum their displayed durations as independent compute operations. The final
publication rendezvous is now explicitly labeled `norm_publish`.

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

Prefetch follow-up artifacts on `viking-prod-260`:

- A/B: `sglang_qwen_variant_ab_20261019_151133_352279.log`.
- Memcheck/racecheck: `sglang_qwen_variant_sanitize_20261019_151233_502975.log`
  and `sglang_qwen_variant_sanitize_20261019_151624_078030.log`.
- Detailed B16 IKET: `qwen3_tts_0p6b_h200_b16_prefetch_iket.pftrace`
  (matching `.trace.json` and `.html` beside it).
- NCU: `qwen_ncu_mega_qwen3_tts.kernels.post_attention_b16_prefetch.ncu-repz`.
- SASS: `qwen3_tts_h200_b16_prefetch.sass`.
- Real-checkpoint replay: `sglang_qwen_backbone_ab_20261019_151842_873156.log`
  (`Original` in this log means `c03b2bc`).

Tools: Nsight Systems 2026.5.1.97 and Nsight Compute 2026.4.0.6. The latter
handled this compiler's generated code; older NCU builds failed to profile it.

Handoff follow-up artifacts on `viking-prod-260`:

- A/B: `sglang_qwen_variant_ab_20261019_154841_711841.log`.
- Validation and profiler collection: `sglang_qwen_handoff_verify_20261019_154941_577483.log`.
- IKET: `qwen3_tts_0p6b_h200_b16_handoff_iket.pftrace`, with matching
  `.trace.json`, `.pftrace.gz`, and `.html` directly under `tmp/`.
- SASS: `qwen3_tts_0p6b_h200_b{16,32}_handoff.sass`.
- NCU: `qwen_ncu_mega_qwen3_tts.kernels.post_attention_b{16,32}_handoff.ncu-repz`.
- Real-checkpoint replay: `sglang_qwen_handoff_backbone_20261019_155209_290791.log`
  (`Original` here means `9465e8b`).

Residual handoff follow-up artifacts on `viking-prod-260`:

- A/B: `sglang_qwen_variant_ab_20261019_161534_380119.log`.
- Validation and profiler collection: `sglang_qwen_all_lamport_verify_20261019_161753_604413.log`.
- IKET: `qwen3_tts_0p6b_h200_b16_all_lamport_iket.pftrace`, with matching
  `.trace.json`, `.pftrace.gz`, and `.html` directly under `tmp/`.
- SASS: `qwen3_tts_0p6b_h200_b{16,32}_all_lamport.sass`.
- NCU: `qwen_ncu_mega_qwen3_tts.kernels.post_attention_b{16,32}_all_lamport.ncu-repz`.
- Real-checkpoint replay: `sglang_qwen_all_lamport_backbone_20261019_162021_767748.log`
  (`Original` here means the version immediately before the residual-handoff change).

Early-consumer experiments (B16):

- Delay sweep: `sglang_qwen_variant_ab_20261019_162719_360642.log`.
- Reversed-order timing: `sglang_qwen_variant_ab_20261019_163157_566710.log`.
- Retained redundant GEMM: `sglang_qwen_variant_ab_20261019_163429_156819.log`;
  SASS verification: `sglang_qwen_polling_finish_20261019_163520_769853.log`.
- Detailed baseline IKET: `qwen3_tts_0p6b_h200_b16_polling_baseline_iket.pftrace`.
- Retained redundant GEMM IKET: `qwen3_tts_0p6b_h200_b16_full_redundant_projection_iket.pftrace`.
- Diagnostic warp-sync IKET: `qwen3_tts_0p6b_h200_b16_norm_lane_sync_iket.pftrace`.
  Matching JSON/HTML files are directly beside each trace in `tmp/`.

Norm/scheduling follow-up on `viking-prod-260` (baseline `465a629`):

- Scheduling sweeps: `sglang_qwen_variant_ab_20261019_172712_660938.log`
  and `sglang_qwen_variant_ab_20261019_172908_059917.log`.
- Norm variants: `sglang_qwen_variant_ab_20261019_172813_634941.log`.
- Combined candidates: `sglang_qwen_variant_ab_20261019_173103_113821.log`.
- B=1/8/16/17/32: `sglang_qwen_variant_ab_20261019_173243_810378.log`.
- Final reversed-order A/B: `sglang_qwen_variant_ab_20261019_174702_286202.log`.
- Order-preserving alternatives: `sglang_qwen_variant_ab_20261019_173856_920946.log`
  and `sglang_qwen_variant_ab_20261019_174232_345634.log`.
- Checkpoint replay: `sglang_qwen_norm_backbone_20261019_174017_152932.log`;
  old-order control: `sglang_qwen_norm_backbone_20261019_174122_667800.log`.
- Final sanitizers and SASS resources: `sglang_qwen_norm_finish_20261019_174344_800808.log`
  (profiling was retried separately after an output-directory error).
- SASS: `qwen3_tts_0p6b_h200_b{16,32}_after_norm_schedule.sass`.
- IKET: `qwen3_tts_0p6b_h200_b16_{before,after}_norm_schedule_iket.pftrace`;
  the slower four-warp trial is `qwen3_tts_0p6b_h200_b16_four_warp_norm_iket.pftrace`.
  Matching JSON/HTML files are directly under `tmp/`.

## Residual Resume-Scan Experiment

H200 `viking-prod-321`, B16/B32, rotating 28 weight sets, interleaved CUDA
graphs. Times below are median microseconds per block across three rounds.
All candidates preserve the current kernel's output bit-for-bit across three
random seeds and pass zero/changed-input graph replay checks.

| Polling implementation | B16 | B32 |
| --- | ---: | ---: |
| Current loop (first run) | 12.160 | 16.476 |
| Resume scan, sentinel-initialized registers | 12.798 | 17.534 |
| Resume scan with persistent completion mask and initial bulk loads | 13.772 | 17.423 |
| Current loop (control run) | 12.167 | 16.509 |
| Resume scan retaining initial bulk loads | 12.401 | 16.866 |

The direct resume scan checks saved groups and reloads only unfinished ones,
checking the newly loaded values on the next pass. The completion-mask variant
also skips checking groups already known to be complete. Neither beats the
current loop. Keeping the initial loads issued together recovers most of the
direct scan's regression, but the changed retry loop still loses about 2%.
No candidate was promoted; normalization arithmetic and the main kernel remain
unchanged by this experiment.

Artifacts directly under `tmp/`:

- `qwen_norm_resume.py`, `qwen_norm_resume_mask.py`, `qwen_norm_resume_prefetch.py`.
- First run: `sglang_qwen_variant_ab_20261019_235416_712812.log`.
- Control run: `sglang_qwen_variant_ab_20261019_235507_188235.log`.

## Earlier Output-Projection TMA

The previous prologue loaded the epoch, synchronized the CTA, initialized
DSMEM mailboxes, synchronized the cluster, and reset inactive payload slots
before issuing the first weight TMA. These operations do not produce the
output-projection weights.

For N16, thread 0 now initializes the two output operand barriers, prefetches
the weight tensor map, and issues both weight TMAs before the epoch setup.
The existing epoch CTA barrier also joins the earlier barrier initialization.
Operand-loader threads still publish their SMEM stores through the proxy
fence and barrier arrival; MMA still waits for both operands. N32 and the
down projection keep their previous staged weight-loading schedule.

On H200 `viking-prod-321`, with 28 rotating weight sets and interleaved graph
timing, B16 improves from 12.149 to 11.711 us/block (3.6% lower latency).
Applying early TMA to B32 did not give a clear improvement (16.515 versus
16.541 us), so it was not retained for that bucket.

Fresh B16 IKET captures show the following medians relative to each warp's
recorded entry, not the host launch timestamp:

| Event | Before (us) | Early TMA (us) |
| --- | ---: | ---: |
| Weight-prefetch range starts | Not separately marked | 0.096 |
| First output MMA issue starts | 1.408 | 1.312 |
| Second output MMA issue starts | 1.984 | 1.760 |
| Output projection completes, MMA warps | 2.240 | 2.016 |

The prefetch marker includes descriptor prefetch and barrier initialization;
it is not the exact timestamp of the TMA instruction. This follows the
guide's "start independent work before waiting" technique. Remaining startup
work includes epoch handling, mailbox setup, resets, and activation staging.

Validation: B16/B32 candidates matched the prior kernel bit-for-bit across
three seeds and 28 weight sets. B16 racecheck reported zero hazards. The
retained implementation passed memcheck and zero/changed-input graph replay
at B=1/8/16/17/32.

Artifacts directly under `tmp/`:

- Baseline source: `qwen_startup_before.py`; candidate: `qwen_startup_early_tma.py`.
- A/B: `sglang_qwen_variant_ab_20261020_001036_999073.log`.
- IKET: `qwen3_tts_0p6b_h200_b16_startup_{before,early_tma}_iket.pftrace`,
  with matching JSON/HTML files.
- Racecheck: `sglang_qwen_norm_sanitize_20261020_001254_146677.log`.
- Memcheck: `sglang_qwen_norm_sanitize_20261020_001353_732512.log`.

## Earlier Activation Loading

H200 `viking-prod-214`, B16, 28 rotating weight sets. Median us/block from
three interleaved rounds:

| Experiment | Paired control | Candidate |
| --- | ---: | ---: |
| Stage both activation tiles early with vector loads/stores | 11.560 | 12.350 |
| Early 16-byte `cp.async` with mbarrier completion | 11.560 | 11.632 |
| Early activation TMA alongside weight TMA | 11.627 | 11.669 |
| Activation TMA, parallel epoch update, inactive resets at the tail | 11.595 | 11.643 |

The TMA experiment views the contiguous `[B,2048]` activation as
`[32,B,64]` with strides `[64,2048,1]`. A `(4,N,64)` box lands directly in
the K256 swizzled operand layout; no extra packing kernel is required.
Its bytes and the weight bytes contribute to the same stage barrier.

All four B16 variants preserve outputs exactly, but none improves the full
block. The committed post-attention schedule remains unchanged. Logs:

- `tmp/sglang_qwen_variant_ab_20261020_003344_527715.log`
- `tmp/sglang_qwen_variant_ab_20261020_003815_489576.log`
- `tmp/sglang_qwen_variant_ab_20261020_004725_649484.log`

## QKV Preparation

The new block covers input RMSNorm through the KV-cache write; attention
itself is still Torch SDPA. Its 128 CTAs form 32 independent four-CTA
clusters, with no global producer/consumer dependency.

The initial IKET capture exposed two costs worth addressing:

- Per-element `st.async` partial publication was slower than staging a
  contiguous partial tile and bulk-copying it to the reducer.
- Eight norm warps each processed two/four rows at B16/B32. Sixteen warps
  process one/two rows, while the WGMMA group remains warps 4-7.

Only ranks 1-3 issue the remote bulk copy. Memcheck caught the invalid
self-destination case in the first prototype; rank 0 now writes directly
into its local reduction buffer, and the barrier expects three transfers.
The extra SMEM is one local FP32 partial tile (4/8 KiB for N16/N32).

SASS also exposed BF16 FMA contraction in RoPE despite explicit casts.
`--fmad=false` restores upstream's separately rounded products. An isolated
RoPE test now requires bitwise agreement, rather than hiding this behind
the GEMM tolerance. Packed RoPE shuffles were tested but not retained:
the B16 improvement was negligible and B32 did not improve.

Final validation and measurements:

- Standalone, including the direct-slot-update compiled control:
  `tmp/sglang_pre_attention_20261020_011410_740838.log`.
- Memcheck B=1/8/16/17/32: `tmp/sglang_qwen_qkv_sanitize_20261020_010008_664356.log`.
- Racecheck B=1/16/32: `tmp/sglang_qwen_qkv_sanitize_20261020_010117_176908.log`.
- All checkpoint layers, actual inputs:
  `tmp/sglang_qwen_qkv_checkpoint_20261020_010240_330214.log`.
- Interleaved full-talker comparison:
  `tmp/sglang_qwen_qkv_backbone_20261020_010443_149473.log`.
- B16 Nsight comparison, with GPU metrics and graph nodes:
  `tmp/qwen3_tts_0p6b_h200_b16_qkv_post_comparison.1.nsys-rep`.
- Final IKET traces: `tmp/qwen3_tts_0p6b_h200_b{16,32}_pre_attention_iket.pftrace`,
  with matching JSON/HTML.
- SASS: `tmp/qwen3_tts_0p6b_h200_b{16,32}_pre_attention.sass`.
  B16/B32 use 48/70 registers, zero local memory, and no spill loads/stores.
  Input/gamma loads and normalized SMEM stores are 128-bit; Q/K/V stores are
  64-bit. RoPE uses separate packed BF16 multiply/add instructions, not FMA.
  The kernel has CTA fences, but no `MEMBAR.ALL.GPU`.

The full-talker trace also exposes full-cache copies in the fully compiled
Torch baseline. The standalone benchmark therefore reports a second control:
compiled QKV arithmetic followed by direct in-place `index_copy_`, all inside
the timed CUDA graph. This reduces the baseline from 30.14/39.69 us to
22.57/22.88 us at B16/B32. Fused preparation measures 5.99/8.85 us in that
same run. The full-talker baseline was not changed, so its speedup includes
eliminating those cache copies.

## Next: Decode Attention

For each batch row and KV head, two Q heads consume the same K/V history.
The sequence-split approach gives each CTA a history partition and computes
local maximum, exponential sum, and unnormalized weighted-V sum. A cluster
combines them by rescaling each partition to the largest maximum, summing
the rescaled denominators/numerators, and dividing once. This avoids a
global partial-result buffer and a second reduction launch.

Choose the history split using batch size and context length: B16 already
has 128 batch/KV-head groups, whereas B1 has only eight. Splitting a short
history too finely can cost more in synchronization than it gains in
parallelism. The B16 full-talker capture confirms a cuDNN SDPA grid of
`(1,8,16)`, 128 threads/CTA, averaging 5.27 us per layer in the combined
path. A separate mask-conversion kernel adds about 1 us per layer.
There is room to specialize for decode, but insufficient CTA count alone
does not explain this B16 case.

The current QKV clusters own individual heads. Combining QKV and attention
in one launch will require grouping the two Q heads with their K/V head
and coordinating current-token readiness. Historical K/V loads can overlap
projection, and current-token K/V can be published to both cache and
consumer SMEM. That is a later kernel design, not implemented here.

## Remaining Opportunities

The block still spends much more time waiting/staging than issuing tensor-core
work. Further experiments could target normalization staging or a different
projection tiling that reduces those costs without adding another expensive
reduction. More CTAs alone were not
enough. Full-talker numerical and generation-quality validation remains separate
from these scheduling improvements.
