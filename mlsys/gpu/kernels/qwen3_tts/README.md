# Qwen3-TTS Talker and Code Predictor

Standalone 0.6B and 1.7B components, benchmarked separately:

- **Talker:** prepared embeddings -> 28 transformer layers -> codec logits.
- **Code predictor:** talker hidden state and first code -> 15 remaining codes
  and embedding feedback, reusing the same fused blocks.

Neither benchmark includes text processing, waveform decoding, or a combined
talker/predictor generation loop. See [design.md](design.md) for the math,
CTA/warp schedules, memory handoffs, and retained design decisions.

## Run

Use a Hopper or datacenter Blackwell GPU and the `mlsys-gpu` environment with
CUTLASS DSL experimental primitives. Run from this directory:

```bash
# Complete post-attention block, correctness checks, and compiled Torch baseline.
# Rotate 28 weight sets so the benchmark does not keep just one layer hot in L2.
python mega_qwen3_tts/kernels/post_attention.py --layers 28

# Input norm, QKV, head norms, RoPE, cache writes, and optional additive mask.
python mega_qwen3_tts/kernels/pre_attention.py --write-mask

# 1.7B talker shapes, without loading a checkpoint.
python mega_qwen3_tts/kernels/pre_attention.py --hidden-size 2048 --write-mask
python mega_qwen3_tts/kernels/post_attention.py --hidden-size 2048

# Full backbone; omit --model for deterministic random weights.
python bench_qwen3_tts_talker.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
python bench_qwen3_tts_talker.py --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Backbone defaults: BF16, B=1/8/16/32, 28 layers, 128 prefix tokens. Multi-token
prefill stays eager; custom decode uses fused QKV preparation, cuDNN attention
through Torch SDPA, fused post-attention, and compiled remaining Torch operations.
The specialization supports B=1..32, one shared decode position, and immutable
weights. The checkpoint config selects hidden/MLP widths 1024/3072 or 2048/6144;
random weights default to 0.6B dimensions. Training is not supported. Architecture dispatch
selects SM90, SM100, or SM103 automatically; device kernels are separate under
`mega_qwen3_tts/kernels/sm90/` and `sm100/`.

Move the reference to the final CUDA device and BF16 before constructing the
fused module; changing device, dtype, or weights requires rebuilding it.
Call each model's `make_cache` before capture. This also precomputes RoPE;
decode only gathers the current position's cos/sin. The fused mask path skips
the unused boolean-mask preparation.

Useful controls:

- `--no-fused-qkv`: only post-attention fusion.
- `--no-fused-mask`: restore SDPA's separate boolean-mask conversion.
- `--layers 2`: shorten development runs.
- `--inputs /path/to/inputs.pt --batch-size 16`: replay captured `prefix` and
  `decode` embeddings, each `[B,T,1024]`, with the matching `--model` checkpoint.

Without `--inputs`, embeddings are synthetic even with real weights. Captured
inputs are teacher-forced: custom predictions are not fed back into the model.
Each implementation has its own KV cache. Timings use interleaved CUDA graphs,
three samples of five replays; compilation, packing, allocation, and cache setup
are excluded. Backbone timing repeatedly overwrites the final fixed KV slot,
not a growing-context generation loop.

## Cache-Slot Interface

`SlotPreAttention` accepts per-token positions and writes directly to separate,
caller-owned K/V buffers of shape `[slots, 8, 128]`. RoPE uses a
`[positions, 128]` table containing 64 cosine and 64 sine values per position.
`PackedPostAttention` accepts projection and normalization weights directly,
without a reference-model dependency. Construct both modules before graph capture.

```bash
python -m pytest test_cache_slots.py -q
```

Tests check changing positions/slots under CUDA graph replay, untouched cache
entries, irregular batches, both hidden widths, and post-attention workspace reuse
against the Torch reference. The cache-slot path has been tested on H200;
Blackwell validation remains outstanding.

## Code Predictor

```bash
python bench_qwen3_tts_predictor.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice

# Standalone greedy selection + embedding lookup, including optional projection.
python mega_qwen3_tts/kernels/argmax_embedding.py
```

Both predictors have five layers with 1024/3072 widths and 16 Q / 8 KV
heads, matching the 0.6B talker. The 1.7B predictor additionally projects its
2048-wide input embeddings to 1024; that projection stays in Torch.
Each frame runs a two-token prefix followed by 14 single-token
steps, producing 15 codes with group-specific heads and embeddings. The custom
path reuses both fused decode blocks; multi-token prefix operations remain Torch.
Greedy argmax and embedding lookup use one custom kernel per code. Use
`--no-fused-selection` to isolate that change; teacher-forced codes bypass it.

Both paths precompute RoPE and additive masks during cache setup. Steps use table
views, avoiding repeated setup kernels while retaining a fixed-capacity KV cache.

The benchmark compares eager, compiled, and fused **complete frame generation**:
prefix, decode, greedy selection, embedding lookup, and the final embedding sum.
All three use interleaved CUDA graphs. Defaults are B=1/8/16/32; `--profile` captures
one frame per variant. Optional `--inputs` loads a `.pt` with `hidden [B,1,1024]`
and `first_code [B,1]` from the matching talker. Otherwise inputs are synthetic.

H200, real 0.6B checkpoint, 16/32 distinct frames from one captured prompt:

| Batch | Torch | Compiled Torch | Fused | Compiled/Fused |
| --- | ---: | ---: | ---: | ---: |
| 16 | 10.068 ms | 4.740 ms | 2.070 ms | 2.29x |
| 32 | 10.217 ms | 4.667 ms | 2.755 ms | 1.69x |

Selection fusion alone reduced B16/B32 frame time from 2.348/2.866 ms to
2.068/2.749 ms in a paired run. Codes, embeddings, logits, and caches remained
bitwise identical, including changed-input and repeated-frame graph replays.
Standalone irregular-width, mixed-dtype, tie/NaN, and projected-table checks pass;
CUDA memcheck/racecheck reported no errors or hazards.

The eager reference matches upstream logits, KV, and feedback exactly at
B=1/16/32 with matching static caches and math SDPA. Changed-input, stale-cache,
and repeated-frame CUDA-graph checks pass at B=1/16/17/32. On this captured input,
fused teacher-forced top-code agreement is 96.7%/95.8%; free-running greedy code
agreement is 81.3%/81.0%.

Stage-by-stage checks attribute the observed differences primarily to projection
reduction/rounding order, with rare Q/K head-normalization rounding differences;
no indexing, cache-write, or BF16-boundary mismatch was found in the tested cases.
Eager Torch is not an exact-arithmetic oracle: using FP64 projection reductions
with the same BF16 output boundaries also changes codes (81.25% agreement at B32).
The benchmark reports numerical differences without a tolerance-based failure;
shape, finiteness, and exact feedback checks still raise errors. These standalone
checks do not establish speech quality.
See [numerical attribution](design.md#predictor-numerical-attribution).

## Auxiliary Kernels

Standalone Hopper implementations; these additions are not wired into a serving
framework or the waveform decoder yet. Each file runs correctness checks and an
interleaved CUDA-graph benchmark against `torch.compile`, without a checkpoint:

```bash
python mega_qwen3_tts/kernels/embedding_update.py
python mega_qwen3_tts/kernels/argmax_embedding.py
python mega_qwen3_tts/kernels/sampling.py
python mega_qwen3_tts/kernels/snake_beta.py
```

| File | Fused work |
| --- | --- |
| `embedding_update.py` | Embedding gather + BF16 accumulation + optional preprojected-table lookup |
| `argmax_embedding.py` | Greedy selection + both lookups + optional accumulation and strided code writes |
| `sampling.py` | Top-k + temperature + top-p + seeded Gumbel sampling, optionally including both lookups and accumulation; also accepts prefiltered log probabilities |
| `snake_beta.py` | Channel scaling + sine + square + inverse-beta scaling + residual, preserving each eager BF16 rounding boundary |

Embedding transfers use 128-bit loads/stores and packed BF16 additions. Independent
loads precede stores to avoid serializing the accumulator read. Sampling packs a
BF16 score and token ID into a 32-bit key, discards losing runs during top-k merge,
and recovers scores from keys without rereading logits. One CTA handles each row.
SnakeBeta uses packed BF16 arithmetic and precomputed immutable coefficients;
large tensors additionally use a shared 128 KiB table covering every BF16 sine input.
Small tensors retain direct sine because the extra lookup costs more there.

New embedding paths require contiguous BF16 tables with widths divisible by eight;
token IDs may be strided. Raw-logit sampling supports BF16 `[B,2048]`, top-k up to
128, per-row GPU sampling parameters, and changing seeds/positions under graph
replay. Equal logits use ascending token IDs, **not PyTorch's unspecified `topk`
tie order**. Prefiltered sampling preserves the supplied candidate order and
supports up to 2048 candidates. Floating-point cutoff boundaries can also affect
same-seed parity. SnakeBeta accepts BCT or BTC storage, BF16 only; construct it on
the final device before capture. The existing greedy kernel retains its mixed-dtype
and irregular-width support.

H200, CUTLASS DSL 4.8.0, Torch 2.13.0+cu132, synthetic inputs, 2026-10-22.
Times below are microseconds per call: 16 calls/graph, median of three interleaved
rounds of five replays. Compilation, allocation and immutable-table preparation
are excluded. SnakeBeta's compiled reference preserves BF16 precision casts;
the sampler baseline implements the same deterministic tie ordering.
These are component measurements, not end-to-end speech throughput.

| Operation | Shape | Compiled | CuTeDSL | Speedup |
| --- | --- | ---: | ---: | ---: |
| Gather + accumulation | B16, D1024 | 1.300 | 1.230 | 1.06x |
| Gather + accumulation + projected lookup | B16, D2048/1024 | 2.388 | 1.449 | 1.65x |
| Greedy + gather + accumulation | B16, V2048, D1024 | 3.221 | 1.794 | 1.80x |
| Greedy + both lookups + accumulation | B16, V2048, D2048/1024 | 3.574 | 2.274 | 1.57x |
| Top-k/top-p sampling | B16, V2048, K50 | 26.591 | 4.615 | 5.76x |
| Sampling + both lookups + accumulation | B16, V2048, K50, D2048/1024 | 28.911 | 5.258 | 5.50x |
| Prefiltered sampling | B16, K50 | 4.242 | 1.944 | 2.18x |
| Prefiltered sampling | B16, K2048 | 6.655 | 3.920 | 1.70x |
| SnakeBeta, BCT storage | 4,64,16384 | 10.389 | 5.216 | 1.99x |
| SnakeBeta, BTC storage | 4,64,131072 | 93.800 | 37.294 | 2.52x |

Not every case wins: prefiltered K50 sampling takes 1.879 vs 1.763 us at B1 and
1.991 vs 1.862 us at B32. Small SnakeBeta and
plain gathers are approximately tied. `--tune` exposes the retained launch choices
in the new files.

Changed-input graph checks pass, including accumulation and untouched code slots.
Both SnakeBeta paths match eager Torch across all 65,536 BF16 input encodings with
eight coefficient pairs. H200 memcheck/racecheck pass. Inspected SASS confirms
vectorized transfers and packed arithmetic without register spills.

Matched comparison against the local Omni Triton implementations (`02993ce46`),
on one H200 (`viking-prod-309`), Torch 2.14.1+cu132 / Triton 3.8.0. Same inputs,
interleaved unprofiled graphs and timing method as above; microseconds per call:

| Operation | Shape | Triton path | CuTeDSL | Speedup |
| --- | --- | ---: | ---: | ---: |
| Gather + accumulation | B16, D1024 | 1.225 | 1.213 | 1.01x |
| Gather + accumulation + projected lookup | B16, D2048/1024 | 1.494 | 1.420 | 1.05x |
| Greedy + both lookups + accumulation | B16, V2048, D2048/1024 | 9.662 | 2.252 | 4.29x |
| Top-k/top-p sampling | B16, V2048, K50 | 6.110 | 4.602 | 1.33x |
| Sampling + both lookups + accumulation | B16, V2048, K50, D2048/1024 | 7.514 | 5.260 | 1.43x |
| Prefiltered sampling | B16, K50 | 2.021 | 1.938 | 1.04x |
| Prefiltered sampling | B16, K1024 | 4.533 | 2.877 | 1.58x |
| SnakeBeta, BCT storage | 4,64,16384 | 8.128 | 5.312 | 1.53x |
| SnakeBeta, BTC storage | 4,64,131072 | 73.827 | 37.630 | 1.96x |

The greedy baseline is **Torch argmax followed by Triton gather**, not a single
Triton kernel. SnakeBeta prepares immutable coefficients and its sine table outside
timing; Triton recomputes coefficients per call. Gather, greedy, prefiltered and
SnakeBeta outputs match exactly on these inputs. A separate multi-seed check matched
160/160 draws for both K50 and K128. K16 matched 137/160; all 23 differing token IDs
had equal BF16 logits. K16 is excluded from the table because its tie ordering is
not a drop-in match.
The full B1/B16/B32 sweep is saved in `tmp/qtts_aux_h200_triton_comparison.json`.

The sampling rows above predate overlapping Gumbel generation with merge/softmax.
An interleaved H200 A/B test (Torch 2.13.0+cu132, same timing method, eight warps)
measured the scheduling change alone:

| Operation | Shape | Before (us) | After (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| Sampling | B16, K50 | 4.609 | 3.613 | 1.28x |
| Sampling + both lookups + accumulation | B16, K50 | 5.212 | 4.238 | 1.23x |
| Sampling | B16, K128 | 6.191 | 4.355 | 1.42x |
| Sampling + both lookups + accumulation | B16, K128 | 6.608 | 4.924 | 1.34x |

All 48 combinations of B1/16/32, K1/16/50/128, four/eight warps, and optional
embedding updates matched the previous kernel exactly across five changed-input
draws per case. Results: `tmp/qtts_sampling_overlap_bench.json`.
The updated kernel passes memcheck/racecheck and the changed-input graph tests.
K50 SASS uses 30 registers/thread with no spills and retains 128-bit transfers
and packed BF16 embedding arithmetic.

The standalone embedding kernel now assigns projected lookups to separate CTAs
in the same launch. On H200, B16/D2048/DP1024 improves from 1.39 to 1.17 us
(32 to 48 CTAs at 128 threads); issuing both lookups early in the original CTAs
takes 1.25 us. Unprojected lookup timings are unchanged within measurement noise.
Interleaved A/B results: `tmp/qtts_embedding_partition_bench.json` and
`tmp/qtts_embedding_partition_tuning.json`. The older tables above predate this change.

## Talker Validation

The Torch reference follows upstream Qwen-TTS BF16 rounding; the compiled
baseline enables `emulate_precision_casts`. Two-layer prefill/decode checks
with real weights match upstream exactly. Individual fused blocks pass their
tolerance checks on random inputs, non-unit gamma, zeros, and checkpoint layers.
QKV's isolated RoPE check is bitwise exact. Changed-input/position graph replays
exercise cache writes and workspace reuse.

H200 checks reported clean memcheck/racecheck for pre/post blocks, including
non-bucket batches and changed-input graph replays.
The RoPE-cache cleanup preserves before/after eager and fused logits and KV
caches bitwise at B16/32, including changing-position CUDA-graph replay.

Full-backbone pointwise differences exceed standalone tolerances, including for
compiled Torch. On one repeated prompt with 32 teacher-forced decode steps,
fused QKV+post had 100% top-token agreement and mean/worst relative L2 logit error
of 0.921%/2.395%. This does not establish free-running speech quality.

Both model benchmarks report numerical differences and top-code agreement,
rejecting shape mismatches and non-finite values. Standalone kernels retain
tolerance-based assertions. Free-running speech-quality validation is outside
the scope of these standalone benchmarks.

## Representative Results

H200, BF16, unprofiled interleaved CUDA graphs. Standalone blocks rotate 28
weight sets. These are selected measurements from development, not a fresh run.

| Block | Batch | Compiled Torch | Fused | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QKV preparation | 16 | 22.57 us | 5.99 us | 3.77x |
| QKV preparation | 32 | 22.88 us | 8.85 us | 2.59x |
| Post-attention, warp-local norm | 16 | 27.14 us | 12.22 us | 2.22x |
| Post-attention, warp-local norm | 32 | 27.84 us | 16.64 us | 1.67x |

QKV's baseline compiles arithmetic but uses direct in-place Torch cache-slot
updates. Fully compiling `index_copy_` introduced full-cache copies and measured
30.14/39.69 us instead, so it is not the standalone comparison above.

Full fused talker after caching RoPE: **0.646 ms at B16, 0.927 ms at B32**.
Real weights, 23-token prefix and 32 captured decode inputs; timing the last
step. B32 duplicates the B16 inputs. The paired before/after run measured
0.663 -> 0.646 ms and 0.942 -> 0.927 ms. The 16 setup kernels became one lookup.
The compiled full-backbone baseline still includes the cache copies described
above; speedup is not solely faster arithmetic. These are component timings,
not end-to-end generation speedups.

### Initial Blackwell Port

B300 (SM103), CUTLASS DSL 4.8.0, BF16, unprofiled interleaved CUDA graphs.
Standalone blocks rotate 28 weight sets; model runs use the real checkpoint
and captured inputs described above.

| Workload | Batch | Compiled Torch | Fused | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QKV preparation | 16 | 24.33 us | 5.31 us | 4.59x |
| QKV preparation | 32 | 22.05 us | 6.73 us | 3.27x |
| Post-attention | 16 | 23.98 us | 11.73 us | 2.04x |
| Post-attention | 32 | 23.54 us | 21.06 us | 1.12x |
| Talker decode step | 16 | 1.572 ms | 0.622 ms | 2.53x |
| Predictor frame | 16 | 4.193 ms | 1.967 ms | 2.13x |
| Predictor frame | 32 | 4.291 ms | 2.826 ms | 1.52x |

These measurements are from `5d0ecc7`, before the scheduling changes below.
Standalone pre/post and selection checks pass, as do
the moved Hopper blocks' B16/B32 regression checks. B300 pre/post memcheck and
racecheck report zero errors/hazards at B1/16/17/32 with changed-input/position
graph replays. B200 dispatch is implemented but has not been tested.

Talker top-token agreement is 100% over the 32 captured teacher-forced steps.
Predictor B16/B32 teacher-forced agreement is 97.9%/96.7%, and free-running code
agreement is 91.3%/84.0%. These are numerical comparisons, not speech-quality
validation; the same full-backbone cache-copy caveat applies to these timings.

### Scheduling Optimization

Paired before/after measurements against `5d0ecc7`, using unprofiled interleaved
CUDA graphs. Post-attention rotates 28 weight sets; model inputs and timing
boundaries are as described above. Values are medians of three timing rounds.

| GPU | Workload | Batch | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| B300 | Post-attention | 16 | 11.78 us | 10.81 us | 1.09x |
| B300 | Post-attention | 32 | 21.08 us | 13.35 us | 1.58x |
| H200 | Post-attention | 16 | 11.62 us | 11.68 us | 1.00x |
| H200 | Post-attention | 32 | 16.54 us | 15.36 us | 1.08x |
| B300 | Talker decode step | 16 | 0.626 ms | 0.587 ms | 1.07x |
| B300 | Talker decode step | 32 | 0.980 ms | 0.750 ms | 1.31x |
| B300 | Predictor frame | 16 | 1.988 ms | 1.881 ms | 1.06x |
| B300 | Predictor frame | 32 | 2.829 ms | 2.289 ms | 1.24x |
| H200 | Talker decode step | 32 | 0.929 ms | 0.899 ms | 1.03x |
| H200 | Predictor frame | 32 | 2.751 ms | 2.659 ms | 1.03x |

Both architectures now distribute gate/up across all 96 CTAs in one wave.
Blackwell also uses warp-uniform startup-hint polling; Hopper retains its
single-lane poll. Descriptor construction moves out of the K64 loops.
H200 B16 model timings remain essentially unchanged.

Before/after kernel outputs are bitwise identical at B1/16/17/32. Full-model
logits and KV caches, plus predictor codes and feedback, are bitwise identical
at B16/32 on both GPUs. Pre/post memcheck and racecheck pass on both GPUs with
changed-input/position graph replays. These checks establish preservation of the
previous fused implementation, not equivalence to eager Torch or speech quality.

### 1.7B Talker

The optimized schedule replaces two M64 channel waves with one M128 wave,
streams normalization into operand buffers, and coalesces activation stores.
Blackwell additionally releases operand buffers directly from MMA completion.
Paired measurements against the initial support commit `a7069ba`, rotating
28 weight sets in interleaved CUDA graphs:

| GPU | Batch | Post-attention before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| H200 | 16 | 36.00 us | 30.09 us | 1.20x |
| H200 | 32 | 45.35 us | 35.99 us | 1.26x |
| B300 | 16 | 32.68 us | 24.40 us | 1.34x |
| B300 | 32 | 44.34 us | 28.78 us | 1.54x |

Real CustomVoice checkpoint, synthetic embeddings, 28 layers, B16/B32,
128-token prefix and three teacher-forced decode steps. Unprofiled, interleaved
CUDA graphs time the final fixed-position decode step:

| GPU | Batch | Compiled Torch | Fused before | Fused after | Compiled/after |
| --- | ---: | ---: | ---: | ---: | ---: |
| H200 | 16 | 2.553 ms | 1.565 ms | 1.361 ms | 1.88x |
| H200 | 32 | 3.168 ms | 2.100 ms | 1.762 ms | 1.80x |
| B300 | 16 | 2.222 ms | 1.374 ms | 1.064 ms | 2.09x |
| B300 | 32 | 2.399 ms | 1.863 ms | 1.563 ms | 1.53x |

The full-backbone compiled baseline still includes its cache-copy overhead;
these speedups are not all attributable to faster arithmetic. Before/after
logits and caches match bitwise on B300. H200 runs can differ first in the
unchanged SDPA call despite identical query/cache inputs; the latest B32 run
has 0.33-0.50% logit relative-L2 difference. Isolated post-attention blocks
fed identical checkpoint activations match bitwise.

Random-input standalone checks pass at the existing tolerances. On H200,
isolated post-attention blocks using all 28 checkpoint layers' decode inputs
exceed `atol=rtol=0.02` for 48/917,504 values at B16 and 76/1,835,008 at B32;
relative-L2 errors are 0.061%/0.058%. FP64 projection reductions with the same
BF16 boundaries also exceed that tolerance (65 and 113 values, respectively), consistent with
rounding sensitivity rather than establishing eager Torch as an exact oracle.
Checkpoint QKV checks pass. Full-talker logit relative-L2 differences are about
0.9-1.2%, with top-token agreement of 93.8-100% on these synthetic inputs.
Real-prompt generation and speech quality remain unvalidated.

Optimized post-attention memcheck and racecheck pass on H200/B300 at
B1/16/17/32 with changed-input graph replays. All 28 rotating-weight blocks
remain bitwise identical to `a7069ba`; H200 checks using checkpoint-layer
activations also match bitwise. H200/B300 SASS has no register spills at B16/B32.
The 0.6B device kernels were not modified; paired B16/B32 checks on both GPUs
remain bitwise identical with unchanged measured latency.

## Profiling

`--profile` emits CUDA-profiler capture ranges and named NVTX graph replays.
For scheduling analysis use graph-level Nsight capture; for kernel attribution
use graph-node capture. Keep traces directly under repository `tmp/`, with model,
GPU, batch, and variant in the filename.

Graph-node tracing can substantially inflate predictor timings and introduce GPU
gaps during `cudaGraphLaunch`. Use unprofiled interleaved CUDA-event timings for
speedup claims.

Pre/post kernels have IKET per-warp markers; `--trace-only --batch-size 16`
launches one custom block without reference GEMMs or timing loops:

```bash
run-iket --output-dir /path/to/repo/tmp profile --postprocess all \
  --max-ts-cnt-per-warp 256 --keep -- \
  python mega_qwen3_tts/kernels/post_attention.py --batch-size 16 --trace-only
```

Useful phases:

| Markers | Meaning |
| --- | --- |
| `qkv_weight_prefetch`, `input_norm`, `qkv_operand_wait`, `qkv_mma` | QKV startup, staging, operand wait, compute |
| `qkv_partial_publish`, `qkv_partial_wait`, `head_norm_rope_cache` | Head reduction and epilogue |
| `output_projection`, `gate_up_projection`, `down_projection` | Post-attention GEMM regions, including nested waits |
| `splitk_store`, `splitk_receive`, `output_reduce`, `down_reduce` | Mailbox transfer/completion, then arithmetic reduction |
| `gate_up_prefetch_wait`, `normalization_input_wait`, `norm_local_stats`, `norm_publish` | Scheduling hint, payload polling, RMS statistics, operand publication |
| `activation_wait`, `silu_and_gate`, `down_cluster_release` | SwiGLU handoff and final DSMEM lifetime rendezvous |

IKET ranges include waits and instrumentation overhead. Nested or overlapping
warp ranges cannot be summed into kernel time. Use unprofiled A/B timing for performance.

Auxiliary kernels also have IKET markers: `logits_load_keys`, `warp_topk`,
`candidate_handoff`, `merge_topk`, `softmax`, `top_p_gumbel`, `sample_handoff`,
`embedding_load`, and `embedding_store_add`. Greedy and prefiltered sampling
mark their reductions; SnakeBeta marks coefficient loading, sine and final stores.
The markers are inactive in normal compilation.

H200 captures in `tmp/` (2026-10-22):

- `qwen3_tts_aux_h200_b16_cta0_iket_20261022.pftrace`: one CTA per variant;
  B16 selection/embedding and B4/C64/T16384 SnakeBeta, both direct sine and lookup.
- `qwen3_tts_sampling_h200_b32_all_ctas_iket_20261022.pftrace`: all CTAs,
  K50 sampling with/without embedding and prefiltered K50/K1024 sampling.

These combined captures include seconds of setup between microsecond launches;
at full zoom the kernel events are almost invisible. Prefer the per-launch exports
`qwen3_tts_<operation>_h200_<shape>_iket_20261022.pftrace` in `tmp/`, for example
`qwen3_tts_sampling_k50_embedding_h200_b32_all_ctas_iket_20261022.pftrace`.
They preserve the original events/timestamps but omit other launches and unused
tracks, so each opens at microsecond scale.

The captures above predate the sampling overlap change. Sampling now computes
rank-based FP64 noise in helper warps (`gumbel_prepare`) while warp 0 merges
candidates and computes softmax. After `gumbel_ready`, warp 0 applies top-p and
selects the winner. Other warps then gather embeddings; their `sample_handoff`
wait overlaps warp 0's work and is not an additional independent cost.
The updated single-launch trace is
`tmp/qwen3_tts_sampling_k50_embedding_h200_b16_overlap_iket_20261022.pftrace`.
