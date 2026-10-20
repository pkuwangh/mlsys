# Qwen3-TTS Talker

Standalone 0.6B talker backbone: prepared embeddings -> transformer layers ->
codec logits. This does not run text preprocessing, the code predictor,
sampling, or the waveform decoder.

## Run

Use a Hopper GPU and the `mlsys-gpu` environment with CUTLASS DSL experimental
primitives. No SGLang runtime is required.

```bash
# Complete fused block, including correctness checks and compiled Torch baseline.
python mega_qwen3_tts/kernels/post_attention.py

# Rotate 28 independent weight sets instead of keeping one block hot in L2.
python mega_qwen3_tts/kernels/post_attention.py --layers 28

# Input norm, QKV projection, head norms, RoPE, and cache writes.
python mega_qwen3_tts/kernels/pre_attention.py

# Full backbone; omit --model for deterministic random weights.
python bench_qwen3_tts.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Backbone defaults: BF16, B=16/32, 28 layers, 128 prefix tokens. The custom path uses
compiled Torch for the remaining decode operations and eager Torch for prefill.
`--no-fused` tests the Torch scaffold; `--no-fused-qkv` keeps only post-attention
fusion; `--layers 2` shortens development runs.
Inputs are synthetic embeddings even when real checkpoint weights are loaded.
For captured inputs, add `--inputs /path/to/inputs.pt --batch-size 16`. The file
contains `prefix` and `decode` embedding tensors shaped `[B,T,1024]`, generated
with the same checkpoint. The benchmark replays all decode steps with separate
KV caches; it does not feed the custom model's sampled tokens back into itself.

Timings use interleaved CUDA-graph replays: three samples, five replays each.
Compilation, allocation, weight packing, and cache initialization are excluded.
The full-backbone timing overwrites one fixed KV slot at a fixed context length.
`--profile` captures graph replays inside an NVTX range named `benchmark`.

## Fused QKV Preparation

`FusedPreAttention` replaces input RMSNorm, Q/K/V projections, Q/K head
RMSNorm, RoPE, and the current KV-cache slot write in one launch. Torch SDPA
still computes attention. Multi-token prefill uses the original Torch path.

There are 32 four-CTA clusters: one per Q/K/V head (16 Q, 8 K, 8 V).
Within a cluster, ranks 0/1 project channels 0-63 and ranks 2/3 project
channels 64-127; each pair splits the 1024-channel input reduction into
two K512 halves. Each CTA issues an M64 x N16/32 x K512 WGMMA sequence.

| Phase | Work |
| --- | --- |
| Startup | Thread 0 starts the 64-KiB weight TMA; all 16 warps normalize input rows and stage this CTA's K512 slice |
| Projection | Warps 4-7 issue WGMMA and write FP32 partials to SMEM |
| Reduction | Ranks 1-3 bulk-copy their partials to rank 0's SMEM; rank 0 writes its own partials locally |
| Epilogue | Rank 0's 16 warps sum the two K partitions, apply Q/K RMSNorm and RoPE, and write Q or the KV-cache slot |

A transaction barrier counts only the three remote copies. No global scratch,
Lamport polling, or cooperative grid launch is needed for this block.
Input norm is deliberately recomputed per CTA to avoid a global handoff.

Supported: Hopper, BF16, B=1..32, the 0.6B dimensions, one shared decode
position across the batch, and contiguous cache `[2,B,8,capacity,128]`.
RoPE uses the same position on all three axes, matching ordinary TTS decoding.
Weights are packed once and must remain unchanged.

FP32 GEMM/reduction order can differ from Torch, but BF16 boundaries are
preserved. In particular, `--fmad=false` prevents RoPE's separately rounded
products from contracting into BF16 FMA. The standalone check isolates RoPE
and requires exact agreement there, independently of GEMM differences.

## Fused Post-Attention

`FusedPostAttention` replaces the whole post-attention sequence in one launch:

1. Attention output projection and residual addition.
2. RMSNorm, gate/up projections, SiLU, and gating multiplication.
3. Down projection and the final residual addition.

The reference preserves upstream BF16 rounding boundaries, including the cast
between normalization and gamma multiplication, and between SiLU and gating.
The compiled baseline enables `emulate_precision_casts` for the same reason.
Matmuls accumulate in FP32, but their reduction order can differ from cuBLAS.

The kernel uses cooperative four-CTA clusters, with eight warps per CTA:

| Batch | Output projection | Gate/up | Down projection |
| --- | --- | --- | --- |
| 1-16 | First 64 CTAs, four-way split-K | All 96 CTAs, one tile each | First 64 CTAs, four-way split-K |
| 17-32 | First 64 CTAs, four-way split-K | Last 48 of 112 CTAs, two tiles each | First 64 CTAs, overlapping the second gate/up wave |

Gate/up does not split K. Warps 0-3 stage output/down operands; warps 4-7
issue WGMMA. Output projection has two K256 stages and down has two K384
stages. These use distinct buffers, so neither needs an empty-buffer handshake.
Gate/up weights occupy 128 KiB of SMEM per tile, loaded in one full-K TMA
transfer before normalization. At N16, CTAs 64-95 wait for a producer's first
output-projection MMA stage, then prefetch their own tiles into SMEM and the
other 64 tiles into L2. The progress hint does not publish residual data or need
a release fence. N32 waits for an existing residual word before its first TMA
and prefetches its second weight tile during SiLU processing. Extra weight storage aliases the output/down
operand buffers.

Normalized inputs remain in WGMMA layout. Split-K partials are pushed into each
reducer's SMEM mailbox with `st.async`; a transaction barrier signals completion.
Gate/up epilogues use a separate swizzled layout. At N16 their storage cannot
alias the incoming down-projection mailbox: peers can reach that phase earlier.

Both residuals and gate/up activations use packed BF16 sentinel polling:
consumers use the words they polled, without a separate flag or release fence.
Normalization retains residuals and gamma in registers. Each warp owns a full
1024-channel row, reduces statistics with shuffles, and writes normalized data
directly into the WGMMA operand layout. Only the final CTA rendezvous remains;
there are no shared partial sums or inverse-RMS broadcasts. BF16 casts are
unchanged, but the FP32 summation order differs from the earlier kernel.
Every payload word is validated, regardless of the prefetch hint. Two slots per
payload alternate between calls; each call clears the inactive slots for the next one.
Live negative zero is canonicalized to positive zero to reserve the sentinel.
Each instance owns its workspace, which must not be used concurrently on
different streams. The cooperative launch ensures cross-cluster forward progress.

Supported: SM90, BF16, B=1..32, hidden=1024, attention width=2048, FFN=3072,
no projection biases. B<=16 uses an N16 MMA bucket; larger batches use N32.
Weights are packed once and must remain unchanged. Larger batches, the 1.7B
dimensions, training, and non-Hopper GPUs are not supported by this kernel.

## Validation Status

The initial implementation was tested on H100; these optimizations were tested
on H200:

- QKV preparation passes random/non-unit-gamma and zero-input checks, changed
  inputs/positions under graph replay, and all 28 checkpoint layers on three
  actual decode inputs at B16/B32. The latter has relative L2 error 0.0095%.
  Memcheck is clean at B=1/8/16/17/32; racecheck is clean at B=1/16/32.
  These are layer-local checks, not whole-model quality validation.
- Two-layer reference matches upstream decoder-layer code exactly, including
  prefill and three decode steps with real checkpoint weights and KV caches.
- Fused block passes random inputs, non-unit gamma, zero inputs, changed inputs
  under graph replay, and layer-local checks using all 28 checkpoint layers.
- Through `465a629`, Compute Sanitizer reports zero memcheck
  errors and zero racecheck hazards for B=1/8/16/17/32 with changed-input graph
  replay. Results match the original fused kernel bit-for-bit.
- At that commit, a real-checkpoint 32-step backbone replay at B=16/32 also produces bitwise
  identical old/new logits and KV caches. B=32 repeats the same captured inputs;
  this checks the optimization, not diverse-request quality.
- The warp-local RMSNorm changes the FP32 sum order. Block tolerance checks
  still pass, but this newer version is **not bitwise equivalent** to `465a629`:
  the same replay changes one of 32 B16 top-token decisions (none at B32).
  Mean relative L2 error against eager is 0.913% / 0.927% at B16/B32, versus
  0.952% / 0.923% before. This is a numerical tradeoff, not a generation-quality
  validation; old-order-preserving alternatives are recorded in the profiling notes.

**Full-backbone numerical validation is not yet passing the strict pointwise
tolerance.** Small matmul rounding differences can amplify through 28 layers;
a Torch control using FP64 projection matmuls also diverges from eager BF16.
On one real text prompt repeated across B=16, a 32-step teacher-forced replay
using upstream code-predictor outputs had 100% top-token agreement. With the
remaining decode operations compiled, mean relative L2 logit error was 0.92%
(2.24% worst step), versus 0.85% for compiled Torch (1.97% worst step). Compiled
Torch also fails the pointwise tolerance. This is not a free-running
generation-quality validation or a diverse-request test.

The backbone benchmark prints those failures, continues to collect diagnostic
timings, then exits unsuccessfully. It does not relax the tolerance or claim a
validated end-to-end speedup. Real-request replay and generation-quality checks
remain necessary before integrating this kernel into a serving engine.

## Performance

QKV preparation on H200 (`viking-prod-214`), BF16, 28 rotating weight sets,
interleaved CUDA-graph timing:

| Batch | Eager Torch | Compiled full | Compiled slots | Fused QKV | Speedup vs slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 77.47 us | 30.14 us | 22.57 us | 5.99 us | 3.77x |
| 32 | 79.93 us | 39.69 us | 22.88 us | 8.85 us | 2.59x |

The timings include both normalization stages, QKV projection, RoPE, and
cache writes, but exclude SDPA and weight packing.
Full compilation introduces full-cache copies when functionalizing
`index_copy_`. The **compiled slots** control compiles the arithmetic and
uses direct in-place Torch slot updates. Both run inside CUDA graphs, so
Python dispatch overhead is excluded.

Full-talker comparison on the same H200, using real weights and one repeated
prompt: a 23-token prefix plus 32 captured decode inputs. Median of three interleaved rounds,
timing the last decode step:

| Batch | Compiled Torch | Post-attention fusion only | QKV + post-attention fusion |
| --- | ---: | ---: | ---: |
| 16 | 1.728 ms | 1.327 ms | 0.686 ms |
| 32 | 1.995 ms | 1.643 ms | 0.972 ms |

These are **diagnostic timings**, not validated generation speedups. The
combined path has 100% top-token agreement on this teacher-forced replay,
with mean/worst-step relative L2 logit error 0.921%/2.395%. Strict pointwise
checks still fail, as they do for compiled Torch. B32 duplicates the B16
inputs. Prefill logits and caches remain bitwise identical to eager.
The full-talker Torch baseline still includes the cache copies above; part
of the end-to-end gain comes from eliminating them, not just faster GEMMs.

Post-attention optimization history follows.

H200 (`s4124-0042`), BF16, 28 rotating weight sets (616 MiB), interleaved
CUDA-graph timing without profiling. Median of three measurement rounds;
each round uses three samples of five replays. Per complete fused block:

| Batch | Compiled Torch | Original fused | Optimized fused | Speedup vs Torch | Latency reduction vs original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 27.19 us | 18.69 us | 16.22 us | 1.68x | 13.3% |
| 32 | 27.86 us | 23.54 us | 19.49 us | 1.43x | 17.2% |

Original is commit `f0a74e6`; optimized is `c03b2bc`. These are standalone-block results, not whole-TTS
speedups. The initial H100 results were 20.87/25.87 us for the original kernel;
do not compare those directly with H200 timings.

Follow-up on H200 (`viking-prod-260`), with the same interleaved protocol:

| Batch | Compiled Torch | `c03b2bc` | Earlier weight/norm prefetch | Speedup vs Torch |
| --- | ---: | ---: | ---: | ---: |
| 16 | 27.11 us | 16.24 us | 14.64 us | 1.85x |
| 32 | 27.81 us | 19.49 us | 19.54 us | 1.42x |

B16 latency falls another 9.9%; B32 is effectively unchanged. That prefetch
change preserved the CTA roles, math, and cross-CTA publication protocol.

The subsequent mailbox/sentinel handoff change, measured on the same H200:

| Batch | Compiled Torch | `9465e8b` | Current | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 16 | 27.14 us | 14.59 us | 13.50 us | 7.5% |
| 32 | 27.76 us | 19.53 us | 18.38 us | 5.9% |

These timings include the inactive activation-slot reset on every invocation.

Converting the remaining residual handoff to sentinel polling gives a small
additional improvement in a fresh interleaved A/B run:

| Batch | Compiled Torch | Before | Both handoffs sentinel-based |
| --- | ---: | ---: | ---: |
| 16 | 27.12 us | 13.48 us | 13.31 us |
| 32 | 27.76 us | 18.27 us | 18.08 us |

These include both inactive-slot resets and local norm-statistics computation.

The warp-local RMSNorm and progress-triggered prefetch follow-up, against
`465a629` on the same H200 (median of three interleaved rounds):

| Batch | Compiled Torch | Before | Current | Latency reduction | Speedup vs Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 27.14 us | 13.30 us | 12.22 us | 8.1% | 2.22x |
| 32 | 27.84 us | 18.16 us | 16.64 us | 8.4% | 1.67x |

No fixed delay is used. The changed norm summation order has the numerical
tradeoff described above; these are block timings, not validated TTS speedups.

Real-checkpoint full-talker replay, 23-token prefix plus 32 captured decode
inputs, timing the last step at a fixed KV position:

| Batch | Compiled Torch | Original fused path | Optimized fused path |
| --- | ---: | ---: | ---: |
| 16 | 1.786 ms | 1.593 ms | 1.514 ms |
| 32 | 2.077 ms | 1.994 ms | 1.862 ms |

These remain **diagnostic timings**: both compiled Torch and the fused path
fail the strict full-backbone pointwise tolerance described above. This excludes
the code predictor, sampling, and vocoder.

See [profiling notes](profiling.md) for NCU/IKET findings, rejected experiments,
and trace paths. To reproduce the current standalone measurement:

```bash
python mega_qwen3_tts/kernels/post_attention.py --layers 28
```
