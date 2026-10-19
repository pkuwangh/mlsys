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

# Full backbone; omit --model for deterministic random weights.
python bench_qwen3_tts.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Backbone defaults: BF16, B=16/32, 28 layers, 128 prefix tokens. The custom path uses
compiled Torch for the remaining decode operations and eager Torch for prefill.
`--no-fused` tests the Torch scaffold; `--layers 2` shortens development runs.
Inputs are synthetic embeddings even when real checkpoint weights are loaded.
For captured inputs, add `--inputs /path/to/inputs.pt --batch-size 16`. The file
contains `prefix` and `decode` embedding tensors shaped `[B,T,1024]`, generated
with the same checkpoint. The benchmark replays all decode steps with separate
KV caches; it does not feed the custom model's sampled tokens back into itself.

Timings use interleaved CUDA-graph replays: three samples, five replays each.
Compilation, allocation, weight packing, and cache initialization are excluded.
The full-backbone timing overwrites one fixed KV slot at a fixed context length.
`--profile` captures graph replays inside an NVTX range named `benchmark`.

## Fused Block

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
Gate/up weights occupy 128 KiB of SMEM per tile: four independently ready K256
stages at N16, or one full-K TMA transfer at N32. N32 prefetches the first tile
while waiting for normalization inputs and the next tile during SiLU processing.
Its extra weight storage aliases the output/down operand buffers.

Normalized inputs remain in WGMMA layout. Split-K partials and gate/up epilogues
use different shared-memory swizzles to suit their different read patterns.
Epoch-tagged GPU release/acquire flags connect the cluster groups. Each instance
owns its workspace, which must not be used concurrently on different streams.
The cooperative launch is required for cross-cluster forward progress.

Supported: SM90, BF16, B=1..32, hidden=1024, attention width=2048, FFN=3072,
no projection biases. B<=16 uses an N16 MMA bucket; larger batches use N32.
Weights are packed once and must remain unchanged. Larger batches, the 1.7B
dimensions, training, and non-Hopper GPUs are not supported by this kernel.

## Validation Status

The initial implementation was tested on H100; these optimizations were tested
on H200:

- Two-layer reference matches upstream decoder-layer code exactly, including
  prefill and three decode steps with real checkpoint weights and KV caches.
- Fused block passes random inputs, non-unit gamma, zero inputs, changed inputs
  under graph replay, and layer-local checks using all 28 checkpoint layers.
- After the scheduling/layout changes, Compute Sanitizer reports zero memcheck
  errors and zero racecheck hazards for B=1/8/16/17/32 with changed-input graph
  replay. Results match the original fused kernel bit-for-bit.
- A real-checkpoint 32-step backbone replay at B=16/32 also produces bitwise
  identical old/new logits and KV caches. B=32 repeats the same captured inputs;
  this checks the optimization, not diverse-request quality.

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

H200 (`s4124-0042`), BF16, 28 rotating weight sets (616 MiB), interleaved
CUDA-graph timing without profiling. Median of three measurement rounds;
each round uses three samples of five replays. Per complete fused block:

| Batch | Compiled Torch | Original fused | Optimized fused | Speedup vs Torch | Latency reduction vs original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 27.19 us | 18.69 us | 16.22 us | 1.68x | 13.3% |
| 32 | 27.86 us | 23.54 us | 19.49 us | 1.43x | 17.2% |

Original is commit `f0a74e6`. These are standalone-block results, not whole-TTS
speedups. The initial H100 results were 20.87/25.87 us for the original kernel;
do not compare those directly with H200 timings.

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
