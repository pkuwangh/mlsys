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

Defaults: BF16, B=16/32, 28 layers, 128 prefix tokens. The custom path uses
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

The kernel launches 28 cooperative four-CTA clusters, with eight warps per CTA:

| Clusters | Work |
| --- | --- |
| 0-15 | Split-K output projection, DSMEM reduce-scatter, publish residual/statistics, then split-K down projection |
| 16-27 | Normalize the published residual, then produce two waves of gate/up activation tiles |

Warps 0-3 load operands; warps 4-7 issue WGMMA. Double-buffered TMA loads
packed weights. Projection and gate/up use K256 stages; down uses K384 stages,
one per activation wave. Twelve loader threads acquire that wave's tile flags
in parallel. Normalized inputs remain in WGMMA layout for both gate/up waves.
Epoch-tagged GPU release/acquire flags connect the cluster groups. Each instance
owns its workspace, which must not be used concurrently on different streams.
The cooperative launch is required for cross-cluster forward progress.

Supported: SM90, BF16, B=1..32, hidden=1024, attention width=2048, FFN=3072,
no projection biases. B<=16 uses an N16 MMA bucket; larger batches use N32.
Weights are packed once and must remain unchanged. Larger batches, the 1.7B
dimensions, training, and non-Hopper GPUs are not supported by this kernel.

## Validation Status

Tested on H100 (`s4124-0130`):

- Two-layer reference matches upstream decoder-layer code exactly, including
  prefill and three decode steps with real checkpoint weights and KV caches.
- Fused block passes random inputs, non-unit gamma, zero inputs, changed inputs
  under graph replay, and layer-local checks using all 28 checkpoint layers.
- Compute Sanitizer: memcheck reports zero errors; racecheck reports zero hazards
  for B=1/17/32 with graph replay.

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

## Initial Performance

H100, BF16, 28 rotating weight sets, interleaved graph timing. Per fused block:

| Batch | Compiled Torch | Custom | Speedup |
| --- | ---: | ---: | ---: |
| 16 | 28.50 us | 20.87 us | 1.37x |
| 32 | 29.16 us | 25.87 us | 1.13x |

Increasing down stages from K128 to K384 reduced latency by approximately
2.4 us at B=16 and 2.0 us at B=32 in the same A/B run. Early gate/up weight
prefetch regressed B=32 and was not retained. These are standalone-block
results, not whole-TTS speedups.

The B=16 captured-input backbone replay measured 1.676 ms custom versus 1.845 ms
compiled Torch without profiling (1.10x). This remains a diagnostic timing:
both paths fail the strict full-backbone pointwise test described above.

The traces report 44/60 registers per thread and 160836/222340 bytes of dynamic
SMEM for B=16/32, with zero per-thread local memory. Active-cluster capacity is
30 for both, exceeding the cooperative grid's 28 clusters.

Local traces with GPU metrics (Nsight Systems 2026.5):

- `tmp/nsys_qwen3_tts_0p6b_h100_post_attention_b16_k384.1.nsys-rep`
- `tmp/nsys_qwen3_tts_0p6b_h100_post_attention_b32_k384.1.nsys-rep`
- `tmp/nsys_qwen3_tts_0p6b_h100_talker_b16_real_replay.1.nsys-rep`

These paths are relative to the repository root. Each report contains named
Torch, Compiled, and Fused/Custom ranges; full-talker results remain diagnostic.
