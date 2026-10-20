# Qwen3-TTS Talker

Standalone 0.6B talker backbone: prepared embeddings -> 28 transformer layers
-> codec logits. This excludes text preprocessing, the code predictor, sampling,
and waveform decoding. See [design.md](design.md) for the math, CTA/warp schedules,
memory handoffs, and retained design decisions.

## Run

Use a Hopper GPU and the `mlsys-gpu` environment with CUTLASS DSL experimental
primitives. No SGLang runtime is required. Run from this directory:

```bash
# Complete post-attention block, correctness checks, and compiled Torch baseline.
# Rotate 28 weight sets so the benchmark does not keep just one layer hot in L2.
python mega_qwen3_tts/kernels/post_attention.py --layers 28

# Input norm, QKV, head norms, RoPE, cache writes, and optional additive mask.
python mega_qwen3_tts/kernels/pre_attention.py --write-mask

# Full backbone; omit --model for deterministic random weights.
python bench_qwen3_tts.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Backbone defaults: BF16, B=16/32, 28 layers, 128 prefix tokens. Multi-token
prefill stays eager; custom decode uses fused QKV preparation, cuDNN attention
through Torch SDPA, fused post-attention, and compiled remaining Torch operations.
The specialization supports B=1..32, one shared decode position, and immutable
weights. It does not support 1.7B dimensions, training, or non-Hopper GPUs.

Useful controls:

- `--no-fused`: compiled Torch scaffold.
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

## Validation

The Torch reference follows upstream Qwen-TTS BF16 rounding; the compiled
baseline enables `emulate_precision_casts`. Two-layer prefill/decode checks
with real weights match upstream exactly. Individual fused blocks pass their
tolerance checks on random inputs, non-unit gamma, zeros, and checkpoint layers.
QKV's isolated RoPE check is bitwise exact. Changed-input/position graph replays
exercise cache writes and workspace reuse.

H200 checks reported clean memcheck/racecheck for pre/post blocks, including
non-bucket batches and changed-input graph replays.
The reference cleanup was also checked with the full 28-layer checkpoint:
before/after logits and KV caches match bitwise for eager prefill/decode at
B1/16/17/32 and compiled Torch/fused graph replay at B16/32.

**Strict full-backbone pointwise validation still fails**, including for compiled
Torch. Preserving BF16 boundaries does not preserve FP32 reduction order, and
small differences can amplify through 28 layers. On one repeated prompt with
32 teacher-forced decode steps, fused QKV+post had 100% top-token agreement,
but mean/worst relative L2 logit error was 0.921%/2.395%. An earlier norm-order
change altered one of 32 top-token decisions; neither result establishes
free-running speech quality.

The backbone benchmark reports failures, collects diagnostic timings, and exits
unsuccessfully. Diverse-request and free-running quality validation are still
required before serving integration.

## Representative Results

H200, BF16, unprofiled interleaved CUDA graphs. Standalone blocks rotate 28
weight sets. These are selected measurements from development, not a fresh run.

| Block | Batch | Compiled Torch | Fused | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QKV preparation | 16 | 22.57 us | 5.99 us | 3.77x |
| QKV preparation | 32 | 22.88 us | 8.85 us | 2.59x |
| Post-attention, warp-local norm | 16 | 27.14 us | 12.22 us | 2.22x |
| Post-attention, warp-local norm | 32 | 27.84 us | 16.64 us | 1.67x |

The retained early-weight-TMA follow-up reduced B16 post-attention from
12.15 to 11.71 us in a separate paired run; it did not improve B32.
QKV's baseline compiles arithmetic but uses direct in-place Torch cache-slot
updates. Fully compiling `index_copy_` introduced full-cache copies and measured
30.14/39.69 us instead, so it is not the standalone comparison above.

Full backbone: real weights, one repeated prompt, 23-token prefix plus 32 captured
decode inputs; timing the last step. B32 duplicates the B16 inputs.

| Batch | Compiled Torch | Post fusion only | QKV + post fusion | With fused mask, later paired run |
| --- | ---: | ---: | ---: | ---: |
| 16 | 1.728 ms | 1.327 ms | 0.686 ms | 0.664 ms |
| 32 | 1.995 ms | 1.643 ms | 0.972 ms | 0.946 ms |

The mask A/B itself measured 684.6 -> 663.7 us at B16 and 965.2 -> 945.7 us at
B32, with bitwise-identical logits/caches. It removes one conversion kernel per
layer. The compiled full-backbone baseline still includes the cache copies
described above: its speedup is not solely faster arithmetic. All backbone
numbers are **diagnostic timings, not validated generation speedups**.

## Profiling

`--profile` emits CUDA-profiler capture ranges and named NVTX graph replays.
For scheduling analysis use graph-level Nsight capture; for kernel attribution
use graph-node capture. Keep traces directly under repository `tmp/`, with model,
GPU, batch, and variant in the filename.

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

Selected existing captures in `tmp/` (local artifacts, not tracked):

- `qwen3_tts_0p6b_h200_b16_qkv_post_comparison.1.nsys-rep`: backbone comparison.
- `qwen3_tts_0p6b_h200_b16_fused_mask_ab.1.nsys-rep`: boolean versus fused mask.
- `qwen3_tts_0p6b_h200_b{16,32}_pre_attention_iket.pftrace`: QKV phases.
