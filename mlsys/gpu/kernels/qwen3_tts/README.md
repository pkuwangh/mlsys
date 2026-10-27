# Qwen3-TTS Kernels

Standalone 0.6B and 1.7B components:

- **Talker:** prepared embeddings -> 28 transformer layers -> codec logits.
- **Code predictor:** talker hidden state and first code -> 15 remaining codes
  and embedding feedback, reusing the same fused blocks.
- **Vocoder:** convolutional waveform tail, optionally measured with the
  unchanged upstream codec prefix.

These are component benchmarks, not an end-to-end text-to-speech pipeline.
See [design.md](design.md) for transformer math, CTA/warp schedules, memory
handoffs, and retained design decisions.

## Layout

```text
torch_qwen3_tts/       Torch references: talker.py, predictor.py, vocoder.py
mega_qwen3_tts/        Custom modules with the same three components
  kernels/            Kernel wrappers with standalone validation/benchmarks
    sm90/             Hopper device kernels
    sm100/            Blackwell device kernels
bench_*.py            Top-level talker, predictor, and vocoder benchmarks
tests/                Correctness, graph replay, and dynamic-shape tests
```

## Supported Inputs

Use CUTLASS DSL 4.8 experimental primitives on Hopper or datacenter Blackwell
(SM90/SM100/SM103). Device kernels are separate under `kernels/sm90/` and
`kernels/sm100/`; the wrappers select the architecture automatically.

| Component | Supported inputs |
| --- | --- |
| Pre-/post-attention | BF16, one-token decode, B=1..32, hidden/MLP widths 1024/3072 or 2048/6144, 16 Q heads, 8 KV heads, head dimension 128 |
| Conv1D | BF16, contiguous BTC input, packed OWC weights, Ci/Co divisible by 32, stride=1, groups=1, positive dilation |
| Embedding update | Contiguous BF16 tables and accumulator, feature widths divisible by eight; token IDs may be strided |
| Greedy selection | FP16/BF16/FP32 logits, vocabulary up to 8192; supports irregular embedding widths |
| Raw-logit sampling | BF16 `[B,2048]`, top-k up to 128, per-row GPU parameters and seeds |
| Prefiltered sampling | FP32 log probabilities and supplied candidate IDs, up to 2048 candidates |
| SnakeBeta | BF16 `[B,C,T]` with contiguous BCT or BTC storage |
| State copies | Gather/scatter multiple contiguous BF16 buffers by Int64 row indices; row widths divisible by eight |

**Post-attention requires a cooperative launch with all 24 four-CTA clusters
resident at the compiled kernel's actual occupancy.** Architecture support or
raw SM count alone does not establish compatibility on smaller or partitioned
GPUs. There is no noncooperative fallback.

Move reference models to their final CUDA device and BF16 before constructing
fused modules. Packed weights and activation coefficients are immutable;
changing weights, dtype, or device requires rebuilding the fused module.
Norm weights are borrowed tensors, not snapshots; leave the source weights
unchanged too. Input and Q/K head norms use Qwen3-TTS's common RMSNorm epsilon.
Training is unsupported. Calls sharing a post-attention instance must be
serialized, including graph replays; concurrent streams need separate instances.

Compilation uses fake tensors with symbolic live batch sizes. Attention
specializes the 16/32-row schedule and model width, not each batch, KV capacity,
or RoPE table length. Conv1D/SnakeBeta keep batch and audio length dynamic;
channel geometry, tiling, vector width, layout, tail masking, dtype, architecture,
and fusion flags remain specialization keys. Post-attention workspaces are
per-batch allocations, separate from the shared compiled binary.
Warm each batch size before CUDA graph capture so workspace initialization
and compilation stay outside graph replay.

## Run

Run from this directory using the `mlsys-gpu` environment:

```bash
# Individual blocks: correctness checks and compiled Torch comparisons.
python mega_qwen3_tts/kernels/pre_attention.py --write-mask
python mega_qwen3_tts/kernels/post_attention.py --layers 28

# 1.7B talker shapes, without a checkpoint.
python mega_qwen3_tts/kernels/pre_attention.py --hidden-size 2048 --write-mask
python mega_qwen3_tts/kernels/post_attention.py --hidden-size 2048

# Full components; omit --model for deterministic random weights.
python bench_talker.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
python bench_predictor.py --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
python bench_vocoder.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice

# Full codec, including its unchanged prefix; also requires compatible qwen-tts.
python bench_vocoder.py --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice --full

python -m pytest tests/ -q
```

Each kernel file is also runnable, with correctness checks and an interleaved
CUDA-graph benchmark against Torch/compiled Torch:

```bash
python mega_qwen3_tts/kernels/embedding_update.py
python mega_qwen3_tts/kernels/argmax_embedding.py
python mega_qwen3_tts/kernels/sampling.py
python mega_qwen3_tts/kernels/snake_beta.py
python mega_qwen3_tts/kernels/state_copy.py
python mega_qwen3_tts/kernels/conv1d.py --shape 4,122880,96,96,7 --variant snake
python mega_qwen3_tts/kernels/conv1d.py --shape 4,122880,96,96,1 --variant residual
```

## Model Interfaces

### Talker And Predictor

Talker defaults are BF16, B=1/8/16/32, 28 layers, and 128 prefix tokens.
Multi-token prefill stays eager. Custom decode uses fused QKV preparation,
Torch SDPA/cuDNN attention, fused post-attention, and compiled remaining Torch
operations. `--no-fused-qkv` and `--no-fused-mask` isolate individual changes;
`--layers 2` shortens development runs.

Both predictors have five layers with hidden/MLP widths 1024/3072. The 1.7B
predictor additionally projects its 2048-wide embeddings to 1024 in Torch.
One frame includes a two-token prefix, 14 single-token decode steps, greedy
selection, embedding lookup, and the final feedback sum. The prefix remains
Torch; `--no-fused-selection` disables the custom greedy/embedding kernel.

Call `make_cache` before graph capture to allocate KV storage and precompute
RoPE/masks. The reference-model path uses a shared decode position.
`SlotPreAttention` instead accepts per-token positions and caller-owned K/V
buffers `[slots,8,128]`, with a `[positions,128]` RoPE table containing 64 cosines
and 64 sines per row. `PackedPostAttention` accepts weights directly without
requiring a reference model.

Without `--inputs`, embeddings are synthetic even with checkpoint weights.
Talker `--inputs` takes `prefix` and `decode` tensors `[B,T,hidden_size]`;
predictor inputs contain `hidden [B,1,hidden_size]` and `first_code [B,1]` from
the matching talker. Captured talker inputs are teacher-forced. Its benchmark
repeatedly overwrites the final fixed KV slot, not a growing-context loop.

### Vocoder

The custom waveform tail keeps BTC storage between layers. Each residual unit
uses K7+bias+SnakeBeta and K1+bias+residual fusions. Transposed convolutions and
the final one-channel convolution remain channels-last cuDNN, with compiled
Torch post-processing. Boundary layout conversions are included in timings.

Blackwell Conv1D uses persistent M128 tiles, double-buffered TMEM accumulators,
separate MMA/TMA warps, and four epilogue warps. Hopper uses persistent M64 tiles
with alternating WGMMA warpgroups, overlapping one tile's epilogue with the
next tile's MMA. Tiled TMA handles causal padding and dilation by shifting
input coordinates; no im2col buffer is materialized. C96 uses K32 operand
stages; other channel sizes use K64 stages. Blackwell pointwise convolutions
use N64 tiles and four operand stages when the workload fits one SM wave;
larger workloads retain their throughput-oriented tiles, including N96 stores
for C96.

The microbenchmark compares against compiled BTC cuDNN with the same padding
and post-ops. The tail benchmark also measures the original BCT reference.
`--full` includes quantization decoding, the codec transformer, and both
ConvNeXt upsamplers, unchanged across variants. Only the tail is compiled or
replaced; this is not a fully compiled codec baseline. Inputs remain synthetic
features or random codec IDs, not generated speech.

### Auxiliary Semantics

Embedding updates use vectorized transfers and BF16 accumulation. Optional
projected embeddings come from a precomputed table, not a fused projection
GEMM. Raw-logit sampling fuses top-k, temperature, top-p, seeded Gumbel selection,
and optional embedding updates. Equal logits use ascending token IDs, not
Torch's unspecified `topk` tie order; cutoff rounding can also affect same-seed
parity. Prefiltered sampling preserves the supplied candidate order.

Gumbel draws remain FP64 to match SGLang-Omni's CUDA/Triton sampler, including
its zero/one hash endpoint handling. This is a deliberate compatibility choice
on Blackwell too: the faster FP32 variant changes seeded draws.

SnakeBeta preserves eager BF16 rounding after each arithmetic operation.
Small tensors use direct sine; large tensors use a shared 128 KiB lookup table
covering all BF16 inputs. Coefficients and the table are prepared before capture.

## Validation And Timing

The Torch references follow upstream Qwen-TTS BF16 rounding. Compiled model
baselines enable `emulate_precision_casts`. Individual kernels use tolerance
checks, and fused Conv1D post-ops are checked bitwise against Torch applied to
the same custom convolution output. Tests cover causal tails, irregular
batches, dynamic-size binary reuse, changing cache slots, and graph replay.
The large-offset SnakeBeta regression needs 12 GiB free memory and skips when
that memory is unavailable. Tests target one GPU per process.

Full-model benchmarks report numerical differences and top-code agreement,
rejecting shape mismatches and non-finite values rather than applying a
pointwise tolerance gate. Reduction-order differences can accumulate, including
with compiled Torch. The 1.7B post-attention checkpoint checks also exceeded
pointwise `atol=rtol=0.02` for a small fraction of outputs, despite relative-L2
errors around 0.06%; FP64 projection references with BF16 boundaries exceeded
that tolerance too. See [numerical attribution](design.md#predictor-numerical-attribution).
These checks do not establish free-running speech or perceptual quality.

Timings use unprofiled, interleaved CUDA graphs, normally three samples of five
replays. Compilation, weight packing, and setup are excluded. Standalone
pre/post-attention rotates 28 weight sets. QKV's standalone baseline compiles
arithmetic but uses direct in-place cache updates: compiling `index_copy_`
introduced full-cache copies. Full-talker compiled baselines still include
that overhead, so their speedups are not solely faster arithmetic.

Selected development measurements below predate the dynamic-size cleanup;
rerun the scripts for current numbers. They are component timings, not TTS
throughput. The transformer measurements use BF16 and interleaved graphs:

| Workload | GPU | Batch | Compiled Torch | Fused | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B QKV preparation | H200 | 16 | 22.57 us | 5.99 us | 3.77x |
| 0.6B post-attention | H200 | 16 | 27.14 us | 12.22 us | 2.22x |
| 1.7B talker decode step | H200 | 16 | 2.553 ms | 1.361 ms | 1.88x |
| 1.7B talker decode step | B300 | 16 | 2.222 ms | 1.064 ms | 2.09x |

The 1.7B model runs use checkpoint weights, synthetic embeddings, a 128-token
prefix, and the final of three teacher-forced decode steps. The waveform-tail
measurements below use the 0.6B tokenizer checkpoint, 64 codec frames, Torch
2.14.1+cu132, and CUTLASS 4.8.0 (October 23 tuning pass):

| GPU | Batch | Compiled BTC | Custom | Speedup |
| --- | ---: | ---: | ---: | ---: |
| H200 | 1 | 1.452 ms | 1.093 ms | 1.33x |
| H200 | 4 | 4.866 ms | 3.781 ms | 1.29x |
| B200 | 1 | 1.077 ms | 0.910 ms | 1.18x |
| B200 | 4 | 3.151 ms | 2.340 ms | 1.35x |

## Profiling

`--profile` emits CUDA-profiler capture ranges and named NVTX graph replays.
Use graph-level Nsight capture for scheduling and graph-node capture for kernel
attribution. Keep traces directly under repository `tmp/`, with model, GPU,
batch, and variant in the filename. Graph-node tracing can inflate timings and
introduce launch gaps; use unprofiled measurements for speedup claims.

Pre/post-attention `--trace-only` launches one custom block for IKET:

```bash
run-iket --output-dir /path/to/repo/tmp profile --postprocess all \
  --max-ts-cnt-per-warp 256 --keep -- \
  python mega_qwen3_tts/kernels/post_attention.py --batch-size 16 --trace-only
```

| Markers | Meaning |
| --- | --- |
| `qkv_weight_prefetch`, `input_norm`, `qkv_operand_wait`, `qkv_mma` | QKV startup, normalization, operand wait, compute |
| `qkv_partial_publish`, `qkv_partial_wait`, `head_norm_rope_cache` | Head reduction and epilogue |
| `output_projection`, `gate_up_projection`, `down_projection` | Post-attention GEMMs, including nested waits |
| `splitk_store`, `splitk_receive`, `output_reduce`, `down_reduce` | Mailbox transfer/completion and reduction |
| `gate_up_prefetch_wait`, `normalization_input_wait`, `norm_local_stats`, `norm_publish` | Scheduling hint, payload polling, RMS statistics, operand publication |
| `activation_wait`, `silu_and_gate`, `down_cluster_release` | Activation handoff and DSMEM lifetime rendezvous |

Auxiliary markers identify sorting, sampling, lookup, and activation phases.
Sampling's `gumbel_prepare` runs in helper warps while warp 0 merges candidates
and computes softmax; `sample_handoff` overlaps that work. IKET instrumentation
is inactive in normal compilation. Instrumented ranges include waits and
overhead, and overlapping warp ranges cannot be summed into kernel time.
