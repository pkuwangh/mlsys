# Qwen 27B Batch-1 Low-Latency Decode Plan

Build a Blackwell decode implementation for the local Qwen3.5-27B and
Qwen3.8-27B checkpoints, using the fusion and scheduling techniques studied in
the GPT-OSS batch-1 implementation. Start with TP1, BF16 weights/activations,
one active request, and one new token per forward pass. Keep vLLM prefill,
cache allocation, request scheduling, and sampling as the initial integration.

This is separate from the [same-GPU prefill/decode proposal](same-gpu-prefill-decode.md).
First optimize standalone decode on the full GPU; SM partitioning is a later
experiment. No speculative decoding or quantization changes in the first version.

## Baseline and Opportunity

The unprofiled B200 run on `umbriel-b200-079`, vLLM `1e5d0ea888`, reports:

| Metric | Result |
| --- | --- |
| Input / output tokens | 1,024 / 1,024 |
| Median request latency, three runs | 10.885 s |
| Output throughput including prefill | 94.1 tokens/s |
| Mean time per subsequent decode token | 10.594 ms |
| Time to first token | 47-48 ms |

[Run log](../../../tmp/qwen3p8-tp1-b1-t1k-rerun.log).
[Trace](qwen3p8-tp1-b1-t1k.nsys-rep): 1,023 decode graph launches, about
10.62 ms between graph starts, and 1,009 kernels per decoder-body graph.
The LM head and other work outside that graph also contribute to token latency.

Initial attribution from kernel sequence, launch geometry, and model source:

| Operation | Calls per decode token | Aggregate kernel time per token |
| --- | ---: | ---: |
| MLP gate/up projection | 64 | 3.745 ms |
| MLP down projection, including split-K reduction | 64 | 2.099 ms |
| GDN QKVZ projection | 48 | 1.349 ms |
| Attention output projection, including split-K reduction | 64 | 0.997 ms |
| Full-attention Q/gate/K/V projection, including reduction | 16 | 0.442 ms |
| GDN b/a projection, including reduction | 48 | 0.282 ms |
| LM head | 1 | About 0.370 ms |
| GDN recurrent update | 48 | 0.251 ms |
| Full attention | 16 | 0.207 ms |

These are profiled kernel durations, not additive savings or unprofiled latency
predictions. Confirm projection attribution with captured operands when building
the microbenchmarks. The priority is the dense projection chain, not replacing
the attention backend first.

The large linear weights total approximately 51.24 GB in BF16 per complete decode
step, excluding other state/cache traffic. At the trace's reported peak bandwidth
of 7.67 TB/s, streaming those weights once would take about 6.68 ms. This is an
idealized bandwidth reference, not measured DRAM traffic or a predicted result.
The MLP GEMMs alone stream about 34.23 GB in 5.64 ms, roughly 6.1 TB/s effective
weight throughput. There is useful headroom, but fusion does not remove the need
to read dense weights. GPT-OSS's quantized MoE throughput is not a target for this
dense BF16 model.

## Model Contract

Both local checkpoint configurations use the same core dimensions:

| Component | Dimensions / behavior |
| --- | --- |
| Decoder | 64 layers: three GDN layers followed by one full-attention layer, repeated |
| Hidden width | 5,120 |
| Dense MLP | 5,120 -> gate/up 34,816 -> SwiGLU 17,408 -> 5,120 |
| GDN projections | 5,120 -> QKVZ 16,384; separate b/a 96; output 6,144 -> 5,120 |
| GDN heads | 16 key heads and 48 value heads, each dimension 128 |
| GDN state | Conv1D history with kernel width 4; FP32 recurrent matrices |
| Full attention | 24 query heads, four KV heads, head dimension 256 |
| Full-attention projections | 5,120 -> Q/gate/K/V 14,336; output 6,144 -> 5,120 |
| Position encoding | Partial rotary dimension 64, preserving checkpoint M-RoPE semantics |
| LM head | 5,120 -> vocabulary 248,320 |

Use configuration checks, not directory names, to select supported kernels.
The local Qwen3.8 checkpoint still uses `Qwen3_5ForConditionalGeneration`.
Its GDN output gate is configured as `swish`, which this vLLM implementation
normalizes to `silu`; Qwen3.5 defaults to `silu`.

Preserve distinctions that matter numerically:

- Layer and Q/K RMSNorm use the checkpoint's `1 + weight` convention and FP32
  arithmetic before the output cast. Do not substitute ordinary RMSNorm.
- GDN Q/K L2 normalization is different from RMSNorm. Its output normalization
  operates on each 128-element value head and precedes the SiLU output gate.
- Full attention uses a sigmoid output gate, not the GDN SiLU gate.
- Preserve FP32 recurrent state, `A_log` handling, recurrence order, and BF16
  rounding at operation boundaries. Establish residual semantics from the actual
  vLLM reference; do not copy GPT-OSS's rounding tricks indiscriminately.
- Preserve paged KV layout, block IDs, state-slot indexing, and dynamic positions.
  No compile-time assumptions about a specific request's cache addresses.

Each residual addition has exactly one owner. The fusion diagrams below describe
candidate regions, not operations to accumulate across implementation stages.
If an output projection adds the residual, its successor normalizes that already
combined value without adding it again. Keep this explicit in the block interface
and in adapters to vLLM's `(hidden_states, residual)` convention.

## Techniques to Adapt

| Technique | Proposed use | Constraint |
| --- | --- | --- |
| Warp-specialized TMA + tensor-core small-N GEMM | Stream large weights while retaining the input vector | Compete with cuBLAS on actual Qwen shapes; compare a SIMT reduction for tiny b/a |
| Cluster-local split-K reduction | Output and down projections; fuse final reduction with the epilogue | Avoid a separate reduction launch without sacrificing memory throughput |
| Fused input normalization / epilogues | Norm + projection, gate/up + SwiGLU, projection + residual | Maintain reference rounding and norm scope |
| Early weight prefetch and PDL | Start successor preparation while the predecessor finishes | Dependent activation reads still wait; verify programmatic graph edges |
| TMEM operand staging | Extend weight prefetch while activations are not ready | Add only when buffering is limiting; account for accumulator storage |
| Tile-level publication | Potential later gate/up-to-down or projection-to-GDN fusion | Prove memory ordering, replay reset, and producer forward progress |
| Persistent scheduling / dynamic work assignment | Large LM head or a measured tail imbalance | Compare with static scheduling; do not add atomics or CLC by default |
| Offline weight packing | Paired gate/up tiles and contiguous projection tiles | Pack outside timing; account for any duplicate storage |

Use CUTLASS primitives where the installed release supports them, with documented
CuTe helpers where needed. No ctypes tensor-map construction or copied private
driver bindings. Pin the tested toolchain, particularly for DSMEM stores and PDL.
The reference extraction itself documents unresolved synchronization questions;
reuse its techniques, not its validation claims.

PDL overlaps independent preparation, not arbitrary dependent computation.
For partial-result consumption, use a separate, explicit publication protocol.
Do not choose a BF16 sentinel that valid activations can produce. Prefer readiness
flags/epochs unless the payload encoding has a proven reserved value. Never rely
on arbitrary CTAs or two streams being co-scheduled to avoid a polling deadlock.
See [CUDA PDL semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Implementation Order

### 1. Establish Projection Microbenchmarks

Capture representative inputs and weights from actual decode, including both
checkpoints. Benchmark the gate/up, down, and attention-output shapes first, then
QKVZ and the tiny b/a projection. Test a small number of tile sizes, stage counts,
and cluster split-K choices rather than building a general autotuning framework.

Compare raw projections against the same cuBLAS-backed operations used by vLLM,
under CUDA graphs. Compare fused chains against the existing compiled chain,
including its reductions and pointwise operations. Keep a raw custom kernel only
if it is competitive or its complete fused chain provides a measured win.

Use rotating layer weights whose working set exceeds L2. Replaying one output
projection's roughly 63 MB weight matrix would otherwise create an unrealistic
cache-hot benchmark on this B200. Keep actual input/state evolution in correctness
tests; do not let graph replay reuse a stale result.

Deliverable: validated small-N projection kernels and a per-shape timing table.
Start with attention-output `6,144 -> 5,120` to establish cluster reduction and a
residual epilogue, then prioritize the much larger aggregate MLP opportunity.

### 2. Fuse the Dense MLP

First target two kernels, not one whole-layer megakernel:

```text
attention output + residual
    -> residual add + RMSNorm + gate/up projection + SwiGLU
    -> down projection + split-K reduction + residual add
```

Pair matching gate/up rows so their results can feed SwiGLU in one epilogue.
Fuse the down projection's final reduction and residual addition. Track the
residual tensor contract so the next layer does not add it a second time.

A next-layer RMSNorm needs the complete 5,120-element hidden vector. It cannot be
computed independently from each output CTA's local rows. Initially leave it in
the next projection's input preparation, after the dependency is satisfied.
Redundant computation of this small norm per CTA is a candidate if weight loads
hide its cost, not an assumed win.

Integrate the winning MLP chain into all 64 layers and remeasure full decode.
Then test PDL between the two custom kernels with successor weight prefetch.
Keep PDL only if it improves the chain under realistic weight traffic. Gate/up
and down both stream large weights, so overlap can also increase contention.

Only investigate a single-kernel tiled producer/consumer MLP after this baseline
works. Down projection consumes the full intermediate vector: this is a real
cross-CTA dependency, not an ordinary epilogue fusion.

### 3. Fuse the GDN Layers

Target the 48-layer path in independently measurable steps:

```text
input RMSNorm + QKVZ and b/a projections
    -> Conv1D history update + SiLU + Q/K L2 norm + gate preparation
    -> recurrent state update + output RMSNorm + SiLU gating
    -> output projection + residual
```

First combine the large QKVZ projection and tiny b/a work under shared input
preparation where profitable; retaining a separate small SIMT path is acceptable
if appending 96 rows hurts the large projection's tiling. Remove avoidable
layout/zeroing kernels only after proving every valid output element is written.

Fuse the recurrent update with per-head output norm/gating, then assess adding
the Conv1D update. Respect the sharing of each Q/K head across three value heads.
Use the existing GDN implementation as an intermediate fallback while optimizing
projections; its recurrent kernel is already fused and relatively inexpensive.

Validate both emitted values and updated convolution/recurrent state. The FP32
recurrent matrices total about 144 MiB across these 48 layers for one request;
their traffic and lifetime cannot be ignored or silently converted to BF16.

### 4. Fuse Full-Attention Preparation

Start by keeping the current FlashInfer attention kernel:

```text
input RMSNorm + Q/gate/K/V projection + Q/K RMSNorm + RoPE + KV append
    -> existing paged attention
    -> sigmoid gating + output projection + residual
```

Align projection tiles with head-level normalization boundaries, or explicitly
reduce across the relevant CTAs. Preserve the Q/gate checkpoint interleaving.
Use the existing KV block table and slot mapping; do not create a private cache.

Direct QKV-to-attention fusion with a cluster-local handoff is a later experiment.
There are only four KV heads here, and a whole-attention cluster arrangement can
underutilize the GPU or scale poorly with context length. Reuse the GPT-OSS idea
only after measuring the Qwen head geometry. At this baseline, FMHA is about
0.21 ms/token; replacing it first would miss the dominant costs.

### 5. Optimize the Decode Tail

Reuse the projection machinery for final residual/RMSNorm plus LM head. Compare
static and persistent scheduling before adding dynamic work assignment. Its
2.54 GB weight stream limits what norm fusion alone can save.

Initially return the full logits tensor to vLLM's existing sampler. A tiled
argmax/top-k tail is a separate greedy-only optimization: it must retain token
masking, penalties, tie handling, and stopping behavior, or explicitly fall back.
Do not restrict the vocabulary to a guessed candidate subset.

## Integration and Code Layout

Keep kernel development under `mlsys/gpu/kernels/qwen27b_decode/`, with a small
shared projection core, operation-specific modules, and standalone benchmarks.
Add modules as their kernels become useful, not a speculative class hierarchy.
Keep weight packing and workspace allocation outside the timed decode path.

Expose one experimental decode-backend selection in the latency benchmark.
Use an explicit model/block integration following vLLM's registered custom-op
and model conventions, including worker-process initialization and compile/fake
implementations. Do not rely on ad hoc monkey-patching during graph replay.

The measured engine uses the **V2 model runner** at
`vllm/vllm/v1/worker/gpu/model_runner.py`; integration must exercise that path,
not only the older `gpu_model_runner.py`. Prefer model/block changes that leave
the runner and scheduler intact. Keep prefill and unsupported decode cases on
the existing implementation; distinguish a genuine single-token decode from
one-token prefill and graph padding using metadata, not tensor shape alone.

Share vLLM's loaded weights where layouts permit. If packed decode copies are
needed while native prefill keeps the original weights, report their exact memory
cost before allocation and reduce cache capacity accordingly. Do not silently
duplicate the roughly 51 GB of linear weights under the existing 85% memory budget.

## Correctness and Performance Gates

For each replacement:

1. Validate the operator with checkpoint and adversarial numerical inputs against
   the reference, including output dtype, rounding, and untouched state slots.
2. Validate a complete layer, then several teacher-forced decode steps with fixed
   input tokens. Compare hidden states, full logits, KV, and recurrent state so
   sampling divergence cannot conceal the first numerical discrepancy.
3. Run end-to-end greedy generation on multiple real prompts and both checkpoints.
   Inspect first mismatches and top-logit margins; do not assume changed reduction
   order guarantees bitwise equality. Choose tolerances from the reference error,
   not after seeing the candidate fail.
4. Replay graphs with changing inputs, fresh and reused request slots, cache-page
   crossings, and short/long contexts. Exercise enough replays to catch stale
   readiness flags and premature workspace reuse. Use sanitizer checks for new
   asynchronous memory/barrier protocols where supported.
5. Interleave baseline/candidate timing on the same B200 after warmup. Use a few
   short samples, stable power settings, and record clocks/temperature. Benchmark
   without profiling; collect separate before/after traces in one fixed directory.

Retain the 1K-input/1K-output synthetic case for continuity, plus real text prompts
and context-length sweeps through the production roughly 28K-token range. Prefill
must supply valid native cache state. Video-conditioned prompts can follow once
text decode and M-RoPE metadata are validated; no vision-encoder rewrite is needed.

Report operator and fused-chain latency, GPU decode-step latency including the LM
head, and end-to-end request/TTFT/decode-token metrics separately. Include packing
memory, state traffic, launch counts, and unsupported cases. Require a repeatable
end-to-end gain before enabling a replacement by default. Keep numerical format
changes and speculative decoding out of the BF16 fusion comparison.

## First Milestone

Implement the cold-weight output-projection and MLP microbenchmarks, establish a
competitive projection core, and integrate a validated two-kernel fused MLP.
Remeasure the full 64-layer decode and capture its trace before committing to
cross-CTA MLP fusion or a custom full-attention kernel.

## Library Experiments: October 8, 2026

Tested Qwen3.8-27B on B200 `umbriel-b200-079`, BF16, TP1, batch 1.
Software: vLLM source `ab905a885d`, PyTorch `2.13.0+cu132`, FlashInfer
`0.7.0.post1`, CUTLASS DSL `4.8.0`, cuDNN backend `9.20.0`, and frontend
`dd335d9442`. The editable vLLM package metadata still reports the older
`1e5d0ea88` revision; these runs use the updated checkout's Python sources.
No upstream library code was modified.

### End-to-End Linear Backend A/B

Same 1,024-input/1,024-output synthetic request, greedy decoding, EOS ignored,
no prefix caching or speculation. Each process runs one warmup followed by
three measured requests, without profiling. Order: default, FlashInfer, default
again, with the MLP experiments before the final default run.

| Backend | Median request (ms) | Median TTFT (ms) | Median mean decode-token time (ms) | Output tokens/s, including prefill |
| --- | ---: | ---: | ---: | ---: |
| Default, before | 10,891.355 | 48.077 | 10.599 | 94.0 |
| FlashInfer CuTeDSL, heuristic GEMM selection | 10,626.424 | 77.336 | 10.312 | 96.4 |
| Default, after | 10,896.372 | 47.242 | 10.605 | 94.0 |

FlashInfer reduces decode-token latency by about **2.7%**; request throughput
improves about **2.5%**, with worse TTFT. Default request timing drift is under
0.05%. All nine measured outputs have identical token hashes, but this is one
synthetic prompt, not a model-quality validation suite.

The initial full FlashInfer autotuning run was stopped during startup because it
tuned many token-count buckets beyond batch 1. The completed run explicitly skips
BF16 GEMM autotuning; it is **not** a fully tuned end-to-end result. Reproduce from
`mlsys/inference/vllm`:

```bash
python benchmark/benchmark_qwen3p8_27b_latency.py
VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=bf16_gemm \
    python benchmark/benchmark_qwen3p8_27b_latency.py --linear-backend flashinfer_cutedsl
```

Logs: [default before](../../../tmp/qwen27b-kernel-experiments/baseline-before.log),
[FlashInfer](../../../tmp/qwen27b-kernel-experiments/flashinfer-heuristic.log),
[default after](../../../tmp/qwen27b-kernel-experiments/baseline-after.log).

### Complete MLP Microbenchmark

Actual layer-0 checkpoint weights: `M=1`, hidden width 5,120, intermediate width
17,408. The baseline combines gate/up into one GEMM, followed by SwiGLU and the
down projection. Inputs are three random BF16 vectors, not captured decode
activations. FlashInfer autotuning is restricted to `M=1` here.

Each complete chain is CUDA-graph captured after compilation/planning. Timings
use five interleaved rounds of three single-replay samples, rotating/reversing
variant order. A 256 MiB buffer evicts L2 before each sample, outside timing.

| Complete MLP chain | Median (us) | Speedup vs torch.compile | GPU kernels |
| --- | ---: | ---: | ---: |
| cuBLAS + vLLM CUDA SwiGLU | 114.688 | 0.982x | 4 |
| torch.compile | 112.640 | 1.000x | 4 |
| FlashInfer CuTeDSL + compiled SwiGLU | 90.112 | **1.250x** | 3 |
| FlashInfer TinyGEMM + compiled SwiGLU | 112.640 | 1.000x | 3 |
| FlashInfer TGV + compiled SwiGLU | 104.512 | 1.078x | 3 |
| cuDNN `swiglu_mlp` | 117.120 | 0.962x | 3 |

Separate diagnostic traces confirm that cuDNN fuses gate/up/SwiGLU, then runs
the down GEMM and its split-K reduction. FlashInfer CuTeDSL uses two GEMMs and
a separate SwiGLU, without a separate down-projection reduction. Thus, fewer
operation boundaries alone do not establish a win: the unfused FlashInfer
chain is faster in this test.

All six variants pass `atol=0.02, rtol=0.02` for all three inputs, with maximum
relative L2 error below 0.1%. The reference uses FP32 activation arithmetic
between BF16 projections; implementations can differ in intermediate rounding.
The standalone cuDNN `GemmSwigluSm100` default `(128,128)` tile passed its support
check and compiled, but execution raised a CUDA illegal instruction. It is
excluded from the table; successful timings were collected in a fresh process.
The cause of that failure remains undiagnosed.

[Microbenchmark source](../../../tmp/qwen_mlp_library_bench.py),
[log](../../../tmp/qwen27b-kernel-experiments/mlp-final.log),
[measurements and kernel names](../../../tmp/qwen27b-kernel-experiments/mlp-results.json),
[failed standalone attempt](../../../tmp/qwen27b-kernel-experiments/mlp.log).
Per-variant PyTorch profiler Chrome traces are in the same results directory.

Next: establish a batch-1-tuned end-to-end FlashInfer baseline and test the other
projection shapes before writing replacement GEMMs. Use its fast MLP chain as
the fusion baseline. The 1.25x microbenchmark result is not a full-model speedup:
only MLP is measured, inputs/cache conditions differ, and the end-to-end run uses
heuristic rather than batch-1-tuned GEMM selection.

## Source Map

- [Qwen3.5 model and weight mapping](vllm/vllm/model_executor/models/qwen3_5.py)
- [Decoder flow and gated full attention](vllm/vllm/model_executor/models/qwen3_next.py)
- [GDN projections, normalization, and state update](vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
- [RMSNorm conventions](vllm/vllm/model_executor/layers/layernorm.py)
- [Low-latency technique notes](../../gpu/kernels/gpu-low-latency.md)
- [GPT-OSS batch-1 reference](../grafia/test/gpt_oss_120b_ctm/reference/bs1)

Use the reference as a source of techniques. Its MoE routing, quantized expert
weights, dimensions, delay constants, and historical speedups are not Qwen defaults.
