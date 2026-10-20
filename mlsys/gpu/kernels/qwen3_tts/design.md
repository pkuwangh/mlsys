# Qwen3-TTS Talker Fusion Design

This document describes the implemented design and the decisions we retain.
Update it when the implementation changes; commands, results, and validation
status belong in [README.md](README.md).

## Scope and Boundaries

Target: **0.6B talker, Hopper, BF16, TP1, single-token decode, B=1..32**.
The backbone has 28 layers, hidden width 1024, 16 query heads, 8 KV heads,
head dimension 128, and SwiGLU width 3072. Projections have no bias.
The 1.7B model, training, code predictor, sampling, and vocoder are outside scope.

The default layer runs:

```text
input RMSNorm + QKV + head RMSNorm + RoPE + cache/mask writes
    -> cuDNN attention through Torch SDPA
    -> output projection + residual + RMSNorm + SwiGLU MLP + residual
```

Multi-token prefill stays in eager Torch. Decode uses custom blocks and compiles
the remaining Torch operations. The caller owns a contiguous KV cache
`[2,B,8,capacity,128]`; all batch rows share one decode position. Packed weights
are prepared once and must remain unchanged.

Implementation: [pre_attention.py](mega_qwen3_tts/kernels/pre_attention.py),
[post_attention.py](mega_qwen3_tts/kernels/post_attention.py), and
[talker.py](mega_qwen3_tts/talker.py).

## Numerical Contract

Follow upstream Qwen-TTS eager BF16 semantics, not SGLang's deferred-residual
and fused-add-normalization rounding. GEMMs accumulate in FP32, but split-K
and warp reductions can change accumulation order: agreement is tolerance-based.
`bf16` and `fp32` below denote explicit casts; weights use `[out,in]` orientation.

```python
def rmsnorm(x, gamma, eps=1e-6):
    h = fp32(x)
    inv_rms = rsqrt(mean(h * h, dim=-1, keepdim=True) + eps)
    unit = bf16(h * inv_rms)
    return bf16(fp32(unit) * fp32(gamma))


z = rmsnorm(x, input_gamma)
q, k, v = bf16(z @ Wq.T), bf16(z @ Wk.T), bf16(z @ Wv.T)
q = q.reshape(B, 16, 128)
k, v = k.reshape(B, 8, 128), v.reshape(B, 8, 128)
q, k = rmsnorm(q, q_gamma), rmsnorm(k, k_gamma)  # Per 128-channel head
# cos/sin use the shared decode position.
q = bf16(fp32(bf16(q * cos)) + fp32(bf16(rotate_half(q) * sin)))
k = bf16(fp32(bf16(k * cos)) + fp32(bf16(rotate_half(k) * sin)))
cache_k[:, :, position], cache_v[:, :, position] = k, v
attn = attention(q, cache_k[:, :, : position + 1], cache_v[:, :, : position + 1])
attn = attn.reshape(B, 2048)

p = bf16(attn @ Wo.T)  # attn: [B,2048], Wo: [1024,2048]
r = bf16(fp32(p) + fp32(x))
z = rmsnorm(r, post_gamma)
g, u = bf16(z @ Wgate.T), bf16(z @ Wup.T)  # Both weights: [3072,1024]
a = bf16(fp32(bf16(silu(fp32(g)))) * fp32(u))
d = bf16(a @ Wdown.T)  # Wdown: [1024,3072]
y = bf16(fp32(d) + fp32(r))
```

Attention is `softmax(Q @ K.T / sqrt(128)) @ V`; each pair of query heads
shares one KV head. The cache slices above restrict it to valid positions.

Keep the casts before gamma multiplication, after SiLU, and before both residual
additions even when intermediates stay in registers. QKV compilation uses
`--fmad=false` to preserve RoPE's separately rounded products rather than BF16
FMA contraction. Signed zeros used in sentinel payloads are canonicalized as
described below. Full-backbone numerical/quality validation remains open.

## Fused QKV Preparation

Launch **32 four-CTA clusters, 512 threads per CTA**. Clusters 0-15 own Q heads,
16-23 own K heads, and 24-31 own V heads. For each head:

| Rank | Output channels | Input reduction |
| --- | --- | --- |
| 0 | 0-63 | K=0..511 |
| 1 | 0-63 | K=512..1023 |
| 2 | 64-127 | K=0..511 |
| 3 | 64-127 | K=512..1023 |

Every projection computes `weight @ activation.T`: MMA M is output channels,
N is the batch bucket (16 or 32), and K is input channels. Token columns beyond
B are padded and excluded from final writes.

1. Thread 0 starts a 64-KiB weight TMA before input normalization.
2. All 16 warps normalize full input rows, then stage this CTA's K512 slice.
   Each warp owns one row at N16 or two rows at N32.
3. Warps 4-7 issue M64 x N16/32 x K512 WGMMA and stage FP32 partials.
4. Ranks 1-3 bulk-copy their partials into rank 0's DSMEM mailbox. Rank 0
   writes its own partial locally; the transaction barrier counts only remote
   copies. Rank 0 sums each K pair, applies Q/K head RMSNorm and RoPE, and
   writes Q or the current KV slot. V needs neither head norm nor RoPE.
5. Optionally, rank 1 of cluster 0 writes an additive `[1,capacity]` BF16 mask:
   zero through the current position, negative infinity afterward.

The default QKV block needs only cluster-local synchronization, not a cooperative
grid or global scratch. Input normalization is intentionally repeated per CTA.
Sharing it across the cluster saved arithmetic but added operand waiting and
was slower at both B16 and B32. Bulk partial copies were faster than per-element
`st.async` publication for this layout.

The additive-mask output is a compile-time option and enabled in the talker.
It removes SDPA's separate boolean-to-additive mask kernel without changing
the attention calculation.

## Fused Post-Attention

Launch cooperative four-CTA clusters with **256 threads per CTA**. All GEMMs
use Hopper WGMMA with BF16 operands and FP32 register accumulators, not TMEM.

| Batch bucket | Grid | Output projection | Gate/up | Down projection |
| --- | ---: | --- | --- | --- |
| N16 | 96 CTAs | CTAs 0-63 | All 96 CTAs, one tile each | CTAs 0-63 |
| N32 | 112 CTAs | CTAs 0-63 | CTAs 64-111, two tiles each | CTAs 0-63 |

### Projection Mapping

For output/down cluster `c=0..15`, all four ranks contribute to output channels
`[64*c,64*c+64)`. Rank `r` computes:

```text
output: K=[512*r,512*r+512), two K256 stages
down:   K=[1536*s+384*r,1536*s+384*r+384), for stage s=0,1
```

Split-K uses **push reduce-scatter**: `st.async` sends each rank's partials to
the appropriate owner mailbox; transaction barriers signal arrival. Rank r
then sums all four contributions for its 16 output channels. Output projection
rounds and publishes the residual; down projection rounds, adds that residual,
and writes the final layer output. There are no global split-K partials.

Gate/up tile `t` produces activation channels `[32*t,32*t+32)`:

```text
N16: t = CTA ID                         # 96 tiles in one wave
N32: t = CTA ID - 64 + 48*wave          # 48 tiles per wave, two waves
weight tile = 32 gate rows + 32 matching up rows, full K1024
GEMM = weight[64,1024] @ normalized_input.T[1024,N]
epilogue = round gate/up -> round SiLU -> multiply -> round activation
```

Gate/up has no split-K. Its SMEM epilogue pairs gate and up rows before publishing
32 activation channels. At N32, down stage 0 consumes tiles `12*r..12*r+11`
while gate/up produces the second wave; stage 1 consumes tiles 48 positions
later. N16 instead gains parallelism by giving one tile to each of 96 CTAs.

### Warp Roles and Storage

| Threads | Output/down | Gate/up |
| --- | --- | --- |
| Warps 0-3 | Weight TMA and activation staging/polling | Thread 0 issues full-tile weight TMA |
| Warps 4-7 | WGMMA | WGMMA on resident normalized inputs |
| All, between GEMMs | Split-K reduction and residual epilogue | RMSNorm and SwiGLU epilogue |

Output/down have two distinct weight/operand stages, with no buffer-reuse
handshake between them. Gate/up holds its entire 128-KiB weight tile in SMEM;
that storage aliases the otherwise unused output/down operand region.
At N32, the next gate/up weight tile loads during the first tile's epilogue.
At N16, the down mailbox must be separate from the gate/up epilogue: a peer can
start down while this CTA still reads its gate/up partials.

Each gate/up CTA independently normalizes the full residual. One warp owns
a complete 1024-channel row, retaining the polled values and gamma in registers.
Shuffles reduce statistics; normalized BF16 data goes directly into WGMMA's
SMEM layout. No shared norm partials or inverse-RMS broadcast remain. A proxy
fence and final CTA rendezvous publish the operands to WGMMA.

### Publication and Lifetime

Residual and activation handoffs use packed BF16 Lamport payloads, not separate
release/acquire flags. Each aligned 32-bit word contains two BF16 values; the
empty word is `0x80008000` (two negative zeros). Producers canonicalize live
negative zeros to positive zero and publish both values in one store.
Consumers issue vector loads before checking readiness, retain completed
vectors, and retry incomplete ones. The polled values are the consumed payload;
there is no flag followed by a separate data read or GPU-wide release fence.

| Workspace | Physical layout | Purpose |
| --- | --- | --- |
| Residual | Int32 `[2,N,512]` | Packed `[N,1024]` BF16 residual, two slots |
| Activation | Int32 `[2,96,N,16]` | Packed `[96,N,32]` BF16 SwiGLU output, two slots |
| Epochs/hints | Int32 `[112]` | Invocation counters; N16 also uses 16 prefetch hints |

Slots start empty. Each call uses `epoch & 1` and clears the inactive slot for
the next call. Calls sharing a module/workspace must be serialized, including
CUDA-graph replays; concurrent streams need separate instances. Normalized
inputs and uncombined gate/up outputs never go to global memory.

The cooperative launch is required for cross-cluster producer progress; the
grid must fit the compiled kernel's actual cluster occupancy. Do not substitute
a normal launch merely because the CTA count is below the SM count. Cluster
rendezvous also protects DSMEM lifetimes before reuse or exit.

### Startup Decisions

- N16 output weight TMA starts before epoch/mailbox setup. Earlier activation
  loads, `cp.async`, and activation TMA did not improve complete-block latency.
- N16 CTAs 64-95 wait for an output producer's first-MMA progress hint, then
  load their own gate/up weights and prefetch the other 64 tiles into L2.
  The hint is only scheduling advice; consumers still validate every payload.
- N32 waits for an existing residual word before gate/up TMA. No fixed delay
  is used: delay experiments helped identify contention, not a stable timing rule.
- Warp-local norm replaced repeated shared-memory handoffs. Preserving the old
  summation order cost latency; the retained order is not bitwise-equivalent.

## Attention Decision

Keep cuDNN attention through Torch SDPA. Standalone SIMT attention, phased
QKV+attention fusion, and PDL were tested but not retained. On H200 at B16,
cuDNN gave 663.7 us per full-backbone step; the best custom variant (standalone
with PDL) took 694.9 us. Phased fusion still read Q/K/V through global memory
and was slower. Removing a launch or overlapping kernels was not sufficient
to offset the attention cost. No custom attention kernel or PDL option is shipped.

## Open Work

The blocking issue for serving integration is full-backbone numerical and
generation-quality validation, not another isolated microbenchmark speedup.
Any future attention work should beat cuDNN in the full layer sequence; possible
directions are sequence partitioning or grouping Q/K/V heads for a local handoff.
Neither is part of the retained design. General low-latency techniques are in
[gpu-mega-fusion.md](../gpu-mega-fusion.md) and
[gpu-low-latency.md](../gpu-low-latency.md).
