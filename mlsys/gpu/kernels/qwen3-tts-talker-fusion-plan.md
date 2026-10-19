# Qwen3-TTS Talker Output Projection and MLP Fusion

Design for the **0.6B talker, BF16, TP1, decode, 1 <= B <= 32**.
Initially targeted at H200; the first implementation is tested on H100.
This is a full-block design, not a sequence of independently fused operators.
Implementation and measured validation status: [qwen3_tts](qwen3_tts/README.md).
The standalone block passes, but full-backbone numerical validation remains open.

Use the CTA specialization, split-K reduction, and producer/consumer ideas in
[gpu-mega-fusion.md](gpu-mega-fusion.md). On Hopper, use WGMMA and register
accumulators, **not Blackwell TCGEN05/TMEM**. Attention stays outside this kernel.

## 1. Math and Boundary

Weights below use PyTorch's `[out, in]` orientation:

| Input | Shape | Meaning |
|---|---|---|
| `attn` | `[B, 2048]` | Attention output: 16 query heads x 128 |
| `residual` | `[B, 1024]` | Residual entering the attention sublayer |
| `Wo` | `[1024, 2048]` | Attention output projection |
| `gamma` | `[1024]` | Post-attention RMSNorm scale; epsilon=1e-6 |
| `Wgate`, `Wup` | Each `[3072, 1024]` | SwiGLU projections |
| `Wdown` | `[1024, 3072]` | MLP output projection |

Use upstream Qwen-TTS eager BF16 semantics, not SGLang's fused add+RMSNorm
rounding. There are no projection biases. With BF16 weights, conceptually:

```python
p = bf16(attn @ Wo.T)                         # FP32 GEMM accumulation
r = bf16(fp32(p) + fp32(residual))             # Round before normalization
h = fp32(r)
inv_rms = rsqrt(sum(h * h, dim=-1) / 1024 + 1e-6)
normalized = bf16(h * inv_rms[:, None])        # Upstream rounds before gamma
z = bf16(fp32(normalized) * fp32(gamma))
g = bf16(z @ Wgate.T)
u = bf16(z @ Wup.T)
activated_gate = bf16(silu(fp32(g)))          # Separate upstream SiLU result
a = bf16(fp32(activated_gate) * fp32(u))
d = bf16(a @ Wdown.T)
y = bf16(fp32(d) + fp32(r))
return y
```

Return **one BF16 layer-output tensor, `y`**, matching upstream. Both residual
additions use FP32 arithmetic followed by BF16 rounding. RMSNorm reads the rounded
residual, and its normalized result rounds again before the gamma multiplication.
SwiGLU likewise rounds the SiLU result before multiplying by the up projection.
Do not substitute SGLang's unrounded residual normalization or deferred output pair.

Preserve these BF16 boundaries even though intermediate values stay inside one
kernel; register conversions do not require separate launches or GMEM writes.
In particular, do not move `inv_rms` outside the gate/up GEMM: that
would bypass rounding of normalized inputs. Split-K changes FP32 accumulation
order, so numerical equivalence is tolerance-based, not bitwise identity.

## 2. Launch and Matrix Orientation

```text
grid        = 112 CTAs
cluster     = 4 CTAs
clusters    = 28
CTA         = 256 threads / 8 warps
launch      = cooperative; verify capacity for this compiled kernel
MMA         = WGMMA m64nNk16, BF16 inputs, FP32 accumulators
N           = 16 or 32: smallest bucket covering B
K stage     = 256 for output/gate/up; 384 for down
weight ring = two stages
```

For every projection, calculate `weight @ activation.T`. The MMA's **M is
output channels**, and **N is batch**, not sequence length. All CTAs handle
the entire batch bucket; there is no separate grid replication per token.
Zero padded token columns and predicate final writes for `b >= B`.

| Cluster IDs | CTAs | First responsibility | Later responsibility |
|---|---:|---|---|
| 0..15 | 64 | Output projection, split-K reduction, residual/stat publication | Down projection, split-K reduction, output |
| 16..27 | 48 | Stage normalized inputs and gate/up weights | Gate/up + SwiGLU, publish two activation tiles |

The first group changes role after projection. There is no permanently idle
third group waiting for down-projection work.

## 3. Cluster and CTA Mapping

### Projection then down: clusters 0..15

For cluster `c` and CTA rank `r`:

```text
output channels = [64*c, 64*c + 64)

Output projection:
    K = [512*r, 512*r + 512)
    partial = Wo[output channels, K] @ attn[:, K].T
    two K256 stages; accumulator [64, N]

Down projection, wave s in {0, 1}:
    K = [1536*s + 384*r, 1536*s + 384*r + 384)
    one K384 stage per wave; accumulator [64, N] spans both waves
```

The four ranks cover all 2048 attention channels, then all 3072 MLP channels.
They do not duplicate projection weights. Each rank handles 768 down-projection
K elements, split into 384 from each production wave.

For both projections, ranks write their FP32 `[64, N]` partials to local SMEM
and synchronize the cluster. **Reduce-scatter**, rather than rank-0 gather:
rank `r` reads all four partials for output rows `[16*r, 16*r+16)` using DSMEM.
It sums in a fixed rank order and owns those 16 final channels for every token.
Keep peer SMEM alive until all reads finish; a second cluster synchronization
allows the partial buffer to be reused.

After output projection, each rank:

1. Rounds its reduced projection to BF16, then adds the BF16 input residual in FP32.
2. Rounds the sum to BF16 and writes its 16 channels to residual workspace `R`.
3. Converts that rounded residual to FP32 and computes its sum of squares over
   16 channels for each token, writing `S[c*4+r, :]`.
4. Publishes `r_ready[c*4+r]` after its payload and statistics are visible.
5. Changes to its down-projection role; stages down weights while awaiting activations.

After down projection, reduce-scatter the four partials in the same way,
round the projection to BF16, add the corresponding `R` slice in FP32, round
again to BF16, and write final output `Y`.

### Gate/up: clusters 16..27

Let `g = 4*(cluster_id-16) + rank`, so `g` runs from 0 to 47.
Each CTA produces **two 32-channel activation tiles**, one per wave:

```text
wave s              = 0, then 1
activation tile t   = 48*s + g
activation channels = [32*t, 32*t + 32)

packed weight rows  = 32 gate rows, then the matching 32 up rows
GEMM                = weight [64, 1024] @ normalized_input.T [1024, N]
accumulator         = [64, N]
K stages            = four K256 stages per wave
epilogue            = BF16 gate/up, BF16 SiLU(gate), multiply by up, BF16 result
publication         = A[t, :, :] and a_ready[t]
```

Each CTA computes all 1024 input channels: no split-K or cross-CTA reduction in
gate/up. Within a gate/up cluster, ranks are independent output-tile producers.
Clustering them is needed for the uniform launch geometry, not a tensor-core
operation spanning four CTAs.

Pack gate/up weights once into `[96, 16, 64, 64]`, ordered by activation tile,
K64 slab, gate/up row, and K within the slab. This is a row permutation,
not an additional mathematical projection. The first 32 accumulator rows are
gate and the second 32 are up. Use a local SMEM epilogue buffer to pair them;
do not assume both rows land in the same thread's WGMMA registers.

### Why two production waves

If each gate/up CTA accumulated 64 gate and 64 up channels before publishing,
most down-projection work would wait for gate/up to nearly finish. Two smaller
products publish the first half of the MLP channels earlier, without increasing
the total weight bytes or useful GEMM arithmetic.

```text
Projection/down group:  [ O projection + DSMEM reduction ] [ D wave 0 ][ D wave 1 ][ reduce ]
Gate/up group:           [ weights / wait / normalize    ] [ G/U wave 0 ][ G/U wave 1 ]
                                                               |             |
                                                 first A tiles +-------------+ final A tiles
```

This is a dependency sketch, not measured timing. Each CTA advances independently;
there is no global barrier between waves. Down rank `r` consumes tiles
`48*s + 12*r + j`, `j=0..11`. Two neighboring 32-channel tiles supply one K64
stage. It waits for **both** tiles before reading that stage, accumulates it,
and proceeds in fixed K order. All down ranks can consume the first wave;
none is assigned only the late half of the activations. The implementation groups
six K64 slabs into one down stage and acquires all twelve activation tiles in
parallel before staging them.

## 4. Warp Roles

Warps 4..7 form one aligned WGMMA warpgroup. Hopper needs all 128 threads
participating, not a single MMA-issue warp as in the Blackwell reference.

| Warps | Output projection | Gate/up | Down projection |
|---|---|---|---|
| 0..3 | Stage attention operands; lane 0 issues weight TMA | Lane 0 issues weight TMA; normalized input is already resident | Acquire twelve activation tiles, then stage operands; lane 0 issues weight TMA |
| 4..7 | WGMMA | WGMMA | WGMMA |
| All, between GEMMs | DSMEM reduce-scatter, residual/statistics and publication | Normalize input; pair gate/up outputs and publish SwiGLU tiles | DSMEM reduce-scatter and final residual |

Control/loader warps finish their bounded work at each cluster synchronization;
they must not wait for future activation data before joining the projection
reduction barriers. Use role-specific barriers during the pipelined loops,
not a full-CTA barrier that a waiting role cannot reach.

Gate/up threads 0..63 acquire all residual-tile flags before a CTA barrier.
Eight threads per batch row reduce the 64 partial sums and compute inverse RMS.
All threads stage the normalized input once, using vectorized loads/stores and
the WGMMA operand layout. Both gate/up waves read it directly, without another
SMEM copy. Initial gate/up weight TMA currently starts after normalization;
prefetching it earlier regressed B=32 in the initial implementation.

The normalizer does not wait for any down-projection work. Down weights can be
prefetched without waiting for activations, but at most two stages ahead, bounded
by the SMEM ring. Keep the FP32 down accumulator in registers while waiting for
the second activation wave.

## 5. Storage and Pipeline Lifetimes

Logical GMEM workspace at N=32:

| Buffer | Layout | Bytes | Writer / reader |
|---|---|---:|---|
| `R` | BF16 `[N, 1024]` | 64 KiB | Projection ranks / gate-up CTAs and down epilogues |
| `S` | FP32 `[64, N]` | 8 KiB | Projection ranks / gate-up control warps |
| `A` | BF16 `[96, N, 32]` | 192 KiB | Gate-up CTAs / down loaders |
| Ready flags, invocation counters | Int32 | 1088 bytes | Owners / consumers |

`Y`, BF16 `[N,1024]` (64 KiB), is the only output. Keep it separate from `R`;
the published residual remains immutable until all readers finish.
`Z` is never written to GMEM. Gate/up products and split-K partials also stay
off GMEM. No FP32 residual mailbox or separate down-projection output is needed.

Each down rank reads 24 activation tiles, grouped into two K384 stages.
The two input tiles are separate GMEM allocations within `A`; loaders assemble
them into the required WGMMA shared-memory layout. No assumed physical
contiguity between the two tiles is required.

Current SMEM regions at N=32 (plus control/alignment):

| Region | Size | Use |
|---|---:|---|
| Normalized input | 64 KiB | Resident `Z` in WGMMA layout |
| Operand ring | 48 KiB | Two BF16 `[6,32,64]` down stages; also used by output projection |
| Two weight stages | 96 KiB | Two BF16 `[6,64,64]` tiles; gate/up uses four slabs per stage |
| Partial/epilogue region | 8 KiB | FP32 `[64,32]` split-K partial or gate/up row pairing |
| Control and auxiliary region | Under 2 KiB | Barriers, inverse RMS, epoch, padding/alignment |

The main WGMMA accumulator costs 16 FP32 registers per compute thread at N=32,
before descriptors, pointers, and epilogue temporaries. This is a budget, not a
compiler register count. Weight, operand, and partial buffers are shared between
roles; normalized input has a separate allocation. The original 96-KiB budget
was increased to reduce pipeline handoffs with larger K stages.

Weight stages are released only after `wgmma.wait_group` proves no outstanding
MMA reads them. TMA completion is tracked with transaction barriers. Shared
memory written by ordinary stores must be published to the async proxy before
WGMMA reads it; also use the required WGMMA fence/commit/wait sequence.
Complete all async operations before aliasing a region for a new phase.

Future gate/up stages could prefetch weights for wave 1 while wave 0 runs its epilogue,
provided the corresponding weight slots have been released. The epilogue buffer
is distinct from those slots. DSMEM readers must finish before an owning CTA
reuses its partial buffer or exits.

## 6. Cross-Cluster Publication

Use **sequence-tagged tile mailboxes**: the same producer/consumer principle as
sentinel Lamport synchronization, but a ready epoch per tile rather than a
reserved floating-point payload. This preserves all BF16/FP32 bit patterns and
avoids speculatively reading a tile while it is being written.

```text
Producer compute warpgroup:
    write tile payload (and norm partials for R)
    participating-writer barrier, including global-memory ordering
    elected lane: store.release.gpu(ready[tile], epoch)

Consumer control warp:
    load.acquire.gpu(ready[tile]) until it equals epoch
    publish readiness to this CTA's loader warps

Loader warps:
    wait for that local handoff
    ordinary vector global loads -> shared-memory stores
    async-proxy fence + local completion handoff

MMA warpgroup:
    wait for local data completion and weight TMA completion
    issue WGMMA
```

The producer barrier must include **all payload writers**. An elected lane's
release alone does not publish unrelated lanes' uncoordinated stores. Likewise,
an acquire performed by warp 3 needs a synchronized handoff before other warps
read the payload. Use an explicit participating-thread named barrier or a
properly counted mbarrier protocol, not just an ordinary shared flag.

Use ordinary vector loads for changing cross-cluster activations in this design;
TMA is for immutable weights and already-ready launch inputs. Do not copy the
reference's speculative TMA/retry scheme without its full memory-order protocol.

### Repeated launches and CUDA graphs

Initialize ready flags and `cta_epoch[112]` to zero once, outside timing/capture.
At invocation entry, each CTA's designated lane increments its own counter and
broadcasts that epoch locally. Every CTA runs exactly once per invocation, so
all expect the same new epoch without a grid barrier or a reset kernel.

All invocations sharing this workspace must be serialized on one stream.
The preceding kernel has finished all reads before the next overwrites payloads;
old ready flags cannot satisfy a new epoch. Graph replay increments the device
counters too: do not capture a constant host epoch. Separate concurrent graphs
or streams need separate workspace. Do not reuse state after a failed/partial
launch. The prototype uses Int32 counters; a production interface should make
their lifetime/wraparound contract explicit.

### Forward progress

The down consumers and gate/up producers share a grid. An ordinary launch with
112 CTAs is **not a proof** that every required producer can run.

Use a cooperative clustered launch and query active-cluster capacity with the
actual block size, compiled register count, actual SMEM usage, and four-CTA cluster.
Require capacity for all 28 clusters, and verify cooperative launch and CUDA
graph capture/replay in the chosen runtime. Do not silently fall back to a normal
launch. If this residency contract fails, revisit the geometry rather than run
the blocking protocol anyway. Cluster co-scheduling alone only protects DSMEM
dependencies inside one cluster, not these cross-cluster waits.

There is no circular compute dependency: output projection does not wait for
gate/up; gate/up does not wait for down. With the launch residency guarantee,
all waits have independent producers that can make progress.

## 7. Expected Cost and Tradeoffs

| Work | Total BF16 weights | CTAs | Weight bytes per CTA | K64 stages per CTA |
|---|---:|---:|---:|---:|
| Output projection | 4 MiB | 64 | 64 KiB | 8 |
| Gate + up | 12 MiB | 48 | 256 KiB | 16 + 16 |
| Down projection | 6 MiB | 64, reused from projection | 96 KiB | 6 + 6 |

Total useful GEMM work is `2*B*(1024*2048 + 2*3072*1024 + 1024*3072)`:
about 0.369 GFLOP at B=16 and 0.738 GFLOP at B=32. Weights total 22 MiB per
layer. Each weight belongs to one CTA, covering all batch columns.

The expected gains are fewer launches, no GMEM gate/up output, no GMEM normalized
input, and overlap of down wave 0 with gate/up wave 1. This is **not** a design
that eliminates all intermediate GMEM traffic or keeps every SM computing.

Costs that can erase the gain:

- Split-K requires two DSMEM reductions per cluster and changes accumulation order.
- All 48 gate/up CTAs load `R` and normalize it independently: 3 MiB of logical
  BF16 payload reads at N=32, normally reusable in L2, but still real traffic.
- Down reads total 16 copies of `A`: 3 MiB at N=32, again primarily an L2 reuse opportunity.
- Gate/up has 48 compute CTAs, while the other 64 prefetch or consume published work.
  Tensor-core utilization is not guaranteed merely by launching 112 CTAs.
- Two gate/up waves add a second epilogue/publication. If that costs more than
  the overlap saves, test one wave **inside the same full-block kernel**.
- Cooperative residency and the larger fused kernel can affect overlap with
  the vocoder. A standalone win must survive the full engine run.

## 8. Implementation and Evaluation

Implement this complete schedule directly. Do not build separate C, D, and E
custom kernels as prerequisites. Keep the initial scope fixed to these dimensions,
BF16, TP1 and B<=32; no routing, quantization, prefill, or attention replacement.

1. Verify the chosen CUTLASS build exposes Hopper WGMMA, cluster DSMEM, ordered
   global loads/stores, and cooperative clustered launches. Latest documentation
   is not a guarantee about the installed 4.8 wheel. Record any missing API before
   choosing a supported wrapper; do not assume a Blackwell primitive substitutes.
2. Compile the complete role layout and verify resources/residency. Add an optional
   diagnostic timestamp per role boundary, not per inner-loop instruction.
3. Validate the layer output against the upstream Qwen-TTS sublayer sequence
   and an explicit Torch reference preserving the rounding boundaries above.
   Use actual layer weights and captured decode inputs, plus randomized inputs.
   Exercise B=1,8,16,32 and non-bucket sizes such as 7,17,31; verify padded rows
   cannot contaminate live rows. Compare norm/gate-up checkpoints in debug builds
   if final error grows, preserving the BF16 boundaries above.
4. Repeat with changing inputs under graph replay to exercise epochs and buffer
   reuse. Check memory/synchronization diagnostics and actual register spills.
   These are checks of this protocol, not an unrelated general test framework.
5. Time the full upstream-equivalent Torch C+D+E sequence (including a compiled
   baseline) versus the full fused kernel under CUDA graphs. Check the compiled
   baseline's numerics against eager upstream too. Alternate ordering; do not
   include one-time weight packing or compilation.
   Include per-invocation mailbox costs. Report both reused-layer and 28-layer
   weight-rotation measurements: one layer's 22 MiB weights can be unusually hot.
6. Check per-CTA timestamps for `O published`, `G/U wave0 published`, `D wave0 done`,
   `G/U wave1 published`, and `D done`. Confirm real wave overlap and measure the
   slowest producer/consumer before adjusting role counts or tile sizes.
7. Only after the block wins, replace this path in the talker and repeat the
   marked engine trace. Measure talker latency, not just whole TTS throughput,
   since the code predictor and vocoder are unchanged.

## References

- Canonical numerical reference: `qwen_tts.core.models.modeling_qwen3_tts`,
  specifically `Qwen3TTSTalkerDecoderLayer`, `Qwen3TTSRMSNorm`, and
  `Qwen3TTSTalkerTextMLP`.
- [SGLang talker](../../inference/sglang/sglang-omni/sglang_omni/models/qwen3_tts/sglang_model.py)
  and [fused add+RMSNorm](../backends/flashinfer/flashinfer/norm/kernels/fused_add_rmsnorm.py)
  for comparison, not the upstream rounding contract.
- [CUTLASS primitive API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html)
  and [kernel launch parameters](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#kernel).
- [CUDA cluster occupancy and synchronization](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html#thread-block-clusters)
  and [PTX WGMMA/proxy ordering](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
