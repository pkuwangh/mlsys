# Mega-Kernel Fusion Pattern

- [Mega-Kernel Fusion Pattern](#mega-kernel-fusion-pattern)
  - [GPT-OSS BS1 Decode](#gpt-oss-bs1-decode)
    - [RMSNorm + QKV + RoPE + Attention](#rmsnorm--qkv--rope--attention)
    - [Output Projection + Bias + Residual](#output-projection--bias--residual)
    - [MoE: Router and Expert Tail](#moe-router-and-expert-tail)
      - [Router:\*\* one eight-CTA split-K cluster. Norm warps build resident SMEM](#router-one-eight-cta-split-k-cluster-norm-warps-build-resident-smem)
  - [4. Final RMSNorm + LM Head](#4-final-rmsnorm--lm-head)
  - [5. Sampling and Cross-Layer Execution](#5-sampling-and-cross-layer-execution)

## GPT-OSS BS1 Decode

We can fuse a whole layer into 4 kernels:

```text
  -> RMSNorm + QKV projection + RoPE + paged attention
  -> output projection + bias + residual
  -> router (RMSNorm + routing projection + top-4 + softmax)
  || expert tail (quantize + gate/up + activation + down + weighted residual)
```

The router and expert tail overlap on two streams.
After all layers, fused (RMSNorm + LM head) produces logits, followed by
sampler kernel(s).

Key shapes:
- hidden size 2880
- 64 Q heads / 8 KV heads
- head dimension 64
- 4 / 128 experts

### RMSNorm + QKV + RoPE + Attention

```text
h       = norm(x)                         # h: [2880]
[q,k,v] = Wqkv @ h + bias                 # Wqkv: [64x64 + 8x64 + 8x64, 2880]
q,k     = RoPE(q, k, position)            # also append k,v to paged KV cache
S[j]    = dot(q, K[j]) / sqrt(head_dim)
E[j]    = exp(S[j] - max(S))
P[j]    = E[j] / sum(E) + exp(sink - max(S))
o       = P * V
```

**Placement**
- 10 CTAs per cluster, covering 8 Q head + 1 K head + 1 V head
- 1 cluster per KV head

| Cluster ranks | Projection phase | Attention phase |
|---|---|---|
| 0-7 | One Q head each | One KV-sequence partition each, computing all eight Q heads |
| 8 | The K head | Publish the current K row |
| 9 | The V head | Publish the current V row |

Within each CTA,
- warp 0 streams projection weights
- warp 1 prefetches historical KV
- warp 2 issues MMA
- warp 3 handles cluster setup
- warp 4-7 run nomalization, project epilogues, and softmax/reduction.

**Handoffs and overlap:**

- Normalize the small input redundantly in each CTA.
  - weight TMA happens concurrently.
- Projection MMA
  - compute transposed view, `W [64, 2880] @ x.T [2880, 1] -> y.T [64, 1]`
  - split along the K reduction dimension, `W [128, 1440] @ x.T [1440, 2] -> y.T [128, 2]`
    - run `M128 x N16` MMA for `M128 x N2` matmul
    - keep upper-left and lower-right corner of `128x2` matrix
- Projected Q is sent to peer SMEM with `mapa + st.async` so every CTA has all Qs for cluster's KV head.
- Projected / RoPE-d K/V is written to both KV cache and the peer SMEM of the CTA responsible for current token.
- Stage early projection weight chunks into TMEM; reuse the projection weight SMEM slots for later attention stages.
- Attention computation
  - `S.T [128, 8] = K [128, head_dim] @ Q.T [head_dim, 8]` for all 8 Q heads
    - softmax needs cross-thread reduction: `redux_sync` within warp and sync through SMEM
  - `O.T [head_dim, 8] = V.T [head_dim, 128] @ P.T [128, 8]` for all 8 Q heads
- Attention partitions exchange max/sum and O partials through DSMEM.

### Output Projection + Bias + Residual

```text
linear = Wo @ attn.T + bias     # Wo: [2880, 4096] @ [4096, 1]
y      = linear + x
```

**Matmul partitioning**
- each cluster handles M=128 output channels, i.e. column parallel among clusters.
- 4 CTA ranks split along reduction dimension, i.e. row parallel among CTAs.
- each CTA own 32 final channels, takes the partial sums through DSMEM, reduces,
  adds bias, rounds, adds residual, and stores to GMEM.

### MoE: Router and Expert Tail

Two launches use different cluster geometries.

```text
h       = norm(y)                       # [1, 2880]
logits  = Wrouter @ h.T + router_bias   # [128, 2880] @ [2880, 1] -> [128, 1]
experts = top4(logits)
weights = softmax(logits[experts])

for e in topK:
    [g,u] = Wgate_up[e] @ quantize_MXFP8(h.T) + bias1[e]    # [5760, 2880] @ [2880, 1] -> [5760, 1]
    g     = min(g, 7)
    u     = clamp(u, -7, 7)
    a     = g * sigmoid(1.702 * g) * (u + 1)
    z[e]  = Wdown[e] @ quantize_MXFP8(a) + bias2[e]
out = y + sum(weights[e] * z[e])
```

#### MoE Router

- one single 8-CTA cluster
- RMS norm:
  - each CTA redundantly computes sum of squares of the whole vector using 4 warps
  - each CTA only normalizes its assigned split-K input chunks and stores in SMEM
- router MMA:
  - each CTA owns a split-K chunk; router MMA is also a split-K GEMM
  - CTA 0 then gather all partials through DSMEM, adds bias, selects experts
  - publish selected expert indices so weight loading can start
  - compute expert weights with softmax over scores

#### Fused Expert

Each 4-CTA cluster handles
- 2x 128-channel tiles (half gate, half up each) for 4 experts in gate/up phase
- 1x 128-channel tile for 4 experts in down phase and produces the 128-channel tile after weighted sum

Both are column parallel, take full activation and perform full K reduction.
So intermediate results b/w gate/up and down projections need go to GMEM and rely on Lamport sync.

| CTA rank | Gate/up proj + bias + SwiGLU + quantize | Down proj + bias + routing weight | Weighted sum + residual |
|---|---|---|---|
| 0 | Tile 0 for expert 0 & 1 | Tile 0 for expert 0 | Gather activation via DSMEM and sum all experts |
| 1 | Tile 0 for expert 2 & 3 | Tile 0 for expert 1 | Send activation to rank 0 through DSMEM |
| 2 | Tile 1 for expert 0 & 1 | Tile 0 for expert 2 | same as above |
| 3 | Tile 1 for expert 2 & 3 | Tile 0 for expert 3 | same as above |

Within each CTA:
- warp 0-3 handle element-wise ops, e.g. bias, requantization, weighted reduction, residule
  - warp 0 may redundantly compute the input norm while routing runs
- warp 4 fetches expert weights & scales after acquiring expert indices
  - use cross-kernel Lamport sync to wait for routing result
- warp 5 TMA load activation after a delay, without real dependency check
- warp 6 copy block scales from SMEM to TMEM
- warp 7 OMMA
- warp 8-11 validates sentinel values and retry incomplete loads

## Final RMSNorm + LM Head

```text
logits = Wvocab @ norm(x.T)     # [201088, 2880] @ [2880, 1]
```

With persistent kernel, each CTA computes the normalization once and retain the input in SMEM.
Use M64xN8xK16 and only one column is useful.
