# GPU Kernel Learnings

This is the learning log for high-level GPU kernel architecture patterns.

- [GPU Kernel Learnings](#gpu-kernel-learnings)
  - [Kernel Patterns](#kernel-patterns)
    - [Embedding Lookup](#embedding-lookup)
    - [RMS Norm](#rms-norm)
    - [RoPE](#rope)
    - [FMHA-Context](#fmha-context)
    - [FMHA-Decode](#fmha-decode)
      - [Reduction across KV splits](#reduction-across-kv-splits)
    - [Dense GEMM](#dense-gemm)
      - [Bias in epilogue](#bias-in-epilogue)
    - [Expert Router with TinyGEMM](#expert-router-with-tinygemm)
    - [MoE TopK](#moe-topk)
    - [MoE Routing](#moe-routing)
    - [MoE Grouped GEMM](#moe-grouped-gemm)
    - [MoE Finalize](#moe-finalize)
    - [Distributed All-Reduce](#distributed-all-reduce)
  - [Primitives](#primitives)
    - [Warp/CTA Barrier Sync](#warpcta-barrier-sync)
    - [Warp Lane Shuffling](#warp-lane-shuffling)
    - [Warp/CTA Reduction](#warpcta-reduction)
    - [Atomic RMW](#atomic-rmw)
    - [Async Data Movement](#async-data-movement)
    - [Cluster DSMEM](#cluster-dsmem)
  - [Techniques](#techniques)
    - [Async Bulk Copy](#async-bulk-copy)
    - [CTA\_2 GEMM](#cta_2-gemm)
    - [Cluster A/B Multicast](#cluster-ab-multicast)
    - [Split-K GEMM](#split-k-gemm)

## Kernel Patterns

### Embedding Lookup

One CTA per token to gather embedding from the weight table: `output[i] = weight[input_ids[i], :]`.
Each thread owns a strided slice help memory access vectorization and coalescing.


### RMS Norm

`RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight`

A CTA processes one row in two passes:
- load into register and then
  - sum(x*x) for `VEC_SIZE` within a thread
  - warp reduction w/ `nvvm.shfl.sync` butterfly
  - cross-warp reduction with `nvvm.red` from lane-0 into SMEM
- CTA barrier and then load square sum to normalize.


### RoPE

Apply rotation to `Q` & `K` (not `V`), write `K` & `V` into KV cache.

Along the `D` dimension, `theta_i = rope_theta^(-2i/D)`.
Then for a token at sequence position `m`,

```text
x[i]       = cos(m * theta_i) * x[i] - sin(m * theta_i) * x[i + D/2]
x[i + D/2] = cos(m * theta_i) * x[i + D/2] + sin(m * theta_i) * x[i]
```

Assume `D=64, dtype=bfloat16`, the work is partitioned as
- each thread handles `128 / 16 = 8` elements, `128` corresponds to `LDG.128`
- `64 / 8 = 8` threads handle one Q head
- each warp handles `32 / 8 = 4` Q heads
- each CTA handles `4 * 4 = 16` Q heads

### FMHA-Context

One work item handles
- one sequence in the batch
- two consecutive Q tiles from one head

Warp specialization:
- 0-3: softmax0 for `Q0`
- 4-7: softmax1 for `Q1`
- 8-11: correction
- 12: MMA
- 13: TMA load
- 14: epilogue TMA store
- 15: persistent scheduler

Dataflow:
- scheduler publish `(seq_tile, head, batch)` coordinates
- TMA load `Q0`, `K0`, `Q1`, `V0` and then `K_i` / `V_i`
- MMA compute `Q0K0`, `Q1K0`, `P0V0`, loop(`Q0Ki`, `P1Vi-1`, `Q1Ki`, `P0Vi`), `P1Vi`
- softmax-0/1 consume `S`, publish old/new `row_max`, softmax and write packed `P` back to TMEM
- MMA consume `P` (overlapped w/ `S` in TMEM) and accumulate `O` in TMEM
- correction takes old/new `row_max` to rescale `O`, and final-normalize `O` w/ `row_sum`
- epilogue TMA store O tile

**Skip correction**

When `|new_row_max - old_row_max| < threshold`, the correction can be skipped.
We can do this skip at the warp granularity, using `nvvm.vote.sync`.

**MUFU emulation**

`MUFU.EX2` is often the bottleneck as 4 threads share one MUFU lane.
So we can replace a subset of them with FMA operations. The key here is to
make sure instructions are properly interleaved, given the in-order issue,
out-of-order execution nature of the scheduler.


### FMHA-Decode

One work item handles
- one sequence in the batch
- multiple Q heads that share the same KV head
- a subset of KV sequence tiles

Warp specialization:
- 0: MMA1, K @ Q -> S in TMEM
- 1: MMA2, P @ V -> O_partial in TMEM
- 2: TMA K/V load, also final reduction
- 3: TMA Q load, also final O store
- 4-7: softmax warpgroup
- 8-11: correction warpgroup

Dataflow:
- TMA Q -> SMEM, TMA K -> SMEM
- MMA Q @ K^T -> S in TMEM
- softmax S -> P in both TMEM/SMEM, running `row_max` in SMEM
- TMA V -> SMEM
- MMA P @ V -> O in TMEM
- correction load `row_max` for correction and load P -> sum reduction for final normalization
- TMA store O

**MMA Shape**

The shape of Tensor Core MMA typically requires a big M (64, 128) while N is more
flexible (8 ~ 256 at a step of 8).
Since Q is `[grouped_heads, head_dim]`, instead of doing `S = Q @ K^T`, we transform it as
`S^T = K @ Q^T`. Then `mma_tile_m` is K/V sequence tile which can be e.g. 128.
Note both Q & K are K-Major in SMEM, no real transpose is needed either doing QK or KQ.

With above transformation, P is also in the transposed state; so maybe cleaner to do
`O^T = V^T @ P^T`.

**Softmax and Correction**

With `S^T` in TMEM, it affects how softmax warpgroup does `row_max` reduction since
a `row` now spreads 128 TMEM lanes/rows and will be loaded into 128 threads. So,
- first `nvvm.redux.sync` to reduce within a warp
- then `nvvm.red` to reduce in SMEM across warps
  - each lane handling one S row which corresponds to one Q head

As `MUFU.EX2` in softmax makes it the bottleneck, the `row_sum` reduction can be
moved into correction warpgroup by storing the full P back to TMEM.
In the correction warpgroup, it only does **per-warp** `row_sum` reduction, and
the warpgroup-level reduction is done later during final reduction.

#### Reduction across KV splits

With local S row max `m_i`, P row sum `l_i` and partial O `o_i`, the final normalization does

```text
m = max_i(m_i)
l = sum_i(exp2(m_i - m) * l_i)
o = sum_i(exp2(m_i - m) * o_i) / l
```

To reduce across KV splits, one can
- run a separate reduction kernel through GMEM workspace
  - decode-attn's reduction warp updates global row_max with `nvvm.red`
  - second reudction kernels loops over KV splits.
- perform cluster reduction
  - schedule all KV splits of one sequence onto the same cluster
  - reduction warp does DSMEM butterfly across KV splits for `M_global` & `L_global`
    - exchange/reduce per-CTA `M` (local `row_max`) in butterfly fashion
    - sum local per-warp `L` into per-CTA `L` and correct `L_local` with `M_global`
    - exchange/reduce per-CTA `L` (per-CTA `row_sum`) in same way
    - write the final correction alpha back to SMEM
  - final correction for each CTA's `O_partial` and store to SMEM
  - final `O` reduction in GMEM with TMA reduce-add using `nvvm.cp.async.bulk.tensor.reduce`

**Butterfly Reduction**

```python
cluster_size = 8
reduction_steps = 3
for step in range(reduction_steps):
  offset_as_xor_mask = 1 << step                # 1, 2, 4
  peer_idx = my_split_idx ^ offset_as_xor_mask  # 0<->1, 0<->2, 0<->4
  peer_buffer_ptr = nvvm.mapa(local_buffer_ptr, peer_idx)
  peer_mbar = nvvm.mapa(local_mbar, peer_idx)
  # send current values to the peer CTA's DSMEM mailbox
  ptx::st.async(peer_buffer_ptr, *values, peer_mbar)
  # order/post my outgoing async store before waiting for my incoming values
  ptx::fence.acq_rel.CTA
  nvvm.mbarrier.try.wait.parity(local_mbar, 0)
  # load data written by peer and reduce into *values
```


### Dense GEMM

**Warp-specialized GEMM pipeline**

```text
MMA Tile:  256x256x64 on CTA_2
per CTA:   A/B 128x64, C 128x256
pipeline:  ab_stages=6 (16KB tiles, 16KB*2*6=192KB), acc_stages=2 (512 TMEM cols / tile_n)
  TMA warp
  -- 6 slots --
  MMA warp
  -- 2 slots --
  Epilogue warp group
```

**M-axis split**

If `mma_tiler_m > 128` per CTA, e.g. `192x192x64`, `M=192` needs to be split for `CTA_1` MMA.
- TMA warp still loads full `192x64` A/B tiles
- MMA warp consecutively queue mma0 (128x192) and mma1 (64x192)

**N-axis split**

If `mma_tiler_n > 256`, e.g. `256x352x64`, `N=352` needs to split for `CTA_2` MMA.
- TMA warp (across 2 CTAs) loads
  - a full `256x64` A tile still;
  - but separate `192x64` B0 & `160x64` B1 tiles.
    - because the first 192 rows are split b/w 2 CTAs, i.e. CTA-0 would load `B[0:96]` and `B[192:272]`.
- MMA warp consecutively queue mma0 (256x192) and mma1 (256x160).

**TMEM column overlap**

For `nvfp4` MMA, scale also takes TMEM space. So for a `256x256x64` MMA, we can allocate
- `TMEM[0:256)` for `acc_stage[0]` and `TMEM[192:448)` for `acc_stage[1]`;
  - 64 cols overlap b/w 2 stage slots.
- remaining columns for scales.

In epilogue warp, `tcgen05.ld` the overlap chunk `TMEM[192:256)` first;
then after `tcgen05.wait` the epilogue can free the acc_stage slot.

#### Bias in epilogue

To add bias in epilogue, we can stage the final output in SMEM and have a separate
warp to launch TMA store.

### Expert Router with TinyGEMM

`logits[tokens, experts] = hidden[tokens, K] @ weight[experts, K]^T + bias[experts]`

Considering the GEMM shape, `M = tokens` & `N = experts`,
- each work tile covers `C[16, 8]`, which would be too small for warpgroup-level MMA.
  - note we can flip M & N given `C^T = B^T @ A^T` if `C = A @ B`
- But `K = hidden_size` is still large.

So, we do Ampere-style warp-level MMA, i.e. each warp uses `mma.sync.m16n8k16` on
a strided slice along K dimension. In the end, each warp has a partially-accumulated
`C[16, 8]` and then warpgroup-level reduction is done through SMEM by warp-0; note
this is effectively Split-K GEMM within a warpgroup.

For each of the 4 compute warp, there is TMA warp for loading activations and weights.

### MoE TopK

`logits[tokens, experts] -> topk_weights[tokens, topK], expert_idx[tokens, topK]`

One warp handles one or more token/row in K iterations:
- first perform thread-local argmax
- then butterfly shuffling to find `(max_val, expert)` on lane-0
- mask found element and look for next top expert

```python
for offset in [16, 8, 4, 2, 1]:
    other_max = shuffle_xor(max_val, offset)
    other_expert = shuffle_xor(expert, offset)
    if other_max > max_val:
        max_val = other_max
        expert = other_expert
    elif other_max == max_val:
        if other_expert < expert:
            expert = other_expert  # tie-break by index
```


### MoE Routing

Instead of having a standalone topK kernel, routing kernel builds expert-grouped
index maps and emits the grouped GEMM dispatch table.

```text
logits[tokens, experts]
  -> topk_weights[tokens, topK], expert_idx[tokens, topK]
  -> expert-grouped permutation indices
  -> grouped-GEMM dispatch schedule
     - padded group index: {expert index, valid M/N limit}
``` 

The algorithm is:

- Same as [MoE TopK](#moe-topk) above, use one warp per token to find topK experts.
  - but weights should go through `softmax` either before or after topK.
- Each warp writes TopK results to SMEM, then cluster barrier makes them visible via DSMEM.
- Each thread atomicAdd into SMEM to build histogram of expert assignment.
  - an `expanded_idx` points to one selected expert of one token in a flattened list.
  - each thread handles 1~2 such `expanded_idx`s.
  - now each block/CTA has an per-expert token count.
- Cross-block histogram reduction via DSMEM reads.
  - each thread handles 1 expert by adding token count from all cluster ranks.
  - all blocks/CTAs do the same so all blocks/CTAs have all the information.
- Block-level exclusive prefix sum computes expert offsets in terms of tiles.
  - `num_cta = ceil(expert_count / TILE_TOKENS_DIM)`.
  - each expert will process its assigned tokens in some number of tiles (GEMM CTA offset)
  - each thread needs to find the tile index (GEMM CTA offset) for its corresponding expert.
    - first, calculate warp-local offset with `ShflKind.up` and write `warp_total` into SMEM
    - second, warp-0 loads 32 `warp_total` to calculate the block-local per-warp offset and
      write `block_total` into SMEM.
  - each block in the cluster calculate the same info.
- Write GEMM dispatch config.
  - each thread handles tiles (GEMM CTAs) for one expert; these tiles are cluster-striped.
  - this builds the mapping from GEMM CTA slot to
    - local expert index in `cta_to_batch_idx`
    - valid token range in `cta_to_mn_limit`
- Derive expert token offset from (GEMM) CTA offset and writes to SMEM.
- Write permutation indices.
  - now each thread handles 1~2 `expanded_idx`s.
  - this builds the mapping b/w `permuted_idx` to `expanded_idx` and `token_idx`.

Now you should realize this kernel launches a single CTA cluster.
- for small-token path, an efficient variant launches a single CTA and rely on SMEM instead of DSMEM.
- for large-token path, many CTAs use global memory plus kernel-boundary/PDL synchronization.


### MoE Grouped GEMM

- `FC1`:
  - if activation is already grouped and padded, which would include
    `tile_idx_to_expert_idx` and `tile_idx_to_mn_limit` mappings, this is simple
    GEMM with a known tile schedule.
  - if activation is in original sequence order, gather into expert-grouped rows,
    pad each expert to `tile_m`; similarly a `tile_idx_to_expert_idx` mapping would
    be populated by the router so scheduling is simple.
  - if activation is already grouped but packed, scan group boundaries in the scheduler.
- `FC2`:
  - input activation should already be grouped/gathered; then depending on whether it is
    ragged/packed, either scan for tiles or simply dispatch tiles in scheduler.

**Token gather and tile padding**

If activation input is in original token order, we need to gather on the fly.
For `topK = 2`, the grouped layout could be:

```text
gathered row:     0  1  2  3  4  5
original token:   2  0  1  0  2  1
expert id:        0  0  0  1  1  1
topK slot:        0  1  0  0  1  1
token_id_mapping: 4  1  2  0  5  3  # original_token_idx * topK + topK_slot
```

Each gathered row copies from GMEM into the staged SMEM tile with `cp.async`.

```python
grouped_row = tile_coord_m * tile_m + local_row
original_token_idx = token_id_mapping[grouped_row] // topK
# conceptually
A_smem[local_row, :] = A_original[original_token_idx, :]
```

**Tile scheduling for ragged input**

Grouped GEMM maps a persistent linear index into a ragged sequence of expert groups.
Given `cu_seqlen[B+1]` or `first_token_offset[G+1]`, we need map a `linear_idx` as
`group_idx` and `tile_idx`.
To speedup the scan, a warp scans 32 groups in parallel.

```python
# each lane owns one group
group_idx += lane_idx
group_begin = first_token_offset[group_idx]
group_end = first_token_offset[group_idx + 1]
# get the tiles per group
group_tiles = ceil_div(group_end - group_begin, tile_m) * clusters_along_n
# get the prefix for each group in terms of tiles
prefix = group_tiles
for delta in [1, 2, 4, 8, 16]:
    other = nvvm.shfl_sync(0xFFFFFFFF, prefix, delta, 0x00, ShflKind.up)
    if lane >= delta:
        prefix += other
# now we get group begin/end index in terms of tiles
group_tile_begin += prefix - group_tiles
group_tile_end = group_tile_begin + group_tiles
# first matching lane wins
winner_mask = nvvm.vote_sync(0xFFFFFFFF, linear_idx < group_tile_end, VoteSyncKind.ballot)
winning_lane = bfind_u32(brev(winner_mask), shift=BfindShift.SHIFTAMT)
# broadcast the group info from the winning lane; this is the base index for next round
group_idx = nvvm.shfl_sync(0xFFFFFFFF, group_idx, winning_lane, 0x1F, ShflKind.idx)
# also set the tile index to the winning group for work tile and next round
group_tile_begin = nvvm.shfl_sync(0xFFFFFFFF, group_tile_begin, winning_lane, 0x1F, ShflKind.idx)
```

**FC1 SwiGLU epilogue**

FC1 computes both `up` and `gate` projections in one GEMM.

```python
# Load paired epilogue subtiles from TMEM:
#   up   = acc[:, even subtile blocks]
#   gate = acc[:, odd subtile blocks]

# SwiGLU math
up = up + up_bias
gate = gate + gate_bias
up = max(min(up, swiglu_limit), -swiglu_limit)
gate = min(gate, swiglu_limit)
exp_neg = exp2(-gate * swiglu_alpha_log2e, fastmath=True)   # exp(-gate * swiglu_alpha)
sigmoid = rcp(one + exp_neg, approx=True, ftz=True)         # 1 / (1 + exp_neg)
out = fma(up, gate, gate) * sigmoid                         # (up + 1) * gate * sigmoid

# Store activated subtile to SMEM, then launch TMA store to GMEM.
```

**FC2 finalize epilogue**

FC2 can fuse finalize into the epilogue:
- scatter valid grouped rows back to the original token row
- accumulate across topK experts.

```python
expanded_idx = permuted_idx_to_expanded_idx[abs_row]
original_token_idx = expanded_idx // topK
for col in subtile_cols:
    value = tmem_acc[epilogue_thread_row, col] + bias[expert_idx, col]
    atomic_add(out[original_token_idx, col], value)     # red.relaxed.gpu.global
```


### MoE Finalize

Finalize maps grouped expert outputs back to original token order and reduces
the `topK` expert contributions.
- standalone finalize reads an already-materialized `expert_output` buffer.
- each CTA owns one token row and each thread handles a strided chunk of hidden elements.
  - reduction across `topK` is done locally within each thread.


## Communication Kernels

### All-Reduce

**One-shot all-reduce via Lamport buffers**

Each rank pushes its local vector into every peer's communication buffer, then
polls its own buffer until every source-rank slot is ready.
Multiple buffer slots available for repeated launches.

```python
# publish
peer_buffers[dst][slot, my_rank, i] = input[i]
# clear next slot
local_buffer[slot + 1, my_rank, i] = sentinel_value
# wait for data from peers to be ready
while local_buffer[slot, source_rank, i] != sentinel
out[i] += local_buffer[slot, source_rank, i]
```

**Two-shot all-reduce with NVSwitch multimem**

Each rank handles `1/world_size` of the output buffer,
- first use `multimem.ld_reduce` to reduce my chunk across peers.
- then use `multimem.st` to broadcast my chunk to peers.
- after a barrier, thread-0 of every CTA increments the flag for each peer.
  - signals `my_rank` at block_idx=X has done.
  - this write should use release semantics with system scope.
- finally wait for all peer ranks has written the flag.


## Primitives

### Warp/CTA Barrier Sync

Use `bar.warp.sync` for warp-scope or sub-warp rendezvous.
This operation also guarantees memory ordering among participating threads.

```python
# Full warp: all lanes wait before reading data produced by lane 0
if lane == 0:
    smem[0] = value
nvvm.bar_warp_sync(FULL_MASK)
out[lane] = smem[0]

# Sub-warp: only lanes named in the mask participate
if lane < 16:
    nvvm.bar_warp_sync(0x0000FFFF)
else:
    nvvm.bar_warp_sync(0xFFFF0000)
```

Use `barrier.cta.sync` for blocking CTA-scope rendezvous and
`barrier.cta.arrive` for split-phase producer/consumer handoff.

```python
# Producer warp: signal arrival and continue.
if warp == PRODUCER:
    nvvm.barrier_cta_arrive(barrier_id=1, thread_count=2 * WARP_SIZE)

# Consumer warp: block until producer + consumer warps have arrived.
if warp == CONSUMER:
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=2 * WARP_SIZE)
    value = smem[0]
```

### Warp Lane Shuffling

Use `shfl.sync` for warp-local register exchange.

```python
# ShflKind.up: lane i reads from lane i - delta; low lanes clamp.
prev = nvvm.shfl_sync(0xFFFFFFFF, value, delta, 0x00, ShflKind.up)

# ShflKind.idx: every lane reads from an absolute source lane.
broadcast = nvvm.shfl_sync(0xFFFFFFFF, value, source_lane, 0x1F, ShflKind.idx)

# ShflKind.bfly: offset is XOR-ed with the calling lane's index.
for delta in [16, 8, 4, 2, 1]:
    value += nvvm.shfl_sync(0xFFFFFFFF, value, delta, 0x1F, ShflKind.bfly)
```

### Warp/CTA Reduction

Use `barrier.cta.red` for CTA-level predicate reduction at a barrier.

```python
# CTA-wide any/all predicate reductions through named barriers
# result broadcast to all participating threads
any_lane0 = nvvm.barrier_cta_red(
    tx == 0,
    barrier_id=0,
    kind="OR",
    thread_count=THREADS,
)
```

Use `vote.sync` for warp-local predicate reduction and lane-mask construction.

```python
# ANY: true if at least one masked lane votes true.
# ALL: true if every masked lane votes true.
# UNI: true if all masked lanes agree, either all true or all false.
any_pos = nvvm.vote_sync(0xFFFFFFFF, per_thread_condition, VoteSyncKind.any)

# BALLOT: bit i is set when lane i votes true.
positive_mask = nvvm.vote_sync(0xFFFFFFFF, per_thread_condition, VoteSyncKind.ballot)
```

`redux.sync` is the numeric peer of `vote.sync`: each lane contributes one
register value, and all masked lanes receive the reduced scalar.
This replaces some explicit shuffle-tree reductions, except for e.g. f32 sum.

```python
# Integer ops: ADD, MIN, MAX, AND, OR, XOR, UMIN, UMAX.
warp_sum = nvvm.redux_sync(lane_value, ReduxKind.ADD, 0xFFFFFFFF)

# Float abs-max: common scale primitive for FP8/MXFP8 quantization.
amax = nvvm.redux_sync(x, ReduxKind.FMAX, 0xFFFFFFFF, abs=True)
```

Use `red` for memory reduction: every thread contributes a value and reduce into
a global or shared cell.

```python
# out_ptr needs initialization
nvvm.red(
    nvvm.ReductionOp.ADD,
    nvvm.ReductionType.S32,
    out_ptr,
    value,
    mem_order=nvvm.MemOrderKind.RELAXED,
    mem_scope=nvvm.MemScopeKind.GPU,
)
```

### Atomic RMW

Use `atomicrmw` to **atomically** perform an operation to a memory location
**and** return the old value.

```python
# return value is caller's offset in the total count
offset_in_bucket = nvvm.atomicrmw("add", counter_ptr, 1)
```

### Async Data Movement

`cp.async.shared.global` copies per-thread slices from global to shared memory.
The same copy can be completed with either wait groups or an mbarrier arrival.

```python
nvvm.cp_async_shared_global(smem_ptr, gmem_ptr, 16, nvvm.LoadCacheModifierKind.CG)

# for same issuing threads to block till copy is done
nvvm.cp_async_commit_group()
nvvm.cp_async_wait_group(0)

# Or: for another warp to wait on an mbarrier
nvvm.cp_async_mbarrier_arrive(full_bar, noinc=True)
```

### Cluster DSMEM

`mapa` translates a local shared-memory address into the same shared-memory
offset in another CTA in the cluster.

```python
if tx == 0:
    # signal the peer CTA
    peer_mbar = nvvm.mapa(mbar, next_rank)
    nvvm.mbarrier_arrive(peer_mbar)
```

`store_async` can be used as a one-shot mailbox handoff inside a CTA cluster.

```python
if tx == 0:
    # sender: map both the peer mailbox and peer mbarrier, then async-store payload.
    peer_slot = nvvm.mapa(mailbox.data_ptr(), next_rank)
    peer_bar = nvvm.mapa(recv_bar, next_rank)
    nvvm.store_async(peer_slot, Int32(1000) + rank, peer_bar)
```

## Techniques

### Async Bulk Copy

```python
# Producer warp: global -> shared
s = k % NUM_STAGES
# prologue pre-signal all empty bars to flip the phase to 1, making all empty slots free
parity = (k // NUM_STAGES) & 1
# producer waits for an empty slot
while not nvvm.mbarrier_try_wait_parity_timelimit(
    empty_bar + s, parity, 10_000_000
):
    pass
# register the expected bytes and launch TMA
nvvm.mbarrier_arrive_expect_tx(full_bar + s, TILE_BYTES)
nvvm.cp_async_bulk_shared_cluster_global(smem_ptr, gmem_ptr, full_bar + s, TILE_BYTES)

# Consumer warp: shared -> global
s = k % NUM_STAGES
parity = (k // NUM_STAGES) & 1
# consumer waits for a full slot
while not nvvm.mbarrier_try_wait_parity_timelimit(
    full_bar + s, parity, 10_000_000
):
    pass
nvvm.cp_async_bulk_global_shared_cta(gmem_ptr, smem_ptr, TILE_BYTES)
# commit a TMA store group
nvvm.cp_async_bulk_commit_group()
# wait for 1 group & release empty slot
nvvm.cp_async_bulk_wait_group(1)
nvvm.mbarrier_arrive(empty_bar + s)
```

### CTA_2 GEMM

Say the MMA tile is `256x256x64`, A & B tiles are split between 2 CTAs:
- CTA-0: `A[0:127, 0:64]` and `B[0:64, 0:127]`
- CTA-1: `A[128:255, 0:64]` and `B[0:64, 128:255]`

For TMA, let both CTAs arrive on the leader CTA's mbar, by clearing bit-24 of each
CTA's local mbar pointer.  Because bit-24 holds the LSB of `cluster_ctarank`, and
hence `mbar[24]=0` points to the leader rank of each 2-SM group.

For MMA, launch from leader CTA only but need signal both CTAs on `ab_empty_mbar`
and `acc_full_mbar`, by setting the `multicast` mask with `tcgen05.commit`.

For epilogue, CTA-0 gets `C[0:127, 0:255]` and CTA-1 gets `C[128:255, 0:255]`.

### Cluster A/B Multicast

In a cluster of shape `(cluster_m, cluster_n)`,
- A is shared across `cluster_n` CTAs - same M rows, different N cols
- B is shared across `cluster_m` CTAs - same N cols, different M rows

TMA is launched from the *leader* rank of each direction, `multicast` mask routes
both data and `complete_tx` to corresponding ranks.

MMA commits to all CTAs as it tries to signal empty mbar for both A & B.

### Split-K GEMM

For small `N` with large `K`, split `K` across CTAs:
- each of `split_k_factor` CTAs covers `ceil(K / split_k_factor)` tiles
- partial outputs are stored to `partial_c[split_k_factor, M, N]`.

Then a separate reduction kernel sums along the split-K axis for each element.
