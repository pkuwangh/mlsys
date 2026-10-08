# Low-Latency GPU Techniques

Low latency is about shortening the dependency chain: start independent work
early, hand off results at the smallest useful granularity, and avoid launches
or copies that only move data between stages.

The examples use CUTLASS primitive-style pseudocode:

```python
import cutlass
from cutlass.experimental import primitives as prims
```

Descriptor construction and indexing are abbreviated. Helpers such as `wait`,
`issue_mma`, and `atomic_fetch_add` describe operations rather than literal APIs.
Barrier initialization, arrival counts, and phases must match the participating
threads and asynchronous operations in a complete kernel.

## 1. Start Independent Work Before Waiting

A wait should block only the operations that need its result. Load static
operands, prepare descriptors, and initialize unrelated buffers before it.

```python
# Epilogue warp: fetch independent operands while the MMA is still running.
bias = prims.load_ext(bias_ptr, dtype=cutlass.Float32, order="relaxed", scope="gpu")
wait(mma_done, phase)
acc = prims.tcgen05_ld("32x32b", acc_tmem, num=8)
prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
store_output(acc + bias)
```

The same principle applies to the kernel prologue. With warp specialization,
the TMA warp can begin weight loads while other warps normalize inputs or stage
metadata. Use role-specific barriers instead of making every warp wait for all
initialization to finish.

Also move unrelated work *after* an important publication point. If MMA needs
normalized input, publish that input before resetting scratch buffers or
computing metadata needed only by the epilogue.

**Tradeoff:** early loads keep registers live longer. Check register pressure
and spills, and do not hoist reads across a dependency on their producer.

## 2. Overlap Consecutive Kernels with PDL

Programmatic Dependent Launch (PDL) lets a consumer start independent preparation
before its producer finishes, while retaining a wait before dependent reads.

```python
# Producer kernel: every CTA reaches the launch signal.
first_part_of_computation()
prims.griddepcontrol("launch_dependents")
finish_computation_and_store_output()

# Consumer kernel, launched on the same stream with use_pdl=True.
prefetch_weights_and_prepare_buffers()
prims.griddepcontrol("wait")
load_producer_output()
compute()
```

The signal means **the consumer may start**, not **the output is ready**.
The consumer's wait resolves its producer-kernel dependency, including output
visibility. Signal where the producer has useful work left to overlap, and
place the consumer's wait immediately before the first dependent access.

Overlap is opportunistic: available resources still determine when the
consumer runs. The producer must not rely on the consumer starting in order
to finish. See [CUDA PDL semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
and [the PDL tutorial](tutorial_pdl_101.py).

## 3. Consume Partial Results with Lamport Sync

When a consumer needs only a chunk of a producer's output, waiting for the
whole producer is unnecessarily coarse. Reserve a payload bit pattern as "not
ready" and let the real data replace it. For floating-point data, negative zero
(`-0.0`) is a useful sentinel: producers replace legitimate negative zeros with
positive zeros before publishing. This assumes the computation does not need
to preserve the sign of zero.

```python
# FP32 mailbox viewed as aligned UInt32 words, initialized before launch.
SENTINEL = cutlass.Uint32(0x80000000)  # FP32 -0.0

# Producer: bitcast preserves bits; it is not a numerical conversion.
bits = bitcast_uint32(produce_element(index))
if bits == SENTINEL:
    bits = cutlass.Uint32(0)  # Publish +0.0 instead of -0.0.
prims.store_ext(bits.ir_value(), mailbox.subview(index), order="release", scope="gpu")

# Consumer: the observed payload is also the readiness signal.
bits = SENTINEL
while bits == SENTINEL:
    bits = prims.load_ext(mailbox.subview(index), dtype=cutlass.Uint32, order="acquire", scope="gpu")
consume(bitcast_float32(bits))
```

Initialize every mailbox word to `SENTINEL`, not zero. Test the **bits**, since
floating-point comparison treats `-0.0 == +0.0`. The encoding depends on the
dtype: FP8 E4M3 uses `0x80`. Other payloads can reserve different sentinels;
negative zero is not a requirement of the protocol.

For a vector or quantized tile, every required element and scale must be ready;
checking only the first element is insufficient. Each element needs an
appropriate publication protocol. A TMA-based consumer can validate a landed
tile and reload it if sentinels remain, but TMA completion alone does not mean
the source data was ready. Its producer/TMA memory-ordering rules also need to
be satisfied; scalar acquire polling is not a substitute for those rules.

This works across CTAs or across kernels, but **data readiness does not guarantee
forward progress**. Waiting consumers must not prevent producers from running.
A small CTA cluster supplies a co-scheduling guarantee for a cross-CTA example;
ordinary grid size or separate streams alone do not.

Reset sentinels only after old readers finish and before new writers begin.
Otherwise stale data can look ready, or a reset can overwrite new data.
See [the Lamport tutorial](tutorial_lamport_101.py).

## 4. Use DSMEM for Cluster-Local Handoffs and Reductions

CTAs in a cluster can deliver results directly into peer shared memory with
`mapa` and `st.async`. A transaction barrier notifies the receiver when the
expected bytes have arrived, avoiding payload polling and a global-memory
round trip.

```python
# Every CTA allocates mailbox and ready at matching shared-memory offsets.
if rank == 0 and tid == 0:
    prims.mbarrier_init(ready, 1)
    prims.mbarrier_arrive_expect_tx(ready, (CLUSTER_SIZE - 1) * 4)
prims.fence_mbarrier_init()
prims.barrier_cluster_arrive()
prims.barrier_cluster_wait()

partial = compute_this_ctas_partial_sum()
if rank != 0 and tid == 0:
    prims.st_async(
        prims.mapa(mailbox.subview(rank), 0),
        partial,
        mbar=prims.mapa(ready, 0),
        type=prims.StAsyncType.F32,
        scope=prims.StAsyncMbarrierScope.CLUSTER,
    )
if rank == 0 and tid == 0:
    wait(ready, phase=0)
    store_output(partial + sum(mailbox[1:CLUSTER_SIZE]))

# Keep peer shared memory alive until all accesses finish.
prims.barrier_cluster_arrive()
prims.barrier_cluster_wait()
```

One use is **split-K GEMM**: partition a long reduction across several CTAs,
then combine partial accumulators inside the cluster. This helps small-output
GEMMs that otherwise launch too few CTAs. It avoids a separate reduction kernel.

For a larger result, distribute ownership across receiving CTAs rather than
making one CTA reduce everything. The same handoff can pass tiles between
different operations, not just reduce sums.

[tutorial_dsmem_101_reduce.py](tutorial_dsmem_101_reduce.py) demonstrates
gather/reduce. The [kernel-pattern notes](gpu-kernel-pattern.md#split-k-gemm)
compare these with a separate global-memory split-K reduction kernel.

## 5. Extend the Prefetch Window with TMEM

A shared-memory ring can fill while MMA is waiting for another operand. On
Blackwell, an MMA form that supports an A operand in TMEM can move prefetched
A tiles there, freeing SMEM slots for the next loads.

```python
# MMA warp, before the other operand is ready.
wait(a_full[slot], phase)
prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
if prims.elect_sync():
    prims.tcgen05_cp(copy_shape, a_tmem_tile, a_smem_descriptor)
    # Release SMEM only after the asynchronous copy has read it.
    prims.tcgen05_commit(a_empty[slot])

wait(b_ready, phase)
issue_mma(a_from_tmem=a_tmem_tile, b_from_smem=b_descriptor, acc=acc_tmem)
```

This adds buffering without increasing the SMEM ring. The TMA producer can
continue loading while the second operand is being prepared. TMEM capacity,
supported operand layouts, and copy overhead limit the benefit; this is not
a general-purpose replacement for shared memory.

[tutorial_tmem_101_staging.py](tutorial_tmem_101_staging.py) compares two SMEM
slots with and without two additional TMEM slots while B is delayed. Each CTA
reads six 64-KiB tiles; the grid spans all SMs with an A working set larger than
twice L2 capacity.

More generally, reuse storage across phases with disjoint lifetimes. A weight
ring can later become an attention workspace, for example. Reuse is safe only
after the last asynchronous reader or writer completes, not merely after its
instruction is issued.

## 6. Prefetch the Stable Part, Fill the Late Part Later

An input tile may contain mostly stable data and a small portion still being
produced. Avoid making the entire load wait for that portion.

```python
# Loader: fetch immutable rows while a producer computes the missing row.
prefetch_stable_rows(tile, excluding=late_row, completion=history_ready)

# Producer: send the late row through a separate synchronized handoff.
publish_late_row(row_data, completion=late_ready)

# Consumer: assemble the tile before MMA reads it.
wait(history_ready, phase)
wait(late_ready, phase)
copy_late_row_into_tile(tile, row_data)
publish_smem_to_async_consumers()  # Required proxy fence and warp/CTA handoff.
issue_mma(tile)
```

Examples include appending a new time step to a cached sequence or assembling
a tile from independently produced regions. Keep concurrent reads and writes
well defined: do not assume it is safe to race a global-memory load with a
store simply because the loaded value will be overwritten later. Also wait
for the prefetch to finish before patching its destination.

## 7. Combine Small Independent MMAs

For very small matrix dimensions, instruction issue and synchronization can
cost more than extra arithmetic. Embed two independent products into one
larger MMA and keep the useful output regions.

```python
# Desired products: C0 = A0 @ B0, C1 = A1 @ B1.
A_stacked = stack_rows(A0, A1)
B_stacked = stack_columns(B0, B1)
issue_mma(A_stacked, B_stacked, C_stacked, clear_accumulator=True)
wait_mma_completion()

# C_stacked = [[A0@B0, A0@B1],
#              [A1@B0, A1@B1]]
C0 = C_stacked.top_left
C1 = C_stacked.bottom_right
```

This trades unused cross-products and a larger accumulator for fewer MMA
instructions and pipeline transitions. It is attractive for narrow,
instruction-bound work, not an already compute-saturated GEMM. Preserve each
product's accumulation order when numerical parity matters.

## 8. Claim Persistent Work Ahead of Compute

Equal tile counts do not guarantee equal completion times. A dynamic work pool
lets faster CTAs take more tiles and slower CTAs take fewer.

```python
# Persistent CTA: the loader warp claims work ahead of MMA and epilogue.
if warp_id == LOADER:
    wait(schedule_slot_empty[slot], phase)
    tile = cutlass.Int32(STOP)
    if prims.elect_sync():
        ticket = atomic_fetch_add(next_tile, 1)  # Pseudocode: device-scope atomic.
        tile = ticket if ticket < NUM_TILES else STOP
        schedule[slot] = tile
        prims.mbarrier_arrive(schedule_ready[slot])
    tile = broadcast_from_elected_lane(tile)
    if tile != STOP:
        prefetch_tile(tile)
else:
    # Other roles use the same tile ID, not independent atomic claims.
    wait(schedule_ready[slot], phase)
    tile = schedule[slot]
    if tile != STOP:
        process_tile(tile)
    release_schedule_slot_after_all_roles_read_it(slot)
```

Claiming from a warp that already runs ahead hides atomic latency under
prefetching. A bounded SMEM ring broadcasts assignments to the other roles.
Publish a stop assignment too, so no role waits forever after the pool drains.

Initialize the work counter before each invocation, ordered after the previous
invocation completes. Independent concurrent invocations need separate state.
This software queue is distinct from Blackwell Cluster Launch Control, which
redistributes work by canceling not-yet-started CTAs or clusters.

## 9. Keep Small Reductions Warp-Local

Assign one small selection problem to each warp rather than involving a whole
CTA. For BF16 top-K, an order-preserving value key and a local index can fit in
one integer, allowing a hardware integer max reduction with deterministic ties.

```python
# One candidate per lane, finite BF16 values; IDs fit in INDEX_BITS.
# Keep the complete key below bit 31 when using signed Int32 MAX.
bits = bf16_bits_as_u16(value)
ordered = (bits ^ 0xFFFF) if (bits & 0x8000) else (bits | 0x8000)
key = (ordered << INDEX_BITS) | (TILE_SIZE - 1 - local_id)

for k in range(K):
    winner = prims.redux_sync(key, prims.ReductionKind.MAX, 0xFFFFFFFF)
    if lane == 0:
        store_winner(k, winner)
    if key == winner:
        key = -1  # Remove this candidate; real keys are nonnegative.
```

For multiple candidates per lane, first take a register-local max. Every lane
in the mask must participate. Define NaN and signed-zero ordering explicitly
when the application requires it.

Fuse other reductions over the same values when useful. For example, compute
maximum and exponential-sum partials while selecting candidates, rather than
reading the entire input again for normalization.

## 10. Remove Boundary Copies and Bookkeeping Launches

Arrange buffers so the producer writes exactly where the next operation reads.
This can remove a copy without fusing the two compute kernels.

```python
# Host-side views: same storage, compatible layout and lifetime.
producer_output = workspace[:ELEMENTS]
consumer_input = producer_output
producer(producer_output, stream=stream)
consumer(consumer_input, stream=stream)
```

Likewise, reset scratch state inside a later kernel once all previous readers
are finished, instead of launching a tiny reset kernel. Keep the reset out of
the later kernel's critical path when possible. If reset becomes part of an
ongoing producer/consumer protocol, it needs explicit ordering just like data.

Buffer aliasing must account for strides, padding, alignment, and concurrent
invocations. CUDA graphs complement this technique by reducing repeated host
launch overhead; capture stable buffers and keep changing state in device
memory rather than accidentally baking a changing value into the graph.

## 11. Pack Streamed Data and Manage Cache Residency

Pack static weights once into the tile order used by the kernel. This makes
each transfer more contiguous without changing the matrix operation or TMA's
destination layout.

```python
# Before repeated execution: tile packing, not per-launch conversion.
packed_weights = pack_tiles(weights, tile_shape, traversal_order)

# TMA producer: streaming weights have little reuse after this tile.
prims.cp_async_bulk_tensor_shared_cta_global(
    smem_tile,
    packed_descriptor,
    tile_coordinates,
    full_bar,
    l2_cache_hint=evict_first_policy,
)
```

An evict-first hint can reduce interference with reusable activations or
metadata; it does not bypass L2 or guarantee eviction. Do not apply it to data
that neighboring CTAs will reuse. Packing may cost extra storage, and its
one-time cost only pays off across enough executions.

## What to Measure

- End-to-end latency, not just the fastest isolated kernel.
- Startup delay and the last-finishing CTA, not only steady-state throughput.
- Actual overlap from device timestamps or traces; fewer barriers alone is not proof.
- Registers, spills, SMEM/TMEM use, and residency after adding buffering or fusion.
- Repeated execution with changing inputs, including scratch-state reuse.

Use the same work and numerical contract for both variants, warm up compilation,
and alternate measurement order to reduce clock and temperature bias.
