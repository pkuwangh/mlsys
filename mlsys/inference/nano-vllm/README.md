# nano-vllm

- [nano-vllm](#nano-vllm)
  - [Install nano-vllm](#install-nano-vllm)
  - [Quick start](#quick-start)
  - [Walk through](#walk-through)
    - [Engine workflow](#engine-workflow)
    - [Scheduling](#scheduling)
    - [Block Manager](#block-manager)
    - [Model Runner](#model-runner)
      - [Setup](#setup)
      - [Preparation before each iteration](#preparation-before-each-iteration)

## Install nano-vllm

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# install nano-vllm
cd nano-vllm
pip install -e .
```

## Quick start

```bash
python example.py
```

## Walk through

### Engine workflow

- make a `Sequence` for each request/prompt
- add each sequence to the `Scheduler`
- step
  - get a subset of sequences from `scheduler.schedule()`
  - call `model_runner` to run a forward pass
  - `scheduler.postprocess` to process output token

### Scheduling

- instantiate a `BlockManager` with `num_kvcache_blocks` and `kvcache_block_size` in # of tokens
- maintain a `waiting` queue and a `running` queue
  - add/preempt to the `waiting` queue
  - schedule to the `running` queue
- `schedule`
  - try schedule prefill requests; limited by `max_num_batched_tokens` and `block_manager.can_allocate`
    - `block_manager.allocate`
    - move from `waiting` queue to `running` queue
      - but only runs newly scheduled prefill sequences for simplicity
  - try schedule decode requests:
    - loop through `running` queue
    - popleft and check `block_manager.can_append`
      - keep preempting recently-enqueued requests from `running` queue
      - if still cannot fit, preempt this very request and done
    - `block_manager.may_append`
    - add `scheduled_seqs` to the head of `running` queue
- `postprocess` after model run
  - append output token to sequence
  - check if sequence is done, i.e. EOS or `max_tokens`
    - mark `FINISHED`, `block_manager.deallocate` and remove from `running` queue.

### Block Manager

```python
class Block:
    block_id: int           # 0 ~ num_blocks-1
    ref_count: int
    hash: int               # hash(token_ids)
    token_ids: list[int]

class BlockManager:
    blocks: list[Block]
    hash_to_block_id: dict[int, int]
    free_block_ids: deque[int]
    used_block_ids: set[int]
```

- `can_allocate(seq)` - for prefill requests
  - `len(free_block_ids) >= seq.num_blocks`
- `allocate(seq)`
  - for each block of tokens in the sequence, i.e. `seq.token_ids[idx*block_size: (i+1)*block_size]`
  - compute hash from this block of token_ids and previous blocks, for **prefix sharing**
    - i.e. this hash is a rolling hash that includes previous blocks
    - when `len(token_ids) < block_size`, there is no valid hash (`-1`) - will always miss
  - check `hash_to_block_id` for cache hit/miss
    - on a miss, get a `block_id` from `free_block_ids` and `_allocate_block(block_id)`
      - if hit but `token_ids` not matching, signals cache collision or stale entry - treat as miss
    - on a hit, check if `block_id` from the hashmap is in `used_block_ids`
      - yes for prefix sharing case, increment `ref_count`
      - no indicating the `token_ids` were known but paged out
        - re-allocate with the same `block_id`
  - for a full block (`h != -1` because `len(token_ids) == block_size`)
    - update hash & token_ids list
    - add to `hash_to_block_id`
  - add `block_id` to `seq.block_table`
- `deallocate(seq)`
  - traverse `seq.block_table`
  - for each block, decrement `ref_count` and `_deallocate_block` if necessary.
- `can_append(seq)` - for decode requests
  - need new block only when the last token we got steps into a new block
  - `len(free_block_ids) >= (seq.num_tokens % block_size == 1)`
- `may_append(seq)`
  - when `seq.num_tokens % block_size == 1`:
    - `last_block` must be a full block with `block.hash != 1`
    - get a `block_id` from `free_block_ids` and allocate a block
  - when `seq.num_tokens % block_size == 0`:
    - calculate the hash and update into the block
    - add to `hash_to_block_id`
  - else, must be a partial block with `block.hash == -1`

### Model Runner

#### Setup

- create an instance of the target model, which includes
  - embedding layer: `weight = nn.Parameter(vocab_size_per_partition, hidden_size)`
    - `y = F.embedding(x, weight)` - expects `weight` shaped `(num_embeddings, embedding_size)`
      - `x: (B, T)`
      - `y = weight[input]`: `(B, T, H)`: gather rows of `weight` corresponding to the indices in `x`.
  - list of decodere layers
    - RMSNorm: `weight = nn.Parameter(hidden_size)`
      - `x = x + residual`
      - `next_residual = x`
      - `var = x.pow(2).mean(dim=-1, keepdim=True)`: per-token mean square across H dimension.
      - `x = x * torch.rsqrt(var + eps)`: apply per-token root mean square with a small eps
      - `x = x * weight`: apply learnable scale vector `weight (H)`
    - Attention
      - Proj for QKV: column-parallel linear `(B, T, H) -> (B, T, num_heads * H + 2 * num_kv_heads * H)`
      - re-shape QKV into `(B * T, num_heads/num_kv_heads, head_dim)` because RMSNorm & rotary are per-token per-head.
      - RMSNorm for Q & K: apply per-head RMSNorm
      - Rotary Emb for Q & K: `positions: (B, T)`
      - Core Attention: `(B * T, num_heads/num_kv_heads, head_dim)`
        - since `B & T` is flattened, which is why `cu_seqlens_q` & `cu_seqlens_k` are needed.
      - Proj O:
        - flatten `(B * T, num_heads, head_dim) -> (B * T, H)`
        - row-parallel linear `(H, H)` and view-back to `(B, T, H)`
    - RMSNorm: post-attention layer norm, residual from input layer norm
    - MLP:
      - gate_up_proj: `u = x @ W_up; g = x @ W_gate`, so the Linear layer `output_sizes` takes `[4H] * 2`.
      - activation: `h = SiLU(g) * u`
      - down_proj: `y = h @ W_down`
  - RMSnorm layer in the end
  - to compute `logits`, apply `lm_head (vocab_size, H)`
    - `y = F.linear(x, weight)` - expects `weight` shaped `(output_features, input_features)`
- allocate KV cache (actual tensors!)
  - `kv_cache = torch.empty(2, n_hidden_layers, n_kvcache_blocks, block_size, n_kv_heads, head_dim)`
  - `module.k_cache = kv_cache[0, layer_id]`
  - `module.v_cache = kv_cache[1, layer_id]`

#### Preparation before each iteration

- construct `input_ids` & `positions` tensors
- calculate `cu_seqlens_q` & `cu_seqlen_k` as attention kernel works on flattened tensor `(B * T, A, d)`
- build `slot_mapping` to tell attention kernel where to write the kv cache for input tokens
- prepare `block_tables` which has the info on where to find the kv cache for previous tokens
