# nano-vllm

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

