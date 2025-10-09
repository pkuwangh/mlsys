# vLLM

## Install vLLM

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# full build
cd vllm
uv pip install -r requirements/build.txt
# build from source in editable mode
uv pip install --no-build-isolation -e .[bench]
# to make vscode/pylance happy and be able to find reference in source code
uv pip install --no-build-isolation -e .[bench] --config-settings editable_mode=compat
```

## Quick start

```bash
cd quickstart

# profile batch inference
nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop ./v01-offline-batch-inference.py
nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true ./v02-tensor-parallel.py

# benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
nsys profile --delay 45 \
    vllm bench throughput \
    --model models/meta-llama/Llama-3.1-8B-Instruct/ \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200
```

## Benchmarks

### Offline throughput benchmark

```bash
# LLM
# nsys profile --delay 50 --duration 30 \
vllm bench throughput \
  --model=models/NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name=sonnet \
  --dataset-path=vllm/benchmarks/sonnet.txt \
  --num-prompts=1000

# Speculative decoding
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_V1=1 \
vllm bench throughput \
    --dataset-name=hf \
    --dataset-path=likaixin/InstructCoder \
    --model=models/meta-llama/Meta-Llama-3-8B-Instruct \
    --input-len=1000 \
    --output-len=100 \
    --num-prompts=2048 \
    --async-engine \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

## Notes

### Request lifecycle

#### Organization & Concept

```python
engine_core: EngineCoreClient   # separate process from the client; own scheduler
    model_executor: UniProcExecutor     # backend abstraction
        driver_worker: WorkerWrapperBase
            worker: gpu_worker.Worker   # separate process that owns CUDA context
                model_runner: GPUModelRunner
                    model: LlamaForCausalLM
                    kv_caches: list[Tensor]
        collective_rpc: to execute a RPC call on the worker
        constructor:
            init_worker
            init_device
            load_model
    _initialize_kv_caches
        model_executor.initialize_from_config -> worker.initialize_from_config
        model_executor.compile_on_warm_up_model -> worker.compile_on_warm_up_model
            model_runner.capture_model
```

#### Generate call trace.

```python
llm._validate_and_add_requests
    llm_engine.add_request  # one by one
        processor.process_input
        engine_core.add_request
            if sampling_params.n > 1: for parallel sampling: call add_request n times
llm._run_engine
    while llm_engine.has_unfinished_requests:
        step_outputs = llm_engine.step()  # one decoding iteration
            EngineCore.step
                scheduler.schedule
                module_executor.execute_model
                scheduler_context.append_output
                process module output
```

#### execute_model

`EngineCore` calls model_executor of `executor`, `worker`, down to `model_runner`.
Details in `execute_model`:

```bash
_prepare_inputs:
  - build PerLayerAttnMetadata
    - block_table: the block_ids for past tokens
    - positions: position for new tokens
    - slot_mapping: slot in KV$ to write KV for new tokens
    - cu_num_tokens: cumulative num of new tokens; since [B, T] dimensions will be flattned, need cumulative count
    - query_start_loc: flash_attn's cu_seqlens_q, shift-right cu_num_tokens
    - seq_lens: length of each sequence
_preprocess:
  - get final input_ids & positions
model.forward:
  - call into specific model implementation
    - Attention layer calls into attn_backend.get_impl_cls().forward()
post-processing:
  - model.compute_logits
  - sampler(logits)
```

### Continuous batching & paged attention

Scheduler

- scheduler does not care about prefill vs. decoding phase
  - each request tracks `num_computed_tokens` and `num_tokens_with_spec`
  - then just compute `num_new_tokens`
- first scan the `running` queue
  - allocate kv_cache with `kv_cache_manager.allocate_slots`
    - on failure, keep preempting low-priority requests
  - if `can_schedule`, add to `scheduled_...` and track `token_budget`
- then scan the `waiting` queue
  - chunked prefill happens w/ `num_new_tokens = min(num_tokens - num_computed_tokens, token_budget)`
  - allocate kv_cache with `allocate_slots` - break if failed to allocate
  - add to `running` queue and update `token_budget`

KVCacheManager

- calculate `num_blocks_to_allocate` based on computed tokens, new-computed tokens (hit prefix cache), and real new tokens.
- allocate a list of new blocks, which simply grab blocks from the `block_pool`
- commit/cache the new blocks, i.e. for blocks that are full, calculate the `block_bash` and track in a map.

### Various Parallelism

Comm groups are constructed during `worker.init_device`.

#### Tensor Parallel

- `ColumnParallelLinear`:
  - cut weight matrix in 2nd dimension, i.e. by columns, the result is a partial matrix; need `all-gather` to get full matrix.
  - note here the dimension refers to matrix `A`'s dimension in `Y = X * A + b` where `A` has shape `(in_features, out_features)`.
    - but to be clear, torch stores weight matrix as `(out_features, in_features)` and it does `Y = X * A.T + b` when calling `functional.linear(X, A, b)`.
- `RowParallelLinear`:
  - cut weight matrix in 1st dimension, i.e. by rows, the input need to be cut by columns; the result is a full matrix but need `all-reduce`.
  - full input -> column-parallel -> partial matrix -> row-parallel -> all-reduce
- `VocabParallelEmbedding`:
  - cut embedding table in vocabulary dimension, i.e. by rows; call `embedding` method from the sharded embedding, then `all-reduce`.
- `ParallelLMHead(VocabParallelEmbedding)`:
  - still cut embedding table in vocabulary dimension, but it is implicitly transposed in matmul, so it's really column-parallel and need a `all-gather`.

In practice,

- `QKV` projection: column-parallel, no gather, partial hidden states
- `O` projection after attention: corresponding partial weights to do row-parallel, then all-reduce
- `Up MLP`: column-parallel, no gather, partial hidden states
- `Down MLP`: row-parallel, then all-reduce

#### Pipeline Parallel

- first rank: `embed_tokens: VocabParallelEmbedding`
- middle ranks: a slice of all `DecoderLayer`s
- last rank: `norm: RMSNorm` and `lm_head`
