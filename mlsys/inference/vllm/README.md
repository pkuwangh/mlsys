# vLLM

- [vLLM](#vllm)
  - [Install vLLM and vLLM-Omni](#install-vllm-and-vllm-omni)
  - [Download Model](#download-model)
  - [vLLM Instructions](#vllm-instructions)
    - [LLM Quick Start](#llm-quick-start)
    - [vLLM Benchmarks](#vllm-benchmarks)
  - [vLLM-Notes](#vllm-notes)
    - [Request lifecycle](#request-lifecycle)
      - [Organization \& Concept](#organization--concept)
      - [Generate call trace.](#generate-call-trace)
      - [execute\_model](#execute_model)
    - [Continuous batching \& paged attention](#continuous-batching--paged-attention)
    - [Various Parallelism](#various-parallelism)
      - [Tensor Parallel](#tensor-parallel)
      - [Pipeline Parallel](#pipeline-parallel)

## Install vLLM and vLLM-Omni

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# install vllm
# python install
VLLM_USE_PRECOMPILED=1 uv pip install -e '.[bench]' --torch-backend=auto

# vllm full build
uv pip install -r requirements/build.txt
# - build from source in editable mode
uv pip install --no-build-isolation -e .[bench]
# - to make vscode/pylance happy and be able to find reference in source code
uv pip install --no-build-isolation -e .[bench] --config-settings editable_mode=compat


# install vllm-omni
micromamba install -y -c conda-forge "libgl"
uv pip install "setuptools-scm==10.2.1"
VLLM_OMNI_TARGET_DEVICE=cuda uv pip install -e '.[demo]' --no-build-isolation
```

## Download Model

```bash
hf download --local-dir ./models/Qwen/Qwen3.8-27B Qwen/Qwen3.8-27B
hf download --local-dir ./models/z-lab/Qwen3.8-27B-DFlash2 z-lab/Qwen3.8-27B-DFlash2

hf download --local-dir ./models/nvidia/Cosmos3-Edge nvidia/Cosmos3-Edge
hf download --local-dir ./models/nvidia/Cosmos3-Super nvidia/Cosmos3-Super
```

## vLLM Instructions

### vLLM Quick Start

```bash
cd quickstart/

# profile batch inference
nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop ./v01-offline-batch-inference.py
nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true ./v02-tensor-parallel.py
```

### vLLM Benchmarks

```bash
cd vllm/

# Offline throughput benchmark
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --delay=50 --duration=30 \
vllm bench throughput \
  --model ../models/Qwen/Qwen3.8-27B \
  --dataset-name sonnet \
  --dataset-path benchmarks/sonnet.txt \
  --num-prompts 1000 \
  --max-num-seqs 64

# Offline latency benchmark
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
vllm bench latency \
  --model openai/gpt-oss-120b \
  --input-len=1 \
  --output-len=128 \
  --batch-size 1 \
  --num-iters-warmup=2 \
  --num-iters 1 \
  --profile \
  --profiler-config.profiler cuda
```

## vLLM Notes

### Request lifecycle

#### Organization & Concept

```python
engine_core: EngineCoreClient   # separate process from the client; own scheduler
    model_executor: UniProcExecutor     # backend abstraction
        driver_worker: WorkerWrapperBase
            worker: gpu_worker.Worker   # separate process that owns CUDA context
                model_runner: GPUModelRunner
                    model: Qwen3_5ForCausalLM
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


## vLLM-Omni Instructions

```bash
# quickstart
python examples/offline_inference/text_to_image/text_to_image.py \
    --model ../models/nvidia/Cosmos3-Edge \
    --prompt "A photorealistic red sports car at golden hour, cinematic lighting." \
    --extra-body '{"guardrails": false}' \
    --output cosmos3_edge_t2i.png
```
