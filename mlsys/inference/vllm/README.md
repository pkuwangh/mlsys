# vLLM

## Install vLLM

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# full build
cd vllm
uv pip install -r requirements/build.txt
# build from source in editable mode
uv pip install --no-build-isolation -e .
# to make vscode/pylance happy and be able to find reference in source code
uv pip install --no-build-isolation -e . --config-settings editable_mode=compat

# additional deps
pip install -r requirements.txt
```

## Quick start

```bash
cd quickstart

# profile batch inference
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ./v01-offline-batch-inference.py
```

## Benchmarks

### Offline throughput benchmark

```bash
# LLM
vllm bench throughput \
  --model=models/NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name=sonnet \
  --dataset-path=vllm/benchmarks/sonnet.txt \
  --num-prompts=100
```

### Archived

```bash
export HF_TOKEN=<hugginface token>
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# throughput benchmark
python3 benchmark_throughput.py \
    --output-json results//throughput_llama8B_tp1.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --load-format dummy \
    --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200 \
    --backend vllm

# latency benchmark
python3 benchmark_latency.py \
    --output-json results//latency_llama8B_tp1.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --load-format dummy \
    --num-iters-warmup 5 \
    --num-iters 15

# serving benchmark
# server
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --swap-space 16 \
    --disable-log-stats \
    --disable-log-requests \
    --load-format dummy
# client
python3 benchmark_serving.py \
    --save-result \
    --result-dir results/ \
    --result-filename serving_llama8B_tp1_sharegpt_qps_16.json \
    --request-rate 16 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200
```
