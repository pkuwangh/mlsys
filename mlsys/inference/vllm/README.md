# vLLM

## Install vLLM

```bash
# install packages in a virtualenv
micromamba create -n mlsys_vllm -c conda-forge python=3.10 pip=23.2 -y
micromamba activate mlsys_vllm

# python-only build
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .

# full build
cd vllm
pip install -e .

# additional deps
pip3 install -r requirements.txt
```

## Benchmarks

```bash
cd vllm/benchmarks
mkdir -p results

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
