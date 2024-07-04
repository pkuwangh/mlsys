# TensorRT-LLM

## Installation

```bash
# use virtual env
micromamba activate mlsys_tensorrt

# install torch
mkdir -p wheels-torch
pushd wheels-torch/
pip download --index-url=https://download.pytorch.org/whl/nightly/cu124 torch==2.4.0.dev20240612
pip install torch-2.4.0*
popd

# install tensorrt
# download tensorrt tar under Linux-aarch64-manylinux_2_31
tar xvf TensorRT-10.1.0.27.Ubuntu-20.04.aarch64-gnu.cuda-12.4.tar.gz
pip3 install TensorRT-10.1.0.27/python/tensorrt-10.1.0-cp310-none-linux_aarch64.whl
cp -r TensorRT-10.1.0.27 tensorrt
rm -rf tensorrt/python/

# pre-install some packages before tensorrt-llm build
pip3 install -r requirements.txt

# build tensorrt-llm (github)
pushd TensorRT-LLM/
git submodule update --init
git lfs pull
python3 ./scripts/build_wheel.py --clean --cuda_architectures "90-real" --trt_root $TRT_ROOT --nccl_root $NCCL_ROOT
pip3 uninstall tensorrt_llm && pip3 install build/*whl
popd
```

## Running GPT2

Download gpt2 model from HF

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm -f pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
```

TRT operations

```bash
# convert checkpoint
python3 TensorRT-LLM/examples/gpt/convert_checkpoint.py --model_dir gpt2 \
    --dtype float16 \
    --output_dir trt_artifacts/ckpts/gpt2-fp16-gpu1

# build trt engine
trtllm-build --checkpoint_dir trt_artifacts/ckpts/gpt2-fp16-gpu1 \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --max_seq_len 1024 \
    --output_dir trt_artifacts/engines/gpt2-fp16-gpu1
```

Run inference

```bash
python3 TensorRT-LLM/examples/run.py --engine_dir trt_artifacts/engines/gpt2-fp16-gpu1 \
    --tokenizer_dir gpt2 \
    --max_output_len 768 \
    --top_k 50
# profiling
python3 TensorRT-LLM/examples/run.py --engine_dir trt_artifacts/engines/gpt2-fp16-gpu1 \
    --tokenizer_dir gpt2 \
    --max_output_len 768 \
    --top_k 50 \
    --run_profiling
```
