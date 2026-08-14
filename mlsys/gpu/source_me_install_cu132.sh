#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../scripts/common.sh" || return 1

splitLine
checkVenv || return 1

# system deps
cleanupCondaBackEnvs
# Keep CUDA and build deps on conda-forge so the solver uses one consistent stack.
micromamba install -y \
    -c conda-forge \
    "cuda-toolkit=13.2.2" \
    "cmake=4.2" \
    "gcc=14.3" \
    "gxx=14.3" \
    "ninja=1.13.0" \
    "libboost-devel" \
    "openmpi-mpicxx" \
    || return 1

# python deps
uv pip install black loguru ruff "huggingface_hub[cli]" "cuda-python==13.2.0" || return 1

# cuda env
source "${CURR_DIR}/../../scripts/source_cuda_env.sh" || return 1

# torch
uv pip install "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu132 || return 1

# cutlass-dsl
uv pip install "nvidia-cutlass-dsl==4.7.0" "apache-tvm-ffi==0.1.12"

splitLine
infoMsg "Checking gcc, nvcc"
which gcc || return 1
gcc --version || return 1
which nvcc || return 1
nvcc --version || return 1

