#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../scripts/common.sh" || return 1

MY_VENV="mlsys-gpu"

micromamba deactivate
if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    micromamba create -n "${MY_VENV}" -c conda-forge python=3.12 pip=25.0 "setuptools<80.0.0" -y
fi
micromamba activate "${MY_VENV}"

splitLine

# system deps
cleanupCondaBackEnvs
# Keep CUDA and build deps on conda-forge so the solver uses one consistent stack.
micromamba install -n "${MY_VENV}" -y \
    -c conda-forge \
    "cuda-toolkit=13.2.2" \
    "cmake=4.4.2" \
    "gcc=14.3.0" \
    "gxx=14.3.0" \
    "ninja=1.13.1" \
    "libboost-devel" \
    "openmpi-mpicxx" \
    || return 1

# python deps
uv pip install black loguru ruff "huggingface_hub[cli]" "cuda-python==13.2.0" || return 1

# cuda env
source "${CURR_DIR}/../../scripts/source_cuda_env.sh" || return 1

splitLine
infoMsg "Checking nvcc"
which gcc || return 1
gcc --version || return 1
which nvcc || return 1
nvcc --version || return 1

splitLine

# torch
uv pip install "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu132 || return 1

# diffusers
uv pip install "diffusers==0.40.0"

# other deps
uv pip install "accelerate==1.14.0"

# cutlass-dsl
uv pip install "nvidia-cutlass-dsl==4.7.1" "apache-tvm-ffi==0.1.12"

python "${CURR_DIR}/check_cuda.py"

