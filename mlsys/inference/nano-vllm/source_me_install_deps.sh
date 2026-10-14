#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../../scripts/common.sh"

MY_VENV="mlsys-nano"

micromamba deactivate
if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    micromamba create -n "${MY_VENV}" -c conda-forge python=3.12 pip=25.0 "setuptools<80.0.0" -y
    # conda-forge only
    micromamba install -n "${MY_VENV}" -y \
        -c conda-forge \
        "cuda-toolkit=13.2.2" \
        "gcc=15.3.0" \
        "gxx=15.3.0" \
        "ninja=1.13.2" \
        || return 1
fi
micromamba activate "${MY_VENV}"

splitLine

# system deps
cleanupCondaBackEnvs

# python deps
uv pip install black loguru ruff huggingface_hub "cuda-python==13.2.0" || return 1

# cuda env
source "${CURR_DIR}/../../../scripts/source_cuda_env.sh" || return 1

splitLine
infoMsg "Checking nvcc"
which nvcc || return 1
nvcc --version || return 1

splitLine

# torch
uv pip install "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu132 || return 1

# flash_attn
export FLASH_ATTN_CUDA_ARCHS="100;103"
# Build the local submodule against installed torch
# FLASH_ATTN_DIR="$(realpath "${CURR_DIR}/../../gpu/backends/flash-attention")"
# FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=16 uv pip install --no-cache --no-build-isolation --no-deps $FLASH_ATTN_DIR || return 1

python "${CURR_DIR}/check_cuda.py"
