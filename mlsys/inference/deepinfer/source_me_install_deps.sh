#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../../scripts/common.sh"

MY_VENV="mlsys-deepinfer"

micromamba deactivate
if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    micromamba create -n "${MY_VENV}" -c conda-forge "python=3.12" "pip=25.0" -y
    # conda-forge only
    micromamba install -n "${MY_VENV}" -y \
        -c conda-forge \
        "cuda-toolkit=13.2.2" \
        "ccache=4.13.6" \
        "cmake=4.4.2" \
        "elfutils==0.196" \
        "gcc=15.3.0" \
        "gxx=15.3.0" \
        "ninja=1.13.0" \
        || return 1
fi
micromamba activate "${MY_VENV}"

splitLine

# system deps
cleanupCondaBackEnvs

# python deps
uv pip install "ruff==0.15.0" "huggingface_hub==0.36.2" "cuda-python==13.2.0" || return 1

# cuda env
source "${CURR_DIR}/../../../scripts/source_cuda_env.sh" || return 1

splitLine
infoMsg "Checking nvcc"
which gcc || return 1
gcc --version || return 1
which nvcc || return 1
nvcc --version || return 1

splitLine

export CCACHE_NOHASHDIR="true"
export CCACHE_DIR="${CURR_DIR}/.ccache"

# torch
uv pip install "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu132 || return 1

# other deps
uv pip install "flash-attn-4==4.0.0b34"

python "${CURR_DIR}/check_cuda.py"
