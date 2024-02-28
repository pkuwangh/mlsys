#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURR_DIR}/scripts/common.sh"

# cuda
if [ -f "${CURR_DIR}/env_cuda_home.sh" ]; then
    source "${CURR_DIR}/env_cuda_home.sh"
else
    debugMsg "env_cuda_home.sh not found, using system default"
    if [ -z "${CUDA_HOME}" ]; then
        debugMsg "CUDA_HOME not set, using default"
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
infoMsg "CUDA_HOME=${CUDA_HOME}"
debugMsg "Set same for CUDA_PATH, CUDA_ROOT, CUDA_TOOLKIT_ROOT_DIR"
export CUDA_PATH="${CUDA_HOME}"
export CUDA_ROOT="${CUDA_HOME}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"

# nccl
export NCCL_ROOT="${CURR_DIR}/packages/nccl/build"
infoMsg "NCCL_ROOT=${NCCL_ROOT}"
if [ ! -d "${NCCL_ROOT}" ]; then
    debugMsg "Locally built NCCL not found (yet)"
    debugMsg "Consider build NCCL or set env NCCL_ROOT properly"
fi

# system paths
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${NCCL_ROOT}/include:${CPATH}"
export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
infoMsg "nvcc=$(which nvcc)"
