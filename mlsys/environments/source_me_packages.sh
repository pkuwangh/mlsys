#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURR_DIR}/scripts/common.sh"

# cuda
if [ -f "${CURR_DIR}/env_cuda_home.sh" ]; then
    source "${CURR_DIR}/env_cuda_home.sh"
else
    checkMsg "env_cuda_home.sh not found, using system default"
    if [ -z "${CUDA_HOME}" ]; then
        checkMsg "CUDA_HOME not set, using default"
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
infoMsg "CUDA_HOME=${CUDA_HOME}"

# nccl
NCCL_PATH="${CURR_DIR}/packages/nccl/build"
if [ -d "${NCCL_PATH}" ]; then
    export NCCL_ROOT="${NCCL_PATH}"
    export NCCL_LIB_DIR="${NCCL_ROOT}/lib"
    export NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"
else
    checkMsg "Locally built NCCL not found, using default NCCL_PATH"
fi
infoMsg "NCCL_ROOT=${NCCL_ROOT}"

# system paths
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${NCCL_PATH}/include:${CPATH}"
export LIBRARY_PATH="${NCCL_PATH}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
infoMsg "nvcc=$(which nvcc)"
