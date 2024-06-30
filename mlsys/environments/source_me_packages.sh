#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURR_DIR}/scripts/common.sh"

# cuda
echo "-------- cuda --------"
if [ -f "${CURR_DIR}/env_cuda_home.sh" ]; then
    source "${CURR_DIR}/env_cuda_home.sh"
else
    debugMsg "env_cuda_home.sh not found, using system default"
    if [ -z "${CUDA_HOME}" ]; then
        debugMsg "CUDA_HOME not set, trying typical location"
        CUDA_SYSTEM_DEFAULT="/usr/local/cuda"
        if [ -d "${CUDA_SYSTEM_DEFAULT}" ]; then
            export CUDA_HOME="${CUDA_SYSTEM_DEFAULT}"
        fi
    fi
fi
if [ -z "${CUDA_HOME}" ]; then
    warnMsg "Could not find CUDA_HOME; CUDA may not work properly"
else
    infoMsg "CUDA_HOME=${CUDA_HOME}"
    debugMsg "Set same for CUDA_PATH, CUDA_ROOT, CUDA_TOOLKIT_ROOT_DIR"
    export CUDA_PATH="${CUDA_HOME}"
    export CUDA_ROOT="${CUDA_HOME}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
fi

# nccl
echo "-------- nccl --------"
if [ -z "${NCCL_ROOT}" ]; then
    debugMsg "NCCL_ROOT not set, trying locally built NCCL"
    NCCL_LOCAL_DEFAULT="${CURR_DIR}/packages/nccl/build"
    if [ -d "${NCCL_LOCAL_DEFAULT}" ]; then
        export NCCL_ROOT="${NCCL_LOCAL_DEFAULT}"
        infoMsg "NCCL_ROOT=${NCCL_ROOT}"
        ls -l "${NCCL_ROOT}/lib/libnccl.so.2"
        export CPATH="${NCCL_ROOT}/include:${CPATH}"
        export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}"
    else
        NCCL_SYSTEM_DEFAULT="/usr/lib/$(uname -m)-linux-gnu"
        debugMsg "Locally built NCCL not found; trying system default at ${NCCL_SYSTEM_DEFAULT}"
        if [ -f "${NCCL_SYSTEM_DEFAULT}/libnccl.so.2" ]; then
            infoMsg "Found NCCL at ${NCCL_SYSTEM_DEFAULT}"
        else
            warnMsg "Could not find NCCL library"
        fi
        if [ -f "/usr/include/nccl.h" ]; then
            infoMsg "Found NCCL header at /usr/include/nccl.h"
        else
            warnMsg "Could not find NCCL header"
        fi
    fi
else
    debugMsg "NCCL_ROOT=${NCCL_ROOT}"
fi

# system paths
echo "-------- system paths --------"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
infoMsg "nvcc=$(which nvcc)"
