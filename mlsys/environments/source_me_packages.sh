#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cuda
source "${CURR_DIR}/env_cuda_home.sh"
echo "CUDA_HOME=${CUDA_HOME}"

# nccl
NCCL_PATH="${CURR_DIR}/packages/nccl/build"
export NCCL_ROOT="${NCCL_PATH}"
export NCCL_LIB_DIR="${NCCL_ROOT}/lib"
export NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"

# system paths
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${NCCL_PATH}/include:${CPATH}"
export LIBRARY_PATH="${NCCL_PATH}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "nvcc=$(which nvcc)"

