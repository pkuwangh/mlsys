#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# nccl
NCCL_PATH="${CURR_DIR}/packages/nccl/build"
export NCCL_ROOT="$(pwd)/nccl/build"
export NCCL_LIB_DIR="${NCCL_ROOT}/lib"
export NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"

# system paths
export CPATH="${NCCL_PATH}/include:${CPATH}"
export LIBRARY_PATH="${NCCL_PATH}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_PATH}/lib:${LD_LIBRARY_PATH}"
