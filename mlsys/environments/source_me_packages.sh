#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# nccl
NCCL_PATH="${CURR_DIR}/packages/nccl/build"

export CPATH="${NCCL_PATH}/include:${CPATH}"
export LIBRARY_PATH="${NCCL_PATH}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_PATH}/lib:${LD_LIBRARY_PATH}"
