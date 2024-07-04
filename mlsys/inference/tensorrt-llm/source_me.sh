#!/bin/bash

export _GLIBCXX_USE_CXX11_ABI=1
export USE_CUDNN=1
export USE_NCCL=1

# get the current dir
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRT_ROOT="${CURR_DIR}/tensorrt"
export TRT_LIB_DIR="${TRT_ROOT}/lib"
export TRT_INCLUDE_DIR="${TRT_ROOT}/include"

export LIBRARY_PATH="${TRT_ROOT}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${LD_LIBRARY_PATH}"
