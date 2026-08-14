#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/common.sh" || return 1

if [ -z "${CONDA_PREFIX:-}" ]; then
    warnMsg "CONDA_PREFIX is not set. Please activate your conda/micromamba environment first."
    return 1
fi

# for cmake find_package
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

# for conda-installed CUDA, real home is under /targets/$(uname -m)-linux
_SCATTERED_CUDA_HOME="${CUDA_HOME}/targets/$(uname -m)-linux"
if [ ! -d "${_SCATTERED_CUDA_HOME}" ]; then
    debugMsg "Cannot find scattered CUDA home: ${_SCATTERED_CUDA_HOME}"
    _OLD_SCATTERED_CUDA_HOME="${_SCATTERED_CUDA_HOME}"
    _SCATTERED_CUDA_HOME="${CUDA_HOME}/targets/sbsa-linux"
    if [ ! -d "${_SCATTERED_CUDA_HOME}" ]; then
        warnMsg "Cannot find scattered CUDA home: ${_OLD_SCATTERED_CUDA_HOME} and ${_SCATTERED_CUDA_HOME}"
        return 1
    fi
fi
infoMsg "Using scattered CUDA home: ${_SCATTERED_CUDA_HOME}"

# link critical header files
_HEADERS=("cuda.h" "cuda_runtime.h" "cuda_runtime_api.h" "device_functions.h")
#for header in "${_HEADERS[@]}"; do
#    if [ ! -f "${CUDA_HOME}/include/${header}" ]; then
#        ln -s "${_SCATTERED_CUDA_HOME}/include/${header}" "${CUDA_HOME}/include/${header}"
#    fi
#done

# cuda.h etc. are under /targets/x86_64-linux/include
export CPATH="${_SCATTERED_CUDA_HOME}/include:${CPATH}"
# cccl, cuda core compute libraries
export CPATH="${_SCATTERED_CUDA_HOME}/include/cccl:${CPATH}"

# libcudart.so is under /targets/x86_64-linux/lib, but is linked to /lib
# libcuda.so is under /targets/x86_64-linux/lib/stub
export LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LD_LIBRARY_PATH}"
