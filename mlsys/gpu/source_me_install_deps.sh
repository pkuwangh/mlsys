#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../scripts/common.sh"

splitLine
checkVenv
if [ $? -ne 0 ]; then
    return 1
fi

cleanupCondaBackEnvs
micromamba install -y \
    -c nvidia/label/cuda-12.9.0 \
    cuda cuda-nvcc cuda-toolkit cuda-runtime \
    -c conda-forge \
    gcc=12.4 libboost-devel=1.88.0 openmpi-mpicxx=5.0.8 \
    --strict-channel-priority

pip install cmake==4.1.0 ninja==1.13.0

pip install black loguru ruff "huggingface_hub[cli]"

# for cmake find_package
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

# for conda-installed CUDA, real home is under /targets/$(uname -m)-linux
_SCATTERED_CUDA_HOME="${CUDA_HOME}/targets/$(uname -m)-linux"
if [ ! -d "${_SCATTERED_CUDA_HOME}" ]; then
    debugMsg "Cannot find scattered CUDA home: ${_SCATTERED_CUDA_HOME}"
    _OLD_SCATTERED_CUDA_HOME=$_SCATTERED_CUDA_HOME
    _SCATTERED_CUDA_HOME="${CUDA_HOME}/targets/sbsa-linux"
    if [ ! -d "${_SCATTERED_CUDA_HOME}" ]; then
        warnMsg "Cannot find scattered CUDA home: ${_OLD_SCATTERED_CUDA_HOME} and ${_SCATTERED_CUDA_HOME}"
        return 1
    fi
fi
infoMsg "Using scattered CUDA home: ${_SCATTERED_CUDA_HOME}"

# link critical header files
_HEADERS=("cuda.h" "cuda_runtime.h" "cuda_runtime_api.h" "device_functions.h")
for header in "${_HEADERS[@]}"; do
    if [ ! -f "${CUDA_HOME}/include/${header}" ]; then
        ln -s "${_SCATTERED_CUDA_HOME}/include/${header}" "${CUDA_HOME}/include/${header}"
    fi
done

# libcudart.so is under /targets/x86_64-linux/lib, but is linked to /lib
# libcuda.so is under /targets/x86_64-linux/lib/stub
export LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LD_LIBRARY_PATH}"

pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

pip install pycuda

splitLine
infoMsg "Checking gcc, nvcc"
which gcc
gcc --version
which nvcc
nvcc --version
