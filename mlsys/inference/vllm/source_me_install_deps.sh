#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURR_DIR}/../../../scripts/common.sh"

MY_VENV="mlsys-vllm"

micromamba deactivate
if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    micromamba create -n "${MY_VENV}" -c conda-forge python=3.10 pip=25.0 "setuptools<80.0.0" -y
fi
micromamba activate "${MY_VENV}"

splitLine
# For whatever reason, installing CUDA adds e.g. $CONDA_BACKUP_CFLAGS back up $CFLAGS
cleanupCondaBackEnvs
micromamba install -n "${MY_VENV}" -y \
    -c nvidia/label/cuda-12.8.0 \
    cuda cuda-nvcc cuda-toolkit cuda-runtime cuda-nvtx-dev \
    -c conda-forge \
    cmake=4.1.0 gcc=12.4 ninja=1.13.1 ccache=4.11 \
    nvtx=0.2.13

pip install black loguru ruff "huggingface_hub[cli]"

splitLine
infoMsg "Checking nvcc"
which nvcc
nvcc --version

splitLine
infoMsg "Set various environment variables for vLLM build"

export CCACHE_NOHASHDIR="true"
export CCACHE_DIR="${CURR_DIR}/.ccache"

# for cmake find_package
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

# for conda-installed CUDA, real home is under /targets/$(uname -m)-linux
_SCATTERED_CUDA_HOME="${CUDA_HOME}/targets/$(uname -m)-linux"

# link critical header files
_HEADERS=("cuda.h" "cuda_runtime.h" "cuda_runtime_api.h" "device_functions.h")
for header in "${_HEADERS[@]}"; do
    if [ ! -f "${CUDA_HOME}/include/${header}" ]; then
        ln -s "${_SCATTERED_CUDA_HOME}/include/${header}" "${CUDA_HOME}/include/${header}"
    fi
done

# cuda.h etc. are under /targets/x86_64-linux/include
export CPATH="${_SCATTERED_CUDA_HOME}/include:${CPATH}"

# libcudart.so is under /targets/x86_64-linux/lib, but is linked to /lib
# libcuda.so is under /targets/x86_64-linux/lib/stub
export LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_SCATTERED_CUDA_HOME}/lib:${_SCATTERED_CUDA_HOME}/lib/stub:${LD_LIBRARY_PATH}"

uv pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

python "${CURR_DIR}/check_cuda.py"
