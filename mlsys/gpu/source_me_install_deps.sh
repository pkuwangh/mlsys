#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/../../scripts/common.sh"

splitLine
checkVenv
if [ $? -ne 0 ]; then
    return 1
fi

micromamba install -y \
    -c nvidia/label/cuda-12.9.0 \
    cuda cuda-nvcc cuda-toolkit cuda-runtime \
    -c conda-forge \
    cmake=4.1.0 gcc=12.4 libboost-devel=1.88.0 openmpi-mpicxx=5.0.8 \
    pytorch-gpu=2.7.1 pycuda

splitLine
infoMsg "Checking gcc, nvcc"
which gcc
gcc --version
which nvcc
nvcc --version

