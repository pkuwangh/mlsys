#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURR_DIR}/scripts/common.sh"

# system dependecies
infoMsg "Installing system dependencies"
sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt-get install libopenblas-dev mpich

# python requirements
infoMsg "Installing python modules"
pip3 install numpy packaging

# nccl!
infoMsg "To install NCCL:"
debugMsg " - To build on your own, follow packages/README.md"
debugMsg " - To install directly, search with ./scripts/find_nccl.sh"

# pytorch!
infoMsg "To install PyTorch:"
debugMsg " - To build on your own, follow packages/README.md"
debugMsg " - To use pre-built, run"
debugMsg "   - ./scripts/install_pytorch_grace_nccl.sh with NCCL/UCC support"
debugMsg "   - ./scripts/install_pytorch_grace_plain.sh without NCCL support"
