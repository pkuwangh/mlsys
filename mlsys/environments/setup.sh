#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURR_DIR}/scripts/common.sh"

# double check python virtual env
checkVenv

# install system deps
infoMsg "Installing system dependencies..."
sudo apt-get install libcudnn9-cuda-12 libopenblas-dev

# install python modules
infoMsg "Installing python modules..."
pip3 install -r "${CURR_DIR}/requirements/requirements-deps.txt"

# nccl!
infoMsg "To install NCCL:"
debugMsg " - To build on your own, follow packages/README.md"
debugMsg " - To install directly, search with ./scripts/find_nccl.sh"

# pytorch!
infoMsg "To install PyTorch:"
debugMsg " - To build on your own, follow packages/README.md"
debugMsg " - To use pre-built, run"
debugMsg "   - pip3 install -r requirements/requirements-pytorch.txt for x86_64"
debugMsg "   - ./scripts/install_pytorch_grace_nccl.sh with NCCL/UCC support for aarch64"
debugMsg "   - ./scripts/install_pytorch_grace_plain.sh without NCCL support for aarch64"
