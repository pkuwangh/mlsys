#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/../../../scripts/common.sh"

splitLine
checkVenv
if [ $? -ne 0 ]; then
    return 1
fi

micromamba install -y \
    -c conda-forge \
    pytorch-gpu=2.7.1

