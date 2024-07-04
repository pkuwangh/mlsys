#!/bin/bash

# alias
alias lt='ls -lhrt'

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/mlsys/environments/scripts/common.sh"

# check if virtualenv is installed
debugMsg "Checking if micromamba is installed ..."
micromamba env list > /dev/null
if [ $? -ne 0 ]; then
    warnMsg "Error: micromamba is not installed."
    debugMsg "To install, follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos"
    return
fi

# let micromamba env ignore home local
export PYTHONNOUSERSITE=1

splitLine
# create virtual envs
ALL_VENVS=("mlsys_general" "mlsys_tensorrt")

for venv in "${ALL_VENVS[@]}"; do
    if micromamba env list | grep -q "${venv}"; then
        debugMsg "Virtual env ${venv} already exists."
    else
        infoMsg "Creating default virtual env ${venv} ..."
        micromamba create -n "${venv}" -c conda-forge python=3.10 pip=23.2 -y
    fi
done

splitLine
debugMsg "Listing all virtual envs ..."
micromamba env list
