#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/mlsys/environments/scripts/common.sh"

# check if virtualenv is installed
debugMsg "Checking if micromamba is installed and list envs if yes ..."
micromamba env list
if [ $? -ne 0 ]; then
    warnMsg "Error: micromamba is not installed."
    debugMsg "To install, follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos"
    return
fi

splitLine
# create default virtualenv
MY_VENV="mlsys_general"

if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    infoMsg "Creating default virtual env ${MY_VENV} ..."
    micromamba create -n "${MY_VENV}" -c conda-forge python=3.10 pip=23.2 -y
fi

splitLine
CURR_VENV=$(getVenv)

if [ "${CURR_VENV}" == "${MY_VENV}" ]; then
    debugMsg "Already in virtual environment: ${CURR_VENV}"
else
    infoMsg "Activating default virtual env (${MY_VENV})..."
    micromamba activate "${MY_VENV}"
fi

splitLine
# alias
alias lt='ls -lhrt'
