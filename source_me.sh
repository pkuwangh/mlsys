#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/mlsys/environments/scripts/common.sh"

# alias
alias lt='ls -lhrt'
alias tile_h='tmux select-layout even-vertical'
alias tile_v='tmux select-layout even-horizontal'
alias tile_4='tmux select-layout tiled'

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_ROOT_PREFIX="${CURR_DIR}/micromamba"
if [ -z "${MAMBA_EXE}" ]; then
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"
fi
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE" # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

# let micromamba ignore ~/.local/
export PYTHONNOUSERSITE=1

mkdir -p "${CURR_DIR}/micromamba"

# check if micromamba is installed
debugMsg "Checking if micromamba is installed ..."
micromamba env list > /dev/null
if [ $? -ne 0 ]; then
    warnMsg "Error: micromamba is not installed."
    debugMsg "To download, follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos"
    return
fi

splitLine
# create default virtual env
MY_VENV="mlsys_base"

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
debugMsg "Listing all virtual envs ..."
micromamba env list
