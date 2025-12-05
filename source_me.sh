#!/bin/bash

# get current directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${ROOT_DIR}/scripts/common.sh"

# alias
alias lt='ls -lhrt'
alias tile_h='tmux select-layout even-vertical'
alias tile_v='tmux select-layout even-horizontal'
alias tile_4='tmux select-layout tiled'

# install micromamba
my_arch=$(uname -m)
if [[ "$my_arch" == "x86_64" ]]; then
    MAMBA_ARCH="linux-64"
elif [[ "$my_arch" == "aarch64" ]]; then
    MAMBA_ARCH="linux-aarch64"
else
    warnMsg "Unsupported architecture: $my_arch. Please install micromamba manually."
    return
fi
# check if micromamba is already installed
export MAMBA_ROOT_PREFIX="${ROOT_DIR}/micromamba"
export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"
mkdir -p "${MAMBA_ROOT_PREFIX}"
pushd "${MAMBA_ROOT_PREFIX}" > /dev/null
echo "${MAMBA_EXE}"
if [ ! -f "${MAMBA_EXE}" ]; then
    debugMsg "Downloading micromamba to ${MAMBA_EXE} ..."
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MAMBA_ARCH}/latest" | tar -xvj bin/micromamba
else
    debugMsg "micromamba is already installed at ${MAMBA_EXE}"
fi
# set up micromamba environment for this shell
eval "$(./bin/micromamba shell hook -s posix)"
infoMsg "Micromamba is set up."
popd > /dev/null

# let micromamba ignore ~/.local/
export PYTHONNOUSERSITE=1

splitLine
# create default virtual env
MY_VENV="mlsys-base"

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

micromamba install -y git-lfs

splitLine
debugMsg "Listing all virtual envs ..."
micromamba env list
