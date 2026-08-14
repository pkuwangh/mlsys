#!/bin/bash

# get current directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${ROOT_DIR}/scripts/common.sh" || return 1

# alias
alias lt='ls -lhrt'
alias tileh='tmux select-layout even-vertical'
alias tilev='tmux select-layout even-horizontal'
alias tile4='tmux select-layout tiled'

if [ -z "${MAMBA_ROOT_PREFIX}" ]; then
    infoMsg "No micromamba setup on this machine; setup locally"

    # install micromamba
    my_arch=$(uname -m)
    if [[ "$my_arch" == "x86_64" ]]; then
        MAMBA_ARCH="linux-64"
    elif [[ "$my_arch" == "aarch64" ]]; then
        MAMBA_ARCH="linux-aarch64"
    else
        warnMsg "Unsupported architecture: $my_arch. Please install micromamba manually."
        return 1
    fi

    # check if micromamba is already installed locally
    export MAMBA_ROOT_PREFIX="${ROOT_DIR}/micromamba"
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"
    mkdir -p "${MAMBA_ROOT_PREFIX}" || return 1
    echo "${MAMBA_EXE}"
    if [ ! -f "${MAMBA_EXE}" ]; then
        debugMsg "Downloading micromamba to ${MAMBA_EXE} ..."
        curl -Ls "https://micro.mamba.pm/api/micromamba/${MAMBA_ARCH}/latest" | tar -xvj -C "${MAMBA_ROOT_PREFIX}" bin/micromamba || return 1
    else
        debugMsg "micromamba is already installed at ${MAMBA_EXE}"
    fi
    # set up micromamba environment for this shell
    eval "$("${MAMBA_EXE}" shell hook -s posix)" || return 1
    infoMsg "Micromamba is set up."
fi

# let micromamba ignore ~/.local/
export PYTHONNOUSERSITE=1

splitLine
# create default virtual env
MY_VENV="mlsys-base"

if micromamba env list | grep -q "${MY_VENV}"; then
    debugMsg "Virtual env ${MY_VENV} already exists."
else
    infoMsg "Creating default virtual env ${MY_VENV} ..."
    micromamba create -n "${MY_VENV}" -c conda-forge python=3.12 pip=25.0 -y || return 1
    micromamba install -n "${MY_VENV}" -c conda-forge git-lfs -y || return 1
fi

splitLine
CURR_VENV=$(getVenv)

if [ "${CURR_VENV}" == "${MY_VENV}" ]; then
    debugMsg "Already in virtual environment: ${CURR_VENV}"
else
    infoMsg "Activating default virtual env (${MY_VENV})..."
    micromamba activate "${MY_VENV}" || return 1
fi

splitLine
debugMsg "Listing all virtual envs ..."
micromamba env list
