#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/mlsys/environments/scripts/common.sh"

# check if virtualenv is installed
if ! [ -x "$(command -v virtualenv)" ]; then
    warnMsg "Error: virtualenv is not installed."
    debugMsg "To install, run: sudo apt install python3-pip && pip3 install virtualenv"
    return
fi

splitLine
# create default virtualenv
MY_VENV="general"

if ! [ -f "${CURR_DIR}/workspace/${MY_VENV}/bin/activate" ]; then
    infoMsg "Creating virtual env ${MY_VENV}..."
    virtualenv "${CURR_DIR}/workspace/${MY_VENV}"
else
    debugMsg "Virtual env ${MY_VENV} already exists."
fi

# create a list of other virtualenvs
OTHER_VENVS=("yt-dlp" "super-resolution" "intern-video2")

for venv in "${OTHER_VENVS[@]}"; do
    if ! [ -f "${CURR_DIR}/workspace/${venv}/bin/activate" ]; then
        infoMsg "Creating virtual env ${venv}..."
        virtualenv "${CURR_DIR}/workspace/${venv}"
    else
        debugMsg "Virtual env ${venv} already exists."
    fi
done

splitLine
PLAIN_ENV=$(getPlainEnv)
CURR_VENV=$(getVenv)

if [ "${CURR_VENV}" != "${PLAIN_ENV}" ]; then
    debugMsg "Already in virtual environment: ${CURR_VENV}"
else
    infoMsg "Activating default virtual env (${MY_VENV})..."
    source "${CURR_DIR}/workspace/${MY_VENV}/bin/activate"
fi

splitLine
# alias
alias lt='ls -lhrt'
