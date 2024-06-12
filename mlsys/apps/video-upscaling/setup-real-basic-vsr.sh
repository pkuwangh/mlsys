#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/../../environments/scripts/common.sh"

splitLine
MY_VENV="real-basic-vsr"
CURR_VENV=$(getVenv)
if [ "${CURR_VENV}" != "${MY_VENV}" ]; then
    warnMsg "Please activate the ${MY_VENV} virtual environment first."
    debugMsg "source ${CURR_DIR}/../../../workspace/${MY_VENV}/bin/activate"
    exit 1
fi

splitLine
pip3 install -r "${CURR_DIR}/requirements/requirements-real-basic-vsr.txt"

splitLine
MAX_JOBS=16 mim install mmcv-full

splitLine
pip3 install mmedit
