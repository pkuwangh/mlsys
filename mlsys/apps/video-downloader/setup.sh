#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/../../../scripts/common.sh"

splitLine
MY_VENV="yt-dlp"
CURR_VENV=$(getVenv)
if [ "${CURR_VENV}" != "${MY_VENV}" ]; then
    warnMsg "Please activate the ${MY_VENV} virtual environment first."
    exit 1
fi

splitLine
pip3 install -r "${CURR_DIR}/requirements.txt"

splitLine
