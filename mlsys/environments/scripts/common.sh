#!/bin/bash

# take one argument as the message
infoMsg() {
    echo -e "\033[32;1m${1}\033[0m"
}

warnMsg() {
    echo -e "\033[31;1m${1}\033[0m"
}

debugMsg() {
    echo -e "\033[33;5m${1}\033[0m"
}

splitLine() {
    echo "----------------------------------------"
}

# take one argument as the message
continueOrExit() {
    read -p "Continue with ${1}? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# get current virtualenv
getVenv() {
    local MY_VENV=$(python3 -c 'import sys ; print( sys.prefix.split("/")[-1] )')
    echo $MY_VENV
}

# check if the virtual environment is activated
checkVenv() {
    infoMsg "Checking python virtual environment..."
    if [ -z $CONDA_DEFAULT_ENV ]; then
        warnMsg "Please activate your virtual environment first."
        exit 1
    fi
    # double check
    which python3
    continueOrExit "the above python virtualenv"
}

# check my dev envrionment
checkDevEnv() {
    infoMsg "Checking my dev environment..."
    local CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "${CURR_DIR}/../check_cuda.py"
}
