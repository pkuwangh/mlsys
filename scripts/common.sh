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
        return 1
    fi
    return 0
}

# cleanup conda leftover
cleanupCondaBackEnvs() {
    for v in $(env | awk -F= '/^CONDA_BACKUP_/ {print $1}'); do unset "$v"; done
}

# get current virtual env
getVenv() {
    local MY_VENV=$(python3 -c 'import sys ; print( sys.prefix.split("/")[-1] )')
    echo $MY_VENV
}

# check if the virtual environment is activated
checkVenv() {
    infoMsg "Checking python virtual environment..."
    if [ -z $CONDA_DEFAULT_ENV ]; then
        warnMsg "Please activate your virtual environment first."
        return 1
    fi
    # double check
    which python3
    continueOrExit "the above python virtual environment"
    return $?
}
