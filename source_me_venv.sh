#!/bin/bash

# get current directory
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# if virtualenv already exists
if ! [ -f "${CURR_DIR}/workspace/my_venv/bin/activate" ]; then
    echo "Creating virtual env my_venv..."
    virtualenv "${CURR_DIR}/workspace/my_venv"
else
    echo "Virtual env my_venv already exists."
fi

if [ -f "${CURR_DIR}/workspace/my_venv/bin/activate" ]; then
    echo "Activating virtual env my_venv..."
    source "${CURR_DIR}/workspace/my_venv/bin/activate"
fi

# alias
alias lt='ls -lhrt'

