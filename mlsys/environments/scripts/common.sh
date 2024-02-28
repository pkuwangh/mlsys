#!/bin/bash

infoMsg() {
    echo -e "\033[32;1m${1}\033[0m"
}

warnMsg() {
    echo -e "\033[31;1m${1}\033[0m"
}

debugMsg() {
    echo -e "\033[33;5m${1}\033[0m"
}

# take one argument as the message

continueOrExit() {
    read -p "Continue with ${1}? (y/N)" -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}
