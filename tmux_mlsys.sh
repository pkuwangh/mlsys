#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# alias
tmux a -t mlsys || tmux -f ${CURR_DIR}/dotfiles/env_tmux.conf new -s mlsys
