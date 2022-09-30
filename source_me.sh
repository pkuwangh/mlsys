#!/bin/bash

export TMOUT=

alias lt='ls -lhrt'

# cpu
lscpu | grep "NUMA node" | grep CPU
# memory
lsmem | grep "online memory"
numastat -m | grep -G "Node.*Total\|MemTotal"
# gpu
lspci | grep -i nvidia
NUM_GPUs="$(lspci | grep -i nvidia | wc -l)"
echo -e "\033[0;32mNumber of NVidia GPUs: ${NUM_GPUs}\033[0m"
# network
lspci | grep Eth | head -n1 | grep Eth
NUM_NICs="$(lspci | grep Eth | wc -l)"
echo -e "\033[0;32mNumber of NICs: ${NUM_NICs}\033[0m"
