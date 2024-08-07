#!/bin/bash

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLSYS_ROOT="${CURR_DIR}/../../.."
TARGET_BIN="${CURR_DIR}/mrun.sh"

MADDR=$(hostname -I | awk '{print $1}')

# still need set LD_LIBRARY_PATH for CUDA

mpirun \
	--hostfile hostlist.local \
	-np 16 \
	-N 1 \
	--bind-to none \
	-x MADDR=$MADDR \
    -x MLSYS_ROOT=$MLSYS_ROOT \
	-x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
	--mca btl tcp,self \
	--mca btl_tcp_if_include enP4s1f1 \
	--allow-run-as-root \
	--tag-output \
	$TARGET_BIN
