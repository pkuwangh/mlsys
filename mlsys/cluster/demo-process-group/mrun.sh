#!/bin/bash

# if $MADDR is not set, set it to the IP address of the current node
if [ -z ${MADDR} ]; then
    MADDR=$(hostname -I | awk '{print $1}')
fi
# init master address and port
export MASTER_ADDR=${MADDR}
export MASTER_PORT=6379

# check if the environment variables are set
if [ -z ${OMPI_COMM_WORLD_RANK} ]; then
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
else
    export RANK=${OMPI_COMM_WORLD_RANK}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
fi

if [ -z ${MLSYS_ROOT} ]; then
    # get the directory of this script
    CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export MLSYS_ROOT="${CURR_DIR}/../../.."
fi

# set python path
export PYTHONPATH=${MLSYS_ROOT}/micromamba/envs/mlsys-pg/lib/python3.10/site-packages
export PYTHONUSERBASE="${PYTHONPATH}"
# set system paths
export LD_LIBRARY_PATH="${MLSYS_ROOT}/mlsys/environments/nccl/build/lib:${LD_LIBRARY_PATH}"
export PATH="${MLSYS_ROOT}/micromamba/envs/mlsys-pg/bin:${PATH}"

# set NCCL interface
export NCCL_SOCKET_IFNAME=enP4s1f1
# export NCCL_DEBUG=INFO

export DEAD_RANK="9"
export NCCL_ALL_REDUCE_HANG_POINT="6"
# echo "Rank=${RANK} on $(hostname)"

python3 "${MLSYS_ROOT}/mlsys/cluster/demo-process-group/distr.py"

