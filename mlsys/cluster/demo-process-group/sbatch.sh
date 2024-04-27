#!/bin/bash

#SBATCH --job-name=process-group --nodes=4 --ntasks-per-node=1 --gpus-per-node=1

CURR_DIR=`pwd`
export MLSYS_ROOT="${CURR_DIR}/../../.."
TARGET_BIN="${MLSYS_ROOT}/mlsys/cluster/demo-process-group/srun.sh"

export MADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun -u -l \
    --container-image=nvcr.io/nvidia/pytorch:23.12-py3 \
    --container-mounts "${MLSYS_ROOT}:${MLSYS_ROOT}" \
    "${TARGET_BIN}"
