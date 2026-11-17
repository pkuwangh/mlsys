#!/usr/bin/env bash
set -euo pipefail

# One node's four TP workers are enough to inspect prefill/decode kernels.
if [[ ${SLURM_PROCID:?} != 0 ]]; then
    exec "$@"
fi
mkdir -p /experiment/nsys
python3 /profiling/capture-profile.py > /experiment/nsys/controller.log 2>&1 &
exec /opt/nsight/target-linux-sbsa-armv8/nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --trace-fork-before-exec=true --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=repeat:2:async --kill=none \
    --output=/experiment/nsys/rank0 "$@"
