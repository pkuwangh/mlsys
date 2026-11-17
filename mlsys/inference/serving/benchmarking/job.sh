#!/usr/bin/env bash
set -euo pipefail
experiment_dir=${1:?Usage: job.sh /absolute/run/directory}
exec python3 "$experiment_dir/run.py" job \
  --engine "$experiment_dir/engine.json" \
  --benchmark "$experiment_dir/benchmark.json" \
  --deployment "$experiment_dir/deployment.json"
