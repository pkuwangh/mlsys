#!/usr/bin/env bash
# Usage: bash docker/build-agentx-enroot.sh OUTPUT.sqsh [AGENTX_VERSION]
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=${1:?Usage: build-agentx-enroot.sh OUTPUT.sqsh [AGENTX_VERSION]}
base_sqsh="${output%.sqsh}.base.sqsh"
container_name=$(basename -- "${output%.sqsh}")
base_image=${ENROOT_BASE_IMAGE:-docker://docker.io#python:3.12-slim}
mkdir -p -- "$(dirname -- "$output")"

if [[ ! -f "$base_sqsh" ]]; then
  enroot import --output "$base_sqsh" "$base_image"
fi
enroot create --force --name "$container_name" "$base_sqsh"
enroot start --root --rw --env NVIDIA_VISIBLE_DEVICES=void \
  --mount "$script_dir/install-agentx.sh:/opt/install-agentx.sh:ro,bind,x-create=file" \
  "$container_name" bash /opt/install-agentx.sh "${2:-${AGENTX_VERSION:-}}"
enroot export --force --output "$output" "$container_name"
enroot remove --force "$container_name"
printf 'Built %s\n' "$output"
