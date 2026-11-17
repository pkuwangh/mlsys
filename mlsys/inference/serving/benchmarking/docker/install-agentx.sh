#!/usr/bin/env bash
# Run inside a Debian-based Python container, through Docker or Enroot.
set -euo pipefail

agentx_version=${1:-${AGENTX_VERSION:-1.0.13}}
if [[ ! "$agentx_version" =~ ^[0-9][0-9A-Za-z.+-]*$ ]]; then
  echo "Expected an exact AgentX version, got: $agentx_version" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates
python3 -m pip install --no-cache-dir "agentx-harness==$agentx_version"
python3 -m pip check
agentx --help >/dev/null

# Keep the resolved dependency versions in both kinds of image.
mkdir -p /opt/agentx
python3 -m pip freeze --all > /opt/agentx/requirements-installed.txt
python3 -c 'import importlib.metadata as m; print("agentx-harness=" + m.version("agentx-harness"))'
rm -rf /var/lib/apt/lists/*
