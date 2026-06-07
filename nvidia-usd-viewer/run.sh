#!/usr/bin/env bash
# Start the USD Viewer RTX streaming server.
# Run after ./setup.sh
set -euo pipefail

source .venv/bin/activate

# Required: skip the USD environment check before ovrtx is imported.
export OVRTX_SKIP_USD_CHECK=1

# Derive OVRTX_BIN_PATH from the installed package (resolves renderer plugins).
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))')"
export LD_LIBRARY_PATH="${OVRTX_BIN_PATH}/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Use project-local cache (avoids permission issues in containers / CI).
mkdir -p .cache/cuda .cache/gl .cache/warp
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$PWD/.cache/gl"

exec python3 -m server.ov_web_viewer_server \
  --width          1920 \
  --height         1080 \
  --fps            30 \
  --signaling-port 49100 \
  --http-port      8888 \
  --public-ip      127.0.0.1 \
  --scene          assets/samples/scene.usda \
  "$@"
