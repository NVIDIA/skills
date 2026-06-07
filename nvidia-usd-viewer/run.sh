#!/usr/bin/env bash
# Start the USD Viewer RTX streaming server.
# Run after ./setup.sh
set -euo pipefail

# Activate venv (Linux/macOS vs Windows Git Bash)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  echo "ERROR: .venv not found. Run setup.sh first." >&2
  exit 1
fi

# Required before any ovrtx import.
export OVRTX_SKIP_USD_CHECK=1

# Derive OVRTX_BIN_PATH from the installed package (resolves renderer plugins).
OVRTX_BIN_PATH=$(python -c \
  'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))' \
  2>/dev/null || true)
if [ -n "$OVRTX_BIN_PATH" ] && [ -d "$OVRTX_BIN_PATH" ]; then
  export OVRTX_BIN_PATH
  export LD_LIBRARY_PATH="${OVRTX_BIN_PATH}/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="${OVRTX_BIN_PATH}:$PATH"
fi

# Project-local caches (avoids permission issues in containers).
mkdir -p .cache/cuda .cache/gl .cache/warp
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$PWD/.cache/gl"

exec python -m server.ov_web_viewer_server \
  --width          1920 \
  --height         1080 \
  --fps            30 \
  --signaling-port 49100 \
  --http-port      8888 \
  --public-ip      127.0.0.1 \
  --scene          assets/samples/scene.usda \
  "$@"
