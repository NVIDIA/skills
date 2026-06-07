#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export OVRTX_SKIP_USD_CHECK=1

exec python3 -m server.ov_web_viewer_server \
  --width  1920 \
  --height 1080 \
  --fps    30 \
  --signaling-port 49100 \
  --http-port      8888 \
  --public-ip      127.0.0.1 \
  --scene  assets/samples/scene.usda \
  "$@"
