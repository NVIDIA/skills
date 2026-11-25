#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM container.
# NGC_API_KEY is read from your environment; its value is never written to this file.
set -euo pipefail
: "${NGC_API_KEY:?export NGC_API_KEY before running this script}"
mkdir -p "$HOME/.cache/nim"
docker run -d \
  --name nim-llama-31-8b \
  --label nim-deploy.run=golden \
  --gpus all \
  --shm-size=16g \
  -u "$(id -u):0" \
  -e NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -p 8000:8000 \
  --restart unless-stopped \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
