#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM container.
# NGC_API_KEY is read from your environment; its value is never written to this file.
set -euo pipefail
: "${NGC_API_KEY:?export NGC_API_KEY before running this script}"
mkdir -p '/data/nim cache'
docker run -d \
  --name nim-llama-31-8b \
  --label nim-deploy.run=golden \
  --gpus '"device=0,1"' \
  --shm-size=32g \
  -u "$(id -u):0" \
  -e NGC_API_KEY \
  -e NIM_MODEL_PROFILE=vllm-bf16-tp2 \
  -v '/data/nim cache:/opt/nim/.cache' \
  -p 8100:8000 \
  --restart unless-stopped \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
