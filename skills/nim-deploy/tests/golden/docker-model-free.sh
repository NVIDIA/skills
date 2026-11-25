#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM container.
# NGC_API_KEY is read from your environment; its value is never written to this file.
set -euo pipefail
: "${NGC_API_KEY:?export NGC_API_KEY before running this script}"
: "${HF_TOKEN:?export HF_TOKEN before running this script}"
mkdir -p "$HOME/.cache/nim"
docker run -d \
  --name nim-llama-31-8b \
  --label nim-deploy.run=golden \
  --gpus 1 \
  --shm-size=16g \
  -e NGC_API_KEY \
  -e HF_TOKEN \
  -e NIM_MODEL_PATH=hf://Qwen/Qwen3-0.6B \
  -e NIM_SERVED_MODEL_NAME=qwen3 \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -p 8000:8000 \
  --restart no \
  nvcr.io/nim/nvidia/llm-nim:2.0.0
