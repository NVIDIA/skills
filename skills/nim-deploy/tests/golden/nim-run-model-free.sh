#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM (nim CLI).
# nim reads NGC_API_KEY and HF_TOKEN from your environment or config; no secret here.
set -euo pipefail
nim run nvcr.io/nim/nvidia/model-free-nim:2.1.4 -p 8000 --gpus all -d --no-auto-start -e NIM_MODEL_PATH=hf://Qwen/Qwen3-0.6B </dev/null
