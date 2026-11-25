#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM (nim CLI).
# nim reads NGC_API_KEY and HF_TOKEN from your environment or config; no secret here.
set -euo pipefail
nim run meta/llama-3.1-8b-instruct -p 8100 --gpus 0,1 -d --no-auto-start --profile vllm-bf16-tp2 -e NIM_MAX_MODEL_LEN=8192 </dev/null
