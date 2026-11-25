#!/usr/bin/env bash
# Rendered by nim-deploy for run golden. Starts one NIM (nim CLI).
# nim reads NGC_API_KEY and HF_TOKEN from your environment or config; no secret here.
set -euo pipefail
nim run meta/llama-3.1-8b-instruct -p 8000 --gpus all -d --no-auto-start </dev/null
