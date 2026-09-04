#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# Step 1 (alternate): Convert SafeTensors model to ONNX.
#
# Uses the shared venv's torch.onnx.export via safetensors_to_onnx.py. The former
# optimum-cli path was removed: optimum held transformers below 4.54.0, which blocked
# the releases that fix its RCE advisories, and optimum 2.1.0 dropped the `onnx`
# subcommand outright.
#
# Writes <output_dir>/model.onnx — the same filename optimum produced.
#
# Usage: ./safetensors-to-onnx.sh <hf_model_id_or_path> <output_dir> [extra args]
# Extra args are passed through to safetensors_to_onnx.py, e.g.
#   --opset 17 · --image-size 640 · --max-batch 16 · --static-batch
# Examples:
#   ./safetensors-to-onnx.sh PekingU/rtdetr_r50vd ./onnx_export
#   ./safetensors-to-onnx.sh facebook/detr-resnet-50 ./onnx_export --opset 17
#   ./safetensors-to-onnx.sh ./local_model_dir ./onnx_export --image-size 800
################################################################################
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hf_model_id_or_path> <output_dir> [extra args for safetensors_to_onnx.py]"
    echo ""
    echo "Examples:"
    echo "  $0 PekingU/rtdetr_r50vd ./onnx_export"
    echo "  $0 facebook/detr-resnet-50 ./onnx_export --opset 17"
    exit 1
fi

MODEL="$1"
OUTPUT_DIR="$2"
shift 2
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Installed layout is <root>/.claude/skills/deepstream-import-vision-model/scripts/model/, so the
# working root (where setup.sh created build/.venv_optimum) is five levels up from this script.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
mkdir -p "$REPO_ROOT/build"
VENV_DIR="$REPO_ROOT/build/.venv_optimum"
VENV_PY="$VENV_DIR/bin/python"

echo "=== SafeTensors → ONNX Export ==="
echo "Model:      $MODEL"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: ${EXTRA_ARGS[*]-}"
echo "Venv:       $VENV_DIR"
echo ""

# The venv is built ONCE, IN THE CONTAINER, by setup.sh — with virtualenv, because the DeepStream
# container's python lacks ensurepip (so `python3 -m venv` would fail here). Reuse it; never recreate.
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv not found at $VENV_DIR — run the one-time bootstrap first (in-container):" >&2
    echo "  docker run --rm -it --gpus all --shm-size=16g -v \"\$PWD\":/work -w /work \\" >&2
    echo "    --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \\" >&2
    echo "    .claude/skills/deepstream-import-vision-model/setup.sh" >&2
    exit 1
fi
echo "Using venv: $VENV_DIR"
echo ""

# ${arr[@]+"${arr[@]}"} expands to nothing when the array is empty. Plain "${arr[@]-}"
# would pass a single empty-string argument instead, which argparse rejects.
"$VENV_PY" "$SCRIPT_DIR/safetensors_to_onnx.py" \
    --model "$MODEL" --output-dir "$OUTPUT_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=== Export Complete ==="
ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null
