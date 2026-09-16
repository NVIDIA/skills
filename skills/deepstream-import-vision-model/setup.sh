#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# One-command environment bootstrap for deepstream-import-vision-model on a FRESH machine.
# Creates the shared venv (build/.venv_optimum), installs the ONNX-export + report Python deps,
# and installs wkhtmltopdf (for the PDF report) — ALL INSIDE the container. Idempotent.
#
# Nothing runs on the host: the host only needs Docker + the NVIDIA driver. Run this INSIDE the
# DeepStream container, from the working root (where models/, reports/, build/ will live) — so
# torch/CUDA/TensorRT and the compiled parser all match the runtime:
#
#   # Linux / WSL2 bash:
#   docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
#     .claude/skills/deepstream-import-vision-model/setup.sh
#   # PowerShell: use -v "${PWD}:/work"   ·   cmd: -v "%cd%:/work"   (see references/windows.md)
set -euo pipefail

SK=".claude/skills/deepstream-import-vision-model"
VENV="build/.venv_optimum"
PY="$VENV/bin/python"

if [ ! -d "$SK" ]; then
  echo "[setup] ERROR: run from the working root (the dir that contains $SK)." >&2
  exit 1
fi

# 1) virtualenv (the container python lacks ensurepip, so bootstrap virtualenv via pip)
if [ ! -x "$PY" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m pip install --quiet --user virtualenv 2>/dev/null || python3 -m pip install --quiet virtualenv
  python3 -m virtualenv "$VENV"
else
  echo "[setup] venv exists: $VENV"
fi

# 2) Python dependencies (ONNX export + report)
echo "[setup] installing Python deps from $SK/scripts/requirements.txt (this can take several minutes)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install -r "$SK/scripts/requirements.txt"

# 3) wkhtmltopdf for the PDF report — installed IN the container (not a host dependency).
#    Self-contained Qt-WebKit renderer; no browser/Chromium needed.
if command -v wkhtmltopdf >/dev/null 2>&1; then
  echo "[setup] wkhtmltopdf already present"
else
  echo "[setup] installing wkhtmltopdf (apt, in-container)"
  if [ "$(id -u)" = "0" ]; then APT=""; else APT="sudo"; fi
  $APT apt-get update -qq && $APT apt-get install -y -qq wkhtmltopdf \
    || echo "[setup] WARN: wkhtmltopdf install failed — the HTML report still works; PDF step will skip"
fi

# 4) verify
echo "[setup] verifying…"
"$PY" - <<'PYV'
import importlib.util, sys
mods = ["torch","torchvision","transformers","onnx","onnxruntime","onnxscript",
        "huggingface_hub","matplotlib","numpy","markdown","reportlab"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if missing: print("  MISSING:", missing); sys.exit(1)
print("  all required packages present")
PYV
command -v wkhtmltopdf >/dev/null 2>&1 && echo "  wkhtmltopdf: $(command -v wkhtmltopdf)" || echo "  wkhtmltopdf: (absent — PDF step will skip)"
command -v /usr/src/tensorrt/bin/trtexec >/dev/null 2>&1 && echo "  trtexec: /usr/src/tensorrt/bin/trtexec" || echo "  trtexec: (check TensorRT in image)"

echo
echo "[setup] DONE. Next: run preflight, then the phases via the container —"
echo "  docker run --rm --gpus all -v \"\$PWD\":/work -w /work --entrypoint bash \\"
echo "    nvcr.io/nvidia/deepstream:9.1-triton-multiarch $SK/scripts/preflight.sh"
echo "  (see $SK/SKILL.md + $SK/references/windows.md for the per-shell mount token)"
