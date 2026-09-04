#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Preflight for deepstream-import-vision-model: verify the environment BEFORE running the
# import loop. Everything runs THROUGH the DeepStream container (no host packages). Checks:
#   1. docker daemon reachable            (host-mode only)
#   2. the DeepStream image is pulled     (host-mode only)
#   3. GPU visible inside the container
#   4. the venv has the ONNX-export + report packages, and trtexec is present
#
# Usage:
#   Host (Linux):  bash scripts/preflight.sh
#   Through Docker (any OS — the portable way):
#     docker run --rm --gpus all -v "$PWD":/work -w /work --entrypoint bash \
#       nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
#       .claude/skills/deepstream-import-vision-model/scripts/preflight.sh
set -u
IMG="${1:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
VENV="${2:-build/.venv_optimum}"
PKGS="torch torchvision transformers onnx onnxruntime onnxscript huggingface_hub matplotlib numpy markdown reportlab"
fail=0
ok(){  echo "  [OK]      $1"; }
warn(){ echo "  [WARN]    $1"; }
bad(){  echo "  [MISSING] $1"; fail=1; }

# Container-mode: when run INSIDE the container (Windows/macOS invoke it as
#   docker run --gpus all -v <pwd>:/work -w /work <image> bash scripts/preflight.sh
# ), the host docker/image checks are moot and there's no nested docker CLI — verify GPU +
# venv + trtexec DIRECTLY. On a Linux host (no /.dockerenv) fall through to host-orchestration.
if [ -f /.dockerenv ]; then
  echo "[preflight] container-mode (inside the container — verifying GPU + venv + trtexec directly)"
  echo "[preflight] 1/3 GPU (nvidia-smi)"
  nvidia-smi -L >/dev/null 2>&1 && ok "GPU visible in container" \
    || bad "no GPU in container — run with --gpus all (Windows: Docker Desktop WSL2 backend + NVIDIA driver)"
  echo "[preflight] 2/3 trtexec"
  [ -x /usr/src/tensorrt/bin/trtexec ] && ok "trtexec present (/usr/src/tensorrt/bin/trtexec)" \
    || bad "trtexec not found — is this the DeepStream/TensorRT image?"
  echo "[preflight] 3/3 python packages in $VENV"
  if [ -x "$VENV/bin/python" ]; then
    miss=$("$VENV/bin/python" -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
    if [ -z "${miss// /}" ]; then ok "all packages present"
    else bad "missing in venv: $miss  (run setup.sh to (re)build the venv)"; fi
  else
    bad "venv not found: $VENV/bin/python — run setup.sh first (creates build/.venv_optimum)"
  fi
  echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — environment ready' || echo 'FAIL — fix [MISSING] items before running')"
  exit $fail
fi

echo "[preflight] 1/4 docker daemon"
docker info >/dev/null 2>&1 && ok "docker reachable" || bad "docker not installed or daemon not running"

echo "[preflight] 2/4 DeepStream image: $IMG"
have_img=0
if docker image inspect "$IMG" >/dev/null 2>&1; then ok "image present"; have_img=1
else warn "image not pulled — run: docker pull $IMG"; fi

if [ "$have_img" = 1 ]; then
  echo "[preflight] 3/4 GPU via docker --gpus all"
  docker run --rm --gpus all --entrypoint nvidia-smi "$IMG" -L >/dev/null 2>&1 \
    && ok "GPU visible in container" \
    || bad "no GPU through --gpus all (check driver + nvidia-container-toolkit)"

  echo "[preflight] 4/4 python packages in $VENV"
  if [ -x "$VENV/bin/python" ]; then
    miss=$(docker run --rm --entrypoint /work/"$VENV"/bin/python -v "$PWD":/work "$IMG" \
      -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
    if [ -z "${miss// /}" ]; then ok "all packages present"
    else bad "missing in venv: $miss  (run setup.sh)"; fi
  else
    bad "venv not found: $VENV/bin/python — run setup.sh first"
  fi
else
  echo "[preflight] 3/4 GPU            — SKIPPED (image not pulled)"
  echo "[preflight] 4/4 python packages — SKIPPED (image not pulled)"
fi

echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — environment ready' || echo 'FAIL — fix [MISSING] items before running')"
exit $fail
