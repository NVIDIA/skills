#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# Convenience wrapper: run a command INSIDE the DeepStream container with the working root
# bind-mounted at /work. Every phase of this skill runs this way — nothing runs on the host.
#
#   bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh '<command to run in-container>'
#   e.g. bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh 'build/.venv_optimum/bin/python -c "import torch;print(torch.__version__)"'
#        bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh 'make -C models/yolov8n/parser && ls models/yolov8n/parser/*.so'
#
# Env overrides: DS_IMAGE (default DeepStream 9.1), DS_GPU (default "--gpus all", set "" for CPU-only steps).
# On PowerShell the -v token is "${PWD}:/work"; on cmd "%cd%:/work" — Claude Code sets it per host shell.
# See references/windows.md.
set -euo pipefail
IMG="${DS_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
GPU="${DS_GPU-"--gpus all"}"
if [ "$#" -eq 0 ]; then echo "usage: bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh '<in-container command>'" >&2; exit 2; fi
exec docker run --rm $GPU --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash "$IMG" -lc "$*"
