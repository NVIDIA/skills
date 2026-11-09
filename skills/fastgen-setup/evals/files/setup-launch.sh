#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd /workspace/project
export FASTGEN_OUTPUT_ROOT=outputs
export HF_HOME=.cache
exec /opt/task-env/bin/python scripts/inference/image_model_inference.py \
  --config fastgen/configs/experiments/Flux/config_sft.py \
  --prompt_file /inputs/prompts.txt --do_teacher_sampling True --do_student_sampling False \
  --image_save_dir /outputs/teacher - trainer.seed=42
