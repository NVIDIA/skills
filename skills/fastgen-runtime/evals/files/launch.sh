#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export FASTGEN_OUTPUT_ROOT=outputs
cd /workspace/project
/opt/task-env/bin/python train.py --config /workspace/project/fastgen/configs/experiments/WanT2V/config_dmd2.py trainer.max_iter=3 --dryrun
