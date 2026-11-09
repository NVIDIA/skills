#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export PYTHONPATH=/workspace/old-project
export FASTGEN_OUTPUT_ROOT=/workspace/outputs
cd /workspace/old-project
exec /opt/task-env/bin/python train.py --config fastgen/configs/experiments/SD15/config_sft.py --dryrun
