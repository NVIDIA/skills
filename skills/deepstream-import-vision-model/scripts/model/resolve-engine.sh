#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Apache-2.0.
#
# Resolve the engine that batch scaling actually produced, plus the values derived from it.
#
# WHY a helper: Step 6/7 (pipeline-run.md) and Step 8 (report-generation.md) both need the same
# engine, and both used to inline the same lookup. They drifted — pipeline-run.md selected with
# `head -1` and so picked b1 while report-generation.md picked the highest MAX_BS, meaning the
# benchmark and the report could describe different engines. One implementation cannot drift.
#
# Selection rule: `sort -V | tail -1` takes the HIGHEST MAX_BS, i.e. the final engine from
# iterative scaling. A lexicographic `head -1` would take b1 out of b1/b16/b8.
#
# USAGE — evaluate the output to set ENGINE / MAX_BS / MODEL_FILENAME in the caller's shell:
#     eval "$(bash .claude/skills/deepstream-import-vision-model/scripts/model/resolve-engine.sh "$MODEL_NAME")"
#
# Exits non-zero with a message on stderr when no engine is present.
set -euo pipefail

MODEL_NAME="${1:?usage: resolve-engine.sh <MODEL_NAME>}"
ENGINE_DIR="models/${MODEL_NAME}/benchmarks/engines"

ENGINE=$(ls "${ENGINE_DIR}"/*_dynamic_b*.engine 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$ENGINE" ]; then
    echo "ERROR: No engine found in ${ENGINE_DIR}/ — run Steps 4-5 first (references/engine-build.md)" >&2
    exit 1
fi

MAX_BS=$(echo "$ENGINE" | grep -oP '_b\K[0-9]+(?=\.engine)')
MODEL_FILENAME=$(basename "$ENGINE" | sed 's/_dynamic_b[0-9]*\.engine//')

printf 'ENGINE=%q\n'          "$ENGINE"
printf 'MAX_BS=%q\n'          "$MAX_BS"
printf 'MODEL_FILENAME=%q\n'  "$MODEL_FILENAME"
printf 'echo "Using engine: %s (MAX_BS=%s)"\n' "$ENGINE" "$MAX_BS"
