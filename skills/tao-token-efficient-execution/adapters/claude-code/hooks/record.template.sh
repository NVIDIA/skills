#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Recorder hook (PostToolUse, matcher: Bash) — append every substantive
# executed command to the run's commands.log so later sessions and later runs
# LOAD AND RUN instead of re-deriving.
#
# Edit two things for your workflow:
#   1. the case patterns — which commands are worth recording
#   2. the run-dir glob  — where the current run's commands.log lives
CMD=$(jq -r '.tool_input.command // empty' 2>/dev/null)
[ -z "$CMD" ] && exit 0
case "$CMD" in
  *docker\ run*|*your_train_command*|*your_eval_command*)
    RD=$(ls -td "${WS:-$HOME/workspace}"/results/run_* 2>/dev/null | head -1)
    [ -n "$RD" ] && { printf '### %s\n%s\n\n' "$(date -u +%FT%TZ)" "$CMD" >> "$RD/commands.log"; } ;;
esac
exit 0
