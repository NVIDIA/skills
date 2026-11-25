#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Guard hook (PreToolUse, matcher: Bash) — block known dead ends BEFORE the
# agent burns tokens on them.
#
# Contract: the hook receives the tool call as JSON on stdin. Exit 2 blocks the
# call and shows stderr to the agent (write the message as advice: say WHY it
# is blocked and WHAT to do instead). Exit 0 allows the call.
#
# Add one guard per quirk your first run discovers. Real examples from the
# study are in the Pi adapter's guard.ts (../../pi/guard.ts): a GPU/cuDNN
# incompatibility, a disk headroom check, and a container CLI quirk.
CMD=$(jq -r '.tool_input.command // empty' 2>/dev/null)
[ -z "$CMD" ] && exit 0

# --- Example guard: require disk headroom before a training launch ---------
# if echo "$CMD" | grep -q "your_train_command"; then
#   FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
#   if [ "${FREE:-0}" -lt 30 ]; then
#     echo "GUARD(disk): only ${FREE}G free. Training writes ~NG of checkpoints; free space first." >&2
#     exit 2
#   fi
# fi

exit 0
