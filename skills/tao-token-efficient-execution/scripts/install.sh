#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Token-efficient execution kit — install / preflight.
#
# Deliberately minimal and non-invasive: checks prerequisites, detects which
# agent harnesses are available, scaffolds ~/.tao-kit/kit.env (never
# overwrites), and prints exact next steps. It does NOT modify any harness's
# global settings and does NOT read or write credentials.
# ============================================================================
set -u
SKILL=$(cd "$(dirname "$0")/.." && pwd)
BANK=$(cd "$SKILL/../../.." && pwd)
KIT_HOME=${KIT_HOME:-$HOME/.tao-kit}
OK=0; MISS=0

say() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
check() { # $1=label $2=cmd
  if have "$2"; then OK=$((OK+1)); say "  OK:      $1"; else MISS=$((MISS+1)); say "  MISSING: $1"; fi
}

say "== tao-token-efficient-execution install =="
say "bank: $BANK"
say ""
say "-- core prerequisites (drivers are plain bash) --"
check "bash" bash
check "jq (driver routing on loop_log.jsonl)" jq
check "python3 (usage accounting, skill helper scripts)" python3
check "docker (shipped card packs launch TAO containers)" docker

say ""
say "-- agent harnesses (need at least one; adapters/ has the matching guard+recorder) --"
HARNESS=""
if have pi; then say "  OK:      pi ($(pi --version 2>/dev/null | head -1)) -> adapters/pi (extensions, used by the shipped pack drivers)"; HARNESS=pi; fi
if have claude; then say "  OK:      claude -> adapters/claude-code (PreToolUse/PostToolUse hooks + templates/driver.template.sh)"; HARNESS=${HARNESS:-claude}; fi
[ -z "$HARNESS" ] && { MISS=$((MISS+1)); say "  MISSING: no agent harness found (install pi or claude)"; }

say ""
say "-- config scaffold --"
mkdir -p "$KIT_HOME"
if [ -f "$KIT_HOME/kit.env" ]; then
  say "  KEPT:    $KIT_HOME/kit.env (already exists, not touched)"
else
  cp "$SKILL/templates/kit.env.template" "$KIT_HOME/kit.env"
  say "  WROTE:   $KIT_HOME/kit.env (edit WS, MODEL, and pack-specific values)"
fi

say ""
say "-- shipped card packs --"
found=0
for pack in "$BANK"/skills/applications/*/cards; do
  [ -d "$pack" ] || continue
  found=1
  say "  $(basename "$(dirname "$pack")"): $pack"
done
[ "$found" -eq 0 ] && say "  (none found)"

say ""
say "== next steps =="
say "  1. Edit $KIT_HOME/kit.env (set WS; VENV for the AutoML pack)"
say "  2. Export the API key for your model provider (e.g. NVIDIA_INFERENCE_API_KEY)"
say "  3. Sanity-check the adapters without a GPU:"
say "       bash $SKILL/scripts/smoke_test.sh"
say "  4. Launch a pack: nohup bash <pack>/driver.sh & then tail ~/.tao-kit/<pack>/driver.log"
say ""
if [ "$MISS" -gt 0 ]; then say "RESULT: $MISS missing prerequisite(s) above"; exit 1; fi
say "RESULT: ready"
