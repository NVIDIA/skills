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
[ -d "$HOME/.local/share/pi-node/current/bin" ] && export PATH="$HOME/.local/share/pi-node/current/bin:$PATH"
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
check "jq (driver routing on run state JSON)" jq
check "python3 (usage accounting, skill helper scripts)" python3
check "docker (shipped card packs launch TAO containers)" docker

say ""
say "-- agent harnesses (adapters/ has the matching guard+recorder) --"
if have pi; then
  OK=$((OK+1)); say "  OK:      pi ($(pi --version 2>/dev/null | head -1)) -> adapters/pi (required by the shipped pack drivers)"
else
  MISS=$((MISS+1)); say "  MISSING: pi (required by the shipped pack drivers; install it or put it on PATH)"
fi
if have claude; then
  say "  OK:      claude -> adapters/claude-code (hooks + templates/driver.template.sh only; cannot run the shipped packs)"
else
  say "  SKIP:    claude (optional; only for packs built from templates/driver.template.sh)"
fi

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
say "  1. Edit $KIT_HOME/kit.env (set WS and MODEL; VENV for the AutoML pack, DS_IMG for the DEFT pack)"
say "  2. Export the API key for your MODEL's provider: NVIDIA_API_KEY for"
say "     build.nvidia.com (MODEL=nvidia/<id>); for your own endpoint (MODEL=nim/<id>)"
say "     set PI_KIT_NIM_BASE_URL and export NVIDIA_INFERENCE_API_KEY; local servers:"
say "     ~/.pi/agent/models.json. MODEL and the nim/ endpoint have no default."
say "  3. Sanity-check the adapters without a GPU (name the model; there is no default):"
say "       SMOKE_MODEL=<provider>/<model-id> bash $SKILL/scripts/smoke_test.sh"
say "  4. Launch a pack: nohup bash <pack>/driver.sh & then tail ~/.tao-kit/<pack>/driver.log"
say ""
if [ "$MISS" -gt 0 ]; then say "RESULT: $MISS missing prerequisite(s) above"; exit 1; fi
say "RESULT: ready"
