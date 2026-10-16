#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# No-GPU smoke test for the kit's Pi adapter. Verifies, with tiny real API
# calls:
#   T1  guard.ts blocks a known-bad command and the reason reaches the model
#       (and blocked commands are NOT recorded)
#   T2  recorder.ts appends executed matching commands to $RD/commands.log
#   T3  per-message usage lands in the session JSONL and analyze_usage.py
#       parses it (also prints the measured per-session floor)
# No docker, no training, no writes outside $SMOKE.
# Requires: the API key for SMOKE_MODEL exported in this shell.
# ============================================================================
set -u
[ -d "$HOME/.local/share/pi-node/current/bin" ] && export PATH="$HOME/.local/share/pi-node/current/bin:$PATH"
SKILL=$(cd "$(dirname "$0")/.." && pwd)
ADAPTER=$SKILL/adapters/pi
SMOKE=${SMOKE_DIR:-$HOME/.tao-kit/smoke}
SMOKE_MODEL=${SMOKE_MODEL:-nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B:off}
PASS=0; FAIL=0

say(){ printf '%s\n' "$*"; }
ok(){ PASS=$((PASS+1)); say "  PASS: $*"; }
bad(){ FAIL=$((FAIL+1)); say "  FAIL: $*"; }

say "== preflight =="
command -v pi >/dev/null || { say "ABORT: pi not on PATH"; exit 1; }
command -v jq >/dev/null || { say "ABORT: jq required"; exit 1; }
say "  pi version: $(pi --version 2>/dev/null | head -1)"
case "$SMOKE_MODEL" in
  nim/*)       KEYVAR=NVIDIA_INFERENCE_API_KEY ;;
  anthropic/*) KEYVAR=ANTHROPIC_API_KEY ;;
  *)           KEYVAR= ;;
esac
if [ -n "$KEYVAR" ] && [ -z "$(eval "printf %s \"\${$KEYVAR:-}\"")" ]; then
  say "ABORT: $KEYVAR is not set. Run:  export $KEYVAR=...  and retry."
  exit 1
fi
[ -n "$KEYVAR" ] && say "  $KEYVAR: SET"
say "  model: $SMOKE_MODEL"

# $SMOKE is recreated fresh each run — refuse to wipe a directory this test
# did not create (a mispointed SMOKE_DIR must not delete user data).
if [ -e "$SMOKE" ] && [ ! -e "$SMOKE/.tao-smoke" ]; then
  say "ABORT: $SMOKE exists and was not created by this smoke test; refusing to delete it."
  say "Point SMOKE_DIR at a fresh, dedicated directory."
  exit 1
fi
rm -rf "$SMOKE"
mkdir -p "$SMOKE/sessions" "$SMOKE/ws/results/run_smoke"
touch "$SMOKE/.tao-smoke"
export PI_KIT_WS="$SMOKE/ws"
unset PI_KIT_RD 2>/dev/null || true

SP="You are a precise task executor operating in a bash environment. You MUST perform every action by calling your tools — never describe, simulate, or invent a result or command output. Work alone; never ask questions."
run_pi(){ # $1=prompt $2=outfile
  timeout 240 pi -p -na --model "$SMOKE_MODEL" \
    --system-prompt "$SP" \
    --tools bash --no-context-files --no-skills --no-extensions \
    -e "$ADAPTER/nvidia-provider.ts" -e "$ADAPTER/guard.ts" -e "$ADAPTER/recorder.ts" \
    --session-dir "$SMOKE/sessions" \
    "$1" > "$2" 2>&1
}

say "== T1: guard blocks known-bad command =="
run_pi "Run exactly this bash command (do not modify it): echo probe image_embeddings --gpus
Then say done." "$SMOKE/t1.out"
if grep -rq "GUARD(sm_75-embedding)" "$SMOKE/sessions" "$SMOKE/t1.out"; then
  ok "guard reason found in session/output"
else
  bad "GUARD(sm_75-embedding) not found — guard did not fire (see $SMOKE/t1.out)"
fi
# The model may legitimately retry WITHOUT --gpus after the guard's advice;
# only the blocked variant (with --gpus) must never be recorded.
if grep "image_embeddings" "$SMOKE/ws/results/run_smoke/commands.log" 2>/dev/null | grep -q -- "--gpus"; then
  bad "blocked command WAS recorded to commands.log (should not be)"
else
  ok "blocked command not recorded"
fi

say "== T2: recorder appends executed command =="
run_pi "Run exactly this bash command (do not modify it): echo smoke run visual_changenet ok
Then say done." "$SMOKE/t2.out"
if grep -q "visual_changenet ok" "$SMOKE/ws/results/run_smoke/commands.log" 2>/dev/null; then
  ok "command recorded to commands.log"
else
  bad "command missing from commands.log (see $SMOKE/t2.out)"
fi

say "== T3: usage accounting parses =="
NSESS=$(ls "$SMOKE/sessions"/*.jsonl 2>/dev/null | wc -l)
if [ "$NSESS" -ge 2 ]; then ok "$NSESS session files written"; else bad "expected >=2 session files, got $NSESS"; fi
if OUT=$(python3 "$SKILL/scripts/analyze_usage.py" "$SMOKE/sessions" 2>&1); then
  say "$OUT" | sed 's/^/  | /'
  if say "$OUT" | grep -q "bill (input-eq):" && ! say "$OUT" | grep -q "bill (input-eq):      0$"; then
    ok "analyzer produced a non-zero bill"
  else
    bad "analyzer output missing or zero bill"
  fi
else
  bad "analyze_usage.py crashed: $OUT"
fi

say ""
say "== result: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
