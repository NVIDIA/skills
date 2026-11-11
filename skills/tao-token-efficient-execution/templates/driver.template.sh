#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Harness-kit stage driver (template)
#
# Runs a staged agentic workflow as a series of FRESH headless Claude Code
# sessions instead of one long conversation. Each session executes exactly one
# stage card, appends "<stage> ok" to progress.log, and exits. State lives on
# disk, never in chat history.
#
# On the very first run (no cards exist yet) the driver runs an AUTHORING
# session: the agent reads your skills, explores the workspace, and writes the
# stage cards itself. See authoring_prompt.md.
#
# ── THE THREE CONFIG POINTS (edit these for your workflow) ──────────────────
#  1. STAGE ROUTING  — the case block mapping last completed stage -> next card
#  2. working()      — how to detect that a long external job is still running
#  3. Paths          — kit dir, workspace, results glob, done marker
# ============================================================================
set -u

# ---- 3. PATHS --------------------------------------------------------------
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}   # kit dir (cards/, .claude/)
WS=${WS:?set WS to your workspace root}      # data + skills workspace
RESULTS=$WS/results                          # where runs create their dirs
RUN_GLOB='run_*'                             # how run dirs are named
MODEL=${MODEL:-claude-fable-5}               # execution model
LOG=$KIT/driver.log
MARKER=$KIT/.launch_marker

# ---- 2. WORK PREDICATE -----------------------------------------------------
# Return 0 while the workflow's external jobs (training, inference, HPO
# runners) are still busy. THIS IS WORKFLOW SPECIFIC — a wrong predicate makes
# the driver fire sessions early or never. Check containers AND host processes.
working() {
  docker ps --format '{{.Image}}' | grep -qE 'your-container-image' && return 0
  pgrep -f "[y]our_runner_process" >/dev/null && return 0
  return 1
}

touch "$MARKER"; cd "$KIT"
echo "[driver] start $(date)" >> "$LOG"

# ---- ROUND 0: self-bootstrap (agent authors its own cards) -----------------
if [ ! -f "$KIT/cards/00"*.md ] 2>/dev/null && ! ls "$KIT"/cards/*.md >/dev/null 2>&1; then
  echo "[driver] round 0 -> AUTHORING $(date)" >> "$LOG"
  timeout 2400 claude -p "$(cat "$KIT/authoring_prompt.md")" \
    --model "$MODEL" --permission-mode auto >> "$LOG" 2>&1
  ls "$KIT/cards/" >> "$LOG"
fi
ls "$KIT"/cards/*.md >/dev/null 2>&1 || { echo "[driver] ABORT: no cards" >> "$LOG"; exit 1; }

noop=0
for round in $(seq 1 60); do
  while working; do sleep 30; done
  sleep 15; working && continue

  RD=$(find "$RESULTS" -maxdepth 1 -type d -name "$RUN_GLOB" -newer "$MARKER" 2>/dev/null | head -1)
  LAST=""
  # Route on the last OK stage only — error entries must not advance routing.
  [ -n "$RD" ] && [ -f "$RD/progress.log" ] && LAST=$(grep ' ok$' "$RD/progress.log" | tail -1 | awk '{print $1}')
  if [ -n "$RD" ] && tail -1 "$RD/progress.log" 2>/dev/null | grep -q ' FAIL$'; then
    echo "[driver] HALT: latest mark is FAIL — no auto-retry $(date)" >> "$LOG"
    exit 2
  fi
  [ -n "$RD" ] && [ -f "$RD/DONE.marker" ] && { echo "[driver] DONE after $((round-1)) sessions $(date)" >> "$LOG"; exit 0; }

  # ---- 1. STAGE ROUTING (edit for your workflow's stages) ------------------
  case "$LAST" in
    "")          CARD=00-first-stage.md ;;
    first_stage) CARD=10-second-stage.md ;;
    second_stage)CARD=20-third-stage.md ;;
    *)           CARD=90-finalize.md ;;
  esac

  # Snapshot the run state into the prompt so the fresh session needs no history
  STATE=""; PROG=""; CMDS=""
  [ -n "$RD" ] && STATE=$(head -c 2500 "$RD/state.json" 2>/dev/null)
  [ -n "$RD" ] && PROG=$(tail -5 "$RD/progress.log" 2>/dev/null)
  [ -n "$RD" ] && CMDS=$(tail -25 "$RD/commands.log" 2>/dev/null | head -c 3000)

  PROMPT="Execute exactly ONE stage card of a staged workflow, tersely, then end your turn.
Constants: WS=$WS RD=${RD:-<create per card>}
Rules: follow the card exactly; sessions are fresh so the card + snapshot below is your whole
memory; reuse commands.log entries instead of re-deriving commands; after any detached launch
END YOUR TURN; no narration; if the card is wrong about reality, fix the blocker, update the
card file, continue.
===== CARD =====
$(cat "$KIT/cards/$CARD")
===== STATE =====
state.json: ${STATE:-<none>}
progress.log tail: ${PROG:-<empty>}
commands.log tail (REUSE THESE):
${CMDS:-<empty>}"

  echo "[driver] round $round -> $CARD (last=$LAST) $(date)" >> "$LOG"
  before=$( [ -n "$RD" ] && wc -l < "$RD/progress.log" 2>/dev/null || echo 0 )
  timeout 2400 claude -p "$PROMPT" --model "$MODEL" --permission-mode auto >> "$LOG" 2>&1
  after=$( [ -n "$RD" ] && wc -l < "$RD/progress.log" 2>/dev/null || echo 0 )
  RD2=$(find "$RESULTS" -maxdepth 1 -type d -name "$RUN_GLOB" -newer "$MARKER" 2>/dev/null | head -1)
  if [ "$after" -eq "$before" ] && ! working && [ "$RD2" = "$RD" ]; then
    noop=$((noop+1)); echo "[driver] no progress ($noop)" >> "$LOG"
    [ "$noop" -ge 4 ] && { echo "[driver] ABORT: 4 no-progress rounds" >> "$LOG"; exit 1; }
  else noop=0; fi
done
echo "[driver] round cap reached $(date)" >> "$LOG"
exit 1
