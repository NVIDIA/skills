#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# AutoML card-pack driver (Pi harness).
#
# Runs the tao-run-automl workflow as a series of FRESH headless agent
# sessions — one stage card per session. Progress marks live in the run dir's
# progress.log; the driver owns run-dir selection (no cross-run pointer file).
# See skills/core/tao-token-efficient-execution/SKILL.md for the framework.
#
# Config comes from the environment or ~/.tao-kit/kit.env. Required: WS, VENV.
# Common overrides: MODEL, TRAIN_IMG, PI_KIT_TURN_BUDGET.
# ============================================================================
set -u
KIT_ENV=${KIT_ENV:-$HOME/.tao-kit/kit.env}
[ -f "$KIT_ENV" ] && . "$KIT_ENV"
[ -d "$HOME/.local/share/pi-node/current/bin" ] && export PATH="$HOME/.local/share/pi-node/current/bin:$PATH"

PACK=$(cd "$(dirname "$0")" && pwd)                 # this cards/ directory
BANK=$(cd "$PACK/../../../.." && pwd)               # skill-bank root
ADAPTER=$BANK/skills/core/tao-token-efficient-execution/adapters/pi
CARDS=${CARDS:-$PACK}
RUN_HOME=${RUN_HOME:-$HOME/.tao-kit/automl}         # sessions + logs live here, never in the bank checkout
SESSION_DIR=${SESSION_DIR:-$RUN_HOME/sessions}
LOG=$RUN_HOME/driver.log
MARKER=$RUN_HOME/.launch_marker

WS=${WS:?export WS (workspace root) or set it in ~/.tao-kit/kit.env}
case "$WS" in *[[:space:]]*) echo "[a-driver] ABORT: WS must not contain whitespace (docker mount flags are word-split): $WS" >&2; exit 1 ;; esac
RESULTS=$WS/results
SB=${SB:-$BANK}                                     # skill bank the cards read references from
VENV=${VENV:?export VENV (venv with the nvidia-tao-automl wheel; see tao-run-automl preflight) or set it in ~/.tao-kit/kit.env}
# Resolve from the same bank that supplies the recommendation spec template.
if [ -z "${TRAIN_IMG:-}" ]; then
  TRAIN_IMG=$("$VENV/bin/python" "$SB/scripts/resolve_versions_key.py" images.tao_toolkit.pyt --skill-bank "$SB") || exit 1
fi
SYSPROMPT="You are a precise task executor operating in a bash environment on a GPU workstation. You MUST perform every action by calling your tools (bash, read, edit, write) — never describe, simulate, or invent a result or command output. Follow the stage card exactly; work alone; never ask questions; end your turn the moment the card says to."

MODEL=${MODEL:-nim/nvidia/qwen/qwen3.6-35b-a3b:off}
export WS SB VENV TRAIN_IMG
export PI_KIT_WS="$WS"
export PI_KIT_RUN_PREFIX="automl2_"
# 120 clears every legitimately-completed session measured in the study
# while still killing 300-700-call wedges.
export PI_KIT_TURN_BUDGET=${PI_KIT_TURN_BUDGET:-120}

PI_FLAGS=(-p -na --system-prompt "$SYSPROMPT"
  --tools read,bash,edit,write --no-context-files --no-skills --no-extensions
  -e "$ADAPTER/nvidia-provider.ts" -e "$ADAPTER/guard.ts" -e "$ADAPTER/recorder.ts"
  --session-dir "$SESSION_DIR")

case "$MODEL" in
  nim/*) [ -n "${NVIDIA_INFERENCE_API_KEY:-}" ] || { echo "[a-driver] ABORT: export NVIDIA_INFERENCE_API_KEY" >&2; exit 1; } ;;
  anthropic/*) [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "[a-driver] ABORT: export ANTHROPIC_API_KEY" >&2; exit 1; } ;;
esac
mkdir -p "$SESSION_DIR"
[ -f "$MARKER" ] || touch "$MARKER"
cd "$RUN_HOME"
echo "[a-driver] start (model=$MODEL) $(date)" >> "$LOG"

working() {
  pgrep -f "$RESULTS/.*runner/driver.py" >/dev/null 2>&1 && return 0
  # Baseline eval gate: PID file written by card 10 (never pgrep on generic
  # command text — an unrelated process containing the name would park us).
  [ -n "${RD:-}" ] && [ -f "$RD/baseline/job.pid" ] && kill -0 "$(cat "$RD/baseline/job.pid" 2>/dev/null)" 2>/dev/null && return 0
  local id img cpu
  for id in $(docker ps -q 2>/dev/null); do
    img=$(docker inspect --format '{{.Config.Image}}' "$id" 2>/dev/null)
    case "$img" in *tao-toolkit*) ;; *) continue ;; esac
    if [ $(( $(date +%s) - $(date -d "$(docker inspect --format '{{.State.StartedAt}}' "$id")" +%s) )) -lt 180 ]; then return 0; fi
    cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' "$id" 2>/dev/null | tr -d '%' | cut -d. -f1)
    [ "${cpu:-0}" -ge 2 ] && return 0
  done
  return 1
}

# Driver-owned run dir (no cross-run pointer file)
[ -f "$MARKER" ] || { echo "[a-driver] ABORT: $MARKER missing — refusing to select/fork a run dir" >> "$LOG"; exit 1; }
RD=$(find "$RESULTS" -maxdepth 1 -type d -name 'automl2_*' -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$RD" ] && { RD="$RESULTS/automl2_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RD"; touch "$RD/progress.log"; }
export RD AUTOML_RD="$RD" PI_KIT_RD="$RD"
echo "[a-driver] RD=$RD" >> "$LOG"

finish_if_terminal() {
  # On DONE, refresh the marker so the NEXT driver launch starts a fresh run
  # instead of re-selecting this completed one and exiting immediately.
  # Halt only when the LATEST mark is a FAIL (a recovered run appends new ok lines after a stale FAIL).
  tail -1 "$RD/progress.log" 2>/dev/null | grep -q " FAIL$" && { echo "[a-driver] HALT: latest mark is FAIL — no auto-retry $(date)" >> "$LOG"; exit 2; }
  [ -f "$RD/AUTOML_DONE.marker" ] && { touch "$MARKER"; echo "[a-driver] DONE after $((round-1)) sessions $(date)" >> "$LOG"; exit 0; }

  return 0
}

noop=0
for round in $(seq 1 40); do
  while working; do sleep 30; done
  sleep 12; working && continue

  finish_if_terminal

  LAST=$(grep ' ok$' "$RD/progress.log" 2>/dev/null | tail -1 | awk '{print $1}')
  case "$LAST" in
    "")             CARD=00-preflight.md ;;
    preflight)      CARD=10-baseline-eval.md ;;
    baseline_eval)  CARD=20-launch-recs.md ;;
    recs_launched|runner_finished) CARD=30-interpret.md ;;
    done)           CARD=30-interpret.md ;;
    *)              CARD=30-interpret.md ;;
  esac

  CMDS=""; [ -f "$RD/commands.log" ] && CMDS=$(tail -20 "$RD/commands.log" 2>/dev/null | head -c 2500)

  PROMPT="Execute exactly ONE stage card of a TAO AutoML workflow, then end your turn.
Constants (all exported as environment variables in your bash shell — use them verbatim):
WS=$WS  RD=$RD  SB=$SB  VENV=$VENV  TRAIN_IMG=$TRAIN_IMG
Rules: follow the card exactly; one step = one command, run it verbatim; never retype or
expand commands; variables you set inside one bash call do not survive to the next call;
progress marks are appended ONLY by the exact card commands; after any detached launch or
when the card says so, print the exact STAGE_DONE token and STOP — no extra commands.
You have a hard budget of $PI_KIT_TURN_BUDGET tool calls this session.
===== CARD =====
$(cat "$CARDS/$CARD")
===== RECENT COMMANDS (reuse, don't re-derive) =====
${CMDS:-<none>}"

  echo "[a-driver] round $round -> $CARD (last=$LAST) $(date)" >> "$LOG"
  before=$( [ -f "$RD/progress.log" ] && wc -l < "$RD/progress.log" || echo 0 )
  touch "$RUN_HOME/.round_marker"
  timeout 2400 pi "${PI_FLAGS[@]}" --model "$MODEL" "$PROMPT" >> "$LOG" 2>&1
  after=$( [ -f "$RD/progress.log" ] && wc -l < "$RD/progress.log" || echo 0 )
  # Progress = new log mark OR active work OR any new JOB OUTPUT under RD
  # (fast jobs can start AND finish inside one session window). Exclude the
  # recorder's commands.log: it is harness telemetry, appended even on stuck
  # rounds, and counting it would defeat the 5-round no-progress abort.
  NEWF=$(find "$RD" -newer "$RUN_HOME/.round_marker" -type f ! -name 'commands.log' 2>/dev/null | head -1)
  # Grace before counting no-progress: a detached launch can take a few seconds
  # to appear in docker ps after the session exits.
  if [ "$after" -eq "$before" ] && ! working && [ -z "$NEWF" ]; then
    sleep 20
    NEWF=$(find "$RD" -newer "$RUN_HOME/.round_marker" -type f ! -name 'commands.log' 2>/dev/null | head -1)
  fi
  if [ "$after" -eq "$before" ] && ! working && [ -z "$NEWF" ]; then
    noop=$((noop+1)); echo "[a-driver] no progress ($noop)" >> "$LOG"
    [ $noop -ge 5 ] && { echo "[a-driver] ABORT: 5 no-progress rounds" >> "$LOG"; exit 1; }
  else noop=0; fi
done
finish_if_terminal
echo "[a-driver] 40-round cap $(date)" >> "$LOG"
exit 1
