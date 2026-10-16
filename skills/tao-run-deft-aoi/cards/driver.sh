#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# DEFT AOI card-pack driver (Pi harness).
#
# Runs the DEFT AOI loop as a series of FRESH headless agent sessions — one
# stage card per session — instead of one long conversation. State lives on
# disk (loop_log.jsonl via commit_stage.py), never in chat history.
# See skills/core/tao-token-efficient-execution/SKILL.md for the framework.
#
# Config comes from the environment or ~/.tao-kit/kit.env (see the kit's
# templates/kit.env.template). Required: WS. Common overrides: MODEL,
# TRAIN_IMG, DS_IMG, PI_KIT_TURN_BUDGET.
# ============================================================================
set -u
KIT_ENV=${KIT_ENV:-$HOME/.tao-kit/kit.env}
[ -f "$KIT_ENV" ] && . "$KIT_ENV"
[ -d "$HOME/.local/share/pi-node/current/bin" ] && export PATH="$HOME/.local/share/pi-node/current/bin:$PATH"

PACK=$(cd "$(dirname "$0")" && pwd)                 # this cards/ directory
BANK=$(cd "$PACK/../../../.." && pwd)               # skill-bank root
ADAPTER=$BANK/skills/core/tao-token-efficient-execution/adapters/pi
CARDS=${CARDS:-$PACK}
RUN_HOME=${RUN_HOME:-$HOME/.tao-kit/deft-aoi}       # sessions + logs live here, never in the bank checkout
SESSION_DIR=${SESSION_DIR:-$RUN_HOME/sessions}
LOG=$RUN_HOME/driver.log
MARKER=$RUN_HOME/.launch_marker

WS=${WS:?export WS (DEFT workspace root) or set it in ~/.tao-kit/kit.env}
case "$WS" in *[[:space:]]*) echo "[driver] ABORT: WS must not contain whitespace (docker mount flags are word-split): $WS" >&2; exit 1 ;; esac
RESULTS=$WS/results
SKILL_ROOT=${SKILL_ROOT:-$(cd "$PACK/.." && pwd)}   # the tao-run-deft-aoi skill this pack ships with
DPY=$SKILL_ROOT/scripts/deft_python.sh
TRAIN_IMG=${TRAIN_IMG:-nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt}
DS_IMG=${DS_IMG:-nvcr.io/nvidian/iva/tao-toolkit-ds:aoi}
# Workspace-layout mounts: authored for the NV_PCB_Siamese layout the cards
# were compiled against. Re-author the pack (see the kit's authoring prompt)
# for a different dataset layout.
MOUNTS_T=${MOUNTS_T:-"-v $WS:/data/workspace -v \$RD:/results -v $WS/kpi/images:/data/datasets/NV_PCB_Siamese/images -v $WS/train/base:/data/datasets/NV_PCB_Siamese/csv -v $WS/kpi:/data/datasets/NV_PCB_Siamese/kpi -v $WS/augmentation/backbone/model.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors"}
SYSPROMPT="You are a precise task executor operating in a bash environment on a GPU workstation. You MUST perform every action by calling your tools (bash, read, edit, write) — never describe, simulate, or invent a result or command output. Follow the stage card exactly; work alone; never ask questions; end your turn the moment the card says to."

MODEL=${MODEL:-nim/nvidia/qwen/qwen3.6-35b-a3b:off}
export WS TRAIN_IMG DS_IMG SKILL_ROOT DPY
export PI_KIT_WS="$WS"
# 120 clears every legitimately-completed session measured in the study
# (heaviest DEFT stage: 117 calls) while still killing 300-700-call wedges.
export PI_KIT_TURN_BUDGET=${PI_KIT_TURN_BUDGET:-120}

PI_FLAGS=(-p -na --system-prompt "$SYSPROMPT"
  --tools read,bash,edit,write --no-context-files --no-skills --no-extensions
  -e "$ADAPTER/nvidia-provider.ts" -e "$ADAPTER/guard.ts" -e "$ADAPTER/recorder.ts"
  --session-dir "$SESSION_DIR")

case "$MODEL" in
  nim/*) [ -n "${NVIDIA_INFERENCE_API_KEY:-}" ] || { echo "[driver] ABORT: export NVIDIA_INFERENCE_API_KEY" >&2; exit 1; } ;;
  anthropic/*) [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "[driver] ABORT: export ANTHROPIC_API_KEY" >&2; exit 1; } ;;
esac
mkdir -p "$SESSION_DIR"
[ -f "$MARKER" ] || touch "$MARKER"
cd "$RUN_HOME"
echo "[driver] start (model=$MODEL cards=$CARDS) $(date)" >> "$LOG"

# ---- 2. WORK PREDICATE (activity-based; idle zombie containers don't count) -
working() {
  local id img cpu
  for id in $(docker ps -q 2>/dev/null); do
    img=$(docker inspect --format '{{.Config.Image}}' "$id" 2>/dev/null)
    case "$img" in *tao-toolkit*) ;; *) continue ;; esac
    [ "$(docker ps -q --filter "id=$id" --filter "status=running" | wc -l)" -eq 1 ] || continue
    # young containers count as working even before load ramps
    if [ $(( $(date +%s) - $(date -d "$(docker inspect --format '{{.State.StartedAt}}' "$id")" +%s) )) -lt 180 ]; then return 0; fi
    cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' "$id" 2>/dev/null | tr -d '%' | cut -d. -f1)
    [ "${cpu:-0}" -ge 2 ] && return 0
  done
  return 1
}

finish_if_terminal() {
  LAST=""; ILAB=""
  if [ -n "$RD" ] && [ -f "$RD/loop_log.jsonl" ]; then
    LAST=$(jq -rRs 'split("\n") | map(select(length>0) | (fromjson? // empty)) | map(select(.status=="ok")) | last | .stage // empty' "$RD/loop_log.jsonl" 2>/dev/null)
    ILAB=$(jq -rRs 'split("\n") | map(select(length>0) | (fromjson? // empty)) | map(select(.status=="ok")) | last | .iter // empty' "$RD/loop_log.jsonl" 2>/dev/null)
  fi
  # No-auto-retry contract: a committed error halts the loop (operator decision).
  ERRLAST=""
  [ -n "$RD" ] && [ -f "$RD/loop_log.jsonl" ] && ERRLAST=$(jq -rRs 'split("\n") | map(select(length>0) | (fromjson? // empty)) | last | select(.status=="error") | "\(.iter)/\(.stage)"' "$RD/loop_log.jsonl" 2>/dev/null)
  [ -n "$ERRLAST" ] && { echo "[driver] HALT: committed error at $ERRLAST — no auto-retry (operator must decide) $(date)" >> "$LOG"; exit 2; }

  # loop_stop commits the training result, not the inference handoff. Finalize
  # deterministically here, including after an interrupted terminal card.
  if [ "$LAST" = "loop_stop" ]; then
    if ! "$DPY" "$SKILL_ROOT/scripts/prepare_inference_spec.py" --results-dir "$RD" >> "$LOG" 2>&1 ||
       ! "$DPY" "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RD" --require-complete >> "$LOG" 2>&1; then
      echo "[driver] HALT: finalization failed; run is NOT complete (operator must decide) $(date)" >> "$LOG"
      exit 2
    fi
    touch "$MARKER"
    echo "[driver] handoff prepared and completion audited - DONE $(date)" >> "$LOG"
    exit 0
  fi

  return 0
}

noop=0
for round in $(seq 1 80); do
  while working; do sleep 30; done
  sleep 12; working && continue

  # A vanished marker would make every find below return nothing and silently
  # fork a fresh run (and a fresh baseline training) each round. Refuse instead.
  [ -f "$MARKER" ] || { echo "[driver] ABORT: $MARKER vanished mid-run — refusing to fork a new run dir" >> "$LOG"; exit 1; }
  RD=$(find "$RESULTS" -maxdepth 1 -type d -name 'run_*' -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  finish_if_terminal

  next_iter() { # baseline -> iter1, iterN -> iterN+1
    if [ "$1" = "baseline" ]; then echo iter1; else echo "iter$(( ${1#iter} + 1 ))"; fi
  }

  # ---- 1. STAGE ROUTING (the skill's committed stage sequence) --------------
  if   [ -z "$RD" ]; then CARD=00-init-baseline-train.md; ITER=baseline
  elif [ -z "$LAST" ]; then
       if [ -s "$RD/baseline/train/train.log" ]; then CARD=10-post-train.md; ITER=baseline
       else CARD=00-init-baseline-train.md; ITER=baseline; fi
  elif [ "$LAST" = "train" ];       then CARD=20-evaluate.md;      ITER=$ILAB
  elif [ "$LAST" = "evaluate" ];    then CARD=30-post-evaluate.md; ITER=$ILAB
  elif [ "$LAST" = "rca" ];         then CARD=40-routing.md;       ITER=$(next_iter "$ILAB")
  elif [ "$LAST" = "routing" ];     then CARD=40-routing.md;       ITER=$ILAB   # anomalygen commit pending
  elif [ "$LAST" = "anomalygen" ];  then CARD=50-mining.md;        ITER=$ILAB
  elif [ "$LAST" = "data_mining" ]; then CARD=60-merge-train.md;   ITER=$ILAB
  elif [ "$LAST" = "data_merge" ];  then CARD=10-post-train.md;    ITER=$ILAB
  else CARD=30-post-evaluate.md; ITER=$ILAB; fi   # unknown stage: let the audit direct

  export ITER
  if [ -n "$RD" ]; then export RD; export MOUNTS="${MOUNTS_T//\$RD/$RD}"; export PI_KIT_RD="$RD"; else MOUNTS="$MOUNTS_T"; export PI_KIT_RD=""; fi
  [ -z "$RD" ] && { RD_NEW="$RESULTS/run_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RD_NEW"; export RD="$RD_NEW"; export MOUNTS="${MOUNTS_T//\$RD/$RD}"; export PI_KIT_RD="$RD"; }

  CMDS=""; [ -f "$RD/commands.log" ] && CMDS=$(tail -20 "$RD/commands.log" 2>/dev/null | head -c 2500)

  PROMPT="Execute exactly ONE stage card of a DEFT AOI loop, then end your turn.
Constants (all exported as environment variables in your bash shell — use them verbatim):
WS=$WS  RD=$RD  ITER=$ITER
TRAIN_IMG=$TRAIN_IMG  DS_IMG=$DS_IMG
SKILL_ROOT=$SKILL_ROOT
DPY=$DPY   (run ALL bundled scripts through \$DPY)
MOUNTS=\"$MOUNTS\"
Rules: follow the card exactly; one step = one command, run it verbatim; never retype or
expand commands; variables you set inside one bash call do not survive to the next call;
all state/log writes go through commit_stage.py ONLY; after any detached launch or when the
card says so, print the exact STAGE_DONE token and STOP — no extra commands, no summaries.
You have a hard budget of $PI_KIT_TURN_BUDGET tool calls this session.
===== CARD =====
$(cat "$CARDS/$CARD")
===== RECENT COMMANDS (reuse, don't re-derive) =====
${CMDS:-<none>}"

  echo "[driver] round $round -> $CARD ($ITER) $(date)" >> "$LOG"
  export STAGE_T0=$(date +%s)   # cards pass a session-relative --duration-sec from this
  before=$( [ -f "$RD/loop_log.jsonl" ] && wc -l < "$RD/loop_log.jsonl" || echo 0 )
  timeout 2400 pi "${PI_FLAGS[@]}" --model "$MODEL" "$PROMPT" >> "$LOG" 2>&1
  after=$( [ -f "$RD/loop_log.jsonl" ] && wc -l < "$RD/loop_log.jsonl" || echo 0 )
  # Grace before counting no-progress: a card's detached docker launch can take
  # a few seconds to appear in docker ps after the session exits (observed live).
  if [ "$after" -eq "$before" ] && ! working; then sleep 20; fi
  if [ "$after" -eq "$before" ] && ! working; then
    noop=$((noop+1)); echo "[driver] no progress ($noop consecutive)" >> "$LOG"
    [ $noop -ge 5 ] && { echo "[driver] ABORT: 5 no-progress rounds" >> "$LOG"; exit 1; }
  else noop=0; fi
done
finish_if_terminal
echo "[driver] hit 80-round cap $(date)" >> "$LOG"
exit 1
