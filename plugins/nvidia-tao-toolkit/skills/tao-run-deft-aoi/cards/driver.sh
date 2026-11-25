#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# DEFT AOI card-pack driver (Pi harness).
#
# Runs the DEFT AOI loop as a series of FRESH headless agent sessions — one
# stage card per session — instead of one long conversation. State lives on
# disk (deft_state.json, written only by commit_stage.py / finalize_run.py),
# never in chat history; routing follows scripts/deft_context.py.
# See skills/tao-token-efficient-execution/SKILL.md for the framework.
#
# Config comes from the environment or ~/.tao-kit/kit.env (see the kit's
# templates/kit.env.template). Required, no default: WS, DS_IMG, MODEL.
# Common overrides: TRAIN_IMG, PI_KIT_TURN_BUDGET.
# ============================================================================
set -u
KIT_ENV=${KIT_ENV:-$HOME/.tao-kit/kit.env}
[ -f "$KIT_ENV" ] && . "$KIT_ENV"
[ -d "$HOME/.local/share/pi-node/current/bin" ] && export PATH="$HOME/.local/share/pi-node/current/bin:$PATH"
command -v pi >/dev/null 2>&1 || { echo "[driver] ABORT: pi not on PATH (this pack runs on the Pi harness; run the kit's install.sh)" >&2; exit 1; }

PACK=$(cd "$(dirname "$0")" && pwd)                 # this cards/ directory
BANK=$(cd "$PACK/../../.." && pwd)                  # skill-bank root
ADAPTER=$BANK/skills/tao-token-efficient-execution/adapters/pi
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
# The pack ships no data-services image: cards 30 and 50 are compiled for one
# command form, so the user names an image that provides it.
[ -n "${DS_IMG:-}" ] || { echo "[driver] ABORT: set DS_IMG (export it or set it in ~/.tao-kit/kit.env) to a TAO data-services image you can pull that provides 'gap_analysis vcn_aoi' with key=value overrides, 'embedding image_embeddings' and 'tmm nearest_neighbors'; this pack has no default image" >&2; exit 1; }
# C-RADIOv2-B backbone on the host; default is what scripts/stage_backbone.py writes.
if [ -z "${BACKBONE:-}" ]; then
  BACKBONE=$WS/augmentation/backbone/c_radio_v2_b.safetensors
  [ ! -s "$BACKBONE" ] && [ -s "$WS/augmentation/backbone/model.safetensors" ] && BACKBONE=$WS/augmentation/backbone/model.safetensors
fi
case "$BACKBONE" in *[[:space:]]*) echo "[driver] ABORT: BACKBONE must not contain whitespace: $BACKBONE" >&2; exit 1 ;; esac
[ -s "$BACKBONE" ] || { echo "[driver] ABORT: backbone not staged at $BACKBONE; run: $DPY $SKILL_ROOT/scripts/stage_backbone.py --workspace $WS (or set BACKBONE)" >&2; exit 1; }
# Workspace-layout mounts: authored for the NV_PCB_Siamese layout the cards
# were compiled against. Re-author the pack (see the kit's authoring prompt)
# for a different dataset layout.
MOUNTS_T=${MOUNTS_T:-"-v $WS:/data/workspace -v \$RD:/results -v $WS/kpi/images:/data/datasets/NV_PCB_Siamese/images -v $WS/train/base:/data/datasets/NV_PCB_Siamese/csv -v $WS/kpi:/data/datasets/NV_PCB_Siamese/kpi -v $BACKBONE:/data/pretrained_models/C-RADIOv2_B.safetensors:ro"}
SYSPROMPT="You are a precise task executor operating in a bash environment on a GPU workstation. You MUST perform every action by calling your tools (bash, read, edit, write) — never describe, simulate, or invent a result or command output. Follow the stage card exactly; work alone; never ask questions; end your turn the moment the card says to."

# Exact accelerator model for init_deft_state.py --gpu-model (references/preflight.md step 8).
GPU_MODEL=${GPU_MODEL:-$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)}
[ -n "$GPU_MODEL" ] || { echo "[driver] ABORT: cannot read the GPU model via nvidia-smi; set GPU_MODEL (e.g. \"NVIDIA RTX PRO 6000 Blackwell, 97887 MiB\")" >&2; exit 1; }

MODEL=${MODEL:-}                                    # required, no default; checked by kit_check_model_key
export WS TRAIN_IMG DS_IMG SKILL_ROOT DPY GPU_MODEL BACKBONE
export PI_KIT_WS="$WS"
# 120 clears every legitimately-completed session measured in the study
# (heaviest DEFT stage: 117 calls) while still killing 300-700-call wedges.
export PI_KIT_TURN_BUDGET=${PI_KIT_TURN_BUDGET:-120}

PI_FLAGS=(-p -na --system-prompt "$SYSPROMPT"
  --tools read,bash,edit,write --no-context-files --no-skills --no-extensions
  -e "$ADAPTER/nvidia-provider.ts" -e "$ADAPTER/guard.ts" -e "$ADAPTER/recorder.ts"
  --session-dir "$SESSION_DIR")

. "$BANK/skills/tao-token-efficient-execution/scripts/model_preflight.sh"
kit_prepare_model_env
kit_check_model_key "[driver]" || exit 1
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

state_file() { echo "$RD/deft_state.json"; }
event_count() { [ -n "${RD:-}" ] && [ -f "$(state_file)" ] && jq '.events // [] | length' "$(state_file)" 2>/dev/null || echo 0; }
state_status() { jq -r '.status // empty' "$(state_file)" 2>/dev/null; }

# Reads the durable next stage from deft_state.json via the skill's own
# deft_context.py; sets NEXT ("" before state exists) and ILAB.
finish_if_terminal() {
  NEXT=""; ILAB=""
  [ -n "$RD" ] && [ -f "$(state_file)" ] || return 0
  local ctx
  if ! ctx=$("$DPY" "$SKILL_ROOT/scripts/deft_context.py" --state "$(state_file)" 2>> "$LOG"); then
    echo "[driver] HALT: deft_context.py cannot read $(state_file) (operator must decide) $(date)" >> "$LOG"; exit 2
  fi
  NEXT=$(printf '%s' "$ctx" | jq -r '.next_stage // empty')
  ILAB=$(printf '%s' "$ctx" | jq -r '.iteration // empty')

  # No-auto-retry contract: a committed error halts the loop (operator decision).
  if [ "$NEXT" = "halt" ]; then
    local err
    err=$(jq -r '[.events // [] | .[] | select(.status=="error")] | last | if . then "\(.iter)/\(.stage): \(.summary)" else "status=failed" end' "$(state_file)" 2>/dev/null)
    echo "[driver] HALT: committed error at $err — no auto-retry (operator must decide) $(date)" >> "$LOG"; exit 2
  fi

  # Terminal evaluate is deterministic: finalize_run.py prepares the inference
  # handoff and commits loop_stop with the stop reason the metric implies.
  if [ "$NEXT" = "finalize" ]; then
    local reason t0
    reason=$(jq -r --arg it "$ILAB" 'if .iterations[$it].metric_result.passed == true then "metric_met" else "max_iterations" end' "$(state_file)")
    t0=$(date +%s)
    "$DPY" "$SKILL_ROOT/scripts/finalize_run.py" --results-dir "$RD" --iter-label "$ILAB" \
      --stop-reason "$reason" --duration-sec $(( $(date +%s) - t0 + 1 )) >> "$LOG" 2>&1
  fi

  if [ "$NEXT" = "complete" ] || [ "$NEXT" = "finalize" ]; then
    if [ "$(state_status)" != "complete" ]; then
      echo "[driver] HALT: finalization failed; deft_state.json status is not complete (operator must decide) $(date)" >> "$LOG"
      exit 2
    fi
    touch "$MARKER"
    echo "[driver] deft_state.json status=complete (handoff prepared by finalize_run.py) - DONE $(date)" >> "$LOG"
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

  # ---- 1. STAGE ROUTING (deft_context.py's durable next_stage) ---------------
  ITER=${ILAB:-baseline}
  case "$NEXT" in
    "")          CARD=00-init-baseline-train.md ;;   # no run dir or no deft_state.json yet
    train)       if [ "$ITER" = baseline ] && [ ! -s "$RD/baseline/train/train.log" ]; then CARD=00-init-baseline-train.md
                 else CARD=10-post-train.md; fi ;;
    evaluate)    CARD=20-evaluate.md ;;
    rca)         CARD=30-post-evaluate.md ;;
    routing|anomalygen) CARD=40-routing.md ;;
    data_mining) CARD=50-mining.md ;;
    data_merge)  CARD=60-merge-train.md ;;
    *) echo "[driver] HALT: no card for next_stage=$NEXT ($ITER) (operator must decide) $(date)" >> "$LOG"; exit 2 ;;
  esac

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
  before=$(event_count); [ -f "$(state_file)" ] || before=-1
  timeout 2400 pi "${PI_FLAGS[@]}" --model "$MODEL" "$PROMPT" >> "$LOG" 2>&1
  after=$(event_count); [ -f "$(state_file)" ] || after=-1
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
