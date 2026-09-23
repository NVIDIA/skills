#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for the DEFT OD state machine: init_deft_state.py,
# commit_stage.py, and audit_deft_run.py driven together over a whole run.
#
# Why this exists: the three scripts share one contract (scripts/deft_stages.py)
# but each enforces a different slice of it — init freezes config, commit gates
# ordering and artifacts, audit re-derives the truth from disk. Each one passes
# its own unit reasoning; only running them against each other proves they agree
# on phase labels, stage order, artifact flags, exit codes, and the next action
# an agent is told to take. The loop's entire memory is those two files, so a
# disagreement between the writer and the auditor is unrecoverable in production:
# a resumed run re-does a stage, skips one, or claims a completion that never
# happened.
#
# Every commit is followed by an assertion on the audit's next_action and
# read_before_action, because that pair is the only instruction the orchestrating
# agent acts on. read_before_action is also checked to name a file that exists.
#
# Negative cases carry equal weight: a rejected commit must leave deft_state.json
# and loop_log.jsonl byte-for-byte unchanged, since a half-applied rejection is
# worse than no rejection at all. Rejections are proven with a checksum of both
# files taken before and after.
#
# Artifacts are fixtures, not real outputs — the scripts only check that a path
# exists and is a file or a directory, never its contents. No TAO, no GPU, no
# network, stdlib Python only.
#
# Usage:
#   bash scripts/tests/test_state_machine.sh
#   DEFT_TEST_PYTHON=/usr/bin/python3.12 bash scripts/tests/test_state_machine.sh
#   DEFT_TEST_KEEP=1 bash scripts/tests/test_state_machine.sh   # keep the temp run
#
# Exit codes: 0 all assertions passed; 1 at least one failed.

set -u
set -o pipefail

TESTS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCRIPTS_DIR=$(CDPATH= cd -- "$TESTS_DIR/.." && pwd)
SKILL_DIR=$(CDPATH= cd -- "$SCRIPTS_DIR/.." && pwd)

# Deliberately not deft_python.sh: these three scripts are stdlib-only by
# contract, and deft_python.sh selects an interpreter that also has pandas,
# pyarrow, and PIL. Requiring those here would make the test unrunnable on a
# machine where the contract it is testing holds perfectly well.
PY=${DEFT_TEST_PYTHON:-python3}

INIT="$SCRIPTS_DIR/init_deft_state.py"
COMMIT="$SCRIPTS_DIR/commit_stage.py"
AUDIT="$SCRIPTS_DIR/audit_deft_run.py"

for script in "$INIT" "$COMMIT" "$AUDIT" "$SCRIPTS_DIR/deft_stages.py"; do
  if [ ! -f "$script" ]; then
    echo "FATAL: missing $script" >&2
    exit 1
  fi
done
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "FATAL: python interpreter not found: $PY" >&2
  exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/deft_state_machine.XXXXXX")
cleanup() {
  if [ "${DEFT_TEST_KEEP:-0}" = "1" ]; then
    echo "kept: $WORK"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

TOTAL=0
FAILURES=0
CURRENT_SECTION=""

section() {
  CURRENT_SECTION=$1
  printf '\n== %s\n' "$1"
}

ok() {
  TOTAL=$((TOTAL + 1))
  printf 'ok %d - %s\n' "$TOTAL" "$1"
}

notok() {
  TOTAL=$((TOTAL + 1))
  FAILURES=$((FAILURES + 1))
  printf 'not ok %d - %s\n' "$TOTAL" "$1"
  if [ "$#" -gt 1 ]; then
    shift
    printf '  # %s\n' "$@"
  fi
}

assert_eq() {  # assert_eq EXPECTED ACTUAL LABEL
  if [ "$1" = "$2" ]; then
    ok "$3"
  else
    notok "$3" "expected: $1" "actual:   $2" "section:  $CURRENT_SECTION"
  fi
}

# ── running the scripts ──────────────────────────────────────────────────────

RUN_RC=0
RUN_OUT=""

run() {  # run CMD...  -> RUN_RC, RUN_OUT (stdout+stderr merged)
  RUN_OUT=$("$@" 2>&1)
  RUN_RC=$?
  return 0
}

assert_rc() {  # assert_rc EXPECTED_RC LABEL
  if [ "$RUN_RC" = "$1" ]; then
    ok "$2"
  else
    notok "$2" "expected exit $1, got $RUN_RC" "output: ${RUN_OUT:-<empty>}"
  fi
}

commit() {  # commit RESULTS_DIR PHASE STAGE [extra args...]
  local results=$1 phase=$2 stage=$3
  shift 3
  run "$PY" "$COMMIT" --results-dir "$results" --iter-label "$phase" \
    --stage "$stage" "$@"
}

# ── reading the audit ────────────────────────────────────────────────────────

AUDIT_RC=0
AUDIT_STATUS=""
AUDIT_LAST=""
AUDIT_NEXT=""
AUDIT_READ=""
AUDIT_TERMINAL=""
AUDIT_COMPLETE=""

audit_kv() {  # audit_kv RESULTS_DIR -> AUDIT_* (key=value lines only; stderr dropped)
  local out key value
  out=$("$PY" "$AUDIT" --results-dir "$1" 2>/dev/null)
  AUDIT_RC=$?
  AUDIT_STATUS=""; AUDIT_LAST=""; AUDIT_NEXT=""
  AUDIT_READ=""; AUDIT_TERMINAL=""; AUDIT_COMPLETE=""
  while IFS='=' read -r key value; do
    case "$key" in
      DEFT_RUN_STATUS)    AUDIT_STATUS=$value ;;
      last_committed)     AUDIT_LAST=$value ;;
      next_action)        AUDIT_NEXT=$value ;;
      read_before_action) AUDIT_READ=$value ;;
      terminal)           AUDIT_TERMINAL=$value ;;
      complete)           AUDIT_COMPLETE=$value ;;
    esac
  done <<<"$out"
}

expect_audit() {  # expect_audit RESULTS_DIR LAST_COMMITTED NEXT_ACTION READ_BEFORE
  local results=$1 last=$2 next=$3 read_before=$4
  audit_kv "$results"
  assert_eq 0 "$AUDIT_RC" "audit exits 0 after $last"
  assert_eq VALID "$AUDIT_STATUS" "DEFT_RUN_STATUS after $last"
  assert_eq "$last" "$AUDIT_LAST" "last_committed after $last"
  assert_eq "$next" "$AUDIT_NEXT" "next_action after $last"
  assert_eq "$read_before" "$AUDIT_READ" "read_before_action after $last"
  if [ "$read_before" != "none" ] && [ ! -f "$SKILL_DIR/$read_before" ]; then
    notok "read_before_action names an existing overlay after $last" \
      "missing: $SKILL_DIR/$read_before"
  else
    ok "read_before_action names an existing overlay after $last"
  fi
}

# ── proving a rejection changed nothing ──────────────────────────────────────

fingerprint() {  # fingerprint RESULTS_DIR -> checksum of state + log
  cksum "$1/deft_state.json" "$1/loop_log.jsonl" 2>&1
}

FROZEN=""
freeze() { FROZEN=$(fingerprint "$1"); }

assert_no_tmp_files() {  # assert_no_tmp_files RESULTS_DIR LABEL
  local leftovers
  leftovers=$(find "$1" -maxdepth 1 -name '*.tmp' 2>/dev/null)
  if [ -z "$leftovers" ]; then
    ok "$2"
  else
    notok "$2" "leftover: $leftovers"
  fi
}

assert_no_audit_warnings() {  # assert_no_audit_warnings RESULTS_DIR LABEL
  # A healthy run must produce a silent audit. Every warning is a place where
  # the writer and the auditor already disagree about what is on disk.
  local noise
  noise=$("$PY" "$AUDIT" --results-dir "$1" 2>&1 >/dev/null)
  if [ -z "$noise" ]; then
    ok "$2"
  else
    notok "$2" "stderr: $noise"
  fi
}

assert_unchanged() {  # assert_unchanged RESULTS_DIR LABEL
  local now
  now=$(fingerprint "$1")
  if [ "$FROZEN" = "$now" ]; then
    ok "$2"
  else
    notok "$2" "before: $FROZEN" "after:  $now"
  fi
}

# ── fixtures ─────────────────────────────────────────────────────────────────

make_file() { mkdir -p "$(dirname -- "$1")" && printf '%s\n' "${2:-fixture}" >"$1"; }

# Artifacts are checked for structure, not just existence, so fixtures carry the
# minimum that makes them readable: parquet needs its magic bytes, a KPI csv needs
# a data row under the header.
make_parquet() { mkdir -p "$(dirname -- "$1")" && printf 'PAR1fixture-rowsPAR1' >"$1"; }

new_workspace() {  # new_workspace NAME -> prints the workspace root
  local ws="$WORK/$1"
  make_file "$ws/ckpt/gdino_zero_shot.pth" "not a real checkpoint"
  make_file "$ws/specs/train_grounding_dino.yaml" "dataset: {train_data_sources: []}"
  make_file "$ws/classes/classes_its.yaml" "car: [car]"
  make_file "$ws/encoder/siglip/config.json" '{"model_type": "siglip"}'
  make_file "$ws/kpi/sequence_a/images/000001.png" "png"
  make_file "$ws/kpi/labels/000001.txt" "car 0.0 0 0.0 10 10 100 100 0 0 0 0 0 0 0"
  mkdir -p "$ws/results"
  printf '%s\n' "$ws"
}

state_json() {  # state_json RESULTS_DIR KEY  -> value from config, JSON-encoded
  "$PY" -c 'import json,sys
c = json.load(open(sys.argv[1] + "/deft_state.json"))["config"]
print(json.dumps(c[sys.argv[2]], sort_keys=True))' "$1" "$2"
}

make_pool() {  # make_pool WORKSPACE  (what the prep stage would emit)
  make_file "$1/source_pool/odvg/pool_odvg.jsonl" '{"filename": "a.png"}'
  make_parquet "$1/source_pool/source_embeddings.parquet"
}

init_run() {  # init_run WORKSPACE RESULTS_DIR MAX_ITERATIONS [extra args...]
  local ws=$1 results=$2 max=$3
  shift 3
  # OMIT_AP50=1 drops the flag entirely so a test can exercise the reference
  # defaults. Every other caller keeps passing thresholds explicitly.
  local ap50=(--ap50-thresholds-json '{"car": 0.9, "person": 0.85}')
  [ "${OMIT_AP50:-0}" = 1 ] && ap50=()
  run "$PY" "$INIT" \
    --results-dir "$results" \
    --workspace "$ws" \
    --max-iterations "$max" \
    --num-epochs 10 \
    --learning-rate 0.0001 \
    --zero-shot-checkpoint "$ws/ckpt/gdino_zero_shot.pth" \
    --train-spec-template "$ws/specs/train_grounding_dino.yaml" \
    --source-pool-embeddings "$ws/source_pool/source_embeddings.parquet" \
    --source-pool-annotations "$ws/source_pool/odvg" \
    --embedding-model-path "$ws/encoder/siglip" \
    --kpi-images-dir "$ws/kpi/sequence_a/images" \
    --ground-truth-labels-dir "$ws/kpi/labels" \
    --class-mapping "$ws/classes/classes_its.yaml" \
    ${ap50[@]+"${ap50[@]}"} \
    "$@"
}

make_phase_artifacts() {  # make_phase_artifacts RESULTS_DIR PHASE
  local results=$1 phase=$2
  mkdir -p "$results/$phase/inference/labels"
  make_file "$results/$phase/inference/labels/000001.txt" \
    "car 0.0 0 0.0 10 10 100 100 0 0 0 0 0 0 0 0.91"
  make_file "$results/$phase/kpi/kpi_calc.csv" "Sequence Name,class,AP50"
  printf 'kpi,car,0.42\n' >>"$results/$phase/kpi/kpi_calc.csv"
  make_file "$results/$phase/kpi/kpi_analyze.log" "mAP: 0.42"
}

make_iter_artifacts() {  # make_iter_artifacts RESULTS_DIR ITER_LABEL
  local results=$1 phase=$2
  make_parquet "$results/$phase/gaps/weak_images.parquet"
  make_file "$results/$phase/gaps/gap_report.json" '{"weak_images": 120}'
  make_parquet "$results/$phase/embeddings/weak_images_embeddings.parquet"
  make_parquet "$results/$phase/mining/final_unique_files.parquet"
  make_file "$results/$phase/mining/summary.json" '{"retrieved": 360}'
  mkdir -p "$results/$phase/tmm/images"
  make_file "$results/$phase/tmm/images/000001.png" "png"
  make_file "$results/$phase/tmm/annotations/tmm_odvg.jsonl" '{"filename": "000001.png"}'
  make_file "$results/$phase/tmm/annotations/labelmap.json" '{"0": "car"}'
  make_parquet "$results/$phase/mined_cumulative.parquet"
  make_file "$results/$phase/train/gdino_model_latest.pth" "checkpoint"
  make_file "$results/$phase/train_grounding_dino.yaml" "train: {num_epochs: 10}"
  make_phase_artifacts "$results" "$phase"
}

# Commit the seven stages of one iteration, asserting the audit after each.
# TAIL_NEXT/TAIL_READ describe what must follow this iteration's kpi_analyze:
# another gap_analysis while iterations remain, loop_stop at max_iterations.
commit_iteration() {  # commit_iteration RESULTS_DIR ITER_LABEL TAIL_NEXT TAIL_READ
  local results=$1 phase=$2 tail_next=$3 tail_read=$4

  commit "$results" "$phase" gap_analysis \
    --weak-images "$results/$phase/gaps/weak_images.parquet" \
    --gap-report "$results/$phase/gaps/gap_report.json" \
    --weak-image-count 120 \
    --summary "gap_analysis: 120 weak images across 2 classes" --duration-sec 91
  assert_rc 0 "commit $phase/gap_analysis"
  expect_audit "$results" "$phase/gap_analysis" embed \
    references/tao-generate-image-embeddings.md

  commit "$results" "$phase" embed \
    --embeddings-parquet "$results/$phase/embeddings/weak_images_embeddings.parquet" \
    --summary "embedded 120 weak images with SigLIP" --duration-sec 45
  assert_rc 0 "commit $phase/embed"
  expect_audit "$results" "$phase/embed" mine references/tao-mine-od-images.md

  commit "$results" "$phase" mine \
    --mining-output "$results/$phase/mining/final_unique_files.parquet" \
    --mining-summary "$results/$phase/mining/summary.json" \
    --summary "mined 360/360 unique images (100% coverage)" --duration-sec 612
  assert_rc 0 "commit $phase/mine"
  expect_audit "$results" "$phase/mine" stage references/stage-mined-data.md

  commit "$results" "$phase" stage \
    --odvg "$results/$phase/tmm/annotations/tmm_odvg.jsonl" \
    --label-map "$results/$phase/tmm/annotations/labelmap.json" \
    --staged-images-dir "$results/$phase/tmm/images" \
    --exclude-parquet "$results/$phase/mined_cumulative.parquet" \
    --summary "staged 360 images with annotations" --duration-sec 73
  assert_rc 0 "commit $phase/stage"
  expect_audit "$results" "$phase/stage" train references/grounding-dino.md

  commit "$results" "$phase" train \
    --checkpoint "$results/$phase/train/gdino_model_latest.pth" \
    --training-spec "$results/$phase/train_grounding_dino.yaml" \
    --summary "trained $phase: 10 epochs, 2 data sources" --duration-sec 4820
  assert_rc 0 "commit $phase/train"
  expect_audit "$results" "$phase/train" inference references/grounding-dino.md

  commit "$results" "$phase" inference \
    --inference-labels-dir "$results/$phase/inference/labels" \
    --summary "inference: 1 label file" --duration-sec 140
  assert_rc 0 "commit $phase/inference"
  expect_audit "$results" "$phase/inference" kpi_analyze \
    references/tao-analyze-detection-kpi.md

  commit "$results" "$phase" kpi_analyze \
    --kpi-csv "$results/$phase/kpi/kpi_calc.csv" \
    --kpi-log "$results/$phase/kpi/kpi_analyze.log" \
    --map-value 0.55 \
    --summary "kpi: mAP=0.55" --duration-sec 62
  assert_rc 0 "commit $phase/kpi_analyze"
  expect_audit "$results" "$phase/kpi_analyze" "$tail_next" "$tail_read"
}

printf 'DEFT OD state machine integration test\n'
printf 'python:   %s (%s)\n' "$PY" "$("$PY" -V 2>&1)"
printf 'scripts:  %s\n' "$SCRIPTS_DIR"
printf 'workdir:  %s\n' "$WORK"

# ═══════════════════════════════════════════════════════════════════════════
# RUN A — the full loop: prep, baseline, iter1, iter2, loop_stop
# ═══════════════════════════════════════════════════════════════════════════

WS_A=$(new_workspace ws_main)
RUN_A="$WS_A/results/run_main"

section "A1. init_deft_state.py"

# The source pool is prepared by its own run before the loop launches, so it is
# already on disk by the time init sees it.
make_pool "$WS_A"
init_run "$WS_A" "$RUN_A" 2
assert_rc 0 "init_deft_state.py exits 0"
[ -f "$RUN_A/deft_state.json" ] && ok "deft_state.json created" \
  || notok "deft_state.json created"
[ -f "$RUN_A/loop_log.jsonl" ] && ok "loop_log.jsonl created" \
  || notok "loop_log.jsonl created"
expect_audit "$RUN_A" none inference references/grounding-dino.md

freeze "$RUN_A"
init_run "$WS_A" "$RUN_A" 2
assert_rc 1 "re-init without --force is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the rejected re-init"

section "A3. baseline (inference -> kpi_analyze, no training)"

make_phase_artifacts "$RUN_A" baseline
commit "$RUN_A" baseline inference \
  --inference-labels-dir "$RUN_A/baseline/inference/labels" \
  --summary "inference: 1 label file from the zero-shot checkpoint" --duration-sec 133
assert_rc 0 "commit baseline/inference"
expect_audit "$RUN_A" baseline/inference kpi_analyze \
  references/tao-analyze-detection-kpi.md

commit "$RUN_A" baseline kpi_analyze \
  --kpi-csv "$RUN_A/baseline/kpi/kpi_calc.csv" \
  --kpi-log "$RUN_A/baseline/kpi/kpi_analyze.log" \
  --map-value 0.41 \
  --summary "kpi: mAP=0.41" --duration-sec 58
assert_rc 0 "commit baseline/kpi_analyze"
expect_audit "$RUN_A" baseline/kpi_analyze gap_analysis \
  references/tao-analyze-gaps-od-map.md

section "A4. negative — out-of-order commit is rejected and writes nothing"

make_iter_artifacts "$RUN_A" iter1
freeze "$RUN_A"

# embed is iter1's second stage; gap_analysis has not run.
commit "$RUN_A" iter1 embed \
  --embeddings-parquet "$RUN_A/iter1/embeddings/weak_images_embeddings.parquet" \
  --summary "embed before gap_analysis"
assert_rc 1 "out-of-order iter1/embed is rejected"
case "$RUN_OUT" in
  *"out-of-order"*) ok "rejection names the ordering violation" ;;
  *) notok "rejection names the ordering violation" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_A" "state and log unchanged after the out-of-order commit"

# Skipping a whole phase is rejected the same way.
commit "$RUN_A" iter1 train \
  --checkpoint "$RUN_A/iter1/train/gdino_model_latest.pth" \
  --training-spec "$RUN_A/iter1/train_grounding_dino.yaml" \
  --summary "train before anything else"
assert_rc 1 "out-of-order iter1/train is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the second out-of-order commit"

# Re-committing a stage that already completed is also out of order.
commit "$RUN_A" baseline kpi_analyze \
  --kpi-csv "$RUN_A/baseline/kpi/kpi_calc.csv" \
  --summary "duplicate kpi_analyze"
assert_rc 1 "re-committing a completed stage is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the duplicate commit"

expect_audit "$RUN_A" baseline/kpi_analyze gap_analysis \
  references/tao-analyze-gaps-od-map.md

section "A5. negative — missing artifacts are rejected and write nothing"

freeze "$RUN_A"

commit "$RUN_A" iter1 gap_analysis \
  --weak-images "$RUN_A/iter1/gaps/does_not_exist.parquet" \
  --gap-report "$RUN_A/iter1/gaps/gap_report.json" \
  --summary "gap_analysis with a missing weak-images parquet"
assert_rc 1 "commit with a nonexistent artifact path is rejected"
case "$RUN_OUT" in
  *"--weak-images"*) ok "rejection names the missing flag" ;;
  *) notok "rejection names the missing flag" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_A" "state and log unchanged after the missing-artifact commit"

commit "$RUN_A" iter1 gap_analysis \
  --weak-images "$RUN_A/iter1/gaps/weak_images.parquet" \
  --summary "gap_analysis without --gap-report"
assert_rc 1 "commit with a required flag omitted is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the omitted-flag commit"

# A file where a directory is required is a missing artifact, not a pass.
commit "$RUN_A" iter1 gap_analysis \
  --weak-images "$RUN_A/iter1/gaps/weak_images.parquet" \
  --gap-report "$RUN_A/iter1/gaps" \
  --summary "gap_analysis with a directory where a file belongs"
assert_rc 1 "commit with the wrong artifact kind is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the wrong-kind commit"

# An artifact belonging to another stage is rejected before anything is written.
commit "$RUN_A" iter1 gap_analysis \
  --weak-images "$RUN_A/iter1/gaps/weak_images.parquet" \
  --gap-report "$RUN_A/iter1/gaps/gap_report.json" \
  --checkpoint "$RUN_A/iter1/train/gdino_model_latest.pth" \
  --summary "gap_analysis carrying train's checkpoint"
assert_rc 1 "commit carrying another stage's flag is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the misfiled-flag commit"

expect_audit "$RUN_A" baseline/kpi_analyze gap_analysis \
  references/tao-analyze-gaps-od-map.md

section "A6. iter1 — seven stages in order"

commit_iteration "$RUN_A" iter1 gap_analysis references/tao-analyze-gaps-od-map.md

section "A7. negative — an unfinished run is neither terminal nor complete"

run "$PY" "$AUDIT" --results-dir "$RUN_A" --require-complete
assert_rc 1 "--require-complete fails on an unfinished run"
case "$RUN_OUT" in
  *"no loop_stop event is committed"*) ok "--require-complete explains why" ;;
  *) notok "--require-complete explains why" "output: $RUN_OUT" ;;
esac

run "$PY" "$AUDIT" --results-dir "$RUN_A" --require-terminal
assert_rc 1 "--require-terminal fails on an unfinished run"

section "A8. iter2 — seven stages in order"

make_iter_artifacts "$RUN_A" iter2
commit_iteration "$RUN_A" iter2 loop_stop none

section "A9. loop_stop"

commit "$RUN_A" iter2 loop_stop \
  --summary "loop complete: 2 iterations, mAP 0.41 -> 0.55" --duration-sec 1
assert_rc 0 "commit iter2/loop_stop"
expect_audit "$RUN_A" iter2/loop_stop complete none
assert_eq true "$AUDIT_TERMINAL" "terminal=true after loop_stop"
assert_eq true "$AUDIT_COMPLETE" "complete=true after loop_stop"

run "$PY" "$AUDIT" --results-dir "$RUN_A" --require-complete
assert_rc 0 "--require-complete passes on the finished run"
run "$PY" "$AUDIT" --results-dir "$RUN_A" --require-terminal
assert_rc 0 "--require-terminal passes on the finished run"
run "$PY" "$AUDIT" --results-dir "$RUN_A" --json
assert_rc 0 "--json exits 0 on the finished run"
run "$PY" -c "import json,sys; json.loads(sys.argv[1])" "$RUN_OUT"
assert_rc 0 "--json emits one parseable JSON object"

section "A10. negative — nothing may follow a committed loop_stop"

freeze "$RUN_A"
commit "$RUN_A" iter2 kpi_analyze \
  --kpi-csv "$RUN_A/iter2/kpi/kpi_calc.csv" \
  --summary "post-terminal commit"
assert_rc 1 "commit after loop_stop is rejected"
assert_unchanged "$RUN_A" "state and log unchanged after the post-terminal commit"

section "A11. the log and state the run left behind"

run "$PY" - "$RUN_A" <<'PYEOF'
"""Structural check of the finished run: exact event sequence, seq integrity,
event schema, and every artifact path recorded in state still on disk."""
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
state = json.loads((results / "deft_state.json").read_text())
events = [json.loads(line) for line in
          (results / "loop_log.jsonl").read_text().splitlines() if line.strip()]

expected = (
    [("baseline", "inference"), ("baseline", "kpi_analyze")]
    + [(f"iter{n}", stage) for n in (1, 2) for stage in
       ("gap_analysis", "embed", "mine", "stage", "train", "inference", "kpi_analyze")]
    + [("iter2", "loop_stop")]
)
actual = [(e.get("iter"), e.get("stage")) for e in events]
problems = []
if actual != expected:
    problems.append(f"event sequence\n  expected {expected}\n  actual   {actual}")

required = {"seq", "ts", "iter", "stage", "status", "summary",
            "duration_sec", "context_tokens"}
for index, event in enumerate(events, 1):
    missing = required - set(event)
    if missing:
        problems.append(f"event {index} missing keys {sorted(missing)}")
    if event.get("seq") != index:
        problems.append(f"event {index} has seq={event.get('seq')!r}")
    if event.get("status") != "ok":
        problems.append(f"event {index} has status={event.get('status')!r}")
    if not isinstance(event.get("duration_sec"), int) or event["duration_sec"] < 0:
        problems.append(f"event {index} duration_sec={event.get('duration_sec')!r}")
    if event.get("context_tokens") != 0:
        problems.append(f"event {index} context_tokens={event.get('context_tokens')!r}")

if state.get("status") != "complete":
    problems.append(f"state.status={state.get('status')!r}, expected 'complete'")
if state.get("current_iteration") != 2:
    problems.append(f"state.current_iteration={state.get('current_iteration')!r}")
if "stopped_at" not in state:
    problems.append("state.stopped_at was never recorded by loop_stop")
if state.get("config", {}).get("max_iterations") != 2:
    problems.append("state.config.max_iterations was not frozen at 2")

for phase in ("baseline", "iter1", "iter2"):
    entry = state.get("iterations", {}).get(phase)
    if not isinstance(entry, dict):
        problems.append(f"state.iterations.{phase} is missing")
        continue
    for field, value in entry.items():
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        if not Path(value).exists():
            problems.append(f"{phase}.{field} points at a missing path: {value}")
for phase, last in (("baseline", "kpi_analyze"),
                    ("iter1", "kpi_analyze"), ("iter2", "kpi_analyze")):
    got = state.get("iterations", {}).get(phase, {}).get("stage_completed")
    if got != last:
        problems.append(f"{phase}.stage_completed={got!r}, expected {last!r}")
# The mAP the loop exists to report has to survive to disk.
for phase in ("baseline", "iter1", "iter2"):
    if "map_value" not in state.get("iterations", {}).get(phase, {}):
        problems.append(f"{phase}.map_value was not recorded")

if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"{len(events)} events, sequence and artifacts verified")
PYEOF
assert_rc 0 "final state and log are structurally sound"
[ "$RUN_RC" = 0 ] || printf '  # %s\n' "$RUN_OUT"

assert_no_audit_warnings "$RUN_A" "the finished run audits without a single warning"
assert_no_tmp_files "$RUN_A" "no tmp files survived any atomic write"

# ═══════════════════════════════════════════════════════════════════════════
# RUN B — a hard stop: --status error is terminal but never complete
# ═══════════════════════════════════════════════════════════════════════════

section "B1. a failing stage marks the run failed"

WS_B=$(new_workspace ws_failed)
make_pool "$WS_B"
RUN_B="$WS_B/results/run_failed"

init_run "$WS_B" "$RUN_B" 2
assert_rc 0 "init_deft_state.py exits 0 (pool already on disk)"
# The pool exists, so prep is skipped entirely and baseline leads.
expect_audit "$RUN_B" none inference references/grounding-dino.md

make_phase_artifacts "$RUN_B" baseline
commit "$RUN_B" baseline inference \
  --inference-labels-dir "$RUN_B/baseline/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 120
assert_rc 0 "commit baseline/inference"
expect_audit "$RUN_B" baseline/inference kpi_analyze \
  references/tao-analyze-detection-kpi.md

# A stage that dies produces no artifacts; the commit records the failure anyway.
commit "$RUN_B" baseline kpi_analyze --status error \
  --summary "kpi_analyze failed: analytics container exited 1" --duration-sec 12
assert_rc 0 "commit baseline/kpi_analyze --status error"
expect_audit "$RUN_B" baseline/kpi_analyze loop_stop none
# A hard stop still owes its loop_stop: next_action says so, so terminal must not
# say the opposite. The reporter gates the final render on --require-terminal, and
# rendering before loop_stop leaves stopped_at unset and no terminal log event.
assert_eq false "$AUDIT_TERMINAL" "terminal=false while the hard stop is unfinalized"
assert_eq false "$AUDIT_COMPLETE" "complete=false after the hard stop"

run "$PY" "$AUDIT" --results-dir "$RUN_B" --require-terminal
assert_rc 1 "--require-terminal fails until the hard stop commits loop_stop"
case "$RUN_OUT" in
  *"no loop_stop is committed"*) ok "--require-terminal names the missing loop_stop" ;;
  *) notok "--require-terminal names the missing loop_stop" "output: $RUN_OUT" ;;
esac
run "$PY" "$AUDIT" --results-dir "$RUN_B" --require-complete
assert_rc 1 "--require-complete fails on the failed run"

run "$PY" -c "
import json, sys
state = json.load(open(sys.argv[1]))
assert state['status'] == 'failed', state['status']
entry = state['iterations']['baseline']
assert entry['status'] == 'failed', entry
assert entry['failed_stage'] == 'kpi_analyze', entry
assert entry['stage_completed'] == 'inference', entry
" "$RUN_B/deft_state.json"
assert_rc 0 "state records the failure without advancing stage_completed"

section "B2. only loop_stop may follow a hard stop"

freeze "$RUN_B"
commit "$RUN_B" baseline kpi_analyze \
  --kpi-csv "$RUN_B/baseline/kpi/kpi_calc.csv" \
  --summary "silent retry after the hard stop"
assert_rc 1 "retrying the failed stage is rejected"
case "$RUN_OUT" in
  *"hard stop"*) ok "rejection names the hard stop" ;;
  *) notok "rejection names the hard stop" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_B" "state and log unchanged after the rejected retry"

commit "$RUN_B" iter1 gap_analysis \
  --weak-images "$WS_B/source_pool/source_embeddings.parquet" \
  --gap-report "$WS_B/source_pool/source_embeddings.parquet" \
  --summary "advancing past the hard stop"
assert_rc 1 "advancing to the next phase after a hard stop is rejected"
assert_unchanged "$RUN_B" "state and log unchanged after the rejected advance"

commit "$RUN_B" baseline loop_stop \
  --summary "halted: kpi_analyze failed at baseline" --duration-sec 1
assert_rc 0 "commit baseline/loop_stop after the hard stop"
audit_kv "$RUN_B"
assert_eq VALID "$AUDIT_STATUS" "finalized failed run is still VALID"
assert_eq true "$AUDIT_TERMINAL" "terminal=true after finalizing the failure"
assert_eq false "$AUDIT_COMPLETE" "complete=false after finalizing the failure"
case "$AUDIT_NEXT" in
  *FAILED*) ok "next_action tells the agent to report FAILED" ;;
  *) notok "next_action tells the agent to report FAILED" "actual: $AUDIT_NEXT" ;;
esac
assert_eq none "$AUDIT_READ" "read_before_action after the finalized failure"

run "$PY" "$AUDIT" --results-dir "$RUN_B" --require-terminal
assert_rc 0 "--require-terminal passes on the finalized failed run"
run "$PY" "$AUDIT" --results-dir "$RUN_B" --require-complete
assert_rc 1 "--require-complete still fails on the finalized failed run"

# ═══════════════════════════════════════════════════════════════════════════
# RUN C — a commit the post-commit audit rejects must roll both files back
# ═══════════════════════════════════════════════════════════════════════════

section "C1. a phase may not start before the phase it is measured against finished"

WS_C=$(new_workspace ws_rollback)
make_pool "$WS_C"
RUN_C="$WS_C/results/run_rollback"

init_run "$WS_C" "$RUN_C" 2
assert_rc 0 "init_deft_state.py exits 0"
make_phase_artifacts "$RUN_C" baseline
commit "$RUN_C" baseline inference \
  --inference-labels-dir "$RUN_C/baseline/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 120
assert_rc 0 "commit baseline/inference"
commit "$RUN_C" baseline kpi_analyze \
  --kpi-csv "$RUN_C/baseline/kpi/kpi_calc.csv" --map-value 0.4 \
  --summary "kpi: mAP=0.40" --duration-sec 60
assert_rc 0 "commit baseline/kpi_analyze"

# iter2/gap_analysis passes the per-phase order check (iter2 has completed
# nothing) but consumes an iter1 that never ran. iter1 is also the phase iter2's
# mAP is compared against, so starting iter2 on it is rejected outright — the
# cross-phase gate fires before anything is written.
make_iter_artifacts "$RUN_C" iter2
freeze "$RUN_C"
commit "$RUN_C" iter2 gap_analysis \
  --weak-images "$RUN_C/iter2/gaps/weak_images.parquet" \
  --gap-report "$RUN_C/iter2/gaps/gap_report.json" \
  --weak-image-count 120 \
  --summary "gap_analysis for iter2 with iter1 never run"
assert_rc 1 "starting iter2 while iter1 never finished is rejected"
case "$RUN_OUT" in
  *"until iter1 finishes kpi_analyze"*) ok "rejection names the unfinished phase" ;;
  *) notok "rejection names the unfinished phase" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_C" "state and log unchanged after the cross-phase rejection"

# Same rule one phase earlier: a baseline abandoned at inference has no mAP, so
# iteration 1 has nothing to be measured against and may not start.
WS_C2=$(new_workspace ws_baseline_abandoned)
make_pool "$WS_C2"
RUN_C2="$WS_C2/results/run_baseline_abandoned"
init_run "$WS_C2" "$RUN_C2" 2
assert_rc 0 "init_deft_state.py exits 0"
make_phase_artifacts "$RUN_C2" baseline
make_iter_artifacts "$RUN_C2" iter1
commit "$RUN_C2" baseline inference \
  --inference-labels-dir "$RUN_C2/baseline/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 120
assert_rc 0 "commit baseline/inference"
freeze "$RUN_C2"
commit "$RUN_C2" iter1 gap_analysis \
  --weak-images "$RUN_C2/iter1/gaps/weak_images.parquet" \
  --gap-report "$RUN_C2/iter1/gaps/gap_report.json" \
  --weak-image-count 120 \
  --summary "gap_analysis before the baseline KPI ran"
assert_rc 1 "starting iter1 while baseline stopped at inference is rejected"
case "$RUN_OUT" in
  *"until baseline finishes kpi_analyze"*) ok "rejection names the abandoned baseline" ;;
  *) notok "rejection names the abandoned baseline" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_C2" "state and log unchanged after the abandoned-baseline commit"
expect_audit "$RUN_C2" baseline/inference kpi_analyze \
  references/tao-analyze-detection-kpi.md

section "C1b. the post-commit audit rolls back only what the commit introduced"

# The gates above catch every inconsistency commit_stage.py can see for itself,
# so the post-commit rollback is now a backstop — and a backstop still has to be
# proven. AUDIT_SCRIPT is pointed at a wrapper around the real audit that injects
# one extra error, which is the only shape of rejection left: an error that was
# not there before this commit.
cat >"$WORK/audit_plus_error.py" <<'PYEOF'
"""Real audit, plus one injected error. Set DEFT_TEST_INJECT=always|after."""
import json
import os
import subprocess
import sys

REAL = os.environ["DEFT_TEST_REAL_AUDIT"]
MARK = "injected: a synthetic audit error"
proc = subprocess.run([sys.executable, REAL] + sys.argv[1:], capture_output=True, text=True)
sys.stderr.write(proc.stderr)
if "--json" not in sys.argv:
    sys.stdout.write(proc.stdout)
    raise SystemExit(proc.returncode)
report = json.loads(proc.stdout)
mode = os.environ.get("DEFT_TEST_INJECT", "after")
# "after": the error appears only once the commit has been written, so it looks
# newly introduced. "always": it is there before and after, i.e. inherited.
if mode == "always" or report["log_entries"] > int(os.environ["DEFT_TEST_BASE_EVENTS"]):
    report["errors"] = report["errors"] + [MARK]
    report["status"] = "INVALID"
print(json.dumps(report, indent=2))
raise SystemExit(1 if report["status"] == "INVALID" else 0)
PYEOF

cat >"$WORK/commit_with_stub_audit.py" <<'PYEOF'
"""Run the real commit_stage.main() against a stubbed audit script."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["DEFT_TEST_SCRIPTS"])
import commit_stage  # noqa: E402

commit_stage.AUDIT_SCRIPT = Path(os.environ["DEFT_TEST_STUB_AUDIT"])
raise SystemExit(commit_stage.main())
PYEOF

export DEFT_TEST_SCRIPTS="$SCRIPTS_DIR"
export DEFT_TEST_REAL_AUDIT="$AUDIT"
export DEFT_TEST_STUB_AUDIT="$WORK/audit_plus_error.py"
export DEFT_TEST_BASE_EVENTS=2   # baseline inference + kpi_analyze

make_iter_artifacts "$RUN_C" iter1
freeze "$RUN_C"
DEFT_TEST_INJECT=after run "$PY" "$WORK/commit_with_stub_audit.py" \
  --results-dir "$RUN_C" --iter-label iter1 --stage gap_analysis \
  --weak-images "$RUN_C/iter1/gaps/weak_images.parquet" \
  --gap-report "$RUN_C/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "gap_analysis the audit will reject"
assert_rc 2 "a commit that introduces an audit error exits 2"
case "$RUN_OUT" in
  *"rolled back"*) ok "rollback is reported on stderr" ;;
  *) notok "rollback is reported on stderr" "output: $RUN_OUT" ;;
esac
case "$RUN_OUT" in
  *"injected: a synthetic audit error"*) ok "the rejection names the error it introduced" ;;
  *) notok "the rejection names the error it introduced" "output: $RUN_OUT" ;;
esac
assert_unchanged "$RUN_C" "state and log rolled back byte-for-byte"
assert_no_tmp_files "$RUN_C" "the rollback left no tmp file behind"
[ -e "$RUN_C/.deft_commit.journal" ] \
  && notok "the rollback cleared its journal" "journal survived the rollback" \
  || ok "the rollback cleared its journal"

# The same error, already present before the commit, must NOT veto it. A run that
# went inconsistent for an unrelated reason still has to be able to record a
# stage, a failure, and above all its loop_stop.
DEFT_TEST_INJECT=always run "$PY" "$WORK/commit_with_stub_audit.py" \
  --results-dir "$RUN_C" --iter-label iter1 --stage gap_analysis \
  --weak-images "$RUN_C/iter1/gaps/weak_images.parquet" \
  --gap-report "$RUN_C/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "gap_analysis over a pre-existing inconsistency"
assert_rc 0 "a pre-existing audit error does not veto the next commit"
case "$RUN_OUT" in
  *"already inconsistent before this commit"*) ok "the inherited error is reported, loudly" ;;
  *) notok "the inherited error is reported, loudly" "output: $RUN_OUT" ;;
esac
unset DEFT_TEST_INJECT DEFT_TEST_SCRIPTS DEFT_TEST_REAL_AUDIT DEFT_TEST_STUB_AUDIT
unset DEFT_TEST_BASE_EVENTS

expect_audit "$RUN_C" iter1/gap_analysis embed \
  references/tao-generate-image-embeddings.md

section "C2. a deleted artifact invalidates the run"

rm -f "$RUN_C/baseline/kpi/kpi_calc.csv"
audit_kv "$RUN_C"
assert_eq 1 "$AUDIT_RC" "audit exits 1 when a recorded artifact vanished"
assert_eq INVALID "$AUDIT_STATUS" "DEFT_RUN_STATUS=INVALID when an artifact vanished"
case "$AUDIT_NEXT" in
  repair*) ok "next_action is the repair instruction" ;;
  *) notok "next_action is the repair instruction" "actual: $AUDIT_NEXT" ;;
esac
assert_eq none "$AUDIT_READ" "read_before_action is none while INVALID"

section "C3. an audit of a directory with no state"

run "$PY" "$AUDIT" --results-dir "$WORK/never_initialized"
assert_rc 2 "audit exits 2 when deft_state.json is missing"

# ═══════════════════════════════════════════════════════════════════════════
# RUN D — loop_stop before the run finished: terminal, VALID, but not complete
# ═══════════════════════════════════════════════════════════════════════════

section "D1. loop_stop committed early must not report itself as complete"

WS_D=$(new_workspace ws_early_stop)
make_pool "$WS_D"
RUN_D="$WS_D/results/run_early_stop"

init_run "$WS_D" "$RUN_D" 3
assert_rc 0 "init_deft_state.py exits 0 with max_iterations=3"
make_phase_artifacts "$RUN_D" baseline
commit "$RUN_D" baseline inference \
  --inference-labels-dir "$RUN_D/baseline/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 120
assert_rc 0 "commit baseline/inference"
commit "$RUN_D" baseline kpi_analyze \
  --kpi-csv "$RUN_D/baseline/kpi/kpi_calc.csv" --map-value 0.4 \
  --summary "kpi: mAP=0.40" --duration-sec 60
assert_rc 0 "commit baseline/kpi_analyze"

make_iter_artifacts "$RUN_D" iter1
commit_iteration "$RUN_D" iter1 gap_analysis references/tao-analyze-gaps-od-map.md

# One of three iterations ran. loop_stop is legal from any phase, so this is a
# well-formed run — it is simply not a finished one.
commit "$RUN_D" iter1 loop_stop \
  --summary "stopped after iter1 at the user's request" --duration-sec 1
assert_rc 0 "commit iter1/loop_stop before max_iterations"
audit_kv "$RUN_D"
assert_eq VALID "$AUDIT_STATUS" "an early loop_stop leaves the run VALID"
assert_eq true "$AUDIT_TERMINAL" "terminal=true after the early loop_stop"
assert_eq false "$AUDIT_COMPLETE" "complete=false after the early loop_stop"
assert_eq none "$AUDIT_READ" "read_before_action after the early loop_stop"
if [ "$AUDIT_NEXT" = "complete" ]; then
  notok "next_action does not claim completion after an early loop_stop" \
    "next_action=complete contradicts complete=false; --require-complete rejects this run"
else
  ok "next_action does not claim completion after an early loop_stop"
fi
run "$PY" "$AUDIT" --results-dir "$RUN_D" --require-complete
assert_rc 1 "--require-complete fails after the early loop_stop"
run "$PY" "$AUDIT" --results-dir "$RUN_D" --require-terminal
assert_rc 0 "--require-terminal passes after the early loop_stop"

section "D2. the documented zero-weak-image early stop is a real completion"

# The early stop has two independent proofs on the phase entry — a zero
# weak_image_count and the explicit zero_weak_images flag — and an overlay may
# record either one. Testing them together would let one of them rot, so each
# spelling drives its own run.
zero_weak_run() {  # zero_weak_run NAME LABEL [gap_analysis flags...]
  local name=$1 label=$2
  shift 2
  local ws results
  ws=$(new_workspace "$name")
  make_pool "$ws"
  results="$ws/results/$name"

  init_run "$ws" "$results" 3
  assert_rc 0 "[$label] init_deft_state.py exits 0 with max_iterations=3"
  make_phase_artifacts "$results" baseline
  commit "$results" baseline inference \
    --inference-labels-dir "$results/baseline/inference/labels" \
    --summary "inference: 1 label file" --duration-sec 120
  assert_rc 0 "[$label] commit baseline/inference"
  commit "$results" baseline kpi_analyze \
    --kpi-csv "$results/baseline/kpi/kpi_calc.csv" --map-value 0.95 \
    --summary "kpi: mAP=0.95" --duration-sec 60
  assert_rc 0 "[$label] commit baseline/kpi_analyze"

  # Every class already meets its AP50 threshold, so the remaining six stages
  # of the iteration have nothing to consume and the loop stops here.
  make_iter_artifacts "$results" iter1
  commit "$results" iter1 gap_analysis \
    --weak-images "$results/iter1/gaps/weak_images.parquet" \
    --gap-report "$results/iter1/gaps/gap_report.json" \
    "$@" \
    --summary "gap_analysis: 0 weak images; every class met its AP50 threshold" \
    --duration-sec 88
  assert_rc 0 "[$label] commit iter1/gap_analysis"
  expect_audit "$results" iter1/gap_analysis loop_stop none

  commit "$results" iter1 loop_stop \
    --summary "early stop: no weak images remain" --duration-sec 1
  assert_rc 0 "[$label] commit iter1/loop_stop"
  expect_audit "$results" iter1/loop_stop complete none
  assert_eq true "$AUDIT_COMPLETE" "[$label] complete=true after the early stop"
  run "$PY" "$AUDIT" --results-dir "$results" --require-complete
  assert_rc 0 "[$label] --require-complete passes on the early stop"
}

zero_weak_run ws_zero_count "--weak-image-count 0" --weak-image-count 0
zero_weak_run ws_zero_flag "--zero-weak-images" --zero-weak-images

# ═══════════════════════════════════════════════════════════════════════════
# RUN F — a stage that dies before producing anything, in a phase state has
#         never seen: the commit must still record the only evidence there is
# ═══════════════════════════════════════════════════════════════════════════

section "F1. a failure with no artifacts at all is still recorded"

WS_F=$(new_workspace ws_first_stage_failure)
make_pool "$WS_F"
RUN_F="$WS_F/results/run_first_stage_failure"

init_run "$WS_F" "$RUN_F" 2
assert_rc 0 "init_deft_state.py exits 0"
make_phase_artifacts "$RUN_F" baseline
commit "$RUN_F" baseline inference \
  --inference-labels-dir "$RUN_F/baseline/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 120
assert_rc 0 "commit baseline/inference"
commit "$RUN_F" baseline kpi_analyze \
  --kpi-csv "$RUN_F/baseline/kpi/kpi_calc.csv" --map-value 0.4 \
  --summary "kpi: mAP=0.40" --duration-sec 60
assert_rc 0 "commit baseline/kpi_analyze"

# iter1's first stage dies. state.iterations.iter1 does not exist yet, and the
# artifact the flag names was never written — the commit has to survive both.
commit "$RUN_F" iter1 gap_analysis --status error \
  --weak-images "$RUN_F/iter1/gaps/weak_images.parquet" \
  --summary "gap_analysis failed: analytics exited 1, no parquet emitted" \
  --duration-sec 31
assert_rc 0 "commit iter1/gap_analysis --status error with no artifacts on disk"
case "$RUN_OUT" in
  *"--weak-images was not on disk"*) ok "the unavailable artifact is reported, not recorded" ;;
  *) notok "the unavailable artifact is reported, not recorded" "output: $RUN_OUT" ;;
esac
expect_audit "$RUN_F" iter1/gap_analysis loop_stop none

run "$PY" -c "
import json, sys
state = json.load(open(sys.argv[1]))
entry = state['iterations']['iter1']
assert state['status'] == 'failed', state['status']
assert entry['status'] == 'failed', entry
assert entry['failed_stage'] == 'gap_analysis', entry
assert 'stage_completed' not in entry, entry
assert 'weak_images_parquet' not in entry, entry
" "$RUN_F/deft_state.json"
assert_rc 0 "the failed phase records no stage_completed and no missing path"

commit "$RUN_F" iter1 loop_stop \
  --summary "halted: gap_analysis failed at iter1" --duration-sec 1
assert_rc 0 "commit iter1/loop_stop after the first-stage failure"
run "$PY" "$AUDIT" --results-dir "$RUN_F" --require-terminal
assert_rc 0 "--require-terminal passes on the finalized failure"
run "$PY" "$AUDIT" --results-dir "$RUN_F" --require-complete
assert_rc 1 "--require-complete fails on the finalized failure"

# ═══════════════════════════════════════════════════════════════════════════
# RUN G — one section per defect an adversarial review found in these three
#         scripts. Each asserts the behaviour that replaced the defect, so the
#         defect cannot come back quietly.
# ═══════════════════════════════════════════════════════════════════════════

g_init() {  # g_init NAME MAX [init flags...] -> G_WS, G_RUN
  local name=$1 max=$2
  shift 2
  G_WS=$(new_workspace "$name")
  make_pool "$G_WS"
  G_RUN="$G_WS/results/$name"
  init_run "$G_WS" "$G_RUN" "$max" "$@"
  make_phase_artifacts "$G_RUN" baseline
}

g_baseline() {  # g_baseline RESULTS LABEL [map]
  local results=$1 label=$2 map=${3:-0.40}
  commit "$results" baseline inference \
    --inference-labels-dir "$results/baseline/inference/labels" \
    --summary "inference: 1 label file" --duration-sec 120
  commit "$results" baseline kpi_analyze \
    --kpi-csv "$results/baseline/kpi/kpi_calc.csv" \
    --kpi-log "$results/baseline/kpi/kpi_analyze.log" --map-value "$map" \
    --summary "kpi: mAP=$map" --duration-sec 60
  assert_rc 0 "[$label] baseline committed"
}

report_field() {  # report_field RESULTS KEY -> one field of the --json report
  "$PY" -c "
import json, subprocess, sys
proc = subprocess.run([sys.executable, sys.argv[1], '--results-dir', sys.argv[2], '--json'],
                      capture_output=True, text=True)
report = json.loads(proc.stdout)
value = report[sys.argv[3]]
print(json.dumps(value) if not isinstance(value, str) else value)
" "$AUDIT" "$1" "$2"
}

section "G1. a baseline abandoned at inference can never be reported complete"

# The reviewer's run: baseline stops at inference, every iteration finishes, and
# loop_stop lands. Committing it is now impossible (C1), so the shape is built by
# hand to prove the audit rejects it on the way back in too — the completion gate
# is what the agent's final claim rests on.
g_init ws_no_baseline_kpi 1
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G1
commit_iteration "$G_RUN" iter1 loop_stop none
commit "$G_RUN" iter1 loop_stop --summary "loop complete" --duration-sec 1
assert_rc 0 "[G1] the honest run commits loop_stop"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-complete
assert_rc 0 "[G1] the honest run is complete"

run "$PY" - "$G_RUN" <<'PYEOF'
"""Delete the baseline kpi_analyze the reviewer's agent skipped, and renumber."""
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
log = results / "loop_log.jsonl"
events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
events = [e for e in events if not (e["iter"] == "baseline" and e["stage"] == "kpi_analyze")]
for index, event in enumerate(events, 1):
    event["seq"] = index
log.write_text("".join(json.dumps(e) + "\n" for e in events))

state_path = results / "deft_state.json"
state = json.loads(state_path.read_text())
baseline = state["iterations"]["baseline"]
for field in ("kpi_csv", "kpi_log", "map_value"):
    baseline.pop(field, None)
baseline["stage_completed"] = "inference"
baseline["status"] = "in_progress"
state_path.write_text(json.dumps(state, indent=2))
PYEOF
assert_rc 0 "[G1] fixture: the baseline KPI is removed from state and log"

audit_kv "$G_RUN"
assert_eq INVALID "$AUDIT_STATUS" "[G1] a run with no baseline KPI is INVALID"
assert_eq false "$AUDIT_COMPLETE" "[G1] complete=false without a baseline mAP"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-complete
assert_rc 1 "[G1] --require-complete rejects it"
case "$RUN_OUT" in
  *"baseline never finished kpi_analyze"*) ok "[G1] the audit names the abandoned baseline" ;;
  *) notok "[G1] the audit names the abandoned baseline" "output: $RUN_OUT" ;;
esac

section "G2. next_action follows run progress, not the last event's phase"

# prep is legal at any point, and committing it late used to rewind next_action
# to baseline/inference on a run that had already trained iteration 1.
g_init ws_late_prep 2
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G2
commit_iteration "$G_RUN" iter1 gap_analysis references/tao-analyze-gaps-od-map.md
make_file "$G_RUN/prep/pool_report.json" '{"annotations_by_class": {"car": 900}}'
commit "$G_RUN" prep prep \
  --pool-odvg "$G_WS/source_pool/odvg" \
  --pool-embeddings "$G_WS/source_pool/source_embeddings.parquet" \
  --pool-report "$G_RUN/prep/pool_report.json" \
  --summary "prep: recorded the pool that was already on disk" --duration-sec 10
assert_rc 0 "[G2] a late prep commit is still accepted"
expect_audit "$G_RUN" prep/prep gap_analysis references/tao-analyze-gaps-od-map.md

section "G3. a missing pool routes to prep; it does not refuse the run"

# Asking for the loop without a prepared pool is a reason to prep, not to refuse.
# But prep needs inputs the loop otherwise never asks for, so the permission is
# tied to having those: without them there is nothing to route to, and starting
# anyway only defers the failure to `mine`.
WS_G3=$(new_workspace ws_pool_missing)   # no make_pool: prep has not run
RUN_G3="$WS_G3/results/run_pool_missing"
init_run "$WS_G3" "$RUN_G3" 1
assert_rc 1 "[G3] init refuses when the pool is missing and prep has no inputs"
case "$RUN_OUT" in
  *"has to build it"*) ok "[G3] the error says prep must build the pool" ;;
  *) notok "[G3] the error says prep must build the pool" "output: $RUN_OUT" ;;
esac
case "$RUN_OUT" in
  *"--codetr-checkpoint"*) ok "[G3] the error names the missing prep inputs" ;;
  *) notok "[G3] the error names the missing prep inputs" "output: $RUN_OUT" ;;
esac
[ -f "$RUN_G3/deft_state.json" ] \
  && notok "[G3] no state is written on a refused init" \
  || ok "[G3] no state is written on a refused init"

# Given prep's inputs, the same missing pool becomes a routing decision.
make_file "$WS_G3/raw/img0.png" "png"
make_file "$WS_G3/ckpt/codetr.pth" "not a real checkpoint"
make_file "$WS_G3/ckpt/coco_classmap.txt" "person"
init_run "$WS_G3" "$RUN_G3" 1 \
  --pool-images "$WS_G3/raw" \
  --codetr-checkpoint "$WS_G3/ckpt/codetr.pth" \
  --codetr-classmap "$WS_G3/ckpt/coco_classmap.txt"
assert_rc 0 "[G3] init succeeds when prep can build the pool"
expect_audit "$RUN_G3" none prep references/prep-source-pool.md
case "$(state_json "$RUN_G3" codetr_checkpoint)" in
  *codetr.pth*) ok "[G3] the labeler is frozen so a resumed prep uses the same one" ;;
  *) notok "[G3] the labeler is frozen so a resumed prep uses the same one" \
       "got: $(state_json "$RUN_G3" codetr_checkpoint)" ;;
esac

# A supplied prep input that does not exist is still an error.
init_run "$WS_G3" "$RUN_G3" 1 --force \
  --pool-images "$WS_G3/raw" \
  --codetr-checkpoint "$WS_G3/ckpt/nope.pth" \
  --codetr-classmap "$WS_G3/ckpt/coco_classmap.txt"
assert_rc 1 "[G3] a prep input that is not on disk is rejected"

# With the pool already prepared, prep is skipped and baseline leads.
WS_G3C=$(new_workspace ws_pool_ready); make_pool "$WS_G3C"
RUN_G3C="$WS_G3C/results/run_pool_ready"
init_run "$WS_G3C" "$RUN_G3C" 1
assert_rc 0 "[G3] a prepared pool needs no prep inputs"
expect_audit "$RUN_G3C" none inference references/grounding-dino.md

# --pool-report cross-checks that the pool was prepared for THESE target classes.
# A pool holding no examples of a target class cannot be mined for it, and mining
# would not fail -- it would return neighbours of something else.
WS_G3B=$(new_workspace ws_pool_classes); make_pool "$WS_G3B"
RUN_G3B="$WS_G3B/results/run_pool_classes"
make_file "$WS_G3B/pool_report.json" '{"annotations_by_class": {"car": 900, "person": 40}}'
init_run "$WS_G3B" "$RUN_G3B" 1 --pool-report "$WS_G3B/pool_report.json" \
  --target-classes car,person,bicycle
assert_rc 1 "[G3] init rejects a pool prepared for a different class set"
case "$RUN_OUT" in
  *"holds no annotations for target class"*"bicycle"*)
    ok "[G3] the error names the unbacked class" ;;
  *) notok "[G3] the error names the unbacked class" "output: $RUN_OUT" ;;
esac

init_run "$WS_G3B" "$RUN_G3B" 1 --pool-report "$WS_G3B/pool_report.json" \
  --target-classes car,person
assert_rc 0 "[G3] a pool covering every target class is accepted"

section "G4. zero-padded iteration labels are not accepted as extra iterations"

g_init ws_padded_label 2
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G4
commit_iteration "$G_RUN" iter1 gap_analysis references/tao-analyze-gaps-od-map.md
make_iter_artifacts "$G_RUN" iter01
freeze "$G_RUN"
commit "$G_RUN" iter01 gap_analysis \
  --weak-images "$G_RUN/iter01/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter01/gaps/gap_report.json" \
  --weak-image-count 120 --summary "an extra iteration wearing iter1's number"
assert_rc 1 "[G4] iter01 is rejected"
case "$RUN_OUT" in
  *"not a canonical label"*) ok "[G4] the rejection names the canonical spelling" ;;
  *) notok "[G4] the rejection names the canonical spelling" "output: $RUN_OUT" ;;
esac
assert_unchanged "$G_RUN" "[G4] state and log unchanged after the padded-label commit"
commit "$G_RUN" iter001 gap_analysis \
  --weak-images "$G_RUN/iter01/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter01/gaps/gap_report.json" \
  --weak-image-count 120 --summary "iter001"
assert_rc 1 "[G4] iter001 is rejected too"
assert_unchanged "$G_RUN" "[G4] state and log unchanged after iter001"

# iter0 parses as an iteration but previous_phase() answers "iter-1", a label
# nothing resolves. The zeroth phase is called baseline.
commit "$G_RUN" iter0 gap_analysis \
  --weak-images "$G_RUN/iter01/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter01/gaps/gap_report.json" \
  --weak-image-count 120 --summary "iter0"
assert_rc 1 "[G4] iter0 is rejected"
case "$RUN_OUT" in
  *"numbered from 1"*) ok "[G4] the rejection explains where iterations start" ;;
  *) notok "[G4] the rejection explains where iterations start" "output: $RUN_OUT" ;;
esac
assert_unchanged "$G_RUN" "[G4] state and log unchanged after iter0"

# And a padded label already on disk is reported, not counted.
run "$PY" - "$G_RUN" <<'PYEOF'
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
state_path = results / "deft_state.json"
state = json.loads(state_path.read_text())
state["iterations"]["iter01"] = {"stage_completed": "gap_analysis", "status": "in_progress"}
state_path.write_text(json.dumps(state, indent=2))
PYEOF
assert_rc 0 "[G4] fixture: a padded phase is planted in state"
audit_kv "$G_RUN"
assert_eq INVALID "$AUDIT_STATUS" "[G4] a padded phase on disk makes the run INVALID"

section "G5. a vanished artifact does not veto recording what happens next"

g_init ws_vanished_artifact 2
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G5
commit "$G_RUN" iter1 gap_analysis \
  --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "gap_analysis: 120 weak images" --duration-sec 91
commit "$G_RUN" iter1 embed \
  --embeddings-parquet "$G_RUN/iter1/embeddings/weak_images_embeddings.parquet" \
  --summary "embedded 120 weak images" --duration-sec 45
commit "$G_RUN" iter1 mine \
  --mining-output "$G_RUN/iter1/mining/final_unique_files.parquet" \
  --mining-summary "$G_RUN/iter1/mining/summary.json" \
  --summary "mined 360 images" --duration-sec 612
commit "$G_RUN" iter1 stage \
  --odvg "$G_RUN/iter1/tmm/annotations/tmm_odvg.jsonl" \
  --label-map "$G_RUN/iter1/tmm/annotations/labelmap.json" \
  --staged-images-dir "$G_RUN/iter1/tmm/images" \
  --exclude-parquet "$G_RUN/iter1/mined_cumulative.parquet" \
  --summary "staged 360 images" --duration-sec 73
commit "$G_RUN" iter1 train \
  --checkpoint "$G_RUN/iter1/train/gdino_model_latest.pth" \
  --training-spec "$G_RUN/iter1/train_grounding_dino.yaml" \
  --summary "trained iter1" --duration-sec 4820
assert_rc 0 "[G5] the run is committed through iter1/train"

# The scratch cleanup every full disk invites: thousands of staged JPEGs, deleted
# after the stage that consumed them already committed.
rm -rf "$G_RUN/iter1/tmm/images"
audit_kv "$G_RUN"
assert_eq INVALID "$AUDIT_STATUS" "[G5] the run goes INVALID when the staged images vanish"

commit "$G_RUN" iter1 inference \
  --inference-labels-dir "$G_RUN/iter1/inference/labels" \
  --summary "inference: 1 label file" --duration-sec 140
assert_rc 0 "[G5] the next stage is still recordable"
case "$RUN_OUT" in
  *"already inconsistent before this commit"*) ok "[G5] the inherited fault is reported" ;;
  *) notok "[G5] the inherited fault is reported" "output: $RUN_OUT" ;;
esac

commit "$G_RUN" iter1 kpi_analyze --status error \
  --summary "kpi_analyze failed: analytics exited 1" --duration-sec 12
assert_rc 0 "[G5] the hard stop is still recordable"
commit "$G_RUN" iter1 loop_stop \
  --summary "halted: the staged image directory is gone" --duration-sec 1
assert_rc 0 "[G5] loop_stop is still recordable"
assert_eq true "$(report_field "$G_RUN" loop_stop_committed)" \
  "[G5] the run has a terminal record"
run "$PY" -c "
import json, sys
state = json.load(open(sys.argv[1]))
assert state['status'] == 'failed', state['status']
assert state.get('stopped_at'), 'stopped_at was never recorded'
" "$G_RUN/deft_state.json"
assert_rc 0 "[G5] the finalized run records status=failed and stopped_at"

section "G6. two overlapping commits never erase one another"

G6_COMMITTED=0
G6_EVENTS=0
for round in 1 2 3 4 5; do
  g_init "ws_race_$round" 2
  make_iter_artifacts "$G_RUN" iter1
  g_baseline "$G_RUN" "G6-$round" >/dev/null 2>&1
  for worker in 1 2; do
    (
      "$PY" "$COMMIT" --results-dir "$G_RUN" --iter-label iter1 --stage gap_analysis \
        --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
        --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
        --weak-image-count 120 --summary "gap_analysis (worker $worker)" \
        --duration-sec 9 >"$WORK/race.$round.$worker.out" 2>&1
    ) &
  done
  wait
  G6_COMMITTED=$((G6_COMMITTED + $(grep -l '^committed ' "$WORK/race.$round".*.out 2>/dev/null | wc -l)))
  G6_EVENTS=$((G6_EVENTS + $(grep -c '"stage": "gap_analysis"' "$G_RUN/loop_log.jsonl")))
  audit_kv "$G_RUN"
  [ "$AUDIT_STATUS" = VALID ] || notok "[G6] round $round leaves the run VALID" \
    "status: $AUDIT_STATUS"
done
assert_eq 5 "$G6_COMMITTED" "[G6] exactly one of each pair reports 'committed'"
assert_eq 5 "$G6_EVENTS" "[G6] exactly one gap_analysis event survives per round"
assert_eq gap_analysis "$(report_field "$G_RUN" stage_completed_by_phase | \
  "$PY" -c 'import json,sys; print(json.load(sys.stdin)["iter1"])')" \
  "[G6] state still records the stage the winner committed"

section "G7. a commit interrupted mid-write is recovered, not wedged"

cat >"$WORK/crash_in_window.py" <<'PYEOF'
"""Run the real commit_stage.main(), but SIGKILL between the two renames — the
window where deft_state.json is replaced and loop_log.jsonl is not."""
import os
import signal
import sys

sys.path.insert(0, os.environ["DEFT_TEST_SCRIPTS"])
import commit_stage  # noqa: E402

real_write = commit_stage.write_state_atomic


def state_then_die(results_dir, state):
    real_write(results_dir, state)
    os.kill(os.getpid(), signal.SIGKILL)


commit_stage.write_state_atomic = state_then_die
raise SystemExit(commit_stage.main())
PYEOF

g_init ws_torn_write 2
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G7
DEFT_TEST_SCRIPTS="$SCRIPTS_DIR" run "$PY" "$WORK/crash_in_window.py" \
  --results-dir "$G_RUN" --iter-label iter1 --stage gap_analysis \
  --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "gap_analysis killed mid-write" --duration-sec 9
assert_rc 137 "[G7] the commit is killed between the state and log writes"
[ -f "$G_RUN/.deft_commit.journal" ] && ok "[G7] the journal survives the kill" \
  || notok "[G7] the journal survives the kill"

# A rollback must not be blockable by something squatting on its scratch path.
mkdir -p "$G_RUN/loop_log.jsonl.rollback.tmp"
commit "$G_RUN" iter1 gap_analysis \
  --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "gap_analysis: 120 weak images" --duration-sec 91
assert_rc 0 "[G7] the next commit recovers the interrupted one and succeeds"
case "$RUN_OUT" in
  *"recovered an interrupted commit"*) ok "[G7] the recovery is reported" ;;
  *) notok "[G7] the recovery is reported" "output: $RUN_OUT" ;;
esac
rmdir "$G_RUN/loop_log.jsonl.rollback.tmp"
[ -e "$G_RUN/.deft_commit.journal" ] \
  && notok "[G7] the journal is cleared once the commit lands" \
  || ok "[G7] the journal is cleared once the commit lands"
expect_audit "$G_RUN" iter1/gap_analysis embed references/tao-generate-image-embeddings.md
run "$PY" -c "
import json, sys
events = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
gaps = [e for e in events if e['stage'] == 'gap_analysis']
assert len(gaps) == 1, gaps
assert [e['seq'] for e in events] == list(range(1, len(events) + 1)), events
" "$G_RUN/loop_log.jsonl"
assert_rc 0 "[G7] the recovered run has exactly one gap_analysis event, seq intact"

section "G8. a non-finite number never reaches deft_state.json"

g_init ws_non_finite 2
g_baseline "$G_RUN" G8
make_iter_artifacts "$G_RUN" iter1
freeze "$G_RUN"
# "=" form for -inf: argparse reads a leading dash as another flag otherwise.
for value in nan inf infinity -inf; do
  commit "$G_RUN" iter1 gap_analysis \
    --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
    --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
    --weak-image-count 120 "--map-value=$value" --summary "mAP=$value"
  assert_rc 1 "[G8] --map-value $value is rejected"
done
case "$RUN_OUT" in
  *"finite"*) ok "[G8] the rejection explains what to do instead" ;;
  *) notok "[G8] the rejection explains what to do instead" "output: $RUN_OUT" ;;
esac
assert_unchanged "$G_RUN" "[G8] state and log unchanged after the non-finite commits"
run "$PY" -c "
import json, sys
def reject(literal):
    raise ValueError(f'non-RFC-8259 literal: {literal}')
json.load(open(sys.argv[1]), parse_constant=reject)
" "$G_RUN/deft_state.json"
assert_rc 0 "[G8] deft_state.json parses under strict RFC-8259 rules"

init_run "$G_WS" "$G_WS/results/run_non_finite_lr" 2 --learning-rate inf
assert_rc 1 "[G8] init_deft_state.py rejects a non-finite --learning-rate"

section "G9. gap_analysis has to record the number that ends the loop"

g_init ws_gap_count_required 3
g_baseline "$G_RUN" G9
make_iter_artifacts "$G_RUN" iter1
freeze "$G_RUN"
commit "$G_RUN" iter1 gap_analysis \
  --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
  --summary "gap_analysis: 0 weak images across 0 classes"
assert_rc 1 "[G9] gap_analysis without a weak-image count is rejected"
case "$RUN_OUT" in
  *"--weak-image-count"*) ok "[G9] the rejection names the missing flag" ;;
  *) notok "[G9] the rejection names the missing flag" "output: $RUN_OUT" ;;
esac
assert_unchanged "$G_RUN" "[G9] state and log unchanged after the countless commit"

commit "$G_RUN" iter1 gap_analysis \
  --weak-images "$G_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 3 --zero-weak-images \
  --summary "gap_analysis claiming both 3 and 0 weak images"
assert_rc 1 "[G9] a count that contradicts --zero-weak-images is rejected"
assert_unchanged "$G_RUN" "[G9] state and log unchanged after the contradictory commit"

# Every overlay's copy-paste commit block must carry the flag it now requires.
if grep -q -- "--weak-image-count" "$SKILL_DIR/references/tao-analyze-gaps-od-map.md"; then
  ok "[G9] the gap_analysis overlay documents --weak-image-count"
else
  notok "[G9] the gap_analysis overlay documents --weak-image-count" \
    "references/tao-analyze-gaps-od-map.md never mentions the flag its commit block needs"
fi

section "G10. the weak-image count belongs to gap_analysis and nothing else"

g_init ws_count_scope 3
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G10
commit_iteration "$G_RUN" iter1 gap_analysis references/tao-analyze-gaps-od-map.md
freeze "$G_RUN"
commit "$G_RUN" iter1 loop_stop --weak-image-count 0 \
  --summary "out of budget, dressed up as the documented early stop"
assert_rc 1 "[G10] --weak-image-count on loop_stop is rejected"
case "$RUN_OUT" in
  *"belong"*gap_analysis*) ok "[G10] the rejection names the owning stage" ;;
  *) notok "[G10] the rejection names the owning stage" "output: $RUN_OUT" ;;
esac
assert_unchanged "$G_RUN" "[G10] state and log unchanged after the loop_stop count"
run "$PY" -c "
import json, sys
entry = json.load(open(sys.argv[1]))['iterations']['iter1']
assert entry['weak_image_count'] == 120, entry
" "$G_RUN/deft_state.json"
assert_rc 0 "[G10] gap_analysis's real count survives"

# The same value smuggled in as an undeclared extra, on the stage before it.
commit "$G_RUN" iter1 loop_stop --zero_weak_images true \
  --summary "the same claim as an undeclared flag"
assert_rc 1 "[G10] an undeclared --zero_weak_images is rejected"
assert_unchanged "$G_RUN" "[G10] state and log unchanged after the undeclared extra"

commit "$G_RUN" iter1 loop_stop --summary "stopped after iter1 of 3" --duration-sec 1
assert_rc 0 "[G10] loop_stop without the count is accepted"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-complete
assert_rc 1 "[G10] --require-complete still rejects 1 of 3 iterations"

section "G11. loop_stop never writes a completion the audit withholds"

run "$PY" -c "
import json, sys
print(json.load(open(sys.argv[1]))['status'])
" "$G_RUN/deft_state.json"
assert_eq stopped "$RUN_OUT" "[G11] an early loop_stop records status=stopped"
assert_eq false "$(report_field "$G_RUN" complete)" "[G11] the audit agrees it is not complete"
noise=$("$PY" "$AUDIT" --results-dir "$G_RUN" 2>&1 >/dev/null)
assert_eq "" "$noise" "[G11] a truthful stopped run audits silently"

# And the disagreement the audit used to leave unflagged is now a warning.
run "$PY" - "$G_RUN" <<'PYEOF'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "deft_state.json"
state = json.loads(path.read_text())
state["status"] = "complete"
path.write_text(json.dumps(state, indent=2))
PYEOF
assert_rc 0 "[G11] fixture: state.status is forced to complete"
noise=$("$PY" "$AUDIT" --results-dir "$G_RUN" 2>&1 >/dev/null)
case "$noise" in
  *"withholds completion"*) ok "[G11] the audit flags a state that claims completion" ;;
  *) notok "[G11] the audit flags a state that claims completion" "stderr: ${noise:-<empty>}" ;;
esac

section "G12. a run-level 'failed' with nothing failing in the log is flagged"

g_init ws_status_only_failure 2
g_baseline "$G_RUN" G12
run "$PY" - "$G_RUN" <<'PYEOF'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "deft_state.json"
state = json.loads(path.read_text())
state["status"] = "failed"
path.write_text(json.dumps(state, indent=2))
PYEOF
assert_rc 0 "[G12] fixture: state.status is forced to failed"
audit_kv "$G_RUN"
assert_eq false "$AUDIT_TERMINAL" "[G12] a state-only failure is not terminal"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-terminal
assert_rc 1 "[G12] --require-terminal rejects it"
noise=$("$PY" "$AUDIT" --results-dir "$G_RUN" 2>&1 >/dev/null)
case "$noise" in
  *"no loop_log event has status=error"*) ok "[G12] the audit warns that nothing recorded a failure" ;;
  *) notok "[G12] the audit warns that nothing recorded a failure" "stderr: ${noise:-<empty>}" ;;
esac

section "G13. the optional recorded paths are re-checked like the required ones"

g_init ws_extra_paths 1
make_iter_artifacts "$G_RUN" iter1
g_baseline "$G_RUN" G13
commit_iteration "$G_RUN" iter1 loop_stop none
commit "$G_RUN" iter1 loop_stop --summary "loop complete: 1 iteration" --duration-sec 1
assert_rc 0 "[G13] commit iter1/loop_stop"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-complete
assert_rc 0 "[G13] the finished run is complete"

# --kpi-log is where mAP is printed; a recorded path that is gone is as
# misleading as a missing checkpoint, which the audit has always caught.
rm -f "$G_RUN/iter1/kpi/kpi_analyze.log"
audit_kv "$G_RUN"
assert_eq INVALID "$AUDIT_STATUS" "[G13] a deleted kpi_log invalidates the run"
run "$PY" "$AUDIT" --results-dir "$G_RUN" --require-complete
assert_rc 1 "[G13] --require-complete rejects it"
case "$RUN_OUT" in
  *"kpi_log"*) ok "[G13] the error names the missing optional path" ;;
  *) notok "[G13] the error names the missing optional path" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G14 — default AP50 thresholds
#
# The thresholds are the one knob whose wrong value fails silently rather than
# loudly: too loose a gate marks no image weak, the mining budget lands at zero,
# and the iteration trains on nothing new while still reporting success. So the
# defaults have to be both present and correct, and every target class must come
# out gated no matter which flags the caller omitted.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G14 default AP50 thresholds"

# 1. Neither flag given: derive the classes from the KPI mapping, which names what
#    the run is scored on. Falling back to the reference ITS set would run a
#    bicycle/car/person loop against an unrelated dataset. new_workspace's mapping
#    declares `car` alone.
G14_WS=$(new_workspace g14a); make_pool "$G14_WS"
G14_RUN="$G14_WS/results"
OMIT_AP50=1 init_run "$G14_WS" "$G14_RUN" 2
assert_rc 0 "[G14] init succeeds with no --ap50-thresholds-json"
assert_eq '{"car": 0.99}' \
  "$(state_json "$G14_RUN" ap50_thresholds)" \
  "[G14] gates only the classes the KPI mapping names"
assert_eq '["car"]' "$(state_json "$G14_RUN" target_classes)" \
  "[G14] target classes are derived from the KPI mapping, not the reference set"
case "$RUN_OUT" in
  *"derived"*) ok "[G14] the derivation is reported" ;;
  *) notok "[G14] the derivation is reported" "output: $RUN_OUT" ;;
esac
case "$RUN_OUT" in
  *"not given; defaulted to"*) ok "[G14] the summary says the value was defaulted" ;;
  *) notok "[G14] the summary says the value was defaulted" "output: $RUN_OUT" ;;
esac

# 2. Target classes given, thresholds not: gate the classes the caller asked for,
#    not the reference's. road_sign is the real case — it is in the ITS ground
#    truth but the reference never gated it.
G14B_WS=$(new_workspace g14b); make_pool "$G14B_WS"
G14B_RUN="$G14B_WS/results"
OMIT_AP50=1 init_run "$G14B_WS" "$G14B_RUN" 2 --target-classes car,bicycle,person,road_sign
assert_rc 0 "[G14] a target class outside the reference set still initializes"
assert_eq '{"bicycle": 0.7, "car": 0.99, "person": 0.7, "road_sign": 0.7}' \
  "$(state_json "$G14B_RUN" ap50_thresholds)" \
  "[G14] an unknown target class is gated at the fallback"
case "$RUN_OUT" in
  *"road_sign"*"by assumption"*) ok "[G14] the assumed gate is named, not buried" ;;
  *) notok "[G14] the assumed gate is named, not buried" "output: $RUN_OUT" ;;
esac
# The defaults must never trip the "target class has no threshold" error, which
# would make omitting the flag a hard failure instead of a convenience.
case "$RUN_OUT" in
  *"have no AP50 threshold"*)
    notok "[G14] defaulting never leaves a target class ungated" "output: $RUN_OUT" ;;
  *) ok "[G14] defaulting never leaves a target class ungated" ;;
esac

# 3. An explicit value still wins — the default must not override the caller.
G14C_WS=$(new_workspace g14c); make_pool "$G14C_WS"
G14C_RUN="$G14C_WS/results"
init_run "$G14C_WS" "$G14C_RUN" 2
assert_rc 0 "[G14] explicit thresholds still accepted"
assert_eq '{"car": 0.9, "person": 0.85}' "$(state_json "$G14C_RUN" ap50_thresholds)" \
  "[G14] an explicit value is not overwritten by the default"
case "$RUN_OUT" in
  *"not given; defaulted to"*)
    notok "[G14] no defaulting notice when the caller supplied a value" "output: $RUN_OUT" ;;
  *) ok "[G14] no defaulting notice when the caller supplied a value" ;;
esac

# 4. multiplier carries the reference default too, so neither knob needs the user.
assert_eq '3' "$(state_json "$G14C_RUN" multiplier)" \
  "[G14] multiplier defaults to the reference value"

# ═══════════════════════════════════════════════════════════════════════════
# G15 — one pool directory, a default train spec, and pool-derived rare classes
#
# Each of these exists so the launch prompt can stay short. The rare-class one
# also has teeth: stratified allocation with the wrong rare set quietly spends the
# budget on classes that did not need it.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G15 pool-dir, default spec, derived rare classes"

init_bare() {  # init_bare WORKSPACE RESULTS [extra args...]
  local ws=$1 results=$2; shift 2
  run "$PY" "$INIT" \
    --results-dir "$results" --workspace "$ws" --max-iterations 1 \
    --num-epochs 1 --learning-rate 0.0001 \
    --zero-shot-checkpoint "$ws/ckpt/gdino_zero_shot.pth" \
    --embedding-model-path "$ws/encoder/siglip" \
    --kpi-images-dir "$ws/kpi/sequence_a/images" \
    --ground-truth-labels-dir "$ws/kpi/labels" \
    --class-mapping "$ws/classes/classes_its.yaml" \
    "$@"
}

G15_WS=$(new_workspace g15); make_pool "$G15_WS"
G15_POOL="$G15_WS/source_pool"
make_file "$G15_POOL/coco.json" '{"images": [], "annotations": [], "categories": []}'
make_file "$G15_POOL/pool_report.json" \
  '{"annotations_by_class": {"car": 72292, "road_sign": 75298, "person": 4841, "bicycle": 1330}}'
G15_RUN="$G15_WS/results/run_g15"

# 1. one directory stands in for four paths, and no train-spec template is needed
init_bare "$G15_WS" "$G15_RUN" --pool-dir "$G15_POOL" \
  --target-classes car,road_sign,person,bicycle
assert_rc 0 "[G15] --pool-dir alone satisfies the pool inputs"
case "$(state_json "$G15_RUN" source_pool_embeddings)" in
  *source_embeddings.parquet*) ok "[G15] embeddings path derived from --pool-dir" ;;
  *) notok "[G15] embeddings path derived from --pool-dir" \
       "got: $(state_json "$G15_RUN" source_pool_embeddings)" ;;
esac
case "$(state_json "$G15_RUN" train_spec_template)" in
  *assets/train_grounding_dino.yaml*) ok "[G15] train spec defaults to the shipped asset" ;;
  *) notok "[G15] train spec defaults to the shipped asset" \
       "got: $(state_json "$G15_RUN" train_spec_template)" ;;
esac

# 2. class_stratified with no rare list derives it from the pool's own counts.
#    bicycle (1330) and person (4841) sit below the mean; car and road_sign do not.
init_bare "$G15_WS" "$G15_RUN" --force --pool-dir "$G15_POOL" \
  --target-classes car,road_sign,person,bicycle \
  --allocation-policy class_stratified \
  --source-detection-file "$G15_POOL/coco.json" \
  --target-detection-file "$G15_POOL/coco.json"
assert_rc 0 "[G15] class_stratified needs no explicit rare-class list"
assert_eq '"bicycle,person"' "$(state_json "$G15_RUN" rare_class_list)" \
  "[G15] rare classes are the ones the pool holds fewest of"
case "$RUN_OUT" in
  *"derived"*"pool's own class counts"*) ok "[G15] the derivation is reported, not silent" ;;
  *) notok "[G15] the derivation is reported, not silent" "output: $RUN_OUT" ;;
esac

# 3. an explicit list still wins over the derivation
init_bare "$G15_WS" "$G15_RUN" --force --pool-dir "$G15_POOL" \
  --target-classes car,road_sign,person,bicycle \
  --allocation-policy class_stratified --rare-class-list bicycle \
  --source-detection-file "$G15_POOL/coco.json" \
  --target-detection-file "$G15_POOL/coco.json"
assert_rc 0 "[G15] an explicit rare-class list is accepted"
assert_eq '"bicycle"' "$(state_json "$G15_RUN" rare_class_list)" \
  "[G15] the explicit list is not overwritten by the derivation"

# 4. a pool missing a target class is still refused, derivation or not
init_bare "$G15_WS" "$G15_RUN" --force --pool-dir "$G15_POOL" \
  --target-classes car,truck
assert_rc 1 "[G15] a target class absent from the pool is rejected"
case "$RUN_OUT" in
  *"holds no annotations for target class"*"truck"*)
    ok "[G15] the error names the class the pool cannot supply" ;;
  *) notok "[G15] the error names the class the pool cannot supply" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G16 — an exhausted source pool is a documented early stop, not an abandoned run
#
# The miner raises rather than returning a short result, so a pool that cannot
# fill the budget ends the run. Only a recorded assertion separates that from a
# run someone walked away from.
#
# Each variant is a full run: commit_stage.py pins state.results_dir, so a copied
# run directory is correctly rejected and cannot stand in for one.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G16 pool exhaustion as a terminal state"

# pool_exhausted_run LABEL EXPECT_RC [loop_stop flags...]
pool_exhausted_run() {
  local label=$1 expect_rc=$2; shift 2
  local ws results
  ws=$(new_workspace "g16_${label}"); make_pool "$ws"
  results="$ws/results/run_g16_${label}"
  init_run "$ws" "$results" 2

  make_phase_artifacts "$results" baseline
  commit "$results" baseline inference \
    --inference-labels-dir "$results/baseline/inference/labels" \
    --summary "inference" --duration-sec 120
  commit "$results" baseline kpi_analyze \
    --kpi-csv "$results/baseline/kpi/kpi_calc.csv" --map-value 0.76 \
    --summary "kpi: mAP=0.76" --duration-sec 60

  # Iteration 1 mines the entire pool and completes every stage.
  make_iter_artifacts "$results" iter1
  commit "$results" iter1 gap_analysis \
    --weak-images "$results/iter1/gaps/weak_images.parquet" \
    --gap-report "$results/iter1/gaps/gap_report.json" \
    --weak-image-count 10891 --summary "10891 weak" --duration-sec 133
  commit "$results" iter1 embed \
    --embeddings-parquet "$results/iter1/embeddings/weak_images_embeddings.parquet" \
    --summary "embedded" --duration-sec 229
  commit "$results" iter1 mine \
    --mining-output "$results/iter1/mining/final_unique_files.parquet" \
    --mining-summary "$results/iter1/mining/summary.json" \
    --summary "mined 4941 of 5000" --duration-sec 35
  commit "$results" iter1 stage \
    --odvg "$results/iter1/tmm/annotations/tmm_odvg.jsonl" \
    --label-map "$results/iter1/tmm/annotations/labelmap.json" \
    --staged-images-dir "$results/iter1/tmm/images" \
    --exclude-parquet "$results/iter1/mined_cumulative.parquet" \
    --summary "staged 4941" --duration-sec 4
  commit "$results" iter1 train \
    --checkpoint "$results/iter1/train/gdino_model_latest.pth" \
    --training-spec "$results/iter1/train_grounding_dino.yaml" \
    --summary "trained" --duration-sec 1272
  commit "$results" iter1 inference \
    --inference-labels-dir "$results/iter1/inference/labels" \
    --summary "inference" --duration-sec 1535
  commit "$results" iter1 kpi_analyze \
    --kpi-csv "$results/iter1/kpi/kpi_calc.csv" --map-value 0.78082 \
    --summary "kpi: mAP=0.78082" --duration-sec 1319
  assert_rc 0 "[G16/$label] iteration 1 completed every stage"

  # Iteration 2 finds weak images but the pool cannot fill another budget.
  make_iter_artifacts "$results" iter2
  commit "$results" iter2 gap_analysis \
    --weak-images "$results/iter2/gaps/weak_images.parquet" \
    --gap-report "$results/iter2/gaps/gap_report.json" \
    --weak-image-count 10362 --summary "10362 weak" --duration-sec 153
  assert_rc 0 "[G16/$label] iteration 2 reached gap_analysis"

  commit "$results" iter2 loop_stop "$@" --summary "stop: $label" --duration-sec 1
  assert_rc 0 "[G16/$label] loop_stop committed"
  run "$PY" "$AUDIT" --results-dir "$results" --require-complete
  assert_rc "$expect_rc" "[G16/$label] --require-complete exits $expect_rc"
  G16_OUT=$RUN_OUT
}

# 1. The real 5k case: 59 images left, far below the budget. Exhaustion is asserted
#    by the flag; the count stays honest.
pool_exhausted_run exhausted 0 --pool-exhausted --pool-remaining 59
case "$G16_OUT" in
  *"source pool was exhausted"*) ok "[G16] the reason names pool exhaustion" ;;
  *) notok "[G16] the reason names pool exhaustion" "output: $G16_OUT" ;;
esac
case "$G16_OUT" in
  *"pool_remaining=59"*) ok "[G16] the honest count is reported, not forced to 0" ;;
  *) notok "[G16] the honest count is reported, not forced to 0" "output: $G16_OUT" ;;
esac

# 2. A literally empty pool, the earlier spelling, still records the stop.
pool_exhausted_run zero 0 --pool-remaining 0

# 3. Without either, an early stop is indistinguishable from abandonment.
pool_exhausted_run bare 1

# Both flags belong to loop_stop, like the weak-image count belongs to gap_analysis.
G16_WS=$(new_workspace g16_stage); make_pool "$G16_WS"
G16_RUN="$G16_WS/results/run_g16_stage"
init_run "$G16_WS" "$G16_RUN" 2
make_phase_artifacts "$G16_RUN" baseline
commit "$G16_RUN" baseline inference \
  --inference-labels-dir "$G16_RUN/baseline/inference/labels" \
  --summary "inference" --duration-sec 1
commit "$G16_RUN" baseline kpi_analyze \
  --kpi-csv "$G16_RUN/baseline/kpi/kpi_calc.csv" --map-value 0.5 \
  --summary "kpi" --duration-sec 1
make_iter_artifacts "$G16_RUN" iter1
commit "$G16_RUN" iter1 gap_analysis \
  --weak-images "$G16_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G16_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 5 --pool-remaining 0 --summary "wrong stage" --duration-sec 1
assert_rc 1 "[G16] --pool-remaining is rejected on a stage that is not loop_stop"
case "$RUN_OUT" in
  *"belongs to loop_stop"*) ok "[G16] the rejection names loop_stop" ;;
  *) notok "[G16] the rejection names loop_stop" "output: $RUN_OUT" ;;
esac
commit "$G16_RUN" iter1 gap_analysis \
  --weak-images "$G16_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G16_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 5 --pool-exhausted --summary "wrong stage" --duration-sec 1
assert_rc 1 "[G16] --pool-exhausted is rejected on a stage that is not loop_stop"

# ═══════════════════════════════════════════════════════════════════════════
# G17 — init and prep must not be each other's precondition
#
# --source-detection-file is source_pool/coco.json, which prep produces. Requiring
# it at init makes a class_stratified run from raw images unstartable.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G17 init before prep on an unprepared pool"

G17_WS=$(new_workspace g17)
mkdir -p "$G17_WS/pool_images" "$G17_WS/ckpt" "$G17_WS/encoder/siglip" "$G17_WS/kpi/labels" "$G17_WS/classes"
make_file "$G17_WS/pool_images/a.jpg" "x"
make_file "$G17_WS/ckpt/gdino_zero_shot.pth" "x"
make_file "$G17_WS/ckpt/codetr.pth" "x"
make_file "$G17_WS/ckpt/coco80.txt" "person"
make_file "$G17_WS/encoder/siglip/config.json" '{}'
mkdir -p "$G17_WS/kpi/sequence_a/images"
make_file "$G17_WS/kpi/labels/a.txt" "car 0 0 0 1 1 2 2 0 0 0 0 0 0 0"
make_file "$G17_WS/classes/classes_its.yaml" 'car: ["car"]'
make_file "$G17_WS/kpi/target.json" '{"images": [], "annotations": [], "categories": []}'
make_file "$G17_WS/pool_report.json" '{"annotations_by_class": {"car": 120}}'
G17_RUN="$G17_WS/results/run_g17"

run "$PY" "$INIT" \
  --results-dir "$G17_RUN" --workspace "$G17_WS" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G17_WS/ckpt/gdino_zero_shot.pth" \
  --embedding-model-path "$G17_WS/encoder/siglip" \
  --kpi-images-dir "$G17_WS/kpi/sequence_a/images" \
  --ground-truth-labels-dir "$G17_WS/kpi/labels" \
  --class-mapping "$G17_WS/classes/classes_its.yaml" \
  --allocation-policy class_stratified --rare-class-list car \
  --pool-report "$G17_WS/pool_report.json" \
  --target-detection-file "$G17_WS/kpi/target.json" \
  --source-detection-file "$G17_WS/source_pool/coco.json" \
  --source-pool-annotations "$G17_WS/source_pool/odvg" \
  --source-pool-embeddings "$G17_WS/source_pool/source_embeddings.parquet" \
  --pool-images "$G17_WS/pool_images" \
  --codetr-checkpoint "$G17_WS/ckpt/codetr.pth" \
  --codetr-classmap "$G17_WS/ckpt/coco80.txt"
assert_rc 0 "[G17] init succeeds when prep will produce the detection file"
case "$RUN_OUT" in
  *"prep\` runs first and produces it"*) ok "[G17] the absent detection file is a warning, not an error" ;;
  *) notok "[G17] the absent detection file is a warning, not an error" "output: $RUN_OUT" ;;
esac

# Without prep's inputs there is nothing to produce it, so it stays an error.
G17_WS2=$(new_workspace g17_noinputs)
mkdir -p "$G17_WS2/ckpt" "$G17_WS2/encoder/siglip" "$G17_WS2/kpi/labels" "$G17_WS2/classes" "$G17_WS2/kpi/sequence_a/images"
make_file "$G17_WS2/ckpt/gdino_zero_shot.pth" "x"
make_file "$G17_WS2/encoder/siglip/config.json" '{}'
make_file "$G17_WS2/kpi/labels/a.txt" "car 0 0 0 1 1 2 2 0 0 0 0 0 0 0"
make_file "$G17_WS2/classes/classes_its.yaml" 'car: ["car"]'
make_file "$G17_WS2/kpi/target.json" '{"images": [], "annotations": [], "categories": []}'
make_file "$G17_WS2/pool_report.json" '{"annotations_by_class": {"car": 120}}'
run "$PY" "$INIT" \
  --results-dir "$G17_WS2/results/run_g17b" --workspace "$G17_WS2" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G17_WS2/ckpt/gdino_zero_shot.pth" \
  --embedding-model-path "$G17_WS2/encoder/siglip" \
  --kpi-images-dir "$G17_WS2/kpi/sequence_a/images" \
  --ground-truth-labels-dir "$G17_WS2/kpi/labels" \
  --class-mapping "$G17_WS2/classes/classes_its.yaml" \
  --allocation-policy class_stratified --rare-class-list car \
  --pool-report "$G17_WS2/pool_report.json" \
  --target-detection-file "$G17_WS2/kpi/target.json" \
  --source-detection-file "$G17_WS2/source_pool/coco.json" \
  --source-pool-annotations "$G17_WS2/source_pool/odvg" \
  --source-pool-embeddings "$G17_WS2/source_pool/source_embeddings.parquet"
assert_rc 1 "[G17] with no prep inputs the missing detection file is still an error"

# class_stratified on an ALREADY-PREPARED pool, with no pool report: nothing proves
# the pool holds the requested classes, so mining would return neighbours of
# something else.
G17_WS3=$(new_workspace g17_prepared); make_pool "$G17_WS3"
make_file "$G17_WS3/kpi/target.json" '{"images": [], "annotations": [], "categories": []}'
run "$PY" "$INIT" \
  --results-dir "$G17_WS3/results/run_g17c" --workspace "$G17_WS3" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G17_WS3/ckpt/gdino_zero_shot.pth" \
  --train-spec-template "$G17_WS3/specs/train_grounding_dino.yaml" \
  --embedding-model-path "$G17_WS3/encoder/siglip" \
  --kpi-images-dir "$G17_WS3/kpi/sequence_a/images" \
  --ground-truth-labels-dir "$G17_WS3/kpi/labels" \
  --class-mapping "$G17_WS3/classes/classes_its.yaml" \
  --allocation-policy class_stratified --rare-class-list car \
  --target-detection-file "$G17_WS3/kpi/target.json" \
  --source-detection-file "$G17_WS3/kpi/target.json" \
  --source-pool-annotations "$G17_WS3/source_pool/odvg" \
  --source-pool-embeddings "$G17_WS3/source_pool/source_embeddings.parquet"
assert_rc 1 "[G17] class_stratified on a prepared pool without --pool-report is rejected"
case "$RUN_OUT" in
  *pool-report*) ok "[G17] the rejection names --pool-report" ;;
  *) notok "[G17] the rejection names --pool-report" "output: $RUN_OUT" ;;
esac

# ...but a run that still has to prep must not be blocked by it: pool_report.json is
# prep's own output, so requiring it there recreates the init/prep deadlock.
run "$PY" "$INIT" \
  --results-dir "$G17_WS/results/run_g17d" --workspace "$G17_WS" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G17_WS/ckpt/gdino_zero_shot.pth" \
  --embedding-model-path "$G17_WS/encoder/siglip" \
  --kpi-images-dir "$G17_WS/kpi/sequence_a/images" \
  --ground-truth-labels-dir "$G17_WS/kpi/labels" \
  --class-mapping "$G17_WS/classes/classes_its.yaml" \
  --allocation-policy class_stratified --rare-class-list car \
  --target-detection-file "$G17_WS/kpi/target.json" \
  --source-detection-file "$G17_WS/source_pool/coco.json" \
  --source-pool-annotations "$G17_WS/source_pool/odvg" \
  --source-pool-embeddings "$G17_WS/source_pool/source_embeddings.parquet" \
  --pool-images "$G17_WS/pool_images" \
  --codetr-checkpoint "$G17_WS/ckpt/codetr.pth" \
  --codetr-classmap "$G17_WS/ckpt/coco80.txt"
assert_rc 0 "[G17] a prep run is not blocked by the --pool-report requirement"
case "$RUN_OUT" in
  *"prep\` runs first and produces it"*) ok "[G17] the pool report is deferred to prep" ;;
  *) notok "[G17] the pool report is deferred to prep" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G18 — an artifact that exists but holds nothing is not a completed stage
#
# A crashed container leaves a zero-byte csv, an empty label directory, or a
# truncated parquet. Committing those as success moves the failure several stages
# downstream, where it reads as a different bug.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G18 artifact validity, not just existence"

G18_WS=$(new_workspace g18); make_pool "$G18_WS"
G18_RUN="$G18_WS/results/run_g18"
init_run "$G18_WS" "$G18_RUN" 1
assert_rc 0 "[G18] init"
make_phase_artifacts "$G18_RUN" baseline

# An inference directory with no labels in it.
rm -rf "$G18_RUN/baseline/inference/labels"; mkdir -p "$G18_RUN/baseline/inference/labels"
commit "$G18_RUN" baseline inference \
  --inference-labels-dir "$G18_RUN/baseline/inference/labels" \
  --summary "empty labels dir" --duration-sec 1
assert_rc 1 "[G18] an empty inference directory is rejected"
case "$RUN_OUT" in
  *"directory is empty"*) ok "[G18] the rejection says the directory is empty" ;;
  *) notok "[G18] the rejection says the directory is empty" "output: $RUN_OUT" ;;
esac

make_file "$G18_RUN/baseline/inference/labels/000001.txt" "car 0 0 0 1 1 2 2 0 0 0 0 0 0 0 0.9"
commit "$G18_RUN" baseline inference \
  --inference-labels-dir "$G18_RUN/baseline/inference/labels" \
  --summary "one label" --duration-sec 1
assert_rc 0 "[G18] the same directory commits once it holds a label"

# A KPI csv with a header and no rows: kpi_analyze wrote nothing.
: >"$G18_RUN/baseline/kpi/kpi_calc.csv"
printf 'Sequence Name,class,AP50\n' >"$G18_RUN/baseline/kpi/kpi_calc.csv"
commit "$G18_RUN" baseline kpi_analyze \
  --kpi-csv "$G18_RUN/baseline/kpi/kpi_calc.csv" --map-value 0.4 \
  --summary "header only" --duration-sec 1
assert_rc 1 "[G18] a header-only KPI csv is rejected"
case "$RUN_OUT" in
  *"no data rows"*) ok "[G18] the rejection says there are no data rows" ;;
  *) notok "[G18] the rejection says there are no data rows" "output: $RUN_OUT" ;;
esac

# A zero-byte file passes an existence check and nothing else.
printf 'kpi,car,0.42\n' >>"$G18_RUN/baseline/kpi/kpi_calc.csv"
commit "$G18_RUN" baseline kpi_analyze \
  --kpi-csv "$G18_RUN/baseline/kpi/kpi_calc.csv" --map-value 0.4 \
  --summary "with a row" --duration-sec 1
assert_rc 0 "[G18] the same csv commits once it has a data row"

make_iter_artifacts "$G18_RUN" iter1
: >"$G18_RUN/iter1/gaps/weak_images.parquet"
commit "$G18_RUN" iter1 gap_analysis \
  --weak-images "$G18_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G18_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "empty parquet" --duration-sec 1
assert_rc 1 "[G18] a zero-byte parquet is rejected"
case "$RUN_OUT" in
  *"file is empty"*) ok "[G18] the rejection says the file is empty" ;;
  *) notok "[G18] the rejection says the file is empty" "output: $RUN_OUT" ;;
esac

# Non-empty but not a parquet: a truncated or wrong-format write.
printf 'this is not a parquet file\n' >"$G18_RUN/iter1/gaps/weak_images.parquet"
commit "$G18_RUN" iter1 gap_analysis \
  --weak-images "$G18_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G18_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "not parquet" --duration-sec 1
assert_rc 1 "[G18] a file without the parquet magic is rejected"

# Malformed JSON that a later stage would fail to parse.
make_parquet "$G18_RUN/iter1/gaps/weak_images.parquet"
printf '{"weak_images": 120' >"$G18_RUN/iter1/gaps/gap_report.json"
commit "$G18_RUN" iter1 gap_analysis \
  --weak-images "$G18_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G18_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "truncated json" --duration-sec 1
assert_rc 1 "[G18] truncated JSON is rejected"
case "$RUN_OUT" in
  *unreadable*) ok "[G18] the rejection says the file is unreadable" ;;
  *) notok "[G18] the rejection says the file is unreadable" "output: $RUN_OUT" ;;
esac

make_file "$G18_RUN/iter1/gaps/gap_report.json" '{"weak_images": 120}'
commit "$G18_RUN" iter1 gap_analysis \
  --weak-images "$G18_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G18_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 120 --summary "valid" --duration-sec 1
assert_rc 0 "[G18] the stage commits once every artifact is readable"

# ═══════════════════════════════════════════════════════════════════════════
# G19 — a pool with duplicate basenames is refused before it is labelled
#
# Pseudo-labels are written one flat file per basename, so two images of the same
# name keep one set of labels between them. Nothing downstream can recover the
# lost one, which makes the pool the only place this is repairable.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G19 duplicate pool basenames"

VPL="$SKILL_DIR/scripts/verify_pseudo_labels.py"
G19=$(new_workspace g19)
make_file "$G19/pool/a/frame_001.jpg" "A"
make_file "$G19/pool/a/frame_002.jpg" "A2"
make_file "$G19/pool/b/frame_001.jpg" "B"

run "$PY" "$VPL" --pool-images-dir "$G19/pool"
assert_rc 1 "[G19] a pool with duplicate basenames is refused"
case "$RUN_OUT" in
  *"duplicate basename"*) ok "[G19] the refusal names the duplicate" ;;
  *) notok "[G19] the refusal names the duplicate" "output: $RUN_OUT" ;;
esac
case "$RUN_OUT" in
  *frame_001.jpg*) ok "[G19] the refusal names the offending file" ;;
  *) notok "[G19] the refusal names the offending file" "output: $RUN_OUT" ;;
esac

rm -f "$G19/pool/b/frame_001.jpg"
run "$PY" "$VPL" --pool-images-dir "$G19/pool"
assert_rc 0 "[G19] the same pool passes once the duplicate is gone"

run "$PY" "$VPL"
assert_rc 1 "[G19] neither directory given is an error, not a silent pass"

# ═══════════════════════════════════════════════════════════════════════════
# G20 — the class list must agree everywhere it is written
#
# Grounding DINO labels a detection by the position of the caption it matched, so
# a short or reordered list relabels every prediction and the run still exits 0.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G20 class contract"

VCC="$SKILL_DIR/scripts/verify_class_contract.py"
G20=$(new_workspace g20)
make_file "$G20/mapping.yaml" '- bicycle: [bicycle]
- car: [car]
- person: [person]
- road_sign: [road_sign]'
make_file "$G20/labelmap.json" '{"0": "bicycle", "1": "car", "2": "person"}'
make_file "$G20/state.json" '{"config": {"target_classes": ["bicycle", "car", "person"]}}'
make_file "$G20/classes.yaml" 'bicycle: [bicycle]
car: [car]
person: [person]'

run "$PY" "$VCC" --captions '["bicycle","car","person"]' --kpi-mapping "$G20/mapping.yaml"
assert_rc 1 "[G20] a scored class missing from captions is rejected"
case "$RUN_OUT" in
  *road_sign*) ok "[G20] the rejection names the unpredictable class" ;;
  *) notok "[G20] the rejection names the unpredictable class" "output: $RUN_OUT" ;;
esac

run "$PY" "$VCC" --captions '["car","bicycle","person"]' --labelmap "$G20/labelmap.json"
assert_rc 1 "[G20] captions ordered differently from the labelmap are rejected"

run "$PY" "$VCC" --captions '["bicycle","car","person"]' \
  --state "$G20/state.json" --classes "$G20/classes.yaml" --labelmap "$G20/labelmap.json"
assert_rc 0 "[G20] agreeing sources pass"

make_file "$G20/classes_wrong.yaml" 'bicycle: [bicycle]
car: [car]
truck: [truck]'
run "$PY" "$VCC" --captions '["bicycle","car","person"]' \
  --state "$G20/state.json" --classes "$G20/classes_wrong.yaml"
assert_rc 1 "[G20] a pool folded to different classes than the run trains is rejected"

run "$PY" "$VCC" --captions '["bicycle","car","bicycle"]' --labelmap "$G20/labelmap.json"
assert_rc 1 "[G20] duplicate captions are rejected"

run "$PY" "$VCC" --captions '["bicycle","car","person"]'
assert_rc 1 "[G20] a single source is not a comparison"

# classes.yaml is documented with a `classes:` root key and is also accepted bare;
# prepare_class_mappings_for_mining_data_prep.py unwraps both, so this must too.
# Without that, a documented classes.yaml yields the single class name "classes"
# and every comparison against it is a false mismatch.
make_file "$G20/classes_wrapped.yaml" 'classes:
  bicycle: [bicycle]
  car: [car]
  person: [person]'
run "$PY" "$VCC" --captions '["bicycle","car","person"]' \
  --state "$G20/state.json" --classes "$G20/classes_wrapped.yaml"
assert_rc 0 "[G20] a classes.yaml with a classes: root key is understood"

make_file "$G20/classes_wrapped_wrong.yaml" 'classes:
  bicycle: [bicycle]
  car: [car]
  truck: [truck]'
run "$PY" "$VCC" --captions '["bicycle","car","person"]' \
  --state "$G20/state.json" --classes "$G20/classes_wrapped_wrong.yaml"
assert_rc 1 "[G20] and a wrapped file that disagrees is still rejected"

# ═══════════════════════════════════════════════════════════════════════════
# G21 — inputs outside the workspace need their own mounts, derived not guessed
#
# Containers see only "$WORKSPACE:$WORKSPACE". KPI data and checkpoints commonly
# live elsewhere, and a warning that names the problem without naming the mounts
# leaves every stage to work it out.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G21 container mounts"

G21_WS=$(new_workspace g21); make_pool "$G21_WS"
G21_OUT="$WORK/g21_outside"
make_file "$G21_OUT/kpi/images/000001.png" "png"
make_file "$G21_OUT/kpi/labels/000001.txt" "car 0.0 0 0.0 10 10 100 100 0 0 0 0 0 0 0"
G21_RUN="$G21_WS/results/run_g21"

run "$PY" "$INIT" \
  --results-dir "$G21_RUN" --workspace "$G21_WS" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G21_WS/ckpt/gdino_zero_shot.pth" \
  --train-spec-template "$G21_WS/specs/train_grounding_dino.yaml" \
  --source-pool-embeddings "$G21_WS/source_pool/source_embeddings.parquet" \
  --source-pool-annotations "$G21_WS/source_pool/odvg" \
  --embedding-model-path "$G21_WS/encoder/siglip" \
  --kpi-images-dir "$G21_OUT/kpi/images" \
  --ground-truth-labels-dir "$G21_OUT/kpi/labels" \
  --class-mapping "$G21_WS/classes/classes_its.yaml" \
  --ap50-thresholds-json '{"car": 0.9}'
assert_rc 0 "[G21] inputs outside the workspace still initialize"
case "$RUN_OUT" in
  *'-v "'*) ok "[G21] the warning gives the -v arguments to add" ;;
  *) notok "[G21] the warning gives the -v arguments to add" "output: $RUN_OUT" ;;
esac
assert_eq 1 "$("$PY" -c 'import json,sys
mounts = json.load(open(sys.argv[1]))["config"]["extra_container_mounts"]
print(1 if any(sys.argv[2] in m for m in mounts) else 0)' "$G21_RUN/deft_state.json" "$G21_OUT")" \
  "[G21] the outside path is recorded in state as a mount"

# Everything inside the workspace needs no extra mount at all.
G21B=$(new_workspace g21b); make_pool "$G21B"
init_run "$G21B" "$G21B/results/run_g21b" 1
assert_eq '[]' "$(state_json "$G21B/results/run_g21b" extra_container_mounts)" \
  "[G21] a fully self-contained run records no extra mounts"

# ═══════════════════════════════════════════════════════════════════════════
# G22 — a reused pool must prove it was built for this run
#
# Prep is idempotent by existence, and existence cannot show a pool was folded to
# the same classes. A stale pool is otherwise reused in silence.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G22 pool provenance"

G22=$(new_workspace g22); make_pool "$G22"
make_file "$G22/pool_report_stale.json" \
  '{"annotations_by_class": {"car": 120, "person": 40}, "prep_inputs": {"target_classes": "bicycle,truck"}}'
make_file "$G22/pool_report_ok.json" \
  '{"annotations_by_class": {"car": 120, "person": 40}, "prep_inputs": {"target_classes": "car,person"}}'
make_file "$G22/pool_report_bare.json" '{"annotations_by_class": {"car": 120, "person": 40}}'

init_run "$G22" "$G22/results/run_stale" 1 --pool-report "$G22/pool_report_stale.json"
assert_rc 1 "[G22] a pool folded to other classes is refused"
case "$RUN_OUT" in
  *"prepared for"*) ok "[G22] the refusal names both class sets" ;;
  *) notok "[G22] the refusal names both class sets" "output: $RUN_OUT" ;;
esac

init_run "$G22" "$G22/results/run_ok" 1 --pool-report "$G22/pool_report_ok.json"
assert_rc 0 "[G22] a pool prepared for this class set is accepted"

init_run "$G22" "$G22/results/run_bare" 1 --pool-report "$G22/pool_report_bare.json"
assert_rc 0 "[G22] a pool without provenance still initializes"
case "$RUN_OUT" in
  *"no prep_inputs"*) ok "[G22] but the missing provenance is reported" ;;
  *) notok "[G22] but the missing provenance is reported" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G23 — a pool-exhaustion claim needs evidence behind it
#
# The flag asserts a terminal state, so on its own it would let any abandoned run
# be relabelled complete. Only an iteration mines, and only a loop that finished
# one has a trend to report.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G23 false pool-exhaustion claims"

G23=$(new_workspace g23); make_pool "$G23"
G23_RUN="$G23/results/run_g23"
init_run "$G23" "$G23_RUN" 3
make_phase_artifacts "$G23_RUN" baseline
commit "$G23_RUN" baseline inference \
  --inference-labels-dir "$G23_RUN/baseline/inference/labels" --summary s --duration-sec 1
commit "$G23_RUN" baseline kpi_analyze \
  --kpi-csv "$G23_RUN/baseline/kpi/kpi_calc.csv" --map-value 0.4 --summary s --duration-sec 1
assert_rc 0 "[G23] baseline committed"

commit "$G23_RUN" baseline loop_stop --pool-exhausted --pool-remaining 999999 \
  --summary "false claim" --duration-sec 1
assert_rc 1 "[G23] exhaustion cannot be claimed on baseline, which never mines"
case "$RUN_OUT" in
  *"only an iteration mines"*) ok "[G23] the refusal says why baseline cannot claim it" ;;
  *) notok "[G23] the refusal says why baseline cannot claim it" "output: $RUN_OUT" ;;
esac

# Claimed on iteration 1 before any iteration finished: still not a completed run.
make_iter_artifacts "$G23_RUN" iter1
commit "$G23_RUN" iter1 gap_analysis \
  --weak-images "$G23_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G23_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 50 --summary s --duration-sec 1
commit "$G23_RUN" iter1 loop_stop --pool-exhausted --pool-remaining 0 \
  --summary "stopped at iter1" --duration-sec 1
assert_rc 0 "[G23] the claim is recordable on an iteration"
run "$PY" "$AUDIT" --results-dir "$G23_RUN" --require-complete
assert_rc 1 "[G23] but 0 completed iterations is not a complete run"
case "$RUN_OUT" in
  *"at least one completed iteration"*) ok "[G23] the reason states the evidence required" ;;
  *) notok "[G23] the reason states the evidence required" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G24 — a run with no baseline mAP has produced no trend
#
# --map-value is optional, because a log printing "mAP: nan" has none to record.
# Completion is not: the baseline mAP is what every iteration is compared against.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G24 completion needs a baseline mAP"

G24=$(new_workspace g24); make_pool "$G24"
G24_RUN="$G24/results/run_g24"
init_run "$G24" "$G24_RUN" 1
make_phase_artifacts "$G24_RUN" baseline
commit "$G24_RUN" baseline inference \
  --inference-labels-dir "$G24_RUN/baseline/inference/labels" --summary s --duration-sec 1
commit "$G24_RUN" baseline kpi_analyze \
  --kpi-csv "$G24_RUN/baseline/kpi/kpi_calc.csv" \
  --summary "mAP: nan — no value to record" --duration-sec 1
assert_rc 0 "[G24] kpi_analyze may commit without a map_value"

make_iter_artifacts "$G24_RUN" iter1
commit "$G24_RUN" iter1 gap_analysis \
  --weak-images "$G24_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G24_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 0 --summary "no weak images" --duration-sec 1
commit "$G24_RUN" iter1 loop_stop --summary "early stop" --duration-sec 1
assert_rc 0 "[G24] the documented early stop is committed"

run "$PY" "$AUDIT" --results-dir "$G24_RUN" --require-complete
assert_rc 1 "[G24] but with no baseline mAP the run is not complete"
case "$RUN_OUT" in
  *"no baseline mAP"*) ok "[G24] the reason names the missing baseline mAP" ;;
  *) notok "[G24] the reason names the missing baseline mAP" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G25 — rare classes are a property of the pool, so prep settles them
#
# init runs before prep, and which target classes are rare is decided by the
# pool's annotation counts. On a run that still has to prep, the list is left
# unset and the prep commit derives it. `mine` is the first stage that reads it
# and refuses to commit while it is still unset.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G25 rare classes are derived at the prep commit"

G25=$(new_workspace g25)
make_file "$G25/classes/classes_its.yaml" "car: [car]
person: [person]
bicycle: [bicycle]"
make_file "$G25/raw/img0.png" "png"
make_file "$G25/ckpt/codetr.pth" "not a real checkpoint"
make_file "$G25/ckpt/coco_classmap.txt" "person"
make_file "$G25/pool/coco_source.json" '{}'
make_file "$G25/pool/coco_target.json" '{}'
G25_RUN="$G25/results/run_g25"

init_run "$G25" "$G25_RUN" 1 \
  --ap50-thresholds-json '{"car": 0.9, "person": 0.85, "bicycle": 0.8}' \
  --pool-images "$G25/raw" \
  --codetr-checkpoint "$G25/ckpt/codetr.pth" \
  --codetr-classmap "$G25/ckpt/coco_classmap.txt" \
  --allocation-policy class_stratified \
  --source-detection-file "$G25/pool/coco_source.json" \
  --target-detection-file "$G25/pool/coco_target.json"
assert_rc 0 "[G25] class_stratified init succeeds with no rare list on a prep run"
case "$RUN_OUT" in
  *"commit_stage.py --stage prep"*)
    ok "[G25] the warning names what will derive it" ;;
  *) notok "[G25] the warning names what will derive it" "output: $RUN_OUT" ;;
esac
assert_eq 'null' "$(state_json "$G25_RUN" rare_class_list)" \
  "[G25] the list is left unset until prep has built a pool"

# Prep produces the counts that decide it: car dominates, the other two are rare.
make_pool "$G25"
make_file "$G25_RUN/prep/pool_report.json" \
  '{"annotations_by_class": {"car": 72287, "person": 4103, "bicycle": 2002}}'
commit "$G25_RUN" prep prep \
  --pool-odvg "$G25/source_pool/odvg" \
  --pool-embeddings "$G25/source_pool/source_embeddings.parquet" \
  --pool-report "$G25_RUN/prep/pool_report.json" \
  --summary "prep" --duration-sec 1
assert_rc 0 "[G25] prep commits"
assert_eq '"bicycle,person"' "$(state_json "$G25_RUN" rare_class_list)" \
  "[G25] the below-mean classes are derived from the pool's own counts"

# Without that derivation, mine cannot commit: class_stratified allocates by it.
G25B=$(new_workspace g25b); make_pool "$G25B"
G25B_RUN="$G25B/results/run_g25b"
init_run "$G25B" "$G25B_RUN" 1
"$PY" - "$G25B_RUN/deft_state.json" <<'EOF'
import json, sys
p = sys.argv[1]
s = json.load(open(p))
s["config"]["allocation_policy"] = "class_stratified"
s["config"]["rare_class_list"] = None
json.dump(s, open(p, "w"))
EOF
make_phase_artifacts "$G25B_RUN" baseline
commit "$G25B_RUN" baseline inference \
  --inference-labels-dir "$G25B_RUN/baseline/inference/labels" --summary s --duration-sec 1
commit "$G25B_RUN" baseline kpi_analyze \
  --kpi-csv "$G25B_RUN/baseline/kpi/kpi_calc.csv" --map-value 0.5 --summary s --duration-sec 1
make_iter_artifacts "$G25B_RUN" iter1
commit "$G25B_RUN" iter1 gap_analysis \
  --weak-images "$G25B_RUN/iter1/gaps/weak_images.parquet" \
  --gap-report "$G25B_RUN/iter1/gaps/gap_report.json" \
  --weak-image-count 5 --summary s --duration-sec 1
commit "$G25B_RUN" iter1 embed \
  --embeddings-parquet "$G25B_RUN/iter1/embeddings/weak_images_embeddings.parquet" \
  --summary s --duration-sec 1
commit "$G25B_RUN" iter1 mine \
  --mining-output "$G25B_RUN/iter1/mining/final_unique_files.parquet" \
  --mining-summary "$G25B_RUN/iter1/mining/summary.json" --summary s --duration-sec 1
assert_rc 1 "[G25] mine refuses while rare_class_list is still unset"
case "$RUN_OUT" in
  *"rare_class_list is unset"*)
    ok "[G25] the refusal names the unset field and how to fill it" ;;
  *) notok "[G25] the refusal names the unset field and how to fill it" "output: $RUN_OUT" ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# G26 — every path a container opens is mounted, including the ones it is handed
#
# The detection files are read by `tmm unique_neighbor_matching` from inside the
# container. The target file is the KPI detections, which live with the read-only
# dataset rather than in the per-run workspace. --extra-mount covers whatever the
# derivation cannot know about.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G26 detection files and arbitrary mounts"

G26_WS=$(new_workspace g26); make_pool "$G26_WS"
G26_OUT="$WORK/g26_dataset"
make_file "$G26_OUT/kpi/images/000001.png" "png"
make_file "$G26_OUT/kpi/labels/000001.txt" "car 0.0 0 0.0 10 10 100 100 0 0 0 0 0 0 0"
make_file "$G26_OUT/hl_coco_its_kpi.json" '{"images": [], "annotations": []}'
make_file "$G26_WS/source_pool/coco.json" '{"images": [], "annotations": []}'
make_file "$G26_WS/source_pool/pool_report.json" '{"annotations_by_class": {"car": 10}}'
make_file "$WORK/g26_aux/notes.txt" "an input no config key names"
G26_RUN="$G26_WS/results/run_g26"

mounts_contain() {  # mounts_contain RESULTS_DIR NEEDLE
  "$PY" -c 'import json,sys
mounts = json.load(open(sys.argv[1]))["config"]["extra_container_mounts"]
print(1 if any(sys.argv[2] in m for m in mounts) else 0)' "$1" "$2"
}

run "$PY" "$INIT" \
  --results-dir "$G26_RUN" --workspace "$G26_WS" --max-iterations 1 \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G26_WS/ckpt/gdino_zero_shot.pth" \
  --train-spec-template "$G26_WS/specs/train_grounding_dino.yaml" \
  --source-pool-embeddings "$G26_WS/source_pool/source_embeddings.parquet" \
  --source-pool-annotations "$G26_WS/source_pool/odvg" \
  --embedding-model-path "$G26_WS/encoder/siglip" \
  --kpi-images-dir "$G26_OUT/kpi/images" \
  --ground-truth-labels-dir "$G26_OUT/kpi/labels" \
  --class-mapping "$G26_WS/classes/classes_its.yaml" \
  --ap50-thresholds-json '{"car": 0.9}' \
  --allocation-policy class_stratified --rare-class-list car \
  --pool-report "$G26_WS/source_pool/pool_report.json" \
  --source-detection-file "$G26_WS/source_pool/coco.json" \
  --target-detection-file "$G26_OUT/hl_coco_its_kpi.json" \
  --extra-mount "$WORK/g26_aux"
assert_rc 0 "[G26] class_stratified init succeeds with detection files outside the workspace"
assert_eq 1 "$(mounts_contain "$G26_RUN/deft_state.json" "$G26_OUT")" \
  "[G26] the target detection file's directory is mounted"
assert_eq 1 "$(mounts_contain "$G26_RUN/deft_state.json" "$WORK/g26_aux")" \
  "[G26] --extra-mount admits a path no config key names"

# A source detection file inside the workspace needs no mount of its own.
assert_eq 0 "$(mounts_contain "$G26_RUN/deft_state.json" "$G26_WS/source_pool")" \
  "[G26] a detection file inside the workspace adds no mount"

# An --extra-mount that is not on disk is a configuration error, not a silent skip.
run "$PY" "$INIT" \
  --results-dir "$G26_RUN" --workspace "$G26_WS" --max-iterations 1 --force \
  --num-epochs 1 --learning-rate 0.0001 \
  --zero-shot-checkpoint "$G26_WS/ckpt/gdino_zero_shot.pth" \
  --train-spec-template "$G26_WS/specs/train_grounding_dino.yaml" \
  --source-pool-embeddings "$G26_WS/source_pool/source_embeddings.parquet" \
  --source-pool-annotations "$G26_WS/source_pool/odvg" \
  --embedding-model-path "$G26_WS/encoder/siglip" \
  --kpi-images-dir "$G26_OUT/kpi/images" \
  --ground-truth-labels-dir "$G26_OUT/kpi/labels" \
  --class-mapping "$G26_WS/classes/classes_its.yaml" \
  --ap50-thresholds-json '{"car": 0.9}' \
  --extra-mount "$WORK/g26_absent"
assert_rc 1 "[G26] an --extra-mount that is not on disk is refused"
case "$RUN_OUT" in
  *"--extra-mount"*) ok "[G26] the refusal names the flag" ;;
  *) notok "[G26] the refusal names the flag" "output: $RUN_OUT" ;;
esac
# G27 — the KPI confidence threshold is the run's, and it is recorded
#
# Every phase is scored at config.kpi_conf_threshold, so it decides whether two
# runs are comparable. It defaults to 0.0, the threshold inference writes at.
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_SECTION="G27 kpi_conf_threshold"

G27=$(new_workspace g27); make_pool "$G27"
init_run "$G27" "$G27/results/run_g27" 1
assert_eq '0.0' "$(state_json "$G27/results/run_g27" kpi_conf_threshold)" \
  "[G27] the default is 0.0, what inference writes its labels at"

init_run "$G27" "$G27/results/run_g27" 1 --force --kpi-conf-threshold 0.3
assert_eq '0.3' "$(state_json "$G27/results/run_g27" kpi_conf_threshold)" \
  "[G27] a supplied threshold is recorded for every phase to score at"

init_run "$G27" "$G27/results/run_g27" 1 --force --kpi-conf-threshold 1.5
assert_rc 1 "[G27] a threshold outside [0, 1] is refused"

# ═══════════════════════════════════════════════════════════════════════════

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf 'PASS %d/%d assertions\n' "$TOTAL" "$TOTAL"
  exit 0
fi
printf 'FAIL %d/%d assertions failed\n' "$FAILURES" "$TOTAL"
exit 1
