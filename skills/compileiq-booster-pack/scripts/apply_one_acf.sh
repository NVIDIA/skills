#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Apply a single ACF candidate, measure it against a baseline, and append a
# row to booster-pack-log.csv. Single-ACF only — no auto-loop, by design.
#
# Usage:
#   apply_one_acf.sh \
#       --acf path/to/candidate.acf \
#       --baseline-cmd "python bench.py" \
#       --candidate-cmd "PTXAS_OPTIONS='--apply-controls=path/to/candidate.acf' python bench.py" \
#       [--trials 10] [--log booster-pack-log.csv] \
#       [--score-regex 'mean: ([0-9.]+)']
#
#   apply_one_acf.sh --self-test
#
# The PTXAS_OPTIONS= prefix above suits a Triton 3.6+ benchmark; for other
# frameworks pass whatever injection your objective uses.
#
# The `--candidate-cmd` must produce a line on stdout that matches the
# `--score-regex` capture-group; default regex assumes the line "mean: 1.2345".
# Lower-is-better. The script returns 0 if the candidate improves over baseline
# with margin > 1% (and prints "KEPT"); else returns 1 and prints "REJECTED:<reason>".

set -u

ACF=""
BASELINE_CMD=""
CANDIDATE_CMD=""
TRIALS=10
LOG="booster-pack-log.csv"
SCORE_REGEX='mean: ([0-9.]+)'
SELF_TEST=0

usage() {
    sed -nE '2,/^$/ s/^# ?//p' "$0"
    exit 2
}

python_bin() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    echo "python3 or python is required" >&2
    return 1
}
PYTHON_BIN="$(python_bin)" || exit 2

while [ "$#" -gt 0 ]; do
    case "$1" in
        --acf) ACF="$2"; shift 2 ;;
        --baseline-cmd) BASELINE_CMD="$2"; shift 2 ;;
        --candidate-cmd) CANDIDATE_CMD="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --score-regex) SCORE_REGEX="$2"; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

run_n() {
    # Run the command N times, extract the first regex match per run, print one number per line.
    # xargs splits the command string (quotes and leading VAR=value assignments
    # included) and env runs it without a shell, so pipes and redirection inside
    # it are not supported.
    local cmd="$1"
    local n="$2"
    for _ in $(seq 1 "$n"); do
        printf '%s\n' "$cmd" | xargs env 2>/dev/null \
          | grep -oE "$SCORE_REGEX" | sed -nE "s/$SCORE_REGEX/\\1/p" | head -1
    done
}

mean() {
    "$PYTHON_BIN" -c "import sys; vs=[float(x) for x in sys.stdin.read().split() if x]; print(sum(vs)/len(vs) if vs else 'nan')"
}

if [ "$SELF_TEST" = 1 ]; then
    # Dry-run: identical "baseline" and "candidate" commands. Helper must report NOT-an-improvement.
    SELF_CMD="$PYTHON_BIN -c \"import random; print(f'mean: {1.0 + random.random()*0.001}')\""
    B=$(run_n "$SELF_CMD" 5 | mean)
    C=$(run_n "$SELF_CMD" 5 | mean)
    DELTA_PCT=$("$PYTHON_BIN" -c "b,c=$B,$C; print((b-c)/b*100)")
    echo "self-test baseline=$B  candidate=$C  delta_pct=$DELTA_PCT"
    if "$PYTHON_BIN" -c "import sys; sys.exit(0 if abs($DELTA_PCT) < 1.0 else 1)"; then
        echo "SELF-TEST PASS (identical distributions correctly classified as NOT a real improvement)"
        exit 0
    else
        echo "SELF-TEST FAIL (helper thinks identical distributions differ materially)"
        exit 1
    fi
fi

[ -z "$ACF" ] || [ -z "$BASELINE_CMD" ] || [ -z "$CANDIDATE_CMD" ] && usage
[ -f "$ACF" ] || { echo "ACF not found: $ACF" >&2; exit 2; }

ACF_SHA=$(sha256sum "$ACF" | awk '{print $1}')

echo "Running baseline ($TRIALS trials)..."
B_TIMES=$(run_n "$BASELINE_CMD" "$TRIALS")
B_MEAN=$(printf '%s\n' "$B_TIMES" | mean)

echo "Running candidate ($TRIALS trials)..."
C_TIMES=$(run_n "$CANDIDATE_CMD" "$TRIALS")
C_MEAN=$(printf '%s\n' "$C_TIMES" | mean)

DECISION=$("$PYTHON_BIN" <<PY
b, c = float("$B_MEAN"), float("$C_MEAN")
if c != c or b != b:  # NaN: every run failed or printed no score
    print("REJECTED:nan_score")
elif c >= b * 0.99:
    print("REJECTED:no_improvement_or_within_1pct")
else:
    speedup = b / c
    print(f"KEPT:speedup={speedup:.4f}x")
PY
)

# Append CSV row
if [ ! -f "$LOG" ]; then
    echo "timestamp,acf,sha256,trials,baseline_mean,candidate_mean,decision,baseline_cmd,candidate_cmd" > "$LOG"
fi
# CSV-quote the free-text fields by doubling embedded double quotes.
printf '%s,%s,%s,%s,%s,%s,"%s","%s","%s"\n' \
    "$(date -u +%FT%TZ)" "$ACF" "$ACF_SHA" "$TRIALS" "$B_MEAN" "$C_MEAN" "$DECISION" \
    "${BASELINE_CMD//\"/\"\"}" "${CANDIDATE_CMD//\"/\"\"}" \
    >> "$LOG"

echo "baseline=$B_MEAN  candidate=$C_MEAN  $DECISION"
case "$DECISION" in
    KEPT*) exit 0 ;;
    *)     exit 1 ;;
esac
