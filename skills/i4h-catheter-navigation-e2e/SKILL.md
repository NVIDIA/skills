---
name: i4h-catheter-navigation-e2e
version: "0.7.0"
description: End-to-end smoke for catheter navigation covering setup, digital twin, DRR, and unit tests. Use when asked to run the full catheter workflow smoke or demo the v0.7 pipeline.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - end-to-end
    - smoke
---

# i4h Catheter Navigation - End-to-End Smoke

## Purpose

Run a v0.7 smoke pipeline: verify setup, build a digital twin from a TotalSegmentator subject, render a DRR from cache, and run CPU unittests. Skips the interactive viewport (requires display).

## Base Code

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

Stages map to individual skills; this skill chains them with shared `RUN_DIR` and `CACHE`.

| Stage | Skill |
|---|---|
| setup + CPU smoke | [[i4h-catheter-navigation-setup]] |
| digital twin | [[i4h-catheter-navigation-digital-twin]] |
| GPU DRR | [[i4h-catheter-navigation-render-drr]] |
| unittest | [[i4h-catheter-navigation-smoke]] |

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - initialize run

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
RUN_DIR="${WF_ROOT}/runs/e2e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"
CACHE="${CACHE:-/tmp/ct_cache}"
SUBJ="${SUBJ:-}"
export I4H_WORKFLOWS="$REPO_ROOT"
```

### Step 2 - setup verify

Follow [[i4h-catheter-navigation-setup]] Steps 1-3; tee to `${RUN_DIR}/logs/setup.log`.

### Step 3 - digital twin (requires SUBJ)

Set `SUBJ` to an extracted TotalSegmentator subject, then follow [[i4h-catheter-navigation-digital-twin]] Steps 3-4.

If `SUBJ` is unset, skip to Step 4 with synthetic DRR only and note partial e2e in the summary.

### Step 4 - render DRR

If Step 3 succeeded, set `CACHE` and follow [[i4h-catheter-navigation-render-drr]] cache variant.
Otherwise run synthetic render to `${RUN_DIR}/drr_synthetic.png`.

### Step 5 - smoke tests

Follow [[i4h-catheter-navigation-smoke]]; log to `${RUN_DIR}/logs/smoke.log`.

### Step 6 - summary

```bash
cat > "${RUN_DIR}/logs/SUMMARY.txt" <<EOF
Catheter navigation e2e smoke
RUN_DIR=${RUN_DIR}
CACHE=${CACHE}
SUBJ=${SUBJ:-<skipped>}
Artifacts:
  $(ls -1 "${RUN_DIR}" 2>/dev/null || true)
EOF
cat "${RUN_DIR}/logs/SUMMARY.txt"
```

## Flags (informal)

- Skip digital twin when `SUBJ` unset -> synthetic DRR only.
- Add viewport manually via [[i4h-catheter-navigation-viewport]] when a display is available.

## Outputs

- `RUN_DIR/logs/` - per-stage logs and `SUMMARY.txt`
- `RUN_DIR/drr.png` or `drr_synthetic.png` - rendered frame
- `CACHE/` - digital twin artifacts when Step 3 ran

## Prerequisites

- [[i4h-catheter-navigation-setup]] host requirements.
- `SUBJ` for full e2e (otherwise partial smoke).
- GPU for Step 4 cache render.

## Limitations

- Does not launch the interactive viewport (display required).
- Does not cover RL/IL training loops.

## Troubleshooting

- Partial run when no CT data - expected; document which stages ran.
- GPU render failure - fall back to synthetic DRR and report GPU setup gap.

## Final Response

Print `RUN_DIR`, stages completed/skipped, DRR output path, smoke-test status, and recommend [[i4h-catheter-navigation-viewport]] for interactive demo.
