---
name: i4h-catheter-navigation-render-drr
version: "0.7.0"
description: Render a single DRR fluoroscopy frame from a CT cache or synthetic phantom. Use when asked to render DRR, generate a fluoro image, or smoke-test the Slang renderer.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - drr
    - fluoroscopy
---

# i4h Catheter Navigation - Render DRR

## Purpose

Render a single digitally reconstructed radiograph (DRR) frame. Works with a preprocessed CT cache from [[i4h-catheter-navigation-digital-twin]], a direct NIfTI/DICOM path, or the built-in synthetic phantom (no data required).

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

- Default mode in `metadata.json`; self-contained with synthetic phantom when no `--cache` is given.
- GPU + slangpy required for actual rendering.
- Entry mode: `./i4h run catheter_navigation render_drr` (preferred).

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - resolve run dir

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
RUN_DIR="${WF_ROOT}/runs/render_drr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"
OUTPUT="${RUN_DIR}/drr.png"
CACHE="${CACHE:-}"
```

### Step 2 - render (pick one variant)

**Synthetic phantom (fastest smoke, no data):**

```bash
"${REPO_ROOT}/i4h" run catheter_navigation render_drr --local \
  --run-args="--output ${OUTPUT}" \
  2>&1 | tee "${RUN_DIR}/logs/render_drr.log"
```

**From preprocessed cache:**

```bash
if [ ! -d "${CACHE}" ] || [ ! -f "${CACHE}/mu_volume.npy" ]; then
  echo "render-drr: set CACHE to a preprocess_ct output dir (missing mu_volume.npy)." >&2
  exit 1
fi
"${REPO_ROOT}/i4h" run catheter_navigation render_drr --local \
  --run-args="--cache ${CACHE} --output ${OUTPUT}" \
  2>&1 | tee "${RUN_DIR}/logs/render_drr.log"
```

## Verify

```bash
test -f "${OUTPUT}"
file "${OUTPUT}"
```

## Prerequisites

- [[i4h-catheter-navigation-setup]] completed.
- NVIDIA GPU with slangpy for rendering (CPU smoke tests do not cover GPU render).

## Limitations

- Single-frame render only; batch multi-env RL rendering uses the fluorosim Python API directly.
- Catheter compositing in DRR requires attaching a `CatheterProvider` in custom scripts (not the default example).

## Troubleshooting

- **Error:** slangpy / CUDA failures - Fix: run without `--local` to use Docker, or verify GPU driver >= 570 and CUDA 12.8.
- **Error:** cache not found - Fix: run [[i4h-catheter-navigation-digital-twin]] first or use synthetic mode (no `--cache`).

## Final Response

Report output PNG path, whether synthetic or patient cache was used, and log path. Recommend [[i4h-catheter-navigation-viewport]] for interactive navigation.
