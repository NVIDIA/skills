---
name: i4h-catheter-navigation-setup
version: "0.7.0"
description: Verify host/GPU requirements and PYTHONPATH for the catheter navigation workflow. Use when asked to set up, install, or bootstrap catheter_navigation, or when hitting import/GPU/slangpy errors.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - setup
    - installation
---

# i4h Catheter Navigation - Setup

## Purpose

Verify host and GPU requirements, confirm the `./i4h` CLI sees the workflow, and run CPU smoke tests. Use when asked to set up catheter navigation or when hitting missing imports, GPU, or slangpy errors.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/catheter_navigation/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` - pick that default without prompting. Run every command below from the resolved root:

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- Catheter navigation registers via `workflows/catheter_navigation/metadata.json` and runs through `./i4h run catheter_navigation <mode>`.
- Runtime is package-first: `render_drr` uses installed `fluorosim` (`python -m fluorosim.examples.render_drr`), while `interactive_viewport` is launched from the local workflow script path in `metadata.json`.
- Docker image: `workflows/catheter_navigation/docker/Dockerfile` (drop `--local` on `./i4h run` to use it).
- GPU modes need slangpy, Warp, and CUDA; CPU smoke tests do not.

## Preflight

```bash
command -v python3
command -v git
nvidia-smi
df -h .
```

Required: Linux x86_64 (Ubuntu 22.04/24.04 tested), NVIDIA GPU (CC >= 7.0), driver compatible with CUDA 12.8, >= 16 GB RAM, >= 20 GB disk.

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - resolve repo and export paths

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
SIM_ROOT="${WF_ROOT}/scripts/simulation"
export PYTHONPATH="${SIM_ROOT}:${PYTHONPATH:-}"
RUN_DIR="${WF_ROOT}/runs/setup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"
```

### Step 2 - verify CLI registration

```bash
"${REPO_ROOT}/i4h" modes catheter_navigation 2>&1 | tee "${RUN_DIR}/logs/modes.log"
```

### Step 3 - CPU smoke tests (no GPU)

```bash
python3 -m unittest workflows/catheter_navigation/tests/test_fluorosim_smoke.py \
  2>&1 | tee "${RUN_DIR}/logs/smoke.log"
```

Expected: `Ran 7 tests ... OK`. Parser error lines in stderr from negative test cases are expected.

### Step 4 - optional GPU sanity (synthetic DRR)

Skip if no GPU or slangpy not installed.

```bash
"${REPO_ROOT}/i4h" run catheter_navigation render_drr --local \
  --run-args="--output ${RUN_DIR}/drr.png" \
  2>&1 | tee "${RUN_DIR}/logs/render_drr.log"
```

## Verify

```bash
test -f "${RUN_DIR}/logs/smoke.log"
grep -q "OK" "${RUN_DIR}/logs/smoke.log"
python3 -c "import fluorosim; print('fluorosim', fluorosim.__file__)"
```

## Prerequisites

- Repo checkout with `workflows/catheter_navigation/` present.
- Python 3 with numpy; full GPU stack (slangpy, torch CUDA, warp) for render/viewport modes.

## Limitations

- No dedicated `setup.sh` yet - this skill verifies and documents host requirements; use Docker when host deps are incomplete.
- Step 4 requires a GPU; Step 3 alone is sufficient for CI-style verification.

## Troubleshooting

- **Error:** `fluorosim` import fails - Cause: PYTHONPATH not set. Fix: re-run Step 1; confirm `SIM_ROOT` exists.
- **Error:** `./i4h` not found - Cause: not at repo root. Fix: `cd "$REPO_ROOT"` where `./i4h` lives.
- **Error:** render_drr fails with slang/GPU - Cause: missing CUDA or slangpy. Fix: use Docker (`./i4h run catheter_navigation render_drr` without `--local`) or install deps per README.

## Final Response

Report setup status, smoke-test result, optional DRR output path, and recommend the next skill ([[i4h-catheter-navigation-digital-twin]] for patient data, [[i4h-catheter-navigation-viewport]] for interactive demo).
