---
name: i4h-lerobot-viz
version: "0.6.0"
description: Serve the LeRobot HTML visualizer for a converted dataset in a browser. Use when asked to visualize, inspect, or open a LeRobot dataset; not for converting HDF5 (use [[i4h-workflow-dataset-convert]]).
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - lerobot
    - visualization
    - dataset
---

# i4h Workflow — LeRobot Viz

## Purpose

Serve the LeRobot HTML visualizer for a converted dataset in a browser. Use when the user asks to visualize, inspect, or open a LeRobot dataset.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- Input is a converted LeRobot dataset directory containing `meta/info.json`.
- Use for visual checks after conversion or video augmentation.

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup and resolve dataset

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# Point DATASET_DIR at a converted LeRobot dataset dir (absolute; must contain meta/info.json),
# produced by [[i4h-workflow-dataset-convert]]. List candidates:
#   find "${RUNS_ROOT}" "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" -name info.json -path '*/meta/*' -printf '%h\n' | sed 's#/meta$##' | sort -u
DATASET_DIR="${DATASET_DIR:-}"
if [ ! -f "${DATASET_DIR%/}/meta/info.json" ]; then
  echo "viz: set DATASET_DIR to a LeRobot dataset dir with meta/info.json (got '${DATASET_DIR:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" -name info.json -path '*/meta/*' -printf '%h\n' 2>/dev/null | sed 's#/meta$##' | sort -u | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/viz_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/viz_state"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — serve visualizer

```bash
"${REPO_ROOT}/workflows/agentic/dataset/viz.sh" "${DATASET_DIR}" \
  --state-dir "${RUN_DIR}/viz_state" \
  2>&1 | tee "${RUN_DIR}/logs/viz.log"
```

## Notes

- The dataset path must be absolute. `viz.sh` treats relative paths as Hugging Face repo ids and looks them up under `~/.cache/huggingface/lerobot/<path>`.
- Override `--state-dir` only when the caller provides one.

## Verify

- The visualizer prints a local URL (e.g. `http://127.0.0.1:9090/`).
- Videos and joint timelines load in the browser.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- A converted LeRobot dataset directory containing `meta/info.json` (see [[i4h-workflow-dataset-convert]]).
- An absolute path to that dataset directory.

## Limitations

- Input must be a converted LeRobot dataset directory with `meta/info.json`; intended for visual checks after conversion or video augmentation.
- The dataset path must be absolute; `viz.sh` treats relative paths as Hugging Face repo ids and looks them up under `~/.cache/huggingface/lerobot/<path>`.
- Override `--state-dir` only when the caller provides one.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** dataset resolved as a Hugging Face repo id / not found - Cause: a relative dataset path was passed. Fix: pass the absolute path to the dataset directory.
- **Error:** no `meta/info.json` - Cause: directory is not a converted LeRobot dataset. Fix: convert first with [[i4h-workflow-dataset-convert]].
- **Error:** address/port already in use - Cause: a visualizer is already serving that local URL. Fix: stop the existing process before starting a new one.

## Final Response

Report dataset path, visualizer URL, stop command, startup failures.
