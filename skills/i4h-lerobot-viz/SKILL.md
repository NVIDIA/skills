---
name: i4h-lerobot-viz
description: Serve and visually inspect a converted LeRobot dataset in the browser. Use for videos and state/action timelines; do not use for raw workflow HDF5 or incomplete conversion output.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - lerobot
    - visualization
---

# Visualize a LeRobot Dataset

## Purpose

Serve one completed local dataset and verify its episode videos and timelines in a browser.

## Instructions

1. Resolve the base checkout and one completed LeRobot dataset.
2. Launch its managed local server.
3. Open the printed URL.
4. Inspect videos, timelines, episode count, and cleanup state.

## Resolve the dataset

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
find runs "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" \
  -name info.json -path '*/meta/*' -printf '%T@ %h\n' 2>/dev/null \
  | sort -nr | head
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit/current-chain converted directory. Otherwise select the newest candidate and state the choice. Require `<dataset>/meta/info.json`; route raw HDF5 to `i4h-workflow-dataset-convert`.

## Serve

Pass an absolute local path:

```bash
DATASET_DIR=/absolute/path/to/lerobot/dataset
STATE_DIR=/absolute/path/to/run/viz-state
tools/dataset/scripts/viz.sh "$DATASET_DIR" --state-dir "$STATE_DIR"
```

The script selects a free local port, waits for HTTP readiness, and prints the URL, PID, state files, log, and exact stop command. Keep it running only while the user wants access.

## Verify visually

Open the printed URL. Confirm:

- the requested episode list loads
- every expected camera video renders
- state and action timelines render with plausible dimensions and motion
- episode count and task text match `meta/info.json`
- no blank page, missing video, or wrong dataset is being served

Reuse a live server only when its target dataset matches. Otherwise stop it using its printed state directory and port, then start the requested dataset.

## Troubleshooting

If startup or the page fails, inspect `meta/info.json`, videos, the printed server log, and port ownership before restarting.

## Prerequisites

Require the dataset tool environment and a completed LeRobot dataset with `meta/info.json`.

## Limitations

The visualizer does not accept raw workflow HDF5 or repair incomplete metadata/videos.

## Examples

- `Open the latest converted LeRobot dataset for inspection.` → choose the newest verified dataset, serve it, and report the observed videos/timelines plus cleanup command.

## Completion gate

Report dataset path, repo id, local URL, episode/camera/timeline observations, PID/state directory, and exact cleanup command.
