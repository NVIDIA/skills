---
name: i4h-workflow-dataset-replay
description: Replay a workflow HDF5 episode through its original Scene. Use for visual trajectory and recording verification; do not use for policy evaluation or LeRobot data.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - replay
    - hdf5
---

# Replay a Workflow Recording

## Purpose

Replay the exact recorded action sequence through the matching workflow and inspect it visibly.

## Instructions

1. Run the checkout resolver and select the exact HDF5.
2. Read the original workflow and requested episode.
3. Inspect action-contract compatibility.
4. Run the replay visibly in the foreground through completion.

## Resolve the recording

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
find runs -name '*.hdf5' -type f -printf '%T@ %p\n' | sort -nr | head
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use an explicit or current-chain recording first. Otherwise select the newest plausible HDF5 and state that choice. Never substitute an older file after a failed recording.

Interpret natural ordinals as zero-based indices: first is `0`, second is `1`.

## Inspect before launch

```bash
HDF5_PATH=/absolute/path/to/recording.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH" --segments
```

Resolve the original workflow from recording metadata and conversation context. Confirm the requested episode exists and its action width matches the workflow's replay Scene contract.

## Replay

```bash
./run.sh <workflow> --replay "$HDF5_PATH" --episode <zero-based-index>
```

Keep the visible simulator and command in the foreground. Poll yielded execution until `run.sh` exits; do not detach or return while replay is still running.

## Verify

Observe the complete trajectory, relevant objects, camera views, segment boundaries, and final status. If motion diverges, report the first mismatching segment or action-contract error. Do not change workflow, episode, or recording silently.

## Troubleshooting

On launch failure, verify workflow metadata, episode existence, and action width. On divergence, report the first mismatching segment.

## Prerequisites

Require the original workflow assets and a readable episode whose action width matches replay mode.

## Limitations

Replay verifies stored actions in one Scene; it does not evaluate a learned policy or guarantee transfer to another Scene.

## Examples

- `Replay the second episode.` → select the current recording, map “second” to index `1`, and observe the visible replay.

## Completion gate

Report workflow, HDF5 path, episode index, frame/segment count, action width, exit/final status, and visible agreement or first mismatch.
