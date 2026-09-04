---
name: i4h-workflow-dataset-convert
description: Convert workflow HDF5 recordings to LeRobot datasets for training or browser inspection. Use for conversion; do not use for replay, augmentation, or raw-data repair.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - hdf5
    - lerobot
---

# Convert Workflow HDF5 to LeRobot

## Purpose

Preserve recorded actions, state, cameras, task text, and embodiment labels in a local LeRobot dataset.

## Instructions

1. Run the checkout resolver and select the source HDF5.
2. Read the workflow, Scene, embodiment, and instruction.
3. Run conversion for the selected successful episodes.
4. Inspect metadata, parquet, videos, and feature widths.

## Resolve and inspect

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
HDF5_PATH=/absolute/path/to/recording.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH" --segments
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit/current-chain recording. Resolve its workflow and Scene from recording metadata/context, then read the Scene manifest for the embodiment and instruction. Use the embodiment manifest for labels. Do not assume state width equals action width; the converter derives both from the recording.

## Convert

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="$RUN_DIR/lerobot/local/<name>"
mkdir -p "$(dirname "$DATASET_DIR")"
[ ! -e "$DATASET_DIR" ] || { echo "Destination already exists: $DATASET_DIR" >&2; exit 2; }
uv run --project tools/dataset i4h-dataset convert \
  "$HDF5_PATH" "$DATASET_DIR" \
  --robot <embodiment> \
  --repo-id "local/<name>" \
  --successful-only \
  --task "<instruction>"
```

Use `--fps` or `--skip-frames` only when the user requests it or source metadata justifies it. Keep the default H.264 video codec for compatibility with GR00T's fast decord loader; select another `--video-codec` only when the target consumer requires it.

Conversion writes aggregate `meta/stats.json` for downstream policy loaders. Native G1 rule-based WBC recordings already contain 43-D state and 50-D action; the converter recognizes that contract and writes GR00T's required semantic `meta/modality.json` automatically. For a G1 recording made through the legacy 23-D Pink/keyboard contract and destined for a 50-D G1 WBC policy Task, add `--g1-wbc-policy-actions`. That explicit mapping combines the measured 43-joint state with the recorded navigation, base-height, and torso commands; require source action width 23 and state width 43.

## Verify

Require:

- `meta/info.json`
- `meta/stats.json`
- `meta/modality.json` when the target trainer requires semantic modality slices
- episode parquet data
- video files for every recorded camera
- converted episode count matching selected successful sources
- action/state feature widths and names matching the recording plus embodiment descriptor

For G1, require modality metadata for both supported paths: native `state=43/action=50`, or explicitly mapped `state=43/source-action=23/output-action=50`. Treat a native 50-D dataset without `meta/modality.json` as incomplete.

Treat missing inputs or zero converted episodes as failure. If conversion leaves a partial destination, quarantine or remove that exact incomplete directory before retrying; never report it as usable.

## Troubleshooting

On dimension errors, resolve the source workflow and embodiment again. On missing videos, confirm frames existed before conversion.

## Prerequisites

Require a readable HDF5 recording and its matching Scene plus embodiment manifests.

## Limitations

Conversion cannot reconstruct missing cameras, actions, state, task text, or successful episodes.

## Examples

- `Convert my scissor pick-and-place recording into a LeRobot dataset.` → resolve `so101`, convert successful episodes, and verify metadata, parquet, and both camera videos.

## Completion gate

Report source HDF5/workflow, embodiment, task text, source/converted/skipped counts, action/state widths, output directory/repo id, aggregate-stats/modality/parquet/video checks, and any missing modality.
