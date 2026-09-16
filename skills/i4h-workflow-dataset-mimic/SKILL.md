---
name: i4h-workflow-dataset-mimic
description: Expand workflow HDF5 demonstrations with action jitter, optionally scoped to node segments. Use for synthetic variants; do not use to collect data, alter state directly, or generate new images.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - augmentation
    - hdf5
---

# Mimic Workflow Demonstrations

## Purpose

Clone successful source episodes and perturb recorded actions.

## Instructions

1. Run the checkout resolver and select the source HDF5.
2. Inspect successful episodes and segments.
3. Run mimic to generate the requested additional action-jitter variants.
4. Compare exact counts and preserved contracts.

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
HDF5_PATH=/absolute/path/to/source.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH" --segments
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit/current-chain recording. Require at least one successful source episode. If the user asks for node-scoped jitter, require that node in segment metadata.

## Expand

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
uv run --project tools/mimic i4h-mimic \
  "$HDF5_PATH" "$RUN_DIR/expanded.hdf5" \
  --episodes <additional-count> \
  --noise 0.01 \
  --seed <seed> \
  --include-source \
  --successful-only
```

`--episodes` is the number of generated variants, not the final total. Add `--node <node-id>` only when requested. Preserve the default deterministic seed unless the user supplies another.

## Verify and compose

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/expanded.hdf5" --segments
```

Confirm source count, generated count, dimensions, success flags, and segment preservation. Reject zero generated episodes.

If the same prompt requests visualization, continue with `i4h-workflow-dataset-convert`, then `i4h-lerobot-viz`; raw HDF5 is not a LeRobot visualizer input.

## Troubleshooting

On failure, inspect the first episode shape or segment error. If a requested node is absent, report available node ids.

## Prerequisites

Require a readable HDF5 file with at least one successful episode and valid segments for node-scoped jitter.

## Limitations

The current tool does not independently add state noise, rerun physics, or synthesize camera frames.

## Examples

- `Mimic 3 more episodes and visualize my dataset.` → generate three variants, convert the expanded HDF5, then launch LeRobot visualization.

## Completion gate

Report input/output paths, source and generated counts, final total, action-noise value, seed, optional node, limitations, and conversion/visualizer result when requested.
