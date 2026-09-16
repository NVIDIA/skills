---
name: i4h-workflow-dataset-teleop
description: Record demonstrations through a workflow's teleop Task into workflow HDF5. Use for keyboard, leader, VR, or bus input; do not use for policy evaluation or autonomous rule-based Tasks.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - teleoperation
    - hdf5
---

# Record Teleop Demonstrations

## Purpose

Run the workflow's declared teleop graph through the shared `SimulationRunner` so actions, state, cameras, segments, attempts, and outcomes use the normal HDF5 contract.

## Instructions

1. Resolve the base checkout and live teleop device contract.
2. Choose a supported human-input device.
3. Record in the foreground through `run.sh`.
4. Inspect visible motion and HDF5 content.

## Resolve support

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
./run.sh list
./run.sh show <workflow> --mode teleop
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Require `teleop` in the live mode list. Read the workflow builder, its scene manifest `teleop` override, and the embodiment manifest's `teleop_devices`. Do not maintain a static support table in the skill.

## Choose input

- Use the named device when supported.
- Otherwise use the workflow builder's default device.
- Keep interactive keyboard/leader/VR/bus sessions visible and in the foreground. Surface the device controls and require the human operator to complete the task.
- Never pretend to provide human input in an unattended shell.

## Record

```bash
./run.sh <workflow> --teleop <device> \
  --episodes <N> --attempts 3 \
  --record
```

Omit `<device>` to use the workflow default. Bare `--record` writes `demos.hdf5` inside the launcher's automatic run directory. Read the absolute directory from the `==> run dir ...` line or `run.json`; do not recreate its timestamp in the shell. When a larger pipeline requires a caller-selected shared directory, pass `--run-dir "$RUN_DIR" --record demos.hdf5`; the launcher creates the directory and anchors the relative recording name inside it. An absolute `--record` path remains supported.

## Verify

Require the final `N/N episodes succeeded` summary. Then inspect content:

```bash
RUN_DIR="<absolute run_dir from run.json or launcher output>"
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/demos.hdf5" --segments
uv run --project tools/dataset i4h-dataset actions "$RUN_DIR/demos.hdf5"
```

Visually confirm that the operator completes the requested task, robot motion matches the input device, and all expected cameras record the same behavior. Treat zero saved episodes, missing observations, absent action motion, or an unsuccessful task outcome as failure. Stop leftovers with `./stop.sh all`.

## Troubleshooting

On device or width errors, compare the workflow teleop builder, Scene mode override, and embodiment devices.

## Prerequisites

Require a workflow with `teleop`, a supported device, a working simulator, and a present operator for interactive input.

## Limitations

Teleop records human input and requires an operator for interactive devices. Record autonomous rule-based Tasks with `i4h-workflow-validate` instead.

## Examples

- `Record 5 keyboard teleop demonstrations for locomanip tray pick and place.` → use G1's supported keyboard device, require a human operator, and verify the recorded action motion.

## Completion gate

Report workflow, mode/device, controls, requested/saved episodes, attempts, visual result, HDF5 path, dimensions/segments, and whether a human operator completed the task.
