---
name: i4h-workflow-dataset-convert
version: "0.6.0"
description: Convert an agentic HDF5 recording into a LeRobot dataset (parquet, meta, videos). Use when asked to convert HDF5, prepare for training, or export to LeRobot; not for viewing — use [[i4h-lerobot-viz]].
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - lerobot
    - conversion
---

# i4h Workflow — Convert Dataset

## Purpose

Convert an agentic HDF5 recording into a LeRobot dataset (parquet + meta + videos). Use when the user asks to convert HDF5, prepare for training, or export to LeRobot.

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

- Use the same `--env` that produced the HDF5.
- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` supplies the robot, task, cameras, and `dataset.*` (action/state names, splits, modality) converter defaults.
- Output goes to `HF_LEROBOT_HOME/<repo-id>`.

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup and resolve HDF5

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# Point HDF5_PATH at a real recording (absolute path). Recordings come from teleop, mimic, or
# validate (which writes data/verify.hdf5 under each runs/eval_* dir). List candidates newest-first:
#   find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head
HDF5_PATH="${HDF5_PATH:-}"
if [ ! -f "${HDF5_PATH}" ]; then
  echo "convert: set HDF5_PATH to an existing .hdf5 (got '${HDF5_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/convert_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
export HF_LEROBOT_HOME="${RUN_DIR}/lerobot"
```

### Step 2 — convert

```bash
"${REPO_ROOT}/workflows/agentic/dataset/run.sh" \
  --env "${ENV_ID}" \
  --hdf5-path "${HDF5_PATH}" \
  --repo-id "local/${ENV_ID}" \
  --video-codec h264 \
  --overwrite \
  2>&1 | tee "${RUN_DIR}/logs/convert.log"
```

## Notes

- `--video-codec h264` is required. The converter's default AV1 codec breaks GR00T's `decord` video reader at finetune time.
- Scissor SO-ARM generates `meta/modality.json` from YAML splits and does not need `dataset.modality_template_path`.
- G1 locomanip and assemble-trocar use `dataset.modality_template_path` from the env YAML.
- All camera streams are resized to the env YAML `policy.image_size` (override with `--image-size H W`), normalizing mixed-resolution cameras (e.g. head cam + overview cam) to the one size the modality config expects.

## Verify

- `${HF_LEROBOT_HOME}/local/${ENV_ID}/meta/info.json` exists.
- Log reports the saved episode count.
- Per-episode video files are present under `${HF_LEROBOT_HOME}/local/${ENV_ID}/`.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to convert (set `HDF5_PATH` to an absolute path; the Run block lists candidates if it's unset or wrong).
- The same `--env` that produced the HDF5 (its YAML supplies robot, task, camera, modality, and converter defaults).
- `HF_LEROBOT_HOME` set to the output location for `<repo-id>`.

## Limitations

- `--video-codec h264` is required; the converter's default AV1 codec breaks GR00T's `decord` reader at finetune time.
- All camera streams are resized to the env YAML `policy.image_size` (override with `--image-size H W`).
- G1 locomanip and assemble-trocar require `dataset.modality_template_path` from the env YAML; scissor SO-ARM generates `meta/modality.json` from YAML splits.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** input HDF5 not found - Cause: wrong or missing `--hdf5-path`. Fix: point `HDF5_PATH` at an existing recording.
- **Error:** `decord` fails to read video at finetune time - Cause: dataset written with the default AV1 codec. Fix: re-convert with `--video-codec h264`.
- **Error:** missing/incorrect modality config - Cause: wrong `--env`, so robot/task/camera/modality defaults do not match the HDF5. Fix: use the same `--env` that produced the recording.

## Final Response

Report source HDF5, dataset path, repo id, episode count, skipped or failed episodes.
