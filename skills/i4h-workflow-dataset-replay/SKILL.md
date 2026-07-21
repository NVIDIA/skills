---
name: i4h-workflow-dataset-replay
version: "0.6.0"
description: Replay a recorded HDF5 episode inside Isaac Sim for visual verification. Use when the user asks to replay, play back, or step through an HDF5 recording.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - replay
    - hdf5
---

# i4h Workflow — Replay Dataset

## Purpose

Replay a recorded HDF5 episode inside Isaac Sim for visual verification. Use when the user asks to replay, play back, or step through an HDF5 recording.

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

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — the `<env>` scene, robot, and cameras Arena replays against.
- Replay runs `arena/run.sh --replay` against the env that produced the HDF5.
- Use it to verify visual correctness before conversion or training.
- Interpret ordinal wording as zero-based episode indices: "first episode" -> `0`, "second episode" -> `1`, etc.

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup and resolve HDF5

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
EPISODE_INDEX="${EPISODE_INDEX:-0}"  # For "Replay second episode", set EPISODE_INDEX=1.

# Point HDF5_PATH at a real recording (absolute path). Recordings come from teleop, mimic, or
# validate (which writes data/verify.hdf5 under each runs/eval_* dir). List candidates newest-first:
#   find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head
HDF5_PATH="${HDF5_PATH:-}"
if [ ! -f "${HDF5_PATH}" ]; then
  echo "replay: set HDF5_PATH to an existing .hdf5 (got '${HDF5_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" -name '*.hdf5' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/replay_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — replay

```bash
"${REPO_ROOT}/workflows/agentic/arena/run.sh" \
  --env "${ENV_ID}" \
  --replay "${HDF5_PATH}" \
  --episode-index "${EPISODE_INDEX}" \
  2>&1 | tee "${RUN_DIR}/logs/replay.log"
```

## Notes

- `HDF5_PATH` must be an absolute path to an existing recording — `--replay` resolves a relative path against `runs/<env>/`, not your cwd, so a bare/relative path silently fails to load. The block lists real candidates if it's unset or wrong.
- Recordings come from [[i4h-workflow-dataset-teleop]], [[i4h-workflow-dataset-mimic]], or [[i4h-workflow-validate]] (validate writes `data/verify.hdf5` under each `runs/eval_*` dir). There is no default `demo.hdf5`.
- `--episode-index` selects the episode within the HDF5 (zero-based).
- For "Replay second episode", use `--episode-index 1`.
- Use the same env id as the env that produced the recording.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to replay.
- The env id that produced the recording.

## Limitations

- Visual verification only; replay does not modify or expand the recording.
- Replays one episode per invocation, selected by `--episode-index`.
- Runs inside Isaac Sim; the env id must match the one that produced the HDF5.

## Troubleshooting

- **Error:** `.venv` not found / replay fails to launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** `replay: set HDF5_PATH to an existing .hdf5` (or recording fails to load) - Cause: `HDF5_PATH` unset or not a real file. Fix: pick an absolute path from the printed candidates (e.g. a `verify.hdf5` under `runs/eval_*/data/`).
- **Error:** episode index out of range - Cause: `--episode-index` exceeds the episodes in the HDF5. Fix: use a valid zero-based index.
- **Error:** mismatched/garbled playback - Cause: `--env` differs from the env that produced the recording. Fix: use the same env id.

## Final Response

Report env, HDF5 path, episode index, launch outcome, visible mismatches.
