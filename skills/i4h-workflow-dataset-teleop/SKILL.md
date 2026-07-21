---
name: i4h-workflow-dataset-teleop
version: "0.6.0"
description: Record episodes for an agentic env via teleoperation (keyboard, SO-ARM leader, or VR) into HDF5. Use when the user wants to teleop or record human demos.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - teleoperation
    - recording
---

# i4h Workflow — Teleop Record

## Purpose

Record episodes for an agentic env via teleoperation (keyboard, SO-ARM leader, or VR) into HDF5. Use when the user wants to teleop or record human demos.

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

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — `robot.type`, `zenoh.camera_names`, and the task for `<env>`.
- Teleop runs through `arena/run.sh --teleop`.
- Device support is env-specific. Check `arena/run.sh --env <env> --help` for valid `--teleop-device` values.
- Do not run teleop headless unless the user explicitly asks. Teleop is a GUI/hardware handoff: launching the recorder is not the same as recording successful demos.

## Controls

Reserved keys (consistent across devices):

| Key | Action |
|---|---|
| `B` | Start episode |
| `N` | Mark success, save, advance |
| `R` | Discard, reset |
| `F` | Reserved by Isaac Sim — do not bind |

Device-specific keybindings (move, rotate, gripper, mode switches) are printed by the teleop process at startup and vary by `--teleop-device`. Report them to the user from the log; they cannot drive the sim without them. See "Surface Device Keybindings". For `keyboard_23d`, the banner can print late, after scene creation and recorder setup; wait for mode/control markers such as `BOTH_HANDS`, `HAND MODE`, `BASE NAVIGATION MODE`, `SPECIAL KEYS`, or `Current Mode` before saying the controls are missing.

Stop from terminal:

```bash
workflows/agentic/stop.sh arena --env <env>
```

## Known Devices

| Env | Devices |
|---|---|
| `scissor_pick_and_place` | `keyboard`, `so101_leader` |
| `locomanip_tray_pick_and_place` | `keyboard_23d` |
| `locomanip_push_cart` | `keyboard_23d` |

For other envs, consult `arena/run.sh --env <env> --help`.

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/teleop_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — teleop record

Use the foreground form when the human operator is ready to drive immediately and the agent can stay attached until completion:

```bash
"${REPO_ROOT}/workflows/agentic/arena/run.sh" \
  --env "${ENV_ID}" \
  --teleop \
  --teleop-device <device> \
  --episodes 3 \
  --record-to "${RUN_DIR}/data/demo.hdf5" \
  2>&1 | tee "${RUN_DIR}/logs/teleop.log"
```

If no human operator is ready, do not fake demos and do not leave a teleop process behind. You may launch long enough to confirm the visible GUI, recorder, and controls, then stop cleanly and report that recording is waiting on a human operator.

## Surface Device Keybindings

The teleop process prints its keybinding table to stdout shortly after launch (look for sections such as `Keybindings`, `Controls`, `Key Map`, `BOTH_HANDS`, `HAND MODE`, `BASE NAVIGATION MODE`, `SPECIAL KEYS`, `WORKSPACE LIMITS`, `Current Mode`, mode-switch lines, or any block enumerating keys/actions). Wait for the table to appear, extract it from the log, and report it to the user before they need to drive the sim.

```bash
# After launching teleop and tailing the log:
for i in $(seq 1 180); do
  if grep -qiE 'Keybind|Controls|Key Map|BOTH_HANDS|HAND MODE|BASE NAVIGATION MODE|SPECIAL KEYS|WORKSPACE LIMITS|Current Mode|Mode\] Switched|run complete|Traceback|Error' "${RUN_DIR}/logs/teleop.log" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -n -A80 -E 'Keybind|Controls|Key Map|BOTH_HANDS|HAND MODE|BASE NAVIGATION MODE|SPECIAL KEYS|WORKSPACE LIMITS|Current Mode|Mode\] Switched' "${RUN_DIR}/logs/teleop.log" | head -120
```

If the block is multi-section (e.g. `BOTH_HANDS`, `BASE_NAV`, `LEFT_HAND`, `RIGHT_HAND` modes), include every mode in the report. Append the reserved keys above so the user has one consolidated reference.

## Notes

- `--record-to` must be absolute. The recorder resolves relative paths against `workflows/agentic/arena` (its CWD) and writes to a nested orphan dir. `${RUN_DIR}/data/demo.hdf5` built from `${REPO_ROOT}` is absolute.
- Use `--save-all-episodes` only when failed attempts must be kept.
- If the prompt asks "Run teleop for N episodes" and no human/operator input is available, launch the visible recorder only for readiness/control verification, stop it cleanly, and report that no demos were recorded; do not claim N episodes were recorded.

## Verify

- `${RUN_DIR}/data/demo.hdf5` exists.
- Log contains `run complete: N/M episodes succeeded`, and N must equal the requested episode count before reporting success. A tiny HDF5 with `0/M episodes succeeded` is only a failed/empty recording artifact, not a usable dataset.
- Before final response, verify no unintended teleop process is left running unless the user explicitly asked to keep it open.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- A valid env id and a `--teleop-device` it supports (see Known Devices or `arena/run.sh --env <env> --help`).
- The chosen teleop device available (keyboard, SO-ARM leader, or VR).
- An absolute `--record-to` HDF5 path (relative paths resolve against `arena`'s CWD).

## Limitations

- Device support is env-specific; not every device works with every env.
- `--record-to` must be absolute or the recording lands in a nested orphan dir.
- By default only successful episodes are saved; use `--save-all-episodes` to keep failed attempts.
- Device-specific keybindings are only printed at startup; they cannot be known before launching.

## Troubleshooting

- **Error:** `.venv` not found / teleop fails to launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** invalid `--teleop-device` for env - Cause: device not supported by that env. Fix: pick a value from `arena/run.sh --env <env> --help` (or Known Devices).
- **Error:** HDF5 written to an unexpected/nested location - Cause: `--record-to` was relative. Fix: pass an absolute path built from `${REPO_ROOT}`.
- **Error:** keybindings missing / cannot drive the sim - Cause: the device keybinding table was not surfaced. Fix: extract the table from the log before the operator starts driving (see Surface Device Keybindings).
- **Error:** run exits with `0/N episodes succeeded` - Cause: no human/operator completed episodes. Fix: record successful source demos with an operator; do not continue to mimic/convert/finetune from an empty HDF5.

## Final Response

Report env, device, the device-specific keybinding table extracted from the log, requested vs saved episodes, HDF5 path, log path.
