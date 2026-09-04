---
name: i4h-workflow-validate
description: Run the root-level workflow runtime policy or rule-based rollouts and verify simulator success. Use for evaluation, checkpoints, or local controllers; do not use for replay or dataset annotation.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - simulation
    - evaluation
---

# Validate a Workflow

## Purpose

Run the selected workflow run mode through the unified launcher, inspect the completed recording, and report simulator success.

## Instructions

1. Resolve the base checkout and a live workflow run mode.
2. Run the unified launcher in the foreground.
3. Require the final episode success summary.
4. Inspect visible behavior and every rollout artifact.

## Resolve live support

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
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Treat `./run.sh list` output as the complete authoritative workflow-by-mode table; it is dependency-light and faster than scanning workflow modules. Do not duplicate that mutable table in this skill. Map the user's natural name to a listed workflow id, then choose only a mode shown on that same line:

| User intent | Required live mode | Launcher argument |
|---|---|---|
| Ordinary learned-policy evaluation | `policy` | `--policy` |
| Requested or only available local controller | `rule-based` | `--rule-based` |
| Explicit named alternative such as N1.7 | matching listed mode such as `policy_n17` | `--mode <name>` |

Inspect `./run.sh show <workflow> --mode <mode>`, the workflow module, Scene manifest, and selected Task manifest when model, prompt, checkpoint, goal, or step-cap behavior matters.

Use precise readiness language:

- **Structurally valid**: `show`, per-mode lint, and `lint --all` pass.
- **Launchable**: the selected simulator mode starts and every required backend/checkpoint preloads.
- **Rollout-validated**: the requested episodes complete and the recorded success evidence passes inspection.

Do not report “validated” without stating which level was actually reached.

## Foreground execution rule

Keep `run.sh` as this agent's foreground tool call. Do not use a subagent, monitor task, shell backgrounding, `nohup`, `tmux`, or a detached process. Poll a yielded session until exit and inspect the final episode summary before responding. When the selected Task is remote, the policy backend subprocess internally owned by `run.sh` is expected; a simulator-compatible exported RSL-RL Task runs in-process.

Run visibly by default. If the user explicitly requests headless execution, or a documented environment constraint makes it necessary, say so before launch and include `--headless`; never switch to headless silently.

Policy:

```bash
./run.sh <workflow> --policy \
  --episodes <N> --attempts 3 \
  --record verify.hdf5
```

Rule-based:

```bash
./run.sh <workflow> --rule-based \
  --episodes <N> --attempts 3 \
  --record verify.hdf5
```

The launcher creates a unique canonical run directory and anchors the relative `verify.hdf5` inside it. Resolve `RUN_DIR` from the `==> run dir ...` line or machine-readable `run.json`; do not recreate the launcher's timestamp. Use `--run-dir "$RUN_DIR"` only when a caller-selected location must be shared with another stage; the launcher creates it. Absolute recording paths remain supported.

For another declared mode, use `--mode <name>`. For a supplied/new checkpoint, resolve its exact path, confirm it belongs to the selected policy Task, and pass `--checkpoint /absolute/path`; an exported RSL-RL Task expects its TorchScript `policy.pt`, while a remote Task expects the owning backend's loadable checkpoint format. For “300 timesteps,” pass `--episode-steps 300`.

Never raise the Scene manifest's cap. `--episode-steps` may only lower it. Remote inference waits do not consume simulation steps. Use a unique `--namespace` when another run of the same workflow is active.

## Verify

Require exit status 0 and final `N/N episodes succeeded`. A failed attempt followed by a successful retry counts as a successful requested episode; report attempts and retries.

```bash
RUN_DIR="<absolute run_dir from run.json or launcher output>"
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/verify.hdf5" --segments
```

Inspect episode metadata, action/state widths, camera frames, node segments, and success flags. For visible runs, observe Scene/camera behavior and final task outcome. On failure, use the first actionable backend, contract, graph, or simulator error; never switch modes or increase the cap silently. Run `./stop.sh all` after crashes that leave processes.

For a success rule that excludes collision, inspect the owning contact-sensor configuration and require a forced-contact negative test at least once after authoring or changing that rule. A sensor that initializes but has never produced a non-zero filtered force does not establish collision rejection.

## Troubleshooting

Use the first actionable backend, graph, contract, or simulator error. Retry within the same mode and cap only after correcting it.

## Prerequisites

Require synced simulator assets and any backend/checkpoint declared by the selected run mode.

## Limitations

Only modes from `run.sh list` are supported, and a runtime step override may lower but never raise the validated Scene cap.

## Examples

- `Evaluate scissor pick and place for 2 episodes.` → run policy mode for two successful episodes, record, inspect, and report attempts plus visible outcome.
- `Run surgical_reach_psm in rule-based mode for 1 episode.` → use only the declared local-controller mode.

## Completion gate

Report workflow, mode, model/checkpoint source, requested successes, attempts/retries, completion steps, visible outcome, HDF5 path and inspection, final exit status, and first unresolved failure if any.
