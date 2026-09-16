---
name: i4h-workflow-e2e
description: Run the maintained workflow data-to-policy pipeline from recording through checkpoint validation. Use for full end-to-end requests; do not use for one individual stage.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - robotics
    - data-to-policy
---

# Run the Workflow End-to-End Pipeline

## Purpose

Use the maintained driver so stage resolution, artifacts, logs, and checkpoint handoff stay consistent with current workflow/task manifests.

## Instructions

1. Resolve the base checkout and policy workflow.
2. Require a successful driver dry-run.
3. Execute the maintained driver in the foreground.
4. Inspect every stage artifact before reporting completion.

## Resolve the checkout

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
```

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Require the workflow's `policy` mode. The driver discovers the remote task, embodiment, task text, and trainability from live workflow/task manifests.

## Dry-run first

```bash
./scripts/e2e/run.sh --env <workflow> --dry-run
```

Require exit status 0 and inspect every printed command and artifact path. The dry-run is the source of truth for current stages and backend ownership.

## Run in the foreground

```bash
./scripts/e2e/run.sh --env <workflow>
```

Use `--run-dir` only when the caller needs a specific location. Apply `--skip-mimic`, `--skip-annotate`, `--skip-replay`, or `--skip-viz` only when the user explicitly omits that optional stage or a documented smoke profile requires it.

Keep the driver as this agent's foreground tool call. Do not use a subagent, monitor task, shell backgrounding, `nohup`, `tmux`, or a detached process. Poll until exit.

The driver performs full setup, then owns its stage sequence, timestamped run directory, `runs/.latest` link, and per-stage logs. Do not replace it with a manually assembled subset.

## Verify

On success, inspect the printed summary and artifacts:

- policy recording
- expanded/filtered HDF5 as applicable
- visible replay result when enabled
- LeRobot metadata, parquet, and videos
- visualizer URL/content when enabled
- training logs and exact checkpoint when trainable
- checkpoint validation recording and final success summary

On failure, stop at the first failed stage, inspect that stage's log, preserve the run directory, and repair the owning stage before rerunning. Do not skip a required failure merely to obtain a green summary. Stop leftovers with `./stop.sh all`.

## Troubleshooting

Use the first failed stage and its log to choose the owning stage skill. Preserve the run directory and rerun only after that stage verifies its output.

## Prerequisites

Require a policy workflow plus host, simulator, backend, VLM, dataset, training, and visualization dependencies for every enabled stage.

## Limitations

The pipeline supports only workflows with a policy mode; inference-only Tasks skip fine-tuning and checkpoint validation.

## Examples

- `Run end-to-end smoke pipeline for scissor pick-and-place.` → dry-run, execute the driver, and report each recording-to-validation stage.

## Completion gate

Report workflow/task/embodiment/trainability, dry-run result, run directory, every stage outcome and skip, dataset/visualizer/checkpoint/verification artifacts, final exit status, and cleanup state.
