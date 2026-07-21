---
name: i4h-workflow-e2e
version: "0.6.0"
description: Run the full end-to-end agentic pipeline (record → mimic → annotate → replay → convert → visualize → finetune → validate). Use when asked to run the whole pipeline or do an e2e, smoke, or demo run.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - end-to-end
    - pipeline
---

# i4h Workflow — End-to-End

## Purpose

Run the full end-to-end agentic pipeline (record, mimic, annotate/filter, replay, convert, visualize, finetune, validate). Use when the user asks to run the full pipeline, smoke the whole workflow, demo the workflow, or do an e2e run.

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

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — drives every stage for `<env>` (robot, task, policy, cameras, `arena.max_timesteps`, `dataset.*` mappings).
- Use the e2e script for full pipeline runs.
- For per-stage work, use the corresponding dataset/finetune/validate skills.
- `assemble_trocar` is inference-only; the e2e script skips finetune and checkpoint validation for it.

## Dry Run

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
"${REPO_ROOT}/workflows/agentic/scripts/e2e/run.sh" --dry-run --env <env>
```

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

For Claude Code `--print` or any other noninteractive runner, keep Step 2 in the foreground. This is a validation requirement: do not use Claude background tasks, async task mode, Bash background mode, `&`, `nohup`, `tmux`, `disown`, or any detached process/task id, and do not answer that the pipeline is still running. Do not return until `run.sh` exits and you have inspected `logs/SUMMARY.txt` on success, or the failing stage log on failure. Report the run dir, skipped stages, per-stage status, key artifacts, and cleanup/stop status before finishing.

### Step 1 — setup

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
```

### Step 2 — e2e pipeline

```bash
"${REPO_ROOT}/workflows/agentic/scripts/e2e/run.sh" --env <env>
```

## Flags

- `--skip-mimic`, `--skip-annotate`, `--skip-replay`, `--skip-viz`
- `--from-stage <stage> --run-dir <existing-run>` resumes from a prior run.
- Policy record/verify stages open the sim window by default. Set `ARENA_HEADLESS=1` before `run.sh` only when the user explicitly asks for headless/no-window execution.

Stages: `setup record mimic annotate replay convert viz finetune validate summary`.

## Outputs

The script prints `RUN_DIR` and symlinks it to `runs/.latest`. Subdirs:

- `logs/` — per-stage logs, `workflow.log` (full teed output), and `logs/SUMMARY.txt` (the final summary report)
- `data/`
- `lerobot/`
- `checkpoint/` (trainable envs only)

## Monitor

`run.sh` runs every stage in the foreground and returns only when the whole pipeline ends, so track a long run from a **separate shell** (do not expect to query it from the shell that launched it):

```bash
tail -f "${REPO_ROOT}/workflows/agentic/runs/.latest/logs/workflow.log"   # live per-stage progress
cat    "${REPO_ROOT}/workflows/agentic/runs/.latest/logs/SUMMARY.txt"     # final report (once DONE)
```

## Stop

### Step 3 — stop (if needed)

```bash
"${REPO_ROOT}/workflows/agentic/stop.sh" all --env <env>
```

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist); `setup` is also the first pipeline stage.
- A valid `--env` name to drive the run.
- For per-stage work, use the corresponding dataset/finetune/validate skills instead.

## Limitations

- `assemble_trocar` is inference-only; the e2e script skips finetune and checkpoint validation for it.
- `checkpoint/` outputs are produced for trainable envs only.
- Resuming requires both `--from-stage <stage>` and `--run-dir <existing-run>`.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** env not recognized - Cause: wrong `--env` name. Fix: pass a valid env name; dry-run first with `--dry-run --env <env>`.
- **Error:** resume fails to find prior outputs - Cause: `--from-stage` used without a matching `--run-dir`. Fix: pass `--from-stage <stage> --run-dir <existing-run>`.
- **Error:** stale processes block a rerun - Cause: a previous pipeline session is still running. Fix: run `stop.sh all --env <env>` before retrying.

## Final Response

Report env, run dir, skipped stages, per-stage success/failure, key artifact paths.
