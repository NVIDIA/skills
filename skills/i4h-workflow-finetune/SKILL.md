---
name: i4h-workflow-finetune
description: Fine-tune a manifest-backed GR00T or openpi remote Task on compatible LeRobot data. Use for training; do not use for inference-only Tasks or checkpoint rollout.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - lerobot
    - gr00t
    - openpi
---

# Fine-tune a Workflow Policy Task

## Purpose

Resolve and run training from the selected workflow run mode and owning remote-task manifest.

## Instructions

1. Resolve the base checkout, policy mode, and remote task.
2. Verify dataset compatibility and a `train` block.
3. Dry-run the exact configuration.
4. Train in the foreground and verify checkpoint artifacts.

## Resolve the workflow, task, and data

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
./run.sh show <workflow> --mode <policy-mode>
test -f /absolute/path/to/dataset/meta/info.json
nvidia-smi
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Read the selected workflow run mode to identify its remote task id. Open `tasks/<project>/i4h_tasks/<project>/manifest/<task>.yaml` and require `train:`. Resolve the project, entry point, base model/config, output defaults, and modality contract from that manifest and the project's `train.py`.

Use the current-chain LeRobot dataset when the prompt omits a path. Verify its embodiment, cameras, task text, feature widths, and episode count are compatible with the remote task.

## Resolve configuration before a long run

All policy train entry points support `--dry-run`:

```bash
uv run --project "tasks/<project>" "i4h-tasks-<project-with-hyphens>-train" \
  --task <project>/<task> \
  --dataset /absolute/path/to/dataset \
  --output-dir /absolute/path/to/checkpoints \
  --max-steps <N> \
  --batch-size <N> \
  --dry-run
```

Inspect the resolved config. Keep user-requested steps, batch size, model/config, and GPU count exact.

For GR00T, “turn off vision tuning” maps to `--no-tune-visual`. Do not pass that flag to openpi, whose CLI does not expose it. Use only flags present in the selected project's current `train.py`.

## Train

Remove `--dry-run` and keep the command in the foreground:

```bash
uv run --project "tasks/<project>" "i4h-tasks-<project-with-hyphens>-train" \
  --task <project>/<task> \
  --dataset /absolute/path/to/dataset \
  --output-dir /absolute/path/to/checkpoints \
  --max-steps <N> \
  --save-steps <N> \
  --batch-size <N> \
  --num-gpus <N>
```

Add backend-specific flags only after resolving them. Do not silently lower requested steps or batch size to make training fit.

## Verify

Require exit status 0, completed requested steps, saved training logs, and at least one loadable checkpoint artifact. Resolve the exact checkpoint path rather than calling an incomplete output directory a checkpoint.

Run a bounded backend load smoke before reporting the checkpoint usable:

```bash
uv run --project "tasks/<project>" python -m "<project>.server" \
  --namespace "checkpoint-smoke-$$" \
  --preload <project>/<task> \
  --checkpoint /absolute/path/to/checkpoint \
  --preload-only
```

Use the selected project's actual module path. `--preload-only` loads the manifest and checkpoint through the inference backend, then exits without starting a rollout. A training exit alone proves that files were written, not that inference can load them.

Report `du -sh` for the task output and the selected checkpoint. Some trainers save both a final model at the output root and numbered checkpoints; identify that duplication, but do not delete either copy unless the user explicitly asks for cleanup.

Hand the exact load-smoked checkpoint path to `i4h-workflow-validate`; do not evaluate unless the user requested it.

## Troubleshooting

Report the first dataset, manifest, model-access, GPU-memory, or backend error. Preserve logs and never silently change requested hyperparameters.

## Prerequisites

Require a compatible LeRobot dataset, synced policy environment, model access, GPU capacity, and a remote-task manifest with `train:`.

## Limitations

Inference-only Tasks cannot be fine-tuned, and this skill does not claim rollout success from training alone.

## Examples

- `Fine-tune for 200 steps with a batch size of 32. Turn off vision tuning.` → preserve exact values, apply GR00T's supported vision flag, dry-run, train, and report the checkpoint.

## Completion gate

Report workflow/mode, task id and manifest, dataset compatibility, resolved config, requested/completed steps, batch/GPU/vision settings, checkpoint path, bounded load-smoke result, output/checkpoint disk sizes, exact validation handoff, exit summary, and any inference-only or resource blocker.
