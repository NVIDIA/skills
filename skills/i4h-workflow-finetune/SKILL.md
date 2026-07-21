---
name: i4h-workflow-finetune
version: "0.6.0"
description: Fine-tune a GR00T or openpi PI0 policy on a LeRobot dataset. Use when asked to finetune, train, or post-train a policy on demos; not for evaluating a checkpoint (use [[i4h-workflow-validate]]).
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - finetune
    - training
---

# i4h Workflow — Finetune

## Purpose

Fine-tune a GR00T or openpi PI0 policy on an existing LeRobot dataset. Use when asked to finetune, train, or post-train a policy on recorded demos.

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

- The dataset path must be an existing LeRobot directory with `meta/info.json`.
- Train support is determined by `policy.train_module` in `workflows/agentic/config/environments/<env>.yaml`. A null value means inference-only.
- `assemble_trocar` is inference-only.

## Stack Map

| Env | Stack | CLI |
|---|---|---|
| `scissor_pick_and_place` | `gr00t_n15` | `i4h-agentic-gr00t-n15-train` |
| `locomanip_tray_pick_and_place` | `gr00t_n16` | `i4h-agentic-gr00t-n16-train` |
| `locomanip_push_cart` | `gr00t_n16` | `i4h-agentic-gr00t-n16-train` |
| `ultrasound_liver_scan` | `openpi_pi0` | `i4h-agentic-openpi-pi0-train` |

N1.6 locomanip envs share `policy.locomanip.train`.

## Preflight

```bash
test -f "${DATASET_PATH}/meta/info.json"
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
workflows/agentic/policy/<stack>/run.sh --list-envs
```

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup and resolve dataset

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
STACK_DIR=gr00t_n15
TRAIN_CLI=i4h-agentic-gr00t-n15-train
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# Point DATASET_PATH at a converted LeRobot dataset dir (absolute; must contain meta/info.json),
# produced by [[i4h-workflow-dataset-convert]]. List candidates:
#   find "${RUNS_ROOT}" "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" -name info.json -path '*/meta/*' -printf '%h\n' | sed 's#/meta$##' | sort -u
DATASET_PATH="${DATASET_PATH:-}"
if [ ! -f "${DATASET_PATH%/}/meta/info.json" ]; then
  echo "finetune: set DATASET_PATH to a LeRobot dataset dir with meta/info.json (got '${DATASET_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" -name info.json -path '*/meta/*' -printf '%h\n' 2>/dev/null | sed 's#/meta$##' | sort -u | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/finetune_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
OUT="${RUN_DIR}/checkpoint"
export TMPDIR=/tmp   # short path: torch DataLoader FD-sharing socket must fit AF_UNIX's 108-byte limit
mkdir -p "${OUT}" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — train

```bash
uv --directory "${REPO_ROOT}/workflows/agentic/policy/${STACK_DIR}" run "${TRAIN_CLI}" \
  --env "${ENV_ID}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUT}" \
  --max-steps 1000 \
  --save-steps 1000 \
  --num-gpus 1 \
  2>&1 | tee "${RUN_DIR}/logs/finetune.log"
```

Tyro flags use kebab case (`--max-steps`, not `--max_steps`).

## Common Flags

- `--dataset-path PATH` (required)
- `--output-dir PATH`
- `--base-model-path PATH_OR_REPO` overrides YAML `policy.model_repo`
- `--max-steps N`, `--save-steps N`
- `--batch-size N`, `--learning-rate FLOAT`
- `--no-tune-visual` — freeze the vision backbone (trains the action head + projector only): ~2× faster, ~half the memory, less overfitting. Good default for small datasets; unfreeze only with lots of data + a real visual domain gap.
- `--num-gpus N` — must not exceed visible GPUs
- `--report-to tensorboard|wandb`

## Verify

- Checkpoint directory `${OUT}/checkpoint-<N>` contains `model-0000*-of-*.safetensors`, `experiment_cfg/`, `processor/`.
- Log contains `train_loss` lines and a final `'train_runtime': ...` summary.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the stack's `.venv` must exist).
- An existing LeRobot dataset directory with `meta/info.json`.
- A train-capable env: `policy.train_module` non-null in `workflows/agentic/config/environments/<env>.yaml` (`assemble_trocar` is inference-only).
- At least one visible GPU (`--num-gpus` must not exceed visible GPUs).

## Limitations

- Inference-only envs (null `policy.train_module`, e.g. `assemble_trocar`) cannot be fine-tuned.
- Requires GPU(s); `--num-gpus` must not exceed the count from `nvidia-smi`.
- N1.6 locomanip envs share `policy.locomanip.train`.
- Each env maps to one stack/CLI (see Stack Map); the dataset must match that env.

## Troubleshooting

- **Error:** train CLI / module import fails - Cause: workflow not set up, stack `.venv` missing. Fix: run [[i4h-workflow-setup]] first.
- **Error:** dataset path rejected / missing `meta/info.json` - Cause: `--dataset-path` is not a valid LeRobot directory. Fix: point to a converted LeRobot dataset (see Preflight `test -f`).
- **Error:** env is inference-only / no train support - Cause: `policy.train_module` is null for that env. Fix: choose a train-capable env from the Stack Map.
- **Error:** unrecognized flag like `--max_steps` - Cause: Tyro flags use kebab case. Fix: use `--max-steps` form.

## Final Response

Report env, stack, dataset path, output checkpoint path, train_loss summary, and blockers.
