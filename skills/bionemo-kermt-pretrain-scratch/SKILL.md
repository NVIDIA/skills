---
name: kermt-pretrain-scratch
description: Pretrain a fresh KERMT model from scratch on a user-provided corpus. Builds a new vocabulary from the corpus, instantiates the model architecture from defaults, and launches pretrain_ddp.py inside the kermt container (detached for long runs). Unlike kermt-continue-pretrain, no starting checkpoint is loaded — the model is randomly initialized.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: workflow-skill
  risk_tier: skill
# Line/token budget: ~210 lines, ~2400 tokens — well within the
# 500-line / 5000-token cap for skill files. Most of the orchestration is
# shared with kermt-continue-pretrain; the differences are documented below.
---

# kermt-pretrain-scratch

Pretrain a brand-new KERMT model from scratch on a user-provided corpus. Useful
when you want to retrain a model on a custom chemistry domain rather than
extending one of the released checkpoints. **Significantly more expensive than
`kermt-continue-pretrain`** — no warm start, so the loss curves need to descend
from scratch over many epochs.

## Skill and runtime paths

Set `SKILL_DIR` to the absolute path of this installed skill directory. Export
`KERMT_REPO` as the absolute path to the KERMT checkout used for model
execution. The bundled container helper mounts that checkout at
`/workspace` and this skill at `/skill` (read-only). Commands inside
the container use `/skill/scripts/`; defaults are bundled in `config/`.

## Hardware requirements

Same as `kermt-continue-pretrain`:

- **GPUs**: 1–N CUDA-capable. The runner auto-detects via
  `torch.cuda.device_count()`; `--gpus 0,2` overrides. Single-GPU fallback:
  `--batch_size 32 --save_interval 500`. Multi-GPU keeps defaults
  (`--batch_size 256` etc.). Note: `--gpus N` uses **torch.cuda** indexing,
  which can differ from `nvidia-smi`'s display order on multi-GPU hosts
  (PCI bus vs. CUDA enumeration). To target a specific physical GPU, set
  `CUDA_VISIBLE_DEVICES` before invoking, or run
  `python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"`
  to confirm which device you're picking.
- **VRAM**: the default `--batch-size 256` is sized for A100-class hardware
  (80 GB VRAM). On smaller GPUs, downscale to avoid OOM:

  | GPU class                | VRAM       | Suggested `--batch-size` |
  |--------------------------|------------|--------------------------|
  | L4, T4, V100 16 GB       | 16–24 GB   | 32–64                    |
  | A100 40 GB, L40, A40     | 40–48 GB   | 128                      |
  | A100 80 GB, H100, H200   | 80 GB      | 256 (default)            |

  These are rough starting points — pass `--batch-size N` to override.
- **Disk**: tens of GB for shards + vocab + checkpoints, scaled by epochs.
- **Wall time**: this is the big difference. Pretraining from scratch on an
  11M-mol corpus at 100 epochs typically takes **days even on a multi-GPU box**.
  The skill prints an estimate before launching; confirm with the user.

## When to invoke

- User wants to train a new model on a custom corpus (e.g. domain-specific
  chemistry that the released ckpts don't cover).
- User wants to reproduce a pretrain config end-to-end without depending on a
  released ckpt.

For continuing an existing released ckpt, use `kermt-continue-pretrain`. For
adding a cMIM decoder to an encoder-only grover_base ckpt, use
`kermt-add-cmim-pretrain`.

## Inputs

Required:

- `--csv <path>` — the pretrain corpus CSV with a `smiles` column. Single file
  by convention; multi-file corpora deferred. Use `--val-csv` for a separate
  validation set.
- `--pretrain-target-mode {vocab|cmim|hybrid}` — which pretrain objective to
  use. **No default** — must be set explicitly so the user makes an informed
  choice:
  - `vocab` — original GROVER-style atom + bond vocab prediction (encoder-only
    output, lightweight).
  - `cmim` — contrastive + SMILES reconstruction objective. Requires building
    a SMILES vocab from the corpus.
  - `hybrid` — both vocab and contrastive objectives jointly (the
    state-of-the-art config from the KERMT manuscript).

Optional:

- `--val-csv <path>` — separate validation CSV. Without it, prepare_data
  auto-splits the input by `--val-frac 0.1` (random shuffle with `--seed`).
- Training-hyperparameter overrides: `--epochs N` / `--batch-size N` /
  `--init-lr F` / `--max-lr F` / `--final-lr F` / `--warmup-epochs F` /
  `--weight-decay F` / `--dropout F` / `--save-interval N` / `--seed N`.
  Anything not given is filled from `config/defaults_pretrain.json`.
- `--vocab-loss-weight F` (hybrid only) / `--latent-dim N` /
  `--contrastive-temperature F` (cmim and hybrid only).
- `--wandb-project NAME` / `--wandb-run-name NAME` — optional Weights & Biases
  logging. When `--wandb-project` is set, rank 0 logs train/val losses; the run
  name is honored only alongside a project. Off by default.
- `--gpus 0,2` — restrict to a GPU subset.

## Workflow

Let `$KERMT_REPO` be the path to your kermt repo checkout.

1. **Pre-flight: ensure container + system probe** (same as
   `kermt-continue-pretrain` step 1). Refuse to proceed if `check_system`
   reports gaps.

2. **Compute run directory.**
   ```
   RUN_DIR=$KERMT_REPO/runs/pretrain-scratch_$(date -u +%Y-%m-%dT%H-%M-%SZ)
   ```

3. **Validate the corpus** (no ckpt to validate, so this is the only input
   check):
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> -- \
       "python /skill/scripts/check_data.py --mode pretrain --csv /data/<basename>"
   ```
   Abort on `ok: false`.

4. **Prepare the data** — no vocab pass-through (we want fresh vocab from
   corpus):
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> --run-dir $RUN_DIR -- \
       "python /skill/scripts/prepare_data.py --mode pretrain \\
            --csv /data/<basename> --out /runs/data \\
            [--val-csv /data/<val-basename>] [--val-frac 0.1] [--seed 0]"
   ```
   Outputs land at `$RUN_DIR/data/prepare_data.json` with
   `vocab_source: "built_fresh"`.

5. **Estimate runtime + warn loudly.** This is critical for pretrain-from-scratch:
   - "Pretraining from scratch is days-scale even on multi-GPU; the released
     KERMT checkpoints were each trained on millions of molecules for hundreds
     of GPU-hours. If you mainly want to leverage existing knowledge for a
     downstream task, consider `kermt-continue-pretrain` from a released ckpt
     instead, which converges in hours instead of days."
   - Show the corpus size × epochs × GPU count → estimated wall time.
   - Ask for explicit confirmation unless `--yes` was given.

6. **Launch the runner detached.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run_detached \\
       --name kermt-pretrain-scratch-<ts> \\
       --run-dir $RUN_DIR -- \\
       "python /skill/scripts/run_pretrain_local.py \\
            --from-scratch --pretrain-target-mode <vocab|cmim|hybrid> \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --out /runs \\
            [--epochs N --batch-size N ...]"
   ```
   Note: NO `--ckpt` flag (the runner refuses if both `--from-scratch` and
   `--ckpt` are given). The runner uses the `arch` group from
   `config/defaults_pretrain.json` to size the model.

7. **Report to the user.** Always include all of the following — do not
   omit the TensorBoard line under output-length pressure:
   - Container name + id
   - `$RUN_DIR/run.json` (the manifest with `workflow: pretrain-scratch`,
     `from_scratch: true`, `vocab_check: null`, `arch` from defaults, full
     `cmd_replay`)
   - Log file: `$RUN_DIR/logs/pretrain_ddp.log`
   - TensorBoard: `$RUN_DIR/logs/tb` (open with `tensorboard --logdir
     $RUN_DIR/logs/tb`)
   - Suggest `kermt-monitor <RUN_DIR>` for progress.

## Hard rules

- **Never accept a `--ckpt` flag.** From-scratch is exclusive with input
  ckpt — the runner enforces this; the skill should too.
- **Never silently default `--pretrain-target-mode`.** This is a significant
  architectural choice (vocab = lightweight, hybrid = SOTA). Prompt the user
  if not given on the CLI.
- **Strong warning before launching.** From-scratch pretrain is the most
  expensive workflow. The user needs to know what they're committing to.

## Common errors

- `--pretrain-target-mode is required when --from-scratch is set` → user
  forgot the mode flag. Prompt.
- `--from-scratch is incompatible with --ckpt` → user provided both; ask which
  one they meant.
- `defaults_pretrain.json has no arch group` → repo state issue (should never
  happen on a fresh clone); points the user at running `kermt-setup` again.

## What's in the manifest after a from-scratch run

Same reproducibility fields as continue-pretrain (`repo.commit`, `kermt_image`,
`cmd_replay`, `args_applied`), plus:

- `workflow`: `"pretrain-scratch"`
- `from_scratch`: `true`
- `inputs.ckpt`: `null`
- `ckpt_symlink`: `null`
- `vocab_check`: `null` (not verified — vocab built from corpus is
  authoritative for from-scratch)
- `arch`: the values pulled from `config/defaults_pretrain.json`'s
  `arch` group (with any future CLI overrides applied).

## Replayability

Same as continue-pretrain: `cmd_replay` is a copy-pasteable command. If
`ok_to_replay: false`, the kermt repo working tree was dirty at launch
time — check `repo.commit` and `git checkout` it first.
