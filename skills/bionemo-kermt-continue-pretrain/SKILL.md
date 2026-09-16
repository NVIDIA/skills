---
name: kermt-continue-pretrain
description: Continue KERMT pretraining on a custom SMILES corpus with a grover_base, cmim, or hybrid checkpoint. Use a local checkpoint or optionally download a pinned Hugging Face model bundle using HF_TOKEN if configured. Run containerized training and write model bundles, prepared data, logs, and checkpoints to user-selected host directories.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: workflow-skill
  risk_tier: skill
# Line/token budget: this file is targeted at ~250 lines / ~3000 tokens —
# well within the 500-line / 5000-token cap. Long examples live in
# /skill/scripts/run_pretrain_local.py's docstring.
---

# kermt-continue-pretrain

Continue pretraining from a user-supplied KERMT checkpoint (grover_base /
cmim / hybrid). The skill is the workflow orchestrator: it validates inputs,
prepares the corpus, launches the runner, and returns a run directory.

## Skill and runtime paths

Set `SKILL_DIR` to the absolute path of this installed skill directory. Export
`KERMT_REPO` as the absolute path to the KERMT checkout used for model
execution. The bundled container helper mounts that checkout at
`/workspace` and this skill at `/skill` (read-only). Commands inside
the container use `/skill/scripts/`; defaults are bundled in `config/`.
See [Released models](references/released-models.md) for checkpoint bundle requirements.

## Downloads and local outputs

The optional released-model branch reads `config/released_model.json` for the
Hugging Face repository, pinned revision, and filenames. The bundled
`scripts/fetch_released_model.py` downloads the model bundle over HTTPS into
the host directory the user selects. Public models work without credentials;
if `HF_TOKEN` is set, the container helper forwards it for Hugging Face
authentication. Prepared data, logs, and workflow results go into the chosen
run directory.

## Hardware requirements

- **GPUs**: 1–N CUDA-capable NVIDIA GPUs. The runner auto-detects via
  `torch.cuda.device_count()`; `--gpus 0,2` overrides. On a single GPU the
  runner falls back to `--batch_size 32 --save_interval 500`; on multi-GPU
  it uses the `defaults_pretrain.json` values (currently `batch_size 256`).
  Note: `--gpus N` uses **torch.cuda** indexing, which can differ from
  `nvidia-smi`'s display order on multi-GPU hosts (PCI bus vs. CUDA
  enumeration). To target a specific physical GPU, set `CUDA_VISIBLE_DEVICES`
  before invoking, or run
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
- **Disk**: tens of GB depending on corpus size + epochs (each checkpoint
  is several hundred MB).
- **Driver / CUDA**: any host supporting CUDA 12.6 (the kermt image base).
  `kermt-setup` validates this up-front.

## Inputs

Required:

- `--csv <path>` — the pretrain CSV (single column `smiles`). If you have
  separate train/val CSVs, pass `--val-csv <path>` too.

Checkpoint (optional — defaults to the released model if omitted):

- `--ckpt <path>` — the input pretrain checkpoint to continue from. Must be
  a grover_base (with vocab heads), cmim, or hybrid ckpt; the validator
  rejects everything else with a redirect to the correct workflow. **If
  omitted**, the skill offers to download the released pretrained hybrid model
  **nvidia/NV-KERMT-70M-v2** and continue-pretrain from it — see "Resolve &
  validate the checkpoint" (workflow step 3). The released bundle ships its
  three vocab files alongside the ckpt, so the authoritative-vocab pass-through
  (step 5) works automatically.
- `--pretrained-release` — explicit opt-in to use the released model without
  the interactive prompt (for non-interactive / agent runs). Mutually
  exclusive with `--ckpt`.
- `--model-dir <dir>` — where to save the downloaded bundle (default
  `$KERMT_REPO/models/NV-KERMT-70M-v2/`). An already-complete bundle there is
  reused, not re-downloaded.

Optional:

- `--val-csv <path>` — separate validation CSV. Without it, the prep step
  auto-splits the input by `--val-frac 0.1` (random shuffle with `--seed`).
- `--epochs N` / `--batch-size N` / `--init-lr F` / `--max-lr F` /
  `--final-lr F` / `--warmup-epochs F` / `--weight-decay F` / `--dropout F` /
  `--save-interval N` / `--seed N` — training-hyperparameter overrides.
  Anything not given is filled from `config/defaults_pretrain.json`.
- `--vocab-loss-weight F` (hybrid only) / `--latent-dim N` /
  `--contrastive-temperature F` (cmim and hybrid only) — loss / decoder
  overrides.
- `--wandb-project NAME` / `--wandb-run-name NAME` — optional Weights & Biases
  logging. When `--wandb-project` is set, rank 0 logs train/val losses; the run
  name is honored only alongside a project. Off by default. (Independent of the
  ckpt's `wandb_run_id` continuity handling under `--resume`.)
- `--resume` — see "Modes" section below.
- `--gpus 0,2` — restrict to a GPU subset. Default uses all visible GPUs.
- `--from-prepare <dir>` — skip the prepare step and reuse an existing
  `prepare_data.json` in `<dir>`. Useful when iterating on hyperparameters.

## Modes

The runner has two modes for ingesting the input ckpt, dispatched on whether
`--resume` is set. Pick based on intent:

### Default (fresh-schedule continue-pretrain)

**Use when**: you have a finished pretrain ckpt and want to continue training
it — on a new corpus, with a different objective, or just for more epochs
than its original plan. The previous training's step counter and schedule
shape are no longer relevant; you want a new learning-rate schedule for the
new run.

**What gets loaded from the ckpt**:
- ✓ Model weights (encoder + vocab heads + contrast head + decoder, whatever
  is there)
- ✓ Optimizer state (Adam's running m1/m2 moments — warm-starts the new
  schedule so the first few hundred steps aren't dominated by noisy
  gradient-estimate startup)
- ✗ Scheduler step counter (reset to 0)
- ✗ Epoch counter (reset to 0)
- ✗ Batch counter (reset to 0)
- ✗ wandb run id (new wandb run, not a continuation)

**Schedule shape** (init/max/final LR, warmup epochs, total epochs): from
your CLI args or `defaults_pretrain.json`. A fresh NoamLR is constructed
from these values and starts at step 0.

### `--resume` (true resume)

**Use when**: a previous run was interrupted (crash, OOM, Ctrl-C) and you
want to pick up exactly where it left off — same dataset, same schedule,
same training trajectory.

**What gets loaded from the ckpt**: **everything** in the
`save_model_for_restart` format. Model weights + optimizer state +
scheduler_step + epoch + batch_idx + wandb_run_id are all restored. The
new run continues from the saved step in the saved schedule (which is
recovered from the ckpt's `saved_args`). Mid-epoch resume works too —
`pretrain_ddp.py`'s sampler skip-count picks up at the saved batch index
within the saved epoch.

**Schedule shape**: inherited from the ckpt's `saved_args`. CLI overrides
of any schedule flag (`--epochs / --warmup-epochs / --init-lr / --max-lr /
--final-lr`) are **rejected with a hard error** — pure resume means pure
resume; if you want to change the schedule, drop `--resume` and start a
fresh-schedule run.

**Requirements**: the ckpt must have been saved via `save_model_for_restart`
(i.e., carry `optimizer / scheduler_step / epoch / batch_idx` keys). If
any of these is missing, the runner errors with a clear message and
suggests dropping `--resume`.

The default mode is the right choice ~90% of the time. Reach for `--resume`
only when you genuinely need to continue a single interrupted training
run.

## Workflow

Let `$KERMT_REPO` be the path to your kermt repo checkout, and assume
`kermt-setup` has already built `kermt:latest`. All paths below are on the
host; the helper bind-mounts them at known container paths.

1. **Pre-flight: ensure container + system probe.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" check_system | python -c "
   import json, sys; d = json.load(sys.stdin)
   if not d['ok']:
       print('System check failed:', d['gaps']); sys.exit(1)
   print(f'OK: {len(d[\"gpus\"])} GPU(s); {d[\"disk\"][\"free_gb\"]} GB free; CUDA via container toolkit')
   "
   ```
   Surface any `gaps` to the user. Refuse to proceed if `ok: false`.

2. **Compute run directory.**
   ```
   RUN_DIR=$KERMT_REPO/runs/continue-pretrain_$(date -u +%Y-%m-%dT%H-%M-%SZ)
   ```

3. **Resolve & validate the checkpoint.**

   **Resolve — only if `--ckpt` was omitted.** Default to the released
   pretrained hybrid model **nvidia/NV-KERMT-70M-v2**:
   - **Consent gate.** Unless `--pretrained-release` was passed, ask the user:
     "No checkpoint given — download the released model nvidia/NV-KERMT-70M-v2
     (NVIDIA Open Model License, https://huggingface.co/nvidia/NV-KERMT-70M-v2)
     and continue-pretrain from it? [y/N]". **Never download without an
     explicit yes** (or `--pretrained-release`). If both `--ckpt` and
     `--pretrained-release` are given, abort — they conflict.
   - **Save location.** Default `$KERMT_REPO/models/NV-KERMT-70M-v2/`; honor
     `--model-dir <dir>` if given. An already-complete bundle is reused.
   - **Download** (foreground; ~282 MB on first fetch):
     ```
     "$SKILL_DIR/scripts/kermt_container.sh" run --model-dir <save-dir> -- \
         "python /skill/scripts/fetch_released_model.py --out /model"
     ```
     Parse the JSON; abort on `ok: false` (surface `errors`). On success set
     `<user-ckpt> = <save-dir>/kermt_contrastive_v2.0.pt`. The bundle's three
     vocab files land in `<save-dir>` too, so step 5's `--vocab-dir`
     auto-detection (which looks in the ckpt's parent directory) finds them
     with no extra work.

   **Validate** the resolved (or user-provided) ckpt:
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --ckpt <user-ckpt> -- \
       "python /skill/scripts/check_checkpoint.py --mode continue_pretrain --ckpt /ckpt"
   ```
   Parse the JSON. Abort on `ok: false`, showing the error verbatim. The error
   message redirects the user to `kermt-add-cmim-pretrain` for encoder-only
   ckpts, or to `kermt-finetune` for finetuned ckpts.

4. **Validate the data.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> -- \
       "python /skill/scripts/check_data.py --mode pretrain --csv /data/<basename>"
   ```
   Abort on `ok: false`.

5. **Prepare the data** (skip if `--from-prepare` given).
   **Pass the ckpt's vocab through.** Look in the ckpt's parent directory for
   the conventional `pretrain_atom_vocab.{json,pkl}`, `pretrain_bond_vocab.{json,pkl}`,
   and `pretrain_smiles_vocab.pkl` files (the bundling convention for released
   models; see `references/released-models.md`). If all three are
   present, auto-pass via `--vocab-dir <ckpt_parent_dir>`. If only some are
   present, pass them via explicit flags (`--atom-vocab`, `--bond-vocab`,
   `--smiles-vocab`). If none are present, ask the user for `--vocab-dir` — or
   refuse to proceed, because rebuilding a fresh vocab from the new corpus
   would silently mismatch the ckpt's vocab heads (the ckpt's vocab is
   authoritative for continue-pretrain).

   Note the **two-layer mount pattern**: pass the host directory to
   `kermt_container.sh --vocab-dir` (which mounts it at `/vocab` inside the
   container), and reference `/vocab` from the inner `prepare_data.py`
   command. The same pattern applies to every host path the inner command
   needs to read (`--data <host-csv>` → `/data/<basename>`,
   `--ckpt <host-ckpt>` → `/ckpt`).

   ```
   VOCAB_DIR=$(dirname <user-ckpt>)
   "$SKILL_DIR/scripts/kermt_container.sh" run \
       --data <user-csv> --vocab-dir $VOCAB_DIR --run-dir $RUN_DIR -- \
       "python /skill/scripts/prepare_data.py --mode pretrain \\
            --csv /data/<basename> --out /runs/data \\
            --vocab-dir /vocab \\
            [--val-csv /data/<val-basename>] [--val-frac 0.1] [--seed 0]"
   ```
   Outputs land at `$RUN_DIR/data/prepare_data.json` with
   `vocab_source: "user_provided"`. The runner step 7 will verify the vocab
   files' entry counts match the ckpt's vocab-head sizes and refuse to launch
   on mismatch.

6. **Estimate runtime + confirm with user.**
   - Pretrain wall time depends on corpus size × epochs × GPU count.
   - Tell the user the estimate; ask "proceed?" unless `--yes` flag was given
     (agent-non-interactive case).
   - Example estimate template:
     `~N hours on K GPUs for E epochs over M molecules (~steps/epoch × seconds/step)`.

7. **Launch the runner detached.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run_detached \\
       --name kermt-continue-pretrain-<ts> \\
       --ckpt <user-ckpt> --run-dir $RUN_DIR -- \\
       "python /skill/scripts/run_pretrain_local.py \\
            --ckpt /ckpt \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --out /runs \\
            [--epochs N --batch-size N --init-lr F ...]"
   ```
   Returns the container name + id + log file path.

8. **Report to the user.** Output a short summary:
   - Container name + id
   - `$RUN_DIR/run.json` (the manifest with cmd_replay + image digest)
   - Log file: `$RUN_DIR/logs/pretrain_ddp.log`
   - TensorBoard: `$RUN_DIR/logs/tb` (open with `tensorboard --logdir
     $RUN_DIR/logs/tb`)
   - Suggest invoking `kermt-monitor <RUN_DIR>` to check progress.

## Hard rules

- **Never download the released model without consent.** When `--ckpt` is
  omitted, download `nvidia/NV-KERMT-70M-v2` only after an explicit user "yes"
  or an explicit `--pretrained-release` flag. `--ckpt` and
  `--pretrained-release` are mutually exclusive.
- **Never modify the user's input ckpt.** The runner symlinks it into the
  save_dir; the symlink is what pretrain_ddp.py auto-resumes from. The
  source file stays untouched.
- **Never silently override arch.** If the user passes a `--hidden-size`
  etc. that doesn't match the ckpt-derived value, the runner aborts loudly.
  Arch params come from the ckpt, period.
- **Never block on the long-running pretrain itself.** The runner is invoked
  via `run_detached`; the skill returns immediately after step 8. Use
  `kermt-monitor` for progress.
- **Echo applied defaults back to the user.** The `args_applied` field of
  `run.json` records every flag's value + source (user / default-config /
  auto-1gpu / auto-multi-gpu). Skill should surface a summary of any flag
  not user-specified so the user knows what was assumed.

## Common errors

- `model_type='finetuned'` rejected → the ckpt is a downstream finetune,
  not a pretrain. The error redirects to the relevant workflow.
- `grover_base ckpt has no vocab head` → encoder-only ckpt (e.g. the
  original-grover `grover_base.pt`). The error redirects to
  `kermt-add-cmim-pretrain`.
- `prepare_data manifest is missing required outputs` → user passed
  `--from-prepare` to a directory where prepare was run with `--skip-vocab`
  or `--skip-split`. Re-run prepare without those flags.
- `--gpus all` not available → install `nvidia-container-toolkit`; check
  `kermt_container.sh check_system`.

## Replayability

The `run.json` `cmd_replay` field is a single-line command that re-runs the
pretrain with the same inputs, hyperparameters, and arch. To replay:

```bash
# Inside the kermt container:
$(jq -r .cmd_replay $RUN_DIR/run.json)
```

If `ok_to_replay: false` in the manifest (because the kermt repo working
tree was dirty at launch time), the replay may not be bit-exact — pin the
exact commit via the `repo.commit` field and `git checkout` it
first.
