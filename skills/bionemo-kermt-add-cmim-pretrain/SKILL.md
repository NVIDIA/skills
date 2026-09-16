---
name: kermt-add-cmim-pretrain
description: Convert a grover_base checkpoint (encoder-only or encoder + vocab heads) into a hybrid checkpoint by adding a randomly-initialized cMIM decoder + latent_dist, then continue pretraining on the user's corpus as hybrid (vocab + contrast). Effectively kermt-continue-pretrain with a one-time ckpt-conversion step prepended.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: workflow-skill
  risk_tier: skill
# Line/token budget: ~165 lines, ~1900 tokens — well within the
# 500-line / 5000-token cap for skill files.
---

# kermt-add-cmim-pretrain

Convert a grover_base checkpoint (legacy original-GROVER `grover.encoders.*`
or modern `kermt.encoders.*`, with or without vocab heads) into a fully-formed
hybrid (cMIM + vocab) checkpoint, then continue pretraining on the user's
corpus as hybrid.

This is a thin wrapper: `upgrade_to_hybrid.py` produces a new ckpt that
classifies as `model_type: hybrid` via `check_checkpoint.py`, and the rest of
the workflow is identical to `kermt-continue-pretrain`.

> **Status: experimental.** This workflow is functional end-to-end but has not
> been benchmarked against the manuscript's from-scratch hybrid training (which
> produces the released checkpoint). Use as an experimental alternative to
> `kermt-pretrain-scratch` when you want to extend an existing grover_base
> checkpoint rather than restart from random init. Validate downstream
> performance on your own benchmark before relying on the upgraded ckpt for
> production work.

## Skill and runtime paths

Set `SKILL_DIR` to the absolute path of this installed skill directory. Export
`KERMT_REPO` as the absolute path to the KERMT checkout used for model
execution. The bundled container helper mounts that checkout at
`/workspace` and this skill at `/skill` (read-only). Commands inside
the container use `/skill/scripts/`; defaults are bundled in `config/`.

## Hardware requirements

Same as `kermt-continue-pretrain` (the cMIM decoder adds parameters but not
substantially; VRAM headroom should be fine). The upgrade step itself is
fast (~5 s) and CPU-only — only the subsequent continue-pretrain consumes
GPU.

## When to invoke

- User has a grover_base checkpoint (encoder-only or with vocab heads) and
  wants to extend it into a hybrid (vocab + cMIM contrastive) pretrain.
- Useful for adding the SMILES-reconstruction contrastive objective to a
  pretrained encoder without restarting pretraining from scratch (which
  `kermt-pretrain-scratch` would do at days-scale).

For continuing an existing hybrid or cmim ckpt: use `kermt-continue-pretrain`
directly. For training a fresh model on a custom corpus: use
`kermt-pretrain-scratch`.

## Inputs

Required:

- `--ckpt <path>` — grover_base ckpt to upgrade. Validated via
  `check_checkpoint.py --mode upgrade_to_hybrid`; rejected if the ckpt
  already has a contrast head or task FFN.
- `--csv <path>` — pretrain corpus CSV. Same shape as
  `kermt-continue-pretrain`'s `--csv` input.

Optional (same as `kermt-continue-pretrain`):

- `--val-csv <path>` — separate validation CSV. Without it, prepare_data
  auto-splits by `--val-frac 0.1`.
- Training-hyperparameter overrides (`--epochs N`, `--batch-size N`, lr triple,
  `--warmup-epochs F`, etc.).
- `--vocab-loss-weight F` / `--latent-dim N` / `--contrastive-temperature F`.
- `--wandb-project NAME` / `--wandb-run-name NAME` — optional Weights & Biases
  logging (run name honored only alongside a project). Off by default.
- `--gpus 0,2`.

## Workflow

Let `$KERMT_REPO` be the path to your kermt repo checkout.

1. **Pre-flight: check_system** (same as `kermt-continue-pretrain` step 1).

2. **Compute run directory:**
   ```
   RUN_DIR=$KERMT_REPO/runs/add-cmim-pretrain_$(date -u +%Y-%m-%dT%H-%M-%SZ)
   ```

3. **Validate the input ckpt with `check_checkpoint --mode upgrade_to_hybrid`.**
   Abort on `ok: false`. The validator rejects ckpts that already have
   contrast head (suggest `kermt-continue-pretrain`) or task FFN heads
   (the ckpt has been finetuned; suggest using the original pretrain
   checkpoint).

4. **Validate the corpus** via `check_data --mode pretrain`. Abort on
   `ok: false`.

5. **Prepare the data** with `--mode pretrain` — *without* `--vocab-dir`.
   The upgrade builds fresh vocab heads sized to the corpus's vocab, so we
   want `prepare_data` to produce a new vocab from the corpus rather than
   passing through the ckpt's old vocab (which may not even exist for
   encoder-only legacy grover_base ckpts):
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> --run-dir $RUN_DIR -- \
       "python /skill/scripts/prepare_data.py --mode pretrain \\
            --csv /data/<basename> --out /runs/data \\
            [--val-csv /data/<val-basename>] [--val-frac 0.1] [--seed 0]"
   ```
   The output manifest has `vocab_source: "built_fresh"` and includes a
   `smiles_vocab` (built from the corpus, needed for the new decoder).

6. **Upgrade the ckpt.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --ckpt <user-ckpt> --run-dir $RUN_DIR -- \
       "python /skill/scripts/upgrade_to_hybrid.py \\
            --ckpt /ckpt \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --out /runs/upgraded.pt"
   ```
   Surface the JSON summary to the user — especially `warnings[]`, which
   includes any encoder-arch drift notes (e.g. legacy GROVER had two extra
   `act_func_*` keys that modern KERMTEmbedding doesn't) and the
   pretrain_ddp.py `--backbone` argparse-restriction note if the upgraded
   ckpt's backbone is anything other than `gtrans`.

7. **Estimate runtime + confirm with the user.** Same heuristic as
   `kermt-continue-pretrain` (corpus size × epochs × GPU count → wall time).

8. **Launch the runner detached.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run_detached \\
       --name kermt-add-cmim-pretrain-<ts> \\
       --run-dir $RUN_DIR -- \\
       "python /skill/scripts/run_pretrain_local.py \\
            --ckpt /runs/upgraded.pt \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --out /runs \\
            [--epochs N --batch-size N ...]"
   ```
   The runner sees the upgraded ckpt as `model_type: hybrid`, so it auto-dispatches
   `--pretrain_mode hybrid --vocab_loss_weight 1.0` with smiles_vocab plumbed
   through.

9. **Report to the user** with the upgraded ckpt path + the same run.json
   pointer / log path / tensorboard URL pattern as `kermt-continue-pretrain`.

## Hard rules

- **Never modify the user's input ckpt.** The upgrade writes a new file at
  `<run_dir>/upgraded.pt`; the source ckpt stays untouched.
- **Vocab heads are always fresh.** Even if the input grover_base has vocab
  heads, they're discarded and rebuilt sized to the new corpus's vocab.
  Continue-pretraining the upgraded ckpt will train those new heads alongside
  the decoder.
- **Don't auto-relax `--backbone` choices.** If the upgrade warning fires
  because the input ckpt's backbone isn't `gtrans` (e.g. legacy `dualtrans`),
  surface the warning and ask the user. Do NOT silently modify parsing.py to
  add the legacy backbone to the choices list.

## Common errors

- `check_checkpoint rejected the ckpt` with model_type=hybrid or cmim →
  user's ckpt already has a contrast head. Redirect to
  `kermt-continue-pretrain`.
- `check_checkpoint rejected the ckpt` with task_ffn=true → the ckpt has
  been finetuned. The upgrade workflow only supports pretrain checkpoints.
- `prepare manifest missing smiles_vocab` → prepare_data was invoked with
  `--skip-vocab` or some equivalent that omitted the smiles vocab. Re-run
  prepare without those flags.
- `unexpected key(s) in encoder load` warning → legacy GROVER architectures
  saved a couple of `act_func_*` weights that modern KERMTEmbedding doesn't
  use. Benign; the rest of the encoder loaded correctly.

## What's in `run.json` after a successful run

Same reproducibility fields as `kermt-continue-pretrain`, plus the upgrade step's
`summary.json` is captured under the `inputs.upgrade_summary` path so the
provenance of the upgraded ckpt is auditable.

## Replayability

Same as `kermt-continue-pretrain`: `cmd_replay` rebuilds the
`run_pretrain_local.py --ckpt <upgraded.pt> ...` invocation. To redo the
full add-cmim flow end-to-end, the user also needs the input grover_base
ckpt and the corpus — both are captured in the prepare_data and upgrade
manifests by absolute path.
