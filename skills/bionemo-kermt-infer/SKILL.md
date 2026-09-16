---
name: kermt-infer
description: Run predictions with a finetuned KERMT checkpoint on a SMILES-only CSV. The skill validates that the input ckpt has task FFN heads (refuses pretrain ckpts with a redirect to kermt-finetune), validates the CSV, prepares the data (clean + rdkit_2d features), then launches main.py predict inside the kermt container (blocking, minutes-scale).
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: workflow-skill
  risk_tier: skill
# Line/token budget: targets ~170 lines / ~2000 tokens — well within the
# 500-line / 5000-token cap for skill files.
---

# kermt-infer

Run predictions with a finetuned KERMT checkpoint on a SMILES-only CSV. The
skill is the workflow orchestrator: validate ckpt, validate CSV, prepare data,
launch the runner blocking, return the predictions CSV.

## Skill and runtime paths

Set `SKILL_DIR` to the absolute path of this installed skill directory. Export
`KERMT_REPO` as the absolute path to the KERMT checkout used for model
execution. The bundled container helper mounts that checkout at
`/workspace` and this skill at `/skill` (read-only). Commands inside
the container use `/skill/scripts/`; defaults are bundled in `config/`.

## Hardware requirements

- **GPUs**: 1 (single-GPU). Multi-GPU inference is not currently supported.
- **VRAM**: ≥ 4 GB for the default `batch_size 32`.
- **Disk**: a few hundred MB per run (cleaned CSV + features + predictions).
- **Driver / CUDA**: any host supporting CUDA 12.6 (the kermt image base).

## Inputs

Required:

- `--ckpt <path>` — finetuned checkpoint (must have task FFN heads). The
  validator refuses pretrain ckpts with a redirect to `kermt-finetune`.
- `--csv <path>` — SMILES-only CSV. First column is `smiles`; other columns
  are ignored.

Optional:

- `--batch-size N` — override the configured default (32).
- `--seed N` — random seed for inference (deterministic featurization paths).
- `--gpus 0` — single GPU id (default 0). Multi-GPU rejected.
- `--from-prepare <dir>` — skip the prepare step and reuse an existing
  `prepare_data.json` in `<dir>`.

## Workflow

Let `$KERMT_REPO` be the path to your kermt repo checkout, and assume
`kermt-setup` has built `kermt:latest`.

1. **Pre-flight: ensure container + system probe.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" check_system
   ```
   Refuse to proceed on `ok: false`.

2. **Compute run directory.**
   ```
   RUN_DIR=$KERMT_REPO/runs/infer_$(date -u +%Y-%m-%dT%H-%M-%SZ)
   ```

3. **Validate the checkpoint.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --ckpt <user-ckpt> -- \
       "python /skill/scripts/check_checkpoint.py --mode inference --ckpt /ckpt"
   ```
   Parse the JSON. Abort on `ok: false`. The validator rejects pretrain ckpts
   (`has_task_ffn: false`) with a redirect to `kermt-finetune`.

4. **Validate the data.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> -- \
       "python /skill/scripts/check_data.py --mode inference --csv /data/<basename>"
   ```
   Abort on `ok: false`.

5. **Prepare the data.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> --run-dir $RUN_DIR -- \
       "python /skill/scripts/prepare_data.py --mode inference \\
            --csv /data/<basename> --out /runs/data"
   ```
   Outputs land at `$RUN_DIR/data/prepare_data.json` with `clean_csv` +
   `clean_npz` paths (rdkit_2d_normalized features).

6. **Launch the runner (blocking).**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run \\
       --ckpt <user-ckpt> --run-dir $RUN_DIR -- \\
       "python /skill/scripts/run_inference.py \\
            --ckpt /ckpt \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --out /runs \\
            [--gpus 0 --batch-size N --seed N]"
   ```
   Returns the predictions CSV path on success.

7. **Report to the user.** Output a short summary:
   - Predictions: `$RUN_DIR/out/predictions.csv` (smiles + per-target columns)
   - Manifest: `$RUN_DIR/run.json` (cmd_replay + image digest + applied args)
   - Log: `$RUN_DIR/logs/inference.log`
   - Row count: <N> molecules predicted across <K> targets

## Hard rules

- **Never modify the user's ckpt.** The runner symlinks the ckpt into a
  unique `<out>/ckpt_link/` subdir so `main.py predict --checkpoint_dir`
  picks it up; the source file stays untouched.
- **Arch comes from the ckpt, never from CLI/defaults.** The runner records
  the validator's arch block in `run.json` but does not pass arch flags into
  `main.py predict` — predict reads them from the loaded ckpt's saved_args.
- **Single-GPU only.** Multi-GPU inference is not currently supported.
- **Echo applied defaults.** The `args_applied` field of `run.json` records
  every flag's value + source (user / default-config). Surface a short
  summary of any default-filled flag.

## Common errors

- `inference requires a finetuned ckpt with task FFN heads` → ckpt is a
  pretrain ckpt; use `kermt-finetune` first.
- `prepare_data manifest reports ok=False` → check the manifest `errors` for
  the failed step (typically clean_smiles or save_features).
- `could not convert string to float: '<value>'` from save_features or main.py
  predict → input CSV has a non-numeric passthrough column (e.g. a 'split'
  label). The prep step now strips the CSV to SMILES-only at inference; if
  this error still surfaces, the CSV is being read by a runner that bypassed
  prepare_data. Re-run via the skill, not `main.py` directly.
- `--gpus '0,1' is single-GPU only` → pass a single id.

## Replayability

The `run.json` `cmd_replay` field is a single-line command that re-runs the
inference with the same inputs. To replay inside the kermt container:

```bash
$(jq -r .cmd_replay $RUN_DIR/run.json)
```

If `ok_to_replay: false` (dirty kermt repo worktree at launch time), pin
the commit via `repo.commit` and `git checkout` it first.
