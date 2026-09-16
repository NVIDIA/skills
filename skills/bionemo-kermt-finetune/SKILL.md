---
name: kermt-finetune
description: Finetune a pretrained KERMT encoder on a labeled CSV. Validate the checkpoint and data, prepare features, and run containerized training. Use a local checkpoint or optionally download a pinned Hugging Face model bundle using HF_TOKEN if configured. Write model bundles, prepared data, logs, and trained models to user-selected host directories.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: workflow-skill
  risk_tier: skill
# Line/token budget: targets ~250 lines / ~3000 tokens — within the
# 500-line / 5000-token cap for skill files.
---

# kermt-finetune

Finetune a pretrained KERMT encoder on a user-supplied labeled CSV. The skill
is the workflow orchestrator: validate ckpt, validate data, prepare data,
launch the runner detached, return a run directory + container name.

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

- **GPUs**: 1 by default (single-GPU); pass `--gpus 0` (or whichever id) to
  select one. For faster training on a multi-GPU host, pass `--num-gpus N`
  (N>1) to run data-parallel DDP across N GPUs — `--batch-size` is then
  per-GPU (effective global batch = batch_size × N).
- **VRAM**: ≥ 8 GB for the default `batch_size 32` configuration. Lower VRAM
  works at smaller batch sizes — pass `--batch-size N` to override.
- **Disk**: a few GB per run (checkpoint + features + logs).
- **Driver / CUDA**: any host supporting CUDA 12.6 (the kermt image base).
  `kermt-setup` validates this up-front.

## Inputs

Required:

- `--csv <path>` — labeled CSV. First column is `smiles`; every other column
  is a target.

Checkpoint (optional — defaults to the released model if omitted):

- `--ckpt <path>` — input pretrain checkpoint (grover_base / cmim / hybrid).
  The validator refuses already-finetuned ckpts with a redirect to
  `kermt-infer`. **If omitted**, the skill offers to download the released
  pretrained hybrid model **nvidia/NV-KERMT-70M-v2** and finetune from it —
  see "Resolve & validate the checkpoint" (workflow step 3).
- `--pretrained-release` — explicit opt-in to use the released model without
  the interactive prompt (for non-interactive / agent runs). Mutually
  exclusive with `--ckpt`.
- `--model-dir <dir>` — where to save the downloaded bundle (default
  `$KERMT_REPO/models/NV-KERMT-70M-v2/`). An already-complete bundle there is
  reused, not re-downloaded.

Optional:

- `--dataset-type {regression | classification | multiclass}` — default
  `regression` (from `defaults_finetune.json`). Drives loss, metric defaults,
  and head initialization. For classification tasks pass
  `--dataset-type classification`.

- `--targets COL [COL ...]` — explicit target column names. If omitted, the
  validator auto-detects numeric non-smiles columns and the skill confirms
  with the user before proceeding.
- `--val-csv <path>` and `--test-csv <path>` — user-provided val + test
  splits. Either pass both or pass neither (the skill auto-splits using the
  configured `--split-type`).
- `--split-type {random | scaffold_balanced | index_predetermined}` —
  default `scaffold_balanced` from `defaults_finetune.json`.
  - `random` and `scaffold_balanced`: build the val/test split internally
    from the train CSV. No `--val-csv` / `--test-csv` needed.
  - `index_predetermined`: **requires** pre-split CSVs passed via
    `--val-csv` + `--test-csv` (and, separately, per-fold index files —
    see `kermt/util/utils.split_data`). Use this when the dataset ships
    its own canonical split (e.g. `tests/data/Biogen_for_grover/scaffold/
    balance/<endpoint>/{train,val,test}.csv`).
- `--metric NAME` — `mae` (regression default), `auc` (classification default),
  or any name `kermt.util.metrics.get_metric_func` accepts.
- `--epochs N` / `--batch-size N` / `--init-lr F` / `--max-lr F` /
  `--final-lr F` / `--warmup-epochs F` / `--weight-decay F` / `--dropout F` /
  `--bond-drop-rate F` / `--dist-coff F` / `--early-stop-epoch N` /
  `--seed N` — training-hyperparameter overrides. Anything not given is
  filled from `config/defaults_finetune.json`.
- `--ffn-hidden-size N` / `--ffn-num-layers N` — shared FFN trunk dims.
- `--ffn-num-task-specific-layers N` / `--ffn-task-specific-hidden-size H` —
  per-target FFN heads (default 0 = off; useful for heterogeneous multi-target
  finetunes). Both must be set together when N > 0.
- `--ensemble-size N` / `--num-folds N` — multi-model / k-fold CV. Default 1
  each.
- `--gpus 0` — single GPU id for single-process finetune (default 0). Ignored
  when `--num-gpus > 1`.
- `--num-gpus N` — number of GPUs for data-parallel DDP finetune. Default 1
  (single-process, unchanged). N>1 runs `main.py finetune` with `WORLD_SIZE=N`
  (one process per GPU); `--batch-size` is per-GPU.
- `--from-prepare <dir>` — skip the prepare step and reuse an existing
  `prepare_data.json` in `<dir>`. Useful when iterating on hyperparameters.

## Workflow

Let `$KERMT_REPO` be the path to your kermt repo checkout, and assume
`kermt-setup` has built `kermt:latest`. All paths below are on the host; the
helper bind-mounts them at known container paths.

1. **Pre-flight: ensure container + system probe.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" check_system | python -c "
   import json, sys; d = json.load(sys.stdin)
   if not d['ok']:
       print('System check failed:', d['gaps']); sys.exit(1)
   print(f'OK: {len(d[\"gpus\"])} GPU(s); CUDA via container toolkit')
   "
   ```
   Refuse to proceed if `ok: false`.

2. **Compute run directory.**
   ```
   RUN_DIR=$KERMT_REPO/runs/finetune_$(date -u +%Y-%m-%dT%H-%M-%SZ)
   ```

3. **Resolve & validate the checkpoint.**

   **Resolve — only if `--ckpt` was omitted.** Default to the released
   pretrained hybrid model **nvidia/NV-KERMT-70M-v2**:
   - **Consent gate.** Unless `--pretrained-release` was passed, ask the user:
     "No checkpoint given — download the released model nvidia/NV-KERMT-70M-v2
     (NVIDIA Open Model License, https://huggingface.co/nvidia/NV-KERMT-70M-v2)
     and finetune from it? [y/N]". **Never download without an explicit yes**
     (or `--pretrained-release`). If both `--ckpt` and `--pretrained-release`
     are given, abort — they conflict.
   - **Save location.** Default `$KERMT_REPO/models/NV-KERMT-70M-v2/`; honor
     `--model-dir <dir>` if given. An already-complete bundle is reused.
   - **Download** (foreground; ~282 MB on first fetch):
     ```
     "$SKILL_DIR/scripts/kermt_container.sh" run --model-dir <save-dir> -- \
         "python /skill/scripts/fetch_released_model.py --out /model"
     ```
     Parse the JSON; abort on `ok: false` (surface `errors`). On success set
     `<user-ckpt> = <save-dir>/kermt_contrastive_v2.0.pt`.

   **Validate** the resolved (or user-provided) ckpt:
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --ckpt <user-ckpt> -- \
       "python /skill/scripts/check_checkpoint.py --mode finetune_init --ckpt /ckpt"
   ```
   Parse the JSON. Abort on `ok: false`. The validator rejects already-
   finetuned ckpts (`has_task_ffn: true`) with a redirect to `kermt-infer`.

4. **Validate the data.**
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> -- \
       "python /skill/scripts/check_data.py --mode finetune --csv /data/<basename> [--targets COL1 COL2 ...]"
   ```
   If `--targets` was not given by the user, surface `auto_detected_targets`
   from the JSON and ask the user to confirm before continuing. Abort on
   `ok: false`.

5. **Prepare the data** (skip if `--from-prepare` given).

   **Pre-flight: check for sibling val.csv / test.csv.** Before invoking
   prepare_data, inspect the parent directory of `<user-csv>`. If a
   canonical-looking sibling `val.csv` (or `val_*.csv` — common variants
   include `val_T.csv`, `val_clean.csv`) AND a matching `test.csv` /
   `test_*.csv` exist next to the train CSV, the dataset ships its own
   pre-defined split. **In that case set `--split-type index_predetermined`
   AND pass `--val-csv` / `--test-csv`** — otherwise the configured
   `split_type` (default `scaffold_balanced`) will re-split the train CSV
   from scratch and silently discard the user's val/test files. When in
   doubt — or when the sibling files use non-canonical suffixes (`_T`,
   `_v2`, etc.) — surface the situation to the user and ask which they
   want.

   **Quoting target names.** If any of the `--targets` column names
   contain shell metacharacters (`>`, `&`, `|`, `(`, `)`, `$`, etc.),
   single-quote each one when passing on the CLI to keep the shell from
   eating part of the name. Example: `--targets 'Log_Caco2_Papp_A>B'
   'logD'`. The CSV header itself is read directly by the downstream
   trainer and is unaffected, but the prepare_data.json manifest's
   `targets[]` field captures whatever the shell delivers — unquoted
   metacharacters get truncated there.

   **Mount note:** `kermt_container.sh --data <host-csv>` mounts the
   parent directory of `<host-csv>` at `/data`. `--val-csv` and
   `--test-csv` must therefore reference files in that same parent
   directory. If val/test live in a separate directory (e.g. a sibling
   `splits/` folder), mount the parent of all three using `--data <dir>`
   on a directory rather than a file.

   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run --data <user-csv> --run-dir $RUN_DIR -- \
       "python /skill/scripts/prepare_data.py --mode finetune \\
            --csv /data/<basename> --out /runs/data \\
            --split-type <split_type> \\
            [--val-csv /data/<val-basename> --test-csv /data/<test-basename>] \\
            [--val-frac 0.1 --test-frac 0.1 --seed 0] \\
            --targets <COL1> [COL2 ...]"
   ```
   Outputs land at `$RUN_DIR/data/prepare_data.json`. For `scaffold_balanced`
   and `index_predetermined`, prep emits a single `clean_full_csv` + `.npz`;
   the runner passes them through to `main.py finetune` which calls
   `split_data` internally with the user-supplied seed.

6. **Estimate runtime + echo applied defaults.**
   - Finetune wall time is typically minutes-to-hours on 1 GPU.
   - Surface a summary of every flag that was filled from the defaults
     vs user-supplied, so the user knows what was assumed. The runner
     records this in `args_applied`.
   - Sample message:
     `"Filling from defaults_finetune.json: epochs=30, batch_size=32,
       split_type=scaffold_balanced. Override any of these with --<flag>."`

7. **Targets confirmation gate (hard requirement).** Before launching the
   runner, regardless of how the targets list was determined (CLI `--targets`,
   auto-detection in step 4, or a user natural-language request like
   "finetune on Caco2 and HLM"), echo the final targets list to the user with
   an explicit count:
   `"Will finetune on N target(s): COL1, COL2, ..."`. If the user's request
   specified a subset that doesn't match this list (e.g., they asked for 2
   tasks via natural language but the list still has 4), treat it as a
   discrepancy and re-prompt with the diff — never silently proceed on the
   wrong target set. Wait for explicit confirmation before launching unless
   `--yes` was given.

8. **Launch the runner detached.** (Consistent with the pretrain skills.)
   ```
   "$SKILL_DIR/scripts/kermt_container.sh" run_detached \\
       --name kermt-finetune-<ts> \\
       --ckpt <user-ckpt> --run-dir $RUN_DIR -- \\
       "python /skill/scripts/run_finetune_local.py \\
            --ckpt /ckpt \\
            --prepare-manifest /runs/data/prepare_data.json \\
            --dataset-type <type> \\
            --out /runs \\
            [--gpus 0] \\
            [--num-gpus N] \\
            [--epochs N --batch-size N --init-lr F ...] \\
            [--ffn-num-task-specific-layers N --ffn-task-specific-hidden-size H]"
   ```
   Returns the container name + id + log file path.

9. **Report to the user.** Output a short summary:
   - Container name + id
   - `$RUN_DIR/run.json` (manifest with cmd_replay + image digest)
   - Log file: `$RUN_DIR/logs/finetune.log`
   - TensorBoard: `$RUN_DIR/logs/tb` (open with `tensorboard --logdir
     $RUN_DIR/logs/tb`)
   - Final checkpoints land at `$RUN_DIR/ckpt/fold_0/model_0/model.pt`
     (best-val) and `last_checkpoint.pt` (sibling, auto-resume target).
     Held-out test predictions + metrics land at
     `$RUN_DIR/ckpt/fold_0/test_result.csv`. Paths vary with `--num-folds`
     / `--ensemble-size`.
   - To follow progress: `kermt-monitor <RUN_DIR>` (one-shot) or
     `docker logs -f <container-name>` (streaming).
   - To block until the run finishes (useful for short test runs):
     `docker wait <container-name>` — prints the exit code on completion.

## Hard rules

- **Never download the released model without consent.** When `--ckpt` is
  omitted, download `nvidia/NV-KERMT-70M-v2` only after an explicit user "yes"
  or an explicit `--pretrained-release` flag. `--ckpt` and
  `--pretrained-release` are mutually exclusive.
- **Never modify the user's input ckpt.** The runner passes its path via
  `--checkpoint_path`; `task/train.py` loads it read-only into the model and
  attaches a new FFN head. The source file stays untouched.
- **Arch comes from the ckpt, not from CLI/defaults.** The runner extracts
  `hidden_size`, `depth`, `num_attn_head`, `activation`, `embedding_output_type`,
  `self_attention` (+ `attn_hidden` / `attn_out` when applicable) from the
  ckpt's saved_args. There is no `--hidden-size` flag on this runner.
- **Never block on the long-running finetune.** The skill launches via
  `run_detached` and returns immediately after step 9. Use `kermt-monitor`.
- **Echo applied defaults back to the user.** The `args_applied` field of
  `run.json` records every flag's value + source (user / default-config).
  Surface a one-line summary of every filled-from-default flag so the user
  knows what was assumed.

## Common errors

- `finetune_init requires a pretrain ckpt (grover_base / cmim / hybrid)` →
  the ckpt you passed is already finetuned (has task FFN heads). Pick a
  pretrain ckpt instead, or use `kermt-infer` if you want to run
  predictions with the existing finetuned model. To resume a finetune on
  the SAME dataset, bypass the skill and call
  `python main.py finetune --checkpoint_path <ckpt> ...` directly — the
  agent skill doesn't support resume because saved-task identity
  can't be machine-verified against the new training data.
- `prepare_data manifest reports ok=False` → check `errors` for the failed
  step (typically clean_smiles or save_features). Fix and re-run.
- `ffn_num_task_specific_layers=N>0 but ffn_task_specific_hidden_size is unset`
  → MTL heads need an explicit hidden size. Pass `--ffn-task-specific-hidden-size H`.
- `finetune is single-GPU` (from `--gpus 0,1`) → `--gpus` selects one device
  for single-process finetune. For multi-GPU, use `--num-gpus N` (DDP) instead.

## Replayability

The `run.json` `cmd_replay` field is a single-line command that re-runs the
finetune with the same inputs, hyperparameters, and arch. To replay inside
the kermt container:

```bash
$(jq -r .cmd_replay $RUN_DIR/run.json)
```

If `ok_to_replay: false` in the manifest (because the kermt repo working
tree was dirty at launch time), the replay may not be bit-exact — pin the
exact commit via the `repo.commit` field and `git checkout` it
first.
