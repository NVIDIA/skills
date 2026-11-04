# Finetune Reference — Phase 0 and Phase 1

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- Phase 0: Environment and Checkpoint Pre-flight
  - Environment check
- Phase 1: Fine-Tuning
  - Expected dataset structure
  - Step 1: Dataset validation output
  - Step 2: Validation JSONL sizing
  - Step 3: Config template and critical flags
  - Step 4: Training launch details
  - Best checkpoint selection
  - Phase 1 error handling


Detail for `anomalygen` Phases 0 and 1. Read before executing either phase.

---

## Phase 0: Environment and Checkpoint Pre-flight

Read `references/setup.md` for the full checkpoint table, script behavior
detail (idempotent skip logic, HF_TOKEN handling, SAM2/Qwen3-VL skip), and
error handling. Summary:

- Defaults fetch the 2B base + T5-large + guardrail/encoders (~40 GB). The 14B
  base (`--model-sizes 14B`) and T5-11B (`--with-t5-11b`) are opt-in; the full
  set is ~150 GB. Artifacts: 2B/14B model, T5-large/T5-11B, Cosmos-Guardrail1,
  RADIO, NVDINOV2, DINOv2, SAM2, Qwen3-VL-4B.
- Pass `--model-sizes <2B|14B>` to `check.sh`/`download_checkpoints.sh` matching
  this run's `model_size` — the 2B-only default otherwise skips a 14B download
- `HF_TOKEN` must be present in the environment — exported in your shell or
  sourced from a user-approved env file (`set -a; source /path/to/.env; set +a`);
  `download_checkpoints.sh` refuses to start if unset
- `check.sh` exits 0 if all present, 1 with a remediation list if not — run it first

### Environment check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')"
```

- If `num_gpus` > detected GPU count → stop and inform the user.
- If CUDA unavailable → stop: `conda activate cosmos-predict2`.
- If `num_gpus > 1` → remind user to verify `context_parallel_size: 1` in the
  generated config (set by default in Step 3).

---

## Phase 1: Fine-Tuning

### Expected dataset structure

```
<dataset_dir>/
├── <TEXTURE>/
│   ├── anomaly_image/
│   │   ├── <ANOMALY_TYPE_A>/   # image_001.png, image_002.png, ...
│   │   └── <ANOMALY_TYPE_B>/
│   ├── mask/
│   │   ├── <ANOMALY_TYPE_A>/   # image_001_mask.png, ...
│   │   └── <ANOMALY_TYPE_B>/
│   └── clean_image/            # optional — omit clean_dir when present here
└── <TEXTURE_2>/
    └── ...
```

Each anomaly image must have a corresponding mask with the `_mask` suffix in
the filename stem. `validate_dataset.py` checks this and exits non-zero on
any mismatch (training would crash mid-iteration on it).

### Step 1: Dataset validation output

`validate_dataset.py` prints a summary:
```
=== Dataset Validation Summary ===
  [TEXTURE_1, TYPE_A]: 50 images, 50 masks
  [TEXTURE_1, TYPE_B]: 30 images, 30 masks
Issues: 0
```

- If no anomaly types detected → exit 1; the dataset structure is wrong.
- Image/mask mismatch or missing-directory issues → listed as WARNINGs, then
  exit 1. Fix the pairs (or remove the offending files) and re-run — the
  training dataloader asserts on exactly these and would crash mid-iteration.
- The anomaly types list from this output is used in `generate_config.py`
  (`--defect-spec` drives it, but validate first to catch structural issues).

### Step 2: Validation JSONL sizing

num_SDG for the validation JSONL = **total training mask count** (sum of all
mask counts printed by `validate_dataset.py`). Combined with `--mode validation`
(required — `prep_testcase.sh` defaults to `inference`), the proportional
allocation gives n_seeds=1 with every training submask appearing exactly once.

The script emits every defect type from `defect_spec` — any type with zero
allocation under `--mode validation` causes `allocate_samples.py` to raise
(the per-defect KPI floor described below). Fix the offending defect spec or
adjust num_SDG before continuing.

**KPI floor (validation only):** every defect type must receive ≥ 1 JSONL
entry, and ≥ 3 is needed for `nn_score` to be statistically meaningful. With
proportional allocation, ≥ 1 is satisfied automatically when num_SDG ≥
`ceil(3 × N_total / N_min)`, where `N_total` is total mask count and `N_min`
is the smallest per-type mask count. If any type has very few masks (e.g. 1–2),
raise num_SDG accordingly.

**Inference mode skips the floor.** `prep_testcase.sh --mode inference`
(default) does NOT enforce the per-defect floor — defects can have 0
allocation when `--per-defect-counts` excludes them. This is intentional for
SDG inference where the user may want only a subset of defect types.

Skip this step if the user supplies `VALIDATION_JSONL` directly. In that case,
verify every row's `anomaly_type` exists in `defect_spec` — a mismatch causes
training validation to drift from the trained defect set silently.

### Step 3: Config template and critical flags

`generate_config.py` uses the FSDP template from
`.agents/skills/anomalygen/assets/ag_config.yaml`. Key flags:

| Flag | Default | Note |
|---|---|---|
| `--model-size` | `2b` | `2b` or `14b`; sets the DiT backbone size |
| `--aug-type` | `random_ratio_crop` | Pass `--aug-type null` to disable crop augmentation |
| `--ratio-range` | `1.5 8.0` | MIN MAX floats; only used when aug_type is set |
| `--early-stop` | off | Enables early stopping on validation `nn_score` |
| `--es-patience` | `5` | Validations without improvement before stopping |

**Critical:** `ad_precision: float32` is injected by the script and must not
be removed — it overrides bfloat16 for the mask encoder, adapter, and anomaly
embedding, producing better quality.

**Config template notes** (`assets/ag_config.yaml`):
- `t5_model_name: checkpoints/google-t5/t5-large` — saves ~20 GB VRAM vs the
  T5-11B default; `d_model=1024` stays compatible with the DiT backbone.
- `anomaly_types` must match between `dataloader_train.dataset.anomaly_types`
  and `model.config.ag_config.anomaly_embedding.anomaly_types` —
  `generate_config.py` populates both from `--defect-spec` automatically.

Show the generated config to the user and ask for confirmation before writing
to `ag_configs/<name>.yaml`.

### Step 4: Training launch details

`launch_training.sh` wraps:
```bash
export IMAGINAIRE_OUTPUT_ROOT=./results
EXP=predict2_anomaly_gen_ddp_<model_size>  # e.g. predict2_anomaly_gen_ddp_2b
torchrun --nproc_per_node=<num_gpus> --master_port=12341 \
    -m scripts.anomaly_gen.ag_train \
    --config=cosmos_predict2/configs/base/ag_config.py \
    --ag_config=ag_configs/<name>.yaml \
    -- experiment=${EXP}
```

`IMAGINAIRE_OUTPUT_ROOT` must be set (defaults to `checkpoints/` otherwise).
The bare `--` before `experiment=` is required (separates argparse from Hydra).

Checkpoints are saved to:
```
results/anomaly_gen/<name>/<name>_training_FP32_lr<lr>_bs=<bs>_<MODEL_SIZE_UPPER>_<H>x<W>/
```

### Best checkpoint selection

After training, review the validation `nn_score` log (emitted every
`validation_iter` steps). Select the step with the **highest average nn_score**
across all defect types — not the final checkpoint. Small datasets overfit early.

If the user enabled `--early-stop`, the training script selects the best step
automatically. Otherwise, scan `results/anomaly_gen/<name>/*/validation_*.log`
or Tensorboard for the step with peak average `nn_score`.

Report the best step and its `nn_score` to the user before proceeding to
inference phases.

### Phase 1 error handling

| Symptom | Action |
|---|---|
| No anomaly types detected | Dataset structure wrong — check `anomaly_image/` and `mask/` subdirs exist |
| Missing `clean_image_path` and auto-discovery fails | Stop; ask the user for the clean images path |
| Missing ROI/submask for PD defects in validation JSONL | Warn and treat as free, or ask for the missing inputs |
| Multi-GPU port conflict | Increment `master_port` in launch_training.sh (default 12341) |
| OOM during training | Reduce `batch_size` or switch to smaller model |
| Training loss diverges | Reduce `lr` (try `0.01` or `0.005`) |
| NCCL errors | Check `CUDA_VISIBLE_DEVICES` and driver compatibility |
