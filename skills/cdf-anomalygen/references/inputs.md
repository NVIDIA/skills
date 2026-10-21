# Inputs reference

Progressive-disclosure detail for the **Inputs** section in the parent `SKILL.md` — the mode table, the
parameter reference (required + optional), and the step map. If this file conflicts with `SKILL.md`,
`SKILL.md` wins.

## Modes

Which steps run depends on the mode:

| User intent                | `mode`            | Must collect up front                            | Steps    |
| -------------------------- | ----------------- | ------------------------------------------------ | -------- |
| Train + generate           | `full`            | `dataset_dir`, `num_sdg`                         | 1 → 7    |
| Train only                 | `finetune_only`   | `dataset_dir`, `num_sdg`                         | 1 → 3    |
| Generate from a checkpoint | `generation_only` | `checkpoint`, `recipe`, `dataset_dir`, `num_sdg` | 2, 4 → 7 |

`generation_only` (a.k.a. `inference_only`) still needs `dataset_dir` — Step 2 reads its clean
images (and, under `amg`, its `anomaly_image/`+`mask/` pairs as exemplars) to build the generation
batch, and Step 5 reads its `anomaly_image/`+`mask/` pairs as the evaluation real-reference root.
`defect_spec` is needed **only when `mask_backend=amp`** (allocate → pair → place routes on it);
the default `amg` never reads it, and Steps 3–7 don't either — they consume `testcase.jsonl` and
the image/mask trees. In `full`, `checkpoint` is derived after Step 3 — don't pass it. Collect
params up front; run straight through.

Each step's section header lists the modes it runs in (e.g. `### Step 6 — … (skip when finetune_only)`).

## Parameters

### Required

| Param         | Maps to                                                                                       | Notes                                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`        | —                                                                                             | `full` \| `finetune_only` \| `generation_only`.                                                                                       |
| `name`        | `set_pipeline_vars.sh --name`                                                                 | Experiment label **only**; names the recipe, run dir, logs and mask-set dirs. Never a dataset location.                               |
| `dataset_dir` | `set_pipeline_vars.sh --dataset_dir`, then `--dataset-path` / `--dataset_dir` / `--real_root` | Step 1 contract layout; also the eval real-reference. Required in every mode and never derived from `name` — the two normally differ. |
| `defect_spec` | `--defect_desc`                                                                               | `defect_spec.jsonl`, one row per `{texture}+{defect}` — **`mask_backend=amp` only**; `amg` never reads it.                            |
| `recipe`      | `--recipe`                                                                                    | `full`/`finetune_only`: derived from `name`; optional, used as the template if passed. **generation_only**: the run-dir copy.         |
| `num_sdg`     | `--num_sdg`                                                                                   | Samples across defect types; final `searched/` count.                                                                                 |
| `checkpoint`  | `--checkpoint`                                                                                | **generation_only** — `iter_<best>.pt`; `full` derives it.                                                                            |

### Optional (defaults shown)

| Param                           | Maps to               | Default               | Notes                                                                                       |
| ------------------------------- | --------------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| `num_search_run`                | — (Step 6 rounds)     | `3`                   | Search rounds; `0` = clone only.                                                            |
| `mask_backend`                  | — (Step 2 backend)    | `amg`                 | `amg` (agentic) or `amp` (AMP).                                                             |
| `legacy_fixed_size`             | `--job_name` + recipe | off                   | Legacy fixed-input-size mechanism; see SKILL.md Step 3. Only when the user explicitly asks. |
| `num_gpus`                      | `--nproc_per_node`    | `1`                   | GPUs (fine-tune + generation).                                                              |
| `per_defect_counts`             | `roi_allocate`        | uniform               | Exact per-defect gen counts.                                                                |
| `per_clean_pairing`             | `roi_allocate`        | `off`                 | `on` = one record per clean image; needs exactly one non-zero defect type. **`amp` only**.  |
| `no_caption`                    | `pseudo_label`        | off                   | Skip Step 7 captioning.                                                                     |
| `max_iter`                      | `recipe`              | `15000`               | Total fine-tune iterations; set `cycle_lengths` to the same value.                          |
| `save_iter`                     | `recipe`              | `1000`                | Checkpoint interval.                                                                        |
| `validation_iter`               | `recipe`              | `1000`                | Validation interval.                                                                        |
| `lr`                            | `recipe`              | `0.001`               | AdamW learning rate.                                                                        |
| `max_sequence_length`           | `recipe`              | `5120`                | Token budget per step.                                                                      |
| `activation_checkpointing_mode` | `recipe`              | `none`                | `none` / `selective` / `full`; memory-only OOM remedy.                                      |
| `input_size_ladder`             | `recipe`              | `[256,512,1024,1536]` | Allowed input sizes (one entry = fixed size).                                               |
| `crop_mode`                     | `recipe`              | `scale`               | Framing source: `scale` or `ratio`.                                                         |
| `crop_ratio_range`              | `recipe`              | `[1.5,2.5]`           | Context ratio: the window is `bbox × ratio` (widen under `ratio`, e.g. `[1.5,8.0]`).        |
| `scale_range`                   | `recipe`              | `[0.333,3.0]`         | Resample-scale draw range (`scale` mode only).                                              |
| `per_class_lora_rank/alpha`     | `recipe`              | `8` / `8`             | Per-defect LoRA capacity.                                                                   |
| `early_stop_enabled`            | `recipe`              | `false`               | Stop when `nn` plateaus.                                                                    |
| `score`                         | `quality_refine run`  | `nn`                  | Step 6 objective; `aq_rank` is optional.                                                    |
| `guidance_range`                | `quality_refine draw` | `1.0 10.0`            | Step 6 guidance draw range.                                                                 |
| `crop_ratio_range`              | `quality_refine draw` | recipe band           | Step 6 crop-ratio draw range; narrowed to the recipe's trained band.                        |
| `bo_min_obs`                    | `quality_refine draw` | `2`                   | Min obs before per-sample BO.                                                               |

`recipe` knobs live in `ag_config/exp_${TASK}_<name>.yaml` (full key list + defaults in
`fine-tuning.md`); `quality_refine` score/search flags in `quality-refinement.md`;
`--per_defect_counts` in `mask-placement.md`. `SKILL.md` links all three.

## Step map

| Step | What runs                                      | Entry point                                                               |
| ---- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| 1    | Dataset in contract layout                     | `prepare_*` or custom                                                     |
| 2    | Mask placement → `testcase.jsonl`              | `agentic_mask_generation.generate` (default) / `roi_place_pipeline` (amp) |
| 3    | Fine-tuning (LoRA adapters)                    | `scripts/skill_utility/train_control.sh`                                  |
| 4    | Generation (SDG)                               | `${SCRIPTS}/generate.py`                                                  |
| 5    | Evaluation (quality KPI)                       | `${SCRIPTS}/evaluate.py`                                                  |
| 6    | Quality refinement → `searched/`               | `${SCRIPTS}/quality_refine.py run`                                        |
| 7    | Pseudo-labeling → COCO/classification/captions | `${SCRIPTS}/pseudo_label.py`                                              |
