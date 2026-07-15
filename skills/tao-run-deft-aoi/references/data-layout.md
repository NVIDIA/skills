# Data Contract and Output Layout

## Bringing Your Own Data

This loop trains on **your** AOI inspection data — there is **no public AOI
dataset to download**. If the user arrives without a workspace, do not hard-stop
silently: read `references/data-onboarding.md` and walk them through what to
supply. That doc is the source of truth for the user-provided / auto-fetched /
loop-created split, the full 14-column CSV schema, and the `baseline_spec.yaml`
template pointer. The `## Data Contract` below summarizes the resulting layout.

## Data Contract

Inputs (all paths under `<workspace>` unless absolute):

```text
<workspace>/
├── .env                                     # NGC_KEY (nvcr.io/* image pulls — both nvstaging/tao and nv-metropolis-dev), HF_TOKEN (HuggingFace pre-flight pulls)
├── specs/baseline_spec.yaml                 # ChangeNet train/eval spec
├── train/base/
│   ├── training_set.csv                     # seed training rows; ChangeNet 14-column siamese schema
│   └── validation_set.csv                   # held-out rows; checked for leakage against every train CSV
├── kpi/
│   ├── images/                              # KPI test images (real data only — no generated images here)
│   └── testing_set.csv                      # labels live in the CSV
├── augmentation/
│   ├── mining_pool/
│   │   ├── mining_pool.csv                  # append-only production-line samples; paths relative to this dir
│   │   └── images/                          # source images referenced by mining_pool.csv (e.g. *_SolderLight.jpg)
│   └── anomalygen/                          # [Optional] User override slots for AnomalyGen assets.
│       │                                    # If pre-staged, the loop uses these host paths verbatim.
│       │                                    # If absent, the paidf-anomalygen skill handles asset acquisition
│       │                                    # internally — exact storage location is its concern, not the loop's.
│       │                                    # `<project>` is the project label (e.g. UC1).
│       │                                    # See references/paidf-anomalygen.md for details.
│       ├── checkpoints/<project>/           # Fine-tuned PCB AnomalyGen model override (ag_config.yaml + checkpoints/{latest_checkpoint.txt, model/iter_<step>.pt}).
│       ├── base_checkpoints/                # Cosmos base models cache override (~22 GB for 2B-only, ~140 GB with 14B + T5-11b).
│       └── datasets/<project>/              # PCB reference data override — defect_spec.jsonl + per-texture image/mask subdirs.
└── results/run_<YYYYMMDD_HHMMSS>/           # created/resumed by this workflow (= ${RESULTS_DIR})
```

**ChangeNet CSV schema (VCN).** Mandatory columns: `input_path`, `golden_path`, `label`, `object_name` (siamese change-detector — a row without `golden_path` is unusable). Preserve `boardname`, scores, and provenance fields when present. TAO builds the full image path as `{images_dir}/{input_path}/{object_name}_{light}{image_ext}` — `input_path` is a directory, not a file. Full 14-column enumeration + example row: `references/data-onboarding.md`.

## Output Layout

Relative to `<workspace>`:

```text
results/run_<YYYYMMDD_HHMMSS>/               # = ${RESULTS_DIR}
├── deft_state.json                          # current resume snapshot (schema: references/deft_state.json)
├── loop_log.jsonl                           # append-only stage log; single source of truth
├── DEFT_Loop_Report.html                    # re-rendered after every stage by agents/reporter.md
├── best_model.json                          # inference handoff metadata (see references/prepare-for-inference.md)
├── best_model_inference_spec.yaml           # ready-to-run TAO inference spec built from training config
├── iter${ITER}_summary.md                   # ≤300-word per-iteration summary
├── baseline/
│   ├── train/                               # TAO train output: model_epoch_<EEE>_step_<SSS>.pth × N, status.json, experiment.yaml, train.log
│   ├── inference/{best_val,latest}/         # per-checkpoint inference.csv + KPI plots from scripts/analyze_kpi.py
│   └── rca_results/<TS>/                    # kpi_gaps.parquet, threshold.txt, weak_samples_breakdown.txt
└── iter${ITER}/
    ├── routing_results/<TS>/                # mining_gaps.parquet, anomalygen_gaps.parquet, routing_summary.txt
    ├── anomalygen/
    │   ├── amp/                             # AMP testcase intermediates (one subdir per sample row in testcase.jsonl)
    │   ├── testcase.jsonl                   # built by prep_testcase.sh; consumed by run_sdg.sh
    │   └── sdg/                             # `synthetic_dataset_generation.py` output (= paidf-anomalygen `output_dir`)
    │       ├── SDG_result.csv               # one row per generated sample with params + PSNR
    │       ├── reconstructed_image/         # NG outputs (used as ChangeNet input_path)
    │       ├── original_image/              # OK inputs paired 1-to-1 (used as ChangeNet golden_path)
    │       ├── original_mask/
    │       ├── cropped_image/
    │       ├── cropped_mask/
    │       └── annotated_image/
    ├── ag_config_sdg.yaml                   # sanitized config (job + model only); bind-mounted at SDG launch onto the real checkpoint's ag_config.yaml
    ├── mining_filter/
    │   ├── mining_pool.csv                  # combined SDG rows + real mined rows (similarity ≥ 0.9); used for training
    │   ├── sdg_rows.csv                     # raw output of scripts/changenet_data_pair_prepare.py before path rewriting
    │   ├── knn_summary.csv                  # candidate_count, kept_count, rejected_count, similarity_threshold=0.9
    │   ├── source_embeddings.parquet        # embeddings of mining_pool candidates
    │   ├── target_embeddings.parquet        # embeddings of weak-target images
    │   └── mining_summary.txt               # per-label breakdown emitted by mining container
    ├── dataset/
    │   ├── train_combined_iter${ITER}.csv
    │   ├── train_combined_iter${ITER}_provenance.csv  # source ∈ {base_train, previous_iter_train, mining_pool}
    │   └── images/synthetic_iter${ITER}_{ng,ok}/      # ChangeNet-ready synthetic image staging
    ├── train/                               # TAO train output for iter${ITER}
    ├── inference/{best_val,latest}/
    └── rca_results/<TS>/                    # next iteration's RCA reads inference/{best_val|latest}/inference.csv
```

A previous combined CSV's rows already include every prior contribution — assemble iter N+1 from `train_combined_iter${N}.csv` plus the new `mining_filter/mining_pool.csv`, not from `train/base/training_set.csv` again.

## Augmentation Pool

Each iteration builds one **mining pool** from two complementary sources:

| Source | Selection | Contribution |
|---|---|---|
| AnomalyGen synthetic generation (Pipeline step 3) | All generated images — no filtering | Defect-type diversity |
| Real images from `augmentation/mining_pool/` (Pipeline step 4) | k-NN cosine similarity ≥ 0.9 to weak-target embeddings | Real-distribution anchor |

Both sources are appended into a single `mining_filter/mining_pool.csv` before fine-tuning. `train_combined_iter${N}.csv` = base training rows + mining pool rows.

**Source pool growth.** `augmentation/mining_pool/mining_pool.csv` is append-only — the production line contributes new real-image samples daily (Day 1 → Day N). Each iteration mines against the current accumulated state of the pool; later iterations naturally benefit from a richer pool. Before running the mining step, verify the file exists and is non-empty; a missing or zero-row pool is a hard stop (no real-image contribution to the mining pool for this iteration).
