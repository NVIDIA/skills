# Data Contract and Output Layout

## Bringing Your Own Data

This loop trains on **your** AOI inspection data. There is **no public AOI
dataset to download** — the `NV_PCB_Siamese` paths throughout the skill are a
naming convention for the mount layout, not a fetchable dataset. If the user
arrives without a workspace, do not hard-stop silently: explain what they must
supply (the three categories below are the whole story), point them at the spec
template (`references/baseline_spec.yaml`), and offer to scaffold the tree.

**You must provide (required — the loop cannot fabricate or download these):**

| Path | What it is |
|---|---|
| `specs/baseline_spec.yaml` | ChangeNet train/eval spec. Copy `references/baseline_spec.yaml` (bundled template) and adjust. Source of truth for architecture, lighting, image size. |
| `train/base/training_set.csv` | Seed training rows using the four mandatory ChangeNet columns below. ~200 rows is a normal first-run size. |
| `train/base/validation_set.csv` | Held-out rows, same schema. Must not overlap training (the loop hard-stops on leakage). |
| `kpi/testing_set.csv` | KPI test rows, same schema. This is what FAR / recall is measured on. |
| `kpi/images/` | The actual image files referenced by every CSV above (real inspection captures + their golden references). |
| `.env` | `NGC_KEY` + `HF_TOKEN`. Copy `.env.example`. |

**Auto-fetched on first use (do not pre-stage unless air-gapped):** the
ChangeNet backbone (`nvidia/C-RADIOv2-B`), the Cosmos/AnomalyGen base
checkpoints, and the AnomalyGen PCB reference dataset
(`nvidia/Cosmos-AnomalyGen-PCB-Dataset`) — all gated by `HF_TOKEN`, cached under
`augmentation/anomalygen/base_checkpoints/`. **Note:** the AnomalyGen PCB
reference dataset is a *generator* fine-tuning set (clean image + mask + defect
spec) — it is **not** your AOI training data and cannot substitute for it.

**Created by the loop (never hand-author):** everything under
`results/run_<TS>/`, the per-iter `synthetic_iter*` staged images, and the
combined training CSVs.

The `augmentation/mining_pool/` real-image pool is **optional** — provide it if
you have a production-line image stream to mine from; the loop runs without it
(synthetic-only augmentation).

## Data Contract

The required inputs above, laid out as a tree — plus the optional
`augmentation/` AnomalyGen override slots and the `results/` root the loop
creates (paths under `<workspace>` unless absolute):

```text
<workspace>/
├── .env                                     # NGC_KEY (nvcr.io/* pulls for all pinned images), HF_TOKEN (HuggingFace pre-flight pulls)
├── specs/baseline_spec.yaml                 # ChangeNet train/eval spec
├── train/base/
│   ├── training_set.csv                     # seed training rows; four mandatory ChangeNet columns
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

**ChangeNet CSV schema (VCN).** All three CSVs (`training_set.csv`,
`validation_set.csv`, `testing_set.csv`) share the same schema with four mandatory columns:

| # | Column | Required? | Meaning |
|---|---|---|---|
| 1 | `input_path` | **yes** | Directory (not a file) holding the component crop, relative to `images_dir`. |
| 2 | `golden_path` | **yes** | Directory of the golden/reference image for the same component. A row without it is unusable (this is a siamese change-detector). |
| 3 | `label` | **yes** | `PASS` (exact case — the dataloader's class-0 sentinel) or a defect string (`Missing`, `Shift`, …). |
| 4 | `object_name` | **yes** | Component id / filename stem, e.g. `C1018@1`; do not include the lighting suffix or file extension. |

Additional production metadata columns are optional and may be preserved when
present; they are not part of the required ChangeNet CSV contract. `label` case
matters — keep `PASS` exactly, lowercase + strip everything else (see
`references/visual-changenet.md`). TAO constructs each image path as
`{images_dir}/{input_path}/{object_name}_{light}{image_ext}`. The
`_{light}{image_ext}` filename suffix is defined in the run's `spec.yaml`, not
in the CSV: `{light}` is a key in `dataset.classify.input_map` (for example,
`SolderLight`), and `{image_ext}` is `dataset.classify.image_ext` (for example,
`.jpg`).

Example row:
```
input_path,golden_path,label,object_name
690-5G190-0510-001P1/AOI_B/FXLH_..._AOI_B_20230317130332/PerComponent,golden/images/690-5G190-0510-001P1BOT/,PASS,C1018@1
```

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
