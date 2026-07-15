# DEFT AOI — Data Onboarding

Read this when a user wants to run the loop and either (a) has no workspace yet,
or (b) needs the exact data formats to assemble one. The main SKILL.md links
here from its data sections.

This loop trains on **your** AOI inspection data. There is **no public AOI
dataset to download** — the `NV_PCB_Siamese` paths throughout the skill are a
naming convention for the mount layout, not a fetchable dataset. If the user
arrives without a workspace, do not hard-stop silently: explain what they must
supply, point them at the spec template (`references/baseline_spec.yaml`), and
offer to scaffold the directory tree. The three categories below are the whole
story.

## What goes in a workspace

**You must provide (required — the loop cannot fabricate or download these):**

| Path | What it is |
|---|---|
| `specs/baseline_spec.yaml` | ChangeNet train/eval spec. Copy `references/baseline_spec.yaml` (bundled template) and adjust. Source of truth for architecture, lighting, image size. |
| `train/base/training_set.csv` | Seed training rows, 14-column siamese schema (below). ~200 rows is a normal first-run size. |
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

## ChangeNet CSV schema (VCN)

All three CSVs (`training_set.csv`, `validation_set.csv`, `testing_set.csv`)
share one 14-column schema, in order:

| # | Column | Required? | Meaning |
|---|---|---|---|
| 1 | `input_path` | **yes** | Directory (not a file) holding the component crop. TAO resolves the image as `{images_dir}/{input_path}/{object_name}_{light}{image_ext}`. |
| 2 | `golden_path` | **yes** | Directory of the golden/reference image for the same component. A row without it is unusable (this is a siamese change-detector). |
| 3 | `label` | **yes** | `PASS` (exact case — the dataloader's class-0 sentinel) or a defect string (`Missing`, `Shift`, …). |
| 4 | `object_name` | **yes** | Component id, e.g. `C1018@1`; combined with `{light}` to form the filename. |
| 5 | `project` | optional | Free-form project tag. |
| 6 | `boardname` | optional | Board id; used in the leakage-check key, so populate it if you can. |
| 7 | `comp_type_2` | optional | Component sub-type. |
| 8 | `mpass_mfail` | optional | Upstream machine verdict (`MPASS`/`MFAIL`). |
| 9 | `is_valid` | optional | 0/1 validity flag. |
| 10 | `comp_name` | optional | Component name. |
| 11 | `part_type` | optional | Part-type code. |
| 12 | `number_of_pins` | optional | Pin count. |
| 13 | `description` | optional | Human-readable part description. |
| 14 | `comp_type_1` | optional | Component primary type. |

Only columns 1–4 are load-bearing; the 10 optional columns carry production
metadata and are preserved/padded with empty strings when absent. `label` case
matters — keep `PASS` exactly, lowercase + strip everything else (see
`references/visual-changenet.md`). `{light}` comes from
`dataset.classify.input_map` in the spec (e.g. `SolderLight`); `{image_ext}`
from `dataset.classify.image_ext`.

Example row:
```
input_path,golden_path,label,object_name,project,boardname,comp_type_2,mpass_mfail,is_valid,comp_name,part_type,number_of_pins,description,comp_type_1
690-5G190-0510-001P1/AOI_B/FXLH_..._AOI_B_20230317130332/PerComponent,golden/images/690-5G190-0510-001P1BOT/,PASS,C1018@1,690-5G190-0510-001P1,30332,C,MPASS,0,C1018,6,4,CAP X6S 0402 1uF 6.3V 10%,3
```
