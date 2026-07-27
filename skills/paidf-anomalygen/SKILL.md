---
name: paidf-anomalygen
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and a CUDA GPU. Pulls the `metropolis_sdg.paidf_anomalygen` image declared in `versions.yaml` at the skill bank root.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
description: >-
  Full PAIDF AnomalyGen pipeline — fine-tune on a new anomaly dataset, generate
  synthetic anomaly images (SDG), evaluate quality (nn_score), and search per-sample
  (guidance, crop_ratio) parameters. Three modes: full (Phase 0→7: finetune then
  generate), finetune_only (Phase 0→1: train only), inference_only (Phase 0, 2→7:
  generate from an existing checkpoint). Use when the user asks to "fine-tune
  AnomalyGen", "generate anomaly images", "run PAIDF SDG", "evaluate SDG output
  quality", "run per-sample search", or run any part of the AnomalyGen pipeline,
  even if they only mention one phase.
tags:
- tao
- data
---

# PAIDF AnomalyGen

Multi-phase pipeline (0–7); the `mode` flag selects which phases run.

| Phase | What runs | Mode(s) |
|---|---|---|
| 0 | Verify / download pretrained checkpoints | all |
| 1 | Fine-tune on `dataset_dir` | `full`, `finetune_only` |
| 2 | Prepare inference JSONL (AMP routing) | `full`, `inference_only` |
| 3 | SDG — generate synthetic anomaly images → `original/` | `full`, `inference_only` |
| 4 | Eval `original/` — emit `per_sample.csv` + `eval.log`, merge `nn_score` into `SDG_result.csv` | `full`, `inference_only` |
| 5 | Per-sample `(guidance, crop_ratio)` search rounds → `rounds/round_NN/` (each round runs SDG + eval) | `full`, `inference_only` |
| 6 | Assemble best-of-rounds into `searched/` (stitch only), plus `rounds/search_summary.csv` | `full`, `inference_only` |
| 7 | Filter `searched/` by `nn_threshold` (default `0.4`), regen dropped samples, then canonical bucket eval → `searched/{per_sample.csv, eval.log}` | `full`, `inference_only` |

Run every phase through to completion without mid-run pauses. Collect all
required parameters up front, and run every command from the repo root.

**Shell setup.** All `${ANOMALYGEN_SCRIPTS}` references resolve to the packaged
helper-script directory. Inside the container this is preset (`ENV
ANOMALYGEN_SCRIPTS=<dir>/scripts/utilities`); on the host, export it once per
shell:

```bash
export ANOMALYGEN_SCRIPTS="$(git rev-parse --show-toplevel)/scripts/utilities"
```

`python3 -m scripts.utilities.<name>` invocations work from any CWD inside the
container (PYTHONPATH is preset) and from the repo root on the host. When inside
a product container (`ANOMALYGEN_PRODUCT_MODE=1`), invoke `anomalygen-guard`
before any GPU work; if it reports `BLOCKED`, fix the listed issues before
continuing.

## Quick Start

The pipeline runs inside the `metropolis_sdg.paidf_anomalygen` container
(declared in `versions.yaml`) or any host with the `cosmos-predict2` conda env
active. All phase commands assume that environment, at the repo root, with
`ANOMALYGEN_SCRIPTS` exported.

Minimal end-to-end run (`mode=full`):

```bash
# 1. Set the shared variables (see "Shared variables" for the full set).
export ANOMALYGEN_SCRIPTS="$(git rev-parse --show-toplevel)/scripts/utilities"
MODE=full
NAME=my_exp
DATASET_DIR=/data/uc1
DEFECT_DESC=assets/defect_spec_template.jsonl
NUM_SDG=20
MODEL_SIZE=2b

# 2. Phase 0 — verify / download checkpoints (~140 GB; needs HF_TOKEN).
${ANOMALYGEN_SCRIPTS}/check.sh || ${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh

# 3. Walk Phases 1→7 in order (see each Phase section).
```

For `mode=inference_only` (reuse a checkpoint) also set `CKPT`/`STEP` and skip
Phase 1. For `mode=finetune_only` run only Phases 0–1.

## Running in Docker — container launch, mounts & permissions

The `paidf-anomalygen` image runs as a non-root baked-in user (`USER
anomalygen`, `uid=10000`), independent of your host uid. Docker does not remap
uids on bind mounts, so a host directory owned by your uid is not writable by
uid 10000 and the container fails the instant it tries to create a file there.
Run as your host uid with `--user "$(id -u):$(id -g)"` plus the mandatory
`/etc/passwd`+`/etc/group` and `HOME`/cache-redirect companions, and run the
fail-fast write preflight before Phase 0. See `references/docker.md` for the
full `docker run` command, the load-bearing-flag table, the preflight snippet,
and the uid-10000 `chown`/`chmod` fallback.

## Reference files — read before executing phases

Read **`references/finetune.md`** before Phase 0/1 and **`references/inference.md`**
before any of Phases 2–7; for `mode=full` read both before starting. The
remaining references below are on-demand — read when troubleshooting or needing
full detail for a specific phase.

| File | Read when |
|---|---|
| `references/finetune.md` | Before Phase 0/1: env check, checkpoint download, dataset validation, config generation, training commands, best-checkpoint selection |
| `references/finetune-commands.md` | Exact Phase 1 Step 1–4 commands and `CKPT`/`STEP` derivation |
| `references/inference-commands.md` | Exact Phase 5 `run_round.sh` and Phase 7 `filter_with_regen` commands |
| `references/inference.md` | Before Phases 2–7: AMP routing, JSONL validation, SDG flags, eval interpretation, search loop, filtering |
| `references/setup.md` | Checkpoint download fails; first-time setup; HF_TOKEN / disk issues |
| `references/datasets.md` | User needs to prepare or obtain a UC1 / UC2 / UC3 dataset; `dataset_dir` doesn't exist yet |
| `references/prep-testcase.md` | AMP fails; need full param table, helper script descriptions, allocation invariant |
| `references/sdg-inference.md` | NCCL hang; checkpoint validation error; multi-GPU VRAM question; full step list |
| `references/eval.md` | Unexpected scores; FID column order confusion; eval output format reference |
| `references/sdg-refine.md` | draws.json alignment; re-AMP heuristics; search output layout |
| `references/guard-and-custom-counts.md` | Full guard preflight command; `--per-defect-counts` example |
| `references/docker.md` | Container launch command, mount-permission flags, write preflight, uid-10000 fallback |
| `references/output-layout.md` | Full `results/<name>/` directory tree with per-file annotations; post-run Verification checklist |
| `references/error-handling.md` | Pipeline-level failure modes: missing mask dirs, short/empty AMP, mid-round resume, off-boundary `step` |

---

## Required parameters

`num_SDG` allocation depends on `prep_testcase.sh --mode`: `inference` (default,
Phase 2) is uniform across defect types, override per-defect via
`--per-defect-counts`; `validation` (Phase 1's validation JSONL) is proportional
to training mask counts (largest-remainder rounding) and enforces ≥1 per defect.
See `references/prep-testcase.md` for the full mode table.

| Parameter | Description |
|---|---|
| `mode` | `full` (Phase 0→7), `inference_only` (skip Phase 1), or `finetune_only` (Phase 0→1 only). |
| `name` | Experiment label. |
| `dataset_dir` | Training/reference dataset root. Drives mask-count allocation, AMP submask templates, and holds `semantic_segmentation_labels.json` for `cad` defects. |
| `defect_spec` | JSONL tagging each defect `spatial_dependency` as `free`/`text`/`cad`. `text` entries need `roi_prompt_defect_location`. Template: `assets/defect_spec_template.jsonl`. |
| `num_SDG` | Total output samples per bucket. *(Ignored when `mode=finetune_only`.)* |

## Conditionally required

| Parameter | Required when | Description |
|---|---|---|
| `checkpoint_dir` / `step` | `mode=inference_only` | Pre-existing fine-tuned model. In `mode=full` these are auto-derived after Phase 1; passing them is an error. In `mode=finetune_only` silently ignored — Phase 1 always trains from scratch (no resume-from-checkpoint support). Both must be present together — supplying only one is an error. |

## Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `clean_dir` | `dataset_dir` | Clean images. Set only when they live outside the training dataset. Forwarded as `--clean-dir` to prep-testcase and `--clean-image-path` to finetune. |
| `validation_jsonl` | auto-generated | Pre-built validation JSONL for Phase 1. When supplied, preflight verifies every `defect_spec` type appears and paths exist. |
| `num_search_run` | `3` | Per-sample search budget for Phase 5. `0` skips search (only `original/`). *(Ignored when `mode=finetune_only`.)* |
| `nn_threshold` | `0.4` | `nn_score` cutoff for Phase 7 (DINOv2 correspondence to real defects — key KPI). Samples below are regenerated; final `searched/` always has `num_SDG`. `0` disables filtering. |
| `max_iter` | `75000` | Phase 1 only. Total fine-tune iterations. |
| `save_iter` | `5000` | Phase 1 only. Checkpoint save interval. |
| `validation_iter` | `5000` | Phase 1 only. Validation (`nn_score`) logging interval. |
| `num_gpus` | `1` | Forwarded to Phase 1 (finetune) and Phase 3 (SDG). Eval and search rounds stay single-GPU. |
| `model_size` | `2b` | `2b` or `14b`. Used by finetune and SDG. On-disk checkpoint path encodes in upper-case (`2b`→`2B`, `14b`→`14B`). |
| `lr` | `0.02` | Phase 1 only. Learning rate. |
| `batch_size` | `2` | Phase 1 only. Per-GPU batch size. |
| `image_size` | `512` | Phase 1 only. Training resolution (square). |
| `guidance_range` | `1.5 10.0` | Phase 5 search draw range for guidance. |
| `crop_ratio_range` | `1.5 10.0` | Phase 5 search draw range for crop_ratio. |

---

## Mode validation (fail fast before any phase)

- `mode` unset → halt: *"`mode` is required (`full` | `inference_only` | `finetune_only`)."*
- `mode=inference_only` missing either `checkpoint_dir` or `step` → halt: *"inference_only requires both `checkpoint_dir` and `step`."*
- `mode=full` with `checkpoint_dir` or `step` supplied → halt: *"full mode runs finetune; use `mode=inference_only` to reuse an existing checkpoint."*

## Shared variables

Set once before Phase 0:

```bash
MODE=<full|inference_only|finetune_only>
NAME=<exp>
DATASET_DIR=<dataset_dir>
CLEAN_DIR=${clean_dir:-${DATASET_DIR}}
CKPT=<checkpoint_dir>      # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
STEP=<iter>                # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
NUM_SDG=<N>
DEFECT_DESC=<defect_spec.jsonl>
DEFECTS=(T+A T+B)          # TEXTURE+TYPE names. For mode=inference_only, derive from ${CKPT}/ag_config.yaml → dataloader_train.dataset.anomaly_types (also printed by validate_checkpoint.py in Phase 0). For mode=full, take from DEFECT_DESC entries. See references/inference.md §Phase 0.
NUM_SEARCH_RUN=${num_search_run:-3}
NN_THRESHOLD=${nn_threshold:-0.4}
MODEL_SIZE=<2b|14b>
NUM_GPUS=${num_gpus:-1}
MAX_ITER=${max_iter:-75000}
SAVE_ITER=${save_iter:-5000}
VALIDATION_ITER=${validation_iter:-5000}
LR=${lr:-0.02}
BATCH_SIZE=${batch_size:-2}
IMAGE_SIZE=${image_size:-512}
VALIDATION_JSONL=${validation_jsonl:-}  # optional; set by Phase 1 Step 2 if not user-supplied

BASE=results/${NAME}
JSONL=ag_inference/${NAME}/testcase.jsonl
ORIGINAL=${BASE}/original
SEARCHED=${BASE}/searched
ROUNDS=${BASE}/rounds
REGENS=${BASE}/regens
```

## Guard preflight (product mode only)

When `ANOMALYGEN_PRODUCT_MODE=1`, run
`.agents/skills/anomalygen-guard/scripts/preflight.py` before any GPU work and
fix any `BLOCKED` issues. `--validation-jsonl` is forwarded only when the user
supplied one; for `MODE=finetune_only` omit `--num-sdg` if not supplied. See
`references/guard-and-custom-counts.md` for the full preflight command with all
forwarded flags and the validation-JSONL / `allocate_samples.py` 0-entry
checks.

---

## Phase 0 — checkpoints

Read `references/finetune.md §Phase 0` for HF_TOKEN requirements and what gets
downloaded (~140 GB). Verify first; download only what is missing.

```bash
${ANOMALYGEN_SCRIPTS}/check.sh \
    || ${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh
```

---

## Phase 1 — fine-tune (skip when `MODE=inference_only`)

Read `references/finetune.md §Phase 1` for dataset structure, config template
details, and best-checkpoint selection. Four steps: (1) validate dataset /
derive anomaly types, (2) generate the validation JSONL (skip if user supplied
`VALIDATION_JSONL`), (3) generate the training config — **show it to the user
and confirm before writing** — (4) launch training in the background. Then
derive `CKPT` (path encodes upper-case `MODEL_SIZE`) and `STEP` (highest
`nn_score` step from validation logs). If `MODE=finetune_only`, stop after
training. See `references/finetune-commands.md` for the exact Step 1–4 commands
and the `CKPT`/`STEP` derivation snippet.

---

## Phase 2 — prep-testcase (skip when `MODE=finetune_only`)

Read `references/inference.md §Phase 2` for AMP routing detail and n_seeds
sizing. Do NOT pass `--seeds` — it is auto-computed and is not a recognized
flag. `prep_testcase.sh` defaults to `--mode inference` (uniform allocation
across defect types, no KPI floor), which Phase 2 always uses.

```bash
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name ${NAME} --num-sdg ${NUM_SDG} \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/${NAME}/amp \
    --output-jsonl ${JSONL}
```

**Custom per-defect counts:** when the user specifies counts per defect type,
translate to `--num-sdg` plus a `--per-defect-counts` JSON dict (types absent
from the dict get 0; sum should equal `--num-sdg`, else the script warns on
stderr and uses the override sum). Confirm the allocation when intent is
ambiguous. See `references/guard-and-custom-counts.md` for the full
`--per-defect-counts` command example and the ambiguity-handling detail.

---

## Phase 3 — SDG → `original/`

Read `references/inference.md §Phase 3` for JSONL validation against the
checkpoint, multi-GPU caveats, and output verification.

```bash
python3 -m scripts.utilities.validate_checkpoint ${CKPT} --step ${STEP}
python3 -m scripts.utilities.validate_jsonl ${CKPT} ${JSONL}

${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir ${CKPT} --step ${STEP} \
    --input_jsonl ${JSONL} --output_dir ${ORIGINAL} \
    --model_size ${MODEL_SIZE} --num_gpus ${NUM_GPUS}

${ANOMALYGEN_SCRIPTS}/verify_output.sh ${JSONL} ${ORIGINAL}
```

---

## Phase 4 — eval `original/`

Read `references/inference.md §Eval` for score interpretation and feature-count
explanation. `run_eval.sh` writes `per_sample.csv` and `eval.log` inside
`original/` and merges `nn_score` into `SDG_result.csv`.

```bash
${ANOMALYGEN_SCRIPTS}/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${ORIGINAL} \
    --anomaly-types ${DEFECTS[@]}
```

---

## Phase 5 — per-sample search rounds

Read `references/inference.md §Phase 5` for draw strategy, ranges, and re-AMP
guidance. For `r` in `1..NUM_SEARCH_RUN`:

1. Read prior round's `per_sample.csv` (or `${ORIGINAL}/per_sample.csv` for `r=1`).
2. Write `${ROUNDS}/round_${r}/draws.json` with selected `(guidance, crop_ratio)` per sample.
3. Run round via `${ANOMALYGEN_SCRIPTS}/run_round.sh` (SDG + eval; the round dir
   gets its own `sdg/{SDG_result.csv, per_sample.csv, eval.log}`). See
   `references/inference-commands.md §Phase 5` for the full command and flags.

`NUM_SEARCH_RUN=0` is valid — skip this phase entirely and let Phase 6
clone `original/` into `searched/`.

---

## Phase 6 — assemble `searched/` (stitch only)

Always run assemble (works with 0 rounds — `searched/` clones `original/`, so
downstream always reads `searched/` regardless of `num_search_run`). Stitch-only:
copies winning images per sample-index into `searched/` and carries over
per-sample `nn_score` / `mnn_score` from each pick's source-round per_sample.csv.
No eval — Phase 7 emits the canonical `searched/eval.log`.

```bash
mkdir -p ${ROUNDS}
python3 -m scripts.utilities.assemble_searched \
    --original-dir ${ORIGINAL} --original-csv ${ORIGINAL}/per_sample.csv \
    --rounds-dir ${ROUNDS} --searched-dir ${SEARCHED}
```

---

## Phase 7 — filter + regen + eval (default `nn_threshold=0.4`)

Phase 7 **runs by default** (`nn_threshold=0.4`) on every `mode=full` and
`mode=inference_only` invocation; pass `nn_threshold=0` to skip it. It filters
`searched/` by `nn_threshold`, regenerates dropped samples via re-AMP (fresh
`(clean, submask)` pairing in the same defect type) for up to 5 attempts, then
falls back to best-scoring non-passing regens and finally to dropped originals,
so the final bucket always equals `num_SDG`.

Run `python3 -m scripts.utilities.filter_with_regen`. It runs the final
`run_eval.sh` internally — the only eval against `searched/`. Read
`references/inference.md §Phase 7` for regen mechanics, source-column tracing,
and the `regens/regen_summary.csv` schema; see
`references/inference-commands.md §Phase 7` for the full command and flags.

---

## Output layout

Every bucket that gets eval'd carries the same triad of files:
`SDG_result.csv` (generation params + `nn_score`), `per_sample.csv`
(per-sample nn + mnn), and `eval.log` (aggregate FID / per-defect avg).
Buckets live under `results/<name>/` as `original/` (Phase 3+4), `searched/`
(Phase 6 stitch + Phase 7 filter+regen+eval), `rounds/round_NN/` (Phase 5,
plus `search_summary.csv`), and `regens/regen_NN/` (Phase 7, plus
`regen_summary.csv`).

See `references/output-layout.md` for the full directory tree with per-file
annotations and the post-run **Verification** checklist (image counts per
bucket, `search_summary.csv` / `regen_summary.csv` row checks, and the per-type
`nn_score` / `mnn_score` / `fid` fields in each `eval.log`).

## Error handling

Common pipeline failure modes (missing mask dirs, short/empty AMP output and
the `0 entries written` halt, mid-round SDG failure resume, off-boundary
`step`) are covered in `references/error-handling.md`; see also
`references/finetune.md` and `references/inference.md` for phase-specific
error handling.
