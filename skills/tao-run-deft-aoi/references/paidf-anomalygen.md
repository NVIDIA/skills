# Cosmos AnomalyGen — DEFT Loop Reference

Read this when the parent runs the `anomalygen` stage. The underlying skill
`tao-skill-bank:paidf-anomalygen` (`data/paidf-anomalygen/SKILL.md`) owns
the standalone 8-phase pipeline and parameter reference. This file is the
DEFT-loop overlay: what to pass, how mounts resolve, the few invariants
that gate the run, and the failure mode the loop has actually hit.

**Per-iteration**, the DEFT loop runs the underlying skill in
`mode=inference_only` and only needs Phases 2 (`prep_testcase.sh`) and 3
(`run_sdg.sh`). Phases 4–7 (eval / search / filter+regen) are SDG-quality
optimization and do not contribute to the loop's training pairs. Skip them
by setting `num_search_run=0` and `nn_threshold=0`, or invoke the two
wrappers directly (see *Per-iteration invocation* below).

**Once, before the loop starts**, the post-gate bootstrap (SKILL.md →
*Workflow* step 2) populates whichever assets are missing — Cosmos base
checkpoints, PCB reference dataset, and the AnomalyGen fine-tuned
checkpoint — all auto-download by default; bring-your-own to override.
Pre-Flight only **probes** status and reports it — no side-effecting work
happens before the user gate.

## Workspace Inputs

Three independent inputs under `<workspace>/augmentation/anomalygen/` plus
the Cosmos base-checkpoints cache.

| Input | Path (e.g. `<project>=nvpcb`) | Holds | Source (bring-your-own OR auto-populate) |
|---|---|---|---|
| `checkpoint_dir` | `augmentation/anomalygen/checkpoints/<project>/` | `ag_config.yaml` + `checkpoints/{latest_checkpoint.txt, model/iter_<step>.pt, …}` | **Auto-download** by default from HF (see *Fine-tuned checkpoint sources* below); **BYO** to override (pre-stage the dir). |
| `dataset_dir` | `augmentation/anomalygen/datasets/<project>/` | PCB reference data + `semantic_segmentation_labels.json` + `defect_spec.jsonl`. Sibling to `checkpoints/`. | **BYO:** pre-stage. **Auto:** `python3 -m scripts.utilities.prepare_dataset_uc1 <dir>` (HF `nvidia/Cosmos-AnomalyGen-PCB-Dataset`). `uc1` is the paidf-anomalygen skill's identifier for the PCB use-case — the script name is unrelated to `<project>`. |
| `defect_spec` | `${dataset_dir}/defect_spec.jsonl` | One entry per defect_type (`<T>+<A>`); `spatial_dependency ∈ {free, text, cad}` | Bundled with `dataset_dir` (either path). Template at `data/paidf-anomalygen/assets/defect_spec_template.jsonl`. |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` (resolved by Pre-Flight) | Cosmos base checkpoints — `nvidia/Cosmos-Predict2-2B-Text2Image/`, `google-t5/`, `NVDINOV2/`, … | **BYO:** pre-stage. **Auto:** container's `${ANOMALYGEN_SCRIPTS}/check.sh \|\| download_checkpoints.sh`. Post-gate bootstrap runs this once with a `:rw` mount; persists across runs. |

DEFT AOI is always a PCB workflow; the `<project>` placeholder is just the
directory label the user picked for this AnomalyGen project's checkpoint +
dataset (commonly `UC1` or `nvpcb`). The loop reads it from
`deft_state.json::config.anomalygen.project`.

`dataset_dir` and `clean_dir` resolve to the same path on this workspace —
clean images live under `<dataset_dir>/<T>/clean_image/` which is the
container's first probe hit. The container handles both flat and
split-by-texture layouts transparently via `validate_amp_inputs.py`; the
loop passes the workspace dir verbatim, no pre-staging.

## Fine-tuned checkpoint sources

The 2B Cosmos-Predict2 finetunes for each UC are published on Hugging Face
and can be downloaded with a helper script:

```bash
scripts/utilities/download_anomalygen_checkpoints.sh --uc {pcb|metal|glass|all} \
    [--checkpoint-dir checkpoints]
```

| UC | HF repo | Iter |
|---|---|---|
| pcb | [`nvidia/Cosmos-AnomalyGen-PCB-2B`](https://huggingface.co/nvidia/Cosmos-AnomalyGen-PCB-2B) | 14000 |
| metal | [`nvidia/Cosmos-AnomalyGen-Metal-2B`](https://huggingface.co/nvidia/Cosmos-AnomalyGen-Metal-2B) | 10000 |
| glass | [`nvidia/Cosmos-AnomalyGen-Glass-2B`](https://huggingface.co/nvidia/Cosmos-AnomalyGen-Glass-2B) | 9000 |

Each repo ships the checkpoint plus its `ag_config.yaml` (which lists the
supported anomaly types and trained `image_size`). Point the AnomalyGen
pipeline at the downloaded directory via `checkpoint_dir=`. DEFT AOI is
always a PCB workflow — the loop selects `--uc pcb` and reads
`step=14000` from `${checkpoint_dir}/checkpoints/latest_checkpoint.txt`.

## Invariants

Verify these before invoking; the rest is up to the container.

1. **`cad_mask` preserves per-class RGB.** `cad2roi` looks up each pixel's
   RGB tuple in `semantic_segmentation_labels.json`. A flattened binary
   `(0,0,0)`/`(255,255,255)` cad_mask yields zero ROIs everywhere (see
   *AMP no-ROI failure mode* below). Verify with
   `Image.open(cad_mask).convert('RGB').getcolors(maxcolors=64)` —
   unique tuples must overlap the labels file.
2. **`defect_spec.jsonl` `text` entries have non-empty
   `roi_prompt_defect_location`.** `cad` and `free` entries don't need it.
3. **`<T>/cad_mask/` and `<T>/clean_image/` are non-empty and paired by
   stem.** Missing pair → record dropped silently.
4. **`semantic_segmentation_labels.json` exists at `datasets/<project>/`.**

Mask file format, image-size agreement, and channel mode do **not** gate
`mode=inference_only` — AMP processes each record at its native size. See
the underlying skill's `references/inference.md` if you need the full
list for `mode=full` / `mode=finetune_only`.

## AMP "no ROI candidates" failure mode

`run_auto_roi_amp.py` silently skips a sample when the cad_mask doesn't
have enough free area for the requested anomaly mask shape. The wrapper
does **not** propagate this — `num_SDG=N` quietly degrades to whatever
AMP could allocate, and the loop only notices via a smaller
`SDG_result.csv`.

Symptoms:

```
WARNING ... <stem>/<T>+<A>: no ROI candidates, skipping
INFO ... <T>+<A>: 0/N with ROI, 0/0 seeds OK
wrote 4 entries to testcase.jsonl       # <-- expected 20
```

Diagnose in this order:

1. **cad_mask class mapping** — invariant #1 above. Most common cause.
2. **Anomaly mask shape vs cad free area** — if the anomaly mask's
   bounding box exceeds every connected component in the cad_mask, AMP
   can't place it. Provide smaller anomaly masks or switch
   `spatial_dependency: free` to skip ROI placement entirely.
3. **Isolate the failing defect** — filter `defect_spec.jsonl` to just
   `<T>+<A>` and re-run `prep_testcase.sh --num-sdg 1`.

After Phase 2, parse `<output_dir>/allocation.json` to confirm per-defect
counts before launching Phase 3 — GPU + model load cost is fixed, so a
4-of-20 yield is worth aborting on.

## DEFT-Loop Parameters

The parent invokes `tao-skill-bank:paidf-anomalygen` (or the wrappers
directly) with:

| Param | Value | Notes |
|---|---|---|
| `mode` | `inference_only` (or omit when calling wrappers directly) | DEFT loop never runs Phase 1 |
| `checkpoint_dir` | `<workspace>/augmentation/anomalygen/checkpoints/<project>` | |
| `step` | int parsed from `checkpoint_dir/checkpoints/latest_checkpoint.txt` | strip `iter_` prefix and `.pt` suffix |
| `dataset_dir` | `<workspace>/augmentation/anomalygen/datasets/<project>/` | passed verbatim |
| `clean_dir` | same as `dataset_dir` | |
| `defect_spec` | `${dataset_dir}/defect_spec.jsonl` | |
| `num_SDG` | per-iter budget from `deft_state.json` | proportionally allocated across defect types by mask count |
| `num_gpus` | `1` | |
| `model_size` | from `ag_config.yaml` (`2b` or `14b`) | |
| `output_dir` | `${RESULTS_DIR}/iter${N}/anomalygen/sdg/` | receives `reconstructed_image/`, `original_image/`, `SDG_result.csv` |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` | resolved in Pre-Flight |
| `num_search_run` | `0` | skip Phase 5 search rounds |
| `nn_threshold` | `0` | skip Phase 7 filter+regen |

## Shared shell setup

Used by both the bootstrap and the per-iteration calls:

```bash
set -a; source <workspace>/.env; set +a
WS=<workspace>
DS=$WS/augmentation/anomalygen/datasets/<project>
CKPT=$WS/augmentation/anomalygen/checkpoints/<project>
COSMOS=$WS/augmentation/anomalygen/base_checkpoints
RUN_DIR=$WS/results/run_<TS>/iter${N}/anomalygen
: "${AG_IMAGE:=$(${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_versions_key.py images.metropolis_sdg.paidf_anomalygen)}"  # reuses Pre-Flight export if set; resolves on demand otherwise
mkdir -p $COSMOS $DS $(dirname $CKPT) $RUN_DIR/amp $RUN_DIR/sdg
chmod 777 $COSMOS $DS $(dirname $CKPT)   # container runs as uid 10000; without this the post-gate bootstrap fails with PermissionError on host-owned mounts
```

Required env across every call: `HF_TOKEN`, `HF_HUB_DISABLE_XET=1`,
`PYTHONPATH=/workspace/paidf-anomalygen`. Required workdir:
`/workspace/paidf-anomalygen` (the `-m scripts.…` invocation needs CWD —
this matches the container's `WORKDIR` / `ENV PYTHONPATH`; older revisions
of this file used `/workspace/paidf-anomalygen`, which does not exist
inside the current `paidf-anomalygen:1.0.0+` image).
`${ANOMALYGEN_SCRIPTS}` is preset inside the container — do not export it
on the host. Use **single quotes** around the inner `bash -lc` so the host
shell doesn't expand `${ANOMALYGEN_SCRIPTS}`; use **double quotes with
escaped `\$`** (`bash -lc "\${ANOMALYGEN_SCRIPTS}/..."`) when you also need
host-side variables (like `$DS`, `$NUM_SDG`) expanded in the same line.

## Post-gate bootstrap (one-time, SKILL.md → Workflow step 2)

Order matters — (a) populates the base checkpoints that (b) depends on.
Run only the steps the Pre-Flight Summary flagged `WILL_AUTO_FETCH`. Both
are idempotent — re-running a completed step exits quickly.

```bash
# (a) Cosmos base checkpoints (~22 GB for 2B-only, ~140 GB with 14B + T5-11b).
# WRITABLE mount (no :ro) so download_checkpoints.sh can populate the cache.
docker run --rm \
  --user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/paidf-anomalygen \
  -v $COSMOS:/workspace/paidf-anomalygen/checkpoints \
  -w /workspace/paidf-anomalygen $AG_IMAGE \
  bash -lc '${ANOMALYGEN_SCRIPTS}/check.sh || ${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh'

# (b) PCB reference dataset — prepare_dataset_uc1.py is the paidf-anomalygen
# skill's PCB-dataset fetcher (`uc1` = the skill's identifier for the PCB
# use-case; unrelated to the host-side <project> directory label).
if [ ! -f "$DS/defect_spec.jsonl" ]; then
  docker run --rm \
    --user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/paidf-anomalygen \
    -v $WS:$WS -w /workspace/paidf-anomalygen $AG_IMAGE \
    python3 -m scripts.utilities.prepare_dataset_uc1 $DS
fi
```

The AnomalyGen fine-tuned checkpoint at `$CKPT` auto-downloads by default
from the per-UC HF repo (see *Fine-tuned checkpoint sources* above) via:

```bash
# (c) AnomalyGen fine-tuned checkpoint (PCB UC; ~5 GB).
if [ ! -f "$CKPT/checkpoints/latest_checkpoint.txt" ]; then
  docker run --rm \
    --user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/paidf-anomalygen \
    -v $WS:$WS -w /workspace/paidf-anomalygen $AG_IMAGE \
    bash -lc "scripts/utilities/download_anomalygen_checkpoints.sh \
      --uc pcb --checkpoint-dir $CKPT"
fi
```

Users who want to override pre-stage the dir before the loop starts.

## Per-iteration invocation (every loop iteration)

After bootstrap, the per-iteration AnomalyGen stage is two `docker run`
calls — same image, READ-ONLY mount on the cosmos cache.

```bash
STEP=$(sed 's/^iter_0*\([0-9]*\)\.pt$/\1/' $CKPT/checkpoints/latest_checkpoint.txt)

# Phase 2: AMP routing → testcase.jsonl  (~10s, no GPU)
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/paidf-anomalygen \
  -v $WS:$WS -v $COSMOS:/workspace/paidf-anomalygen/checkpoints:ro \
  -w /workspace/paidf-anomalygen $AG_IMAGE \
  bash -lc "\${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name iter${N} --num-sdg $NUM_SDG \
    --dataset-dir $DS --clean-dir $DS --defect-spec $DS/defect_spec.jsonl \
    --amp-output-dir $RUN_DIR/amp --output-jsonl $RUN_DIR/testcase.jsonl"

# Phase 3: SDG diffusion → reconstructed_image/ + original_image/  (1-3 min on Blackwell)
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/paidf-anomalygen \
  -v $WS:$WS -v $COSMOS:/workspace/paidf-anomalygen/checkpoints:ro \
  -w /workspace/paidf-anomalygen $AG_IMAGE \
  bash -lc "\${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir $CKPT --step $STEP \
    --input_jsonl $RUN_DIR/testcase.jsonl --output_dir $RUN_DIR/sdg \
    --model_size 2b --num_gpus 1"
```

Required mounts (per-iteration): `<workspace>:<workspace>` (same path) +
`<cosmos_models_dir>:/workspace/paidf-anomalygen/checkpoints:ro`.

## Output layout

```
<output_dir>/
├── SDG_result.csv                          # one row per generated sample (image, mask, params, PSNR)
├── reconstructed_image/<T>+<A>_<idx>.png   # synthetic NG — ChangeNet input_path
├── original_image/<T>+<A>_<idx>.png        # paired OK — ChangeNet golden_path
├── original_mask/, cropped_image/, cropped_mask/, annotated_image/   # intermediates
└── timing_summary.json
```

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label iter${N} \
    --stage anomalygen --status ok \
    --summary "SDG: requested=N, AMP-allocated=M, generated=K by type"
```

When `M < N` (AMP yield gap), include both requested and allocated counts
— that's the signal a reviewer needs to spot allocation-vs-generation
bottlenecks.
