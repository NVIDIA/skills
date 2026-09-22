# Visual ChangeNet — DEFT Loop Reference

Read this when the parent runs the `train`, `inference`, or `evaluate` stage. The
underlying skill `tao-skill-bank:tao-train-visual-changenet` (`skills/models/tao-train-visual-changenet/SKILL.md`)
owns the docker invocation, spec format, CSV format, lighting conventions, and error
patterns — its `## Local Docker Invocation` section has the exact docker run command
(including `--shm-size=8g`, backbone file mount, and how to override
checkpoint/results_dir on the command line without editing the spec). This file only
covers the DEFT-loop-specific overlay: mounts, spec paths, two-checkpoint compare,
KPI sweep and `deft_state.json` updates.

DEFT AOI is intentionally plain-train for Visual ChangeNet. When invoking the
underlying model skill for any train stage, pass `automl_policy: off` so this
workflow bypasses model-level AutoML while leaving Visual ChangeNet metadata
unchanged for other workflows.

## DEFT-Loop Mount Layout

Read `IMAGES_DIR` from `deft_state.json::config.images_dir`; for a new
workspace it is `<workspace>/images` (legacy `<workspace>/kpi/images` is only a
Pre-Flight fallback).

`STAGED` is the host file returned by `stage_backbone.py`.
`BACKBONE_CONTAINER_PATH` is the matching container path written into the
training spec. Use this resolved pair for every train, evaluate, and inference launch.

For every direct Docker action in the underlying reference, first verify the
DEFT results root is writable, then compose this application-owned identity
block:

```bash
mkdir -p "$RESULTS_DIR"
probe="$RESULTS_DIR/.tao-write-probe.$$"
(umask 077 && : >"$probe" && rm -f "$probe") || {
  echo "FATAL: $RESULTS_DIR is not writable by uid $(id -u)" >&2
  exit 2
}

VCN_IDENTITY_ARGS=(
  --user "$(id -u):$(id -g)"
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
```

Insert `"${VCN_IDENTITY_ARGS[@]}"` into the underlying `docker run` command for
train, evaluate, inference, export, and quantize, including resumed actions.
Every resulting checkpoint, status file, inference CSV, and export is then
owned by the submitting host user, so two-checkpoint evaluation and cleanup
need no sudo.

```
-v <workspace>:/data/workspace                                  # combined iter CSVs + staged images
-v ${RESULTS_DIR}:/results                                      # canonical run root; never /results/iterN
-v ${IMAGES_DIR}:/data/datasets/NV_PCB_Siamese/images            # canonical real-image root from state
-v <workspace>/train/base:/data/datasets/NV_PCB_Siamese/csv      # training_set.csv, validation_set.csv
-v <workspace>/kpi:/data/datasets/NV_PCB_Siamese/kpi             # testing_set.csv
-v "${STAGED}:${BACKBONE_CONTAINER_PATH}:ro"                    # selected backbone file
```

## Spec Key Paths (container-side)

| What | Container path |
|---|---|
| Training CSV (iter N) | `/data/workspace/results/run_<TS>/iter${N}/dataset/train_combined_iter${N}.csv` |
| Validation CSV | `/data/datasets/NV_PCB_Siamese/csv/validation_set.csv` |
| KPI test CSV | `/data/datasets/NV_PCB_Siamese/kpi/testing_set.csv` |
| images_dir | `/data/datasets/NV_PCB_Siamese/images` |
| Results dir (baseline / iter N) | `/results/baseline` / `/results/iter${N}` |

## Spec `output_dir` Contract

`baseline_spec.yaml` (and every per-iter spec the loop derives from it) **must**
set the train task's `output_dir` to the canonical `<stage>` subdirectory under
the iteration root, **not** to the iteration root itself:

| Task | Required spec `output_dir` |
|---|---|
| baseline train | `/results/baseline/train/` |
| baseline inference | `/results/baseline/inference/` |
| baseline evaluate | `/results/baseline/evaluate/` |
| iter N train | `/results/iter${N}/train/` |
| iter N inference | `/results/iter${N}/inference/` |
| iter N evaluate | `/results/iter${N}/evaluate/` |

Writing to the iteration root (e.g. `/results/baseline/`) causes the
parent's pre-create / checkpoint-discovery / Output Layout (see
`SKILL.md → ## Output Layout`) to diverge from where TAO actually writes,
which manifests as "checkpoint not found" downstream. Edit the spec to match
the container paths above before launching; they map to the corresponding
`${RESULTS_DIR}/...` host directories through the single `/results` mount.
Do not change or nest that mount.

Before every train launch, validate these coupled spec invariants:

- Use `--shm-size=8g` for ChangeNet and do not combine it with `--shm-size=8g`.
  `--shm-size=8g` makes the container inherit a CI host `/dev/shm` that may be
  only 64 MiB, causing a mid-epoch bus/shared-memory failure even though the
  command also says `--shm-size=8g`.
- `train.checkpoint_interval <= train.num_epochs`. A user override such as
  `epoch 1` must also lower `checkpoint_interval`; otherwise TAO aborts before
  the first epoch.
- Derive inference/evaluate specs from the exact training spec. Preserve the
  model architecture, set `dataset.classify.augmentation_config.augment=false`,
  and change only task paths/checkpoint/results overrides. Keep only the
  matching `train.classify.loss` compatibility stub while the skill baseline
  remains TAO 7.1; TAO 7.2 images containing NVIDIA-TAO/tao-pytorch#107 no
  longer require it.
- Use the underlying skill's documented `visual_changenet <task> -e <spec>`
  entrypoint. Do not switch to direct package-module/Hydra commands after an
  error; their config-path semantics differ.

## DEFT Iter Training — Init Convention

For every iteration N≥1, **init from the previous iter's best checkpoint via `train.pretrained_model_path`, not `train.resume_training_checkpoint_path`.**

```bash
# CORRECT for DEFT iter N (fresh epoch counter, weights from prev best)
train.pretrained_model_path=${prev_best_ckpt}

# WRONG for DEFT iter N — Lightning inherits current_epoch from the checkpoint,
# sees current_epoch >= max_epochs (baseline already used up max_epochs),
# and exits with `Trainer.fit stopped: max_epochs=N reached` after zero training steps.
train.resume_training_checkpoint_path=${prev_best_ckpt}
```

`resume_training_checkpoint_path` is for **interrupted-run resumption** within the same iteration (preserves optimizer state, scheduler, epoch counter — semantics designed for "kill -9 → restart" cases). DEFT iters logically restart the trainer for a new dataset + epoch budget, so they need fresh `pretrained_model_path` init.

Failure mode is silent: `Execution status: PASS` despite no training. Symptom: iter N's train output dir has no new `model_epoch_*.pth`. If you see this, switch the flag.

`commit_stage.py --stage train` requires the exact training spec and rejects a
checkpoint outside `${RESULTS_DIR}/<iter-label>/train/`. For iter N this means
the baseline or previous-iteration checkpoint can initialize training, but it
cannot be committed as iter N's output. Do not copy an old checkpoint into the
new directory to satisfy this check; the file must be emitted by the successful
current train invocation.

## Per-Iter Spec `images_dir` — Asymmetric

When deriving `iter${N}_spec.yaml` from `baseline_spec.yaml`, **only `train_dataset.images_dir` moves to the workspace root**; the other dataset blocks keep the canonical real-images mount:

| Dataset block | images_dir (container path) | Why |
|---|---|---|
| `train_dataset` | `/data/workspace` | iter combined CSV mixes base rows (`images/...`) and SDG rows (`results/run_<TS>/iter${N}/dataset/images/...`) — both are workspace-root-relative after assembly |
| `validation_dataset` | `/data/datasets/NV_PCB_Siamese/images` | validation_set.csv carries paths relative to the canonical real-image root; unchanged from baseline |
| `test_dataset` | `/data/datasets/NV_PCB_Siamese/images` | same — usually points at validation_set.csv |
| `infer_dataset` | `/data/datasets/NV_PCB_Siamese/images` | testing_set.csv carries paths relative to the canonical real-image root |

A bulk `sed 's|/data/datasets/NV_PCB_Siamese/images|/data/workspace|g'` on the spec catches all four and breaks the latter three. Edit `train_dataset.images_dir` surgically.

## Two-Checkpoint Compare

The baseline template intentionally leaves all downstream checkpoint fields
empty. After a successful train, discover the concrete `model_epoch_*_step_*.pth`
files and the `changenet_model_classify_latest.pth` symlink under that
iteration's `train/` directory. Pass the selected checkpoint as an explicit
container path to inference, evaluate, or export. Never use
`${results_dir}/train/...`: Hydra rebases `${results_dir}` to the current
action's output directory.

Run inference on both the best-val checkpoint (lowest `val_loss`) and the
latest checkpoint (highest epoch). `val_loss` and the customer's deployment
metric can diverge. Run the evaluator from `state.metric_contract` on every
candidate and select lowest for `<`/`<=` or highest for `>`/`>=`; never infer
direction from the metric name.

Only checkpoints from a train command that exited zero and whose TAO status
JSONL contains no `FAILURE` entry are candidates. A checkpoint emitted before
a failed batch is partial residue: log `train/status=error`, hard-stop, and do
not run inference or evaluation on it.

## analyze_kpi.py

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/analyze_kpi.py \
    ${RESULTS_DIR}/iter${N}/inference/<label>/inference.csv \
    --output-dir ${RESULTS_DIR}/iter${N}/inference/<label>
```

Run the evaluator selected by `state.metric_contract` and commit its standard
`metric_result.json`. Evaluator-specific diagnostics are secondary artifacts;
command and artifact evaluators follow `references/metric-contract.md`.

## Output to deft_state.json

```json
{
  "iterations": {
    "iter${N}": {
      "status": "complete",
      "stage_completed": "evaluate",
      "best_ckpt_path": "<abs_host_path>",
      "best_ckpt_kind": "best_val|latest",
      "metric_result": {
        "name": "<metric_contract.name>",
        "value": <float>,
        "unit": "<metric_contract.unit>",
        "passed": <bool>,
        "constraints": {},
        "evidence_path": "<abs_path>/metric_result.json"
      },
      "threshold": <float_or_null>,
      "val_loss": <float>,
      "inference_csv": "<abs_host_path>",
      "training_spec": "<abs_host_path_to_the_exact_iter_spec>"
    }
  }
}
```

The baseline uses the same shape at
`state["iterations"]["baseline"]`. After train but before evaluate, commit
`stage_completed="train"` and keep `status="in_progress"`.
Pass the evaluator artifact and evidence paths to `scripts/commit_stage.py`;
it invokes the metric recorder and writes the evaluate fields transactionally.
Compatibility fields, when applicable, are written by the recorder. Treat the
JSON above as schema, not an editing recipe; only the bundled scripts may write
it.

## ChangeNet backbone resolution

`model.backbone.pretrained_backbone_path` **must point to an existing local file on the host that is bind-mounted into the container.** TAO's `ptm_utils.load_pretrained_weights()` hands the string straight to `torch.load(path, ...)` (with a special-case branch when the suffix is `.safetensors`, calling `safetensors.torch.load_file`). It does **not** dereference `https://`, `hf://`, or HuggingFace repo IDs — passing a URL produces `FileNotFoundError: [Errno 2] No such file or directory: 'https://...'` and `Execution status: FAIL` within ~3 s.

Accepted forms (TAO 7.0.0-rc-224):

| Form | Status |
|---|---|
| Local path to `.pth` / `.ckpt` checkpoint | ✓ works (`torch.load`) |
| Local path to `.safetensors` file | ✓ works (`safetensors.torch.load_file`) |
| `https://huggingface.co/...` URL | ✗ FileNotFoundError |
| HF repo id like `nvidia/C-RADIOv2-B` | ✗ FileNotFoundError |
| `null` or empty | ✗ silently degrades held-out evaluation quality; failure mode looks like a training bug |

### Pre-Flight responsibility

Pre-Flight **must stage the backbone locally** before launch. The HuggingFace repo `nvidia/C-RADIOv2-B` ships only `model.safetensors` (no `.pth`). Use the packaged staging script (idempotent; reuses an existing staged file; hard-fails if it cannot produce one):

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
STAGED=$("$PYTHON" <skill_root>/scripts/stage_backbone.py --workspace <workspace>)
BACKBONE_CONTAINER_PATH="/data/pretrained_models/$(basename "$STAGED")"
# Rewrite model.backbone.pretrained_backbone_path in the baseline and derived
# specs to exactly ${BACKBONE_CONTAINER_PATH}.
```

Equivalent manual recipe (only if running the script is not possible):

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(repo_id="nvidia/C-RADIOv2-B", filename="model.safetensors")
dst = "<workspace>/augmentation/backbone/c_radio_v2_b.safetensors"
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)
PY
```

Then mount that exact host/container pair in every train, evaluate, and inference
Docker invocation:

```bash
-v "${STAGED}:${BACKBONE_CONTAINER_PATH}:ro"
```

Set the spec field to the same container-side path:

```yaml
model:
  backbone:
    pretrained_backbone_path: /data/pretrained_models/<staged-backbone-filename>
```

If `HF_TOKEN` is unset or the workspace already has a staged file, Pre-Flight uses the staged file as-is and skips the download. If neither is available, Pre-Flight **hard stops** — there is no working URL fallback in this TAO version, so silently falling through would just produce the FileNotFoundError above after the container starts.

### Frozen DINOv3 profiles

DEFT supports `vit_small_dinov3`, `vit_small_plus_dinov3`,
`vit_base_dinov3`, `vit_large_dinov3`, `vit_huge_plus_dinov3`, and
`vit_7b_dinov3`. The baseline spec must use the same type and set
`model.backbone.freeze_backbone: true`:

```yaml
model:
  backbone:
    type: vit_large_dinov3
    freeze_backbone: true
    pretrained_backbone_path: ''
```

In network mode, accept the DINOv3 license and export `HF_TOKEN`; Pre-Flight
then stages the matching `timm/vit_*_patch16_dinov3.lvd1689m` file with:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
STAGED=$("$PYTHON" \
  <skill_root>/scripts/stage_backbone.py \
  --workspace <workspace> --backbone-type vit_large_dinov3)
BACKBONE_CONTAINER_PATH="/data/pretrained_models/$(basename "$STAGED")"
```

Pre-Flight rewrites `pretrained_backbone_path` to that container path and uses
the same mount for train, evaluate, and inference. In air-gap mode, pre-stage
the exact profile file; the helper rejects a missing or mismatched checkpoint.

## Label case rule (CSV assembly)

TAO's ChangeNet classify dataloader does case-sensitive equality against the
literal string `"PASS"` to detect class 0. Lowercasing it puts every row into
class 1 and the `fpratio_sampling` weighted sampler fails immediately at
training start:

```
RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
RuntimeError: Please call iter(combined_loader) first.
```

Failures reproduce within ~30 s of launching training. The rule: keep `PASS`
exactly as-is; lowercase + strip only the non-`PASS` labels, so `"Missing"`
and `"missing"` collapse to one defect class while `"PASS"` stays the class-0
sentinel.

```python
row["label"] = row["label"] if row["label"] == "PASS" else row["label"].lower().strip()
```

## Log Stage

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label <baseline|iter${N}> \
    --stage train \
    --best-ckpt <absolute selected checkpoint> \
    --best-ckpt-kind <best_val|latest> \
    --training-spec <absolute exact training spec> \
    --val-loss <float> \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "Train: val_loss=Z best_ckpt=<kind>:<absolute path>"

# Commit the evaluator result and ordered evaluate event together.
"$PYTHON" <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label <baseline|iter${N}> \
    --stage evaluate \
    --metric-result <absolute evaluator result JSON> \
    --best-ckpt <absolute selected checkpoint> \
    --inference-csv <absolute inference CSV> \
    --training-spec <absolute exact training spec> \
    --threshold <float; include when required by the evaluator contract> \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "Evaluate: <metric>=X <operator> target; threshold=Y|n/a"
```
