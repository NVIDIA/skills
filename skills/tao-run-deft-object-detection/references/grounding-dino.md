# DEFT OD — Train / Inference Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-train-grounding-dino`. Read that skill's `SKILL.md` for the full spec reference. This file documents only the loop-specific paths, arguments, and commit contract.

## `results_dir` appends the task name

TAO's `update_results_dir` appends the task name to whatever `results_dir` you pass:

| Pass | Train writes | Inference writes |
|---|---|---|
| `results_dir=${RESULTS_DIR}/iter${N}` | `${RESULTS_DIR}/iter${N}/train/` | `${RESULTS_DIR}/iter${N}/inference/` |

Never append `/train` or `/inference` yourself — doing so produces `iter${N}/train/train/`.

## The checkpoint and the container are version-coupled

A Grounding DINO checkpoint only loads into a TAO image whose model definition matches the
architecture it was **trained** with. There is no "just use the pinned image" answer: confirm
the pairing in Pre-Flight, because the failure lands after the container has started and
looks like a checkpoint problem rather than a config one.

### `class_embed_bias` must match the checkpoint

`ContrastiveEmbed` (the `class_embed` head) takes a `bias` argument that defaults to
**`False`**, sourced from the spec as `model.class_embed_bias`. A checkpoint trained with
`class_embed_bias: True` carries 13 extra tensors, and omitting the field at inference fails
with:

```
RuntimeError: Error(s) in loading state_dict for GDINOPlModel:
  Unexpected key(s) in state_dict:
    model.model.transformer.decoder.class_embed.{0..5}.bias
    model.model.transformer.enc_out_class_embed.bias
    model.model.class_embed.{0..5}.bias
```

**Only `class_embed.*.bias` keys unexpected, and nothing else, means exactly this** — set
`model.class_embed_bias: True` in the inference spec (and any spec that loads that
checkpoint). It is not a corrupt checkpoint, not a backbone mismatch, and **not a
driver/CUDA problem**: `load_state_dict` compares parameter names in pure Python before any
kernel runs, so drivers cannot add or remove a `bias`. Genuine CUDA faults look different —
`no kernel image is available`, cuDNN errors, device-side asserts.

The field is easy to lose because it is absent from the shipped `infer.yaml` template and
defaults to the value the checkpoint does *not* use.

### `log_scale` must be set, and a wrong value fails silently

`ContrastiveEmbed.forward` scales the visual·text similarity before the sigmoid:

```python
res = visual_feat @ text_feat.transpose(-1, -2)
if isinstance(self.log_scale, nn.Parameter): res = res * self.log_scale.exp()
elif self.log_scale == 'auto':               res = res / math.sqrt(visual_feat.shape[-1])
```

With `model.log_scale: null` **neither branch fires and there is no scaling at all**. Raw
dot products of 256-dim features reach the sigmoid unscaled and saturate: every detection
scores `1.000`, `conf_threshold` filters nothing, and every one of `num_select` slots is
written for every image.

Unlike `class_embed_bias`, this **does not error**: the run exits 0 with a full set of label
files. `null` is the schema default and is absent from the shipped `infer.yaml`, so it is the
out-of-the-box failure.

**Symptom:** exactly `num_select` boxes per image, every score at ~1.000. The boxes are fine;
only the confidences are meaningless. Set `model.log_scale: auto`, or the learnable float the
checkpoint trained with — a checkpoint carrying no `log_scale` tensor trained with `auto`.

Left unnoticed this poisons the whole loop: gap analysis sees overwhelming FP, precision
collapses to ~0 on every image, and *every* image is flagged weak — so mining gets no signal
while the underlying detector is working.

### Confirming a pairing

Read the architecture straight out of the checkpoint rather than guessing:

```python
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
# backbone: patch_embed.proj.weight (96,3,4,4) -> embed_dim 96 -> swin_tiny_224_1k
# num_queries: query_embed.weight rows
# enc/dec layers: max index in {encoder,decoder}.layers.N.
# class_embed bias present? -> model.class_embed_bias must be True
```

Then set `model.backbone`, `num_queries`, `enc_layers`, `dec_layers`, `num_feature_levels`,
`class_embed_bias`, and `log_scale` to match. A mismatch on any of these surfaces as `Unexpected key(s)`,
`Missing key(s)`, or `size mismatch` at load time.

## NVIDIA ships the authoritative spec with the checkpoint

The NGC download carries `experiment.yaml` alongside the `.pth`. Its model block is the
authority for everything below, including both traps documented above:

```yaml
model:
  backbone: swin_tiny_224_1k
  num_feature_levels: 4
  dec_layers: 6
  enc_layers: 6
  num_queries: 900
  dropout_ratio: 0.0
  dim_feedforward: 2048
  log_scale: auto
  class_embed_bias: True
```

Keep that file next to the checkpoint. When a spec and a checkpoint disagree, this is the
tiebreaker — it is NVIDIA's own configuration for these exact weights, not a value recovered
by inspecting tensors.

## Reference inference spec

Both traps above are settings, not code, and settings that live only in prose are settings
nobody applies. They are therefore held in `assets/overlays/grounding_dino_inference.yaml`
and applied on every run:

```bash
INFER_SPEC="${RESULTS_DIR}/<phase>/infer_grounding_dino.yaml"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$INFER_SPEC" \
  --apply-workflow-defaults <skill_root>/assets/overlays/grounding_dino_inference.yaml \
  --set inference.checkpoint=<checkpoint> \
  --set inference.num_gpus="$NUM_GPUS" \
  --set results_dir="${RESULTS_DIR}/<phase>" \
  --set dataset.infer_data_sources.image_dir="[<config.kpi_images_dir>]" \
  --set dataset.infer_data_sources.captions='["bicycle", "car", "person", "road_sign"]' \
  --set dataset.max_labels=4 \
  --require-no-mandatory-under inference
```

The overlay pins `inference.conf_threshold: 0.0` (keep the full PR curve — gap analysis and
KPI both score these labels), `log_scale: auto`, `class_embed_bias: true`, `batch_size: 8`
and the model geometry. What stays in `--set` is the checkpoint, the paths, the GPU count,
and — deliberately — `captions` and `max_labels`.

**`captions` is never pinned in the overlay**, because it is the label map and belongs to the
run's class set, not to the stage. It must list every class the ground truth contains, not
just the classes being trained: a class present in the ground truth and absent from `captions`
can never be predicted, so every one of its boxes is a false negative *and* its objects
resurface as false positives on whichever caption the model matches instead. `max_labels`
must equal `len(captions)`.

**`captions` order is the label map.** Grounding DINO has no class list; it assigns a detection
to a class by the *position* of the caption token it matched. Reorder the list and every
prediction is relabeled, silently and consistently — the run still exits 0, the box count barely
moves, and only the per-class KPI reveals it. The order must match the one the checkpoint was
trained against; the reference uses alphabetical, which is also what the ODVG labelmap emitted
by the prep stage produces. Do not sort it by class frequency or by the order the user happened
to list their classes in.

Nothing about this is enforced by TAO, so check it before every inference:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/verify_class_contract.py \
  --inference-spec "$INFER_SPEC" \
  --kpi-mapping "$CLASS_MAPPING" \
  --state "${RESULTS_DIR}/deft_state.json" \
  --labelmap "${RESULTS_DIR}/iter${N}/tmm/annotations/labelmap.json"
```

It compares the caption list against the KPI classes, the staged ODVG labelmap
order, `max_labels`, and the run's target classes, and exits non-zero naming both
sides of any disagreement. At baseline there is no staged labelmap yet, so drop `--labelmap`. `--classes` is
not its replacement — it is an independent fourth source and takes the **path** to the
pool's `classes.yaml`, not a class list. Pass it in every phase.

`inference.color_map` in the reference spec is cosmetic — it only tints `images_annotated/`.

## Baseline inference (`iter_0`)

No training at baseline. Score the user-supplied zero-shot / pretrained checkpoint:

```bash
docker run --rm --name "deft_${PHASE}_infer" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  grounding_dino inference -e "$INFER_SPEC" \
  results_dir="${RESULTS_DIR}/baseline" \
  inference.checkpoint="$ZERO_SHOT_CHECKPOINT" \
  inference.num_gpus="$NUM_GPUS"
```

Wait on the labels, the same way training does — `results_dir` gains the action name,
so both the labels and `status.json` sit under `inference/`:

```bash
LAUNCH_MARKER="${RESULTS_DIR}/<phase>/.inference_launched"
mkdir -p "$(dirname "$LAUNCH_MARKER")" && touch "$LAUNCH_MARKER"
# ... launch the container ...

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/await_stage.py \
  --newer-than "$LAUNCH_MARKER" \
  --status-json "${RESULTS_DIR}/<phase>/inference/status.json" \
  --status-contains "finished successfully"
```

Labels land in `${RESULTS_DIR}/baseline/inference/labels/`. The same wait applies to
each iteration's inference with `<phase>` set to `iter${N}`.

Also copy the user's train-spec template to `${RESULTS_DIR}/train_grounding_dino.yaml`. Iteration 1 extends that copy; nothing trains from it at baseline.

## Iteration train

Build the spec first. Every flag below is load-bearing: the script refuses to emit a spec
that cannot train.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_val_split_for_train.py \
  --coco "<pool>/coco.json" --out "${RESULTS_DIR}/val_coco.json"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_spec_for_train.py \
  --previous-spec        "<template or assets/train_grounding_dino.yaml>" \
  --output-spec          "${RESULTS_DIR}/iter${N}/train_grounding_dino.yaml" \
  --tmm-image-dir        "${RESULTS_DIR}/iter${N}/tmm/images" \
  --tmm-odvg-file        "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_odvg.jsonl" \
  --tmm-label-map-file   "${RESULTS_DIR}/iter${N}/tmm/annotations/labelmap.json" \
  --val-image-dir        "<pool images>" \
  --val-json-file        "${RESULTS_DIR}/val_coco.json" \
  --pretrained-model-path "<config.zero_shot_checkpoint>" \
  --num-epochs "$NUM_EPOCHS" --learning-rate "$LEARNING_RATE"
```

| Flag | Why it is not optional |
|---|---|
| `--pretrained-model-path` | Left unset, training starts from **no pretrained weights**, reports success, and emits a model that detects nothing. The failure surfaces only at KPI, a full training run later. |
| `--val-*` | Validation is mandatory and the COCO must be 0-based — see above. |
| `--tmm-label-map-file` | Also the source of `dataset.max_labels`, which must equal the class count. Hardcoding it silently drops classes from captions when the target set changes size. |

Then:

```bash
docker run --rm --name "deft_iter${N}_train" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  grounding_dino train -e "${RESULTS_DIR}/iter${N}/train_grounding_dino.yaml" \
  results_dir="${RESULTS_DIR}/iter${N}" \
  train.num_gpus="$NUM_GPUS"
```

**Wait on the artifact, never on a process name.** `pgrep -f "grounding_dino train"` matches
the waiting shell's own command line, so the wait never ends.

Take a launch marker first, and pass it as `--newer-than`. Without it a retry of a
failed job is satisfied instantly by the checkpoint and success line the previous
attempt left behind:

```bash
LAUNCH_MARKER="${RESULTS_DIR}/iter${N}/train/.launched"
mkdir -p "$(dirname "$LAUNCH_MARKER")" && touch "$LAUNCH_MARKER"
# ... launch training ...

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/await_stage.py \
  --newer-than "$LAUNCH_MARKER" \
  --status-json "${RESULTS_DIR}/iter${N}/train/status.json" \
  --status-contains "finished successfully"
```

`gdino_model_latest.pth` is deliberately not an `--artifact` here: training rewrites it
at every checkpoint interval, so a wait naming it returns at the first interval and the
stage is read as finished several epochs early. `status.json` is the only artifact that
appears once, at the end.

`results_dir` and `train.num_gpus` are Hydra overrides, not flags. Everything else must already be in the spec — do not add further overrides on the command line.

**Never write `automl_policy` or a `workflow:` key into the spec.** TAO's Hydra `ExperimentConfig` schema does not recognize them and the run fails at config-merge time. Plain `docker run … train` is already non-AutoML.

### Every iteration fine-tunes the base checkpoint, not the previous one

`train.pretrained_model_path` stays pointed at the **original base checkpoint** on every
iteration. It is inherited from the spec template and `prepare_spec_for_train.py` never touches it.
That is deliberate, and it is the single most surprising property of this loop:

- What grows across iterations is the **dataset** (`dataset.train_data_sources` gains one
  mined ODVG source per iteration), not the weights.
- The previous iteration's checkpoint is used **only** for inference and the gap analysis
  that follows it — never as a training initialisation.

Do not "improve" this by chaining `pretrained_model_path` to `iter{N-1}`'s checkpoint. That
converts the run into continual fine-tuning, which compounds drift across iterations and
makes any mAP change unattributable — you could no longer tell whether iteration N improved
because the mined data helped or because it inherited N-1's state.

Required output: a newly emitted checkpoint under `${RESULTS_DIR}/iter${N}/train/` (`gdino_model_latest.pth`). A non-zero exit, a TAO status of `FAILURE`, or a run that emits no new checkpoint is a hard stop — never run inference against a checkpoint written before the failure, and never reuse the previous iteration's checkpoint as if it were this iteration's.

## Iteration inference

```bash
docker run --rm --name "deft_${PHASE}_infer" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  grounding_dino inference -e "$INFER_SPEC" \
  results_dir="${RESULTS_DIR}/iter${N}" \
  inference.checkpoint="${RESULTS_DIR}/iter${N}/train/gdino_model_latest.pth" \
  inference.num_gpus="$NUM_GPUS"
```

## Inference label format

TAO writes one KITTI-style `.txt` per image into `inference/labels/`, plus annotated images into `inference/images_annotated/`. Each detection line is 15 fields plus a trailing score:

```
<class_name> 0.00 0 0.00 <x1> <y1> <x2> <y2> 0.00 0.00 0.00 0.00 0.00 0.00 0.00 <score>
```

Boxes are absolute `xyxy`. Detections below `inference.conf_threshold` are **already dropped at write time**, so it bounds what every later stage can see: a downstream threshold composes with this one rather than replacing it, and no stage can recover a box inference did not write.

That is the reason to keep it at `0.0`. The labels then carry the full curve, and gap analysis and KPI are free to pick any threshold independently — including scoring the same labels repeatedly at several thresholds, which costs a re-run of KPI rather than a re-run of inference. A non-zero value here forecloses that permanently and silently compounds with whatever a later stage applies.

## Commit

```bash
# train
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage train \
  --checkpoint "${RESULTS_DIR}/iter${N}/train/gdino_model_latest.pth" \
  --training-spec "${RESULTS_DIR}/iter${N}/train_grounding_dino.yaml" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "trained iter${N}: <epochs> epochs, <N> data sources"

# inference
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "<phase>" --stage inference \
  --inference-labels-dir "${RESULTS_DIR}/<phase>/inference/labels" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "inference: <N> label files"
```

`commit_stage.py` records `inference_labels_dir` under the phase. The next iteration's `gap_analysis` reads that path from state — never hardcode it.
