---
name: tao-train-codetr
description: Co-DETR (CoDINO) for object detection. A DETR-family detector with collaborative hybrid
  assignment — auxiliary one-to-many heads supervise the encoder during training, giving strong
  closed-set accuracy at high inference cost. Use when training, evaluating, or running inference for
  a TAO Co-DETR model. Trigger phrases include "train Co-DETR", "run CoDINO", "codetr inference",
  "collaborative DETR", "autolabel detections with Co-DETR".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- object
- detection
---

# Co-DETR

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Co-DETR trains a DETR detector alongside auxiliary one-to-many assignment heads (`num_co_heads`), which supervise the encoder more densely than DETR's one-to-one matching alone. The auxiliary heads exist only during training; inference runs the primary DETR head. Backbones are large by default — `vit_large_codetr` for training, `swin_large_patch4_window7_224` for the reference inference/eval configs — so this is an accuracy-first model, not a latency-first one.

## Availability — probe, then fall back to the module

The `codetr` **console script is not registered** in the TAO PyTorch images checked so far
(`7.0.1-pyt` and the `2026.7.31-rc-12-multiarch` nightly), even though the module itself
ships. So a bare `codetr` invocation fails while the network is perfectly usable.

Probe in two steps and use whichever works:

```bash
# 1. console script (preferred when present)
docker run --rm "$TAO_PYT_IMAGE" codetr --help >/dev/null 2>&1 && CODETR="codetr"

# 2. module fallback — works whenever the package is installed
[ -z "${CODETR:-}" ] && docker run --rm --entrypoint sh "$TAO_PYT_IMAGE" \
  -c 'python3 -c "import nvidia_tao_pytorch.cv.codetr"' >/dev/null 2>&1 \
  && CODETR="python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr"

[ -z "${CODETR:-}" ] && { echo "FATAL: Co-DETR not available in $TAO_PYT_IMAGE"; exit 1; }
```

Both forms take identical arguments — the module entrypoint reports itself as `codetr` and
exposes the same `{train, evaluate, inference, default_specs}` subtasks. Substitute `$CODETR`
wherever this document writes `codetr`.

Only if **both** probes fail does this image genuinely lack Co-DETR. Then stop and ask which
image to use — do not silently substitute another detector.

## Dataclass Schemas

Generated TAO Core schemas are **not yet packaged** for this model, so `schemas/<action>.schema.json` and `references/spec_template_<action>.yaml` are absent except for the hand-written `references/spec_template_inference.yaml`. Read spec keys from the upstream experiment specs in `nvidia_tao_pytorch/cv/codetr/experiment_specs/` (`train.yaml`, `eval.yaml`, `inference.yaml`, `export.yaml`) until a maintainer regenerates them. AutoML is therefore not runnable for this model.

## Train Action Policy

This model is **not** AutoML-enabled (`automl_enabled: false` in `references/skill_info.yaml`). Run train directly; do not route through `tao-skill-bank:tao-run-automl`. Never add an `automl_policy` or `workflow:` key to the spec — TAO's Hydra `ExperimentConfig` schema rejects unknown top-level keys at config-merge time.

## Supported Actions

The packaged Co-DETR CLI exposes `train`, `evaluate`, and `inference`. This skill exposes all three. `export` and TensorRT deploy flows are not available for this model.

Every action follows the standard TAO form, where anything after the spec is a Hydra override:

```bash
$CODETR <action> -e /absolute/path/spec.yaml [key=value ...]
```

`results_dir` **auto-appends the action name**: passing `results_dir=X` writes to `X/train/`, `X/evaluate/`, or `X/inference/`. Never append the subdirectory yourself.

## Training Requirements

- **Dataset type:** object_detection
- **Formats:** coco (train/evaluate), image directory + classmap (inference)
- **Monitoring metric:** mAP50

### Per-Action Dataset Requirements

| Action | Spec Key | Files |
|---|---|---|
| train | `dataset.train_data_sources` | image_dir, json_file (COCO) |
| train | `dataset.val_data_sources` | image_dir, json_file (COCO) |
| evaluate | `dataset.test_data_sources` | image_dir, json_file (COCO) |
| inference | `dataset.infer_data_sources.image_dir` | directory of images |
| inference | `dataset.infer_data_sources.classmap` | newline-separated class list, one name per line |

`dataset.num_classes` is required for every action and must match both the classmap length and the checkpoint head.

### Typical Spec Overrides

```bash
# $CODETR is either `codetr` or the module form — see ## Availability
$CODETR inference -e "$SPEC" \
  inference.checkpoint=/abs/codetr.pth \
  dataset.infer_data_sources.image_dir=/abs/images \
  dataset.infer_data_sources.classmap=/abs/classmap.txt \
  results_dir=/abs/results \
  inference.conf_threshold=0.3 \
  inference.num_gpus=8
```

`references/spec_template_inference.yaml` is a starting point.

## Getting Weights

Two fields control where training starts:

| Field | Holds |
|---|---|
| `model.pretrained_backbone_path` | a backbone-only checkpoint initialising `model.backbone` |
| `train.pretrained_model_path` | a full Co-DETR (or compatible DETR-family) checkpoint to fine-tune from |

**Prefer `train.pretrained_model_path` with the COCO detector below.** It carries a trained
backbone *and* trained detection heads, so it is the better starting point for fine-tuning on a
new class set, and it is the checkpoint this skill is verified against.

A backbone-only checkpoint must match the architecture exactly. `vit_large_codetr` is **ViT-L/16**
— readable from the checkpoint as `backbone.patch_embed.proj.weight (1024, 3, 16, 16)`, a 16x16
patch embedding. A ViT-L/**14** checkpoint such as
`nvidia/tao/pretrained_dinov2_classification_imagenet:vit_large_patch14_dinov2` has a
`(1024, 3, 14, 14)` embedding and cannot load into it; verify the patch size of any candidate
against the target backbone before using it. For the Swin, FAN, ResNet and EfficientViT
backbones, use their own matching checkpoints.

Left `null`, the backbone initialises randomly and needs substantially longer training.

**Inference needs a full detector checkpoint, not a backbone** — `inference.checkpoint` wants trained detection heads, which a backbone does not have. Training emits `model_epoch_<N>.pth` into `train.results_dir`.

### A COCO-trained detector checkpoint

No Co-DETR detector appears under `nvidia/tao` on NGC. The upstream authors publish one on HuggingFace, and it loads into this build directly:

```bash
huggingface-cli download zongzhuofan/co-detr-vit-large-coco pytorch_model.pth --local-dir <dir>
```

`zongzhuofan/co-detr-vit-large-coco` — Apache-2.0, a single 2.8 GB `pytorch_model.pth`. Verified against `7.0.1-pyt`: it is the ViT-Large COCO-80 model, and pairs with

```yaml
model:
  backbone: vit_large_codetr
  num_queries: 1500
  num_feature_levels: 5
  return_interm_indices: [0, 1, 2, 3, 4]
  num_co_heads: 1
dataset:
  num_classes: 80
```

Loading reports `66 missing, 0 unexpected` — the missing keys are the training-only collaborative heads and are expected. Use the COCO-80 classmap with it, and fold to your own classes with `inference.category_mapping` (above) rather than post-processing the labels.

### The classmap file

**A ready-made COCO-80 classmap ships with this skill at `references/coco80_classmap.txt`** — use it directly for any COCO-trained checkpoint. It is extracted from the container's own `METAINFO['classes']` in `nvidia_tao_pytorch/cv/deformable_detr/model/post_process.py`, which is the source of truth; do not retype a COCO list from memory, since the VOC-style names (`aeroplane`, `motorbike`) and the 91-id-with-gaps ordering both look plausible and both mis-map every class silently.

`dataset.infer_data_sources.classmap` is a plain-text file, one class name per line, in `category_id` order starting at 1 (the first foreground class). Its length must equal the number of foreground classes the model predicts — 80 for a COCO-trained checkpoint with `dataset.contiguous_labels: True`. These names are what `inference.color_map` and `inference.category_mapping` refer to.

## Deriving the spec from a checkpoint

`model.*` must match the checkpoint or loading fails. Read the architecture out of the
checkpoint rather than guessing — every field below is recoverable:

```python
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
# backbone.patch_embed.proj.weight (1024,3,16,16) -> ViT-L/16      -> vit_large_codetr
# query_head.transformer.query_embed.weight (1500,256)             -> num_queries 1500
# max index in query_head...{encoder,decoder}.layers.N.            -> enc_layers / dec_layers
# query_head...cls_branches.0.weight (80,256)                      -> dataset.num_classes 80
# roi_head.<i>. indices present                                    -> num_co_heads (count)
# neck.p2..p6                                                      -> num_feature_levels 5
```

A correct load reports **`0 unexpected`**:

```
Missing keys: 66 total (66 expected for collab heads / buffers, 0 unexpected)
```

Missing keys are normal — the collaborative heads are training-time only. *Unexpected* keys are
not: they mean the spec describes a different architecture than the checkpoint holds.

### Verified inference spec — ViT-Large, COCO-80

Confirmed against a 2.8 GB `vit_large_codetr` COCO checkpoint on `7.0.1-pyt`:

```yaml
results_dir: /abs/results
model:
  backbone: vit_large_codetr
  num_queries: 1500
  num_feature_levels: 5
  return_interm_indices: [0, 1, 2, 3, 4]
  two_stage_type: standard
  num_co_heads: 1
  hidden_dim: 256
  nheads: 8
  enc_layers: 6
  dec_layers: 6
  dim_feedforward: 2048
dataset:
  num_classes: 80
  batch_size: 2
  augmentation:
    fixed_padding: true
    fixed_random_crop: 1536        # REQUIRED by vit_large_codetr — see below
    input_mean: [0.485, 0.456, 0.406]
    input_std: [0.229, 0.224, 0.225]
    test_random_resize: 1280
  infer_data_sources:
    image_dir: /abs/images
    classmap: /abs/coco_classmap.txt
inference:
  checkpoint: /abs/codetr_pytorch_model.pth
  conf_threshold: 0.3
  input_width: 640
  input_height: 640
  num_gpus: 2
  category_mapping:
    car: ["car", "bus", "truck"]
```

### The published docs are ahead of the 7.0.1 image

Two fields in the online Co-DETR spec example are **not in this build's schema** and fail the
Hydra merge with `Key '<name>' not in '<Config>'`:

| Field | Status in `7.0.1-pyt` |
|---|---|
| `dataset.contiguous_labels` | absent from `DINODatasetConfig` |
| `dataset.augmentation.pad_size_divisor` | absent from `DINOAugmentationConfig` |

Co-DETR reuses DINO's dataset and augmentation configs, so enumerate the real fields before
trusting an example:

```bash
python3 -c "
from nvidia_tao_pytorch.config.dino.dataset import DINODatasetConfig, DINOAugmentationConfig
import dataclasses
print([f.name for f in dataclasses.fields(DINODatasetConfig)])
print([f.name for f in dataclasses.fields(DINOAugmentationConfig)])"
```

**`vit_large_codetr` hard-requires `dataset.augmentation.fixed_random_crop`** — the ViT backbone
needs a fixed input size, and `build_nn_model.py` raises without it. It is not optional at
inference despite the name suggesting a training-time augmentation.

## Important Parameters

| Parameter | Notes |
|---|---|
| `model.backbone` | `vit_large_codetr` (train default) or `swin_large_patch4_window7_224` (inference/eval default). Must match the checkpoint. |
| `model.num_queries` | 1500 for the ViT config, 900 for Swin. Paired with the backbone — do not mix. |
| `model.num_feature_levels` / `return_interm_indices` | 5 / `[0,1,2,3,4]` for ViT; 4 / `[1,2,3,4]` for Swin. |
| `model.num_co_heads` | Auxiliary collaborative heads. 2 for the ViT train config, 1 for the reference inference config. Training-time only. |
| `model.soft_nms_enabled` | Train config enables soft-NMS (`linear`, IoU 0.8) to match the original Co-DETR test config. |
| `inference.conf_threshold` | Detections below this are dropped **at write time**. TAO default `0.5`. |
| `dataset.num_classes` | Must match classmap length and checkpoint head. |

**Class indexing.** The reference Co-DETR checkpoint uses **0-indexed** class labels — TAO's inference path sets `start_from_one=False`. Verify the offset before mapping predicted class names onto another schema's integer ids.

## Inference Output

With `results_dir=X`:

| Artifact | Location |
|---|---|
| KITTI detection labels | `X/inference/labels/<image_stem>.txt` |
| Annotated images | `X/inference/images_annotated/` |
| Run config + status | `X/inference/experiment.yaml`, `X/inference/status.json` |

Label lines are KITTI-style — 15 fields plus a trailing confidence, boxes absolute `xyxy`:

```
<class_name> 0.00 0 0.00 <x1> <y1> <x2> <y2> 0.00 0.00 0.00 0.00 0.00 0.00 0.00 <score>
```

### Predicted vocabulary

Co-DETR predicts whatever vocabulary its checkpoint was trained on — COCO-80 for the reference checkpoint, in the order given by `dataset.infer_data_sources.classmap`. Emitted labels carry class **names**, not indices.

### Folding to a smaller class set — use `inference.category_mapping`

When a downstream model trains on a smaller set, fold **here**, not afterwards:

```yaml
inference:
  category_mapping:
    bicycle: ["bicycle", "motorcycle"]
    car:     ["car", "bus", "truck"]
    person:  ["person"]
```

Values are **lists**. Semantics, from `model/category_mapping.py`:

| Behaviour | Detail |
|---|---|
| unmapped originals | dropped |
| a name in two groups | keeps the first, logs a warning |
| a name absent from the classmap | warns, ignores — **matching is exact, including case** |
| an empty remap | raises `ValueError` |
| output category ids | `0..K-1` in the order the mapping is written |

Then it runs `apply_category_mapping_groupnms` — per-output-category soft-NMS **after** the merge. This is why folding here beats renaming labels afterwards: one object detected as both `truck` and `car` becomes two boxes of the *same* class the instant those fold together, and only a post-fold NMS removes the duplicate. A downstream rename ships the duplicates onward.

**That dedup needs `model.soft_nms_enabled: True`, which is not the default.** The schema ships it `False`, and with it off the fold renames classes without merging anything — the same result a downstream rename would give, duplicates included. Folding `{vehicle: ["car", "bus", "truck"]}` over 8 traffic images at `conf_threshold: 0.05`:

| `model.soft_nms_enabled` | boxes emitted |
|---|---|
| `False` (schema default) | 277 — exactly `146 car + 102 truck + 29 bus`, nothing merged |
| `True` | 151 — 126 duplicates removed |

Set it whenever `category_mapping` groups classes that the detector confuses with each other, which for COCO vehicles it reliably does. `soft_nms_iou_threshold` (default `0.8`) controls how aggressively the merge happens.

`inference.color_map` keys and the class names in the emitted labels both follow the *output* categories once this is set.

An empty label file is meaningful and is still written: in KITTI it means "this image has no objects".

### Converting labels for downstream training

KITTI is rarely the final format. Both conversions are published in the TAO data-services container via `annotations convert -e <spec>`:

- **KITTI → COCO** — `data.input_format: KITTI`, `data.output_format: COCO`, plus `kitti.{image_dir, label_dir, mapping}`. `kitti.mapping` is a YAML list of single-key dicts whose values are **lists** — `- car: [car]`, never `- car: car`. The converter builds its lookup with `{label: k for k, v in cat_map.items() for label in v}`, iterating the value, so a bare string yields the class names `c`, `a`, `r`: nothing matches, every box is dropped, and the run still prints `Execution status: PASS` and exits 0.
- **COCO → ODVG** (Grounding DINO training) — `data.input_format: COCO`, `data.output_format: ODVG`, plus `coco.ann_file`. Emits `<name>_odvg.jsonl` and `<name>_odvg_labelmap.json`.

There is no direct KITTI → ODVG conversion; chain the two.

## Multi-GPU / Multi-Node

Set `train.num_gpus` / `inference.num_gpus`. `gpu_spec_key` is `train.num_gpus`. Large backbones make this model memory-hungry — reduce `dataset.batch_size` before reducing GPU count when hitting OOM.

## Hardware

Accuracy-first with large backbones and high query counts (900–1500). Expect substantially higher memory and latency than DINO or RT-DETR at equal input size. `train.activation_checkpoint` trades compute for memory when training OOMs.

## Error Patterns

| Symptom | Cause | Fix |
|---|---|---|
| `codetr: command not found` | Console script unregistered — **expected** on current builds; the module usually still ships | Use `python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr`. Only if the import also fails is Co-DETR genuinely absent |
| Nested `inference/inference/` | Appended the action name to `results_dir` | Pass the parent directory only |
| Empty label files across the board | `conf_threshold` above the model's score range | Lower it (reference pipelines use `0.3`) |
| Class names shifted by one | Reference checkpoint is 0-indexed | Check the offset when mapping to another schema |
| Size/shape mismatch loading checkpoint | `model.backbone` does not match the checkpoint | Align backbone, `num_queries`, `num_feature_levels`, `return_interm_indices`, `num_co_heads` |
| `Key 'X' not in 'DINODatasetConfig'` / `'DINOAugmentationConfig'` | Spec uses a documented field absent from this build | Enumerate the real dataclass fields; `contiguous_labels` and `pad_size_divisor` are documented but not in 7.0.1 |
| `vit_large_codetr requires dataset.augmentation.fixed_random_crop` | ViT backbone needs a fixed input size | Set `fixed_random_crop` (1536 for the reference ViT config) — required at inference too |
| Non-zero **unexpected** keys at load | Spec architecture differs from the checkpoint | Re-derive `model.*` from the checkpoint tensors; missing keys alone are normal |
| CUDA OOM in train | Large backbone + high query count | Lower `dataset.batch_size`; enable `train.activation_checkpoint` |
| Paths not found in container | Host/container path mismatch | Mount so paths are identical; confirm images are under the mount, not just the spec |

## Deployment

`export` and TensorRT deploy are not available for Co-DETR in this build. For a deployable detector, train Co-DETR as a teacher and distil into a student that supports export — `tao-train-rtdetr` and `tao-train-dino` both expose `export` and `gen_trt_engine`.

## References

- `references/checkpoint-spec-pairing.md` — deriving `backbone`, `num_queries`, `num_feature_levels` and `num_classes` from a checkpoint's tensors. A mismatch fails silently: PASS, exit 0, empty labels.
