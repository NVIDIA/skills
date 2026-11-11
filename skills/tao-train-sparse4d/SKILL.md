---
name: tao-train-sparse4d
description: Prepare calibrated multi-camera data and train, evaluate, run inference, export, or quantize TAO Sparse4D 3D detection and tracking models. Includes 2D-to-3D geometric distillation with released LTT weights. Use for Sparse4D adaptation and temporal multi-camera perception workflows.
license: Apache-2.0
compatibility: Requires a compatible TAO runtime; Docker execution requires docker and nvidia-container-toolkit.
metadata:
  version: "0.2.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- temporal
- 3d
- detection
- tracking
---

# Sparse4D

> **Standalone install?** If the TAO skill bank plugin did not initialize the session, run `tao-setup` for host preflight, credentials, and cross-skill discovery.

Sparse4D predicts 3D boxes and tracking IDs from synchronized, calibrated multi-camera images. Start with the selected model's experiment specification and checkpoint. The [canonical TAO documentation](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/sparse4d/sparse4d.html) covers the model workflow.

## Instructions

1. Identify the requested action, data supervision, selected checkpoint, and platform. Read [the action contract](references/skill_info.yaml) and the selected platform skill. Preserve the user's platform, model, data, and training-budget choices.
2. Match the backbone, ordered classes, image preprocessing, anchors, and temporal dimensions to the selected checkpoint. Follow **Backbone and checkpoint selection** below before preparing the specification.
3. Use existing compatible info PKLs when supplied. For raw data, follow [local conversion and artifact checks](references/local_docker_conversion.md). For calibrated real scenes with 2D targets, read [geometric distillation](references/geometric_distillation.md); that route does not require 3D annotations or conversion-generated anchors.
4. Start from `references/spec_template_<action>.yaml` and the selected release's experiment specification. Resolve paths against actual staged artifacts. Bind `results_dir` and any action-specific `results_dir` to the current job output directory; released specifications can contain old output paths. Write nested YAML, not flat dotted keys; do not fabricate train/val/test files or scene suffixes. Check the installed runtime exposes the fields used by the requested workflow.
5. Resolve the model-level image for model actions and the **action-level Data Services image** for `dataset_convert`. For a requested TAO release, use the matching skill-bank release branch and its image pins; do not substitute the default branch's release. Follow `tao-launch-workflow` and the chosen platform's `submit` / `status` / `logs` / `cancel` contract, retaining its job record and explicit output directory. Existing user authorization applies; request missing launch approval only for work not already authorized.
6. Verify the produced artifacts and model metrics. A successful process or short training run is a smoke result, not evidence of improved 3D accuracy. Do not report unpublished experimental fixes or another run's results as validation of the selected image.

## Backbone and checkpoint selection

- Respect an explicit ResNet-50 or ResNet-101 choice. For an existing Sparse4D checkpoint, use the accompanying experiment specification and release metadata to determine `model.backbone.type` (`resnet_50` or `resnet_101`). If they conflict with the requested backbone, resolve that mismatch before launch; do not change the architecture to make a checkpoint load partially.
- If no checkpoint/backbone has been selected, ask which compatible released model to use. The schema default `resnet_101` is a configuration default, not a recommendation or evidence about the user's checkpoint.
- Preserve the selected checkpoint's class order, anchors, input dimensions, and temporal/query dimensions. A filename alone is insufficient when its provenance is unclear.
- The LTT MLP is a separate geometry adapter. Downloading it from the ResNet-50 bundle does **not** select the Sparse4D backbone. Its ordered taxonomy must match the training data and model.
- Fine-tuning uses `train.pretrained_model_path`. Continuing a run uses `train.resume_training_checkpoint_path` and a compatible saved training state. Random initialization with an empty pretrained path is suitable only when explicitly requested, for example a wiring smoke test.

## Train action policy

Read `references/skill_info.yaml` before train requests. This model is AutoML-enabled. Preserve an explicit `automl_policy: on` / `off`; requests for plain training, no HPO, a fixed-specification reproduction, a bounded wiring smoke test, or an exact continuation mean `off` for that run. Otherwise default to `on` and route through `tao-skill-bank:tao-run-automl` when both `schemas/train.schema.json` and `references/spec_template_train.yaml` are available. Preserve dataset, checkpoint, specification, GPU/platform, and budget overrides. If the schema/template is missing, report that AutoML is unavailable and use direct training within the user's request.

AutoML's monitoring name is `val_mAP`; Sparse4D emits `img_bbox_NuScenes/mAP` and `mAP` as aliases. A promoted resume job may carry the source rung's metric without a new evaluation: report its provenance and verify the resumed checkpoint independently. Follow the AutoML skill's baseline-evaluation requirement. Evaluate, inference, export, and quantize remain in this skill.

## Training Requirements

Training needs converted OVPKL data, held-out labeled validation data, matching anchors, and the selected pretrained checkpoint or resume state. Preserve the selected backbone and ordered taxonomy. Geometric distillation additionally needs the released LTT checkpoint and prepared supervision caches. Choose the GPU count and training budget from the user's request and available resources.

### Per-Action Dataset Requirements

The model uses OVPKL annotations. The action contract declares staged inputs, commands, outputs, and optional artifacts. Paths embedded in info PKLs, split lists, and lazy indexes must also remain visible inside the selected runtime; staging a split file alone does not stage its referenced data.

| Action | Runtime command | Required data / model artifacts |
|---|---|---|
| `dataset_convert` | `annotations convert -e SPEC` (Data Services) | Raw scene root, calibration, images; 3D annotations for labeled conversion only |
| `train` | `sparse4d train -e SPEC` | Training and held-out validation PKLs/splits, image root, matching anchors, selected pretrained or resume checkpoint; optional LTT/cache artifacts for the chosen supervision route |
| `evaluate` | `sparse4d evaluate -e SPEC` | `evaluate.checkpoint`, matching anchors, labeled held-out PKLs at **`dataset.test_dataset.ann_file`**, image root |
| `inference` | `sparse4d inference -e SPEC` | `inference.checkpoint`, matching anchors, input PKLs at `dataset.test_dataset.ann_file`, image root |
| `export` | `sparse4d export -e SPEC` | `export.checkpoint`, matching anchors and architecture, explicit `export.onnx_file`; no training datasets or external LTT checkpoint |
| `quantize` | `sparse4d quantize -e SPEC` | `quantize.model_path` and the selected backend's calibration inputs; check runtime backend support |

The evaluation CLI calls the test dataloader. Setting only `dataset.val_dataset.ann_file` does not select its evaluation data; that field is used during training validation. Keep the unused dataset sections from the complete specification without inventing or staging unused annotation files.

Resolve `model.head.instance_bank.anchor` as an explicit model artifact from the selected bundle or a compatible labeled conversion. It is not inferred from a mandatory training dataset for evaluation or export.

Use [checkpoint and output resolution](references/spec_param_inference.md) for exact artifact handoff. Do not choose the first checkpoint found or rely on a removed SDK job resolver.

### Typical Spec Overrides

For AutoML, construct `spec_overrides` from concrete staged paths in the table above and `skill_info.yaml`'s `data_sources`; dotted keys are an AutoML override map, not the container YAML format. Set only the inputs needed by the selected action.

| Scope | Overrides to resolve |
|---|---|
| Training data | `dataset.data_root`, `dataset.train_dataset.ann_file`, `dataset.val_dataset.ann_file` |
| Model initialization | `model.backbone.type`, `model.head.instance_bank.anchor`, `dataset.classes`, and `train.pretrained_model_path` or `train.resume_training_checkpoint_path` |
| Training resources | `train.num_gpus`, `train.num_epochs`, `dataset.batch_size`, and `dataset.num_workers`, preserving the requested budget |
| Geometric distillation | The LTT/cache paths and route fields in [the training fragment](references/geometric_distillation_train.yaml); preserve their validated class order and supervision policy during HPO |
| Evaluation and inference | `dataset.data_root`, `dataset.test_dataset.ann_file`, matching anchors, and the action's checkpoint |
| Output | `results_dir` and the action-specific output paths bound to the current job |

With `dataset.lazy_load: true`, stage the split's parent directory as a folder, including `<split>_lazy_index.pkl` beside the split and `_pkl_cam_counts.pkl` when used. Point `ann_file` into that staged directory; do not let file-only staging relocate it away from its index. Keep referenced PKLs and images visible, and rebuild the index after relocation. `dataset.pkl_cam_counts_path` is an optional camera-count override, not the lazy index.

## Configuration and resource checks

Generated schemas and templates describe the packaged contract; `schemas/manifest.json` lists actions. Do not require a TAO Core checkout on the user's machine. Use the runtime's matching release specification for launch: a schema that knows a new field does not make an older image support it.

- `model.backbone.type`: match the selected checkpoint.
- `train.optim.lr`, `train.optim.lr_scheduler`, and `train.optim.grad_clip`: optimizer settings are nested under `optim`.
- `dataset.batch_size`: per-GPU batch size. `dataset.max_cameras` can reduce sampled training cameras when supported; validate the resulting supervision coverage.
- `train.precision`: `bf16`, `fp16`, or `fp32`, subject to hardware/runtime support. BF16 is not mandatory.
- `train.num_gpus`, `train.gpu_ids`, `train.num_nodes`: use the selected platform's allocation. One-GPU smoke training is supported; memory needs depend on the backbone, resolution, cameras, batch size, and temporal configuration.
- `dataset.num_frames` and `dataset.num_bev_groups` are scheduling/data-size settings, not an activation-memory sequence-length knob. Inspect actual dataloader length and iteration counts before comparing training budgets across datasets or GPU counts.
- `dataset.eval_dist_fcn` and `dataset.eval_hota`: select supported detection/tracking evaluation. Keep thresholds, class order, dataset, and raw-versus-EMA checkpoint choice identical for comparisons.

Lightning manages the local workers. Do not add a second distributed launcher. Use the platform's multi-node environment contract; TAO's `WORLD_SIZE` launcher variable denotes node count, not the global GPU count. Mixed 3D/2D distributed training additionally needs the synchronized route settings in the geometric-distillation reference.

## Error patterns

- **Missing annotation or image:** inspect the converter outputs and embedded paths. Preserve the emitted BEV-group suffix, camera IDs, frame IDs, and mount roots. A conversion of `train` does not also create `val` and `test` outputs.
- **Missing anchor:** reuse the selected pretrained bundle's matching anchors, or use labeled conversion's generated anchors when compatible. Annotation-free conversion intentionally skips anchor initialization.
- **Legacy HDF5 depth tuple:** inspect the affected labeled PKL and use the optional normalization helper in the conversion reference only when its documented layout matches. Unlabeled scenes have no depth paths to normalize.
- **Temporal OOM:** first reduce per-GPU batch size or supported camera sampling; preserve checkpoint architecture. Do not blindly shrink anchor/temporal dimensions or treat `num_frames` as a memory fix.
- **LTT or teacher cache mismatch:** verify ordered classes, scene/camera/frame joins, and container paths. The released LTT route supports its original seven classes; changing class names does not adapt the weights.
- **Export reshape:** check the exporter contract in the conversion reference before training a reduced smoke model. Current export uses 20 dummy cameras and 600 cached temporal instances; successful inference alone does not establish export compatibility.
- **Unsupported quantize backend/image:** report the selected image, backend, exact checkpoint, and failure. Do not claim a result from a historical validation image proves the current runtime works, or remove the action to hide the failure.
