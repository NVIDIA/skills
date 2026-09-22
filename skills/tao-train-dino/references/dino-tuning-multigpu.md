# DINO Tuning And Multi-GPU Notes

Full AutoML/HPO guidance and multi-GPU spec consistency rules. Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Multi-GPU Spec Consistency

When increasing `train.num_gpus`, also set `train.gpu_ids` to the same visible
device range. For example, an 8-GPU single-node Slurm run must include both
`"train.num_gpus": 8` and `"train.gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]`.
Leaving the template default `train.gpu_ids: [0]` while requesting multiple
GPUs can make distributed startup inconsistent and can surface as NCCL
collective timeouts instead of an immediate validation error.

## AutoML / HPO Notes

AutoML runs training — all requirements from **Training Requirements** above apply. The agent must read that section first.

For no-input local DINO AutoML smoke runs, use `DINO_AUTOML_PROFILE` from
**Training Requirements**. Do not inspect previous AutoML runs to infer dataset
URIs, `num_classes`, recommendation count, or interval settings.

**Recommended AutoML metric:** for training-log-only quick operational checks,
use `metric="val_mAP50"` with `direction="maximize"` and pass a custom metric
extractor that reads `Validation mAP50`. For training-log-only COCO or
paper-style comparisons, use `metric="val_mAP"`. When recommendations are
scored by the standalone evaluate action, use `test_mAP50` or `test_mAP`
respectively. Do not
rely on `metric="kpi"` for generated DINO runners unless you have verified the
local resolver maps it to the intended detection metric; loose fallback parsing
can otherwise optimize `val_loss`.

Use a `metric_extractor` that reads the last `Validation mAP50` value from the
logs, then run training-log-only AutoML with `automl_settings={"metric": "val_mAP50",
"direction": "maximize", ...}`.

When a benchmark run remains below target but the per-epoch `val_mAP` curve is
still climbing at the final epoch, extend the best full-budget configuration
before declaring the search plateaued. For dense datasets such as aerial or
driving-scene detection, also preserve high-resolution input overrides and
structural settings (`model.backbone`, `model.num_queries`,
`model.num_select`, class metadata) when evaluating or resuming the checkpoint.

**Recommended hyperparameters:**

Suggested knobs: `train.optim.lr`, `train.optim.weight_decay`,
`model.backbone`, `model.num_queries`, and `model.dropout_ratio`. Constrain
`model.backbone` to supported names such as `resnet_50` and `resnet_34`; the
LLM brain may otherwise propose legacy or invalid DINO backbone names.

`train.optim.weight_decay` is not in the default DINO spec schema — the runner accepts it with a warning. It still works; the DINO training code picks it up from the config.

All model-specific metadata is documented in the Training Requirements table and
`references/skill_info.yaml`. DINO data-source arrays are not auto-resolved from
TAO Core metadata; provide dataset paths explicitly in the spec overrides.

## Spec Param / Parent Model Inference

Model-specific parent-model mappings belong in this MD file. For
`parent_model`/`parent_model_folder`, pass the upstream train/export/AutoML
child job id as the parent job id; list the parent result folder, filter
checkpoint artifacts, and select the resolved model file or folder.

DINO checkpoint layout:

```text
checkpoint format: pth
checkpoint files: results_dir/train/model_epoch_<epoch>_step_<step>.pth
latest alias: results_dir/train/dino_model_latest.pth
evaluate.checkpoint: parent_model
export.checkpoint: parent_model
inference.checkpoint: parent_model
quantize.model_path: parent_model
distill.pretrained_teacher_model_path: parent_model
```

Full inference-mapping table (per action):

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| distill | `distill.pretrained_teacher_model_path` | `parent_model` | model file inferred from the parent job results folder |
| distill | `encryption_key` | `key` | encryption key |
| distill | `results_dir` | `output_dir` | current job results directory |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `encryption_key` | `key` | encryption key |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `encryption_key` | `key` | encryption key |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `model.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |

TensorRT mappings (`gen_trt_engine.onnx_file`, `evaluate.trt_engine`, and
`inference.trt_engine`) live in `deploy/skill_info.yaml` because TensorRT runs
through the DINO deploy workflow, not the PyT model skill.
