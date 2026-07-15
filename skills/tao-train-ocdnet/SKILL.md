---
name: tao-train-ocdnet
description: OCDNet for scene text detection. Detects arbitrary-oriented text regions in natural images using a
  differentiable binarization approach. Use when training, evaluating, exporting, pruning, quantizing, retraining, or running
  inference for a TAO OCDNet model. Trigger phrases include "train OCDNet", "scene text detection", "arbitrary-oriented text
  boxes", "differentiable binarization detector".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- text
- detection
---

# OCDNet

OCDNet for scene text detection. Detects arbitrary-oriented text regions in natural images using a differentiable binarization approach.

Set `model.pretrained_model_path` for pretrained weights.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-ocdnet.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

The PyT OCDNet CLI supports `train`, `evaluate`, `export`, `inference`, `prune`, `quantize`, and `default_specs`. It does not expose PyT-side `retrain` or `gen_trt_engine` subcommands. The model skill exposes `retrain` by running `ocdnet train` with `model.load_pruned_graph: true` and `model.pruned_graph_path`. Resume from an epoch checkpoint uses `ocdnet train` plus `train.resume_training_checkpoint_path`. TensorRT engine generation is owned by the deploy workflow.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

For AutoML train, use `train_loss_epoch` or `train_loss` as the optimization
metric with `direction=minimize`. The Lightning progress log emits
`train_loss_epoch`, and TAO `status.json` records the same final value under
`train_loss`. For one-epoch local AutoML smoke runs, set
`train.lr_scheduler.args.warmup_epoch: 0`; leaving warmup equal to the epoch
budget causes the trainer to fail before a recommendation can report a metric.
Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** ocdnet
- **Formats:** default
- **Monitoring metric:** hmean

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Runtime value | List? |
|---|---|---|---|---|
| evaluate | dataset.validate_dataset.data_path | eval_dataset | extracted validation split folder with `img/` and `gt/` | Yes |
| inference | inference.input_folder | inference_dataset or eval_dataset | extracted image folder | No |
| prune | dataset.validate_dataset.data_path | eval_dataset | extracted validation split folder with `img/` and `gt/` | Yes |
| quantize | dataset.train_dataset.data_path | train_datasets | extracted train split folder with `img/` and `gt/` | Yes |
| quantize | dataset.validate_dataset.data_path | eval_dataset | extracted validation split folder with `img/` and `gt/` | Yes |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets or calibration_dataset | extracted calibration image folder | No |
| train | dataset.train_dataset.data_path | train_datasets | extracted train split folder with `img/` and `gt/` | Yes |
| train | dataset.validate_dataset.data_path | eval_dataset | extracted validation split folder with `img/` and `gt/` | Yes |
| retrain | dataset.train_dataset.data_path | train_datasets | extracted train split folder with `img/` and `gt/` | Yes |
| retrain | dataset.validate_dataset.data_path | eval_dataset | extracted validation split folder with `img/` and `gt/` | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`. OCDNet does not unpack dataset archives at runtime. If the source is `train.tar.gz`, `test.tar.gz`, or `img.tar.gz`, extract it first and pass the split folder or image folder into the spec. The split folder must contain `img/` and `gt/`; alternatively, pass a UTF-8 datalist text file whose lines map image paths to label paths.

```python
TRAIN_ROOT = "/path/to/extracted/train"
EVAL_ROOT = "/path/to/extracted/test"
INFER_IMG_DIR = "/path/to/extracted/test/img"
CALIB_IMG_DIR = "/path/to/extracted/train/img"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.train_dataset.loader.batch_size": 16,
    "dataset.train_dataset.data_path": [TRAIN_ROOT],
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "inference.input_folder": INFER_IMG_DIR,
}
```

**prune (mandatory data sources):**
```python
{
    "prune.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
}
```

**quantize (mandatory data sources):**
```python
{
    "quantize.model_path": "<selected train checkpoint or exported ONNX>",
    "dataset.train_dataset.data_path": [TRAIN_ROOT],
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
    "dataset.quant_calibration_dataset.images_dir": CALIB_IMG_DIR,
}
```

**resume training (mandatory data sources):**
```python
{
    "train.resume_training_checkpoint_path": "<exact model_epoch checkpoint>",
    "dataset.train_dataset.data_path": [TRAIN_ROOT],
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
}
```

**retrain from prune output (mandatory data sources):**
```python
{
    "model.load_pruned_graph": True,
    "model.pruned_graph_path": "<selected prune output>",
    "dataset.train_dataset.data_path": [TRAIN_ROOT],
    "dataset.validate_dataset.data_path": [EVAL_ROOT],
}
```

**default_specs:**
```python
{
    "results_dir": "<writable output directory>",
}
```
## Eval Dataset

Optional. Test dataset provided as separate tarball.

## Important Parameters

- **model.backbone**: Default deformable_resnet18. Deformable convolutions improve text region detection for irregular text.
- **train.optimizer.args.lr**: Learning rate. Default 0.001 (Adam).
- **postprocess.thresh**: Binarization threshold for text region extraction.
- **postprocess.box_thresh**: Box confidence threshold for filtering detections.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.distributed_strategy` | `ddp`, `fsdp`, or `deepspeed_stage_3_offload` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces FP16
- **`deepspeed_stage_3_offload`** is uniquely supported for OCDNet (forces FP16)
- FAN backbones auto-enable `sync_batchnorm`

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. OCDNet is lightweight. Single GPU is sufficient for most datasets.

## Error Patterns

**Low detection rate**: Tune postprocess.thresh and box_thresh. Default thresholds may be too aggressive for some datasets.

**One-epoch smoke train with default scheduler**: `train.num_epochs` must not equal `train.lr_scheduler.args.warmup_epoch`. For one-epoch validation, set `warmup_epoch: 0`; for normal starter runs, keep `num_epochs > warmup_epoch`.

**Archive passed as dataset path**: `dataset.*.data_path` is not an archive path for OCDNet. Passing `train.tar.gz` or `test.tar.gz` directly causes the dataloader to open the gzip as a UTF-8 datalist. Extract the archive and pass the split folder containing `img/` and `gt/`, or pass a real UTF-8 datalist file.

**Quantize checkpoint type**: Do not pass `model_best.pth` to the PyTorch quantize path. Some older PyT runtimes wrote `model_best.pth` without full Lightning checkpoint metadata. The default `torchao` quantize path should use the intended full `model_epoch_<epoch>_step_<step>.pth` checkpoint and write `quantized_model_torchao.pth`.

**Default specs output directory**: `ocdnet default_specs` requires a writable `results_dir` override, for example `results_dir=/workspace/run/results/default_specs`.

## Checkpoint Handoff

OCDNet train writes `model_best.pth` plus full Lightning epoch checkpoints such as `model_epoch_001_step_00046.pth`; it may also write `ocd_model_latest.pth` as a latest symlink. Use `model_best.pth` for `evaluate.checkpoint`, `inference.checkpoint`, `export.checkpoint`, and `prune.checkpoint` when the user asks for the best checkpoint. Use a specific `model_epoch_<epoch>_step_<step>.pth` for `train.resume_training_checkpoint_path` and for any action that explicitly needs a full Lightning checkpoint. Prune writes artifacts such as `pruned_<ch_sparsity>.pth`; use the exact pruned `.pth` artifact for `model.pruned_graph_path` when retraining from a pruned graph. Use a latest checkpoint only when the user explicitly asks for latest.

If quantize is retried with a PyTorch backend, resolve the full `model_epoch_<epoch>_step_<step>.pth` that corresponds to the intended best epoch or requested epoch; do not pass `model_best.pth` to the PyTorch quantize path. If quantize is retried with `modelopt.onnx`, pass the exported ONNX as `quantize.model_path` and verify that the runtime image actually contains `modelopt.onnx.quantization`.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Model handoff mappings:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| prune | `prune.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| prune | `results_dir` | `output_dir` | current job results directory |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| retrain from prune | `model.pruned_graph_path` | `parent_model` | exact pruned model file inferred from the parent prune results folder |
| retrain from prune | `results_dir` | `output_dir` | current job results directory |
| train | `model.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-ocdnet](references/tao-deploy-ocdnet.md)
