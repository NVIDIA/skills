---
name: tao-train-ocrnet
description: OCRNet for scene text recognition. Recognizes text content from cropped text-region images and supports CTC
  and attention-based decoders. Use when training, evaluating, exporting, pruning, quantizing, retraining, or running
  inference for a TAO OCRNet model. Trigger phrases include "train OCRNet", "scene text recognition", "OCR cropped text",
  "CTC / attention text decoder".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- text
- recognition
---

# OCRNet

OCRNet for scene text recognition. Recognizes text content from cropped text region images. Supports CTC and attention-based decoders.

Set train.pretrained_model_path for pretrained OCR weights.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-ocrnet.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** ocrnet
- **Formats:** default
- **Monitoring metric:** val_acc_1

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert | dataset_convert.input_img_dir | train_datasets or eval_dataset | extracted folder containing cropped text images | No |
| dataset_convert | dataset_convert.gt_file | train_datasets or eval_dataset | train/gt_new.txt or test/gt_new.txt | No |
| evaluate | dataset.character_list_file | eval_dataset | character_list | No |
| evaluate | evaluate.test_dataset_dir | eval_dataset | extracted test image folder | No |
| evaluate | evaluate.test_dataset_gt_file | eval_dataset | test/gt_new.txt | No |
| evaluate | evaluate.checkpoint | parent train/AutoML job | best_accuracy.pth or exact requested epoch checkpoint | No |
| export | dataset.character_list_file | eval_dataset | character_list | No |
| export | export.checkpoint | parent train/AutoML job | best_accuracy.pth or exact requested epoch checkpoint | No |
| deploy/gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | extracted calibration image folder for INT8 calibration | Yes |
| deploy/gen_trt_engine | gen_trt_engine.onnx_file | parent export job | exported .onnx artifact | No |
| deploy/gen_trt_engine | dataset.character_list_file | eval_dataset | character_list | No |
| inference | dataset.character_list_file | eval_dataset | character_list | No |
| inference | inference.inference_dataset_dir | inference_dataset | extracted inference image folder | No |
| inference | inference.checkpoint | parent train/AutoML job | best_accuracy.pth or exact requested epoch checkpoint | No |
| prune | dataset.character_list_file | eval_dataset | character_list | No |
| prune | prune.checkpoint | parent train/AutoML job | best_accuracy.pth or exact requested epoch checkpoint | No |
| quantize | dataset.train_dataset_dir | dataset_convert train job | LMDB folder containing data.mdb and lock.mdb | Yes |
| quantize | dataset.val_dataset_dir | dataset_convert eval job | LMDB folder containing data.mdb and lock.mdb | No |
| quantize | dataset.character_list_file | eval_dataset | character_list | No |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | extracted calibration image folder | No |
| quantize | quantize.model_path | parent train/AutoML job | checkpoint selected by resolver | No |
| retrain | dataset.train_dataset_dir | dataset_convert train job | LMDB folder containing data.mdb and lock.mdb | Yes |
| retrain | dataset.val_dataset_dir | dataset_convert eval job | LMDB folder containing data.mdb and lock.mdb | No |
| retrain | dataset.character_list_file | eval_dataset | character_list | No |
| retrain | model.pruned_graph_path | parent prune job | pruned .pth artifact | No |
| train | dataset.train_dataset_dir | dataset_convert train job | LMDB folder containing data.mdb and lock.mdb | Yes |
| train | dataset.train_gt_file | train_datasets | train/gt_new.txt when using raw folders instead of LMDB | No |
| train | dataset.val_dataset_dir | dataset_convert eval job | LMDB folder containing data.mdb and lock.mdb | No |
| train | dataset.val_gt_file | eval_dataset | test/gt_new.txt when using raw folders instead of LMDB | No |
| train | dataset.character_list_file | eval_dataset | character_list | No |

### Checkpoint Selection

OCRNet training writes both `best_accuracy.pth` and epoch-step checkpoints such as `model_epoch_000_step_00003.pth`. Use the SDK/model checkpoint resolver through the `spec_params` mappings in `references/skill_info.yaml`; do not guess by sorting for the newest `.pth`.

- Use `best_accuracy.pth` for best-checkpoint `evaluate`, `inference`, `export`, and `prune` requests.
- Use the exact requested `model_epoch_*_step_*.pth` for epoch/step-specific actions.
- Use `train.resume_training_checkpoint_path` only for resume training, and use `model.pruned_graph_path` for retrain from a prune output. OCRNet does not expose a separate `ocrnet retrain` CLI subtask in the PyT image; the model-skill `retrain` action routes through `ocrnet train -e` with the pruned graph path set.
- OCRNet `quantize` loads the model through PyTorch. For trusted checkpoints created by the same local run, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` if PyTorch 2.6+ rejects the checkpoint as a weights-only load.

### Typical Spec Overrides

Data source overrides are **mandatory for every action**. Run `dataset_convert` separately for train and validation splits, then pass the LMDB folders that directly contain `data.mdb` and `lock.mdb` into train, quantize, and retrain. Tarballs from remote storage must be extracted before they are used as image directories.

```python
TRAIN_IMAGES = "<extracted train image folder>"
TRAIN_GT = "<train gt_new.txt>"
EVAL_IMAGES = "<extracted eval image folder>"
EVAL_GT = "<eval gt_new.txt>"
TRAIN_LMDB = "<train dataset_convert results_dir>"
EVAL_LMDB = "<eval dataset_convert results_dir>"
CHAR_LIST = "<character_list>"
```

**dataset_convert (run once per split):**
```python
{
    "dataset_convert.input_img_dir": TRAIN_IMAGES,
    "dataset_convert.gt_file": TRAIN_GT,
}
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.batch_size": 16,
    "dataset.train_dataset_dir": [TRAIN_LMDB],
    "dataset.val_dataset_dir": EVAL_LMDB,
    "dataset.train_gt_file": "",
    "dataset.val_gt_file": "",
    "dataset.character_list_file": CHAR_LIST,
}
```

**deploy/gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.onnx_file": "<selected export ONNX>",
    "gen_trt_engine.trt_engine": "<output engine path>",
    "gen_trt_engine.tensorrt.calibration.cal_cache_file": "<output calibration cache path>",
    "gen_trt_engine.tensorrt.data_type": "fp16",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [TRAIN_IMAGES],
    "dataset.character_list_file": CHAR_LIST,
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.character_list_file": CHAR_LIST,
    "evaluate.test_dataset_dir": EVAL_IMAGES,
    "evaluate.test_dataset_gt_file": EVAL_GT,
}
```

**export (mandatory data sources):**
```python
{
    "export.checkpoint": "<selected train/AutoML checkpoint>",
    "export.onnx_file": "<output ONNX path>",
    "dataset.character_list_file": CHAR_LIST,
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.character_list_file": CHAR_LIST,
    "inference.inference_dataset_dir": EVAL_IMAGES,
}
```

**prune (mandatory data sources):**
```python
{
    "prune.checkpoint": "<selected train/AutoML checkpoint>",
    "prune.pruned_file": "<output pruned PTH path>",
    "dataset.character_list_file": CHAR_LIST,
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset_dir": [TRAIN_LMDB],
    "dataset.val_dataset_dir": EVAL_LMDB,
    "dataset.character_list_file": CHAR_LIST,
    "dataset.quant_calibration_dataset.images_dir": TRAIN_IMAGES,
    "quantize.model_path": "<selected train/AutoML checkpoint>",
}
```

**retrain (mandatory data sources):**
```python
{
    "dataset.train_dataset_dir": [TRAIN_LMDB],
    "dataset.val_dataset_dir": EVAL_LMDB,
    "dataset.character_list_file": CHAR_LIST,
    "model.pruned_graph_path": "<selected prune output>",
}
```
## Eval Dataset

Optional. Test data provided as separate tarball.

## Important Parameters

- **dataset.character_list_file**: Path to character list defining the supported character set. This determines the output vocabulary size.
- **model.backbone**: Default ResNet.
- **model.prediction**: Decoder type. CTC or Attn (attention-based).
- **train.optim.lr**: Learning rate. Default 1.0 (Adadelta optimizer). High default is specific to Adadelta.
- **dataset.batch_size**: Per-GPU batch size. Default 16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.distributed_strategy` | Strategy name | `auto` |

- Strategy: `auto` for single-GPU, reads `train.distributed_strategy` from config when multi-GPU
- No explicit `num_nodes` in train script — single-node oriented
- Lightweight model, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. OCR text recognition is lightweight. Single GPU is typically sufficient.

## Error Patterns

**dataset_convert required**: If using raw images + gt files, run dataset_convert first to produce LMDB format.

**dataset_convert output folder**: Direct `ocrnet dataset_convert` writes `data.mdb` and `lock.mdb` directly under `dataset_convert.results_dir`. Use that folder itself for `dataset.train_dataset_dir`, `dataset.val_dataset_dir`, quantize, and retrain inputs. SDK-backed runs may wrap the same LMDB folder inside job artifact directories; resolve the actual folder containing `data.mdb` and `lock.mdb`.

**GT file BOM**: Some text-recognition GT files can start with a UTF-8 BOM on the first filename. If dataset conversion logs a missing path with an invisible prefix before the first image name, strip the BOM from a local copy of the GT file before conversion or evaluation.

**Character list mismatch**: All characters in training data must be present in the character_list file.

**Export/prune output fields required**: `export.onnx_file` and `prune.pruned_file` must be writable output paths. These are declared in `references/skill_info.yaml` so SDK-backed model runs can create the paths automatically.

**TensorRT lives in deploy**: The PyT OCRNet CLI exposes `dataset_convert`, `evaluate`, `export`, `inference`, `prune`, `quantize`, and `train`, but not `gen_trt_engine`. Use `references/tao-deploy-ocrnet.md` and `deploy/skill_info.yaml` for TensorRT engine generation and TensorRT-backed evaluate/inference.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `ocrnet.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| dataset_convert | `results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `model.pruned_graph_path` | `pruned_model` | parent pruned model |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| deploy/gen_trt_engine | `encryption_key` | `key` | encryption key |
| deploy/gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | ONNX file inferred from the parent export job results folder |
| deploy/gen_trt_engine | `gen_trt_engine.tensorrt.calibration.cal_cache_file` | `create_cal_cache` | calibration cache path |
| deploy/gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| deploy/gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `model.pruned_graph_path` | `pruned_model` | parent pruned model |
| inference | `results_dir` | `output_dir` | current job results directory |
| prune | `encryption_key` | `key` | encryption key |
| prune | `prune.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| prune | `prune.pruned_file` | `create_pth_file` | output PTH path |
| prune | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| retrain | `encryption_key` | `key` | encryption key |
| retrain | `model.pruned_graph_path` | `parent_model` | model file inferred from the parent job results folder |
| retrain | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-ocrnet](references/tao-deploy-ocrnet.md)
