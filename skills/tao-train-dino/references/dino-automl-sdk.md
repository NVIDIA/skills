# DINO AutoML And SDK Internals

AutoML/HPO notes, SDK orchestration internals, and parent-model inference details.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Optional: SDK orchestration internals
- Internal Details
- Spec Param / Parent Model Inference

## AutoML / HPO Notes

AutoML runs training — all requirements from **Training Requirements** above apply. The agent must read that section first.

For no-input local DINO AutoML smoke runs, use `DINO_AUTOML_PROFILE` from
**Training Requirements**. Do not inspect previous AutoML runs to infer dataset
URIs, `num_classes`, recommendation count, or interval settings.

**Recommended AutoML metric:** use explicit `metric="mAP50"` with
`direction="maximize"` and pass a custom `metric_extractor` that reads
`Validation mAP50`. Do not rely on `metric="kpi"` for generated DINO runners
unless you have verified the local resolver maps it to mAP50; loose fallback
parsing can otherwise optimize `val_loss`.

```python
import re

def extract_dino_map50(logs, metric_name):
    matches = re.findall(
        r"Validation mAP50\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        logs,
    )
    return float(matches[-1]) if matches else None

runner.run(
    ...,
    automl_settings={"metric": "mAP50", "direction": "maximize", ...},
    metric_extractor=extract_dino_map50,
)
```

**Recommended hyperparameters:**

```python
automl_hyperparameters=[
    "train.optim.lr",
    "train.optim.weight_decay",
    "model.backbone",
    "model.num_queries",
    "model.dropout_ratio",
]
custom_param_ranges={
    "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
    "model.backbone": {
        "valid_options": ["resnet_50", "resnet_34"],
        "option_weights": [0.75, 0.25],
    },
    "model.num_queries": {"valid_min": 100, "valid_max": 900},
    "model.dropout_ratio": {"valid_min": 0.0, "valid_max": 0.3},
}
```

`train.optim.weight_decay` is not in the default DINO spec schema — the runner accepts it with a warning. It still works; the DINO training code picks it up from the config.

**Backbone constraint for AutoML:** The LLM brain may propose backbone names not
in the supported list (see Important Parameters above), especially legacy names
from older DINO docs. Use `custom_param_ranges` to constrain categorical params
when possible.

## Optional: SDK orchestration internals

The following details are only relevant when running DINO via the TAO SDK
(`script_runner` orchestration, S3 I/O wrapping, AutoML). Skills consumed by
the SDK read `references/skill_info.yaml` for these mappings. Skip this
section if running locally with `docker run`.

### Internal Details

#### Spec templates

DINO packages `references/spec_template_<action>.yaml` for the advertised
parent model actions. Use those templates directly and apply the required
dataset/checkpoint overrides from this file. TensorRT templates for the deploy
workflow use the `spec_template_deploy_*.yaml` names.

#### Data Sources Gap

DINO's `config.json` has `"data_sources": {}` (empty). The runner's `_apply_data_sources()` only handles flat spec keys (like cosmos-rl's `custom.train_dataset.annotation_path`), but DINO's data sources are **arrays of objects** (`dataset.train_data_sources[{image_dir, json_file}]`). The tao-core microservices config (`tao-core/nvidia_tao_core/microservices/handlers/network_configs/dino.config.json`) has the full mapping using a `mapping` sub-structure, but the runner doesn't support that format.

**Consequence:** The runner cannot auto-resolve data URIs for DINO. Data paths MUST be set manually via `spec_overrides` (see Training Requirements above). The skill's `config.json` instead declares `inputs` in the train action with `[0]`-indexed spec keys so the SDK's script_runner downloads S3 data at runtime:

```json
"inputs": {
    "dataset.train_data_sources[0].image_dir": {"type": "file"},
    "dataset.train_data_sources[0].json_file": {"type": "file"},
    "dataset.val_data_sources[0].image_dir": {"type": "file"},
    "dataset.val_data_sources[0].json_file": {"type": "file"}
}
```

The skill also declares evaluate inputs so generated eval runners do not need
to patch `script_runner` by hand:

```json
"inputs": {
    "evaluate.checkpoint": {"type": "file"},
    "dataset.test_data_sources.image_dir": {"type": "file"},
    "dataset.test_data_sources.json_file": {"type": "file"}
}
```

This model MD is the source of truth for DINO checkpoint inference:

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

All model-specific metadata (dataset type, formats, metrics, required datasets) is documented in the **Training Requirements** section above.

**Current behavior:** Provide DINO data-source paths explicitly until the launcher advertises first-class support for the tao-core `mapping` sub-structure. Do not rely on automatic data-source mapping for DINO.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `dino.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| distill | `distill.pretrained_teacher_model_path` | `parent_model` | model file inferred from the parent job results folder |
| distill | `encryption_key` | `key` | encryption key |
| distill | `results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

TensorRT mappings (`gen_trt_engine.onnx_file`, `evaluate.trt_engine`, and
`inference.trt_engine`) live in `deploy/skill_info.yaml` because TensorRT runs
through the DINO deploy workflow, not the PyT model skill.
