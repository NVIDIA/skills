---
name: tao-train-reid
description: Person re-identification (ReID). Learns discriminative embeddings to match the same person across different
  camera views, based on metric learning. Use when training, evaluating, exporting, or running inference for a TAO person
  re-identification model. Trigger phrases include "train ReID", "person re-identification", "cross-camera person matching",
  "ReID embeddings", "person re-id".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- re
- identification
---

# Re-Identification

Person re-identification. Learns discriminative embeddings to match the same person across different camera views. Metric learning based.

Set model.pretrained_model_path for pretrained weights.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Supported Actions

The packaged Re-Identification PyT CLI supports `train`, `evaluate`, `inference`, `export`, and `default_specs`. This model skill exposes the runnable user actions `train`, `evaluate`, `inference`, and `export`; resume/retrain is performed through `train` with `train.resume_training_checkpoint_path`.

Do not advertise or synthesize `dataset_convert`, `deploy`, `prune`, `quantize`, `gen_trt_engine`, or standalone `retrain` for this model unless the packaged model skill and real CLI add those actions.

## Training Requirements

- **Dataset type:** re_identification
- **Formats:** default
- **Monitoring metric:** cmc_rank_1, maximize

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | evaluate.test_dataset | train_datasets | sample_test.tar.gz | No |
| evaluate | evaluate.query_dataset | train_datasets | sample_query.tar.gz | No |
| inference | inference.test_dataset | train_datasets | sample_test.tar.gz | No |
| inference | inference.query_dataset | train_datasets | sample_query.tar.gz | No |
| train | dataset.train_dataset_dir | train_datasets | sample_train.tar.gz | No |
| train | dataset.test_dataset_dir | train_datasets | sample_test.tar.gz | No |
| train | dataset.query_dataset_dir | train_datasets | sample_query.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
CHECKPOINT = "/results/{train_job_id}/results_dir/model_epoch_000_step_00099.pth"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.num_classes": 100,
    "dataset.num_workers": 4,
    "dataset.batch_size": 16,
    "dataset.num_instances": 4,
    "dataset.train_dataset_dir": f"{S3_TRAIN}/sample_train.tar.gz",
    "dataset.test_dataset_dir": f"{S3_TRAIN}/sample_test.tar.gz",
    "dataset.query_dataset_dir": f"{S3_TRAIN}/sample_query.tar.gz",
}
```

**resume train (mandatory checkpoint):**
```python
{
    "train.num_epochs": 31,
    "train.resume_training_checkpoint_path": CHECKPOINT,
    "dataset.num_classes": 100,
    "dataset.batch_size": 16,
    "dataset.num_instances": 4,
    "dataset.train_dataset_dir": f"{S3_TRAIN}/sample_train.tar.gz",
    "dataset.test_dataset_dir": f"{S3_TRAIN}/sample_test.tar.gz",
    "dataset.query_dataset_dir": f"{S3_TRAIN}/sample_query.tar.gz",
}
```

**evaluate (mandatory data sources and checkpoint):**
```python
{
    "evaluate.test_dataset": f"{S3_TRAIN}/sample_test.tar.gz",
    "evaluate.query_dataset": f"{S3_TRAIN}/sample_query.tar.gz",
    "evaluate.checkpoint": CHECKPOINT,
    "evaluate.output_cmc_curve_plot": "/results/{evaluate_job_id}/results_dir/cmc_curve.png",
    "evaluate.output_sampled_matches_plot": "/results/{evaluate_job_id}/results_dir/sampled_matches.png",
}
```

**export (mandatory checkpoint and output):**
```python
{
    "export.checkpoint": CHECKPOINT,
    "export.onnx_file": "/results/{export_job_id}/results_dir/reid.onnx",
}
```

**inference (mandatory data sources and checkpoint):**
```python
{
    "inference.test_dataset": f"{S3_TRAIN}/sample_test.tar.gz",
    "inference.query_dataset": f"{S3_TRAIN}/sample_query.tar.gz",
    "inference.checkpoint": CHECKPOINT,
    "inference.output_file": "/results/{inference_job_id}/results_dir/reid_inference.json",
}
```

For export and inference, provide explicit file paths for `export.onnx_file` and `inference.output_file`. For evaluate, provide explicit file paths for `evaluate.output_cmc_curve_plot` and `evaluate.output_sampled_matches_plot`. Keep these as spec values or `spec_params` mappings; do not declare them as file outputs in `skill_info.yaml` for local Docker until the runner distinguishes files from folders during output pre-creation.

## Eval Dataset

Required. Evaluation requires test and query datasets for retrieval-based metrics (CMC, mAP).

## Important Parameters

- **dataset.num_classes**: Number of identities. Default 751. Must match the number of unique identities in training data.
- **model.backbone**: Default resnet_50.
- **optim.base_lr**: Base learning rate. Default 3.5e-4.
- **dataset.batch_size**: Per-GPU batch size. Default 64. Re-ID benefits from large batches for better triplet/contrastive sampling.
- **dataset.num_instances**: Number of instances per identity in a batch. Controls sampling strategy for metric learning.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Multi-GPU strategy: `ddp_find_unused_parameters_true`
- `sync_batchnorm` is always enabled
- Precision forced to FP16 (`16-mixed`)
- No explicit `num_nodes` config — single-node oriented

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. Re-ID models are relatively lightweight but benefit from large batch sizes for metric learning.

## Error Patterns

**num_classes mismatch**: Ensure dataset.num_classes equals the number of unique identity folders in the training set.

**Invalid triplet batch shape**: `dataset.batch_size` must be compatible with `dataset.num_instances` so each mini-batch can be reshaped for hard-example mining. For local AutoML smoke runs, keep `dataset.batch_size` fixed to a known valid multiple such as 16 with `dataset.num_instances: 4`, and tune `train.optim.base_lr` instead of unconstrained batch size.

**Query/gallery mismatch**: Query and test (gallery) datasets must share the same identity namespace.

**PyTorch 2.6 checkpoint load failure on checkpoint consumers**: Current Re-ID
checkpoints include OmegaConf containers. For checkpoints produced by the same
trusted TAO train/AutoML workflow, set
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in downstream resume, evaluate, inference,
and export job env vars so Lightning/PyTorch can load the full checkpoint. Do
not use this env var for untrusted checkpoints.

**AutoML metric extraction**: Re-ID train status files report retrieval KPIs such as `cmc_rank_1`, `cmc_rank_5`, `cmc_rank_10`, and `mAP`, plus train loss. Default AutoML train launches must optimize `cmc_rank_1` with `direction: maximize`; do not use `val_loss` as the metric for this model.

**Checkpoint handoff**: Use the checkpoint resolver on the best AutoML child job's `results_dir/train/` folder and select the action-appropriate `model_epoch_*.pth` checkpoint. Re-ID also writes `reid_model_latest.pth`, but that is a latest symlink and should only be used when a caller explicitly requests latest. Preserve the same dataset identity count and query/gallery archives for downstream actions.

**Default spec generation**: The packaged `default_specs` CLI action does not
consume the normal `-e <spec.yaml>` experiment file for `results_dir`. Invoke it
with a Hydra override such as
`re_identification default_specs results_dir=/workspace/run/results/default_specs`.
Passing only `-e` leaves `cfg.results_dir` unset and fails with
`MissingMandatoryValue: results_dir`.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `re_identification.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.output_cmc_curve_plot` | `create_evaluate_cmc_plot_reid` | ReID CMC plot path |
| evaluate | `evaluate.output_sampled_matches_plot` | `create_evaluate_matches_plot_reid` | ReID sampled matches plot path |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.output_file` | `create_inference_result_file_reid` | ReID inference JSON path |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
