# Spec Param / Parent Model Inference

Model-specific parent-model mappings are declared in `references/skill_info.yaml` under `spec_params`. Keep this section aligned with that metadata so generated runners and agents resolve checkpoints before `create_job()` instead of guessing file names.

Inference mappings from this model skill:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| train | `results_dir` | `output_dir` | current job results directory |
| train | `model.backbone.pretrained_backbone_path` | C-RADIO download descriptor | staged classify backbone |
| train | `train.resume_training_checkpoint_path` | `resume_model` | resume checkpoint inferred from parent train results |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| evaluate | `evaluate.checkpoint` | `parent_model` | checkpoint inferred from parent train or AutoML child results |
| inference | `results_dir` | `output_dir` | current job results directory |
| inference | `inference.checkpoint` | `parent_model` | checkpoint inferred from parent train or AutoML child results |
| export | `results_dir` | `output_dir` | current job results directory |
| export | `export.checkpoint` | `parent_model` | checkpoint inferred from parent train or AutoML child results |
| export | `export.onnx_file` | `create_onnx_file` | ONNX artifact path created for deploy |
| quantize | `results_dir` | `output_dir` | current job results directory |
| quantize | `quantize.model_path` | `parent_model` | checkpoint inferred from parent train or AutoML child results |
| segment_train | `results_dir` | `output_dir` | current job results directory |
| segment_train | `train.resume_training_checkpoint_path` | `resume_model` | resume checkpoint inferred from parent train results |
| segment_evaluate | `results_dir` | `output_dir` | current job results directory |
| segment_evaluate | `evaluate.checkpoint` | `parent_model` | checkpoint inferred from parent train or AutoML child results |
| segment_inference | `results_dir` | `output_dir` | current job results directory |
| segment_inference | `inference.checkpoint` | `parent_model` | checkpoint inferred from parent train or AutoML child results |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
