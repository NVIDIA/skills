# Checkpoint and Output Resolution

Read the selected action in `skill_info.yaml`. The producing workflow stages inputs and writes a concrete nested specification before the platform submits the job. These mappings describe artifact roles; ordinary model execution does not require an SDK, a service job ID, or generated runner patches.

| Action | Specification field | Resolve from |
|---|---|---|
| All | `results_dir` | Current job record's explicit results directory |
| Train (fine-tune) | `train.pretrained_model_path` | Selected compatible pretrained Sparse4D checkpoint |
| Train (resume) | `train.resume_training_checkpoint_path` | Explicit checkpoint from the run being continued |
| Evaluate | `evaluate.checkpoint` | Selected completed training/AutoML child checkpoint |
| Inference | `inference.checkpoint` | Selected completed training/AutoML child checkpoint |
| Export | `export.checkpoint` | Selected Sparse4D checkpoint |
| Export | `export.onnx_file` | Explicit output path in the current job's results |
| Quantize | `quantize.model_path` | Exact selected checkpoint for the chosen backend |

When the user supplies a job identifier, read its recorded result location and verify the exact checkpoint exists there. Preserve its epoch/step and raw-versus-EMA identity; do not select the first filesystem match or silently replace a resume checkpoint with initialization weights. If several candidates remain and the intended one is not established, resolve the selection before launching.

Keep architecture, anchors, class order, and preprocessing paired with that checkpoint. Supply an encryption key only if the artifact requires it, through the platform's approved secret handling. The external LTT MLP is needed for LTT training and resume but not for evaluation, inference, or export.

The legacy `spec_params` mapping in `skill_info.yaml` also serves AutoML consumers. AutoML's dotted `spec_overrides` map is expanded before launch; actual container YAML must use nested keys. It must not override an explicitly selected checkpoint with an inferred parent artifact.
