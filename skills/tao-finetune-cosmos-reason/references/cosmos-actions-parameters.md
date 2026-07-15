# Cosmos-Reason Actions And Parameters

Evaluate behavior, dataset notes, important parameters, hardware, error patterns, and parent-model inference.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Config format
- Task type
- LoRA Evaluation
- Selective download
- Results
- Datasets
- Important Parameters
- Training Loop
- Model & Policy
- Parallelism (Multi-GPU / Multi-Node)
- Optimization & Data Loading
- Vision Encoders
- Checkpointing
- Validation
- Logging
- Hardware
- Error Patterns
- Spec Param / Parent Model Inference

## Evaluate

The `actions.evaluate` block in `references/skill_info.yaml` declares the action's inputs (annotation file + media folder + model) and outputs (results directory). For SDK invocation see `skills/platform/tao-run-platform/SKILL.md`.

### Config format

The evaluator reads a **flat TOML** config with top-level keys: `dataset`,
`model`, `task`, `evaluation`, `vision`, `generation`, `metrics`, `results`,
`num_gpus`, and `results_dir`. The defaults template
(`references/spec_template_evaluate.yaml`) matches this flat structure. Use
dotted overrides such as `dataset.annotation_path`, `model.model_name`, and
`evaluation.batch_size`.

### Task type

- Empty string (`""`) — General Evaluator. Auto-detects binary classification (yes/no) from ground truth and computes TP/FP/TN/FN/accuracy/precision/recall/F1.
- `"its_directionality"` — ITS-specific evaluator for left/right/straight classification. Do NOT use for collision detection.

### LoRA Evaluation

To evaluate a fine-tuned LoRA model, pass the checkpoint path via spec_overrides:

```python
spec_overrides={
    'model.model_name': 's3://bucket/results/{train_job_id}/safetensors/epoch_2',
    'model.enable_lora': True,
    'model.base_model_path': 'hf_model://nvidia/Cosmos3-Nano',
    'evaluation.batch_size': 10,
}
```

The LoRA adapter is downloaded from S3/Lustre before the evaluator runs; the evaluator merges it with the base model and runs inference on the merged weights.

### Selective download

When the input declaration carries a `selective` block (`{annotation, format, keys}`), only the files referenced in `dataset.annotation_path` (under the `video` key) are pulled — not the full media folder. For a 112-sample collision dataset, this downloads ~500MB instead of the full 4.8GB folder.

### Results

- `results.json` — per-sample predictions with `video_id`, `response`, `question`, `gt`
- Binary metrics: accuracy, balanced accuracy, precision, recall, F1
- Text metrics: BLEU, ROUGE, BERTScore
- When Lustre is available, results write to Lustre for cross-job persistence (e.g., gap analysis reads directly), then upload to S3.

## Datasets

The `data_sources` config in config.json maps dataset URIs to spec paths. It
appends `annotations.json` to the dataset directory URI by convention. If your
annotations and media do not share a root, or if the annotation file has a
different name, use direct spec overrides instead of forcing a root:

```python
spec_overrides={
    'custom.train_dataset': {
        'annotation_path': 's3://bucket/train/my_annotations.json',
        'media_path': 's3://bucket/media/videos_train.tar.gz',
    },
    'custom.val_dataset': {
        'annotation_path': 's3://bucket/eval/my_annotations.json',
        'media_path': 's3://bucket/eval/videos/',
    },
}
```

**Eval dataset** is optional for plain training only when `train.train_policy.dataset.test_size` is used to auto-split training data. For AutoML or any workflow optimizing a validation metric such as `val/avg_loss`, require either an explicit `custom.val_dataset` or a deliberate auto-split setting before launch preflight passes. If a validation dataset is provided, validation metrics are computed at the frequency set by `validation.freq_in_epoch`.

Before runner generation, verify the annotation JSON is readable and the
referenced media path or archive is visible from the selected platform. Missing
optional annotation fields are not a launch blocker for current Cosmos-RL SFT
training, and the agent must not patch source annotations unless the user
explicitly asks for that dataset mutation.

## Important Parameters

### Training Loop
- **train.epoch**: Number of training epochs. Default 10. Use at least 2 for
  local smoke or AutoML runs that need a host-visible best checkpoint for
  evaluate/inference; one-epoch runs can leave only a broken `best` symlink
  after checkpoint cleanup.
- **train.train_batch_per_replica**: Global batch size per training step. Ideally >= 32 for stability. CRITICAL: must be divisible by `train.train_policy.mini_batch` (default 1 in the packaged smoke-safe template). Recommended production value: 32.
- **train.compile**: Set to true for potential speedup on newer GPUs (H100), else false.
- **train.output_dir**: Output directory for checkpoints and logs.

### Model & Policy
- **policy.model_name_or_path**: HuggingFace model path. The packaged default is `hf_model://nvidia/Cosmos3-Nano`. Override this only when the user provides a different HuggingFace model id, `hf_model://...` URI, or cluster-local snapshot path.
- **policy.model_max_length**: Context window size. Must be 40960 for video SFT. Affected by FPS, resolution, and prompt length.
- **policy.model_gradient_checkpointing**: Save VRAM by recomputing activations. Keep true for large models.

### Parallelism (Multi-GPU / Multi-Node)
- **policy.parallelism.dp_shard_size**: Data-parallel shard size. CRITICAL: should equal **GPUs per node** (the Cosmos-RL equivalent of `num_gpus`).
- **policy.parallelism.dp_replicate_size**: Data-parallel replication = **node count** (equivalent of `num_nodes`). For single-node training set to 1.
- **policy.parallelism.tp_size**: Tensor parallelism. Default 1.
- **policy.parallelism.cp_size**: Context parallelism. Default 1.
- **policy.parallelism.pp_size**: Pipeline parallelism. Default 1.

For multi-node, set `dp_replicate_size = num_nodes` and `dp_shard_size = gpus_per_node`. Cosmos-RL handles the distributed init internally via FSDP — it does **not** rely on the platform-level `MASTER_ADDR` / `WORLD_SIZE` env vars the way `torchrun`-launched jobs do. Just submit with `gpu_count=<gpus_per_node>` and `num_nodes=<N>` on the SDK; the Cosmos-RL spec keys drive the actual sharding.

For platform-side multi-node setup (sbatch flags on SLURM, Indexed Job + Service on Kubernetes), see the platform skill's "Multi-node training" section: `skills/platform/tao-run-on-slurm`, `skills/platform/tao-run-on-kubernetes`. Brev and local Docker are single-host only.

### Optimization & Data Loading
- **train.optm_lr**: Learning rate. Default 1e-6.
- **train.train_policy.type**: Training policy. Default `sft`.
- **train.train_policy.mini_batch**: Micro-batch size per GPU. If OOM, reduce this. Constraint: `train_batch_per_replica % mini_batch == 0`.
- **train.train_policy.dataset.name**: Unique ID for dataset cache. IMPORTANT: change this if you modify `fps` or `total_pixels` to force cache regeneration.
- **train.train_policy.dataset.test_size**: Validation split. Float (0.0–1.0) = ratio; Int = absolute number.
- For AutoML or small subsets, verify every generated recommendation before
  launch with `scripts/check_tao_launch_preflight.py
  --effective-batch-limit train_annotation=<batch_size>,<dp_shard_size>` and
  reject/cap configs where `train_batch_per_replica >
  num_train_samples / dp_shard_size`.

### Vision Encoders
- **custom.vision.fps** *or* **custom.vision.nframes** — **mutually exclusive**, set exactly one.
  - `nframes` (default in template): extract this many frames evenly across the clip. This is the safest default for 1-GPU AutoML smoke runs.
  - `fps`: extract frames at this rate. High motion: 3. Low motion/static: 1–2. Use when the selected videos, `policy.model_max_length`, and GPU memory can absorb the expanded token count.
  - Setting both makes qwen-vl-utils' decord backend error out (`Only accept either fps or nframes`) and silently fall back to torchvision, which deadlocks under multi-worker dataloading (`BlockingIOError [Errno 11]` swscaler errors). If you switch from `fps` to `nframes`, also delete `fps` from your spec.
- Do not require per-record `video_fps` for the packaged `nframes` template.
  If a run switches to `custom.vision.fps` or a selected dataset/image profile
  requires per-record timing, validate annotations before any download or job
  launch:
  ```bash
  scripts/check_tao_launch_preflight.py --platform <platform> \
    --path train_annotation=/path/to/train.json \
    --path val_annotation=/path/to/val.json \
    --json-required-field train_annotation=video_fps \
    --json-required-field val_annotation=video_fps
  ```
- **custom.vision.total_pixels**: Resolution constraint. Increase if the object of focus is small relative to the frame. Default 3136000.
- **custom.system_prompt**: Instructions prepended to every prompt.

### Checkpointing
- **train.ckpt.save_freq_in_epoch**: Save every N epochs. Default 1.
- **train.ckpt.max_keep**: Keep N most recent checkpoints. Default 2 for
  AutoML/minimal runs so the best LoRA adapter remains available even when the
  container records the best validation step before later epoch cleanup.
- **train.ckpt.export_safetensors**: Export in safetensors format. Default true.

When verifying downstream handoff, prefer `train_output_dir/best/safetensors`
only if it resolves inside the results mount. In the current local Docker image,
epoch-based saving writes concrete artifacts under
`train_output_dir/<timestamp>/safetensors/epoch_N` and
`train_output_dir/<timestamp>/checkpoints/epoch_N/policy`, but
`best/best_score.json` and the `best/{safetensors,checkpoints}` symlinks can
record `step_N` targets that do not exist. If a `best` symlink points at a
missing `step_*` directory, resolve the best validation step back to the
corresponding retained `epoch_*` directory and use that exact folder. Do not
fall back to "latest" silently.

For evaluate, pass the resolved LoRA folder directly:
`model.model_name=<train_output_dir>/<timestamp>/safetensors/epoch_N`,
`model.enable_lora=true`, and
`model.base_model_path=<same base model used for training>` (default
`hf_model://nvidia/Cosmos3-Nano`, or the local base-model snapshot path). For
resume/retrain, pass the exact Cosmos checkpoint policy folder as a string:
`train.resume=<train_output_dir>/<timestamp>/checkpoints/epoch_N/policy`.
Avoid `train.resume=true` for local Docker epoch-based checkpoints because the
current resolver scans `step_*` checkpoint directories and can miss the
`epoch_*` folders. Do not count downloaded base-model shards under `ptm/`,
launcher staging files under `inputs/`, or the broken `best` symlink itself as
fine-tuned checkpoints for handoff.

### Validation
- **validation.freq_in_epoch**: Run validation every N epochs. Too frequent slows training.

### Logging
- **logging.logger**: Options: `console`, `wandb`.
- **logging.project_name** / **logging.experiment_name**: W&B experiment tracking.

## Hardware

Cosmos-RL models are 8B parameters and benefit from multi-GPU training with FSDP sharding. `dp_shard_size` should equal total GPU count. Recommended: 8x A100 or H100 (80GB each).

## Error Patterns

**CUDA out of memory (train)**: Reduce `train.train_policy.mini_batch` or increase `dp_shard_size`. Enable `fsdp_offload` if GPU memory is limited. Also check `custom.vision.total_pixels` — high resolution increases memory significantly.

**OOM during evaluation with LoRA**: Loading the base model + LoRA adapter uses more memory than zero-shot eval. If zero-shot eval passes but post-training eval OOMs, reduce `evaluation.batch_size` (e.g., from 10 to 1) or lower `vision.total_pixels`. The OOM typically manifests as the node killing the process mid-run (no Python traceback — just `ERR_PROGRAM` with a node-level OOM event). This is especially likely in DEFT workflows where the same eval spec is used for both zero-shot and post-training evaluation.

**NaN loss**: Learning rate may be too high. Reduce `optm_lr` and increase `optm_warmup_epochs`.

**vision_embeds.shape[0] must be equal to n_tokens**: `model_max_length` is too small for the video input at the current FPS and resolution. Increase `policy.model_max_length` to 40960.

**Quantize image/video token mismatch**: `Mismatch in image token count between
text and input_ids` during calibration means `quantize.max_sequence_length` is
too small for the sampled media tokens. The packaged smoke template uses 4096;
do not lower it to tiny values such as 128 for video calibration.

**train_batch_per_replica not divisible by mini_batch**: The default `train_batch_per_replica=1` from the TAO Core schema is invalid because `mini_batch` defaults to 4. Immediate AssertionError on all ranks. Fix: set `train_batch_per_replica` to a multiple of `mini_batch` (recommended: 32 for large datasets, 4 for small datasets).

**train_batch_per_replica larger than samples per rank**: With FSDP, each rank sees `total_samples / dp_shard_size` samples. If `train_batch_per_replica` exceeds this, the trainer completes 0 training steps and attempts to save a checkpoint before the optimizer/scheduler is initialized, crashing with `'NoneType' object has no attribute 'state_dict'`. Fix: ensure `train_batch_per_replica <= total_samples / dp_shard_size`. For small datasets (e.g., 31 DEFT-generated samples on 8 GPUs = ~4 per rank), set `train_batch_per_replica` to 4.

**Stale dataset cache after changing fps/total_pixels**: Change `train.train_policy.dataset.name` to a new unique identifier to force cache regeneration.

**Checkpoint save failure (scheduler is None)**: The cosmos-rl trainer crashes with `'NoneType' object has no attribute 'state_dict'` when saving a checkpoint before any training step has executed. This happens when the dataset is too small for the batch size (0 steps per epoch). See the batch size error above.

**You are trying to access a gated repo**: The HuggingFace model `nvidia/Cosmos3-Nano` requires authentication. All ranks will retry in a loop until they time out. Fix: ensure `HF_TOKEN` is set in your environment (e.g., `export HF_TOKEN=...` in your shell) and passed into the container with `-e HF_TOKEN`. The user must also accept the model agreement at <https://huggingface.co/nvidia/Cosmos3-Nano>.

**Cosmos-RL GPU resource and architecture gate**: The actionable launch gate is
at least 4 GPUs with 80GB-class memory or higher, plus a GPU architecture
supported by the selected Cosmos-RL image, plus normal platform, container, S3,
and credential preflight. Run
`scripts/check_tao_launch_preflight.py --gpu-min-count 4 --gpu-min-memory-gb 80 --gpu-arch-allowlist cosmos_rl=sm_80,sm_90,sm_100,sm_120`
before launching. If the target architecture is known but cannot be detected
from the launch host, pass `--gpu-arch sm_XX` explicitly. Spark/GB10 `sm_121`
is not launchable with this image unless image introspection confirms `sm_121`
support or a newer compatible image is selected. If a resource-qualified
platform still fails with a kernel JIT error such as
`nvrtc: invalid --gpu-architecture`, classify it as an image/toolchain defect to
fix with a compatible image, not as a platform resource incompatibility.

**TAO_API_JOB_ID status logging warnings in direct Docker**: `cosmos-rl-evaluate`, `cosmos-rl-inference`, and `cosmos-rl-quantize` may log a traceback from `tao_status_logger.py` when `TAO_API_JOB_ID` is unset. For direct local-Docker model-skill validation this is nonfatal if the process exits 0 and the action writes its expected result files. Do not hide a real action failure behind this warning, but do not mark an otherwise successful local run failed only because status-file logging was unavailable.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

- **Checkpoint metadata:** format: safetensors, folder: true

Inference mappings from TAO Core `cosmos-rl.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `model.model_name` | `parent_model_folder` | model folder inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `model_path` | `parent_model_folder` | model folder inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `model.model_path` | `parent_model_folder` | model folder inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.output_dir` | `output_dir` | current job results directory |
| train | `train.resume` | `resume_model` | exact checkpoint policy folder inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
