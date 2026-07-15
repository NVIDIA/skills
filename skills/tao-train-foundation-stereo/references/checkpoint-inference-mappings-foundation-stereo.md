# FoundationStereo Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

DepthNet Stereo training writes checkpoint files under `<results_dir>/train/` using `model_epoch_<epoch>_step_<step>.pth` and a `dn_model_latest.pth` symlink. For `evaluate`, `inference`, `export`, `quantize`, and resume/retrain, select checkpoints through the SDK/model resolver so a requested best, epoch, or step checkpoint resolves to that exact file. Use `dn_model_latest.pth` only when the user explicitly asks for latest.

Parent PyT `gen_trt_engine` is intentionally absent from the supported action set because the current `depth_net` entrypoint rejects it. The TensorRT engine mappings are owned by `tao-deploy-foundation-stereo.md`.

Inference mappings from TAO Core `depth_net_stereo.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `model.model_type` | `FoundationStereo` | FoundationStereo |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `model.model_type` | `FoundationStereo` | FoundationStereo |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `model.model_type` | `FoundationStereo` | FoundationStereo |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| quantize | `model.model_type` | `FoundationStereo` | FoundationStereo |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| train | `model.model_type` | `FoundationStereo` | FoundationStereo |
| train | `model.stereo_backbone.depth_anything_v2_pretrained_path` | `{'link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth', 'destination_path': '/ptm/depth_net/stereo_backbone/depth_anything_v2_vits.pth'}` | {'link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth', 'destination_path': '/ptm/depth_net/stereo_backbone/depth_anything_v2_vits.pth'} |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
