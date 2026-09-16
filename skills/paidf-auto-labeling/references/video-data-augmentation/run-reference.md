# Video Data Augmentation Run Reference

Concise operator reference for running the end-to-end Video Data Augmentation
pipeline (traffic-safety use case) through `workflow-runner`.

Cookbook: `cookbooks/video_data_augmentation/configs/pipeline_video.yaml`
(pipeline: `video`). The committed cookbook ships placeholder paths; supply the
real media path, SAM3 weights host mount, and served VLM/LLM model names at run
time (placeholder paths plus `NVIDIA_API_KEY` env vars), keeping
secrets and absolute host paths out of the tracked file.

## Stage-by-Stage Config Summary

Fixed execution order:
`super_resolution` -> `detection_and_tracking` -> `captioning` -> `visual_qa` ->
`reasoning`.

| Stage | Key config | Notes |
|---|---|---|
| `super_resolution` | resolver `seedvr2`, variant `seedvr2_3b` | `--resolution-policy auto` upscales only low-res clips; high-res clips keep active media. `--empty-output-policy fail`. Heavy GPU stage. |
| `detection_and_tracking` | model `sam3`, tracker `sam3`, traffic `classes` | SAM3 text prompts (a car, a bus, a truck, a motorcycle, a bicycle, a pedestrian, a traffic light). `--sam3-write-annotated-video` emits the annotated video. |
| `captioning` | `--input-source tracking` | Dense-caption prompt `../prompts/dense_caption/traffic_scene_prompt.md`. |
| `visual_qa` | `--generation-mode window-vlm-llm`, `--input-source tracking` | Shared `question_bank_file: ../question_bank.json`. |
| `reasoning` | targets + subtasks | Targets: `scene_description`, `video_summarization`, `open_qa`, `mcq_openended`, `bcq_openended`, `causal_linkage`. Plus `events`, `msted`, `temporal_localization`, and `causal_linkage` (`mode: auto_from_events`). |

## Mounts, Endpoints, and GPU Needs

- **SAM3 weights**: mount host weights at `/models/sam3:ro`; the cookbook sets
  `SAM3_MODEL_PATH=/models/sam3`. The committed file ships a placeholder host
  path; provide the real host path at run time, keeping the tracked file's
  placeholder.
- **Endpoints**: reachable VLM and LLM OpenAI-compatible endpoints under
  `endpoints.vlm` and `endpoints.llm`. Set the served model names to the actual
  served model ids.
- **Model cache**: `runtime.model_cache_path` (e.g. `ckpts`) for SeedVR2 and
  other stage weights; first run may download checkpoints.
- **GPUs**: `runtime.gpu_ids` (e.g. `all` or specific ids). SeedVR2
  super_resolution is heavy; plan a dedicated GPU for it and keep endpoint GPUs
  separate.

## Dry-Run and Execute Commands

Always dry-run first:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

Execute (only after a passing pre-flight and explicit user approval):

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml'
```

Add `--container-build-images` when the runner image needs to be built.

## Expected Artifacts

Under the sample `out_dir`:

- **super_resolution**: super-resolved / active media (upscaled clips only).
- **detection_and_tracking**: tracking sidecars plus the annotated video.
- **captioning**: contextual / task caption outputs.
- **visual_qa**: task QA outputs.
- **reasoning**: DAFT `task/` files for the enabled reasoning targets.

Validate by listing `ls -lhR <out_dir>/` and confirming each enabled stage wrote
its outputs.

## Reasoning max_tokens Note

For reasoning/"thinking" VLM/LLM models (for example `gcp/google/gemini-3-*`),
raise the reasoning-substage `max_tokens` (e.g. `32768`, within the model output
limit). Reasoning models spend part of the budget on internal thinking, so a low
cap truncates or empties outputs (thinking-token tax). Non-reasoning instruct
models keep defaults. Make the change in the cookbook config, not in code.
