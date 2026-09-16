# Scenario Authoring

Use this reference when creating a PAIDF auto-labeling cookbook for a domain or
when recreating an experiment in the new runner. The committed result must have
no runtime dependency on non-PAIDF pipelines, commands, modules, or file locations.

## Authoring Steps

1. Identify media mode: video or image.
2. Define the stage sequence and dependencies explicitly with `workflow.nodes`.
3. Choose reusable domain assets: prompts, question bank, stage choices, model
   budgets, detector classes/prompts, and DAFT task toggles.
4. Replace one-off paths with placeholders or repo-local sample paths.
5. Put runtime settings under `runtime:` and endpoint settings under
   `endpoints:`.
6. Put captioning prompt files and budgets under the captioning node's `args`.
7. Put the shared VQA/DAFT question bank at `visual_qa.question_bank_file`.
8. Put detector/tracker backend and class list in `detection_and_tracking`, and
   detailed RF-DETR/SAM3 knobs under the detection node's `args`.
9. Run asset validation and a container dry-run.

## Field Mapping Pattern

| Concept | Cookbook target |
|---|---|
| model cache | `runtime.model_cache_path` |
| GPU ids | `runtime.gpu_ids` |
| video or image captioning | `captioning.enabled: true` |
| dense-caption prompt file | captioning node `args: [--prompt-file, <path>]` |
| image caption prompt file | captioning node `args: [--image-prompt-file, <path>]` |
| frame/window budgets | captioning node service flags |
| VQA/reasoning question bank | `visual_qa.question_bank_file` |
| DAFT task toggles | `reasoning` section in the same config |
| training dataset export | `training_export` section in the same config |
| RF-DETR/SAM3 thresholds | detection node `args` |
| SAM3 text prompts | detection node `args: [--sam3-prompts, ...]` |

## Current Runner Realities

- `workflow.nodes` locks production order while keeping each stage's flags local.
- Directory media inputs are expanded into sorted per-file entries by the runner.
- Detailed endpoint retry/backoff/timeouts are service flags today; pass them via
  the relevant node's `args`.
- Detector thresholds and SAM3 advanced settings are service flags today; pass
  them via the detection node's `args`.
- Always dry-run after authoring. A valid YAML file is not enough; the generated
  container command must include the intended flags, mounts, images, and paths.

## Domain Coverage Checklist

For each scenario, confirm the cookbook includes:

- Domain-specific prompt file, not only a generic caption.
- Question bank with all expected sections.
- Detector classes or SAM3 prompts selected for visible evidence needs.
- DAFT `reasoning` task toggles that make sense for the media mode; still images
  should not enable temporal/event tasks unless the service explicitly supports
  them.
- Validation commands in the cookbook README or MR notes.
