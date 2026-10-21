# Captioning stage

Single-stage reference for the `captioning` stage that runs a VLM over windowed
active media (plus tracking sidecars when `--input-source tracking`) to emit
contextual captions consumed by `visual_qa` and `reasoning`.

## When to use / not use

- Use: selecting the VLM endpoint, choosing input source, tuning window size or
  frame budget, or debugging captions that are empty, off-style, or too costly.
- Do not use: to run a full pipeline end to end, author a whole cookbook, or
  write prompt files or question banks.

## Config

Cookbook block `captioning:` (essential fields):
- `enabled: true`
- VLM endpoint via `endpoints.vlm`.

Two mutually exclusive choices in `stage_args.captioning` (pick exactly one
option in each; no other branching):
- Input source (`--input-source`): `original` captions the raw/active media;
  `tracking` captions the tracked crops/media from `detection_and_tracking`.
- Window unit: set `--window-seconds` OR `--window-frames`, and leave the other
  at `0`.

Flat tuning flags (independent, no branching):
- `--prompt-file <dense caption prompt md>` — caption prompt/style.
- `--remainder-threshold` — whether trailing partial windows are kept.
- `--sampling-fps`, `--max-frames` — bound frames sampled per window.
- `--resolution` — frame resolution fed to the VLM.

## Instructions

This skill returns captioning configuration or debugging guidance; it does not
run the pipeline. Provide the config immediately - do not gate it behind
execution.

1. **Classify the task.** Configuring/tuning `captioning`, or debugging empty,
   off-style, or too-costly captions. Either is in scope. If the request is a full
   pipeline run, whole-cookbook authoring, or prompt/question-bank authoring, STOP
   and hand off to the matching skill.
2. **Gather needs.** Identify the VLM endpoint, input source, window unit, and
   frame budget (or the symptom). If a required input (endpoint URL/model,
   cookbook path, or input media) is missing or ambiguous, ask - do not guess.
3. **Return the config or fix.** Produce the `captioning:` block and `stage_args`
   per *Config* - one `--input-source`, one window unit, plus the flat tuning
   flags. Suggest a dry-run (`--container-dry-run`) so the caller can verify the
   generated command before running.

**Execution is out of scope.** This skill only produces configuration; it does
not run the stage. Executing (Docker via `workflow-runner`, which requires
explicit user approval) is the operator skill's responsibility.

**When debugging.** On a missing or unreachable VLM endpoint, missing input media,
or empty/malformed captions, report the specific cause from the failing output; do
not fabricate captions.

## Examples

Dense captions over tracked crops with a custom prompt:

```yaml
captioning:
  enabled: true
endpoints:
  vlm:
    url: http://host.docker.internal:18002/v1
    model: qwen-vl
stage_args:
  captioning: >-
    --input-source tracking
    --prompt-file ../prompts/dense_caption/traffic_scene_prompt.md
    --window-seconds 4 --window-frames 0
    --sampling-fps 2 --max-frames 16
    --resolution 448
```

Dry-run to inspect the generated Docker command before running:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

## Gotchas

- A custom `--prompt-file` changes caption style (e.g. a traffic-scene prompt).
- `--resolution` and `--max-frames` trade caption quality against cost.

## Guardrails

Follow [guardrails.md](guardrails.md). Captioning cookbooks use `<model-cache>`
for VLM cache mounts; `--prompt-file` is a repository path, not inlined text.
