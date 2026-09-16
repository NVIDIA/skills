# Detection and Tracking stage

Single-stage reference for `detection_and_tracking`: detect and track objects with
RF-DETR/BoostTrack or SAM3, and optionally emit ID overlays and per-track crops for
downstream captioning, visual_qa, and person_attribute_search.

## When to use / not use

- Use: choosing a backend, writing class lists / SAM3 prompts, tuning thresholds,
  overlays, or crop extraction, or reviewing a `detection_and_tracking` section.
- Do not use: to run a full pipeline end to end, or to author or restructure a
  whole cookbook.

## Backend choice

- RF-DETR + BoostTrack/ByteTrack -> COCO-style boxes, counts, stable IDs. Class
  names must be COCO-valid. Image `detection-and-tracking-rfdetr-service`.
- SAM3 -> promptable object masks/tracks, ID overlays, and non-COCO objects. Needs
  local SAM3 weights + CUDA. Image `detection-and-tracking-sam3-service`.

Read [detection-tracking-patterns.md](detection-tracking-patterns.md)
before writing or reviewing a section: it has the full runner mapping, per-backend
knob lists, the COCO class list, per-domain class/prompt guidance, and validation
checks.

## Config

Only `enabled`, `model` (rfdetr|sam3), `tracker` (boosttrack|bytetrack|sam3), and
`classes` are direct cookbook fields under `detection_and_tracking:`. Everything
else (thresholds, overlays, SAM3 knobs, crop extraction, media copy) goes under
`stage_args.detection_and_tracking`.

Mapping: `model: rfdetr` + `tracker: boosttrack` -> `--tracker rfdetr-boosttrack`;
`model`/`tracker` sam3 -> `--tracker sam3` (selects the SAM3 image).

Common `stage_args`:
- RF-DETR: `--threshold`, `--iou-threshold`, `--per-class`, `--min-track-frames`,
  `--save-video`, `--save-red-id-overlay`.
- SAM3: `--sam3-prompts <noun phrases>`, `--sam3-score-threshold-detection`,
  `--sam3-new-det-thresh`, `--sam3-target-fps`, `--sam3-session-reset-s`,
  `--sam3-max-duration-s`, `--sam3-write-annotated-video`,
  `--sam3-annotated-video-label-style id|name|none`, `--sam3-annotated-video-mask-opacity`.
- Crops (feed person_attribute_search): `--extract-crops`, `--crop-classes`,
  `--crops-per-track`, `--crop-padding`, `--min-crop-size`.

Confirm every intended flag with a dry run:
`make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`

## Instructions

This skill returns detection/tracking configuration or review guidance; it does
not run the pipeline. Provide the config immediately - do not gate it behind
execution.

1. **Classify the task.** Configuring/reviewing `detection_and_tracking`, or
   debugging detections/tracks. Either is in scope. If the request is a full
   pipeline run or whole-cookbook authoring, STOP and hand off.
2. **Gather needs.** Identify the target objects and which backend is available
   (RF-DETR service, or SAM3 weights + CUDA). If a required input is missing or
   ambiguous, ask - do not guess.
3. **Pick the backend.** COCO objects with stable IDs and boxes -> `model: rfdetr`
   + BoostTrack/ByteTrack. Non-COCO or promptable masks -> `model: sam3` +
   `tracker: sam3` (needs SAM3 weights + CUDA).
4. **Return the config, tuning within bounds.** Produce the block and `stage_args`
   per *Config* for the chosen backend. If quality knobs need tuning, iterate in a
   bounded loop - stop when detections/tracks look correct or after at most 2-3
   passes (the reference lists which knobs), else escalate (switch backend or
   revisit prompts/classes). Suggest a dry-run (`--container-dry-run`) so the
   caller can verify the generated command.

**Execution is out of scope.** This skill only produces configuration; it does
not run the stage. Executing (Docker via `workflow-runner`, which requires
explicit user approval) is the operator skill's responsibility.

**When debugging.** On missing SAM3 weights/CUDA, an unreachable RF-DETR service,
missing input media, or empty detections/tracks, report the specific cause from
the failing output; do not fabricate detections.

## Inputs -> Outputs

- Consumes: super-resolved active media when present, else the original media
  (annotation-only; does not promote a new active media file).
- Produces: `contextual/objects.json`, `contextual/instances.json`, per-track crops,
  RF-DETR overlays under `sidecars/rfdetr/` or SAM3 annotated media under
  `sidecars/sam3/` when requested, and `task_artifacts["detection_and_tracking"]`
  in `pipeline_state.json`.

## Examples

RF-DETR + BoostTrack (COCO boxes, stable IDs, per-track crops):

```yaml
detection_and_tracking:
  enabled: true
  model: rfdetr
  tracker: boosttrack
  classes: [person, bicycle, car, motorcycle, bus, truck]
stage_args:
  detection_and_tracking: >-
    --threshold 0.4 --iou-threshold 0.5 --per-class --min-track-frames 5
    --save-video --save-red-id-overlay
    --extract-crops --crop-classes person --crops-per-track 4 --crop-padding 0.1
```

SAM3 (promptable masks/tracks for non-COCO objects):

```yaml
detection_and_tracking:
  enabled: true
  model: sam3
  tracker: sam3
stage_args:
  detection_and_tracking: >-
    --sam3-prompts "a forklift" "a pallet" "a person wearing a hard hat"
    --sam3-score-threshold-detection 0.5 --sam3-target-fps 4
    --sam3-write-annotated-video --sam3-annotated-video-label-style id
```

Dry-run to inspect the generated Docker command before running:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

## Gotchas

- Must run before `person_attribute_search` (Visual Attribute Search,
  `event-and-person-attribute-search-service`), whose per-object unit is the track/crop.
- Do not add non-COCO names (`forklift`, `hard hat`, `pallet jack`, ...) to RF-DETR
  class lists - RF-DETR rejects them; use SAM3 prompts or VLM/VQA instead.
- SAM3 prompts are concrete visible noun phrases ("a pedestrian"), never event
  labels ("collision", "unsafe act").
- Do not carry over unsupported detector keys (`use_reid`, `asso_func`, `min_hits`,
  `max_age`, `save_video_red_id`); translate only to supported service flags.
- Keep thresholds conservative until tested on representative clips; treat detector
  output as evidence, not truth.

## Guardrails

Follow [guardrails.md](guardrails.md). Mount SAM3/RF-DETR weights read-only via
placeholders such as `<model-cache>`.
