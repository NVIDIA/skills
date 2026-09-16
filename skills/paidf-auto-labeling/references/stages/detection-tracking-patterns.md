# Detection And Tracking Patterns

The PAIDF `workflow-runner` service exposes a small stable surface for detection
and tracking. Backend selection and class/prompt lists live in the cookbook;
all backend tuning lives in `stage_args.detection_and_tracking`.

## Runner Mapping

Direct fields read from `detection_and_tracking:`:

```yaml
detection_and_tracking:
  enabled: true
  model: rfdetr       # rfdetr or sam3
  tracker: boosttrack # boosttrack, bytetrack, or sam3
  classes:
    - person
    - car
```

Mapping rules:

- `model: rfdetr` with `tracker: boosttrack` becomes `--tracker rfdetr-boosttrack`.
- `model: rfdetr` with `tracker: bytetrack` becomes `--tracker rfdetr-bytetrack`.
- `model: sam3` or `tracker: sam3` becomes `--tracker sam3` and selects the SAM3
  service image/build target by default.
- `rfdetr-deepocsort` is reserved but not registered in this branch; do not select it.
- Only `model`, `tracker`, and `classes` are mapped directly. Put thresholds,
  overlays, SAM3 knobs, and media-copy behavior in `stage_args`.

Do not carry over unsupported detector keys such as `use_reid`, `asso_func`,
`min_hits`, `max_age`, `save_video_red_id`, or DeepOCSORT-specific settings.
Translate only to supported service flags.

## Pass-Through Args

```yaml
stage_args:
  detection_and_tracking:
    - --threshold
    - "0.25"
    - --iou-threshold
    - "0.3"
    - --per-class
    - --min-track-frames
    - "5"
    - --save-video
    - --save-red-id-overlay
```

Always dry-run and inspect the generated command:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <config> --container-dry-run'
```

## Data Flow

Detection/tracking is annotation-only for active media. It consumes the prior SR
artifact when `pipeline_state.enhanced_media.success` points to an existing file;
otherwise it consumes the original `DataEntry.media_path`. It does not promote a
new active media file.

Outputs are recorded under `pipeline_state.task_artifacts["detection_and_tracking"]`:

- `contextual/objects.json`
- `contextual/instances.json`
- RF-DETR overlays under `sidecars/rfdetr/` when requested
- SAM3 annotated media under `sidecars/sam3/` when requested

Captioning `input_source=auto` prefers SAM3 `annotated_video_path`, then RF-DETR
`red_id_overlay_path`, then enhanced media, then original media. If the caption
prompt should reference object IDs, request an overlay and make the captioning
input source explicit with `--input-source tracking`.

## RF-DETR Backends

Use RF-DETR when the annotation needs COCO-style boxes, counts, and object IDs.
The default production choice is `rfdetr-boosttrack`; `rfdetr-bytetrack` is a
simpler fast path without the BoostTrack association layer.

Runtime requirements:

- Matching RF-DETR service image: `detection-and-tracking-rfdetr-service`.
- Checkpoint at `RFDETR_MODEL_PATH` or `<model_cache_path>/rfdetr/rf-detr-base.pth`
  (default `/models/rfdetr/rf-detr-base.pth`).
- `--allow-model-download` can download the base checkpoint when network access
  is allowed; keep it off in locked-down production runs.
- `RFDETR_MODEL_SHA256` validates custom checkpoint files when set.

Supported RF-DETR knobs:

- `--threshold`: detector confidence threshold; start around `0.25` for traffic
  and `0.2` to `0.3` for cluttered indoor scenes, then tune with real clips.
- `--iou-threshold`: BoostTrack association threshold; default `0.3`.
- `--per-class`: keeps separate association state per class; recommended when
  multiple classes are tracked.
- `--min-track-frames`: filters short tracks from final DAFT artifacts; use `5`
  for noisy video, `1` for images or sparse clips.
- `--save-video`: writes RF-DETR detection and tracking overlays.
- `--save-red-id-overlay`: writes a re-ID overlay that captioning can consume.
- `--save-rgb`: writes diagnostic RGB frames under `sidecars/rfdetr/rgb/`.
- `--copy-media`: copies analyzed media into `raw/` instead of symlinking.

RF-DETR class lists must be COCO-valid. Common COCO names include:

```text
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog,
horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

RF-DETR rejects unknown names during tracker initialization. Do not add open
vocabulary labels such as `forklift`, `pallet jack`, `hard hat`, `robot arm`,
`gripper`, `weapon`, or `firearm`; cover those with VLM/VQA prompts or SAM3
when promptable masks/tracks are required.

## SAM3 Backend

Use SAM3 when the annotation needs promptable object tracks, mask contours,
object-ID overlays, or non-COCO objects. SAM3 is the right choice for event
verification over short clips when the VLM should see visible IDs attached to
prompted actors.

Runtime requirements:

- Matching SAM3 service image: `detection-and-tracking-sam3-service`.
- CUDA-capable GPU. The backend fails fast without CUDA.
- Local SAM3 weights at `SAM3_MODEL_PATH` or `<model_cache_path>/sam3`
  (default `/models/sam3`). This branch does not download SAM3 weights.
- At least one `--sam3-prompts` value, or `classes` values to use as prompts.

Prompt rules:

- Prefer explicit `--sam3-prompts` in `stage_args` instead of relying on
  `classes` fallback.
- Use concrete visible object noun phrases: `a pedestrian`, `a car`,
  `a forklift`, `a robot arm`, `road debris`.
- Do not use event labels as prompts: `collision`, `near miss`, `unsafe act`,
  `theft`, `policy violation`.
- Keep prompt inventory small enough for the clip; too many prompts increase
  compute and can create ambiguous IDs.

Core SAM3 knobs:

- `--sam3-target-fps`: sampled processing FPS. Use `3.0` to `5.0` for traffic or
  long-ish event verification, `10.0` for short clips where motion detail matters.
- `--sam3-session-reset-s`: chunk duration before resetting model state. Smaller
  values such as `5.0` reduce memory and isolate IDs by chunk; `10.0` is default.
- `--sam3-max-duration-s` or `--sam3-max-clip-duration-s`: maximum accepted clip
  duration. Default is `30.0`; raise only when GPU memory has been tested.
- `--sam3-write-annotated-video`: writes the overlay under `sidecars/sam3/`.
- `--sam3-annotated-video-label-style id|name|none`: choose ID labels for VLM
  grounding, names for human review, or none for clean visualization.
- `--sam3-annotated-video-mask-opacity 0..100`: mask fill opacity; `0` keeps
  outlines only, `20` is a useful review overlay.
- `--sam3-annotated-video-trails`: draw object trails when trajectory matters.

Advanced SAM3 quality knobs map directly to `Sam3VideoConfig` when provided:

- `--sam3-score-threshold-detection`
- `--sam3-det-nms-thresh`
- `--sam3-new-det-thresh`
- `--sam3-fill-hole-area`
- `--sam3-recondition-every-nth-frame`
- `--sam3-recondition-on-trk-masks true|false`
- `--sam3-high-conf-thresh`
- `--sam3-high-iou-thresh`

Treat these as experiment knobs. Keep defaults unless a real clip shows missed
objects, fragmented masks, or duplicate tracks. When tuning, change one knob at a
time and re-check on the same representative clip. Stop as soon as detections and
tracks look correct, or after at most 2-3 tuning passes. If quality is still
inadequate after that bound, stop and escalate (switch backend, revisit
prompts/classes, or flag the clip) rather than looping further.

Example SAM3 stage args for traffic ID overlays:

```yaml
stage_args:
  detection_and_tracking:
    - --sam3-prompts
    - a car
    - a bus
    - a truck
    - a motorcycle
    - a bicycle
    - a pedestrian
    - a traffic light
    - --sam3-target-fps
    - "3.0"
    - --sam3-session-reset-s
    - "5.0"
    - --sam3-max-duration-s
    - "45.0"
    - --sam3-write-annotated-video
    - --sam3-annotated-video-label-style
    - id
  captioning:
    - --input-source
    - tracking
```

## Use-Case Guidance

Traffic / roadway safety:

- RF-DETR: use `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`,
  `traffic light`, `stop sign` for stable COCO boxes.
- SAM3: use when overlays/masks should ground incidents or when prompts such as
  `road debris`, `an emergency vehicle`, or `a pedestrian` matter.

Warehouse / operational liability:

- RF-DETR: use `person` and sometimes `truck` as a coarse equipment proxy;
  optionally add visible COCO carried items such as `backpack`, `handbag`, or
  `suitcase`.
- SAM3: use for `a forklift`, `a pallet jack`, `a pallet`, `a worker`,
  `a box`, or zones/objects that RF-DETR cannot label.

Security surveillance:

- RF-DETR: usually `person`; add `backpack`, `handbag`, `suitcase`, or vehicle
  classes only when boxes for those objects are needed.
- SAM3: use concrete visible objects and people; do not use prompts for intent,
  identity, or protected attributes.

Employee conduct:

- RF-DETR: `person` for station presence and coarse movement.
- SAM3: use for workcell objects, tools, or PPE-like visible items only when
  masks/IDs are needed; VQA should still express uncertainty.

Robotics:

- RF-DETR: `person` plus any COCO tools/objects actually relevant.
- SAM3: use for `a robot arm`, `a gripper`, `a tool`, `a bolt`, `a workpiece`,
  or `a human hand` when promptable tracks support task-phase evidence.

Generic images:

- Skip detection unless boxes or object inventories are required.
- RF-DETR supports single images; `min_track_frames=1` is appropriate.
- SAM3 can process image inputs as one-frame clips when local SAM3 runtime is
  available, but it is expensive for simple scene captions.

## Validation Checks

After the stage runs, verify:

- `contextual/objects.json` and `contextual/instances.json` exist.
- Expected RF-DETR overlays exist when `--save-video` or `--save-red-id-overlay`
  was requested.
- Expected SAM3 annotated media exists when `--sam3-write-annotated-video` was
  requested.
- `pipeline_state.json` contains `task_artifacts["detection_and_tracking"]`.
- A downstream captioning stage that needs IDs uses `--input-source tracking` or
  leaves `input_source=auto` with an existing overlay path.
