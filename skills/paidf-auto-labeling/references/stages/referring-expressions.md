# Referring Expressions stage

Single-stage reference for `referring_expressions`: known boxes → short
discriminative phrases via a VLM (SoM-lite numbered overlays). Inverse of
`grounding_2d` (caption → boxes).

- [When to use / not use](#when-to-use--not-use)
- [Packaging](#packaging)
- [Config](#config)
- [Instructions](#instructions)
- [Inputs → Outputs](#inputs--outputs)
- [Examples](#examples)
- [Gotchas](#gotchas)
- [Guardrails](#guardrails)

## When to use / not use

- Use: wiring boxes→phrases after detection or grounding, tuning VLM/IoU/overlay
  knobs, or debugging empty/mismatched region descriptions.
- Do not use: to run a full pipeline end to end, author a whole cookbook, or
  ground captions into new boxes (that is [grounding-2d.md](grounding-2d.md)).

## Packaging

| Image | Role |
|-------|------|
| `referring-expressions-service` | Slim VLM product image (no SAM3/CUDA bake) |
| `detection-and-tracking-sam3-service` | Upstream boxes when composing detect → refer |

Stages scale independently. Mount SAM3 weights only on the detection stage; the
referring stage needs VLM + `contextual/objects.json`. No product `PYTHONPATH`
mounts.

Read [referring-expressions-patterns.md](referring-expressions-patterns.md)
for inputs, type vocabulary, and validation checks.

## Config

Cookbook block `referring_expressions:` (essential):
- `enabled: true`
- VLM via `endpoints.vlm`
- Image: `container.images.referring_expressions: referring-expressions-service`

Common compositions:

```yaml
# Detect then refer
stages: [detection_and_tracking, referring_expressions]

# Refer only (objects.json already present)
stages: [referring_expressions]
```

Flat `stage_args.referring_expressions` knobs:
- `--draw-box-overlay` — write SoM overlay under sidecars
- `--min-match-iou` — link VLM marks to DAFT boxes
- `--frame-number` — frame index in `objects.json`
- `--max-tokens`, `--temperature`, `--top-p`, `--timeout-s`, `--retries`
- `--force-reprocess`

Object `type` is open vocabulary (no closed traffic enum). Detection prompts
belong under `stage_args.detection_and_tracking` when that stage is included.

Confirm flags with a dry run:
`make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`

## Instructions

This skill returns referring configuration or debugging guidance; it does not
run the pipeline. Provide the config immediately - do not gate it behind
execution.

1. **Classify the task.** Configuring/tuning `referring_expressions`, or
   debugging phrases/overlays. If the request is a full pipeline run,
   whole-cookbook authoring, or caption→boxes grounding, STOP and hand off.
2. **Gather needs.** Identify whether boxes already exist
   (`contextual/objects.json`) or detection must run first; VLM endpoint; image
   paths. If a required input is missing or ambiguous, ask - do not guess.
3. **Return the config.** Produce `referring_expressions:` / `stage_args` /
   images per *Config*. For detect→refer, set SAM3 prompts on detection only;
   keep referring mounts free of weight trees when objects already exist.
   Suggest `--container-dry-run`.

**Execution is out of scope.** Docker via `workflow-runner` requires explicit
user approval (operator skill).

**When debugging.** On missing `contextual/objects.json`, unreachable VLM, or
empty `sidecars/referring_expressions/referring_expressions.json`, report the specific cause; do not
fabricate phrases.

## Inputs → Outputs

- Consumes: image `media_path`; `contextual/objects.json` (and optional
  `contextual/instances.json`) from detection or grounding.
- Produces: `sidecars/referring_expressions/referring_expressions.json`,
  `sidecars/referring_expressions/` (step0 + optional `marked_boxes.jpg`).

## Examples

Detect → refer:

```yaml
stages:
  - detection_and_tracking
  - referring_expressions
container:
  images:
    detection_and_tracking: detection-and-tracking-sam3-service
    referring_expressions: referring-expressions-service
  env:
    NVIDIA_API_KEY: EMPTY
    SAM3_MODEL_PATH: /models/sam3
  mounts:
    - <sam3-weights>:/models/sam3:ro
endpoints:
  vlm:
    url: http://host.docker.internal:8000/v1
    model: Qwen/Qwen3-VL-30B-A3B-Instruct
detection_and_tracking:
  enabled: true
  model: sam3
  tracker: sam3
  classes: [person, vehicle]
referring_expressions:
  enabled: true
stage_args:
  detection_and_tracking: >-
    --sam3-prompts person vehicle
    --sam3-score-threshold-detection 0.5
  referring_expressions: >-
    --draw-box-overlay
    --min-match-iou 0.3
    --frame-number 0
```

Refer only:

```yaml
stages:
  - referring_expressions
container:
  images:
    referring_expressions: referring-expressions-service
  env:
    NVIDIA_API_KEY: EMPTY
referring_expressions:
  enabled: true
stage_args:
  referring_expressions: >-
    --draw-box-overlay --min-match-iou 0.3 --frame-number 0
```

Dry-run:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/image_spatial_grounding/configs/pipeline_referring.yaml --container-dry-run'
```

## Gotchas

- Requires upstream boxes; referring does not detect objects itself.
- Open `type` strings are expected; do not reintroduce a closed traffic enum.
- Mark/ID mismatches can still occur when many small boxes crowd the frame —
  tighten detection thresholds or `--min-match-iou` before rewriting prompts.
- Rebuild `referring-expressions-service` after task/service code changes.

## Guardrails

Follow [guardrails.md](guardrails.md). Referring cookbooks mount SAM3 via
`<sam3-weights>`; this stage does not detect objects itself.
