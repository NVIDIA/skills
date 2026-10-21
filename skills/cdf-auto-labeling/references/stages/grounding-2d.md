# Grounding 2D stage

Single-stage reference for `grounding_2d`: caption → VLM referring expressions →
SAM3 boxes/masks on images. Inverse of `referring_expressions` (boxes → language).

## When to use / not use

- Use: choosing caption source, tuning VLM/SAM3 knobs, selecting the live GPU
  image, or debugging empty/over-broad expressions or missing boxes.
- Do not use: to run a full pipeline end to end, author a whole cookbook, or
  generate phrases from known boxes (that is [referring-expressions.md](referring-expressions.md)).

## Packaging

| Image | Role |
|-------|------|
| `grounding-2d-service` | Slim CI / VLM-only |
| `grounding-2d-sam3-service` | Live GPU product image |

Build the live image with `make build IMAGE=grounding-2d-service:sam3` (build
target alias); the resulting tag is `grounding-2d-sam3-service`. Build from
`services/grounding_2d_service/docker/Dockerfile.gpu` with `PACKAGE_NAME=grounding-2d-service`
and `RUNTIME_FLAVOR=sam3`. Independent of the detection service image (no sibling
`FROM`, no `PYTHONPATH` mounts). Mount SAM3 weights at `/models/sam3`.

Read [grounding-2d-patterns.md](grounding-2d-patterns.md) for caption
precedence, filter policy guidance, and validation checks.

## Config

Cookbook block `grounding_2d:` (essential):
- `enabled: true`
- VLM via `endpoints.vlm`
- Live image: `container.images.grounding_2d: grounding-2d-sam3-service`
- Weights mount only: `<sam3-weights>:/models/sam3:ro`

Common composition (runner order puts captioning before grounding on this branch):

```yaml
stages:
  - captioning
  - grounding_2d
```

Flat `stage_args.grounding_2d` knobs (independent):
- `--filter-ungroundable-expressions` / `--no-filter-ungroundable-expressions`
- `--max-instances-per-expression`, `--min-instance-score`, `--min-bbox-area`
- `--max-tokens`, `--temperature`, `--top-p`, `--timeout-s`, `--retries`
- `--sam3-target-fps`, `--sam3-session-reset-s`, `--sam3-max-duration-s`
- `--sam3-write-annotated-media`, `--sam3-annotated-media-label-style`,
  `--sam3-annotated-media-mask-opacity`
- `--force-reprocess`, `--caption` (optional caption override)

Optional `--expression-filter-policy-path` exists for advanced custom deny JSON;
default product path uses **no** keyword pack (VLM `groundable` + SAM3 limits).

Confirm flags with a dry run:
`make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`

## Instructions

This skill returns grounding configuration or debugging guidance; it does not
run the pipeline. Provide the config immediately - do not gate it behind
execution.

1. **Classify the task.** Configuring/tuning `grounding_2d`, or debugging
   expressions/boxes. If the request is a full pipeline run, whole-cookbook
   authoring, or boxes→phrases, STOP and hand off.
2. **Gather needs.** Identify image inputs, caption source (captioning stage vs
   sidecar vs `--caption`), VLM endpoint, and SAM3 weights path. If a required
   input is missing or ambiguous, ask - do not guess.
3. **Return the config.** Produce `grounding_2d:` / `stage_args` / image + mount
   per *Config*. Prefer `captioning` then `grounding_2d` without
   `sidecars/input.json` so dense captions feed grounding. Suggest
   `--container-dry-run`.

**Execution is out of scope.** Docker via `workflow-runner` requires explicit
user approval (operator skill).

**When debugging.** On missing SAM3 weights/CUDA, unreachable VLM, empty
captions, or empty `sidecars/grounding_2d/grounding_2d.json`, report the specific cause from the
failing output; do not fabricate boxes.

## Inputs → Outputs

- Consumes: image `media_path`; caption from `--caption`, optional
  `sidecars/input.json`, or captioning artifacts.
- Produces: `sidecars/grounding_2d/grounding_2d.json`, `sidecars/grounding_2d/` (step0/step1),
  optional SAM3 overlays under `sidecars/sam3/`.

## Examples

Captioning → grounding (live GPU image):

```yaml
stages:
  - captioning
  - grounding_2d
container:
  images:
    captioning: captioning-service
    grounding_2d: grounding-2d-sam3-service
  env:
    NVIDIA_API_KEY: EMPTY
    SAM3_MODEL_PATH: /models/sam3
  mounts:
    - <sam3-weights>:/models/sam3:ro
endpoints:
  vlm:
    url: http://host.docker.internal:8000/v1
    model: Qwen/Qwen3-VL-30B-A3B-Instruct
grounding_2d:
  enabled: true
stage_args:
  grounding_2d: >-
    --filter-ungroundable-expressions
    --max-instances-per-expression 10
    --min-instance-score 0.5
    --min-bbox-area 64
    --sam3-write-annotated-media
    --sam3-annotated-media-label-style name
```

Dry-run:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/image_spatial_grounding/configs/pipeline_grounding.yaml --container-dry-run'
```

## Gotchas

- Image-only MVP; pick video keyframes outside this stage.
- Do not bake domain keyword denylists into the task; keep cookbooks
  domain-agnostic by default.
- Nested `grounding_2d.sam3:` YAML is not auto-mapped — put knobs in `stage_args`.
- Rebuild `grounding-2d-sam3-service` after task/service code changes.

## Guardrails

Follow [guardrails.md](guardrails.md). Grounding cookbooks mount SAM3 at
`<sam3-weights>:/models/sam3:ro` and use `<model-cache>` for any VLM cache.
