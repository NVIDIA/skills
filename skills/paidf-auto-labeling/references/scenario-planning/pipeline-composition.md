# Pipeline Composition

The PAIDF runner composes supported stages into a pipeline. The agent selects a
subset of the canonical stages; the runner builds and executes Docker commands.
There is no single fixed pipeline: each cookbook picks the stages it needs, run
in their fixed relative order.

- [Supported Stage Names](#supported-stage-names)
- [Common Pipelines](#common-pipelines)
- [Stage Selection Rules](#stage-selection-rules)
- [Data Flow Contract](#data-flow-contract)
- [Custom Pipeline Compatibility Checks](#custom-pipeline-compatibility-checks)
- [Dry-Run First](#dry-run-first)

## Supported Stage Names

The canonical stages, in fixed relative order, are:

```text
super_resolution
detection_and_tracking
referring_expressions
captioning
grounding_2d
visual_qa
reasoning
training_export
person_attribute_search
```

Branch note: `grounding_2d` and `referring_expressions` may land on separate MRs.
Verify the current checkout's `workflow_runner` stage list before composing both
in one cookbook. Typical product subsets:

- Caption → boxes: `captioning` → `grounding_2d`
- Boxes → phrases: `detection_and_tracking` → `referring_expressions`
  (or `referring_expressions` alone when `contextual/objects.json` exists)

The image pipeline default subset remains `captioning -> visual_qa -> reasoning ->
training_export`. Include `person_attribute_search` only when
`detection_and_tracking` has already run (PAS requires per-track crops). The single
`daft_export` stage is
retired: DAFT `task/` writing belongs to `reasoning`, and dataset aggregation
belongs to `training_export`. `person_attribute_search` is the stage key for the
Visual Attribute Search product (service image/build target
`event-and-person-attribute-search-service`); it requires `detection_and_tracking`
first for per-track attributes.

Verify current branch support before execution because some services may be
pending in separate MRs.

## Common Pipelines

Image caption to training export:

```yaml
stages:
  - captioning
  - reasoning
  - training_export
```

Image VQA to training export, minimal when `visual_qa` does not require
`captioning` artifacts:

```yaml
stages:
  - visual_qa
  - reasoning
  - training_export
```

Add `captioning` before `visual_qa` only when the VQA flow explicitly consumes
caption artifacts; otherwise `visual_qa` can feed `reasoning` directly.

Video captions to training export:

```yaml
stages:
  - captioning
  - reasoning
  - training_export
```

Tracked video auto-labeling:

```yaml
stages:
  - detection_and_tracking
  - captioning
  - visual_qa
  - reasoning
  - training_export
```

Visual Attribute Search (per-track person attributes), as shipped by the
`visual_attribute_search` cookbook:

```yaml
stages:
  - detection_and_tracking
  - person_attribute_search
```

Image 2D grounding (caption → expressions → SAM3 boxes):

```yaml
stages:
  - captioning
  - grounding_2d
```

Referring expressions (boxes → phrases):

```yaml
stages:
  - detection_and_tracking
  - referring_expressions
```

Low-resolution video auto-labeling:

```yaml
stages:
  - super_resolution
  - detection_and_tracking
  - captioning
  - visual_qa
  - reasoning
  - training_export
```

Mixed-resolution video auto-labeling:

```yaml
stages:
  - super_resolution
  - detection_and_tracking
  - captioning
  - visual_qa
  - reasoning
  - training_export

stage_args:
  super_resolution:
    - --resolution-policy
    - auto
    - --min-input-short-side
    - "720"
    - --min-input-long-side
    - "1280"
```

Additional shipped example cookbooks include `event_verification_reasoning`
(evidence-first `reasoning` verdicts) and `video_data_augmentation`.

## Stage Selection Rules

- Use `super_resolution` with `--resolution-policy auto` for mixed-resolution
  video datasets; only use unconditional SR when every input is known to need
  enhancement.
- Choose SR output resolution and window size based on hardware. Validate the
  real container on representative clips before a full run because SeedVR2 3B at
  720p can exceed A40 memory.
- Use `detection_and_tracking` when labels need object IDs, trajectories,
  overlays, masks, counts, or object-level grounding.
- Use `captioning` for image/video evidence prose and VLM-derived visual
  descriptions.
- Use `visual_qa` when a question bank should produce reusable visual evidence.
- Use `reasoning` when outputs should be written into DAFT `task/`/contextual
  artifacts or need evidence-first verdicts.
- Use `training_export` when per-scene outputs should be aggregated into a
  training/export dataset.
- Use `person_attribute_search` (after `detection_and_tracking`) for the Visual
  Attribute Search product's per-track attribute labels.

## Data Flow Contract

Every custom linear pipeline must preserve:

```text
DataEntry.media_path       effective input after prepare_input
DataEntry.data_path        shared scene directory
sidecars/raw.*             preserved original media
sidecars/active.*          current media handoff
sidecars/pipeline_state.json
contextual/
task/
sidecars/logs/
```

Media-transforming stages update active media only after verifying the output
artifact exists. Annotation-only stages should leave active media unchanged.

## Custom Pipeline Compatibility Checks

Before proposing a custom combination, check:

- Does each stage consume outputs produced by earlier stages or by the original
  scene state?
- Does any stage require active media from SR or tracking overlays?
- Does the selected captioning `input_source` match the intended evidence
  source: original, enhanced, tracking, or auto?
- Are prompt files and question banks mounted read-only through cookbook config
  or runner stage args?
- Are model cache and output scene directories mounted with write access where
  needed?
- Does the generated dry-run command include every intended image, build target,
  mount, endpoint, GPU, and stage arg?

## Dry-Run First

Always inspect a dry-run before execution:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <config> --container-dry-run'
```

Only run for real after the dry-run confirms the stage order and data-flow
assumptions.
