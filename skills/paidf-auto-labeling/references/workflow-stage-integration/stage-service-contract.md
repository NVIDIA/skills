# Stage And Service Contract

Use this checklist before adding a new runner stage or service package.

## Stage Identity

- Choose one canonical snake_case stage name for the runner and cookbooks.
- Choose one service package name, image name, and build target.
- Add or update `[project.scripts]` so `make run SCRIPT=<package>:main` can
  discover the entrypoint.
- Add or update `[tool.build.images]` so `make build IMAGE=<package>:<target>`
  can build the image.
- Update runner stage choices, image defaults, build targets, service args, and
  README tables together.
- A stage key can differ from its image and build target. For example,
  `person_attribute_search` (the Visual Attribute Search product) keeps its
  snake_case stage key while its image and build target are named
  `event-and-person-attribute-search-service`. `training_export` is a shipped example
  of a well-scoped generic export stage that aggregates datasets.

## CLI Contract

Every stage service should accept:

- `--input-file <path>`: JSONL of core `DataEntry` records.
- `--log-level <level>`: compatible with repo logging.
- Stage-owned flags only. Runner-owned flags such as `--container-*`,
  `--stage-arg`, and stage ordering flags stay in the runner.

The service should read each entry's `media_path` and `data_path`. It should not
invent a second manifest schema unless the runner explicitly transforms into it.

## Data And Sidecar Contract

The scene root is `DataEntry.data_path`. The original media path stays caller
input; active media for local task execution is staged under `sidecars/active.*`.
Preserved original media is stored under `sidecars/raw.*`.

Stage outputs belong in predictable locations:

- DAFT contextual artifacts: `contextual/`.
- DAFT task artifacts: `task/`.
- Diagnostics and rich handoff evidence: `sidecars/<stage>/`.
- Shared cross-stage state: `sidecars/pipeline_state.json`.

Media-transforming stages may produce a new media file, but must only promote it
to active media after the output exists and passes stage validation. Stages that
do not transform media should leave active media unchanged.

## Pipeline State Keys

The shared state model has these top-level slices:

- `schema_version`
- `data_entry_id`
- `media_path`
- `enhanced_media`
- `task_artifacts`
- `annotation_export`

Use typed state models in the task package for stage-specific payloads. Store
them under `task_artifacts["<stage_name>"]` unless the core model already has a
dedicated slice, such as `enhanced_media` or `annotation_export`.

When writing state:

- Load the existing state first.
- Update only the owned slice.
- Preserve other slices.
- Write through `core.write_pipeline_state` so the file is updated atomically.

## Cookbook Integration

Expose stable stage knobs through cookbook fields when they are part of the
recipe. Use `stage_args.<stage>` for pass-through options that are still
service-specific or experimental. Cookbook examples should use sample
paths such as `data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4`
(staged from NGC, not shipped in git) and `output/<scenario>`.
