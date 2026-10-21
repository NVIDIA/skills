# Migration Playbook

Use this reference to turn an external annotation or dataset-generation repo
into UPA-native capabilities and cookbooks.

## 1. Inventory The Source Repo

Record the current pipeline as dataflow, not scripts:

- Inputs: raw videos/images, manifests, metadata JSON, model checkpoints,
  endpoint environment variables, prompts, schemas, and labels.
- Stages: what each script reads, writes, and requires.
- Outputs: intermediate JSON, media chunks, crops, annotations, query files,
  DAFT files, visualizers, logs, and resume markers.
- Dependencies: GPU-only steps, API-only steps, model cache, ffmpeg/OpenCV,
  external submodules, and secrets.
- Concurrency: internal thread pools, batch workers, resumability, and skip
  conditions.

Prefer a small table:

| Source step | Reads | Writes | UPA target | Decision |
|---|---|---|---|---|
| chunk videos | raw videos | chunk mp4 + chunks.json | `media_chunking` | new generic service |
| dense captions | chunk mp4 | caption JSON | `captioning` | extend existing stage |

## 2. Classify Each Step

Use this order:

1. Cookbook-only: existing stage already supports the behavior through prompts,
   question banks, endpoints, classes, or `stage_args`.
2. Existing service extension: the behavior belongs to a current reusable stage
   such as captioning, detection/tracking, visual QA, reasoning (which now writes
   DAFT `task/` artifacts), or training_export (dataset aggregation).
3. New generic service: the behavior is a reusable primitive that multiple
   future pipelines could use.
4. External platform orchestration: the behavior is resource scheduling,
   distributed retry, queues, or fleet placement and should stay in OSMO,
   Airflow, or another platform layer.

Do not jump to option 3 until options 1 and 2 are ruled out.

## 3. Keep Branches Simple

Use as few follow-up branches as the review structure allows:

- Skills/docs/cookbook guidance can live together when they are review-only.
- A generic service primitive should own its task/service package, tests, Docker
  registration, and sidecar contract.
- Runner updates should be limited to stage registration, argument plumbing,
  cookbook parsing, or registry support needed by that primitive.
- Cookbook examples should sit on top of the service and runner support they
  require.

Avoid separate runner branches for every small cookbook or prompt adjustment.

## 4. Preserve The UPA Shape

Every migrated runtime stage should:

- Accept `--input-file <jsonl>` with core `DataEntry` rows.
- Use `DataEntry.media_path` and `DataEntry.data_path`.
- Write under the scene `data_path`.
- Put rich handoff evidence under `sidecars/<stage-or-node>/`.
- Write discoverable state to `sidecars/pipeline_state.json` when downstream
  stages need to find outputs.
- Register `[project.scripts]` for `make run`.
- Register `[tool.build.images]` for `make build`.
- Be dry-runnable or unit-testable without requiring a full dataset.

## 5. Author The Cookbook Last

Create the cookbook after service boundaries are stable. The cookbook should
contain:

- `pipeline: video` or `pipeline: image`.
- `workflow.nodes` when dependencies matter more than a simple list.
- `runtime:` for GPU ids and model cache.
- `endpoints:` for VLM/LLM settings.
- Stage sections for stable knobs.
- `stage_args` for service-specific or experimental flags.
- Repo-local sample paths, never user-specific dataset paths.

## 6. Validate Migration Readiness

Minimum validation:

- Unit tests for new parser, service, or sidecar logic.
- Contract test that each new service accepts `--input-file`.
- Runner dry-run that shows expected stage order, images, mounts, prompts,
  endpoints, and stage args.
- One tiny smoke input if models/endpoints are locally available.
- MR note that lists unsupported behavior and follow-up capabilities.
