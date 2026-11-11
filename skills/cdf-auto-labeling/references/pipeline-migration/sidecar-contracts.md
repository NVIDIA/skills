# Sidecar Contracts

Use this reference when a migrated pipeline needs new outputs or downstream
handoffs.

## Required Runtime Contract

Every UPA service stage should:

- Read `--input-file <path>` as JSONL of core `DataEntry` rows.
- Treat `DataEntry.data_path` as the scene root.
- Use `DataEntry.media_path` or the current active-media state as input.
- Create outputs under `data_path`, not beside the source dataset.
- Preserve sibling state by updating `sidecars/pipeline_state.json` through a
  shared core locked merge helper, not ad hoc read/write replacement.
- Fail fast with useful errors for malformed entries or missing required inputs.

## Sidecar Layout

Use predictable folders:

```text
<data_path>/
  sidecars/
    pipeline_state.json
    active.<ext>
    raw.<ext>
    <stage_name>/
      ...
  contextual/
    ...
  task/
    ...
  logs/
    workflow_runner.jsonl
```

Rules:

- Put rich intermediate evidence under `sidecars/<stage_name>/`.
- Put final DAFT contextual artifacts under `contextual/`.
- Put final DAFT task artifacts under `task/`.
- Put workflow-runner execution logs under `logs/`.
- Do not write outside `data_path` except for explicitly configured export
  destinations.

## Pipeline State

Use `sidecars/pipeline_state.json` when a downstream stage must discover a
producer's output without knowing producer-specific paths.

Writers should use a shared core helper that acquires the per-scene lock,
validates the existing JSON, deep-merges incoming keys into the existing state,
writes a temporary file, and atomically renames it into place. Malformed state
must fail before writing. Merge behavior should be deterministic and preserve
sibling top-level keys and `task_artifacts` entries written by parallel DAG
nodes. Implement this in the core/runtime branch that owns pipeline-state
behavior, not as cookbook- or skill-local logic.

Good state entries include:

- Sidecar path.
- Source media path or chunk path.
- Stage name or node id.
- Version/schema marker for the payload.
- Minimal metadata needed for downstream validation.

Avoid storing:

- Large model responses that already live in a sidecar file.
- Absolute user-only paths when relative scene paths are enough.
- Secrets, API keys, or endpoint credentials.

## Repeated Stages And Workflow Nodes

If the same stage can run more than once, each node needs its own namespace:

```text
sidecars/<node_id>/
pipeline_state.task_artifacts["<node_id>"]
```

Repeated stage nodes are allowed when they have distinct node IDs and
namespaces. Do not reuse a sidecar path or `task_artifacts` key across nodes.

## Intermediate Representation Policy

Perception stages should produce reusable intermediate representations. For
example, captioning should produce dense captions/window metadata that other
stages can consume. It should not be forced to emit only DAFT-specific outputs
unless the product explicitly narrows the service's purpose.

Final artifact writing belongs in export/artifact stages so evolving schemas can
be changed in one place.
