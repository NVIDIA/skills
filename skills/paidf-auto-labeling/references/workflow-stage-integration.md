# PAIDF Workflow Stage Integration

Use this skill when the user asks to add a new stage/service, wire a pending
architecture into the workflow runner, or review a stage MR for runner readiness.

## Workflow

Linear spine, Step 1 -> 7. The seven numbered steps are the only flow nodes. The
three decision gates are Step 2 (requirements), Step 4 (validation), and Step 7
(branch separation); the only loop is the bounded fix-and-revalidate self-loop in
Step 4. The Step 4 validation scopes (focused package/service tests, then
`make lint-check`, `make mypy`, `make test`, then the asset scan) run in that
order *inside* Step 4 as one gate - they are ordered content of that step, not
separate nodes each with its own loop. *Required Checks* are the content of Step 3
and *Guardrails* are constraints applied within Steps 3-6, not separate flow
steps. Terminals: blocked-requirements stop (Step 2), unresolved-validation stop
(Step 4), and done (Step 6, with optional Step 7).

1. **Identify the active runner path.** The current target service path is
   `services/workflow_runner`; verify the branch before touching runner-owned
   files. Then go to Step 2.

2. **Gate - are stage requirements clear?** Trigger: before implementing.
   - Unclear (target modality, annotation outputs, stage dependencies, model
     endpoints, GPU/model-cache constraints, or sidecar namespace) -> ask one
     concise question and offer concrete options (a new linear stage, a
     `training_export` dataset-aggregation stage, or a generic stage that owns
     its own sidecar namespace); wait for the answer. If it stays blocked, STOP.
   - Clear -> go to Step 3. Do not implement on assumptions.

3. **Implement against the contract (single step).** Read
   [stage-service-contract.md](workflow-stage-integration/stage-service-contract.md) and satisfy
   every item in *Required Checks* (stage identity, CLI/DataEntry contract,
   Docker registration and runner constants, canonical stage order, linear
   cookbook shape, owned sidecars and `pipeline_state.json` slice, cookbook
   fields, and tests). Treat this as one implementation step, then go to Step 4.

4. **Gate - does validation pass?** Run validation narrowest scope first, then
   broaden: focused package/service tests, then `make lint-check`, `make mypy`,
   `make test`, and the asset scan, per
   [validation-checklist.md](workflow-stage-integration/validation-checklist.md).
   - Fails -> fix the code and re-run the same scope. The loop stays inside this
     Step 4 (re-run the current scope only; it does not re-enter Steps 1-3).
     Bounded: repeat only until that scope passes, or STOP and report a blocker
     you cannot resolve. Do not broaden until the narrower scope passes.
   - Passes at all scopes -> go to Step 5.

5. **Runner dry-run.** Run the runner with `--container-dry-run` and confirm
   stage order, image names/build targets, mounts, model cache, endpoints,
   question banks, prompts, and stage args without launching containers. Then go
   to Step 6.

6. **Confirm MR readiness (done).** Re-check the MR-readiness items in
   [validation-checklist.md](workflow-stage-integration/validation-checklist.md): unrelated
   changes excluded; new stage names, cookbook fields, and dry-run examples
   documented; error-case tests included; pending service/image dependencies
   stated. The integration task is complete here; Step 7 is optional.

7. **Gate - did the user request branch separation?** Runs after Step 6.
   - Yes -> keep each integration part on its own branch (service rename, runner
     schema, stage implementation, cookbook recipe, skill/doc updates).
   - No -> keep the change stacked as the user prefers.

## Required Checks

- Stage CLI contract: every container entrypoint accepts `--input-file` JSONL
  and uses the shared `DataEntry` schema.
- Docker registration: the service package has `[project.scripts]` and
  `[tool.build.images]`; the runner constants in `services/workflow_runner` map
  each stage name to a container image and build target.
- Stage set: canonical stages run in the fixed relative order
  `super_resolution` -> `detection_and_tracking` -> `captioning` ->
  `visual_qa` -> `reasoning` -> `training_export` -> `person_attribute_search`.
  A stage key can differ from its image/build target; e.g.
  `person_attribute_search` (the Visual Attribute Search product) keeps its stage key
  but maps to the `event-and-person-attribute-search-service` image and build target.
- Cookbook shape: author a `workflow.nodes` DAG. It is validate-only today
  (parsed, topologically flattened, and run sequentially, not executed as a
  parallel DAG), but every shipped cookbook uses it and
  `test_repo_cookbooks_use_workflow_nodes_and_list_environment` requires it;
  do not author a `stages:` list.
- Input/output sidecars: stage outputs stay under the scene `data_path`, with
  durable diagnostics under `sidecars/<stage>/` unless the DAFT contract
  requires `contextual/` or `task/`.
- `pipeline_state.json`: each stage writes only its owned state slice and
  preserves sibling slices.
- Cookbook fields: new knobs have declarative cookbook fields or explicit
  `stage_args` guidance, plus dry-runnable example paths.
- Dry-run behavior: runner dry-run prints stage order, images, mounts,
  endpoints, model cache, question banks, prompts, and stage args without
  launching containers.
- Tests: add focused unit tests for parser/runner behavior and service contract
  tests for the new stage.

## Examples

Adding a new `media_chunking` stage.

Service package `pyproject.toml` (Docker registration):

```toml
[project.scripts]
media-chunking = "media_chunking.cli:main"

[tool.build.images]
media-chunking-service = { dockerfile = "Dockerfile", context = "." }
```

Runner constants in `services/workflow_runner/.../cookbook.py` — add the per-stage
image-override entry so a cookbook can pin the image:

```python
_IMAGE_ARG_BY_STAGE: dict[StageName, tuple[str, str]] = {
    # ...existing stages...
    "media_chunking": ("media_chunking_image", "--media-chunking-image"),
}
```

Entrypoint honors the shared contract (JSONL `DataEntry`, owned sidecar + state):

```bash
media-chunking --input-file "$INPUT_JSONL"   # writes sidecars/media_chunking/ and its own pipeline_state.json slice
```

Cookbook can pin the image per stage, then dry-run to confirm the plan:

```yaml
container:
  images:
    media_chunking: media-chunking-service:dev
```

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/<scenario>/configs/pipeline.yaml --container-dry-run'
```

## Guardrails

- Do not add runner-only flags to stage CLIs unless the service truly owns the
  behavior.
- Do not hard-code absolute experiment paths, endpoints, secrets, or local model
  cache paths in committed cookbooks.
- Do not claim parallel or non-linear DAG execution; the runner only linearizes
  and validates a stage graph.
- Do not let two stages write the same artifact key or sidecar path.
- Do not reference the retired single `daft_export` stage. DAFT `task/` writing
  now belongs to `reasoning`, and dataset aggregation belongs to the shipped
  `training_export` stage (a well-scoped generic export stage).
