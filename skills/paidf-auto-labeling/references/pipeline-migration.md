# PAIDF Auto-Labeling Pipeline Migration

Use this skill to guide repo-to-UPA migrations. The goal is to make new
pipelines arrive as reusable UPA capabilities plus cookbooks, not as piles of
pipeline-specific services.

## Core Rule

Start with a cookbook. Extend existing services when the missing behavior is a
mode, prompt, question bank, detector option, windowing choice, or output format.
Add a new service only for a reusable primitive with a stable sidecar contract.

Good service names describe capabilities:

- `media_chunking`
- `query_generation`
- `anomaly_vote`
- `artifact_export` (illustrative generic name; the real shipped export stage is
  `training_export`)

Avoid service names that describe one pipeline or repo:

- `agentic_captioning`
- `agentic_tracking`
- `warehouse_pipeline_service`
- `traffic_query_task`

## Migration Workflow

Linear planning spine, Step 1 -> 8; it produces a migration plan and does not run
anything. Notes so independent readers reconstruct the same graph:

- The reference reads in Steps 2-5 are prerequisite reads (Step 5 only when the
  source is non-linear), not branches.
- The per-stage reuse decision (cookbook-only, extend an existing service, or new
  generic service) happens inside Step 7. It is a per-stage classification, not a
  control-flow branch or loop: each stage resolves independently to exactly one
  outcome, and no outcome loops back into the reference-read steps.
- *Core Rule*, *Output Shape*, and *Guardrails* are constraints/output content
  applied while planning, not separate flow steps.
- The only loop is the bounded validate-revise loop in Step 8.

1. Inventory the source repo: scripts, entrypoints, inputs, outputs, models,
   endpoints, GPU requirements, resume behavior, and output schemas.
2. Read [migration-playbook.md](pipeline-migration/migration-playbook.md) before
   proposing branch structure, service boundaries, or cookbook scope.
3. Read [service-reuse-matrix.md](pipeline-migration/service-reuse-matrix.md) before
   suggesting a new service or task package.
4. Read [sidecar-contracts.md](pipeline-migration/sidecar-contracts.md) before changing
   UPA service IO, `pipeline_state.json`, active-media handoff, or repeated
   stage behavior.
5. Read [nonlinear-video-pipeline-example.md](pipeline-migration/nonlinear-video-pipeline-example.md)
   when a source repo has a non-linear video pipeline, such as tracking plus
   chunking plus VLM annotation plus query generation.
6. Verify the current branch's available services and workflow-runner stage
   registry before claiming a cookbook is executable.
7. Produce a migration plan that separates cookbook-only work, generic service
   capability work, runner registration work, and validation. Classify each
   source stage here into exactly one reuse outcome (see *Output Shape*).
8. Validate the outline with a runner dry-run and contract tests. On revealed
   gaps, revise the outline/classification in place and re-validate at most twice
   (a bounded self-loop on Step 8; do not re-enter Steps 1-7). Then terminate:
   finalize the plan if it passes, or STOP and record the blocking gap in the MR
   note as a follow-up capability.

## Output Shape

For a migration plan, include:

- Source pipeline summary: stages, inputs, outputs, and dependencies.
- Reuse decision: cookbook-only, existing service extension, or new generic
  service for each stage.
- Target UPA cookbook outline with `pipeline`, `runtime`, `endpoints`, stage
  sections, and `workflow.nodes.<node>.args` (`stage_args` is compatibility-only),
  plus the stage shape that fits the source:
  - a linear `stages:` list when the source pipeline runs stages in a simple
    sequence; or
  - `workflow.nodes` (with `needs:`) to capture dependency intent when the source
    is non-linear or dependencies matter more than a flat list.
  The runner currently linearizes and validates `workflow.nodes` rather than
  parallel-executing the DAG, so use it to express dependency intent - do not
  claim parallel execution.
- Sidecar contract for any new or changed service.
- Minimal branch/MR plan; avoid unnecessary follow-up runner branches.
- Validation commands: unit tests, service contract tests, runner dry-run, and
  one tiny end-to-end smoke input when feasible (Step 8 bounds the validate-revise
  loop and its exit).

## Examples

Migrating a source "warehouse anomaly" tracker + VLM captioner + query generator.

Reuse decision (all cookbook-only, no new services):

- tracker -> `detection_and_tracking` (SAM3 prompts).
- VLM captioner -> `captioning` (`--prompt-file`).
- query generator -> `person_attribute_search` LLM query generation.

Target cookbook outline (`workflow.nodes` with colocated args, adapted from a
shipped example):

```yaml
pipeline: video
workflow:
  nodes:
    detection_and_tracking:
      stage: detection_and_tracking
      args: [--model, sam3, --sam3-prompts, a forklift, a pallet]
    captioning:
      stage: captioning
      needs: [detection_and_tracking]
      args: [--input-source, tracking, --prompt-file, ../prompts/warehouse_caption.md]
    visual_qa: { stage: visual_qa, needs: [captioning] }
    reasoning: { stage: reasoning, needs: [visual_qa] }
    training_export: { stage: training_export, needs: [reasoning] }
endpoints:
  vlm: { url: <vlm-url>, model: <vlm-model> }
  llm: { url: <llm-url>, model: <llm-model> }
```

Validate the outline before proposing the branch/MR:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/warehouse/configs/pipeline.yaml --container-dry-run'
```

## Guardrails

- Do not create a new task or service only because the source repo has a new
  project name.
- Do not migrate hard-coded experiment paths, secrets, endpoints, or local
  dataset locations into committed cookbooks.
- Do not copy secret values (API keys, tokens, credentials) or user-specific
  paths into the migration plan, summaries, examples, or MR notes. Reference them
  by env-var name or placeholder and redact the values, even outside committed
  cookbooks.
- Do not put final DAFT/artifact writing into captioning or VQA unless the
  product explicitly wants those stages to stop being reusable intermediate
  representation producers.
- Do not claim parallel or non-linear execution if the current runner only
  validates and topologically flattens `workflow.nodes`.
- Do not allow two stages or two workflow nodes to write the same sidecar path
  or `pipeline_state.json` key.
- Prefer capability gaps over branch sprawl: one well-scoped generic service or
  registry improvement is better than several pipeline-specific branches.
- Map migrated stages onto the current canonical stages (fixed relative order;
  cookbooks select a subset): `super_resolution`, `detection_and_tracking`,
  `captioning`, `visual_qa`, `reasoning`, `training_export`,
  `person_attribute_search`. The single `daft_export` stage is retired: DAFT
  `task/` writing now belongs to `reasoning`, and training dataset aggregation
  belongs to the new `training_export` stage. `person_attribute_search` stays the
  stage key for the Visual Attribute Search product (service image/build target
  remains `event-and-person-attribute-search-service`).
- Adapt a shipped example cookbook (`visual_attribute_search`,
  `event_verification_reasoning`, `video_data_augmentation`) before inventing a
  new pipeline layout.
