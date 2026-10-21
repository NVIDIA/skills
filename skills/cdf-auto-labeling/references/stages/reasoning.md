# Reasoning stage

The `reasoning:` cookbook block turns captions, visual_qa items, and events
into DAFT `task/` artifacts that include reasoning traces — for example
`task/mcq.json`, `task/bcq.json`, `task/open_qa.json`, plus the enabled
reasoning targets. It uses VLM/LLM endpoints only; no local GPU.

## When to use / not use

- Use to select reasoning targets, tune substage `max_tokens`, or fix empty /
  truncated DAFT banks.
- Do not use to run a full cookbook or to write `training_export`/detector
  config. DAFT `task/` writing lives here now; the retired single `daft_export`
  stage no longer does this and dataset aggregation is `training_export`.

## Instructions

This skill returns reasoning-stage configuration or debugging guidance; it does
not run the pipeline. Provide the config immediately - do not gate it behind
execution. The flow is the three numbered steps below (Classify -> Gather ->
Return); the target/file lists inside a step are config values it sets, not flow
nodes or branches, and *Config*, *Examples*, *Gotchas*, and *Guardrails* are
reference content, not steps.

1. **Classify the task.** Configuring reasoning targets / `max_tokens`, or fixing
   empty/truncated DAFT banks. Either is in scope. If the request is a full
   cookbook run, or writing `training_export`/detector config, STOP and hand off.
2. **Gather needs and select targets.** Choose the reasoning-enrichment `targets`
   (scene_description, video_summarization, temporal_description, open_qa,
   mcq_openended, bcq_openended, causal_linkage) and confirm the VLM/LLM
   endpoints. `open_qa`, `mcq_openended`, and `bcq_openended` are not alternatives:
   if used, they must appear in both `reasoning.targets` and their corresponding
   task stage blocks, with source files confirmed (`open_qa.question_file`,
   `mcq_openended.item_file`, `bcq_openended.question_file`).
   `temporal_localization` is a task stage only, not a `reasoning.targets` value;
   set `temporal_localization.query_file` when enabling it. If a required file or
   endpoint is missing or ambiguous, ask - do not guess.
3. **Return the config.** Produce the `reasoning:` block per *Config*: causal pairs
   via `causal_linkage.mode: auto_from_events` with a `max_pairs_auto` cap;
   `max_tokens` per *Config* (default for instruct models, larger for reasoning
   models). Suggest a dry-run (`--container-dry-run`) to verify the command.

**Execution is out of scope.** This skill only produces configuration; it does
not run the stage. Executing (Docker via `workflow-runner`, which requires
explicit user approval) is the operator skill's responsibility.

**On missing or unresolvable input.** This skill runs no tools or models, so the
only failure mode is missing or ambiguous input. If a required target file,
endpoint, or value stays missing after you ask once, STOP and report exactly what
is needed; return partial config only if the user explicitly accepts it, and
never guess values.

**When debugging.** On an unreachable VLM/LLM endpoint, a missing
question/item/query file, or empty/truncated `task/` banks (often a low
`max_tokens` on a reasoning model), report the specific cause; do not fabricate
answers.

## Config

`reasoning:` sub-blocks:
- `reasoning:` (enrichment pass) `targets`: scene_description,
  video_summarization, temporal_description, open_qa, mcq_openended,
  bcq_openended, causal_linkage
- `events:`, `msted:`
- `temporal_localization:` (query_file) - a separate task stage, not a
  `reasoning.targets` value
- `open_qa:` (question_file)
- `mcq_openended:` (item_file)
- `bcq_openended:` (question_file)
- `causal_linkage:` (mode: auto_from_events, max_pairs_auto)
- `anomaly:` (include_person_attributes)

`max_tokens`: the substages temporal_localization / open_qa / mcq_openended /
bcq_openended default to ~4096, tuned for non-reasoning instruct models (Qwen).
For reasoning/"thinking" models (e.g. gcp/google/gemini-3-*) set a larger
explicit `max_tokens` (e.g. 32768) or outputs truncate and items drop mid-bank
(the thinking-token tax). Leave the default for non-reasoning models.

## Examples

Enable QA targets plus event-derived causal linkage on a reasoning model:

```yaml
reasoning:
  reasoning:
    targets: [scene_description, open_qa, mcq_openended, bcq_openended, causal_linkage]
  open_qa:
    question_file: questions/open_qa.json
  mcq_openended:
    item_file: questions/mcq_items.json
  bcq_openended:
    question_file: questions/bcq.json
  causal_linkage:
    mode: auto_from_events
    max_pairs_auto: 32
  max_tokens: 32768        # reasoning/thinking model; keep ~4096 for instruct
```

Dry-run the stage before a real run:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml --container-dry-run'
```

## Gotchas

- A low `max_tokens` on a reasoning model silently empties banks; raise it.
- Question/item/query files must exist for their substages before the run.

## Guardrails

Follow [guardrails.md](guardrails.md). Question, item, and query files are
repository paths; do not commit LLM endpoint keys.
