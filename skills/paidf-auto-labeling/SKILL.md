---
name: paidf-auto-labeling
description: >-
  Use when a user needs to get started with PAIDF Auto-Labeling, plan a
  scenario, run or debug a shipped cookbook, author prompts or cookbooks,
  migrate a pipeline, or configure a stage. Confirm critical inputs (data
  path, output path, endpoints) and ask when any are missing. This is a
  router: read the matching reference instead of inventing a workflow.
license: Apache-2.0
owner: NVIDIA
service: physical-ai-data-factory
version: 1.1.0
reviewed: 2026-08-25
author: "NVIDIA <opensource@nvidia.com>"
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  tags: [getting-started, onboarding, new-domain, quickstart, new-use-case]
---

# PAIDF Auto-Labeling

Use this skill when a user wants to kick off PAIDF Auto-Labeling on their
own data, domain, or use case, or when the request matches a shipped cookbook,
stage, authoring, or migration task. This is a router: sequence the specialized
references instead of duplicating their detail.

## Routing (Read First)

| Request looks like | Read |
| --- | --- |
| New user, clean checkout, first validated run, "how do I get started" | This file, then the matching reference below |
| Choose annotation targets / stage subset for a domain | [`references/scenario-planning.md`](references/scenario-planning.md) |
| Create, review, or adapt a cookbook | [`references/cookbook-authoring.md`](references/cookbook-authoring.md) |
| Write or adapt VLM/LLM prompts or question banks | [`references/prompt-authoring.md`](references/prompt-authoring.md) |
| Migrate an existing annotation repo into this one | [`references/pipeline-migration.md`](references/pipeline-migration.md) |
| Run the video data augmentation cookbook | [`references/video-data-augmentation.md`](references/video-data-augmentation.md) |
| Run or choose an EPAS / PAS cookbook | [`references/event-and-person-attribute-search.md`](references/event-and-person-attribute-search.md) |
| Run event-verification reasoning | [`references/event-verification-reasoning.md`](references/event-verification-reasoning.md) |
| Debug an already-integrated workflow | [`references/workflow-runner-debugging.md`](references/workflow-runner-debugging.md) |
| Implement or review a new stage or Dockerized service | [`references/workflow-stage-integration.md`](references/workflow-stage-integration.md) |
| Configure or debug one production stage | The matching file under [`references/stages/`](references/stages/) |

Stage references: [super-resolution](references/stages/super-resolution.md),
[detection-and-tracking](references/stages/detection-and-tracking.md),
[captioning](references/stages/captioning.md),
[visual-qa](references/stages/visual-qa.md),
[reasoning](references/stages/reasoning.md),
[person-attribute-search](references/stages/person-attribute-search.md),
[grounding-2d](references/stages/grounding-2d.md),
[referring-expressions](references/stages/referring-expressions.md),
[training-export](references/stages/training-export.md).

## Instructions

1. Confirm the critical run inputs with the user before doing anything else, and
   ask a concise question whenever one is missing or ambiguous - never guess or
   silently invent a default. At minimum confirm: input data path, output path,
   VLM/LLM endpoint URLs and model names, model cache path, GPU ids, and (for
   reasoning-capable models) the `max_tokens` cap. Restate the confirmed values
   back to the user before the first execution.
2. Verify the environment: repository cloned, `make` targets available, the
   model cache path exists, the VLM/LLM endpoints are reachable, and a GPU is
   available. State any missing prerequisite as a blocker instead of assuming it.
3. Run a shipped example first to confirm the stack works end to end before
   customizing. Pick the closest operator pipeline - video data augmentation,
   event-and-person-attribute-search, or event-verification-reasoning - and
   run its committed cookbook. Use the matching operator reference.
4. Plan the target scenario: define modality, domain, intended consumer, and
   required annotations, and get a minimal stage subset. Use
   [scenario-planning](references/scenario-planning.md).
5. Adapt the closest shipped cookbook to the new domain rather than authoring
   from scratch. Use [cookbook-authoring](references/cookbook-authoring.md).
6. Author the domain prompts and question banks. Use
   [prompt-authoring](references/prompt-authoring.md).
7. Configure the per-stage settings for the domain (detector classes or SAM3
   prompts, endpoints, windowing, `max_tokens`). Use the relevant stage
   reference, starting with
   [detection-and-tracking](references/stages/detection-and-tracking.md).
8. Dry-run the adapted cookbook, then execute and validate the outputs. Use
   [workflow-runner-debugging](references/workflow-runner-debugging.md).

Adopting an existing external annotation or dataset-generation repository into
PAIDF instead of starting from a shipped cookbook is a migration task; use
[pipeline-migration](references/pipeline-migration.md) for that path.

## Examples

New user, new domain: "I cloned the repo and have my own warehouse-safety video.
How do I produce auto-labels for my domain?"

Guided path:

- Confirm env (model cache, VLM/LLM endpoints, GPU), then prove the stack on a
  shipped example before customizing:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

- Plan the domain ([scenario-planning](references/scenario-planning.md)) -> subset
  `detection_and_tracking -> captioning -> visual_qa -> reasoning -> training_export`
  (add `grounding_2d` for caption→boxes or `referring_expressions` for boxes→phrases;
  use [grounding-2d](references/stages/grounding-2d.md) /
  [referring-expressions](references/stages/referring-expressions.md)).
- Copy the closest cookbook to `cookbooks/warehouse_safety/configs/pipeline.yaml`
  and adapt inputs, detector classes/SAM3 prompts, prompts, and question banks.
- Dry-run the new cookbook, then run for real and validate outputs:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/warehouse_safety/configs/pipeline.yaml --container-dry-run'
```

## Guardrails

- Do not guess or fabricate the critical inputs enumerated in step 1; if any is
  missing or ambiguous, ask the user and confirm before executing.
- Do not customize a cookbook before a shipped example runs clean; a broken base
  makes domain debugging ambiguous.
- Keep the first custom pipeline minimal - only the stages needed for the
  requested annotations - and expand later.
- Verify that every selected stage's service package and image exist in the
  current branch before promising an end-to-end run.
- Do not put secrets, tokens, or absolute home paths in committed cookbooks; use
  placeholders such as `<model-cache>` and env vars for endpoint keys.
- For reasoning-capable models (for example Gemini 3 Flash), raise `max_tokens`
  on the `visual_qa` and `reasoning` LLM substages to avoid the thinking-token
  tax; keep the default cap for non-reasoning models.
- Do not rely on non-PAIDF pipelines, commands, or file locations. A first run
  must be reproducible through `workflow-runner:main` inside this repo.
