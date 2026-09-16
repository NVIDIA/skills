# Intake And Decisions

Use this checklist to decide what to build. Ask only for information that blocks
a good plan; otherwise make conservative assumptions and state them.

## Intake Dimensions

- Modality: image, video, mixed, still crops, short clips, long recordings.
- Domain: traffic, warehouse/operations, security, employee conduct, robotics,
  generic image, person attributes, or a new domain.
- Consumer: DAFT `task/` artifacts, training-export datasets, retrieval/search
  (Visual Attribute Search), QA evaluation, safety review, experiment
  debugging, or visual inspection.
- Required outputs: captions, tracks, masks, VQA evidence, open QA, MCQ/BCQ,
  events, temporal localization, causal linkage, summaries, active-media SR.
- Quality constraints: visible evidence only, uncertainty requirements, track-ID
  grounding, time localization granularity, overlay needs.
- Runtime constraints: available images, GPUs, model cache, VLM/LLM endpoints,
  network access, expected clip/image volume.

## When To Ask The User

Ask a concise question when any of these are unclear:

- The modality is unknown.
- The desired annotation outputs are unknown and the domain has multiple valid
  interpretations.
- The user asks for production execution but endpoints, GPU, images, or model
  cache are unknown.
- The user asks for identity, protected attributes, or intent/fault inference.

If the user asks for a quick recommendation, do not block on exhaustive intake.
State assumptions and propose a dry-run plan.

## Decision Order

1. Pick annotation goal before stage sequence.
2. Decide whether media enhancement is needed.
3. Decide whether object grounding is needed.
4. Decide whether VQA/question-bank evidence is needed.
5. Decide whether `reasoning` (DAFT `task/` writing) and/or `training_export`
   (dataset aggregation) are needed. The single `daft_export` stage is retired.
6. Decide whether `person_attribute_search` (per-track attributes) is needed;
   it requires `detection_and_tracking` first.
7. Choose prompts, detector config, and runtime knobs.
8. Confirm branch/service availability and dry-run.

## Branch Readiness Check

Before execution claims, inspect current code for:

- `services/workflow_runner/src/workflow_runner/container_runner.py` stage choices.
- Service package directories under `services/` for selected stages.
- Registered build images in each service `pyproject.toml`.
- Cookbook support for required prompts, question banks, and stage args.

If a stage is planned but its service is pending in another MR, say so clearly
and produce a plan/cookbook outline rather than claiming it can run now.
