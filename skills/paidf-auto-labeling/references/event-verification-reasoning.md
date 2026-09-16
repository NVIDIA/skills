# Video PAS with event-verification reasoning

Cookbook:
`cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml`.
Read [run-reference.md](event-verification-reasoning/run-reference.md) before generating commands.

## Workflow

1. Confirm video input, output root, SAM3 weights, model cache, VLM endpoint, LLM
   endpoint, and whether either model is a reasoning/thinking model.
2. Copy the shipped cookbook to `*.local.yaml` and change deployment values only.
   Preserve this node order:
   `detection_and_tracking -> captioning -> event_verification_visual_qa ->
   reasoning -> person_attribute_visual_qa -> person_attribute_search ->
   training_export`.
3. Preserve the event-verification and person-attribute question banks and their
   separate sidecar/state namespaces. The event pass owns DAFT flat QA and
   reasoning traces; the person pass uses `--no-flat-qa-tasks`.
4. Raise Visual QA/reasoning `max_tokens` only for reasoning models and stay
   within the model output limit.
5. Dry-run with `--container-dry-run`. Validate node order, repeated Visual QA
   arguments, images, mounts, endpoints, and PAS input producers. Retry an
   invalid plan at most three times, then report the blocker.
6. Preflight inputs, endpoints, images, and mounts without printing secrets.
7. Present the exact real-run command and resource/output effects, then require
   explicit approval before Docker execution.
8. After a successful run, validate event QA/reasoning outputs, per-track VQA,
   PAS artifacts, and the training export. Do not validate a partial failed run.

## Guardrails

- PAS is assembly-only and receives the LLM endpoint, not VLM arguments.
- Do not merge the two Visual QA nodes or their artifact namespaces.
- Do not remove detection/PAS from this cookbook; use a different cookbook if
  the user wants a reasoning-only workflow.
- Never invent paths, model IDs, endpoint URLs, or credentials.
