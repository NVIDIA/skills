# PAIDF Auto-Labeling Scenario Planning

Use this skill when the user asks what auto-labeling workflow to run, which
annotations to generate, how to combine Dockerized stages, or how to design a
new scenario cookbook.

## Workflow

1. Determine the modality, domain, intended consumer, and required annotations.
   If a blocker is unknown, ask a concise question; otherwise state assumptions.
2. Read [intake-and-decisions.md](scenario-planning/intake-and-decisions.md) for the
   planning checklist.
3. Read [annotation-recommendations.md](scenario-planning/annotation-recommendations.md)
   to recommend outputs by use case and domain.
4. Read [pipeline-composition.md](scenario-planning/pipeline-composition.md) before
   proposing stage order, optional stages, or custom combinations.
5. Verify available runner stages and service packages in the current branch
   before claiming a pipeline is executable.
6. Hand off to the cookbook-authoring, prompt-authoring, detection-and-tracking
   stage, and workflow-runner-debugging skills for implementation and execution
   details.

## Planning Output

For a nontrivial scenario, produce:

- The assumed modality/domain/use case.
- Recommended annotation outputs and why they fit.
- The selected stage subset, in the canonical fixed relative order from
  [pipeline-composition.md](scenario-planning/pipeline-composition.md)
  (`super_resolution` -> `detection_and_tracking` -> `referring_expressions` ->
  `captioning` -> `grounding_2d` -> `visual_qa` -> `reasoning` ->
  `training_export` -> `person_attribute_search`).
- Required prompts, question banks, detector/SAM3 choices, endpoints, and model
  cache needs.
- Any unavailable pending services or implementation gaps.
- The dry-run command that should be inspected before execution.

## Examples

Scenario: "traffic-safety video; need MCQ/BCQ QA with reasoning traces and a
training dataset; clips are already high-res (no super-resolution)."

Plan:

- Modality/domain: video / traffic safety; consumer: VL-reasoning training set.
- Stage subset (canonical fixed relative order): `detection_and_tracking` ->
  `captioning` -> `visual_qa` -> `reasoning` -> `training_export`.
- Needs: SAM3 prompts or RF-DETR classes, a dense-caption prompt, an
  event-verification question bank, VLM + LLM endpoints, and a model cache.

Selected stages, then dry-run before execution:

```yaml
pipeline: video
stages: [detection_and_tracking, captioning, visual_qa, reasoning, training_export]
```

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/traffic_safety/configs/pipeline.yaml --container-dry-run'
```

## Guardrails

- Do not treat every use case as requiring every stage. Keep pipelines minimal
  enough to answer the requested annotation goal.
- Do not promise an end-to-end run for a stage whose service package or image is
  not present in the current branch.
- Do not rely on non-PAIDF pipelines, commands, or file locations.
- For custom pipelines, preserve the shared `DataEntry.data_path` scene contract
  and active-media handoff rules.
- There is no single fixed pipeline. Each cookbook selects a subset of the
  canonical stages run in their fixed relative order: `super_resolution` ->
  `detection_and_tracking` -> `referring_expressions` -> `captioning` ->
  `grounding_2d` -> `visual_qa` -> `reasoning` -> `training_export` ->
  `person_attribute_search`. See
  [pipeline-composition.md](scenario-planning/pipeline-composition.md).
- Prefer evidence-first annotations over verdict-first labels. The retired single
  `daft_export` stage no longer exists: DAFT `task/` writing belongs to
  `reasoning`, and dataset aggregation belongs to `training_export`.
