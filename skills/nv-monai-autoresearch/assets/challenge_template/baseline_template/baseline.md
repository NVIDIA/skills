<!-- miagent-template-source: assets/challenge_template/baseline_template/baseline.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: baseline-direction-v5 -->
# Baseline: <Direction>

Main workspace for one solution direction. The challenge folder owns this file
and its runs; the active session operates them. A baseline is a sustained
solution direction and full pipeline. It may contain many controlled
experiments; it is not one fixed configuration.

Keep component replacements, recipe changes, and configurations under this
baseline while they test the same primary formulation. A materially independent
direction gets a new baseline folder and a recorded trajectory transition.

## Human Steering

Record only human commands that change this baseline's implementation plan or
solution direction. Put queue/resource steering in `dashboard.md` and
experiment-specific steering in that run's `experiment.md`.

| Date | Human Decision | Reason | Applied Action |
|---|---|---|---|
| YYYY-MM-DD | Example: implement multi-GPU training with MONAI transforms | The selected baseline should be strong enough for the approved compute budget | Update Implementation Plan training recipe and environment |

## Implementation Plan

Write this before coding. Use your own thoughts to implement a complicated workflow.

This is the current committed pipeline, not the research backlog. When evidence
changes the pipeline, update the plan to the adopted design and leave the reason
in the experiment Outcome or `trajectory.md`.

You can create a step table to store your implementation plan, for your own record and for human inspection.


Here are some good practices you should know of:
1. Always prefer to reuse the existing codebase of a workflow instead of implement a toy example.
For a composed direction, do this for each borrowed component and
define its role, interface, and reuse/build boundary. Do not replace a mature
implementation with shallow new code without an evidence-backed reason.

2. Cover only the parts that matter for this baseline: preprocessing,
model architecture, training recipe, augmentation, runtime logging/evidence,
postprocessing, and ensemble strategy.

3. The implementation time does not count as GPU compute budget. You can take your time to implement and smoke test on very complicated
baselines and spent time on fixing running issues. Hard to implement baselins should not be considered as budget-wasting

4. The plan should match the target and compute budget. Do not use a toy
single-GPU recipe when the approved baseline needs a serious multi-GPU,
multi-node, larger-patch, fold, or ensemble setup. Profile batch size, patch
size, input pipeline, and distributed strategy. When independent comparisons or
folds fit the resources, run them in parallel instead of serially.

5. The below table are some categories you should consider put in your step table:
| Area | Plan/Changes | Reason |
|---|---|---|
| repo link / source strategy | Pinned upstream repos/checkpoints, local paths, primary substrate, and reuse/build boundary |  |
| environment | Docker / PyTorch / MONAI / other runtime setup | Default |
| preprocessing | resampling, normalization, cropping, patch size, task-specific processing | Example: changed intensity normalization following human steering |
| model architecture | backbone, heads, and planned architecture changes | Example: follow the baseline's proposed novelty |
| pretrained checkpoint | use pretrained ConvNext checkpoints | |
| training recipe | optimizer, learning rate, loss, batch size, schedule, and multi-GPU/multi-node plan. If the source config owns the details, link it and state what MIAgent changed, such as adding multi-GPU support. |  |
| compute / GPU efficiency | Hardware, precision, accumulation/checkpointing, workers/prefetch, caching/I/O, distributed strategy, memory headroom, throughput target, and profiling evidence; link the batch and input/patch values above |  |
| augmentation | flips, crops, intensity transforms, task-specific restrictions | Removed flips due to human steering |
| postprocessing | thresholding, connected components, masks, submission adapter | Added connected components to remove small objects |
| ensemble plan | none / folds / multiple checkpoints / multiple models | Example: default five-fold ensemble from the baseline |


## Internal Exploration

This is the direction's concise, living map of questions that may materially
improve it. Do not prefill a generic checklist during planning or smoke testing.
After the first valid decision-bearing result, use the evidence and your own
judgment to add only task-specific possibilities worth remembering.

- Update this map when a result reveals, changes, or resolves an improvement
  opportunity. New overfitting, distribution shift, preprocessing, optimization,
  or failure-mode findings belong here; an independent pipeline belongs in
  `candidates.md`.
- Each next experiment should test the most valuable current question. When
  several fair, independent comparisons fit the resources, screen them in
  parallel and promote the strongest or most complementary result for deeper
  training and robust confirmation.
- A short screen must be long and representative enough to inform a decision.
  One fold or seed is evidence for promotion, not sufficient evidence to reject
  an entire method family.
- Merge stale or duplicate rows. Keep detailed results in experiment Outcomes
  and consequential shifts in `trajectory.md`.

Before leaving the internal loop, apply the current judgment and reminders in
`state.md`. Record here only the task-specific possibilities that matter.

| ID | Possible Improvement / Question | Why It May Materially Help Now | Current Judgment | Evidence | Revisit Trigger |
|---|---|---|---|---|---|
|  |  |  | <concise causal judgment; name the active experiment, evidence-backed decision, or verified blocker when applicable> |  |  |

## Visualization And Logging For Human Inspection

This is a required part of the baseline. Design task-appropriate visualization
and logging that humans can use for debugging and result analysis, then
implement it once under `shared/visualization/` for all baselines.

Before implementation, answer:

1. Which task-specific views are necessary for performance debugging?
   For segmentation, this may include the raw image, ground-truth and prediction
   overlays, and their differences. For registration, it may include fixed,
   moving, affine-warped, and final-warped images, the deformation field, and
   warped annotations. Read the relevant `references/<task>-analysis.md`.
2. Which training signals are necessary for optimization?
   For example, record training loss, validation loss, and validation metrics
   with TensorBoard or an equivalent logging system.
3. Which evidence will help a human or agent identify likely performance
   bottlenecks?
   For example, show incorrect VLM answers with their prompts and references, or
   visualize the segmentation classes with the lowest Dice scores.

Use those answers to complete this table:

| Component | Details |
|---|---|
| raw data, ground truth, and inference visualization | Examples: segmentation mask overlay on raw data; deformation field and warped image; lesion visualization |
| training and validation curves | Example: TensorBoard, W&B, or local CSV/PNG curves |
| per-case result evidence | Example: per-case metric table, worst-case examples, representative predictions, result viewer path |

## Implementation Acceptance

Complete this after implementation and before non-smoke training. Use
`not checked`, `blocked: <reason>`, `not applicable: <reason>`, or
`done: <evidence link>`. Do not accept source presence or prose as evidence.

| Gate | Status / Evidence |
|---|---|
| upstream paper/source inspected and revision pinned, or from-scratch rationale recorded |  |
| candidate alternatives reviewed and selected direction/escape criteria recorded |  |
| selected candidate's complete committed composition and every defining component implemented; no control-only substitute |  |
| canonical config plus train/resume/infer/evaluate/package commands recorded |  |
| preprocessing, transforms, model, and checkpoint loading smoke-tested |  |
| train/resume and inference/evaluation paths smoke-tested |  |
| every scored or protected requirement covered, or gap made explicit |  |
| runtime telemetry and shared visualization adapter smoke-tested |  |
| distributed/resource plan exercised, or omission justified |  |
| representative GPU profile recorded; bottleneck and selected resource settings justified |  |
| quality-affecting compute changes are controlled experiments, or explicitly deferred |  |
