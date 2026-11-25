# Representation Learning Result Analysis

Use this for self-supervised, foundation-model, shared-encoder, and multitask
benchmarks. One strong downstream result does not establish a strong shared
representation.

## Initial Viewer For Every Experiment

Implement the challenge's reusable evaluation viewer under
`shared/visualization/`, invoke it after every representation experiment across
all baselines, and inspect it before interpreting aggregate metrics. Start with
a coverage index for every required downstream task and show the shared
checkpoint plus each adapter/recipe. Use each downstream task's visualization
contract for its outputs: segmentation overlays, registration correspondence,
generation errors, classification examples and confusion, or other
task-appropriate evidence. For a key experiment, add representative and failure
cases for every affected task. Feature maps, embeddings, and nearest neighbors
are supplementary and must be linked to observable downstream behavior.

## Evidence To Produce

- every required downstream task, its weight, protocol, and best trusted result;
- frozen-probe, finetuned, spatial, and global tasks separated explicitly;
- modality, site, cohort, label-volume, and demographic subgroup results;
- feature variance/collapse checks, embedding distributions, and checkpoint
  compatibility across task interfaces;
- compute/runtime and evidence that the same legal checkpoint was used.

Maintain a compact challenge-wide coverage table in the dashboard Result Table
when the benchmark has several required downstream tasks. For each task,
identify the shared checkpoint, adapter or head, current result, evaluation
data, and missing proof. Keep the complete multi-task requirement explicit in
`state.md`; do not let progress on one task stand in for complete representation
coverage.

## Diagnostic Checklist

Consider and disposition applicable items; they are possible explanations, not
the diagnosis or a limit on the agent's reasoning.

- Strong finetuning with weak frozen probes may indicate task adaptation rather
  than transferable representation quality.
- Strong classification/regression with weak segmentation suggests loss of
  spatial detail; the reverse can indicate insufficient global semantics.
- A task-specific input stem, fusion path, or copied encoder may violate a
  shared-checkpoint contract even when individual tasks run.
- Improvements on dominant tasks must be weighed against task weights and
  regressions on other required capabilities.
- Site or demographic separability may help acquisition prediction while
  harming invariance or fairness.
- A weak downstream result can come from the adapter or finetuning recipe rather
  than the pretrained checkpoint; compare a strong downstream-only control.

For shared-checkpoint challenges, assess every backbone or objective change
against all task families. A targeted run may use a proxy, but its design must
state when complete coverage will be checked and what capabilities must remain.
Treat a weak downstream result as a possible adapter or recipe failure before
rejecting the shared representation; compare against a strong task-specific
pipeline and preserve the required shared-encoder contract.
