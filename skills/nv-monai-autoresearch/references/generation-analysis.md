# Generation And Synthesis Result Analysis

Use this for paired or unpaired synthesis, reconstruction, translation, and
image generation. Pixel metrics, visual realism, anatomical fidelity, and
downstream utility may disagree.

## Initial Viewer For Every Experiment

Implement the challenge's reusable viewer under `shared/visualization/`, invoke
it after every experiment across all baselines that produces generated or
reconstructed images, and inspect it before interpreting metrics. Show input,
target, prediction, signed residual, and absolute error using fixed display
ranges. For volumetric data, use synchronized axial, coronal, and sagittal
planes at identical coordinates. For a key experiment, add relevant region
masks, representative, worst, shifted-domain, and region-specific failures.
When another stage computes the final score, show every decision-relevant
intermediate and the downstream prediction/result. A visually pleasing sample
gallery without target/error/downstream evidence does not pass.

## Evidence To Produce

- provenance naming the run/checkpoint, split/cases, source modalities,
  preprocessing, evaluator, and fixed display ranges;
- official metrics per case and by meaningful region, anatomy, and foreground;
- input, target, prediction, signed residual, absolute error, and difference
  histograms using fixed display ranges;
- inside-body, boundary, outside-body, lesion, organ, or acquisition-specific
  errors as relevant;
- downstream segmentation, reconstruction, detection, or clinical metrics when
  the challenge uses them;
- training curves and representative, worst, shifted-domain, and failure cases.

When masks, calibration, reconstruction, or other downstream stages affect the
result, show the source prediction, every decision-relevant intermediate, the
final output, and the downstream metric in the same evidence packet. A visually
plausible generated image is not sufficient when the challenge score is
computed after another algorithm.

## Diagnostic Checklist

Consider and disposition applicable items; they are possible explanations, not
the diagnosis or a limit on the agent's reasoning.

- Global scores can hide a small region that dominates downstream failure;
  decompose by region before changing the whole objective.
- Blur with acceptable MSE but poor perceptual or downstream performance can
  indicate averaging, insufficient high-frequency supervision, or misalignment.
- Sharp but anatomically incorrect output suggests hallucination or an
  over-weighted perceptual/adversarial objective.
- Paired misregistration can make faithful predictions look numerically poor
  and can reward blur.
- Intensity bias with preserved anatomy suggests calibration or normalization;
  spatial errors suggest preprocessing, context, or architecture.
- When one region is already strong, start from the strongest compatible
  checkpoint and use localized supervision, consistency, or bounded
  postprocessing rather than relearning the whole image.

Use an oracle mask, calibration, or region replacement when possible to test
whether a suspected error actually explains the downstream metric. Track both
the target region and protected regions after every intervention.
