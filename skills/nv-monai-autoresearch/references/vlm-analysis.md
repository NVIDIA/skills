# Medical VLM Result Analysis

Use this for image-text reasoning, report generation, visual question answering,
grounding, and multimodal retrieval. Lexical agreement is not equivalent to
clinical correctness or image grounding.

## Initial Viewer For Every Experiment

Implement the challenge's reusable case viewer under `shared/visualization/`,
invoke it after every experiment across all baselines, and inspect it before
interpreting aggregate metrics. Show the source image, prompt, reference, model
output, supported findings, unsupported claims, grounding or retrieved
evidence, and error category. For a key experiment, add representative, rare,
worst, disagreement, and safety-critical cases. For volumetric inputs, show the
relevant axial, coronal, and sagittal locations with grounding overlays.
Aggregate text metrics without case-level clinical and visual evidence do not
pass.

## Evidence To Produce

- task metric plus finding-level correctness, calibration, and subgroup results;
- representative, rare, worst, disagreement, and safety-critical examples;
- image, prompt, reference, model output, supported findings, unsupported
  claims, and grounding/retrieval evidence;
- text-only, image-ablation, prompt-variation, and output-format controls;
- error categories for omission, hallucination, negation, laterality, severity,
  anatomy, and uncertainty.

## Diagnostic Checklist

Consider and disposition applicable items; they are possible explanations, not
the diagnosis or a limit on the agent's reasoning.

- Similar lexical scores can conceal opposite clinical errors; inspect
  structured findings and negation/laterality explicitly.
- Strong text-only performance indicates language priors or leakage rather than
  visual reasoning.
- Output changes under harmless prompt wording indicate prompt or formatting
  brittleness, not medical capability.
- Unsupported confident findings require grounding and calibration work; adding
  retrieval can help factuality but can introduce irrelevant evidence.
- Aggregate accuracy can hide rare-condition, demographic, institution, or
  modality failures.
- Better verbosity can improve overlap metrics while increasing hallucination;
  preserve clinical precision and uncertainty behavior.

Use controlled ablations and human-readable error packets before changing the
model. Track clinical benefit, hallucination risk, grounding, calibration, and
runtime for each proposed intervention.
