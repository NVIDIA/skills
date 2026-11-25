# Segmentation Result Analysis

Use this to plan evidence and diagnose semantic, instance, or small-structure
segmentation. Adapt it to the evaluator; do not assume Dice alone defines
success.

## Initial Viewer For Every Experiment

Implement the challenge's reusable viewer under `shared/visualization/` and
invoke it after every experiment that produces segmentations, regardless of
baseline. Its `visualize_case(...)` call accepts image NIfTI path(s), prediction,
optional ground truth, case ID, metrics, and output directory; its
`visualize_run(...)` path builds the indexed run viewer from the same normalized
fields. For nnU-Net baselines, adapt available log parsing and visualization tools
to the shared renderer for training curves, orthogonal case rendering, and
indexed HTML. Inspect the viewer before interpreting metrics. Provide an
all-case index.
For 3D data, show synchronized axial, coronal, and sagittal views at the same
location for each selected case and class; for 2D data, show the corresponding
full-image panels. Each view must show these panels side by side:

```text
original image
original image + ground truth overlay
original image + prediction overlay
original image + error overlay with TP, FP, and FN in distinct labeled colors
```

Use correct orientation, spacing/aspect ratio, fixed intensity windows, a fixed
color legend, and identical slice coordinates across panels. Show every input
modality when relevant. The initial viewer includes fixed reference and
representative cases. For a key experiment, expand it with best, median, worst,
empty-label, small-target, and failure-specific cases plus relevant
intermediates. Interactive slice/class controls are preferred; a static viewer
must select the largest-target and largest-error locations in all three planes.
An axial-only montage does not pass.

Before trusting an apparent image/mask mismatch, prove that the source image and
label, network input and target, saved prediction, and displayed overlay refer
to the same case and coordinate system. For multimodal input, show every
modality rather than silently choosing one channel. If train-set predictions
are empty, noisy, or implausible, verify checkpoint head/decoder loading and
overfit one case or foreground-positive patch before changing the objective or
architecture.

## Evidence To Produce

- provenance naming the run/checkpoint, split/cases, input modalities,
  preprocessing, evaluator, orientation, and display ranges;
- per-case and per-class Dice plus the official surface metric;
- target presence, ground-truth/predicted volume, and object-size strata;
- raw image, ground truth, prediction, FP/FN, and boundary overlays in the
  correct orientation and spacing;
- fixed reference, best, median, worst, empty-label, and failure-specific cases;
- training/validation curves, patch foreground frequency, and inference output
  geometry.

The training curves belong in the same run viewer as the held-out visual and
metric evidence. Clearly label nnU-Net pseudo-Dice as an internal validation
signal rather than the official evaluator result. When training resumed, merge
all logs by epoch and let the newest complete record win so duplicated epochs
do not distort the plot.

Do not show an unexplained crop or montage without the source case, checkpoint,
mask semantics, and final prediction.

## Diagnostic Checklist

Consider and disposition applicable items; they are possible explanations, not
the diagnosis or a limit on the agent's reasoning.

- Good Dice with poor HD95 often indicates isolated false positives or one bad
  boundary; component filtering may help but can delete true small targets.
- Small or absent structures with low recall suggest resolution, crop, class
  mapping, foreground sampling, or loss imbalance before model capacity.
- Systematic left/right or anatomical confusion requires checking orientation,
  label mapping, and unsafe flips.
- Large predicted volume with low precision suggests threshold/calibration,
  background sampling, or weak localization; tiny volume with low recall
  suggests the opposite.
- A few catastrophic cases require site/modality/size subgroup inspection, not
  blind global tuning.
- Improving loss without official-metric or visual improvement can indicate a
  loss/evaluator mismatch.

Choose interventions only after identifying the failure pattern. Track the
classes, sizes, and previously strong cases that the intervention could harm.
