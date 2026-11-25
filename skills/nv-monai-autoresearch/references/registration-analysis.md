# Registration Result Analysis

Use this for rigid, affine, deformable, multimodal, or longitudinal
registration. Image similarity is not sufficient evidence of anatomical
correspondence.

## Initial Viewer For Every Experiment

Implement the challenge's reusable viewer under `shared/visualization/`, invoke
it after every registration experiment regardless of baseline, and inspect it
before interpreting metrics. Its `visualize_case(...)` inputs name the moving
and fixed images; prediction artifacts name the final warped image and optional
deformation field; ground truth and intermediates carry available annotations,
landmarks, initial alignment, or affine-warped outputs. For volumetric data,
show synchronized axial, coronal, and sagittal planes with fixed orientation,
spacing, intensity windows, and coordinates. Display moving, fixed,
initial/affine-warped, and final-warped images plus before/after overlays or
checkerboards. When segmentations or landmarks exist, also show moving and
fixed annotations plus the warped moving annotation against the fixed
reference. For a key experiment, add displacement magnitude, Jacobian/folding
evidence, and representative, largest-deformation, worst-metric, and
topology-violating cases. A final warped image without the initial alignment
does not pass.

## Evidence To Produce

- official TRE, warped-label overlap, image similarity, or task metric by case;
- moving, fixed, affine-warped, and final-warped overlays/checkerboards;
- landmarks or structure boundaries where available;
- displacement magnitude, Jacobian determinant/folding, smoothness, and inverse
  consistency when applicable;
- results stratified by deformation size, anatomy, modality, and acquisition.

## Diagnostic Checklist

Consider and disposition applicable items; they are possible explanations, not
the diagnosis or a limit on the agent's reasoning.

- Persistent global offset or scale after warping suggests orientation,
  resampling, moving/fixed order, or missing affine initialization.
- Good image similarity with poor landmarks/label overlap can be intensity
  ambiguity or anatomically invalid deformation.
- Good global alignment with local failure suggests the deformable stage,
  receptive field, resolution, correspondence features, or regularization.
- Large-deformation failures often need an affine or coarse-to-fine stage before
  a stronger local network.
- Better overlap with folding or implausible Jacobians is not a clean gain;
  stronger regularization may preserve topology but reduce alignment.
- Lesion or object-volume changes require task-preservation checks even when
  organ registration improves.

Prefer an affine oracle, landmark subset, or controlled warp diagnostic before
changing the full registration system. Preserve topology and downstream task
behavior when optimizing alignment.
