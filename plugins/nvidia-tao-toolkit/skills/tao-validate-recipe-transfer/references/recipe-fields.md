# Recipe field classification

Every field in the paper's recipe is one of three things. Classify all of them before writing
a config; unclassified fields are how paper values silently leak onto data they were never
meant for.

- **INVARIANT** — carry over unchanged
- **SCALE** — adjust for dataset size / batch size / initialization
- **DERIVED** — recompute from the customer's data; never copy

## General (all task families)

| Field | Bucket | Notes |
|---|---|---|
| Architecture, depth, width | INVARIANT | changing these means you are no longer transferring this recipe |
| Loss formulation & weights | INVARIANT | component weights are tuned jointly with the architecture |
| Optimizer choice (SGD/AdamW) | INVARIANT | switching changes the LR scale entirely |
| Momentum / betas | INVARIANT | rarely worth touching |
| Norm layer type | INVARIANT | but see BN batch-size caveat below |
| Base learning rate | SCALE | linear scaling with effective batch size; then ~10x lower if fine-tuning from a checkpoint |
| Effective batch size | SCALE | batch × grad-accum × num-GPUs. Compute it explicitly, do not assume |
| Schedule length (epochs/iters) | SCALE | **the big one** — see below |
| Warmup length | SCALE | shorter when fine-tuning; proportional to total schedule |
| Weight decay | SCALE | often reduced for small datasets; check whether paper excluded bias/norm params |
| EMA decay | SCALE | tied to total step count; a paper's decay over 90k steps is wrong over 2k |
| LR schedule shape | INVARIANT | cosine/step/poly — keep, but rescale to the new length |
| Gradient clipping | INVARIANT | copy the value; its absence is often unspecified and matters |
| Mixed precision mode | INVARIANT | but re-check for NaN if the loss has exp/log terms |
| Input resolution | DERIVED | from target object size distribution, not from the paper |
| Normalization mean/std | INVARIANT | must match the pretrained backbone, not the new data |
| Class count | DERIVED | check N vs N+1 semantics |
| Class weighting / sampler | DERIVED | from target class balance |
| Augmentation *types* | INVARIANT | subject to the semantic-validity filter below |
| Augmentation *strength* | SCALE | weaker for small datasets and short schedules |
| Frozen stages / backbone | SCALE | freezing more is often right on small datasets |

### Schedule length — the dominant transfer error

The paper's schedule assumes their initialization and their dataset size. If you start from
their released checkpoint on a small customer set, their schedule overfits by a wide margin.

Reason from **total gradient steps seen**, not epochs. Epochs are meaningless across datasets
of different sizes. A reasonable starting point when fine-tuning a released checkpoint on a
small set is a small fraction of the paper's total steps, with LR reduced roughly an order of
magnitude and warmup shortened proportionally — then confirm empirically with a short run and
early stopping on the val curve rather than trusting any fixed number.

State the initialization in the config. If you cannot say in one sentence whether you are
fine-tuning or training from scratch, resolve that before anything else.

## Detection

| Field | Bucket | Notes |
|---|---|---|
| Anchor scales / aspect ratios | DERIVED | recompute by clustering target box dimensions; COCO priors rarely fit specialized domains |
| FPN levels used | DERIVED | tiny objects need finer levels; large-only objects can drop the finest |
| Assigner thresholds (IoU) | INVARIANT | but revisit for extreme aspect ratios |
| NMS IoU threshold | DERIVED | crowded scenes need higher; well-separated objects tolerate lower |
| Score threshold | DERIVED | this is the operating point — choose from the PR curve, never inherit |
| Max detections per image | DERIVED | COCO's 100 is wrong for dense scenes; silently truncates recall |
| Mosaic / mixup | SCALE | helps at scale, hurts small datasets; disable for the last epochs; harmful for very small objects |
| Multi-scale training range | SCALE/DERIVED | center it on the target's actual object size range |

## Segmentation

| Field | Bucket | Notes |
|---|---|---|
| Mask head resolution | DERIVED | raise for thin structures |
| Crop size (semantic seg) | DERIVED | relative to structure size, not copied |
| Class balance / ignore index | DERIVED | ignore-region handling differs between annotation formats and is easily lost in conversion |
| Boundary-aware loss terms | INVARIANT | keep if present |
| Sliding-window inference overlap | DERIVED | from target image size vs crop size |

## Classification

| Field | Bucket | Notes |
|---|---|---|
| Crop ratio / resize policy | INVARIANT | must match pretraining to use the backbone properly |
| Mixup / CutMix alpha | SCALE | typically reduced or removed for small datasets |
| Label smoothing | SCALE | less useful with few classes |
| RandAugment magnitude | SCALE | scale down with dataset size |
| Head init / final layer LR multiplier | SCALE | higher LR on a fresh head is standard when fine-tuning |

## Keypoints

| Field | Bucket | Notes |
|---|---|---|
| Heatmap sigma | DERIVED | relative to object size in pixels |
| Input aspect ratio | DERIVED | from target subject proportions |
| Flip pairs | DERIVED | **must** be remapped for a new skeleton — silently wrong otherwise |
| OKS sigmas | DERIVED | per-keypoint variance is dataset-specific; COCO's values are for human pose |

## Augmentation semantic-validity filter

Augmentation types are INVARIANT only if they preserve label semantics in the target domain.
Check each against the customer's data:

- **hflip** — breaks OCR/text, left/right classes, chirality (L/R hands, mirrored parts),
  and any scene where handedness is the signal
- **rotation** — breaks gravity-dependent scenes, document layout, and orientation-labeled
  defects
- **color jitter / grayscale** — destroys the signal when color *is* the class (ripeness,
  corrosion, dyed samples, status LEDs)
- **mosaic / mixup** — degrades very small objects; unhelpful on small datasets; usually
  disabled for the final epochs
- **random crop** — can crop away the only object; check the empty-target path exists
- **elastic / grid distortion** — invalid for metrology or dimensional inspection

Record the decision and the reason for each. "The paper used it" is not a reason.

## Fields the paper probably did not state

These are the usual `unspecified` entries. Check the official code and the repo issues before
guessing, and log the source:

- whether weight decay excludes bias and norm parameters
- exact warmup shape (linear vs constant) and its length
- gradient clipping value, or its absence
- loss normalization: per-image, per-batch, or per-GPU-then-averaged
- EMA presence and decay
- exact augmentation ordering and per-op probabilities
- whether LR was linearly scaled from a reference batch size
- test-time resize policy and whether TTA was used
- which mAP implementation produced the table
