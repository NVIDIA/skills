# Post-mortem: diagnosing a failed transfer

A run happened and the result was wrong, disappointing, or suspiciously good. Do **not**
start by rerunning anything, and do not start by tuning hyperparameters. Almost all failed
CV transfers are a skipped verification gate, not a bad learning rate.

## Procedure

### Step 1 — Classify the symptom

Ask for the numbers before theorizing. Specifically:

- Training loss curve — shape, final value, any spikes or NaN
- Validation metric over time — did it rise then fall, plateau early, or never move
- The gap between val and test, if a separate test set exists
- Production or holdout behavior, if the model was deployed
- What the predictions actually look like on ~20 images (this is worth more than any curve)

Then match to a symptom class in the table below. If you have no visualized predictions,
get them before continuing — the failure mode is usually obvious on sight and invisible in
aggregate metrics.

### Step 2 — Walk checks in likelihood order

For the matched symptom, run the checks in the order given. They are ordered by
(frequency × cheapness), not chronologically. Stop when one fires.

### Step 3 — Confirm by repair, not by argument

A hypothesis is confirmed when fixing it changes the number, not when it sounds plausible.
Prefer the cheapest discriminating experiment — usually a single-batch overfit or an eval
re-run, not a retrain.

### Step 4 — Report which gate would have caught it

Close the loop: name the Phase 0/A/G/T/R gate that was skipped. This is what makes the
post-mortem improve the process instead of just fixing one run.

---

## Symptom → ordered checks

### S1. Great validation number, bad production / holdout behavior

The most common and most damaging outcome. In rough order:

1. **Split leakage.** Near-duplicate images across train and val. Run
   `split_leakage_check.py`. Video frames, fixed cameras, burst captures, same
   scene/patient/lot/site. A random split over any of these produces this exact symptom.
2. **Preprocessing mismatch between train and inference.** Letterbox vs stretch, BGR/RGB
   swap, normalization constants, resize-before-vs-after augmentation, different interpolation.
   Run `preprocess_parity.py`. Nothing errors — that is why it survives to production.
3. **Operating point never chosen.** Val mAP is threshold-free; production runs at one fixed
   confidence. A model can have fine mAP and a useless precision/recall balance at 0.5.
   Plot the PR curve and check where the deployed threshold actually sits.
4. **Test set is not production-representative.** Random slice of a convenience export, not a
   held-out site/time/camera. Compare test-set statistics against real production inputs with
   `domain_gap_report.py`.
5. **Post-processing differs.** NMS IoU, max detections, class-agnostic vs per-class NMS,
   letterbox coordinates never un-padded back to original image space.

Note that (5) and the un-padding bug produce boxes that are systematically shifted or scaled —
visually obvious, so check visualizations early.

### S2. Metric is far below the paper, everywhere, and the loss looks fine

Loss descending normally while the metric stays near zero almost always means a
representation mismatch between labels and predictions, not a learning failure.

1. **Class index off-by-one.** Background handling differs between frameworks — some reserve
   index 0, some do not. Symptom: metric near zero or every prediction one class off.
2. **Box coordinate convention.** xyxy vs xywh vs cxcywh, absolute vs normalized, and
   x/y transposition. Conversion between annotation formats is where this enters.
3. **Class name → id mapping mismatch** between train and eval, especially after a format
   conversion that sorted classes alphabetically on one side only.
4. **Silently partial checkpoint load.** Research repos call `load_state_dict(strict=False)`
   routinely. Half the backbone loads, the head is random, nothing warns. Explicitly log
   missing and unexpected keys and assert the counts.
5. **Eval running on the wrong split or wrong annotation file.** Embarrassing, common.
6. **Mask/polygon rasterization** shifted by a half pixel or transposed, for segmentation.

Check (1)–(3) by visualizing *ground truth boxes drawn from the loaded dataloader tensors* —
not from the annotation file. That single visualization catches most of this class.

### S3. Trains fine, but a few points below the paper on the paper's own benchmark

You are still in Phase A. This is a fidelity problem.

1. **Did the released checkpoint reproduce the reported number?** If you skipped that gate,
   go do it now — it isolates eval-side from training-side causes in a couple of hours.
2. **Eval config differences**: resize/crop policy, test-time image size, TTA, NMS settings,
   which mAP implementation (COCO API vs a reimplementation vs the framework's own).
3. **EMA not used at eval**, or used at train but not saved.
4. **Effective batch size differs** and LR was not linearly scaled; gradient accumulation
   counted wrong; multi-GPU loss averaging applied twice.
5. **Weight decay applied to bias and norm params** when the paper excluded them (or vice versa).
6. **Warmup length/shape**, gradient clipping value, loss normalization (per-image vs
   per-batch vs per-GPU-then-averaged).
7. **Seed variance.** Most papers report one run. If the gap is under ~0.5 points, it may not
   exist. Run 3 seeds before spending a week on it.

Items (4)–(6) are the classic unspecified-in-the-paper fields. Check the repo's issues.

### S4. Metric plateaus early and low on customer data, paper's benchmark reproduces fine

Phase A passed, so this is a transfer problem. Now the domain gap report earns its keep.

1. **Schedule inherited from the paper while fine-tuning from a checkpoint.** The single most
   common transfer error. 300 epochs on 2,000 images from a pretrained init is
   overfitting-by-recipe. Expect to cut the schedule hard and drop LR ~10x.
2. **Object size mismatch.** Run `domain_gap_report.py`. If target objects are much smaller
   than source, the input resolution, FPN level assignment, and anchor scales are all wrong,
   and small-object recall will be near zero while large-object metrics look acceptable.
   Check per-size breakdown — aggregate mAP hides this completely.
3. **Anchor/prior mismatch** more generally: aspect ratios inherited from COCO applied to,
   say, long thin objects.
4. **Augmentation too strong for the dataset size**, or semantically invalid for the domain:
   hflip on text/OCR or left/right classes, color jitter on anything where color *is* the
   signal, mosaic on small datasets or small objects, rotation on gravity-dependent scenes.
5. **Class imbalance** far more extreme than the source benchmark; loss is dominated by the
   majority class.
6. **Annotation quality.** Missing boxes are wrong labels, not absent ones — the model is
   explicitly trained to suppress those objects. Train a quick model, then review its
   *high-confidence false positives*; that surfaces missing annotations reliably.
7. **Not enough data for the method.** Some recipes assume scale. A smaller model or heavier
   pretraining may simply be the right answer.

### S5. Loss goes NaN or diverges

1. **LR too high for the new effective batch size** (or scaled the wrong direction)
2. **AMP/fp16 overflow** — check whether the loss has an exp/log, try bf16 or loss scaling
3. **Degenerate boxes in the data**: zero or negative width/height, coordinates outside the
   image, after conversion or after augmentation clipping
4. **Missing or wrong-value gradient clipping** relative to the paper
5. **Empty images / empty targets** not handled by the loss (some repos assume ≥1 object)
6. **Warmup removed or shortened** when porting the config

### S6. Result is suspiciously good

Treat this as a failure until proven otherwise. It is the same checklist as S1, run
adversarially:

1. Split leakage — run the script, do not eyeball it
2. Test set contamination via augmented copies, or the test set having been used for
   checkpoint selection (that makes it a val set, and its number is optimistic)
3. Label leakage: a feature that encodes the label — filename patterns, image resolution
   differing by class, a watermark or timestamp that correlates
4. Metric computed over the wrong subset, or averaging that hides empty classes
5. Baseline was undertrained or unfairly configured, making the delta look large

### S7. Works at one batch size / GPU count but not another

1. LR not scaled with effective batch size
2. BatchNorm with a small per-GPU batch — needs SyncBN or a switch to GroupNorm
3. Loss reduction applied per-GPU then averaged again
4. Gradient accumulation interacting with the LR schedule stepping per-iteration
5. Dataloader worker count changing augmentation RNG behavior

---

## Output format for a post-mortem

```
# Post-mortem: <run identifier>

## Symptom
What was observed, with numbers.

## Root cause
The confirmed cause, with the evidence that confirmed it.

## Evidence trail
Checks run, in order, and what each ruled out.

## Fix
What to change, and the expected effect on the number.

## Gate that would have caught this
Which phase gate was skipped, and the specific check.

## Residual uncertainty
What is still unverified.
```

Keep "Gate that would have caught this" honest even when the answer is unflattering. That
section is the entire reason to write a post-mortem rather than just fixing the run.
