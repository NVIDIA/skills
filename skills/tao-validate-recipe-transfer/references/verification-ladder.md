# The verification ladder

Gate progression on cheap checks before spending GPU hours. Each rung costs seconds to
minutes; the full run costs days. Most reproduction disasters are a wrong pipeline trained
for three days.

Run these in order. Do not skip ahead because a rung "obviously" passes.

## Rung 1 — Shape, dtype, and parameter count

Trace one batch through the model. Print shapes at every stage boundary.

Then compare total parameter count against the paper's stated count:

```python
n = sum(p.numel() for p in model.parameters())
print(f"{n/1e6:.2f}M params")
```

A mismatch of more than ~1% means you misread the architecture — wrong depth, wrong width
multiplier, wrong number of heads, or a variant config. This catches an enormous number of
architecture misreadings for essentially zero cost.

## Rung 2 — Loss at initialization equals its analytic value

An untrained model's loss is predictable. If it does not match, the loss, the labels, or the
head initialization is wrong.

| Loss | Expected at init |
|---|---|
| Cross-entropy, C classes | ln(C) — e.g. 4.605 for 100 classes, 2.303 for 10 |
| Binary CE, balanced | ln(2) ≈ 0.693 |
| Focal loss | below BCE; depends on alpha/gamma — compute it analytically for your settings, and check the prior-bias init is applied |
| Dice | ≈ 1 − (2·p̄·ȳ)/(p̄+ȳ); ≈1.0 for a sparse mask with random preds |
| L1/GIoU box loss | compute the expected value for random boxes rather than eyeballing |

For detection heads with a focal-loss prior bias (the standard `-log((1-π)/π)` init), verify
the bias was actually applied — if it was not, early training is unstable and the run may
"work" while converging to a worse point.

## Rung 3 — Gradient flow

Every parameter that should receive gradient does; nothing is silently detached or frozen.

```python
loss.backward()
no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
zero_grad = [n for n, p in model.named_parameters()
             if p.grad is not None and p.grad.abs().sum() == 0]
print("no grad:", no_grad[:20])
print("zero grad:", zero_grad[:20])
```

Expected: frozen stages appear (intentionally); nothing else does. A silently detached branch
trains to a plausible-looking loss while a whole component does nothing.

Also verify frozen-BN behavior matches intent — a "frozen" backbone with BN in train mode is
still updating running statistics.

## Rung 4 — Overfit a single batch

Take one batch of 2–8 images. Train on it, evaluating on the same batch, until the loss
approaches zero. Disable augmentation and any regularization.

**If it cannot reach near-zero loss, the pipeline is wrong. Full stop.** Do not proceed, do
not tune the learning rate, do not train longer. Something in the label pipeline, loss, or
target assignment is broken.

This is the highest-value rung on the ladder. It takes about a minute and it catches the
majority of implementation bugs.

## Rung 5 — Overfit ~100 images, then evaluate on them

Same data for train and eval. The metric should be very high — near-perfect for
classification, high mAP for detection.

If the loss goes to zero but the *metric* stays low, the bug is in the **eval path**, not the
training path: post-processing, coordinate conversion, class mapping, NMS, or the metric
implementation itself. This rung exists specifically to separate those two, and it is the one
people skip most often.

Visualize predictions here. Twenty images tell you more than any curve.

## Rung 6 — Short schedule at reduced scale

A fraction of the full schedule on a data subset. Check:

- loss curve shape through warmup and into the main schedule
- component losses at sane relative magnitudes
- a mid-training eval producing a non-trivial metric
- throughput and memory, to project the cost of the full run

If the paper published a curve, compare shape rather than absolute values.

## Rung 7 — Full run

Only now. Log dataset version, config, and commit hash together. Checkpoint frequently,
evaluate on a schedule, and keep the regression suite (pretrained baseline and prior best) on
the same frozen test set.

---

## Applying the ladder in post-mortem mode

The ladder is also a diagnostic bisection. Given a failed run, find the highest rung that
still passes:

- Fails rung 4 → label pipeline, loss, or target assignment
- Passes 4, fails 5 → eval path, post-processing, or metric
- Passes 5, fails 6 → optimization: LR, schedule, augmentation strength
- Passes 6, underperforms at 7 → data quality, dataset size, or genuine domain gap

That bisection turns a vague "the numbers are bad" into a bounded hypothesis space in about
fifteen minutes of compute.
