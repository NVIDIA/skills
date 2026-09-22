---
name: tao-validate-recipe-transfer
description: Port a published computer vision paper's official code and training recipe onto a customer's own dataset, or diagnose why such a transfer produced bad numbers. Use this whenever someone wants to reproduce a CV paper, run a paper's repo on their own images, fine-tune a published detection/segmentation/classification/keypoint model on customer data, adapt a training recipe to a new dataset, or figure out why a fine-tuned vision model scores well on validation but fails in production. Also use for post-mortems on any failed or disappointing CV training run, and whenever a user mentions mAP that looks too good, a model that "worked in training but not in deployment", or transferring hyperparameters from a paper to their own data. Trigger even if the user only says "train a model on my dataset" and a published architecture or repo is involved.
license: Apache-2.0
compatibility: >-
  Methodology skill — no container of its own. Three of the four scripts need only Python 3
  with numpy and Pillow; render_status.py is stdlib-only. They run inside a research repo's
  own environment without disturbing it. Phases A and R execute the ported repo's
  training/eval, so they inherit that repo's requirements (typically docker +
  nvidia-container-toolkit + an NVIDIA GPU).
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
tags:
  - computer-vision
  - training
  - evaluation
  - reproducibility
  - data-quality
allowed-tools: Read Bash Write
---

# CV Recipe Transfer

Port published CV work onto customer data without inheriting the silent bugs that make
the resulting numbers meaningless.

## The core idea

This work splits into two phases with **fundamentally different epistemic status**:

- **Phase A (Port)** — get the official code running faithfully. There is external ground
  truth here: the authors' released checkpoint and reported number.
- **Phase B (Transfer)** — move the recipe onto customer data. There is **no** external
  ground truth. The paper's number stops being meaningful the moment the dataset changes.

Everything verifiable must be verified in Phase A, because the safety net disappears at the
boundary. Once you are in Phase B, an implementation bug and a genuine domain mismatch
produce identical symptoms: a mediocre number with a clean-looking loss curve.

Two consequences that drive everything below:

1. Never skip the Phase A gate to "save time." It costs hours and it is the only thing that
   licenses trust in every number that follows.
2. Stop treating the paper's number as the target once the dataset changes. The target is
   *beat the strongest cheap baseline on the customer's own test set, measured at a
   deliberately chosen operating point.*

## Mode selection

Read the request and pick a mode. Say which one you picked and why.

**Post-mortem mode** — a run already happened and the result was wrong, disappointing, or
suspiciously good. Go to `references/postmortem.md`. Do not start by rerunning anything.
Walk the checks in likelihood order and find the gate that was skipped.

**Forward mode** — no run yet, or starting over. Work Phase 0 → A → G → T → R → E below.

**Audit mode** — a pipeline exists and works, someone wants it checked before it goes to a
customer. Run the Phase A gate and the Phase G report against the existing artifacts, then
the parity and leakage scripts. Skip the porting work.

If the request is ambiguous, ask which of the three it is before doing anything expensive.

---

## Phase 0 — Feasibility gate

Cheap, and it prevents the most expensive class of mistake. Do this before reading any code.

Ask and answer, in writing:

1. **Does this need training at all?** Zero-shot and promptable models cover a lot of what
   used to require a fine-tune — Grounding DINO, SAM-family, OWL-ViT, CLIP, and plain
   pretrained COCO detectors. If the customer's classes overlap COCO/LVIS vocabulary, or the
   task is "find the obvious object," measure a pretrained baseline **first**. A surprising
   fraction of three-week fine-tuning projects are answered by an afternoon of prompting.
2. **Is this the right paper?** Papers get picked by benchmark rank, but rank on COCO
   predicts very little about performance on 800 images of factory parts. Check that the
   paper's benchmark resembles the customer's data on the axes in Phase G.
3. **License.** Research-only, non-commercial, and AGPL clauses are common in CV repos and
   this is customer-facing work. Check the repo license, the weights license (often
   different from the code), and the license of any pretrained backbone. This is a hard gate
   — flag it now, not after three weeks of work.
4. **Is there enough labeled data, and does a test set exist that reflects production?** If
   the only data is a random export, the test set will be optimistic no matter what you do.

Record the answers. If the feasibility gate fails, say so plainly and stop — that is a
successful outcome of this skill, not a failure.

---

## Phase A — Port

Goal: prove the code, the data pipeline, the preprocessing, and the metric implementation
are all faithful. Read `references/port-gate.md` for the detailed procedure.

The gate, in short:

1. **Pin the commit.** Repos drift and released code routinely differs from the paper. Pin
   the hash. Then read the repo's issue threads — undocumented hyperparameters, known
   discrepancies with the paper, and "why can't I reproduce Table 2" live there. Assume the
   code and the paper disagree somewhere until proven otherwise.
2. **Build the environment.** Custom CUDA ops (deformable attention, DCN, custom NMS) in
   older research repos are the single biggest time sink specific to CV. Detect them early
   and containerize. Do not fight the host toolchain.
3. **THE GATE: run their checkpoint through their eval on their benchmark and match the
   reported number.** This single check validates environment, data pipeline, preprocessing,
   and metric implementation without training anything. Hours, not days.
   - Match within ~0.2 points: pass.
   - Off by 0.5–2 points: something in preprocessing or eval config is wrong. Do not proceed.
   - Off by a lot: wrong checkpoint, wrong class mapping, or a silently partial state-dict
     load. See the failure atlas.
4. **Validate the training path** with a short run on a benchmark subset. You are checking
   loss-curve *shape* and that it descends as expected, not the final number.
5. **Scan for hardcoded benchmark assumptions.** Official code is written for exactly one
   dataset. Class counts, normalization constants, anchor priors, path structure, and
   sometimes image size are baked in — occasionally inside the model definition rather than
   the config. Finding all of them is most of the porting work. `references/port-gate.md`
   has the grep patterns.

Do not enter Phase B until step 3 passes. If it cannot be made to pass, say so and report
the gap — "this repo does not reproduce its own paper" is a legitimate and valuable finding.

---

## Phase G — Domain gap report

Make the gap measurable rather than assumed. Run:

```bash
python scripts/domain_gap_report.py --source <source_annotations> --target <target_annotations> --out gap_report.md
```

It compares the source benchmark and the customer set on image count, resolution and aspect
distribution, objects per image, class balance, and — most importantly — **object size as a
fraction of image area**.

That last axis drives more recipe decisions than any other. COCO objects average a few
percent of image area. If the customer's aerial, microscopy, or inspection imagery puts them
at 0.1%, that immediately implicates input resolution, FPN level assignment, anchor scales,
and rules out mosaic-style augmentation. The script emits these as explicit flags with
reasons, not generic warnings.

The magnitude of the gap on each axis tells you which recipe fields in Phase T are at risk.
Carry the flags forward — they are the justification for every deviation from the paper.

Also run split hygiene here, before any training:

```bash
python scripts/split_leakage_check.py --splits train=<dir> val=<dir> test=<dir> --report leakage.md
```

Near-duplicate images across splits are the single biggest source of inflated CV numbers.
Video frames sampled at 5fps, the same scene from a fixed camera, the same patient, the same
production lot — a random split gives 0.95 mAP and a model that collapses on deployment.
Detectable via perceptual hashing and group metadata; almost nobody checks. If this skill
does one thing well, make it this.

---

## Phase T — Transfer the recipe

The intellectual core. Classify **every** field in the paper's recipe into one of three
buckets before writing a config. `references/recipe-fields.md` has the full field-by-field
table; the buckets are:

- **Transfer-invariant** — architecture, loss formulation, optimizer choice, most
  augmentation *types*. Carry over unchanged.
- **Scale-dependent** — schedule length, LR (via batch size and linear scaling),
  augmentation strength, weight decay, EMA decay. Must be *adjusted* for dataset size, and
  the direction of adjustment is usually predictable.
- **Dataset-derived** — anchor and box size priors, input resolution, class weighting, FPN
  level assignment, NMS/score thresholds. Must be **recomputed** from the customer's data,
  never copied.

### The trap that catches almost everyone

If you initialize from the authors' released checkpoint — which you almost always should on
a small customer dataset — **their schedule is wrong for you by a large factor.** Their
recipe assumes ImageNet init or from-scratch training on 118k images. You are fine-tuning on
2,000. Copying the paper's 300-epoch schedule is a common, expensive, and entirely
predictable mistake. Expect: much shorter schedule, lower LR (often 10x), shorter warmup,
possibly a frozen stem, and weaker augmentation.

State the initialization explicitly in the config and derive the schedule from it. If you
cannot say in one sentence whether you are fine-tuning or training from scratch, stop and
resolve that first.

### Preprocessing is a single artifact

Write the preprocessing spec — resize policy (letterbox vs stretch), channel order,
normalization constants, augmentation ordering — into one file that the training pipeline,
the eval path, and the export path all read. Then verify numerically:

```bash
python scripts/preprocess_parity.py --config <preproc.yaml> --image <sample.jpg> --paths train,eval,export
```

Preprocessing mismatch between training and inference is the failure mode that actually
kills deployments: excellent validation mAP, garbage in production, and a long debugging
loop because nothing ever errors. Assert tensor equality rather than trusting that two code
paths do the same thing.

---

## Phase R — Run

Apply the verification ladder from `references/verification-ladder.md` before spending real
GPU hours. Abbreviated:

1. Shape and dtype trace; parameter count against the paper's stated count
2. Loss at initialization equals its analytic value
3. Gradient flow — every parameter that should get gradient does, nothing silently frozen
4. **Overfit a single batch to near-zero loss.** If it cannot, the pipeline is wrong, full stop
5. Overfit ~100 images; confirm the eval path agrees with the training path on that data
6. Short schedule at reduced scale; check trend
7. Full run

Fail early, fail cheap. Most disasters are a wrong pipeline trained for three days.

Alongside the full run, keep a **regression suite**: the pretrained baseline from Phase 0 and
any prior best model, evaluated on the same frozen test set. Log dataset version, config, and
commit hash together so any checkpoint is reproducible.

---

## Phase E — Evaluate and report

**Choose an operating point.** Everyone reports mAP@50-95 and then deploys at a confidence
threshold nobody chose deliberately. Pick the threshold from the PR curve against the
customer's actual precision/recall tradeoff — the cost of a miss versus a false alarm is a
business question, so ask it — and report metrics *at that threshold* alongside the aggregate.

**Break down, don't average.** Per-class and per-slice (object size, lighting, camera,
site, time-of-day). A single mAP hides the fact that the model fails entirely on the one
class the customer cares about.

**Error analysis.** For detection, separate classification errors, localization errors,
duplicates, background false positives, and missed detections — they have different fixes.
High-confidence false positives are often *missing annotations*, not model errors: a missing
box is not a missing label, it is a wrong label, and the model was explicitly taught to
suppress that object. Feed these back into annotation QC.

**Status artifact — emit one at every evaluation, including mid-run.** Write `status.json`
and render it:

```bash
python scripts/render_status.py --status status.json --out status.html
```

It shows exactly two things: the delta against the paper on top, and the best result so far
against its baseline below. `assets/status.example.json` is the schema by example. Two rules
that make it honest:

- A metric the paper never published gets `"paper": null`, which renders as `n/p` with no
  delta. Never fill that column with a number from a different variant or a blog post.
- If the customer dataset has no published result — the normal case — say so in `callout`
  and name what the comparison is actually against. "vs the previous run" and "vs a published
  result" are different claims.

Mid-run is a first-class state: label the cards `still climbing` and say so in the footnote,
so nobody quotes a partial number as final.

**Report structure** — use this template:

```
# <Task> recipe transfer: <paper> → <customer dataset>

## Outcome
One paragraph. The number, at the chosen operating point, versus the baseline.

## Phase A verification
Did the official checkpoint reproduce the reported number? Exact figures.

## Domain gap
The measured gap and which recipe fields it forced us to change.

## Recipe deviations
Table: field | paper value | our value | bucket | reason.

## Results
Per-class and per-slice at the chosen threshold. PR curve. Regression vs baseline.

## Error analysis
Failure categories with counts and examples.

## Known risks
What we could not verify, and what would change the conclusion.
```

The "Known risks" section is not optional. It is the difference between a number a customer
can act on and a number that will embarrass someone in three months.

---

## Reference files

Read these as needed rather than upfront:

- `references/postmortem.md` — **diagnose a failed run.** Symptom-ordered, cheapest checks
  first. Start here in post-mortem mode.
- `references/port-gate.md` — Phase A detail: commit pinning, CUDA op survival, checkpoint
  gate procedure, hardcoded-assumption grep patterns.
- `references/recipe-fields.md` — the full field classification table, per task family.
- `references/verification-ladder.md` — the seven rungs, with the expected analytic values.
- `references/failure-atlas.md` — symptom → cause → check, for the ~25 recurring failures.
- `references/stacks.md` — framework-specific notes: raw research repo, MMDetection /
  Detectron2, Ultralytics, timm / torchvision, TAO. Read only the relevant one.

## Scripts

- `scripts/domain_gap_report.py` — quantified source-vs-target gap with risk flags
- `scripts/split_leakage_check.py` — perceptual-hash near-duplicate detection across splits
- `scripts/preprocess_parity.py` — numerical assertion that train/eval/export preprocessing agree
- `scripts/render_status.py` — the Phase E status artifact: paper delta + best result so far

Run them with `--help` for options. The first three are dependency-light (numpy + Pillow) and
`render_status.py` is stdlib-only, so all of them run inside a research repo's container
without disturbing its environment.

## Honest framing

For a customer's private task **there is no SOTA.** There is no leaderboard for "find defects
in this customer's parts." There is only the baseline and your number. So do not promise
state-of-the-art; promise a well-validated result with no methodology bugs, defensible to
someone who wants to poke at it.

That is worth more than it sounds, because a large share of impressive-looking fine-tuning
results are inflated by test leakage, evaluating on the training distribution, or comparing
against a deliberately weak baseline. Those errors are mechanical, and this skill exists to
catch them.
