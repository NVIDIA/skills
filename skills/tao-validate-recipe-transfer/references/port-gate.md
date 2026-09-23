# Phase A: the port gate

Goal: prove the code, data pipeline, preprocessing, and metric implementation are faithful
**before** touching customer data. This is the last point at which external ground truth
exists.

## A0. License check (hard gate, do first)

Three separate licenses, often different:

- **Code license** — the repo's LICENSE file. Watch for research-only, non-commercial, and
  AGPL. AGPL in particular is a problem for hosted customer deployments.
- **Weights license** — frequently stricter than the code and stated only in the README or a
  model-zoo table. "Non-commercial research use" on weights is common.
- **Backbone / pretraining data license** — an ImageNet-21k or proprietary-data-pretrained
  backbone can carry its own restrictions downstream.

Also check any dataset the recipe depends on for calibration or pretraining. Report the
finding before doing any work; a license blocker discovered in week three is a wasted project.

## A1. Pin the commit

```bash
git clone <repo> && cd <repo>
git log -1 --format="%H %ci %s"   # record this in the run config
git checkout <hash>
```

Record the hash next to the config and dataset version. Any result that cannot name its
commit is not reproducible.

**Then read the issues.** This is not optional and it is the highest-value hour in Phase A.
Search the repo's issues and PRs for: `reproduce`, `mAP`, `lower than paper`, the table
number you care about, and the config filename. What you are looking for:

- Undocumented hyperparameters the authors gave out in replies
- Acknowledged differences between the released code and the paper
- Known-broken configs, or a config that only works at a specific GPU count
- Which released checkpoint corresponds to which table row (often ambiguous)

**Assume the code and the paper disagree somewhere until proven otherwise.** When they
disagree, the code is usually what produced the reported number.

## A2. Environment and custom ops

Survey before building:

```bash
# find custom CUDA/C++ extensions
find . -name "*.cu" -o -name "setup.py" -o -name "*.cpp" | head -50
grep -rn "load(name=\|CUDAExtension\|CppExtension\|torch.utils.cpp_extension" --include=*.py .
```

Common culprits: deformable attention (Deformable DETR and descendants), DCNv2, custom NMS
or ROIAlign kernels, sparse conv libraries, older `mmcv-full` builds.

Guidance:

- Match the repo's **stated** torch/CUDA versions in a container rather than fighting the
  host toolchain. This is almost always faster than porting the op.
- If the op will not build against any reachable toolchain, check whether a maintained
  equivalent exists (e.g. native `torchvision.ops` implementations, or newer framework
  versions that absorbed the op) — but treat substitution as a change requiring re-validation
  at the A3 gate.
- Record the container image alongside the commit hash.

## A3. THE GATE — reproduce the reported number from the released checkpoint

Run the authors' checkpoint through the authors' eval script on the authors' benchmark.

This validates, without training anything:
- environment and custom ops
- data loading and annotation parsing
- preprocessing (resize policy, normalization, channel order)
- the metric implementation
- post-processing (NMS, score thresholds, coordinate un-mapping)

Interpretation:

| Result | Meaning | Action |
|---|---|---|
| within ~0.2 | pass | proceed |
| 0.5–2 low | preprocessing or eval config wrong | fix before proceeding — this is the bug you would otherwise carry into every later number |
| >5 low | wrong checkpoint, class mapping, or partial state-dict load | see failure atlas S2 |
| higher than reported | you are evaluating on the wrong split, or a different metric variant | investigate; do not celebrate |

**Always log state-dict load results explicitly:**

```python
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"missing={len(missing)} unexpected={len(unexpected)}")
assert len(missing) == 0, missing[:20]
```

Research repos use `strict=False` routinely. A head that silently fails to load is the single
most common cause of a wildly-below-paper number.

If the gate cannot be made to pass after genuine effort, that is a reportable finding: *this
repo does not reproduce its own published number under these conditions.* Document the gap
and the ruled-out causes. Do not proceed to customer data — you would have no way to
attribute any later failure.

## A4. Validate the training path

Short run on a benchmark subset. You are checking the loss curve **shape**, not the final
number:

- Does the loss start at roughly its analytic initial value? (see verification-ladder.md)
- Does it descend smoothly through warmup?
- Do the component losses (cls / box / obj / mask) have sane relative magnitudes?
- Does a mid-training eval produce a non-trivial metric?

If the paper published a training curve, compare shape against it.

## A5. Scan for hardcoded benchmark assumptions

Official code is written for exactly one dataset. Assumptions hide in the model definition,
not just the config. Grep patterns:

```bash
# class counts
grep -rn "num_classes\|NUM_CLASSES\|nc=\|80\b\|91\b" --include=*.py --include=*.yaml . | head -40

# normalization constants baked in
grep -rn "0.485\|0.229\|123.675\|58.395\|mean=\|std=" --include=*.py --include=*.yaml .

# anchors / priors / strides
grep -rn "anchor\|ANCHOR\|strides\|base_sizes\|scales=" --include=*.py --include=*.yaml .

# image size and paths
grep -rn "img_scale\|image_size\|input_size\|1333\|800\|640\b" --include=*.py --include=*.yaml .
grep -rn "coco\|COCO\|train2017\|val2017\|annotations/" --include=*.py --include=*.yaml .

# class name lists
grep -rn "CLASSES\s*=\|classes\s*=\s*(\|class_names" --include=*.py .
```

Build an explicit inventory of everything found, marked as *must change*, *must recompute*,
or *safe to keep*. That inventory becomes the input to Phase T.

Two specific traps:

- **Class name lists defined in more than one place** (dataset class, eval config, and a
  constants file). Changing one and not the others yields the S2 symptom.
- **`num_classes` semantics differ** — some frameworks want N, others N+1 for background.
  Read how it is consumed, do not assume.

## A6. Record the recipe spec

Before leaving Phase A, write out the complete recipe as a structured spec, tagging each
field with provenance:

- `paper` — explicitly stated in the publication
- `code` — taken from the official config/implementation
- `issue` — obtained from a maintainer reply or issue thread
- `default` — inherited framework or family default
- `unspecified` — nobody said, and it matters

**The `unspecified` list is your risk register.** When the number later comes up short, that
list is the hypothesis space, and it is much smaller than the space of all hyperparameters.
Use `assets/recipe_spec.yaml` as the template.
