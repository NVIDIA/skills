# DEFT OD — Source-Pool Prep (runs once, before baseline)

Prep turns a directory of raw unlabeled images into the two artifacts every iteration depends on:

| Artifact | Produced by | Consumed by |
|---|---|---|
| `source_pool/odvg/*_odvg.jsonl` + `*_odvg_labelmap.json` | Co-DETR pseudo-labeling (folding via `category_mapping`) → KITTI→COCO→ODVG | `stage` (annotation lookup by basename) |
| `source_pool/source_embeddings.parquet` | `embedding image_embeddings` over the pool | `mine` (the search corpus) |
| `source_pool/coco.json` | the KITTI→COCO step, retained | `mine` as `source_detection_file` (class_stratified only) |

## Paths

Every command below uses `${PREP_DIR}`. Export it before the first step:

```bash
export PREP_DIR="${RESULTS_DIR}/prep"
mkdir -p "$PREP_DIR"

export CLASSES_YAML="<path to the run's classes.yaml>"
export POOL_IMAGES="<config.pool_images>"   # the raw images this prep labels
export CODETR_SPEC="${PREP_DIR}/codetr_inference.yaml"
```

Emit `$CODETR_SPEC` before step 3 overrides it:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
  --stage codetr_inference --pyt-image "$TAO_PYT_IMAGE" --out "$CODETR_SPEC"
```

`${RESULTS_DIR}/prep` is the layout the rest of the workflow expects: the output tree
in `references/pipeline-and-state.md` places the mappings and `pool_report.json` there,
and the report reads them from that path. The pool's own artifacts are separate —
they live under `<workspace>/source_pool/`, outside the results tree, so a second run
against the same pool reuses them.

**This runs exactly once, before the baseline.** Every iteration afterwards is a lookup against what prep produced — no labeler and no encoder runs inside the loop. That is the point: it keeps the per-iteration path deterministic and confines GPU work to train and inference.

**Prep is idempotent.** Skip any step whose output already exists. A user arriving with a pool that is already labeled and embedded pays nothing; a user with raw images pays for the whole chain once. Never re-label or re-embed a pool that already has current artifacts.

**A pre-supplied pool still needs its report.** `class_stratified` mining requires
`pool_report.json`, and `init_deft_state.py` refuses without one on a pool it is not
about to build. A pool prepared before that file existed, or by anything other than
this skill, will not have it — generate one from the pool's own COCO rather than
re-labelling:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_pool_coco.py \
  --coco "<pool>/coco.json" --classes "<classes.yaml>" \
  --record target_classes="<comma-separated target classes>" \
  --record source="<where this pool came from>" \
  --report-json "<pool>/pool_report.json"
```

Seconds, not GPU hours: it reads the COCO, so it both produces the report and
verifies the pool actually holds the classes this run targets.

**Cost warning.** Labeling and embedding are proportional to *pool size*, not to what mining eventually selects. A large pool means a substantial one-time job that must finish before the baseline starts. Surface the pool image count and this expectation in the Pre-Flight Summary so it is not discovered mid-run.

## Target classes are a run-wide contract

The user supplies the classes they want to train on. That same list has to be consistent across four places, or the loop silently misbehaves:

| Where | Use |
|---|---|
| `classes.yaml` (this stage) | Folds Co-DETR's predicted vocabulary down to the target set |
| `weak_thresholds` in `gap_analysis` | Per-class AP50 gates — every target class listed explicitly |
| `rare_class_list` in `mine` | Which target classes drive class-stratified allocation |
| `kpi/mapping.yaml` in `kpi_analyze` | Class mapping for scoring |

Resolve the target class list once in Pre-Flight and derive all four from it. A class present in the KPI ground truth but absent from `classes.yaml` can never be pseudo-labeled, so mining will never find examples for it — the loop will look like it is working while being structurally unable to improve that class.

## Step 0 — Refuse a pool with duplicate basenames

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/verify_pseudo_labels.py \
  --pool-images-dir "$POOL_IMAGES"
```

Every stage after this keys on the basename: `codetr inference` writes one flat
`<basename>.txt`, `annotations convert` reads that directory, and ODVG records carry
a basename. Two pool images with the same name therefore share one set of labels —
the second image's pseudo-labels are overwritten before anything can observe it, and
no later stage can recover them. `stage_mined_odvg.py` refuses such a pool too, but
by then the GPU time is already spent.

A directory walk, so run it before the labelling pass rather than after.

## Step 1 — Emit the two mappings

TAO does the folding; this step only writes the files that tell it how.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_class_mappings_for_mining_data_prep.py \
  --classes "$CLASSES_YAML" \
  --emit-codetr-category-mapping "${PREP_DIR}/codetr_category_mapping.yaml" \
  --emit-kitti-mapping           "${PREP_DIR}/kitti_mapping.yaml" \
  --emit-classmap                "${PREP_DIR}/classmap_target.txt"
```

Two accepted forms for `classes.yaml`. **Explicit folding** — several predicted classes collapse into one target:

```yaml
classes:
  car:     [car, truck, bus]
  person:  [person]
  bicycle: [bicycle]
```

This is the shipped ITS default (`references/example_classes_its.yaml`): trucks and buses count as `car`, because the target vocabulary has no separate heavy-vehicle class and dropping them would discard real road users the model must detect.

**Identity form** — when the target names already match the predicted names and you only want to drop the rest:

```yaml
classes: [car, person, bicycle]
```

**Confirm the fold rather than inventing one.** For a new dataset, whether two predicted classes belong together is a decision about what the metric should mean, so ask instead of guessing. Where a class sits ambiguously between targets — `motorcycle` is neither a car nor a bicycle — surface the choice, since folding it either way silently changes what the numbers report.

### Class names are matched exactly, including case

Both consumers do exact-string lookups — Co-DETR checks `orig_name not in name_to_id`, and `annotations convert` does `labels2cat.get(row_p[0])`. A source written `Car` against a detector that emits `car` matches nothing and every one of its boxes is dropped. Neither raises; Co-DETR logs a warning among many, and the converter says nothing at all.

Copy source names verbatim from the detector's classmap. `validate_pool_coco.py` (step 4) names case-only mismatches explicitly, because this is the most likely way a fold goes wrong and the least visible.

Matching is case-sensitive in both `annotations convert` and `analytics kpi_analyze`,
which share the same lookup, so a mapping must enumerate every spelling it expects to
see — `Bicycle` and `bicycle`, `AutoMobile` and `Automobile`.

### Target names must be single tokens

Targets become COCO categories, then ODVG labels, then the Grounding DINO caption list, and finally the class field of KITTI inference labels. KITTI is space-delimited, so a multi-word target makes those unparseable — the script rejects it. Source names may contain spaces; COCO has `traffic light`.

## Step 2 — Pseudo-label the pool with Co-DETR, folding as you go

Invoke `tao-skill-bank:tao-train-codetr` (read its `SKILL.md` first).

The `codetr` console script is unregistered in the TAO PyTorch images, though the module
ships. Probe both forms and use whichever answers; only if *both* fail is Co-DETR absent:

```bash
docker run --rm "$TAO_PYT_IMAGE" codetr --help >/dev/null 2>&1 && CODETR="codetr"
[ -z "${CODETR:-}" ] && docker run --rm --entrypoint sh "$TAO_PYT_IMAGE" \
  -c 'python3 -c "import nvidia_tao_pytorch.cv.codetr"' >/dev/null 2>&1 \
  && CODETR="python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr"
```

Both forms take identical arguments. Substitute `$CODETR` wherever this document writes `codetr`.

### The fold belongs here, not downstream

`inference.category_mapping` groups the detector's own classes into output categories **after the forward pass**, and it is the right place to fold COCO-80 down to the target set:

```yaml
inference:
  checkpoint: <co-detr checkpoint>
  category_mapping:            # emitted by scripts/prepare_class_mappings_for_mining_data_prep.py
    bicycle:   ["bicycle", "motorcycle"]
    car:       ["car", "bus", "truck"]
    person:    ["person"]
    road_sign: ["road_sign", "traffic light", "stop sign"]
```

`assets/overlays/codetr_inference.yaml` supplies the rest and must be applied to the
emitted spec before launching. Two of its settings decide what the pool contains:

* `model.num_select: 1000` (TAO: 300) caps how many decoder queries become
  candidates, *before* the fold and soft-NMS. At TAO's 300 the abundant class fills
  the candidate set and the rare ones are gone before the fold runs — bicycle drops
  from 55 boxes to 2 across the same 8 images. Mining rare classes is the point of
  the loop, so this is the field to check first if a pool looks class-skewed.
* `inference.conf_threshold: 0.3` (TAO: 0.5) is the quality gate on the labels.

Sizing is by resize-and-pad — `test_random_resize: 1280`, `random_resize_max_size:
2048` — with `inference.input_width`/`input_height` left unset.

Two consequences to respect: output category IDs are assigned `0..K-1` **in the order the
mapping is written**, and the fold runs a per-output-category soft-NMS afterwards. That NMS
is why the fold belongs here — one object detected as both `truck` and `car` becomes two
boxes of the *same* class the moment they merge, and only a post-fold pass removes the
duplicate. Renaming labels later ships the duplicates into training. See `tao-train-codetr`
for how unmapped and conflicting names are handled.

### The spec's architecture must match the checkpoint

A mismatch does not raise. `codetr inference` loads what fits, discards what does not,
prints `Execution status: PASS`, exits 0, and writes one label file per image with **zero
boxes** — the unloaded layers stay randomly initialised. `annotations convert` then succeeds
on an empty COCO and the first hard failure lands stages later.

`assets/overlays/codetr_inference.yaml` pins the six fields the ViT-Large COCO-80 checkpoint
needs, so the schema defaults never apply. Step 2's gate (`verify_pseudo_labels.py`) catches
it if they are wrong anyway.

For a different checkpoint, derive the values from its tensors rather than guessing — see
`tao-train-codetr`'s **Checkpoint/spec pairing** reference, which also covers the
`cls_branches` vs `roi_head` class-count trap and the patch-size constraint on the backbone.

### Launching

`dataset.infer_data_sources` is `null` in the schema, so it cannot be populated from
the Hydra command line — `dataset.infer_data_sources.image_dir=...` fails with
`AssertionError: Unexpected type for root: NoneType` before the run starts. Set the
whole mapping into the spec first, together with the workflow defaults and the
`category_mapping` block emitted by step 1:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$CODETR_SPEC" \
  --apply-workflow-defaults <skill_root>/assets/overlays/codetr_inference.yaml \
  --allow-new \
  --set dataset.infer_data_sources.image_dir="$POOL_IMAGES" \
  --set dataset.infer_data_sources.classmap="$CODETR_CLASSMAP" \
  --report-json "${PREP_DIR}/codetr_spec_report.json"
```

Set the fold from step 1's emitted file into the same spec — the spec, not the
command line, carries it for the same reason. Load the file; do not retype it:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$CODETR_SPEC" --allow-new \
  --set-from-file inference.category_mapping="${PREP_DIR}/codetr_category_mapping.yaml"
```

The emitted file's single top-level `category_mapping:` key is unwrapped, so its
value lands directly on `inference.category_mapping`.

```bash
docker run --rm --name "deft_prep_codetr" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  $CODETR inference -e "$CODETR_SPEC" \
    inference.checkpoint="$CODETR_CHECKPOINT" \
    results_dir="${PREP_DIR}" \
    inference.num_gpus="$NUM_GPUS"
```

Add `-v` for every path outside `$WORKSPACE` — the checkpoint and the classmap
commonly live elsewhere, and a container cannot read what is not mounted.

Runtime scales with pool size, so on a large pool this is the longest stage in the
workflow. Wait on its artifacts, not on the process:

```bash
LAUNCH_MARKER="${PREP_DIR}/.codetr_launched"
mkdir -p "$(dirname "$LAUNCH_MARKER")" && touch "$LAUNCH_MARKER"
# ... launch the container ...

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/await_stage.py \
  --newer-than "$LAUNCH_MARKER" \
  --status-json "${PREP_DIR}/inference/status.json" \
  --status-contains "finished successfully"
```

TAO appends the action name to `results_dir`, so passing `results_dir=${PREP_DIR}`
puts every output under `${PREP_DIR}/inference/`. A wait pointed at
`${PREP_DIR}/status.json` never matches and times out while the stage runs normally.

Do not add `--artifact "${PREP_DIR}/inference/labels"` here. `await_stage.py` returns on
the first condition that holds, and TAO creates the labels directory when it starts, so
the wait would return within seconds and prep would convert a fraction of the pool as
though it were complete. Wait on `status.json` alone for any stage whose outputs appear
before it finishes, and use `--newer-than` so a previous attempt's status cannot satisfy
it.

`dataset.infer_data_sources.classmap` is the detector's **own** vocabulary — one class name per line in `category_id` order starting at 1, COCO-80 for a COCO-trained checkpoint. It is not the target list; `category_mapping` names are matched against it. `results_dir` auto-appends the action, so labels land in `${PREP_DIR}/inference/labels/` already carrying target class names.

`conf_threshold` is the main quality control on the pseudo-labels and is applied at write time.

Obtaining a checkpoint is outside this skill — see `tao-train-codetr`'s SKILL.md.

### Gate the output before converting it

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/verify_pseudo_labels.py \
  --labels-dir "${PREP_DIR}/inference/labels" \
  --expect-images "$POOL_IMAGE_COUNT" \
  --report-json "${PREP_DIR}/pseudo_label_report.json"
```

Non-zero exit stops prep here. This costs a directory walk and is the difference
between catching an architecture mismatch now and discovering it after the
conversion, embedding and mining stages have all succeeded on an empty pool.

## Step 3 — KITTI → COCO

Emit `annotations default_specs`, then apply `assets/overlays/kitti_to_coco.yaml`
(which pins the formats and `kitti.project`) plus the run's paths:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "${PREP_DIR}/kitti_to_coco.yaml" \
  --apply-workflow-defaults <skill_root>/assets/overlays/kitti_to_coco.yaml \
  --set kitti.image_dir=<pool images> \
  --set kitti.label_dir="${PREP_DIR}/inference/labels" \
  --set kitti.mapping="${PREP_DIR}/kitti_mapping.yaml" \
  --set results_dir=<workspace>/source_pool \
  --require-no-mandatory-under kitti
```

`label_dir` is Co-DETR's output, already folded; `mapping` is the identity over
the targets. `results_dir` is **not** a scratch dir — see below. The overlay pins
`kitti.project: coco`, which names the output file `coco.json` that the verify
step and the ODVG conversion both look for.

```bash
docker run --rm --name "deft_prep_kitti2coco" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" annotations convert -e "${PREP_DIR}/kitti_to_coco.yaml"
```

**The mapping is identity here, and still explicit.** Co-DETR already folded, so each target maps only to itself. Omitting `mapping` entirely would make the converter auto-derive one by scanning the label directory — but then COCO category IDs follow filesystem discovery order, and a target absent from this particular pool disappears from `categories`. Those IDs travel into the ODVG labelmap and then into training, so they have to be stable across runs.

**Every TAO task invocation needs `--gpus`, including the CPU-bound ones.** The launcher shells
out to `nvidia-smi -L` unconditionally before dispatching, so a container started without
`--gpus` dies with `FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'` —
`annotations convert` does no GPU work and still requires it. A bare `--help` is the exception;
it short-circuits before the probe, which is why the step 1 availability check works without one.

**`kitti.project` names the output file, and it is not optional here.** The converter writes `<results_dir>/<project>.json`, falling back to the second-to-last component of `kitti.image_dir` (`project = name or img_dir.split('/')[-2]`). Leave it unset and a pool at `.../my_pool/images` silently produces `source_pool/my_pool.json` — not the `source_pool/coco.json` that step 5, the `source_detection_file` gate, and `config.source_detection_file` all name.

**This COCO is not a throwaway.** It is step 5's input *and* the `source_detection_file` that `class_stratified` mining consumes on every iteration. Write it to `<workspace>/source_pool/coco.json` and record that path in state as `config.source_detection_file`.

**Empty label files must exist, not be omitted.** The converter calls `os.path.getsize()` on the label file for *every* image in `image_dir`, inside a multiprocessing worker, so a label file that is absent rather than empty raises `FileNotFoundError` and takes the whole conversion down with `Execution status: FAIL`. Co-DETR writes one file per image, so this holds by construction — it matters only if something prunes them in between.

Do not read that as "the image survives to training". It does not: this step skips images with no valid annotation (`kitti.no_skip` defaults to `False`) and step 5 skips zero-annotation images unconditionally, so an image whose boxes all dropped reaches neither `coco.json` nor the ODVG. It *does* stay in the embedding parquet, which covers the whole pool — the parquet/ODVG divergence `references/data-layout.md` describes.

## Step 4 — Verify the pool before anything trusts it

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_pool_coco.py \
  --coco        "<workspace>/source_pool/coco.json" \
  --classes     "$CLASSES_YAML" \
  --labels-dir  "${PREP_DIR}/inference/labels" \
  --images-dir  "$POOL_IMAGES" \
  --record      target_classes="<comma-separated target classes>" \
  --record      codetr_checkpoint="$CODETR_CHECKPOINT" \
  --record      pyt_image="$TAO_PYT_IMAGE" \
  --report-json "${PREP_DIR}/pool_report.json"
```

`--record` writes those under `prep_inputs` in the report. Prep is idempotent by
existence, and existence cannot show a pool was folded to the same classes with the
same checkpoint — `init_deft_state.py` compares the recorded classes against the
run's and refuses a pool prepared for a different set.

Both TAO folds fail *quietly* when names do not line up: an unmatched name is a log warning, an unmapped detection is silently dropped, and the conversion still prints `Execution status: PASS` and exits 0 while writing a COCO with no annotations. Nothing downstream notices until training produces a model that cannot detect a class, several GPU-hours later. So verify the artifact.

| Check | Why it is a hard error |
|---|---|
| every target class carries annotations | The pool holds no examples of it. If it also appears in the KPI ground truth, gap analysis marks those images weak every iteration while mining cannot find anything — the loop runs its full course looking healthy and cannot improve it. |
| the COCO has any annotations at all | The classic mapping-shape failure: a string where a list belongs. |
| case-only class mismatches | `Car` vs `car` drops every box of that class, silently. |

`pool_report.json` also settles **`max_labels`** — one per target class. Grounding DINO caps
each training caption at that many class phrases: the classes present in the image plus
randomly sampled ones that are not, which is what teaches the model what a class *is not*. Any
larger value behaves identically, since there are no further negatives to sample; any smaller
one truncates them. Fixing it here means the train spec never carries a stale constant, which
is silently wrong the moment the target set changes size.

Reported as warnings, not errors: unmapped source classes with counts (a large `truck=1523` against an identity mapping is a missing fold), the pool-vs-COCO image shortfall, and an annotation count below the folded box count — the converter calls `drop_duplicates()` per image, so byte-identical KITTI lines collapse into one.

Pass `--allow-empty-classes` only when a class is listed defensively and its absence from this pool is expected.

## Step 5 — COCO → ODVG

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "${PREP_DIR}/coco_to_odvg.yaml" \
  --apply-workflow-defaults <skill_root>/assets/overlays/coco_to_odvg.yaml \
  --set coco.ann_file=<the COCO json written by step 3> \
  --set results_dir=<workspace>/source_pool/odvg \
  --require-no-mandatory-under coco
```

```bash
docker run --rm --name "deft_prep_coco2odvg" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" annotations convert -e "${PREP_DIR}/coco_to_odvg.yaml"
```

Output is named after the input: `<basename>_odvg.jsonl` and `<basename>_odvg_labelmap.json`. There is no direct KITTI → ODVG conversion, which is why steps 3 and 4 are separate.

Confirm the ODVG records key on `file_name` with the image **basename** — that is how `stage_mined_odvg.py` looks them up later.

## Step 6 — Embed the pool

First build the input list. Do not hand-write it:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_input_for_image_embeddings.py \
  --images-dir "$POOL_IMAGES" \
  --out        "${PREP_DIR}/pool_input.parquet"
```

Then embed it. `tao-skill-bank:tao-generate-image-embeddings` covers this stage, but its
spec block is written for an iteration — `weak_images.parquet` in, embeddings for the weak
set out. Prep embeds the **pool**, so the two paths differ and the encoder fields do not:

```bash
EMBED_SPEC="${PREP_DIR}/image_embeddings.yaml"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
  --stage embedding --ds-image "$TAO_DS_IMAGE" --out "$EMBED_SPEC"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$EMBED_SPEC" \
  --set input_parquet="${PREP_DIR}/pool_input.parquet" \
  --set output_parquet="<workspace>/source_pool/source_embeddings.parquet" \
  --set model="$EMBEDDING_MODEL" \
  --set model_path="$EMBEDDING_MODEL_PATH" \
  --set model_config_path='""' \
  --set batch_size=64
```

```bash
docker run --rm --name "deft_prep_embed" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  embedding image_embeddings -e "$EMBED_SPEC"
```

The encoder is whatever Pre-Flight check 9 resolved, and it must be the same one every
iteration uses: mining compares an iteration's embeddings against this pool parquet, so
two encoders produce vectors that are not comparable and the failure is silent — mining
succeeds and returns confidently wrong neighbours.

`model_config_path` needs the doubled quoting shown above; see
`references/tao-generate-image-embeddings.md` for why.

**Embed the whole pool, not just what reached the COCO.** Step 3 skips images with no surviving
box — 35 of 5,000 on the reference pool — so `coco.json` is not the image list. Mining searches
the parquet, and an image absent from it can never be selected at all.

The `filepath` column must hold the **same absolute paths** the ODVG `file_name` basenames
resolve against. Mining selects by `filepath`; staging then looks up annotations by basename. If
those two disagree, mining succeeds and staging reports every image as missing an annotation.
The script resolves symlinks for this reason: a pool of links into an unmounted tree yields a
container that sees no images, and an error that blames the image list rather than the mount.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label prep --stage prep \
  --pool-odvg "<workspace>/source_pool/odvg" \
  --pool-embeddings "<workspace>/source_pool/source_embeddings.parquet" \
  --pool-report "${PREP_DIR}/pool_report.json" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "prep: labeled <N> pool images, <M> boxes kept across <K> classes"
```

`--pool-report` is what settles `rare_class_list`: under `class_stratified` allocation
the commit derives it here from the pool's annotation counts and writes it into state,
because which classes are rare is a property of the pool and nothing knows it before
prep. Omitting the flag leaves the field unset and `mine` refuses.

## Gates

Hard-stop before the baseline when any of these hold:

- Co-DETR is unavailable in the resolved image. Probe **both** forms — the bare `codetr`
  console script *and* `python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr` (step 1).
  A failing `codetr --help` on its own is not this gate: it fails in every image checked so
  far while the module works, so gating on it alone hard-stops every run.
- `verify_pseudo_labels.py` exits non-zero (step 1). Co-DETR wrote label files but almost
  no boxes — check the checkpoint/spec architecture pairing before anything else.
- `validate_pool_coco.py` exits non-zero (step 4). It covers the three failures that are
  otherwise silent: a target class with no annotations, a COCO with no annotations at all,
  and a case-only class mismatch.
- The ODVG output is empty or absent.
- `class_stratified` is configured but `source_pool/coco.json` was not retained from step 3.
- The embedding parquet is empty, or its row count is wildly below the pool image count.

Read `pool_report.json`'s warnings before proceeding even when it exits zero — `dropped_by_class`
is where a missing fold shows up, and it is not an error because dropping classes is often
exactly what was intended.
