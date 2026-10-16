---
name: tao-run-deft-od-aoi
description: >-
  Run a binary industrial-inspection DEFT loop with TAO RT-DETR: measure on
  frozen KPI/test roles, mine gap-similar real and clean images with SigLIP,
  accumulate admitted COCO data, retrain from one base checkpoint, and select
  by KPI AP50. Use for iterative AOI defect detection, not generic multiclass OD.
license: Apache-2.0
compatibility: Requires TAO RT-DETR and Data Services images, CUDA GPUs, and normalized COCO roles.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags: [application, workflow, deft, object-detection, aoi, rtdetr]
---

# TAO DEFT OD AOI

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

This application is a disk-backed RT-DETR loop for one foreground class,
`defect`. Its core is real-data-only; AnomalyGenNext synthesis is an optional
route with separate preparation, generation, and admission gates.

## References

Read only the references needed for the current stage:

- intake and launch: `references/defaults.md`, `references/data-contract.md`,
  `references/source-manifest.md`, and `references/preflight.md`;
- orchestration: `references/pipeline.md` and
  `references/scripts-and-agents.md`;
- gaps and retrieval: `references/gap-routing.md` and
  `references/tao-analyze-gaps-od-map.md`;
- training and selection: `references/training-policy.md`;
- optional synthesis: `references/anomalygen-pool.md`.

## Start

Select an installed platform, read its skill, then invoke
`tao-launch-workflow`. The single launch review must include the four normalized
COCO roles, trainable RT-DETR base checkpoint, maximum iterations, image and
Data Services containers, GPU shape, and expected runtime. After approval,
prepare source COCO files when needed:

```bash
scripts/prepare_deft_od_aoi_sources.py \
  --manifest /data/dataset_sources.json --check-only

scripts/prepare_deft_od_aoi_sources.py \
  --manifest /data/dataset_sources.json \
  --output-dir /new/results/normalized
```

`--check-only` runs without a container. Materialization must run in the pinned
TAO Data Services image because it delegates each role's canonical COCO merge
to the existing `annotations merge` action. The application then validates the
merged role contracts before emitting `sources.json`. Include `--gpus` when
launching the materialization container, as documented in
`references/source-manifest.md`.

The user-facing manifest calls the held-out input `benchmark`. The second
command maps it to the existing internal `kpi` role and emits `sources.json`
with the canonical `kpi`, `test`, `real`, and `clean` mapping. Copy
`assets/default_policy.yaml`, use that mapping for `sources`, fill the other
required values, and initialize once:

Source COCOs must already satisfy the canonical KPI metadata contract in
`references/data-contract.md`. The generic source preparer preserves those
fields; it does not infer dataset-specific identity or mask paths.

```bash
scripts/init_deft_od_aoi.py \
  --config /workspace/deft_policy.yaml \
  --output-dir /new/results/deft_contract
```

Never reinitialize an existing result. The validator requires disjoint KPI,
test, defective-real, and verified-clean roles; every COCO must declare only
`defect`. KPI and test may mix boxed and boxless images because they never enter
training. The defective-real role must be nonempty and needs at least one box
on every image. Clean images remain explicit zero-annotation COCO entries; an
empty clean role is accepted as an unavailable retrieval capability with a
startup warning.

## Loop boundary

The loop composes existing bank actions:

1. Baseline gap initialization in explicit `cold_start` or `checkpoint` mode;
   later measurements use `tao-train-rtdetr` inference on KPI and test.
2. Two `tao-analyze-gaps-od-map` actions: loose confidence for FP routing and
   strict confidence for FN routing.
3. `tao-generate-image-embeddings` with one frozen SigLIP encoder, followed by
   `tao-mine-od-images` unique-neighbor matching against the real or clean role.
4. Application-owned admission and cumulative binary COCO assembly.
5. Direct `tao-train-rtdetr` training from the same frozen base checkpoint,
   then KPI-only checkpoint selection. Test remains report-only.

All specs are nested YAML dictionaries. Every GPU/Data Services action uses the
selected platform's `submit/status/logs/cancel` contract and a job record.
Stop on missing or invalid artifacts, role overlap, class drift, or
failed training. Never infer live state from the JSON record alone.

## Retrieval preparation

Build the reusable candidate cache once, then prepare queries after each pair
of loose/strict gap jobs:

```bash
scripts/prepare_deft_od_aoi_retrieval.py candidates \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --output-dir "$RESULTS/candidates"

scripts/prepare_deft_od_aoi_retrieval.py queries \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$ITER/strict/box_gaps.parquet" \
  --loose-gaps "$ITER/loose/box_gaps.parquet" \
  --previous-coco "$PREVIOUS/train.json" \
  --iteration 1 --candidate-root "$RESULTS/candidates" \
  --output-dir "$ITER/retrieval"
```

The candidate manifest records a zero count and emits no candidate parquet or
embedding spec for an empty clean source role.

Run every emitted embedding spec through `tao-generate-image-embeddings`.
For `max_similarity`, run each emitted mining spec through
`tao-mine-od-images`. For `round_robin_similarity`, materialize the selected
artifacts after query embedding instead:

```bash
scripts/deft_od_aoi_round_robin_selection.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --candidate-root "$RESULTS/candidates" \
  --retrieval-root "$ITER/retrieval" \
  --previous-coco "$PREVIOUS/train.json"
```

Omit `--previous-coco` for iteration 1. Later iterations exclude candidate
crops whose source image is already in the cumulative training COCO, while
admission remains the authoritative source-level deduplication gate. Both
strategies must produce an
enabled role's `mine_<role>/final_unique_files.parquet` before committing
`iteration_retrieval`; round-robin also commits
`round_robin_selection_report.json` as `selection_report` and
`round_robin_admission_index.npy` as `admission_index`. Defective candidates
and gap queries use the independently selected
`retrieval.preprocessing.profile`:
the default `square_context` mean-pads and resizes to 224×224, while
`tight_context` preserves an aspect-ratio-aware native-size crop. Clean
candidates use the same profile over the frozen grid.
Strict FNs and loose near-miss FPs route to real data. Background-like loose
FPs route only to the verified-clean role. Pass the prior cumulative COCO as
`--previous-coco`; optional role-specific exclusion parquets use
`--real-exclusions` and `--clean-exclusions`. Empty or fully excluded roles
emit audited exhaustion evidence and no mining action. If every producer is
exhausted and synthesis is not pending, the stage records convergence.
The independent `retrieval.selection.strategy` defaults to
`round_robin_similarity`, which balances candidates across stable per-query
rank rounds and pocket-scoped quotas. `max_similarity` retains the global
maximum-similarity path.

## Admission and cumulative COCO

After each enabled role has a committed selection artifact, admit its selected
candidate crops back to unique source images and publish the next cumulative
dataset:

```bash
scripts/admit_deft_od_aoi_coco.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --candidate-root "$RESULTS/candidates" \
  --retrieval-root "$ITER/retrieval" \
  --previous-coco "$PREVIOUS/train.json" \
  --output-dir "$ITER/training_data"
```

Omit `--previous-coco` only for iteration 1. For `max_similarity`, the helper
recomputes maximum cosine similarity from the frozen candidate/query
embeddings. For `round_robin_similarity`, it consumes the committed per-query
selection, screened boxes, audit report, and admission index without rerunning
selection, then publishes `admission_index.npy` for the next iteration. Both paths apply the
minimum similarity upstream, deduplicate crop hits to source images, exclude
prior sources, and cap cumulative clean negatives against cumulative real defects.
It retains every prior image and box and emits one binary COCO with explicit
zero-annotation clean images. Use `--link-mode hardlink` only when source and
output share a filesystem; portable staging should keep the copy default.

## Measurement specs

For the baseline, pass `--baseline`. The default `cold_start` mode creates an
empty prediction file for every KPI image, making all KPI ground-truth boxes
initial false negatives without loading an incompatible multiclass checkpoint
into the binary inference head. It also emits cardinality-matched empty test
prediction evidence so the measurement manifest retains the same KPI/test
completion contract without running inference. Explicit `checkpoint` mode
emits normal baseline KPI/test inference specs and therefore requires a
binary-compatible checkpoint. Neither mode changes training initialization.

```bash
scripts/prepare_deft_od_aoi_measurement.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --checkpoint "$CHECKPOINT" \
  --kpi-predictions "$MEASURE/kpi/inference/labels" \
  --results-root "$MEASURE" --output-dir "$MEASURE/specs" \
  --baseline
```

Submit only the inference specs listed in `measurement_manifest.json`; a
cold-start baseline lists no inference specs. Then submit `gap_loose.yaml` and
`gap_strict.yaml` through
`tao-analyze-gaps-od-map`. The helper projects the frozen KPI COCO to KITTI,
keeps the two-line inference class map durable, and emits only nested specs.
Test inference is report-only and its output cannot influence routing or
checkpoint selection.

After training, omit `--baseline`; every iteration measurement uses the
selected binary checkpoint and emits KPI/test inference regardless of the
baseline mode.

## RT-DETR training specs

Build nested specs from the current cumulative COCO:

```bash
scripts/prepare_deft_od_aoi_training.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" --iteration 1 \
  --train-coco "$ITER/training_data/train.json" \
  --train-images "$ITER/training_data/images" \
  --results-root "$ITER/training" --output-dir "$ITER/train_specs"
```

Invoke the RT-DETR leaf as direct training with `automl_policy: off`; the AOI
application already owns its frozen probe policy. Iterations 1–2 use 36 epochs.
Later iterations use the clipped size-adaptive budget. When probes are enabled
and the iteration reaches the frozen `training.probes_start_iteration`,
the helper emits independent incumbent, data-growth-scaled, and deterministic
jitter specs, all initialized from the same frozen base checkpoint. Run all
three before selecting and materializing `train.yaml`; never initialize the
next iteration from a previous iteration checkpoint.

Select all three probes before main training:

```bash
scripts/select_deft_od_aoi_training.py probes \
  --manifest "$SPECS/training_manifest.json" \
  --main-template "$SPECS/main_template.yaml" \
  --status "$P0/status.json" --status "$P1/status.json" --status "$P2/status.json" \
  --output-dir "$SPECS/selected"
```

After main training, select the KPI-best checkpoint from structured status
rows. Repeat `--status` for resumed phases. If the best epoch is within the
frozen late window, the first call emits one same-iteration `extension.yaml`
from the terminal checkpoint; submit it, then reselect with
`--extension-applied`. The checkpoint carried to measurement is always the
maximum KPI `val_mAP50`, never the latest checkpoint or test result.

After each stage succeeds, commit at least one completion artifact with
`commit_deft_od_aoi_stage.py`. It accepts only the next frozen stage, verifies
and hashes every named file, atomically updates `deft_state.json`, and appends
`loop_log.jsonl`. The final iteration becomes `COMPLETE` only after its gap
artifacts are committed. Poll native backends for live state; this record is
durable workflow history, not a scheduler substitute.

## Optional synthesis with existing task weights

Enable `synthesis` only when every KPI sample retains its real, nonempty
`dataset_id`; a missing ID is malformed KPI metadata and is never an opt-out.
`synthesis.routes` is the synthesis allowlist. Dataset IDs absent from it skip
synthesis but remain in the separate normal real-data DEFT path. Every FN in a
configured route must also carry `texture_id`, `defect_class`, and a pixel
`fn_mask_source`, and the route must provide an existing AnomalyGenNext
checkpoint and matching recipe. After strict gap analysis, normalize exact
FN/annotation matches:

```bash
scripts/prepare_deft_od_aoi_synthesis.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$MEASURE/gap_strict/box_gaps.parquet" \
  --iteration 1 \
  --real-coco "$ITER/admission/train.json" \
  --output-dir "$ITER/synthesis_request"
```

The default `synthesis.fn_selection.mode: generated_per_type_plan` bounds work
from the current admitted real COCO. It derives a deterministic per-type plan
from `cumulative_fraction_of_total_defects`, writes and hashes the plan, and
records the full allocation calculation. A fraction of `0.25` means synthetic
images are 25% of the combined real-plus-synthetic defect pool. Set the mode to
`all_eligible` explicitly to preserve every routed FN without pre-generation
budgeting.

The planner derives the finalized yield as `2 * max_neighbors_per_fn`: the
finalizer emits two fixed mask branches and requests one image from each branch
for every retained neighbor. `candidate_topn` only supplies fallback neighbor
candidates and does not enter this calculation. It is synthesis-only and is
independent of both max-similarity overfetch and round-robin refill. The derived
value is recorded as `images_per_fn` in the frozen plan evidence; it is not a
policy input. `candidate_topn` is the AMP compute knob: preparation emits two
AMP requests per eligible candidate, so one FN costs up to
`2 * candidate_topn` placement attempts to retain
`2 * max_neighbors_per_fn` finalized images. For backward compatibility,
initialization copies an explicitly
supplied legacy `retrieval.candidate_overfetch` into synthesis when
`candidate_topn` is omitted, and already-frozen policies without the new field
retain that fallback.
Set both fields explicitly to configure them independently.

Pass the emitted filtering YAML through `tao-prepare-anomalygennext-inputs`,
mounting the complete checkpoint root for its `run_amp` action, then pass its
finalized generation plan through `tao-generate-od-defects` with the same
checkpoint root mounted at the same canonical path. Commit
`iteration_synthesis` before training. Read the selected
`tao-generate-od-defects` action contract and resolve
`actions.generate.outputs.binary_coco.relative_path` against that action's
results directory. Re-run admission with the resolved file passed through
`--synthetic-coco` and the generation results directory passed through
`--synthetic-images`; admission never locates a companion skill itself. Synthetic
categories are folded to `defect`, and
the frozen cumulative fraction cap is applied against admitted real defects.
Commit initial admission before requesting synthesis. The stage controller
uses its calculated synthetic capacity: zero room records a
`no_synthetic_budget` skip and bypasses iteration synthesis. Train only when
this iteration admitted new images; otherwise converge. Positive capacity
can remain from prior real admissions even without new real matches.
The post-generation cap and quality checks still apply when synthesis runs;
provide the same-iteration real admission as `--previous-coco` and set
`--synthetic-only` so synthetic inputs do not implicitly rerun mining admission.
`SYNTHETIC_ADMISSION_CAPPED` reports eligible generated images excluded by the
cap. See `references/pipeline.md` for budget-skip evidence. A zero cumulative
real count permits zero synthetic admissions.

Boxes never substitute for the required pixel mask. If the request has no
routed FNs, or preparation reports no mask-eligible FNs, commit that typed skip
contract as `iteration_synthesis` and continue to training without running AMP
or fabricating generation/admission success. Missing or empty AnomalyGen clean
reference pools likewise skip only affected FN/types with
`no_clean_reference_images`; eligible types and real/clean retrieval continue.
Initialization rejects synthesis when the entire configured reference pool has
no supported image under any `TEXTURE/clean_image` directory.
If no FN remains eligible, the preparation-level reason is
`no_eligible_false_negatives` while per-FN evidence retains the specific cause.
If every producer is unavailable, the typed skip commits convergence instead of
training on an unchanged iteration.

## Missing AnomalyGenNext task weights

A synthesis route may replace `checkpoint` and `recipe` with a `finetune` block
containing `dataset_root`, frozen `validation_testcase`, Cosmos3-Nano
`base_checkpoint`, `vae_path`, `checkpoint_root`, future `result_handoff`, and
optional user `recipe_template`/`defect_spec`. These are AnomalyGenNext inputs;
the application policy is not an upstream training recipe.

Run `resolve_deft_od_aoi_synthesis.py` before the candidate cache. If a handoff
is absent, it emits `finetune_requests.json`; execute each request through
`tao-finetune-anomalygennext` once. Resolve again into a new directory, verify
the hash-bound handoff, commit `synthesis_bootstrap`, and use the emitted
`resolved_synthesis_policy.yaml` for all later synthesis preparation. Never
start or resume AnomalyGenNext training inside a DEFT iteration.
