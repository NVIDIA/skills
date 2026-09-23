# DEFT OD — KPI Analyze Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-analyze-detection-kpi`. Read that skill's `SKILL.md` for the full field reference and pitfalls.

## When to invoke

Twice per phase boundary:

- **Baseline**: after the zero-shot `inference`, as the final baseline stage.
- **Iteration N**: after `inference`, as the final stage of the iteration.

## No pre-KPI label filtering

The reference pipeline optionally filtered inference labels before scoring — exclude-ROI (masking out a region of the frame) and exclude-PIC (dropping person-inside-car detections) — and its `exp7` configuration ran with PIC enabled.

**This loop applies neither.** Both are dataset-specific label cleanups for one intersection deployment, not general OD loop mechanics, so `kpi_analyze` scores the raw inference labels as written.

The practical consequence: **mAP here will not match `exp7`'s numbers**, because exp7's persons-inside-cars were removed from ground truth before scoring. The trend across iterations is still valid — it's a consistent measurement — but do not compare the absolute values against the reference experiment's.

## mAP is reported, not gated

This loop does **not** early-exit on a metric target. `kpi_analyze` records where the model stands so the report can show the trend across iterations; a regression does not stop the loop and does not trigger a retry. The loop runs until `max_iterations` or a hard-stop gate.

Do not add metric-based stopping logic here. If the user wants target-gated stopping, that is a change to the workflow contract, not something to improvise mid-run.

## Spec

Do not hand-write it. Emit `analytics default_specs`, then apply the overlay —
`assets/overlays/kpi_analyze.yaml` holds every documented setting, so a field
cannot keep a TAO default just because nobody typed it:

```bash
PHASE=<baseline|iter${N}>          # names the container and this phase's output tree
KPI_SPEC="${RESULTS_DIR}/${PHASE}/kpi_spec.yaml"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
  --stage kpi_analyze --ds-image "$TAO_DS_IMAGE" \
  --out "$KPI_SPEC"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$KPI_SPEC" \
  --apply-workflow-defaults <skill_root>/assets/overlays/kpi_analyze.yaml \
  --set results_dir="${RESULTS_DIR}/<phase>/kpi" \
  --set kpi.conf_threshold=<state.config.kpi_conf_threshold> \
  --set visualize.tag=<phase> \
  --set data.kpi_sources="[{image_dir: <KPI images>, ground_truth_ann_path: <KPI labels>, inference_ann_path: ${RESULTS_DIR}/<phase>/inference/labels}]" \
  --set data.mapping=<class-mapping YAML> \
  --set data.image_dir=<KPI images> \
  --set data.ann_path=<KPI labels> \
  --report-json "${RESULTS_DIR}/<phase>/kpi_overrides.json" \
  --require-no-mandatory
```

Everything in the overlay is fixed; everything in `--set` is a path or a phase
label. A `--set` naming a key the overlay already set is rejected — that
collision is the drift the split exists to prevent.

The resulting `kpi:` block:

```yaml
kpi:
  iou_threshold: 0.5
  conf_threshold: 0.0        # from state; the run's own setting — see below
  num_recall_points: 11      # 11-point interpolated AP; 101 selects COCO-style
  ignore_sqwidth: 40         # TAO emits 0, which scores smaller boxes
  filter: false
  is_internal: false
```

`conf_threshold` is the one value here the run owns. `init_deft_state.py` freezes it as
`config.kpi_conf_threshold` and every phase is scored at that same value, so the baseline
and each iteration stay comparable. Read it from state rather than retyping it.

The default is `0.0`, and that is the value to score at: inference writes its labels at
`0.0`, so scoring there covers the full PR curve. A higher threshold truncates the
low-confidence tail and produces a number that cannot be compared against a run scored at
`0.0`. `--kpi-conf-threshold` exists so a run that needs a different operating point can
say so once, at init, and have every phase honour it.

### These values are the leaf skill's defaults too

`tao-analyze-detection-kpi` ships the same `conf_threshold: 0.0`, `num_recall_points: 11`
and `ignore_sqwidth: 40`, so delegating to it and applying this overlay agree by default.
They stop agreeing on `conf_threshold` as soon as a run sets its own, which is why this
stage passes it explicitly rather than relying on the leaf's default. Apply the overlay
and the `--set` anyway: the agreement is a fact about the current versions, not a
guarantee, and a spec that states its own scoring settings is auditable after the fact.

## Scoring at more than one confidence threshold

Inference runs at `0.0`, so the labels carry every detection and this stage can score them
at any threshold without re-running inference — re-apply the overlay with
`--set kpi.conf_threshold=<t>` and a distinct `results_dir` per point, since
`kpi_calc.csv` is written unconditionally and a shared directory keeps only the last. No
`--allow-workflow-default-override` is needed: the overlay no longer pins this key.
`conf_threshold` selects from what inference wrote; it cannot recover a box inference
dropped.

Only a result at `config.kpi_conf_threshold` is comparable to the loop's reported mAP. The
loop scores every phase once, at that one value; sweeps are a diagnostic.

**`input_format` is uppercase here.** `analytics kpi_analyze` accepts only `KITTI` or `COCO`. This differs from `gap_analysis object_detection` in the same container, which takes lowercase `kitti` / `coco`. The two stages in this loop therefore spell the same format differently — that is expected, not a typo to "fix".

**Keep `visualize.platform: local`.** The `wandb` path needs credentials and a reachable endpoint; a loop stage should not depend on either.

**`is_internal` must stay `false`.** Setting it true drops every class except `person` and appends a `Summary` row, silently changing what the report means.

## Narrow the mapping to the target classes first

The user's mapping usually names more classes than the run targets. Scoring an
untargeted class adds a constant-0 AP row and averages it into the mAP, so the number
stops being comparable with any run that did not:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_mapping_for_kpi_analyze.py \
  --mapping "<the user's class mapping>" \
  --target-classes "<comma-separated target classes>" \
  --out "${RESULTS_DIR}/<phase>/kpi/mapping.yaml"
```

Pass the narrowed file as `data.mapping`, not the user's original.

## Invocation

**Launch it detached.** Runtime scales with the KPI set, and the baseline is the
slowest phase to score because inference writes its labels at `conf_threshold: 0.0`, so
every detection is present to be read. The mAP
appears *only* on stdout, and a foreground `docker run | tee` loses it if the client
dies while the container keeps running — name the container, redirect to the log, and
wait on the log with `await_stage.py`:

```bash
docker run -d --name "deft_${PHASE}_kpi" ... > /dev/null
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/await_stage.py \
  --artifact "${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv" --timeout-sec 5400
docker logs "deft_${PHASE}_kpi" > "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log" 2>&1
```

Pass `--gpus all` even though the scoring itself is CPU-bound: the TAO launcher calls
`nvidia-smi -L` unconditionally at startup, so omitting it fails with
`FileNotFoundError: 'nvidia-smi'` before any work begins.

```bash
<skill_root>/scripts/deft_python.sh <skill_bank>/skills/data/tao-analyze-detection-kpi/scripts/verify_kpi_analyze_spec.py \
  --spec "$KPI_SPEC"

docker run -d --name "deft_${PHASE}_kpi" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  analytics kpi_analyze -e "$KPI_SPEC" 2>&1
# Detached, and NOT --rm: the mAP exists only on this container's stdout, so removing
# it on exit discards the number. Wait on the artifact, then capture the log, then
# remove the container yourself:
#   await_stage.py --artifact "${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv" --timeout-sec 5400
#   docker logs "deft_${PHASE}_kpi" > "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log" 2>&1
#   docker inspect -f '{{.State.ExitCode}}' "deft_${PHASE}_kpi"   # check before trusting it
#   docker rm "deft_${PHASE}_kpi"
```

The aggregate **mAP is printed to stdout only** — nothing writes it into `kpi_calc.csv`,
which carries one row per class and no aggregate row. Do not depend on holding that stream:
the stage runs tens of minutes on a large KPI set, and a driver whose shell calls time out,
a dropped pipe, or a container reaped before `docker logs` runs all lose the one number the
loop compares phases on.

Derive it from the CSV instead. The aggregate is the unweighted mean of the per-class APs,
so it recomputes exactly:

```bash
MAP_VALUE=$(<skill_root>/scripts/deft_python.sh <skill_root>/scripts/summarize_kpi.py \
  --kpi-csv "${RESULTS_DIR}/${PHASE}/kpi/kpi_calc.csv" \
  --expect-classes <number of target classes> | tail -1)
```

It writes `kpi_summary.json` beside the CSV and prints the value for `--map-value`.
`--expect-classes` is what makes the mean trustworthy: the CSV has no class column, so a
row that is not a target class — the `Summary` row `kpi.is_internal: true` appends — would
shift the mean silently. A disagreement is an error instead.

Still capture the log. On an image predating tao-data-services#31 it is the only place the
per-row class names appear, and it remains the record of what the stage actually reported.

## Outputs

| Artifact | Path |
|---|---|
| Per-class metrics | `${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv` |
| PR curve plot | `${RESULTS_DIR}/<phase>/kpi/` |
| Captured log (class names for the CSV rows) | `${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log` |
| Aggregate mAP | `${RESULTS_DIR}/<phase>/kpi/kpi_summary.json` |

`kpi_calc.csv` columns: `Sequence Name`, `class_name`, `TP`, `FP`, `FN`, `TN`, `Pr`, `Re`,
`Acc`, `AP`.

`class_name` arrived with tao-data-services#31. On an image built before it the column is
absent and the rows are unlabeled — one per class, in the order `kpi_analyze.log` prints
them. `summarize_kpi.py` handles both: with the column it labels each AP and drops any
`Summary` row by name; without it, `--expect-classes` is the only thing standing between a
stray row and a quietly wrong mean. Pass `--expect-classes` either way.

`Sequence Name` is derived as the **second-to-last** component of `image_dir`, not configured. Keep `image_dir` free of a trailing slash and laid out so that component is the sequence identifier you want; two sources can otherwise collide under one name.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "<phase>" --stage kpi_analyze \
  --kpi-csv "${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv" \
  --kpi-log "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log" \
  --map-value "$MAP_VALUE" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "kpi: mAP=<value>"
```
