# DEFT AOI RCA (VCN) — DEFT Loop Reference

Read this when the parent runs the `rca` stage on a VCN Classify inference CSV.
The underlying skill `tao-skill-bank:tao-analyze-gaps-visual-changenet` (`skills/data/tao-analyze-gaps-visual-changenet/SKILL.md`)
owns the full gap analysis contract: threshold sweep, weakness ranking,
per-lighting expansion, mandatory visual follow-up, and report format. This file only covers the
DEFT-loop-specific overlay: required inputs, output directory layout, and
`deft_state.json` updates.

## DEFT-Loop Inputs

Pass these as Hydra overrides to the `gap_analysis vcn_aoi` container (see `skills/data/tao-analyze-gaps-visual-changenet/SKILL.md` for the full `docker run` line):

- `inference_results_dir` — **directory** containing `inference.csv` (e.g. `${RESULTS_DIR}/<iter>/inference/best_val/`), not the CSV path. The container reads `inference_results_dir/inference.csv`. Required CSV columns: `input_path`, `object_name`, `label`, `siamese_score`. Use the inference subdirectory recorded in `deft_state.json` (`best_val` or `latest`).
- `train_config` — VCN train YAML from the experiment directory; provides `dataset.classify.input_map` (lighting list) and `dataset.classify.image_ext` for per-lighting expansion
- `kpi_media_path` — dataset image root prepended to relative `input_path` entries in the CSV
- `results_dir` — RCA output directory (this loop uses `${RESULTS_DIR}/<baseline|iter${N}>/rca_results/<timestamp>/`); the container writes `kpi_gaps.parquet`, `threshold.txt`, and `weak_samples_breakdown.txt` here
- `min_recall` — from loop KPI target (default `1.0`; zero-miss)
- `top_k_per_label` — augmentation budget per label (default `50`); always pass an explicit positive integer

## Output Directory

`${RESULTS_DIR}/<baseline|iter${N}>/rca_results/<timestamp>/`

The three artifact classes below are sourced from
`references/rca-artifact-manifest.json` in this application skill.

**Container-produced required:**

- `kpi_gaps.parquet` — top-K weakest per label, expanded per lighting (columns: `filepath`, `label`, `siamese_score`, `weakness`)
- `threshold.txt` — chosen decision threshold (single float)
- `weak_samples_breakdown.txt` — per-label count / misclassified / marginal counts

**Agent-produced required:**

- `RCA_Report.md` — the seven-section report from the gaps skill's
  `references/output-template.md`; the abridged `## KPI Unreachable` form is
  permitted only beside a non-empty `unreachable_kpi.txt`.

**Agent-produced conditionally required:**

- `rca_images/` — must contain at least one non-empty image after the visual
  spot check; it may be absent only beside `unreachable_kpi.txt`.

The 7.2 container does not generate `metrics.json`, `RCA_Report.md`, or
`rca_images/`. `metrics.json` stays outside the contract; the report and image
evidence are mandatory agent work. The RCA stage commit loads the same
manifest and rejects a successful commit until the applicable artifacts are
present and valid.

If the model cannot reach `min_recall` at any threshold, `unreachable_kpi.txt` is written instead of `kpi_gaps.parquet`. When this file exists, skip the spot-check and write the abridged report — do not attempt routing or mining.

The gaps skill's Steps 5–7 are mandatory before commit: perform the visual
spot check and create `rca_images/` on a reachable run, then write
`RCA_Report.md` (or write the abridged report when the KPI is unreachable).

## Output to deft_state.json

```python
label = "baseline" if running_baseline else f"iter{N}"
phase = state["iterations"].setdefault(label, {"status": "in_progress"})
phase["rca_target_defects"] = [...]  # labels with FN / high-FP, sorted by impact
phase["rca_gaps_parquet"] = "<abs_path>/kpi_gaps.parquet"
phase["rca_threshold_file"] = "<abs_path>/threshold.txt"
phase["rca_breakdown_file"] = "<abs_path>/weak_samples_breakdown.txt"
phase["rca_report"] = "<abs_path>/RCA_Report.md"
phase["rca_images_dir"] = "<abs_path>/rca_images"
phase["rca_threshold"] = <float>
phase["stage_completed"] = "rca"
```

For an unreachable KPI, state records `rca_report` and
`rca_unreachable_kpi`; the success-only container fields and
`rca_images_dir` are absent.

`rca_target_defects`: list of label strings present in misclassified / high-weakness samples, sorted by impact (FN count descending, then FP rate descending). The downstream routing stage reads `rca_gaps_parquet` directly from disk — write the absolute path here, not a relative one.
The CLI form is `[--rca-target-defect <label>]...`: each occurrence accepts
exactly one non-empty label. Repeat the flag for every label. Duplicate labels
are stored once in first-seen order.
The snippet documents the schema only; do not run it as inline Python. Commit
the path and threshold with the command below.

## Log Stage

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label <baseline|iter${N}> \
    --stage rca \
    --rca-gaps <absolute path to kpi_gaps.parquet> \
    --rca-report <absolute path to RCA_Report.md> \
    --rca-threshold <float> \
    --rca-target-defect missing \
    --rca-target-defect shift \
    --rca-target-defect lifted_lead \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "RCA (VCN): threshold=X recall=Y; gaps=K rows across N labels"
```
