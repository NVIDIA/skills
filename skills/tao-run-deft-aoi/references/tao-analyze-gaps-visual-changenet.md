# DEFT AOI RCA (VCN) — DEFT Loop Reference

Read this when the parent runs the `rca` stage on a VCN Classify inference CSV.
The underlying skill `tao-skill-bank:tao-analyze-gaps-visual-changenet` (`skills/data/tao-analyze-gaps-visual-changenet/SKILL.md`)
owns the full gap analysis contract: threshold sweep, weakness ranking, per-lighting
expansion, visual spot-check, and report format. This file only covers the
DEFT-loop-specific overlay: required inputs, output directory layout, and
`deft_state.json` / `loop_log.jsonl` updates.

## DEFT-Loop Inputs

Pass these as Hydra overrides to the `gap_analysis vcn_aoi` container (see `skills/data/tao-analyze-gaps-visual-changenet/SKILL.md` for the full `docker run` line):

- `inference_results_dir` — **directory** containing `inference.csv` (e.g. `${RESULTS_DIR}/<iter>/inference/best_val/`), not the CSV path. The container reads `inference_results_dir/inference.csv`. Required CSV columns: `input_path`, `object_name`, `label`, `siamese_score`. Use the inference subdirectory recorded in `deft_state.json` (`best_val` or `latest`).
- `train_config` — VCN train YAML from the experiment directory; provides `dataset.classify.input_map` (lighting list) and `dataset.classify.image_ext` for per-lighting expansion
- `kpi_media_path` — dataset image root prepended to relative `input_path` entries in the CSV
- `results_dir` — RCA output directory (this loop uses `${RESULTS_DIR}/<baseline|iter${N}>/rca_results/<timestamp>/`); the container writes `kpi_gaps.parquet`, `threshold.txt`, `metrics.json`, `weak_samples_breakdown.txt` here
- `min_recall` — from loop KPI target (default `1.0`; zero-miss)
- `top_k_per_label` — augmentation budget per label (default `50`); always pass an explicit positive integer

## Output Directory

`results/<baseline|iter${N}>/rca_results/<timestamp>/`

Required files:
- `kpi_gaps.parquet` — top-K weakest per label, expanded per lighting (columns: `filepath`, `label`, `siamese_score`, `weakness`)
- `threshold.txt` — chosen decision threshold (single float)
- `metrics.json` — confusion matrix + per-label distribution stats at chosen threshold
- `weak_samples_breakdown.txt` — per-label count / misclassified / marginal counts
- `rca_images/` — thumbnails of the 10 spot-checked weak samples

If the model cannot reach `min_recall` at any threshold, `unreachable_kpi.txt` is written instead of `kpi_gaps.parquet`. When this file exists, skip the spot-check and write the abridged report — do not attempt routing or mining.

## Output to deft_state.json

```python
# For baseline:
state["baseline"]["rca_target_defects"] = [...]         # labels with FN / high-FP, sorted by impact
state["baseline"]["rca_gaps_parquet"]   = "<abs_path>/kpi_gaps.parquet"
state["baseline"]["rca_threshold"]      = <float>
# For iter N:
state["iterations"][f"iter{N}"]["rca_target_defects"] = [...]
state["iterations"][f"iter{N}"]["rca_gaps_parquet"]   = "<abs_path>/kpi_gaps.parquet"
state["iterations"][f"iter{N}"]["rca_threshold"]      = <float>
```

`rca_target_defects`: list of label strings present in misclassified / high-weakness samples, sorted by impact (FN count descending, then FP rate descending). The downstream routing stage reads `rca_gaps_parquet` directly from disk — write the absolute path here, not a relative one.

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label <baseline|iter${N}> \
    --stage rca --status ok \
    --summary "RCA (VCN): threshold=X recall=Y; gaps=K rows across N labels"
```
