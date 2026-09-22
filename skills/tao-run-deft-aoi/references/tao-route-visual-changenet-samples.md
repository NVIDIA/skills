# DEFT AOI Routing (VCN) — DEFT Loop Reference

Read this when the parent runs the `routing` stage to split RCA gaps into
per-augmentation-module subsets. The underlying skill
`tao-skill-bank:tao-route-visual-changenet-samples` (`skills/data/tao-route-visual-changenet-samples/SKILL.md`)
owns the full routing contract: label eligibility for each module, the Python
recipe (two `.isin(...)` masks), per-label routing breakdown, and report format.
This file only covers the DEFT-loop-specific overlay: required inputs, output
layout, and `deft_state.json` updates.

## DEFT-Loop Inputs

- `gaps_parquet` — absolute path from `deft_state.json` (`rca_gaps_parquet` field set by the RCA stage); required columns: `filepath`, `label`
- `source_pool_csv` — the required VCN-format
  `<workspace>/augmentation/mining_pool/mining_pool.csv`, with a `label`
  column. Never pass an empty string or substitute KPI rows.
- `anomalygen_supported_labels` — default `{"PASS", "EXCESS_SOLDER", "MISSING", "BRIDGE"}`; override only if AnomalyGen generator coverage has changed

If `rca_gaps_parquet` is absent from `deft_state.json` or the file does not exist on disk, stop and return failure — do not invent a path.

## Output Directory

`${RESULTS_DIR}/iter${N}/routing_results/<timestamp>/`

Required files:
- `mining_gaps.parquet` — subset routed to k-NN Mining (same schema as input `gaps.parquet`; may be empty)
- `anomalygen_gaps.parquet` — subset routed to AnomalyGen/Cosmos SDG (same schema; may be empty)
- `routing_summary.txt` — per-label routing decisions and dropped-label warnings

## Output to deft_state.json

```python
phase = state["iterations"].setdefault(f"iter{N}", {"status": "in_progress"})
phase["routing_mining_parquet"] = "<abs_path>/mining_gaps.parquet"
phase["routing_anomalygen_parquet"] = "<abs_path>/anomalygen_gaps.parquet"
phase["stage_completed"] = "routing"
```

Always write both paths, even when a subset is empty — downstream stages read
these fields unconditionally and record a documented branch skip. If both
subsets are empty (all labels dropped), set `phase["status"] = "failed"`, stop
after writing the report and state, log `status=error`, and surface the
dropped-label list.
The snippet documents the schema only; use `commit_stage.py` for the write.

## Log Stage

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label iter${N} \
    --stage routing \
    --routing-mining <absolute path to mining_gaps.parquet> \
    --routing-anomalygen <absolute path to anomalygen_gaps.parquet> \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "Routing: mining=N_mn rows, anomalygen=N_ag rows; N_drop labels dropped"
```
