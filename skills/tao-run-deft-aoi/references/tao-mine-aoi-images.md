# DEFT AOI Mining — DEFT Loop Reference

Read this when the parent runs the `data_mining` stage (embed-then-mine workflow).
The underlying skill `tao-skill-bank:tao-mine-aoi-images` (`skills/data/tao-mine-aoi-images/SKILL.md`)
owns the full docker invocation (three calls into the pinned TAO data-services
image), encoder consistency requirement,
output schema, and common pitfalls. This file only covers the DEFT-loop-specific
overlay: required inputs, three-step order, output layout, and
`deft_state.json` updates.

## DEFT-Loop Inputs

- `target_parquet` — absolute path from `deft_state.json` (`routing_mining_parquet` field set by the routing stage); required columns: `filepath` (and `label` if `filter_by_label=true`)
- `source_pool_parquet` — parquet of candidate images to mine against with a `filepath` column; convert from CSV up front if needed (preserve `filepath` and `label`)
- `model` — embedding model: `CLIP`, `SigLIP`, or a TAO `.pth`/`.ckpt` checkpoint; default `SigLIP`
- `model_path` — the exact `SIGLIP_MODEL_PATH` resolved during Pre-Flight; do not re-resolve it here. In air-gap or DNS-limited runs this must be a local `google/siglip-base-patch16-224` snapshot containing `config.json`, whether it lives under the workspace or an external `HF_HOME` such as `/scratch/hf-cache`. Bind-mount the snapshot at the identical absolute path (`-v "$SIGLIP_MODEL_PATH:$SIGLIP_MODEL_PATH:ro"`) and put that same path in `embedding_spec.yaml`. Never point SigLIP at a DINO or C-RADIO cache. Use the HuggingFace ID only after outbound HuggingFace access has been verified.
- `topn` — nearest neighbours per target (default `5`). Preserve an explicit
  user value; increase it only when history filtering repeatedly leaves too few
  novel candidates.
- `knn_metric` — `cosine` (default, recommended for CLIP/SigLIP), `euclidean`, or `manhattan`
- `min_similarity` — cosine similarity cutoff used at retention time. Read from `state.config.mining_filter.min_similarity` in `deft_state.json`; fall back to `0.9` only when the field is unset/null. **Always log the value actually used** into `knn_summary.csv` (`similarity_threshold` column) so the report shows what cutoff produced the row count, not the prose-default.
- `filter_by_label` — `true` or `false` (default `false`); requires `label` in both embedding parquets

If `routing_mining_parquet` is absent from `deft_state.json` or the file does not exist on disk, stop and return failure without running any docker steps.

Start from the source CSV recorded in state or explicitly supplied by the
user/harness. In the common workspace layout it is
`<workspace>/augmentation/mining_pool/mining_pool.csv`, while its images and
golden pairs resolve under the shared `state.config.images_dir` (normally
`<workspace>/images`). These are discovery hints, not evidence that a path
exists. Inspect the CSV fields and verify the resolved files; do not infer an
`augmentation/mining_pool/images/` directory solely from the CSV's parent
directory. If the selected source CSV is absent, empty, or its declared paths
cannot be resolved, stop in Pre-Flight; never scan the workspace image tree to
invent source rows.

Before converting CSV to parquet, run `"$PYTHON" scripts/resolve_mining_pool.py` with
the three paths persisted in state. The resolver checks direct paths, paths
relative to the independent images root, golden-directory plus basename, and
the ChangeNet object/light filename form. It writes one canonical `filepath`
and hard-stops on zero or multiple matches. Downstream stages consume only
`config.resolved_mining_pool_csv`; never discover paths by scanning snapshots
or assuming the CSV's parent owns an `images/` directory.

For selected PASS rows in OK-only source pools, stage the source OK image as the
golden pair when no separate golden tree exists; for selected non-PASS rows,
missing golden files remain a hard stop.

## Pre-mine yield precheck (cheap; runs before Step 1 embedding)

Run this on the host before spending GPU time on Step 1+2. For each label in `target_parquet`, count rows in `source_pool_parquet` (or the source CSV) with the same label. If any target label has **zero** source-pool rows of the same label, log a warning and surface it to the user:

```
Pre-mine precheck: target labels {missing} have 0 candidates in mining_pool —
guaranteed 0 yield regardless of similarity. Consider expanding mining_pool.csv
or routing these labels to AnomalyGen exclusively.
```

This is a warning, not a hard stop — k-NN by embedding can still pull rows of a *different* nominal label when their visual content matches (it's the post-routing decision that filters by label, not the source pool itself). But making the zero-coverage cases visible up-front gives the user a chance to fix the pool before the next iteration, instead of discovering it via the post-mine yield monitor below.

## Four-Step Execution Order

1. **Embed targets** (`embedding image_embeddings … input_parquet=<target_parquet>`) → `target_embeddings.parquet`
2. **Embed source pool** (`embedding image_embeddings … input_parquet=<source_pool_parquet>`) → `source_embeddings.parquet`; use the **identical** `model` and `model_path` as Step 1
3. **Mine nearest neighbours and apply the cosine floor** (`tmm nearest_neighbors …`) → `mined_raw.parquet`, then the existing cosine retention step → `mined_candidates.parquet` + `knn_summary.csv`. Keep both immutable.
4. **Drop samples mined in earlier iterations** with the mapped data skill's
   `scripts/filter_mined_history.py` → final `mined.parquet`, per-iteration
   `mining_history_summary.json`, and the run-level
   `${RESULTS_DIR}/mining_history.json` ledger. Pass the 1-based iteration
   number and `state.config.mining_filter.top_k_per_target`; use `--resume`
   only when the ledger already contains this exact iteration.

The first three steps use the pinned TAO data-services image (set as `$DS_IMAGE` at the top of the run — see `skills/data/tao-mine-aoi-images/SKILL.md` § Setup); Step 4 runs on the host. Mount the workspace root at an identical path inside the container (`-v $WORKSPACE:$WORKSPACE`) so absolute paths in parquet args resolve the same on both sides.

Capture each invocation's complete stdout/stderr in a separate immutable log:
`target_embeddings.log`, `source_embeddings.log`, and `nearest_neighbors.log`.
Preserve the Docker return code (do not hide it behind `| tail`), then require
both `rc=0` and `Execution status: PASS`; any `Execution status: FAIL` is a
hard stop even when Docker exits zero. In air-gap mode pass
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Run host-side parquet and CSV
checks through the interpreter selected by
`bash <skill_root>/scripts/deft_python.sh`, not a
similarly named path guessed under the mapped data skill.

Use this capture shape for each mapped command (replace `<docker command>` and
`$LOG`; never append a retry to a failed log):

```bash
set -o pipefail
<docker command> 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] || grep -q 'Execution status: FAIL' "$LOG" ||
   ! grep -q 'Execution status: PASS' "$LOG"; then
  echo "FATAL: TAO stage failed (rc=$rc, log=$LOG)" >&2
  exit 2
fi
```

**Pre-create `experiment_specs/`.** Both `embedding image_embeddings` and `tmm nearest_neighbors` are Hydra-driven and abort with `Primary config directory not found` if no `experiment_specs/` directory exists at the container's working dir. The container does not auto-create it. Before each docker run, `mkdir -p <mining_dir>/experiment_specs/` on the host (the mount makes it visible inside the container), or pass `-w <mining_dir>` and let Hydra find an empty dir there. An empty directory is sufficient — the CLI supplies its own spec via flags. Without this, both steps 1+2 (embedding) and step 3 (mining) fail with the same opaque Hydra error.

## Output Directory

`${RESULTS_DIR}/iter${N}/mining_results/<timestamp>/`

Required files:
- `mined_raw.parquet` — immutable TAO k-NN output before cosine/history filters
- `mined_candidates.parquet` — cosine-qualified candidates before history filtering
- `mined.parquet` — final novel source filepaths (columns: `filepath`); this is
  the only mining parquet allowed into the current iteration's training rows
- `mining_summary.txt` — TAO-emitted read-only query/neighbour summary; do not overwrite it from the host. Write host-side cosine-filter counts to `knn_summary.csv` (and, if needed, `host_mining_summary.txt`).
- `mining_history_summary.json` — current candidate/new/duplicate counts,
  recorded `topn`, artifact hashes, and an increase-`topn` recommendation when
  the candidate neighborhood is dominated by prior selections
- `${RESULTS_DIR}/mining_history.json` — run-level append-only ledger written
  only by `filter_mined_history.py`
- `target_embeddings.parquet` — Step 1 output (reusable across future mining runs against the same targets)
- `source_embeddings.parquet` — Step 2 output (reusable against the same source pool)
- `target_embeddings.log`, `source_embeddings.log`, `nearest_neighbors.log` — complete container logs whose final TAO status is `PASS`

Before committing, verify both embedding parquets have non-zero rows and the
columns `filepath` and `embedding`; verify the mined parquet has `filepath`.
`knn_summary.csv` must have exactly one data row with `candidate_count =
kept_count + rejected_count`, `kept_count` equal to the pre-history
`mined_candidates.parquet` row count, and a numeric `similarity_threshold`.
Keep this cosine summary immutable; `mining_history_summary.json` separately
records the prior-selection rejection count, `topn`, and final novel row count.
Here `candidate_count` means the unique rows in TAO's unfiltered
`mined_raw.parquet` after neighbour deduplication, not
`target_count * topn`; the latter belongs only in TAO's read-only
`mining_summary.txt`. Placeholder parquets, copied
source rows, handwritten PASS text without the three real outputs, or a summary
whose counts disagree are not valid stage artifacts.

Run the history filter after cosine retention:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" \
  <bank_root>/skills/data/tao-mine-aoi-images/scripts/filter_mined_history.py \
  --candidate-parquet <abs>/mined_candidates.parquet \
  --output-parquet <abs>/mined.parquet \
  --history-file "${RESULTS_DIR}/mining_history.json" \
  --summary <abs>/mining_history_summary.json \
  --iteration "${N}" \
  --topn "${TOPN}"
```

If the final parquet is empty but AnomalyGen produced rows, continue with the
synthetic contribution and surface the history summary warning. If both
producers add zero new rows, hard-stop instead of retraining an unchanged
dataset. Increasing `topn` can expose novel neighbors; it cannot help when the
source pool itself has no remaining variance.

## Mined rows → ChangeNet CSV

`mined.parquet` holds source **file** paths (e.g. `images/BOARD/comp_SolderLight.jpg`). ChangeNet's siamese dataloader does **not** open that path directly — it builds `{images_dir}/{input_path}/{object_name}_{light}{image_ext}`, so when turning a mined filepath into a training row:

- `input_path` = the **directory** of the file (`images/BOARD/`), not the file itself.
- `object_name` + `{light}` + `{image_ext}` must reconstruct the file's basename (`comp_SolderLight.jpg`). Carry `object_name` from the source pool row, or derive it by stripping the trailing `_{light}{image_ext}`.
- `golden_path` = the paired golden **directory**, rewritten to be workspace-root-relative (the per-iter training spec sets `images_dir` to the workspace root). For selected PASS rows from an OK-only mining pool, copy the source image into both staged mined-input and staged mined-golden directories, then set `golden_path` to the staged mined-golden directory. Do not apply this fallback to non-PASS rows.

Both `input_path` and `golden_path` need this file→directory collapse — not just `golden_path`. `scripts/validate_training_csv.py` reconstructs the full siamese path and hard-stops if a row doesn't resolve, so a missed conversion is caught before training rather than mid-run.

## Pool Composition Requirement

`augmentation/mining_pool/mining_pool.csv` must contain **NG samples** for every defect type listed in the KPI testing set — not just PASS samples. The mining stage retrieves nearest neighbours by SigLIP embedding similarity, so if the pool has zero NG examples for a defect type, no candidate ever crosses the configured `min_similarity` threshold and the iteration silently contributes no real-image augmentation for that type. Document defect-type coverage in the workspace setup; do not work around in code. Past production pools have been missing `SHIFT`, `LIFTED_LEAD`, `UPSIDE_DOWN`, `TOMBSTONE`, and `POLARITY` simultaneously, which leaves 5/8 KPI defect types with no augmentation path.

## Yield Monitor

After Step 4 finishes, read `mining_filter/knn_summary.csv` and
`mining_history_summary.json`. For N>1, compare
`kept_count` with
`state["iterations"][f"iter{N-1}"]["mining_mined_count"]`; iter1 has no
prior mining yield and skips this comparison. If
`current_kept < 0.5 * previous_kept` (a >50% drop), surface a warning to the
user including both counts and the implied drop percentage:

```
Mining yield dropped {drop_pct}% (iter{N-1}: {prev_kept} → iter{N}: {cur_kept}) —
pool near exhaustion for the current weak-sample targets.
Consider expanding mining_pool.csv with new production samples before the next iteration.
```

This is a warning, not a hard stop. The loop should continue, but the iteration summary must flag the drop so the user notices before the next iteration. A 30→5 collapse in iter2 (83% drop) has happened in past runs without any signal reaching the user.

## Output to deft_state.json

```python
phase = state["iterations"][f"iter{N}"]
phase["mining_mined_parquet"] = "<abs_path>/mined.parquet"
phase["mining_candidate_parquet"] = "<abs_path>/mined_candidates.parquet"
phase["mining_mined_count"] = <int>  # rows in mined.parquet
phase["mining_summary"] = "<abs_path>/knn_summary.csv"  # host-authored cosine-filter summary; keep TAO mining_summary.txt read-only
phase["mining_history"] = "<results_dir>/mining_history.json"
phase["mining_history_summary"] = "<abs_path>/mining_history_summary.json"
phase["mining_target_embeddings"] = "<abs_path>/target_embeddings.parquet"
phase["mining_source_embeddings"] = "<abs_path>/source_embeddings.parquet"
phase["mining_target_log"] = "<abs_path>/target_embeddings.log"
phase["mining_source_log"] = "<abs_path>/source_embeddings.log"
phase["mining_knn_log"] = "<abs_path>/nearest_neighbors.log"
phase["stage_completed"] = "data_mining"
```

This snippet documents the schema only; use `commit_stage.py` for the write.

## Log Stage

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label iter${N} \
    --stage data_mining \
    --mining-parquet <absolute path to mined.parquet> \
    --mining-candidates <absolute path to mined_candidates.parquet> \
    --mining-summary <absolute path to knn_summary.csv> \
    --mining-history "${RESULTS_DIR}/mining_history.json" \
    --mining-history-summary <absolute path to mining_history_summary.json> \
    --mining-target-embeddings <absolute path to target_embeddings.parquet> \
    --mining-source-embeddings <absolute path to source_embeddings.parquet> \
    --mining-target-log <absolute path to target_embeddings.log> \
    --mining-source-log <absolute path to source_embeddings.log> \
    --mining-knn-log <absolute path to nearest_neighbors.log> \
    --mining-count <int> \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "Mining (VCN): mined=N_mined source images for N_targets targets"
```
