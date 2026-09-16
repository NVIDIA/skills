# DEFT OD — Mining Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-mine-od-images`. Read that skill's `SKILL.md` for the full field reference.

## When to invoke

Iteration stage 3, after `embed`. Mines the source pool for images resembling this iteration's weak set.

## Inputs from state

| State field | Description |
|---|---|
| `iterations.iter${N}.embeddings_parquet` | This iteration's `weak_images_embeddings.parquet` — the mining **target**. |
| `config.source_pool_embeddings` | Pre-embedded source pool — the mining **source**. Must use the same encoder as the target. |
| `iterations.iter$((N-1)).exclude_parquet` | Previous cumulative mined set. **Absent on iteration 1.** |
| `config.source_detection_file` | Source-pool COCO JSON — required for `class_stratified`. |
| `config.target_detection_file` | KPI COCO JSON — required for `class_stratified`. It describes the **KPI** set, not the pool: stratified allocation reads it to find how many boxes of each rare class the target set holds. If the KPI package ships one, use it. If the ground truth is KITTI, build one with `annotations convert` the same way prep converts the pool (`references/prep-source-pool.md`, step 3), pointing it at the KPI labels. Without it, use `--allocation-policy global`. |
| `config.rare_class_list` | Comma-separated rare classes, e.g. `"person,bicycle"`. |

## `exclude_path` must be `null` on iteration 1

This is a hard failure, not a warning. TAO DS `load_datasets` raises `FileNotFoundError` when `exclude_path` is set but is not a file — and on iteration 1 no cumulative parquet exists yet. Pointing at the not-yet-written path kills the stage before any mining happens.

A *spent* pool fails differently and worse. When the exclude set already covers every
pool image, `split_datasets_by_class` raises `TypeError: Series object is not iterable`
from `selection.py` — an unhandled cudf error naming neither the pool nor the exclude
set. Count the pool rows not already excluded **before** invoking, and stop the loop
instead of calling the miner with nothing left to give
(`commit_stage.py --stage loop_stop --pool-exhausted --pool-remaining N`).

- **Iteration 1**: emit `exclude_path: null`.
- **Iteration N > 1**: emit the absolute path to `${RESULTS_DIR}/iter$((N-1))/mined_cumulative.parquet`, and confirm the file exists before writing the spec.


## `detection_format` is never inferred

TAO DS raises `detection_format is required (coco or kitti) when a detection file is provided` rather than guessing. Set it explicitly to `coco` and supply COCO JSONs — a KITTI label directory passed as a detection file would be misparsed.

Note this stage uses **COCO** detection files while `gap_analysis` in the same loop reads **KITTI** inference labels. That split is correct: they are different inputs to different steps.

## Rare classes

`class_stratified` allocation needs to know which classes are rare. The loop derives it from
`pool_report.json`: any target class holding a below-mean share of the pool's annotations.
Scarcity in the **pool** is the signal, not scarcity in the KPI set — stratified allocation
exists so classes the pool holds few of still get their share of the budget.

The derivation is reported with the counts behind it, so the choice is visible rather than
silent. It belongs in `tmm unique_neighbor_matching` itself, which already reads both inputs
it would need; tracked separately as a TAO DS request.

## Spec

Write per-iteration under `${RESULTS_DIR}/iter${N}/mining/unique_neighbor_matching.yaml` — the invocation below reads it from
`$MINE_SPEC`, so bind the two:

```bash
MINE_SPEC="${RESULTS_DIR}/iter${N}/mining/unique_neighbor_matching.yaml"
```


```yaml
source_path:            <config.source_pool_embeddings>
target_path:            <iterations.iter${N}.embeddings_parquet>
output_dir:             <absolute path to ${RESULTS_DIR}/iter${N}/mining>
desired_unique_count:   <from prepare_budget_for_mining.py — see below>
allocation_policy:      class_stratified     # or global — see below
distance_metric:        euclidean
candidate_expansion_factor: 5
source_embedding_column: embedding
target_embedding_column: embedding
source_filepath_column: filepath
target_filepath_column: filepath
exclude_path:           null                 # iteration 1 ONLY; else the prev cumulative path
source_detection_file:  <config.source_detection_file>
target_detection_file:  <config.target_detection_file>
detection_format:       coco
rare_class_list:        "person,bicycle"
save_embeddings:        false
visualize:              false
```

**Allocation policy.** Use `class_stratified` when the user supplied rare classes — budget is apportioned by target class ratio using largest-remainder, with a non-rare fallback pool, and each source is owned by the first class that claims it. Use `global` when no rare classes are configured. The reference pipeline's `class_balanced` mode maps to `class_stratified`.

**`rare_class_list` comes from state, not from this stage.** Read
`config.rare_class_list` and write it into the spec verbatim. It is derived once, at
the prep commit, from the pool's own annotation counts — every target class holding a
below-mean share. Under `class_stratified` the commit refuses a `mine` while it is
still unset, since allocation apportions the budget by exactly that list.

**`candidate_expansion_factor` is not the loop's iteration count.** It seeds the miner's internal candidate-pool growth (starting at 5, widening each internal retry, capped at 8 internal passes). It is unrelated to `max_iterations`.

## Invocation

```bash
<skill_root>/scripts/deft_python.sh <skill_bank>/skills/data/tao-mine-od-images/scripts/verify_unique_neighbor_matching_spec.py \
  --spec "$MINE_SPEC"

docker run --rm --name "deft_iter${N}_mine" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  tmm unique_neighbor_matching -e "$MINE_SPEC"
```

## Outputs

| Artifact | Path |
|---|---|
| `final_unique_files.parquet` | `${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet` |
| `summary.json` | `${RESULTS_DIR}/iter${N}/mining/summary.json` |

`final_unique_files.parquet` is a single column named by `source_filepath_column` (`filepath`). It feeds the `stage` step.

Read `coverage_pct` from `summary.json`:

- `retrieved_unique_count == 0` with weak images present → **hard stop**. The source pool has nothing left to give, or the exclude set has consumed it.
- `coverage_pct < 50` → warn in the iteration summary; the pool is running dry and later iterations will add little.
- Otherwise proceed.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage mine \
  --mining-output "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --mining-summary "${RESULTS_DIR}/iter${N}/mining/summary.json" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "mined <retrieved>/<desired> unique images (<coverage_pct>% coverage)"
```

## Computing `desired_unique_count`

The budget is produced before this stage by `prepare_budget_for_mining.py`, and its full
invocation lives in `references/stage-mined-data.md`. Reading only this overlay leaves
the field with no documented way to compute it, so the short form is repeated here:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_budget_for_mining.py \
  --weak-parquet "${RESULTS_DIR}/iter1/gaps/weak_images.parquet" \
  --multiplier "<multiplier>" --pool-size "<pool rows>" \
  --already-mined "<cumulative exclude rows, 0 on iteration 1>" \
  --remaining-iterations "<iterations left, including this one>"
```

Seed it from **iteration 1's** weak parquet on every iteration, so the amount of data
added per iteration stays constant as the gap set shrinks.
