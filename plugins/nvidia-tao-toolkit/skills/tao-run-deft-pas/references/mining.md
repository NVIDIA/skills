# Pool Embedding, Mining, and History Selection

Read only when the audit selects `pool_embed`, `data_mining`, or
`history_select`. Every `run_deft_action.py prepare` call only writes the action
request. Immediately execute and finalize it through
`platform-execution.md` before committing or starting the next step. All
container platforms use the immutable `/specs` mount from that request.

## Contents

- [Baseline pool embedding](#baseline-pool-embedding)
- [Iteration data mining](#iteration-data-mining)
- [History-aware selection](#history-aware-selection)

## Baseline pool embedding

Embed the dataset-setup caption pool once:

```bash
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi
POOL_DIR="$RESULTS_DIR/embeddings/source"
POOL_OUT="$POOL_DIR/embeddings.parquet"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$POOL_DIR" --name pool_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$POOL_OUT" -- \
    embedding text_embeddings -e /specs/text_embed_spec.yaml \
    input_parquet=/results/embeddings/source/source_pool.parquet \
    output_parquet=/results/embeddings/source/embeddings.parquet
```

Commit with the exact generated status file:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label baseline --stage pool_embed \
    --pool-embeddings-parquet "$POOL_OUT" \
    --pool-embed-command-status "$POOL_DIR/pool_embed.status.json" \
    --summary "PAS caption pool embedded"
```

On resume, an existing parquet is reusable only when the matching status has
exit code zero, names that file as a fresh output, and passes commit/audit
schema and row checks. Otherwise prepare the one permitted retry; the producer
removes only the exact output before launch. File existence alone is never a
cache hit.

## Iteration data mining

For iteration N, `iter_N/gaps/kpi_gaps.parquet` was created by the prior
label's gap analysis. Run these three steps in order, fully executing and
finalizing each prepared action before its consumer step.

### 1. Embed target gap captions

```bash
ITER_DIR="$RESULTS_DIR/iter_$N"
TARGET_DIR="$ITER_DIR/embeddings/target"
TARGET_OUT="$TARGET_DIR/embeddings.parquet"
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$TARGET_DIR" --name target_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$TARGET_OUT" -- \
    embedding text_embeddings -e /specs/text_embed_spec.yaml \
    input_parquet=/results/iter_$N/gaps/kpi_gaps.parquet \
    output_parquet=/results/iter_$N/embeddings/target/embeddings.parquet
```

### 2. Mine source neighbors

```bash
MINING_DIR="$ITER_DIR/mining"
MINED_OUT="$MINING_DIR/mined_samples.parquet"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$MINING_DIR" --name knn \
    --fresh-output "$MINED_OUT" -- \
    tmm nearest_neighbors -e /specs/mining_spec.yaml \
    source_parquet=/results/embeddings/source/embeddings.parquet \
    target_parquet=/results/iter_$N/embeddings/target/embeddings.parquet \
    output_parquet=/results/iter_$N/mining/mined_samples.parquet \
    topn="$MINING_TOPN" knn_metric="$KNN_METRIC"
```

`MINING_TOPN` and `KNN_METRIC` must equal immutable state/config. Do not tune
them between iterations without starting a separately approved run.

### 3. Convert candidates

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_pas_stage.py" mining-postprocess \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml" --iter-num "$N"
```

With history enabled, uncapped candidates live under
`mining/history_candidates/`; history selection owns the budget. With history
disabled, the adapter writes the budgeted outputs directly under `mining/`.
Zero gaps, zero mined rows, missing embeddings, invalid parquet schema, or an
empty candidate-pairs JSON is a hard stop.

Commit data mining:

```bash
CANDIDATE_DIR="$MINING_DIR"
if [ "$HISTORY_AWARE" = true ]; then
  CANDIDATE_DIR="$MINING_DIR/history_candidates"
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" --stage data_mining \
    --target-embeddings-parquet "$TARGET_OUT" \
    --target-embed-command-status "$TARGET_DIR/target_embed.status.json" \
    --mined-parquet "$MINED_OUT" \
    --knn-command-status "$MINING_DIR/knn.status.json" \
    --candidate-pairs "$CANDIDATE_DIR/mined_pairs.json" \
    --mining-postprocess-status \
      "$MINING_DIR/mining-postprocess.host.status.json" \
    --summary "iter$N gap captions embedded and neighbors mined"
```

## History-aware selection

Run the adapter without `--resume` on a clean attempt:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_pas_stage.py" history-select \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml" \
    --iter-num "$N"
```

It partitions candidates into novel/replay samples, spends the configured
budget, updates cumulative names, and compares mined/eval basenames. Any eval
overlap is a hard stop. When history is enabled, its root ledger must contain
an entry for exactly N.

If the adapter reports that N is already committed while the DEFT audit still
selects `history_select`, rerun once with `--resume` as documented in
`scripts-and-agents.md`. Never delete selection artifacts or edit the ledger.

Commit:

```bash
HISTORY_ARGS=()
if [ "$HISTORY_AWARE" = true ]; then
  HISTORY_ARGS=(--mining-history "$RESULTS_DIR/mining_selection_history.json")
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" \
    --stage history_select \
    --mined-image-list "$MINING_DIR/mined_image_list.txt" \
    --mined-pairs "$MINING_DIR/mined_pairs.json" \
    --mined-manifest "$MINING_DIR/mined_dataset.json" \
    --cumulative-names "$MINING_DIR/cumulative_mined_unique_names.json" \
    --history-select-status "$MINING_DIR/history-select.host.status.json" \
    "${HISTORY_ARGS[@]}" \
    --summary "iter$N history-aware mining selection completed"
```

Run the audit after each commit. Advancing without a successful stage commit
is prohibited.
