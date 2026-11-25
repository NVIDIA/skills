# Visualization

Visualization is one ordered stage with two independently configured outputs:
contact sheets (`visualize`) and an embedding t-SNE
(`visualize_embeddings`). It is the only skippable stage.

Every `run_deft_action.py prepare` call only writes an action request. Execute
and finalize each one through `platform-execution.md` before starting the next
embedding or the finish adapter.

## Contents

- [Prepare host artifacts](#prepare-host-artifacts)
- [Embed image sets](#embed-image-sets)
- [Finish and commit](#finish-and-commit)

If both approved flags are false, commit the stage with `--skip` and no
artifacts. If either is true, a failure is not silently downgraded to a skip.

## Prepare host artifacts

```bash
ITER_DIR="$RESULTS_DIR/iter_$N"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_pas_stage.py" visualize-prepare \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml" --iter-num "$N"
```

When contact sheets are enabled, this must populate
`iter_N/visualization/samples/`. When embedding visualization is enabled, it
creates exact input parquets for weak and mined images, plus a previous-data
pool only when prior continual training data exists.

## Embed image sets

Skip this section when `visualize_embeddings=false`. Otherwise launch weak and
mined embedding commands, each with its exact output marked fresh:

```bash
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi
WEAK_DIR="$ITER_DIR/embeddings/viz_weak"
WEAK_OUT="$WEAK_DIR/embeddings.parquet"
MINED_EMBED_DIR="$ITER_DIR/embeddings/augmented"
MINED_EMBED_OUT="$MINED_EMBED_DIR/mined_embeddings.parquet"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$WEAK_DIR" --name viz_weak_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$WEAK_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/embeddings/viz_weak/input.parquet \
    output_parquet=/results/iter_$N/embeddings/viz_weak/embeddings.parquet

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$MINED_EMBED_DIR" --name viz_mined_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$MINED_EMBED_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/mining/mined_unique_images.parquet \
    output_parquet=/results/iter_$N/embeddings/augmented/mined_embeddings.parquet
```

If `iter_N/embeddings/previous/prev_pool.parquet` exists and is non-empty,
also run:

```bash
PREV_DIR="$ITER_DIR/embeddings/previous"
PREV_OUT="$PREV_DIR/embeddings.parquet"
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$PREV_DIR" --name viz_previous_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$PREV_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/embeddings/previous/prev_pool.parquet \
    output_parquet=/results/iter_$N/embeddings/previous/embeddings.parquet
```

Do not synthesize an empty previous-data input; absence at iteration 1 without
a seed training set is the normal branch.

## Finish and commit

When `VISUALIZE_EMBEDDINGS=true`, run t-SNE only after all required embedding
actions finalize successfully. Do not run this adapter for contact-sheets-only mode:

```bash
if [ "$VISUALIZE_EMBEDDINGS" = true ]; then
  "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
    "$SKILL_ROOT/scripts/run_pas_stage.py" visualize-finish \
      --results-dir "$RESULTS_DIR" \
      --deft-config "$RESULTS_DIR/config/deft_config.yaml" --iter-num "$N"
fi
```

Always run this host step through `deft_python.sh`; its thread caps prevent
many-core OpenBLAS failures during scikit-learn t-SNE.

Build commit arguments only for enabled/configured outputs. Repeat
`--visualize-command-status` for weak, mined, and—when run—previous embedding
statuses:

```bash
VIS_ARGS=()
VIS_ARGS+=(--visualize-prepare-status \
  "$ITER_DIR/visualization/visualize-prepare.host.status.json")
if [ "$VISUALIZE" = true ]; then
  VIS_ARGS+=(--samples-dir "$ITER_DIR/visualization/samples")
fi
if [ "$VISUALIZE_EMBEDDINGS" = true ]; then
  VIS_ARGS+=(--tsne-plot "$ITER_DIR/visualization/tsne_plot.png")
  VIS_ARGS+=(--visualize-finish-status \
    "$ITER_DIR/visualization/visualize-finish.host.status.json")
  VIS_ARGS+=(--visualize-command-status "$WEAK_DIR/viz_weak_embed.status.json")
  VIS_ARGS+=(--visualize-command-status "$MINED_EMBED_DIR/viz_mined_embed.status.json")
  if [ -n "${PREV_OUT:-}" ]; then
    VIS_ARGS+=(--visualize-command-status "$PREV_DIR/viz_previous_embed.status.json")
  fi
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" --stage visualize \
    "${VIS_ARGS[@]}" --summary "iter$N approved visualizations completed"
```

When both flags are false, use instead:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" \
    --stage visualize --skip \
    --summary "visualization disabled in approved config"
```

Empty contact sheets, missing t-SNE, nonzero embedding status, or stale image
embeddings are stage failures. Apply the normal one-correction limit; do not
change the approved visualization flags to make the audit pass.
