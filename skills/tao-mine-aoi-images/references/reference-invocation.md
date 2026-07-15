# Reference Invocation

This is the minimal end-to-end recipe — paste-and-edit the workspace, the three parquet paths, and the encoder, and it runs. Run as a single Bash block so the script-check hook sees one streamed log.

```bash
WORKSPACE=<absolute path>           # mounted identically inside the container
TARGETS=<target_parquet>            # e.g. .../routing_results/<ts>/mining_gaps.parquet
SOURCE_POOL=<source_pool_parquet>   # parquet with `filepath` (and optional `label`)
OUT="$WORKSPACE/mining_results/$(date +%Y-%m-%d_%H%M%S)"
EMBED_SPEC="$OUT/embedding_spec.yaml"
MINE_SPEC="$OUT/mining_spec.yaml"
MODEL=SigLIP                        # or CLIP, or a TAO checkpoint name
MODEL_PATH=google/siglip-base-patch16-224  # or a local checkpoint path
TOPN=5
METRIC=cosine
FILTER_BY_LABEL=false
IMG=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")

mkdir -p "$OUT"

# Write the two spec files for this iteration
cat > "$EMBED_SPEC" <<EOF
model: $MODEL
model_path: $MODEL_PATH
batch_size: 64
EOF

cat > "$MINE_SPEC" <<EOF
topn: $TOPN
knn_metric: $METRIC
filter_by_label: "$FILTER_BY_LABEL"
EOF

# Step 1: embed targets
docker run --gpus all --rm --ipc=host \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" embedding image_embeddings \
    -e "$EMBED_SPEC" \
    input_parquet="$TARGETS" \
    output_parquet="$OUT/target_embeddings.parquet"

# Step 2: embed source pool (SAME embedding spec as Step 1)
docker run --gpus all --rm --ipc=host \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" embedding image_embeddings \
    -e "$EMBED_SPEC" \
    input_parquet="$SOURCE_POOL" \
    output_parquet="$OUT/source_embeddings.parquet"

# Step 3: mine nearest neighbours
docker run --gpus all --rm --ipc=host \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" tmm nearest_neighbors \
    -e "$MINE_SPEC" \
    source_parquet="$OUT/source_embeddings.parquet" \
    target_parquet="$OUT/target_embeddings.parquet" \
    output_parquet="$OUT/mined.parquet"

# Chown outputs back to the host UID (container runs as root)
docker run --rm -v "$WORKSPACE:$WORKSPACE" alpine \
    chown -R "$(id -u):$(id -g)" "$OUT"

# Sanity print so the script-check hook sees row counts
python3 -c "
import pandas as pd
for name, p in [('target_embeddings', '$OUT/target_embeddings.parquet'),
                ('source_embeddings', '$OUT/source_embeddings.parquet'),
                ('mined',             '$OUT/mined.parquet')]:
    df = pd.read_parquet(p)
    print(f'{name}: rows={len(df)}, cols={list(df.columns)}')
"
```

Print the row counts and column lists at the end so the script-check hook can verify each step actually produced output.
