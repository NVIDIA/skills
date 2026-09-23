# DEFT OD — Embed Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-generate-image-embeddings`. Read that skill's `SKILL.md` for the full field reference.

## When to invoke

Iteration stage 2, immediately after `gap_analysis`. Skipped only when `gap_analysis` produced zero weak images, in which case the loop stops rather than continuing.

## Input comes straight from gap analysis

`weak_images.parquet` already carries a `filepath` column, so it is fed to the embedder directly. No projection step is needed.


## Encoder consistency is the whole ballgame

The embedding written here is compared against the **source-pool** embedding parquet during mining. Both must come from the same encoder — same `model` and same `model_path`. Mismatched encoders produce vectors that are not comparable, and the failure is silent: mining succeeds and returns confidently wrong neighbours.

Resolve `model` / `model_path` once in Pre-Flight (check 9) and reuse those exact values on every iteration from `state.config.embedding_model` / `state.config.embedding_model_path`. Never let one iteration pick a different encoder, and never let this stage fall back to a bare HuggingFace id at run time — Pre-Flight already decided between a local snapshot and a verified online id, and re-deciding mid-loop is how the two sides drift apart.

## Spec

Write per-iteration under `${RESULTS_DIR}/iter${N}/embeddings/image_embeddings.yaml` — the invocation below reads it from
`$EMBED_SPEC`, so bind the two:

```bash
EMBED_SPEC="${RESULTS_DIR}/iter${N}/embeddings/image_embeddings.yaml"
```


```yaml
input_parquet:  <absolute path to iter${N}/gaps/weak_images.parquet>
output_parquet: <absolute path to iter${N}/embeddings/weak_images_embeddings.parquet>
model:          <SigLIP|CLIP — from state.config.embedding_model>
model_path:     <from state.config.embedding_model_path>
model_config_path: ""      # only when model_path is a TAO .pth/.ckpt
                           # Setting this via --set needs the empty string quoted
                           # twice: --set model_config_path='""'. A bare
                           # --set model_config_path="" parses as YAML null and TAO
                           # rejects it with `Incompatible value 'None' for field of
                           # type 'str'`.
batch_size:     64
```

## Invocation

```bash
<skill_root>/scripts/deft_python.sh <skill_bank>/skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py \
  --spec "$EMBED_SPEC"

docker run --rm --name "deft_iter${N}_embed" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  embedding image_embeddings -e "$EMBED_SPEC"
```

## Output

| Artifact | Path |
|---|---|
| `weak_images_embeddings.parquet` | `${RESULTS_DIR}/iter${N}/embeddings/weak_images_embeddings.parquet` |

Columns: `filepath`, `embedding` (list-like), plus every extra column carried through from `weak_images.parquet`.

The embedding column is named **`embedding`**, which is what `tmm unique_neighbor_matching` expects by default — no column override is needed anywhere in this loop.


## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage embed \
  --embeddings-parquet "${RESULTS_DIR}/iter${N}/embeddings/weak_images_embeddings.parquet" \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "embedded <N> weak images with <model>"
```
