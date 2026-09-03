# Curation and Retrieval Workflow — Curator → Data Mining

Concise cross-product map for Cosmos Curator and Data Mining.
There is no shared orchestrator, scheduler, or cross-step workflow state. Make
is the local operator façade; external automation uses the equivalent
CLI/container contracts.

## Ordered handoffs

### 1. Curate and produce embeddings

Run Curator with the existing curation guidance and leave IV2 or CE1 embedding
parquet on disk:

```bash
make check-curator-runtime MODELS_DIR=$MODELS_DIR
make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml \
  MODELS_DIR=$MODELS_DIR \
  DATA_DIR=/path/to/data \
  DOCKER_NETWORK=bridge \
  MODELS_MOUNT_MODE=rw \
  CURATOR_TMP=/path/to/large/scratch
```

Curator IV2 and CE1 can feed TMM.

### 2. Generate TAO DS image embeddings when needed

For still images, TAO DS can generate CLIP/SigLIP vectors:

```bash
make image-embeddings-build \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_JSON=$PWD/data/images/rows.json \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet
make run-image-embeddings \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/images/output.parquet \
  IMAGE_EMBEDDING_MODEL_TYPE=clip \
  IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
make image-embeddings-validate-output \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/images/output.parquet
```

For captions, `make run-text-embeddings` produces CLIP/SigLIP/SigLIP2 vectors
from a parquet with a `text` column; see `references/data-mining.md`.

Validate input rows, producer family, finite vectors, dimensions, and output
metadata before mining. Dry-run evidence is not live Docker/GPU qualification.

### 3. Mine compatible embeddings

Data Mining consumes Curator IV2/CE1 or CLIP/SigLIP embeddings; it does not
generate them. Prepare non-empty target (`S`) and source (`B`) parquet
selections under `DATA_DIR`, then run:

```bash
make run-data-mining-select \
  DATA_DIR=$PWD/data/nearest-neighbor-mining \
  TARGET_SUBDIR=selected/target.parquet \
  SOURCE_SUBDIR=staged/source.parquet \
  TDM_EMBEDDING_BACKEND=ce1
```

Require non-empty S/B selections, finite matching-dimension vectors, an explicit
backend declaration, Docker/GPU prerequisites, and handoff evidence.

For greedy unique-assignment mining (including `class_stratified` with
detection files), use `make run-data-mining-unique-match` and
`cookbook/unique-neighbor-matching/`. See [Data Mining](data-mining.md) for the full Make
and CLI knob list (`DISTANCE_THRESHOLD`, embed column names, UNM stratified
flags, and post-run `final_unique_files.parquet` validation).

## Completion boundary

Each step is finite. Persist its validated artifact path, embedding family, and
JSON result before starting the next step. Mine only after the preceding
validation succeeds. Live model and GPU acceptance remains deployment
qualification.

Use Make locally to discover exact mappings (`make help`). Domain recipes live
under `cookbook/traffic-video-analytics/`, `cookbook/warehouse-safety/`,
`cookbook/nearest-neighbor-mining/`, and `cookbook/unique-neighbor-matching/`.
