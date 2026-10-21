# Curation and Retrieval Workflow — Curator → CDS → Data Mining

Concise cross-product map for Curator, Cosmos Dataset Search (CDS), and Data
Mining. In Curation and Retrieval 1.2, adapter and integration code provide
invocation coverage for the CDS GA core, but live deployment acceptance remains separate. There is no shared
orchestrator, scheduler, or cross-step workflow state. Make is the local
operator façade; external automation uses the equivalent CLI/container/API
contracts.

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

### 2. Send CE1 to CDS

Operator Path D on local compose (wait for health, collection **id**, lab
document ingest): `docs/user-guide/operations-cds.md`.

For GA bulk ingest (validate, stage to `s3://`, create the collection, insert,
wait, then search), run the command sequence in
[Public GA bulk-ingest workflow](cosmos-dataset-search.md#public-ga-bulk-ingest-workflow).
`COLLECTION` must be the create-collection **id**.

`POST /insert-data` 404 on a local file is expected on compose. Follow
operations-cds.md lab ingest. Do not treat `documents:N` as GA bulk-ingest
success.

### 3. Generate image embeddings when needed

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

### 4. Mine compatible embeddings

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
`cookbook/physical-ai-av/`, `cookbook/nearest-neighbor-mining/`, and
`cookbook/unique-neighbor-matching/`.
