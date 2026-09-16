# Data Mining

Handoff reference for Curator embedding outputs that can feed Data Mining.
This page does not own mining command execution.

- Curator video embeddings (IV2 or CE1 parquet) can become S/B tables.
- Still-image CLIP/SigLIP vectors are a separate embedding producer, not a
  Curator stage.
- Data Mining consumes precomputed embeddings; it does not generate them.

Operator execution stays in `make help` and `cookbook/nearest-neighbor-mining/`,
`cookbook/unique-neighbor-matching/`. Narrative:
[`docs/user-guide/`](../../../docs/user-guide/).
Ordered stitch:
[curation-retrieval-workflow.md](curation-retrieval-workflow.md).

`default_specs` is not exposed: both `tmm default_specs` and
`embedding default_specs` exit non-zero with
`Module 'tmm' is not supported. Supported modules: analytics, annotations,
augmentation, auto_label, image`. Every run receives a fully resolved,
validated experiment spec that is returned as run evidence.

## Prerequisites and qualification boundary

- Pull the TAO Toolkit Data Services image with `make pull-data-mining`.
- The Make default is
  `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services`; Make passes the
  explicitly configured local image to the internal CLI.
- Provide Docker, NVIDIA Container Toolkit, a compatible GPU, and sufficient
  shared memory (`DATA_MINING_SHM_SIZE`, default `16g`).
- Keep input, local model artifacts, and output paths under `DATA_DIR`; it is
  mounted at `/data`. Engine-native config files are mounted separately
  read-only.
- Use `GPUS` to select devices.

Manifest validation, Docker argument construction, dry-run behavior, and output
validation are covered by local tests. NVIDIA documents driver `595.45.04` as the
TAO Data Services minimum; Curator compatibility remains a separate qualification
boundary.

Engineering smoke runs on this host exercised `nearest_neighbors` (custom
embedding columns, `filter_by_label`, `distance_threshold`),
`unique_neighbor_matching` (`global` and `class_stratified` with both COCO and
KITTI detections), and `text_embeddings`. Formal release qualification still
requires SQA evidence against the approved image digest.

## Generate image embeddings

TAO DS image embeddings support:

- Hugging Face CLIP models;
- Hugging Face SigLIP models; and
- TAO CLIP-compatible `.ckpt`/`.pth` checkpoints with a model config.

For caption/text vectors see **Generate text embeddings** below.

### Build and validate the input

Create a JSON array with one unique, non-empty `filepath` per row. Paths must
resolve to non-empty files under `DATA_DIR`. Additional columns are preserved
as metadata.

```bash
make image-embeddings-build \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_JSON=$PWD/data/images/rows.json \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet

make image-embeddings-validate-input \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet
```

The input must not already contain the reserved `embedding` column.

### Run CLIP or SigLIP

```bash
make run-image-embeddings \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/images/output.parquet \
  IMAGE_EMBEDDING_MODEL_TYPE=clip \
  IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
```

Use `IMAGE_EMBEDDING_MODEL_TYPE=siglip` with a compatible model. For a TAO
checkpoint, use `clip`, point `IMAGE_EMBEDDING_MODEL` at the checkpoint, and
set `IMAGE_EMBEDDING_MODEL_CONFIG`.

An engine-native YAML containing `input_parquet`, `output_parquet`, `model`,
and `model_path` may be supplied through `IMAGE_EMBEDDING_CONFIG`. Do not mix
that config with direct model/input/output variables.

Set `IMAGE_EMBEDDING_DRY_RUN=1` to validate and emit the Docker argument list
without starting the container. TAO Data Services always receives `-e` with a
read-only experiment spec. Direct variables generate a temporary validated
spec. Dry-run evidence contains its YAML, but the ephemeral host path is
descriptive rather than replayable.

### Validate output

```bash
make image-embeddings-validate-output \
  DATA_DIR=$PWD/data/images \
  IMAGE_EMBEDDING_INPUT=$PWD/data/images/input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/images/output.parquet
```

Validation requires:

- unique `filepath` rows matching the input;
- a non-empty, finite `embedding` vector per row;
- one vector dimension across the output; and
- exact preservation of input metadata columns and values.

## Generate text embeddings

`embedding text_embeddings` consumes a parquet with a `text` column and supports
CLIP, SigLIP, and SigLIP2. Extra columns are preserved as metadata.

```bash
make text-embeddings-validate-input \
  DATA_DIR=$PWD/data/captions \
  TEXT_EMBEDDING_INPUT=$PWD/data/captions/captions.parquet

make run-text-embeddings \
  DATA_DIR=$PWD/data/captions \
  TEXT_EMBEDDING_INPUT=$PWD/data/captions/captions.parquet \
  TEXT_EMBEDDING_OUTPUT=$PWD/data/captions/text_embeddings.parquet \
  TEXT_EMBEDDING_MODEL=clip \
  TEXT_EMBEDDING_MODEL_PATH=openai/clip-vit-base-patch32

make text-embeddings-validate-output \
  DATA_DIR=$PWD/data/captions \
  TEXT_EMBEDDING_INPUT=$PWD/data/captions/captions.parquet \
  TEXT_EMBEDDING_OUTPUT=$PWD/data/captions/text_embeddings.parquet
```

`TEXT_EMBEDDING_MODEL` accepts `clip`, `siglip`, or `siglip2`.
`TEXT_EMBEDDING_MODEL_PATH` accepts a Hugging Face model id or a local model
under `DATA_DIR`. Supply a vendor YAML through `TEXT_EMBEDDING_CONFIG` instead
of the direct variables, and use `TEXT_EMBEDDING_DRY_RUN=1` to emit the spec and
Docker arguments without running.

Output validation requires a finite `embedding` vector per row, one dimension
across the output, the same row count and text order as the input, and every
input metadata column preserved. Text values may repeat, so rows are compared
positionally, matching how the engine rejoins metadata.

## Prepare TMM inputs

TMM accepts precomputed embeddings from:

- Curator IV2 video output;
- Curator CE1 video output; or
- TAO DS CLIP/SigLIP image output.

Prepare target selection `S` and source selection `B` as parquet files or
directories under `DATA_DIR`. Before Docker/GPU execution, the CLI requires:

- non-empty selections;
- finite, non-empty vectors;
- one dimension within each selection; and
- matching dimensions between S and B.

Declare the producer contract explicitly:

```text
TDM_EMBEDDING_BACKEND=iv2|ce1|clip|siglip
```

The declaration routes compatibility checks; vector values alone cannot prove
their producer family.

Run TMM with prepared selections under `DATA_DIR`:

```bash
make run-data-mining-select \
  DATA_DIR=$PWD/data/nearest-neighbor-mining \
  TARGET_SUBDIR=_tmm_prep/target.parquet \
  SOURCE_SUBDIR=_tmm_prep/source.parquet \
  TDM_EMBEDDING_BACKEND=ce1
```

For existing `S.parquet`/`B.parquet`, use `TARGET_SUBDIR`, `SOURCE_SUBDIR`, or
`TMM_CONFIG_FILE=cookbook/nearest-neighbor-mining/tmm.yaml`, plus the matching
`TDM_EMBEDDING_BACKEND`. IV2 remains TMM-only.

TAO DS requires `-e/--experiment_spec_file` for every
`tmm nearest_neighbors` and `tmm unique_neighbor_matching` run. Direct Make
inputs generate a temporary read-only spec; `TMM_CONFIG_FILE` /
`TMM_UNM_CONFIG_FILE` mount the supplied spec read-only. Supported metrics are
`cosine`, `euclidean`, and `manhattan`; `l2` is not a supported mining metric.

### nearest_neighbors (`make run-data-mining-select`)

Direct Make/CLI knobs when `TMM_CONFIG_FILE` is omitted:

- `TOPN`, `DATA_MINING_METRIC`, `DISTANCE_THRESHOLD` (float; `-1.0` disables)
- `FILTER_BY_LABEL=0|1` (drops source/target pairs whose `label` values disagree;
  the engine warns and skips filtering when either side has no `label` column)
- `SOURCE_EMBED_COLUMN_NAME` / `TARGET_EMBED_COLUMN_NAME` (default `embedding`)
- `TARGET_SUBDIR`, `SOURCE_SUBDIR`, `OUTPUT_SUBDIR`, `TDM_EMBEDDING_BACKEND`
- In YAML, `filter_by_label` must be the string `"true"` or `"false"`

```bash
make run-data-mining-select \
  DATA_DIR=$PWD/data/nearest-neighbor-mining \
  TARGET_SUBDIR=S.parquet \
  SOURCE_SUBDIR=B.parquet \
  TOPN=3 \
  DISTANCE_THRESHOLD=1.5 \
  FILTER_BY_LABEL=1 \
  SOURCE_EMBED_COLUMN_NAME=source_vec \
  TARGET_EMBED_COLUMN_NAME=target_vec \
  TDM_EMBEDDING_BACKEND=iv2
```

Custom embedding and filepath column names are honored by host-side preparation
and validation, so parquets that never use the literal `embedding` or `filepath`
names work on both the direct and YAML paths.

### unique_neighbor_matching (`make run-data-mining-unique-match`)

Use for greedy unique-assignment mining. Direct Make/CLI knobs when
`TMM_UNM_CONFIG_FILE` is omitted:

- `DESIRED_UNIQUE_COUNT`, `ALLOCATION_POLICY=global|class_stratified`
- `CANDIDATE_EXPANSION_FACTOR`, `DATA_MINING_METRIC`, `UNM_OUTPUT_SUBDIR`
- Column overrides: `SOURCE_EMBEDDING_COLUMN`, `TARGET_EMBEDDING_COLUMN`,
  `SOURCE_FILEPATH_COLUMN`, `TARGET_FILEPATH_COLUMN`
- Optional: `EXCLUDE_PATH`, `SAVE_EMBEDDINGS=0|1`, `VISUALIZE=0|1`
- For `class_stratified` (all required): `SOURCE_DETECTION_FILE`,
  `TARGET_DETECTION_FILE`, `DETECTION_FORMAT=coco|kitti`, `RARE_CLASS_LIST`

Detection inputs follow the engine's annotation shapes: `DETECTION_FORMAT=coco`
requires a single COCO JSON file, and `DETECTION_FORMAT=kitti` requires a
directory of per-image `.txt` label files. A directory is rejected for `coco`, a
file is rejected for `kitti`, and an empty KITTI directory is rejected before the
container starts.

```bash
make run-data-mining-unique-match \
  DATA_DIR=$PWD/data/unique-neighbor-matching \
  DESIRED_UNIQUE_COUNT=100 \
  ALLOCATION_POLICY=global \
  UNM_OUTPUT_SUBDIR=unm_out \
  DATA_MINING_METRIC=euclidean
```

Class-stratified with COCO JSON files:

```bash
make run-data-mining-unique-match \
  DATA_DIR=$PWD/data/unique-neighbor-matching \
  ALLOCATION_POLICY=class_stratified \
  SOURCE_DETECTION_FILE=detections/source.json \
  TARGET_DETECTION_FILE=detections/target.json \
  DETECTION_FORMAT=coco \
  RARE_CLASS_LIST=person,bicycle
```

Class-stratified with KITTI label directories:

```bash
make run-data-mining-unique-match \
  DATA_DIR=$PWD/data/unique-neighbor-matching \
  ALLOCATION_POLICY=class_stratified \
  SOURCE_DETECTION_FILE=labels_source \
  TARGET_DETECTION_FILE=labels_target \
  DETECTION_FORMAT=kitti \
  RARE_CLASS_LIST=person,bicycle
```

Successful non-dry-run UNM validates `final_unique_files.parquet` under the
output directory. Optional `summary.json` and visualization artifacts may also
appear. See `cookbook/nearest-neighbor-mining/` and `cookbook/unique-neighbor-matching/`.

## Machine-readable handoff

Prefer the installed `paidf_curation_and_retrieval` CLI for automation:

- image and text embedding build/validate/run commands emit JSON status, paths,
  row counts, dimensions, and Docker evidence;
- embedding validation failures emit JSON and exit nonzero;
- `data-mining-select` emits JSON containing status, backend, dry-run state,
  command, and `evidence` with the generated experiment spec;
- `data-mining-unique-match` emits the same JSON shape, and its `evidence` also
  carries the validated `final_unique_files.parquet` and `summary.json` paths;
- `status` is `dry-run` for dry runs and `completed` for real runs;
- Click usage/parsing failures may use human-readable stderr.

Treat every nonzero exit as failure. Persist the validated input/output paths,
embedding backend, vector dimension, and command evidence as the handoff
manifest for downstream orchestration.
