---
name: tao-mine-nearest-neighbors
description: >-
  Run TAO Data Services TMM nearest-neighbor mining from embedding parquet files.
  Use when a workflow needs to mine source samples closest to target samples.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, one or more CUDA GPUs, and the TAO data-services container pinned in versions.yaml.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- data
- mining
- nearest-neighbors
- tmm
---

# TAO Mine Nearest Neighbors

Use this skill to run TAO Data Services TMM nearest-neighbor mining. The skill consumes embedding parquets and writes a mined source-sample parquet plus a mining summary. It does not compute embeddings; upstream steps must produce the source and target embedding parquets first.

The container entrypoint is:

```bash
tmm nearest_neighbors -e /absolute/path/to/nearest_neighbors.yaml
```

TAO Data Services requires `-e/--experiment_spec_file`. The `tmm` console script converts that YAML into Hydra `--config-path` and `--config-name` arguments internally.

## Inputs

The user can provide either an existing nearest-neighbors YAML spec or the fields needed to generate one.

Required spec fields:

| Field | Meaning |
|---|---|
| `source_parquet` | Absolute path to the candidate/source embeddings parquet. |
| `target_parquet` | Absolute path to the target/query embeddings parquet. |
| `output_parquet` | Absolute path where TAO Data Services should write mined source filepaths. |

Common optional fields:

| Field | Default | Meaning |
|---|---:|---|
| `topn` | `5` | Number of nearest source samples to retrieve per target sample. |
| `knn_metric` | `cosine` | One of `cosine`, `euclidean`, or `manhattan`. |
| `source_embed_column_name` | `embedding` | Embedding column in `source_parquet`. |
| `target_embed_column_name` | `embedding` | Embedding column in `target_parquet`. |
| `filter_by_label` | `"false"` | String flag. When `"true"`, TAO DS filters neighbors by matching `label` columns when both parquets provide labels. |
| `distance_threshold` | `-1.0` | Maximum distance to keep. Negative disables thresholding. |

Both input parquets must contain a `filepath` column and a list-like embedding column. If `filter_by_label` is `"true"`, both parquets should also contain `label`.

The default template is `assets/default_nearest_neighbors.yaml`.

## Quick Start

Run from the `tao-skill-bank` repo root. Resolve the pinned TAO Data Services image from `versions.yaml`, verify the spec, mount the run root with identical host/container paths, and stream the Docker logs.

```bash
SPEC=/absolute/path/to/nearest_neighbors.yaml
RUN_ROOT=/absolute/path/that/contains/specs/data/and/results
GPU_COUNT=1

python3 skills/data/tao-mine-nearest-neighbors/scripts/verify_nearest_neighbors_spec.py \
  --spec "$SPEC"

DS_IMAGE="$(scripts/resolve_versions_key.py images.tao_toolkit.data_services)"

docker run --rm --gpus "$GPU_COUNT" --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -w "$RUN_ROOT" \
  "$DS_IMAGE" \
  tmm nearest_neighbors -e "$SPEC"
```

Use at least one GPU. Choose `GPU_COUNT` from the hardware available to the host or platform that will run the container. If the user does not know the right value, inspect the host with `nvidia-smi -L` or ask which GPU allocation the run should use.

Do not pass `--user $(id -u):$(id -g)` to the TAO data-services container unless you have verified the image supports that UID. Some TAO DS images import Python packages that call `getpass.getuser()` at startup and fail when the UID is not present in `/etc/passwd`.

## Generate A Spec

If the user provides source/target/output parquet paths instead of a ready spec, generate a spec from the default template:

```bash
python3 skills/data/tao-mine-nearest-neighbors/scripts/prepare_nearest_neighbors_spec.py \
  --source-parquet /absolute/path/source_embeddings.parquet \
  --target-parquet /absolute/path/target_embeddings.parquet \
  --output-parquet /absolute/path/results/mined.parquet \
  --output-spec /absolute/path/specs/nearest_neighbors.yaml \
  --topn 5 \
  --knn-metric cosine \
  --filter-by-label false \
  --distance-threshold -1.0
```

The generated YAML uses absolute paths. Keep the spec, input parquets, and output directory under `RUN_ROOT` so the same paths resolve inside the container.

## Preflight

Before launching Docker:

1. Verify Docker and GPU access:

```bash
docker info > /dev/null
nvidia-smi -L
```

2. Resolve and pull the data-services image if needed:

```bash
DS_IMAGE="$(scripts/resolve_versions_key.py images.tao_toolkit.data_services)"
docker image inspect "$DS_IMAGE" > /dev/null || docker pull "$DS_IMAGE"
```

3. Validate the spec:

```bash
python3 skills/data/tao-mine-nearest-neighbors/scripts/verify_nearest_neighbors_spec.py \
  --spec "$SPEC"
```

4. Confirm `RUN_ROOT` contains the spec, both input parquets, and the output directory. Mount `RUN_ROOT` to the same absolute path inside Docker.

## Outputs

The skill promises the artifacts named by the spec:

| Artifact | Location |
|---|---|
| mined parquet | `output_parquet` |
| mining summary | `mining_summary.txt` next to `output_parquet` |

The current TAO Data Services `nearest_neighbors` task writes a mined parquet with unique source `filepath` rows. The summary file reports mining counts such as queries processed, neighbors considered, duplicates removed, and any label/distance filtering.

## Troubleshooting

**`The subtask nearest_neighbors requires -e/--experiment_spec_file`**: rerun with `tmm nearest_neighbors -e "$SPEC"`. Hydra overrides alone are not enough.

**Input parquet not found inside Docker**: the YAML path must be visible inside the container. Use a `RUN_ROOT` mount where the host and container paths are identical.

**Output directory is not writable after Docker exits**: the TAO DS container may have written files as root. Inform the user, report which artifacts were produced, and ask whether to repair permissions on the output directory before continuing.

**No GPU or cuDF/cuML errors**: nearest-neighbor mining requires at least one CUDA GPU. Check `nvidia-smi -L`, the Docker `--gpus` flag, and the NVIDIA container toolkit installation.
