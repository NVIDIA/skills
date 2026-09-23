---
name: tao-mine-od-images
description: >-
  Run TAO Data Services TMM unique-neighbor matching mining from embedding parquet files for object detection workflows.
  Use when an object detection workflow needs to mine a bijectively-assigned set of unique source images closest to target samples.
  Use global allocation when mining without class constraints. Use class_stratified when rare classes are specified.
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
- object-detection
- unique-neighbors
- tmm
---

# TAO Mine OD Images (Unique Neighbor Matching)

Use this skill to run TAO Data Services TMM unique-neighbor matching mining for object detection. The skill consumes pre-embedded source and target parquets and writes a directory of outputs including `final_unique_files.parquet` and `summary.json`. It does not compute embeddings; upstream steps must produce the source and target embedding parquets first.

The container entrypoint is:

```bash
tmm unique_neighbor_matching -e /absolute/path/to/unique_neighbor_matching.yaml
```

## Inputs

The user can provide either an existing spec or the fields needed to generate one.

Required spec fields:

| Field | Meaning |
|---|---|
| `source_path` | Absolute path to the source embeddings parquet or directory of parquets. |
| `target_path` | Absolute path to the target embeddings parquet or directory of parquets. |
| `output_dir` | Absolute path to the output directory. Writes `final_unique_files.parquet`, `summary.json`, and per-iteration parquets. |
| `desired_unique_count` | Total number of unique source files to retrieve. |

Common optional fields:

| Field | Default | Meaning |
|---|---:|---|
| `allocation_policy` | `global` | `global` or `class_stratified`. |
| `distance_metric` | `euclidean` | One of `euclidean`, `cosine`, or `manhattan`. Embeddings are L2-normalized before search. |
| `candidate_expansion_factor` | `5` | Candidate-pool multiplier per iteration. Increase if desired count is not reached. |
| `source_embedding_column` | `embedding` | Embedding column in `source_path`. |
| `target_embedding_column` | `embedding` | Embedding column in `target_path`. |
| `source_filepath_column` | `filepath` | Filepath column in `source_path`; also the column of `final_unique_files.parquet`. |
| `target_filepath_column` | `filepath` | Filepath column in `target_path`. |
| `exclude_path` | `null` | Parquet with a `filepath` column; those images are removed from the source pool. |
| `source_detection_file` | `null` | COCO `.json` or KITTI label directory for the source. Required for `class_stratified`. |
| `target_detection_file` | `null` | COCO `.json` or KITTI label directory for the target. Required for `class_stratified`. |
| `detection_format` | `null` | `coco` or `kitti`. Required whenever a detection file is set; never inferred from the path. |
| `rare_class_list` | `""` | Comma-separated rare class names, e.g. `"person,bicycle"`. Required for `class_stratified`. |
| `save_embeddings` | `false` | Include embeddings in per-iteration parquet outputs. |
| `visualize` | `false` | Save per-class visualization grids (requires Pillow and matplotlib). |

Both input parquets must contain the filepath and embedding columns. Source and target embeddings must have been produced by the same encoder; mismatched encoders produce garbage output.

The default template is `assets/default_unique_neighbor_matching.yaml`.

## Quick Start

Run from the `tao-skill-bank` repo root. Resolve the pinned TAO Data Services image from `versions.yaml`, verify the spec, mount the run root with identical host/container paths, and stream the Docker logs.

**Write the spec into the output directory.** The run does not retain it, so a mined
set otherwise carries no record of the budget, allocation policy or rare-class list
that produced it — and those decide which images were selected. Keeping them together
makes the selection recoverable from the run alone.

```bash
OUTPUT_DIR=/absolute/path/for/this/run           # output_dir in the spec
SPEC="$OUTPUT_DIR/unique_neighbor_matching.yaml" # spec lives beside its outputs
RUN_ROOT=/absolute/path/that/contains/specs/data/and/results
GPU_COUNT=1

python3 skills/data/tao-mine-od-images/scripts/verify_unique_neighbor_matching_spec.py \
  --spec "$SPEC"

DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services

docker run --rm --gpus "$GPU_COUNT" --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -w "$RUN_ROOT" \
  "$DS_IMAGE" \
  tmm unique_neighbor_matching -e "$SPEC"
```

Do not pass `--user $(id -u):$(id -g)` to the TAO data-services container; some TAO DS images call `getpass.getuser()` at startup and fail when the UID is not in `/etc/passwd`.

## Generate A Spec

If the user provides source/target paths and an output directory instead of a
ready spec, copy the template and fill in the `null`s. Every tuning value it
already carries is the one this stage wants — change one only deliberately.

```bash
cp skills/data/tao-mine-od-images/assets/default_unique_neighbor_matching.yaml "$SPEC"
```

Fill `source_path`, `target_path`, `output_dir` and `desired_unique_count`, all
as absolute paths, then validate:

```bash
python3 skills/data/tao-mine-od-images/scripts/verify_unique_neighbor_matching_spec.py --spec "$SPEC"
```

```yaml
source_path: /absolute/path/source_embeddings.parquet
target_path: /absolute/path/target_embeddings.parquet
output_dir: /absolute/path/results/mining_output
desired_unique_count: 500
allocation_policy: global          # or class_stratified — see below
distance_metric: euclidean
```

For class-stratified mode set `allocation_policy: class_stratified` and supply
`rare_class_list`, `source_detection_file`, `target_detection_file` and
`detection_format`. `verify` rejects the policy without them: absent those
fields the miner falls back to a global match, which mines the wrong images
rather than failing.

The template is the only place a default value lives, so nothing can disagree
with it. `verify` reports the budget, policy and metric, since the mined parquet
is a list of filepaths and records nothing about why those files were chosen.

Keep the spec, input parquets, and output directory under `RUN_ROOT` so the same
paths resolve inside the container.

## Preflight

Before launching Docker:

1. Verify Docker and GPU access:

```bash
docker info > /dev/null
nvidia-smi -L
```

2. Resolve and pull the data-services image if needed:

```bash
DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services
docker image inspect "$DS_IMAGE" > /dev/null || docker pull "$DS_IMAGE"
```

3. Validate the spec:

```bash
python3 skills/data/tao-mine-od-images/scripts/verify_unique_neighbor_matching_spec.py \
  --spec "$SPEC"
```

4. Confirm `RUN_ROOT` contains the spec, both input parquets (or directories), and the output directory. Mount `RUN_ROOT` to the same absolute path inside Docker.

## Outputs

| Artifact | Location |
|---|---|
| Mined source filepaths | `output_dir/final_unique_files.parquet` |
| Coverage and allocation stats | `output_dir/summary.json` |
| Per-iteration intermediates | `output_dir/<subset>_iteration_<N>_topn_<K>.parquet` |
| Per-class viz grids | `output_dir/*.png` (only if `visualize: true`) |

`final_unique_files.parquet` contains one filepath column. `summary.json` includes `retrieved_unique_count`, `coverage_pct`, and (when detection files are provided) per-class breakdowns for the target and selected source sets.

## Troubleshooting

**`The subtask unique_neighbor_matching requires -e/--experiment_spec_file`**: rerun with `tmm unique_neighbor_matching -e "$SPEC"`.

**Input path not found inside Docker**: use a `RUN_ROOT` mount where host and container paths are identical.

**`ValueError: detection_format is required`**: set `detection_format: coco` or `detection_format: kitti` whenever `source_detection_file` or `target_detection_file` is set.

**`ValueError: rare_class_list is required when allocation_policy is class_stratified`**: set `rare_class_list` and both detection files when using `class_stratified`.

**Low `coverage_pct` in `summary.json`**: the source pool is smaller than `desired_unique_count`. Expand the pool or increase `candidate_expansion_factor`.

**No GPU or cuDF/cuML errors**: mining requires at least one CUDA GPU. Check `nvidia-smi -L`, the Docker `--gpus` flag, and the NVIDIA container toolkit installation.
