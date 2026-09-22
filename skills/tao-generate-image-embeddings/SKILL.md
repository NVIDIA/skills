---
name: tao-generate-image-embeddings
description: >-
  Run TAO Data Services image embedding to turn a parquet of image filepaths into an embedding parquet
  using CLIP, SigLIP, or a TAO checkpoint. Use when a workflow needs embeddings before nearest-neighbor
  or unique-neighbor mining, or when the user asks to "embed images", "compute image embeddings",
  or "generate SigLIP embeddings".
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, one or more CUDA GPUs, and the TAO data-services container pinned in versions.yaml.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- data
- embeddings
- mining
- siglip
- clip
---

# TAO Generate Image Embeddings

Use this skill to run TAO Data Services image embedding. The skill consumes a parquet of image filepaths and writes a parquet with an embedding column. Downstream mining skills (`tao-mine-od-images`, `tao-mine-nearest-neighbors`) consume its output.

The container entrypoint is:

```bash
embedding image_embeddings -e /absolute/path/to/image_embeddings.yaml
```

## Inputs

The user can provide either an existing spec or the fields needed to generate one.

Required spec fields:

| Field | Meaning |
|---|---|
| `input_parquet` | Absolute path to a parquet containing image filepaths. |
| `output_parquet` | Absolute path where the embedding parquet is written. |
| `model` | `CLIP` or `SigLIP`. |
| `model_path` | HuggingFace model id, local HF snapshot directory, or a TAO `.pth`/`.ckpt` checkpoint. **Must match `model`.** SigLIP: `google/siglip-base-patch16-224` (768-dim, the template default). CLIP: `openai/clip-vit-base-patch32` (512-dim). The validator rejects a recognizable model/path mismatch before launch. |

Common optional fields:

| Field | Default | Meaning |
|---|---:|---|
| `model_config_path` | `""` | TAO experiment spec path. Required only when `model_path` is a TAO checkpoint. |
| `batch_size` | `64` | Number of images processed in parallel. Lower it if the GPU runs out of memory. |

The input parquet must contain a `filepath` column. Any additional columns are carried through to the output verbatim, so metadata such as `label` survives into the embedding parquet.

The default template is `assets/default_image_embeddings.yaml`.

## Encoder Consistency

When embeddings feed a mining step, every parquet compared against another must be produced with the **same** `model` and `model_path`. Embedding dimensionality follows the encoder — 768 for the SigLIP default, 512 for CLIP ViT-B/32 — and nothing in the output parquet records which encoder wrote it. Embeddings from different encoders are not comparable, and mismatched encoders are the most common cause of mining output that looks unrelated to the targets. Reuse one spec across every parquet in a mining run and override only `input_parquet` / `output_parquet`.

## Quick Start

Run from the `tao-skill-bank` repo root.

**Write the spec beside the output parquet.** The run does not retain it, so the
embeddings otherwise carry no record of the encoder that produced them. That
matters here more than elsewhere: every parquet compared against another in a
mining step must come from the same `model` and `model_path`, and mismatched
encoders are the usual cause of mining output that looks unrelated to its targets.

```bash
OUT_DIR=/absolute/path/for/this/run              # where output_parquet is written
SPEC="$OUT_DIR/image_embeddings.yaml"            # spec lives beside its output
RUN_ROOT=/absolute/path/that/contains/parquets/images/and/results
GPU_COUNT=1

python3 skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py \
  --spec "$SPEC"

DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services

docker run --rm --gpus "$GPU_COUNT" --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w "$RUN_ROOT" \
  "$DS_IMAGE" \
  embedding image_embeddings -e "$SPEC"
```

Do not pass `--user $(id -u):$(id -g)` to the TAO data-services container; the image imports `transformers` at startup, which calls `getpass.getuser()` and fails when the UID is not present in `/etc/passwd`.

To embed several parquets with one encoder, reuse the same spec and override the two paths per run:

```bash
docker run --rm --gpus "$GPU_COUNT" --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" -w "$RUN_ROOT" "$DS_IMAGE" \
  embedding image_embeddings -e "$SPEC" \
  input_parquet=/abs/path/other_input.parquet \
  output_parquet=/abs/path/other_output.parquet
```

## Generate A Spec

If the user provides parquet paths and an encoder instead of a ready spec, copy
the template and fill in the `null`s. Every tuning value it already carries is
the one this stage wants — change one only deliberately.

```bash
cp skills/data/tao-generate-image-embeddings/assets/default_image_embeddings.yaml "$SPEC"
```

Fill `input_parquet`, `output_parquet` and — if not using the default encoder —
`model` and `model_path`, all as absolute paths, then validate:

```bash
python3 skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py --spec "$SPEC"
```

```yaml
input_parquet: /absolute/path/filepaths.parquet
output_parquet: /absolute/path/results/embeddings.parquet
model: SigLIP
model_path: google/siglip-base-patch16-224
model_config_path: ""          # required only when model_path is a TAO .pth/.ckpt
batch_size: 64
```

The template is the only place a default value lives, so nothing can disagree
with it. `verify` reports the encoder, since embeddings are only comparable to
others produced by the same `model` and `model_path`.

Keep the spec, input parquet, image files, and output directory under `RUN_ROOT`
so the same paths resolve inside the container.

## Preflight

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
python3 skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py \
  --spec "$SPEC"
```

4. Confirm `RUN_ROOT` contains the spec, the input parquet, the image files its `filepath` column points at, and the output directory. Mount `RUN_ROOT` to the same absolute path inside Docker.

## Outputs

| Artifact | Location |
|---|---|
| embedding parquet | `output_parquet` |

The output parquet contains `filepath`, an `embedding` column of list-like vectors, and every extra column carried through from the input. Print its row count and column list after the run so the caller can confirm the embedding column exists.

## Troubleshooting

**`The subtask image_embeddings requires -e/--experiment_spec_file`**: rerun with `embedding image_embeddings -e "$SPEC"`.

**Input parquet or images not found inside Docker**: the `filepath` values are read verbatim. Use a `RUN_ROOT` mount where host and container paths are identical, and confirm the images themselves are under that mount — not just the parquet.

**Model loading error with a `.pth` / `.ckpt` `model_path`**: TAO checkpoints need `model_config_path` set to the training spec so the architecture can be rebuilt. HuggingFace ids and snapshot directories do not.

**CUDA out of memory**: lower `batch_size` (try 32 or 16).

**Mined results look unrelated downstream**: the parquets compared during mining were embedded with different encoders. Re-embed them with one shared spec — see `## Encoder Consistency`.

**No GPU available**: embedding requires at least one CUDA GPU. Check `nvidia-smi -L`, the Docker `--gpus` flag, and the NVIDIA container toolkit installation.
