---
name: paidf-cosmos-predict
description: >-
  Prepare and run PAIDF Cosmos Predict video generation for DEFT media samples.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit, a reachable OpenAI-compatible VLM captioning endpoint, and access to the PAIDF augmentation image.
metadata:
  tags:
  - paidf
  - cosmos-predict
  - video
  - vlm-captioning
  - deft
  - data-generation
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write

---

# PAIDF Cosmos Predict Generation

Prepare and run PAIDF Cosmos Predict generation for media samples. The skill emits a JSONL handoff that maps each input id to the original media path and generated video path.

## Purpose

Use this skill when a DEFT workflow already has media samples and needs synthetic/generated videos from PAIDF Cosmos Predict. This skill does not start the VLM captioning service. A reachable OpenAI-compatible base URL for the model used to caption input media must be provided at runtime.

## Prerequisites

- Docker with NVIDIA GPU support and `nvidia-container-toolkit`.
- Access to the PAIDF augmentation image declared by `images.metropolis_sdg.paidf_augmentation` in `versions.yaml`.
- A running VLM captioning service with an OpenAI-compatible API base URL. The base URL must be provided by the user or upstream workflow at runtime; reuse that exact base URL for every `--vlm-captioning-endpoint` argument. Do not include `/models` in `VLM_CAPTIONING_ENDPOINT`.
- `HF_TOKEN` in the run environment when Cosmos model downloads require HuggingFace access — exported, or in a user-approved env file (bare `KEY=value` lines) that the run block sources.
- `VLM_API_KEY` in the run environment the same way when the VLM captioning endpoint requires authentication.
- Input media paths that are absolute paths on the host under the required media directory. Pass the host media directory with `--media-dir`; the skill mounts it into the PAIDF container at the exact same path.

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| Input JSONL | Yes | Path to the generic media JSONL. The user or upstream workflow must provide it. Each row must include string fields `id` and `media_path`; `id` values must be unique. |
| Output directory | Yes | Host directory for prepared PAIDF config, generated videos, captions, metadata, logs, and final handoff. The user or upstream workflow must provide it. |
| VLM captioning endpoint base URL | Yes | The user or upstream workflow must provide this OpenAI-compatible base URL, for example a URL ending in `/v1`. Do not include `/models`. Pass the exact same base URL to `verify_vlm_captioning_base_url.py` before any other step and to `prepare_paidf_config.py` when writing `config.yaml`. |
| Generation settings | No | If the user provides a generation settings JSON, use it. Otherwise set `GENERATION_SETTINGS` to `skills/data/paidf-cosmos-predict/assets/default_generation_settings.json`. Always pass the resolved path to `prepare_paidf_config.py` with `--generation-settings`. |
| PAIDF GPU count | Yes | Number of GPUs for PAIDF augmentation. Pass it to `prepare_paidf_config.py` with `--paidf-num-gpus N` and to Docker with `--gpus "$PAIDF_NUM_GPUS"`. |
| Media directory | Yes | Host directory containing the input media referenced by `media_path`. Pass it as `--media-dir /path/to/media_dir` to `prepare_paidf_config.py` and mount it into Docker 1:1. |
| VLM captioning prompt | Yes | Prompt text file for VLM captioning. Pass it to `prepare_paidf_config.py` with `--caption-prompt-file`; the script inlines the prompt into `config.yaml`. |
| `HF_TOKEN` | Yes for Cosmos model downloads | The agent checks that `HF_TOKEN` is already set in the run environment, whether exported or sourced from a user-approved env file, and forwards it to Docker with `-e HF_TOKEN`. |
| `VLM_API_KEY` | Yes when the VLM captioning endpoint requires authentication | The agent warns when `VLM_API_KEY` is not set, then forwards it to Docker with `-e VLM_API_KEY` when present. If the endpoint does not require authentication, PAIDF can run without it. |

Input JSONL row shape:

```json
{"id": "stable-sample-id", "media_path": "/abs/input.mp4"}
```

The input JSONL may contain duplicate `media_path` values, but every `id` must be unique. This lets an upstream workflow attach multiple logical samples to the same source video while avoiding repeated PAIDF generation for that video.

Example:

```jsonl
{"id": "sample-a-question-1", "media_path": "/abs/video_a.mp4"}
{"id": "sample-a-question-2", "media_path": "/abs/video_a.mp4"}
{"id": "sample-b-question-1", "media_path": "/abs/video_b.mp4"}
```

`prepare_paidf_config.py` converts `media_path` values to absolute paths and deduplicates them before writing `config.yaml`, so each unique video appears only once in PAIDF `data[]` and goes through captioning/generation only once. `write_paidf_handoff.py` then expands back to the input row level so every original `id` appears in either `generated_videos.jsonl` or `failed_videos.jsonl`, with duplicate media rows pointing to the same generated or expected-generated video path.

## Runtime Image

The PAIDF augmentation image is resolved from `images.metropolis_sdg.paidf_augmentation` in `versions.yaml`; users do not need to provide it for the standard workflow.

## Agent Run Procedure

Before running, determine these runtime values from the user request or upstream workflow output:

- `INPUT_JSONL`: path to the media JSONL.
- `OUTPUT_DIR`: path to the PAIDF output directory.
- `VLM_CAPTIONING_ENDPOINT`: user-provided VLM captioning OpenAI-compatible base URL. Do not include `/models`.
- `GENERATION_SETTINGS`: user-provided generation settings JSON, or `skills/data/paidf-cosmos-predict/assets/default_generation_settings.json` when the user did not provide one.
- `PAIDF_NUM_GPUS`: PAIDF GPU count.
- `MEDIA_DIR`: media directory to mount 1:1 into Docker.
- `CAPTION_PROMPT_FILE`: VLM captioning prompt file.

If required values are missing, ask the user for them before launching PAIDF.

Set the runtime values:

```bash
INPUT_JSONL="<user-provided-input-jsonl>"
OUTPUT_DIR="<user-provided-output-dir>"
VLM_CAPTIONING_ENDPOINT="<user-provided-vlm-captioning-base-url>"
GENERATION_SETTINGS="skills/data/paidf-cosmos-predict/assets/default_generation_settings.json"
PAIDF_NUM_GPUS="<user-provided-paidf-gpu-count>"
MEDIA_DIR="<user-provided-media-dir>"
CAPTION_PROMPT_FILE="<user-provided-caption-prompt-file>"
```

If the user provides a generation settings JSON, replace `GENERATION_SETTINGS` with that path.

Check the VLM captioning base URL before preparing PAIDF inputs. This sends a preflight request to `<VLM_CAPTIONING_ENDPOINT>/models` only to verify that the base URL is reachable and OpenAI-compatible. PAIDF still receives `VLM_CAPTIONING_ENDPOINT` itself in `config.yaml`, not the derived `/models` URL.

```bash
python skills/data/paidf-cosmos-predict/scripts/verify_vlm_captioning_base_url.py \
  --vlm-captioning-endpoint "$VLM_CAPTIONING_ENDPOINT"
```

If this fails, stop the PAIDF workflow and issue an error indicating that the user-provided base URL did not pass the OpenAI-compatible `/models` preflight check.

Prepare the PAIDF output directory:

```bash
python skills/data/paidf-cosmos-predict/scripts/prepare_paidf_config.py \
  --input-jsonl "$INPUT_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --vlm-captioning-endpoint "$VLM_CAPTIONING_ENDPOINT" \
  --media-dir "$MEDIA_DIR" \
  --generation-settings "$GENERATION_SETTINGS" \
  --paidf-num-gpus "$PAIDF_NUM_GPUS" \
  --caption-prompt-file "$CAPTION_PROMPT_FILE"
```

`prepare_paidf_config.py` performs a writeability preflight, writes `config.yaml`, writes `path_map.jsonl`, writes `run_metadata.json`, and creates container-writable output directories. If an existing mounted directory or file is not writable by the agent, it fails early with a permission-fix command.

The generated config uses VLM-only captioning plus Cosmos Predict text-to-world generation. PAIDF's separate LLM prompt-augmentation step is disabled by omission: `captioning.llm` is not present. `prepare_paidf_config.py` inlines `CAPTION_PROMPT_FILE` into `captioning.vlm.user_prompt`.

Prepared helper files:

- `path_map.jsonl` maps each unique absolute input `media_path` to the deterministic generated video path, caption path, and metadata path. `write_paidf_handoff.py` uses it to produce one generated or failed handoff row per input row.
- `run_metadata.json` records the VLM captioning endpoint, captioning model, and PAIDF GPU count used when `config.yaml` was prepared. It is provenance only.

Run PAIDF:

```bash
set -a; source /path/to/.env; set +a   # omit if already exported

[ -n "${HF_TOKEN:-}" ] || {
  echo "MISSING: HF_TOKEN is not set. Export it, or point the loader above at a user-approved env file."
  exit 1
}

if [ -z "${VLM_API_KEY:-}" ]; then
  echo "WARNING: VLM_API_KEY is not set. Continue only if the VLM captioning endpoint does not require authentication."
fi

PAIDF_IMAGE="$(scripts/resolve_versions_key.py images.metropolis_sdg.paidf_augmentation)"

set -o pipefail
docker run --rm \
  --gpus "$PAIDF_NUM_GPUS" \
  --shm-size=8g \
  --network host \
  -e HF_TOKEN \
  -e VLM_API_KEY \
  -v "$MEDIA_DIR:$MEDIA_DIR:ro" \
  -v "$OUTPUT_DIR:$OUTPUT_DIR" \
  "$PAIDF_IMAGE" \
  --config "$OUTPUT_DIR/config.yaml" \
  2>&1 | tee "$OUTPUT_DIR/paidf_docker.log"
```

The Docker command resolves the PAIDF image from `versions.yaml`, mounts `MEDIA_DIR` 1:1 read-only, mounts `OUTPUT_DIR` 1:1 writable, forwards `HF_TOKEN` and `VLM_API_KEY`, and keeps a copy of Docker stdout/stderr in `$OUTPUT_DIR/paidf_docker.log` via `tee`. `set -o pipefail` preserves the Docker exit status when using `tee`.

Create the handoff after PAIDF writes the generated videos:

```bash
python skills/data/paidf-cosmos-predict/scripts/write_paidf_handoff.py \
  --input-jsonl "$INPUT_JSONL" \
  --path-map "${OUTPUT_DIR}/path_map.jsonl" \
  --generated-jsonl "${OUTPUT_DIR}/generated_videos.jsonl" \
  --failed-jsonl "${OUTPUT_DIR}/failed_videos.jsonl"
```

## Workflow

1. Verify the user-provided VLM captioning base URL with `verify_vlm_captioning_base_url.py`. If this fails, stop and report the OpenAI-compatible `/models` preflight error.
2. Run `prepare_paidf_config.py` on the input media JSONL using that same endpoint. It reads VLM captioning and PAIDF generation values from the generation settings, preflights write access, writes the PAIDF config and path map, and dedupes PAIDF generation to one entry per unique media path.
3. Run the PAIDF Docker command. The agent should monitor Docker output, Docker exit status, and generated output counts.
4. Run `write_paidf_handoff.py` to produce `generated_videos.jsonl` and `failed_videos.jsonl`.

## Long-Running Run Updates

PAIDF generation can run for a long time. During an agent-driven run, keep the user informed without requiring them to ask for status.

- Before launching PAIDF, report the output directory, unique media count from `path_map.jsonl`, PAIDF GPU count, and media directory.
- While PAIDF is running, provide a short progress update every 30 minutes. Include elapsed time, generated video count under `generated/videos/`, expected unique media count, and the latest useful PAIDF log signal if available.
- If using a tool environment where one foreground Docker command would block user-facing updates, launch PAIDF in a monitorable way, then poll Docker status/logs and filesystem output counts between updates.
- On completion, report whether the container exited successfully, generated video count, missing output count if any, and the `generated_videos.jsonl` and `failed_videos.jsonl` paths after running `write_paidf_handoff.py`.
- On failure, report the container exit code, the relevant Docker output, and whether any partial generated videos were produced.

## Reference Files

`assets/default_generation_settings.json` is the default VLM captioning and Cosmos Predict generation settings file. If the user provides a different generation settings JSON, set `GENERATION_SETTINGS` to that path; otherwise set it to the default asset path. Always pass the resolved path with `--generation-settings`. Use `assets/paidf_config_template.yaml` only as a readable shape for the generated PAIDF config.

## Deterministic Mapping

The helper converts each `media_path` to an absolute path, hashes that absolute path with SHA-256, and uses the first 16 hex characters as the sample id.

For each unique media path:

```text
generated video: <output_dir>/generated/videos/<hash16>.mp4
caption:         <output_dir>/captions/<hash16>.txt
metadata:        <output_dir>/generated/metadata/<hash16>.json
```

Duplicate input media paths share one generated path. The handoff still writes one row per input row, so multiple unique input ids may point to the same generated video.

## Outputs

| Output | Producer | Notes |
| --- | --- | --- |
| `config.yaml` | `prepare_paidf_config.py` | PAIDF config with one entry per unique absolute media path. |
| `path_map.jsonl` | `prepare_paidf_config.py` | Internal helper file mapping each unique absolute media path to its expected generated video, caption, and metadata paths. `write_paidf_handoff.py` uses this file to emit one generated or failed row per input JSONL row. Host/container paths are identical. |
| `run_metadata.json` | `prepare_paidf_config.py` | Internal provenance file recording the VLM captioning endpoint/model and PAIDF GPU count used to prepare the config. |
| `paidf_docker.log` | Docker command | Docker stdout/stderr captured with `tee` during PAIDF generation. |
| `generated/videos/*.mp4` | PAIDF | One generated video per unique absolute media path. |
| `generated_videos.jsonl` | `write_paidf_handoff.py` | JSONL for successful PAIDF outputs; one row per input row whose generated video exists. |
| `failed_videos.jsonl` | `write_paidf_handoff.py` | JSONL audit for missing generated videos; one row per input row whose expected generated video is absent. |

Each `generated_videos.jsonl` row has exactly:

```json
{"id": "stable-sample-id", "original_media_path": "/abs/input.mp4", "generated_video_path": "/abs/generated.mp4"}
```

Each `failed_videos.jsonl` row has exactly:

```json
{"id": "stable-sample-id", "original_media_path": "/abs/input.mp4", "expected_generated_video_path": "/abs/generated.mp4", "error": "missing_generated_video"}
```

## Troubleshooting

| Symptom | Likely Cause | Agent Response |
| --- | --- | --- |
| VLM captioning base URL preflight fails | The captioning service is not running, the base URL is wrong, `/v1` is missing, the OpenAI-compatible `/models` probe is unavailable, or the service is not reachable from the agent host | Report the base URL, the derived `/models` probe URL, and error detail. The agent does not have enough context to start or repair the service; ask the user whether to provide a different base URL, start the service externally, or stop the run. |
| PAIDF fails after launch | The PAIDF container exited non-zero | Report the exit status, the relevant Docker output, and the count of generated videos under `<output_dir>/generated/videos/` versus expected unique media count from `path_map.jsonl`. Then ask the user how to proceed before retrying, deleting outputs, changing settings, or continuing with partial results. |
| Cosmos checkpoint download fails | `HF_TOKEN` is unset, expired, or lacks access to the gated Cosmos repo | Report the authentication/download error. Suggest that the user accept the model license if needed and make `HF_TOKEN` available before retrying, by exporting it or adding it to a user-approved env file the run sources. Do not ask the user to paste the token into chat; never create the token value yourself and never print it. |
| VLM captioning returns authentication errors | `VLM_API_KEY` is unset, expired, or not authorized for the provided VLM captioning endpoint | Report the authentication error from `paidf_docker.log`. Ask the user to export `VLM_API_KEY` or add it to a user-approved env file the run sources before retrying. After the user says to proceed, double-check that `VLM_API_KEY` is set before launching PAIDF again. Do not ask the user to paste the key into chat; never create the key value yourself and never print it. |
| User has not provided PAIDF GPU count | The skill cannot infer the correct PAIDF GPU count from hardware alone because the desired allocation depends on the run plan and shared resources | Ask the user for `PAIDF_NUM_GPUS` before prepare/run. Do not guess. |
| PAIDF appears to use the wrong GPU count | `PAIDF_NUM_GPUS` was set incorrectly for the run | Report the value used for `--paidf-num-gpus` and Docker `--gpus`. Ask the user for the corrected GPU count; if they provide it, rerun `prepare_paidf_config.py` and then the Docker command with the same corrected value. |
| `prepare_paidf_config.py` fails with `file is not writable` | A previous Docker run or manual setup left root-owned or read-only files under `output_dir` | Report the exact path and permission error. If the script printed a `chown`/`chmod` command, ask the user before running it because it changes file ownership/permissions. |
| PAIDF cannot write outputs | `output_dir` is not writable by the container, or the container is writing outside the prepared output directories | Report the failing output path from Docker output when available and the current ownership/permissions of `OUTPUT_DIR`. Ask the user whether to fix permissions, choose a different output directory, or stop. |
| PAIDF cannot read input media | One or more `media_path` values are outside `MEDIA_DIR`, missing on the host, or not visible in Docker through the 1:1 mount | Report the failing media path and `MEDIA_DIR`. Ask the user whether to correct `media.jsonl`, provide a different media directory, or stop. |
| `failed_videos.jsonl` is non-empty | PAIDF did not produce every expected `.mp4` | Report failed row count, generated row count, expected unique media count, and generated video count. |
| Missing input field error | An input row is missing `id` or `media_path` | Report the input file and row number from the error. Ask the user or upstream workflow to provide a corrected input JSONL; do not fabricate ids or media paths. |
