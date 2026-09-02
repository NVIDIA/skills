---
name: paidf-augmentation
description: >-
  Use when authoring or validating PAIDF augmentation YAML configs, or running
  remote Cosmos Transfer/Predict, image-edit, or image-to-video inference.
license: Apache-2.0
metadata:
  owner: NVIDIA
  service: physical-ai-data-factory
  version: 1.1.0
  reviewed: '2026-08-31'
  author: NVIDIA
  tags:
    - physical-ai
    - augmentation
    - cosmos
    - image-edit
---

# PAIDF Augmentation Pipeline Skill

Unified pipeline for augmenting camera data through NVIDIA generative AI models with automated captioning, generation, and quality evaluation. **BYOM (bring-your-own-model):** every model is reached over a remote HTTP endpoint described by one entry in the config's `endpoints:` list; adding a model is usually a config change, not code.

## Purpose

Use this skill to drive the PAIDF augmentation pipeline end to end:

- **Select the right model** — Cosmos Transfer 2.5 (transform a video), Cosmos Predict 2.5 (generate/extend video), image-edit (edit an image), or image-to-video (animate a first frame: Cosmos3 or Veo 3.1).
- **Author and validate YAML configs** against the `PipelineConfig` Pydantic schema.
- **Configure captioning** (VLM, LLM, deterministic VLM-template, text, or file) and **evaluators** (hallucination check, attribute verification, VLM verification).
- **Launch and run** inference inside the `paidf-augmentation:1.1.0` Docker container (remote-API only — no local model weights).

Use this skill when running inference, authoring or editing configs, debugging validation or runtime errors, adding data samples, configuring captioning, tuning generation parameters, registering BYOM endpoints/adapters, or setting up evaluators. Trigger keywords: augmentation, cosmos transfer, cosmos predict, image edit, image-to-video, veo, image attribute augmentation, defect image generation, captioning, attribute verification, config validation.

Do **not** use this skill for training or fine-tuning models, deploying clusters or NIM endpoints, or unrelated application/database development.

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Docker** | `docker --version`. The image is **remote-API only** — it bundles no Cosmos/torch weights, so plain remote inference needs **no GPU and no `HF_TOKEN`**. |
| **NVIDIA GPU** (conditional) | Only for the `data_processing.alignment` post-processor (cupy) and **H.264** decode (evaluators, `data_processing.transcode`). See Limitations. |
| **Endpoint URLs** | One reachable URL per role the config uses: the model role (`video_transfer`/`video_predict`/`image_edit`/`image2video`) plus `vlm`/`llm` for captioning and evaluation. Defaults are local Qwen vLLM servers (`Qwen/Qwen3.6-27B-FP8` on `vlm`, `Qwen/Qwen2.5-14B-Instruct` on `llm`). If the user has none running, ask for URLs. |
| **API keys** (conditional) | Only for endpoints requiring auth. Passed by env var named in each endpoint's `api_key_env` — never hardcoded in YAML. Common: `VLM_API_KEY`, `LLM_API_KEY`, `VEO_API_KEY`, `BUILD_NVIDIA_API_KEY`. Local endpoints need none. |
| **Input media** | A video (transfer/predict) or image (edit/image2video) reachable by `multistorageclient` — local path, `s3://`, `gs://`, `az://`, or HTTP. |

## Inputs

Resolve each value in this precedence order: **state file → explicit prompt arguments → agent context → user prompt.** Ask the user only for what remains unresolved.

| Input | Required | Description |
|-------|----------|-------------|
| `config_path` | Yes | Path to the pipeline YAML, e.g. `configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml`. If absent, pick a starting config from *Supported Models* and confirm with the user. |
| `input_media` | Yes | Source video/image → `data[].inputs.rgb`. Overridable at run time via `data.0.inputs.rgb=...`. |
| `output_paths` | Yes | `data[].output.{video,caption,metadata}`; `evaluation` optional. |
| `model_name` | Yes | `augmentation.model.name` — an endpoint `id`, a role, or a known model name. Free-form string, not an enum. |
| `endpoint_urls` | Yes | One `endpoints[]` entry per role in use. |
| `api_key_env` | If auth | Env-var *name* per endpoint; the value comes from the environment. |
| `target_attributes` | No | `captioning.llm.variables` (e.g. `weather_condition`, `lighting_condition`). |
| `generation_params` | No | `augmentation.parameters` — pass-through; only set knobs are sent. |
| `seed` | No | Under `augmentation.parameters`; `null` = random, re-rolled on retry. |

## BYOM model: endpoints, adapters, roles

The pipeline never embeds an SDK per model. Instead:

- **`endpoints:` is a list.** Each entry has `role`, `url`, `model` (the wire model string), an optional `id` (only to disambiguate 2+ endpoints sharing a role), an optional `adapter` (API contract; defaults from the role), `api_key_env`, and `timeout`.
- **Roles**: `vlm`, `llm` (captioning + evaluators), `image_edit`, `video_transfer` (Cosmos Transfer), `video_predict` (Cosmos Predict), `image2video` (Cosmos3 / Veo).
- **Adapters (API contracts)**: `openai.chat.completions`, `openai.images.edits`, `openai.video.sync`, `openai.video.async`, `nim`, `passthrough`. The same model can be served over different contracts by changing only the endpoint's `adapter` field.
- **Model selection**: `augmentation.model.name` resolves to an endpoint by `id`, else by `role`, else by the model-name→role map (`image-edit`→`image_edit`, `cosmos-transfer2.5`→`video_transfer`, `cosmos-predict`→`video_predict`, `cosmos3-image2video`→`image2video`).

## Supported Models

When the user hasn't specified a model, choose from their **input type and goal**:

| Input Type → Goal | `model.name` | Role / default adapter | Input → Output |
|-------------------|--------------|------------------------|----------------|
| Video — change scene attributes (weather, lighting, style) | `cosmos-transfer2.5` | `video_transfer` / `nim` | Video (+ controls) → Video |
| Video + text — extend or predict continuation | `cosmos-predict` | `video_predict` / `nim` | Video+Text → Video |
| Text only — generate video from scratch | `cosmos-predict` (`inference_type: text2world`) | `video_predict` / `nim` | Text → Video |
| Image — edit specific attributes | `image-edit` | `image_edit` / `nim` (or `openai.chat.completions`, `openai.images.edits`) | Image → Image |
| Image — animate a first frame | `cosmos3-image2video` (or your Veo endpoint id) | `image2video` / `openai.video.sync` (Veo: `openai.video.async`) | Image + prompt → Video |

**Key rule**: video in + scene-attribute change → **Cosmos Transfer**. Generate new video from text/image/video conditioning → **Cosmos Predict**. Single image edit → **image edit**. Still image → moving clip → **image-to-video**.

All models run via remote HTTP through one `BaseExecutor`; there is no local `torchrun` and no `executor_type` field.

## Usage

### Step 1: Launch the Docker Container

Set `PAIDF_IMAGE_ID` to the immutable `sha256:` image ID recorded from the
trusted local build (or supplied in trusted release metadata). The image ID is
build- and architecture-specific, so this repository cannot provide one
universal value. Verify that the mutable convenience tag still resolves to the
expected ID, then run the ID directly:

```bash
set -e

PAIDF_IMAGE_ID="sha256:<expected-image-id>"
test "$(docker image inspect --format '{{.Id}}' paidf-augmentation:1.1.0)" = "$PAIDF_IMAGE_ID"
docker network inspect paidf >/dev/null 2>&1 || \
  docker network create paidf

docker run -it --rm \
  --network paidf \
  -v "$(pwd)/modules:/workspace/modules" \
  -v "$(pwd)/configs:/workspace/configs" \
  -v "$(pwd)/data:/workspace/data" \
  --entrypoint /bin/bash \
  "$PAIDF_IMAGE_ID"
```

Do not derive `PAIDF_IMAGE_ID` from the tag and immediately trust it; compare
the tag against the digest recorded when the image was built or published. If
a registry release provides a signed manifest, verify that signature before
pulling and use its `name:tag@sha256:<manifest-digest>` reference instead.

- **Networking:** augmentation only makes outbound requests, so it needs no
  `-p`/`--publish` ports. Keep the shared `paidf` bridge shown above for remote
  endpoints. For another model container, attach it to the same bridge and use
  its container name in the endpoint URL. Run host-local models in a container
  on that bridge, or use a remote endpoint; do not grant the augmentation
  container access to the host network.
- **API keys:** prefer a platform secrets manager that injects the required
  environment variables. Otherwise, export only the required keys and forward
  their names with `-e VAR_NAME`; never mount or load a broad credential file.
- **No GPU needed for remote inference** — add `--gpus` for `data_processing.alignment` and any **H.264** decode; pick a GPU not shared with a busy model server. Container runs as uid 10000; ensure `data/` is writable (or `--user "$(id -u):$(id -g)"`).

> **Security:** Host networking is prohibited for this workflow, especially
> when API keys are present. Review
> [pipeline-operations.md](references/pipeline-operations.md#security-notes).

### Step 2: Run the Pipeline (Inside the Container)

```bash
uv run --no-sync modules/cli.py --config configs/<config_file>.yaml

# With OmegaConf CLI overrides (dot-list syntax)
uv run --no-sync modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml \
  data.0.inputs.rgb=/workspace/data/input.mp4 \
  augmentation.parameters.seed=42
```

Environment variables: keys resolve as the `api_key_env` var → the role's default env var. If `api_key_env` names an **unset** var, resolution falls back to the role default; leave it off for unauthenticated endpoints. `LOG_LEVEL` sets logging.

## Configuration Schema

Configs are validated against `PipelineConfig` (`modules/aug_utils/schema/`) and have seven top-level sections: `data`, `endpoints` (a **list**), `pipeline`, `captioning`, `augmentation`, `data_processing`, and `evaluators`. Full per-section YAML is in [configuration-schema.md](references/configuration-schema.md); runtime flow and common editing tasks are in [pipeline-operations.md](references/pipeline-operations.md).

## Examples

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../configs/cookbook/README.md) for the folder layout.

| Use case | Config(s) |
|----------|-----------|
| Video scene-attribute transfer (CT2.5, `nim`) | `config_video_transfer_CT25_nim.yaml` |
| Image → video | `config_image2video_cosmos3.yaml` (VLM→LLM) · `config_image2video_cosmos3_vlm_template.yaml` (VLM→template) · `config_image2video_veo31.yaml` (Veo 3.1, async) |
| Image Attribute Augmentation | `config_image_edit_attribute_{chat_api,images_api,nim}.yaml` · `…_gemma_llm.yaml` (hosted-Gemma LLM swap) |
| Defect Image Generation + MI alignment | `config_image_edit_defect_{chat_api,images_api}.yaml` |
| Batch config generation | `workflow_example.yaml` · `attribute_distribution_1000_v1.yaml` |
| Smart-space seed image / event video | `config_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml` · `config_event_video_gen_cosmos3_smart_spaces.yaml` |

Per-config captioning / evaluator / adapter details are in [config-decision-tree.md](references/config-decision-tree.md).

## Troubleshooting

Run all inference and schema validation **inside the Docker container** for a consistent environment. For config-validation errors, runtime/endpoint errors, and typical per-stage timings, see [troubleshooting.md](references/troubleshooting.md).

## Limitations

- **Remote inference only.** All models run behind remote HTTP endpoints; no local weights, no `torchrun`, no `executor_type`, no Gradio executor.
- **GPU for alignment and H.264 decode.** Remote inference needs no GPU. A CUDA GPU is required by `data_processing.alignment` (cupy) and by anything decoding H.264 — the evaluators and `data_processing.transcode` — because the image ships only the hardware `h264_cuvid` decoder (software AVC decode is off for licensing). VP9 decodes in software. Video **output** is VP9-only.
- **Inference only.** This pipeline augments and generates media — it does not train or fine-tune models.
- **Auth varies by endpoint.** Hosted endpoints (e.g. Veo) need a key via `api_key_env`; local endpoints (e.g. vLLM) need none.

## Reference files

- [configuration-schema.md](references/configuration-schema.md) — full per-section YAML for every config section.
- [config-decision-tree.md](references/config-decision-tree.md) — which config to start from, model/captioning selection, alignment override rules.
- [pipeline-operations.md](references/pipeline-operations.md) — pipeline flow, worked example, common tasks, storage, security notes.
- [captioning-strategy-guide.md](references/captioning-strategy-guide.md) — all 6 captioning modes with complete YAML.
- [evaluator-setup-guide.md](references/evaluator-setup-guide.md) — hallucination tuning, attribute verification, MCQ wiring.
- [troubleshooting.md](references/troubleshooting.md) — validation/runtime errors and per-stage timings.
- [image-attribute-augmentation.md](references/image-attribute-augmentation.md) — Image Attribute Augmentation image-edit workflow and dataset packaging.
- [event-video-gen.md](references/event-video-gen.md) — smart-space image-to-video event generation.
