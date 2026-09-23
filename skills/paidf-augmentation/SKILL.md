---
name: paidf-augmentation
description: >-
  Use when authoring or validating PAIDF augmentation YAML configs, or running
  remote Cosmos Transfer (including Cosmos3 WSM controls), Cosmos Predict,
  image-edit, or image-to-video inference.
license: Apache-2.0
metadata:
  owner: NVIDIA
  service: physical-ai-data-factory
  version: 1.2.0
  reviewed: '2026-09-16'
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - physical-ai
    - augmentation
    - cosmos
    - wsm
    - image-edit
---

# PAIDF Augmentation Pipeline Skill

Remote-API pipeline for captioning, generating, and evaluating augmented camera
data. Models are configured as HTTP endpoints; no local model weights are
included.

## Purpose

Use it to select a supported generation mode, author and validate
`PipelineConfig` YAML, configure captioning/evaluators, and run the
`paidf-augmentation:1.2.0` container. It covers Cosmos Transfer/Predict,
Cosmos3 WSM controls, image editing, image-to-video, BYOM endpoints, and
quality gates.

Do **not** use this skill for training or fine-tuning models, deploying clusters or NIM endpoints, or unrelated application/database development.

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Docker** | `docker --version`. The image is **remote-API only** — it bundles no Cosmos/torch weights, so plain remote inference needs **no GPU and no `HF_TOKEN`**. |
| **NVIDIA GPU** (conditional) | Only when a configured local stage requires CUDA or decodes **H.264** in the augmentation container. See Limitations. |
| **Endpoint URLs** | One reachable URL per role used by the selected config. The examples name local Qwen services (`Qwen/Qwen3.6-27B-FP8` on `vlm`, `Qwen/Qwen2.5-14B-Instruct` on `llm`), but those are not guaranteed to be running. Resolve every required role independently: keep each reachable configured endpoint and ask only for the missing role URLs. |
| **API keys** (conditional) | Only for endpoints requiring authentication. Pass the environment variable named by each endpoint's `api_key_env`; never hardcode values in YAML. Local unauthenticated endpoints need none. |
| **Input media** | A video/image reachable by `multistorageclient` (`s3://`, `msc://`, `gs://`, `az://`, or HTTPS). Local paths are allowed **only when no external Cosmos evaluator is enabled**; external evaluation requires checker-readable remote URIs. |
| **Approved release commit** (execution only) | A 40-character commit hash confirmed by the user as release-owner-approved or retrieved from the verified release manifest. Never invent it or rely on an unverified environment value. |

## Inputs

Resolve each value in this precedence order: **state file → explicit prompt arguments → agent context → user prompt.** Ask the user only for what remains unresolved.

| Input | Required | Description |
|-------|----------|-------------|
| `config_path` | Yes | Path to the pipeline YAML, e.g. `configs/cookbook/video-data-augmentation/config_video_transfer_CT3_omni.yaml`. If absent, pick a starting config from *Supported Models* and confirm with the user. |
| `input_media` | Conditional | Source video/image → `data[].inputs.rgb`. Required for every mode except Cosmos Predict `inference_type: text2world`, where `inputs` may be null or `rgb` omitted. Overridable at run time via `data.0.inputs.rgb=...`. |
| `control_media` | For precomputed control | Control video → `data[].inputs.controls.<type>`. For Cosmos3 WSM transfer, set `controls.wsm`; the client uploads it rather than exposing its local path to the server. |
| `output_paths` | Yes | `data[].output.{video,caption,metadata}`; `evaluation` optional. |
| `model_name` | Yes | `augmentation.model.name` — an endpoint `id`, a role, or a known model name. Free-form string, not an enum. |
| `endpoint_urls` | Yes | One `endpoints[]` entry per role in use. |
| `api_key_env` | If auth | Env-var *name* per endpoint; the value comes from the environment. |
| `target_attributes` | No | `captioning.llm.variables` (e.g. `weather_condition`, `lighting_condition`). |
| `generation_params` | No | `augmentation.parameters` — pass-through; only set knobs are sent. |
| `seed` | No | Under `augmentation.parameters`. Before the first candidate, `null` resolves once to the generator/executor seed when available, otherwise the current Unix time. Retry `n` uses that resolved base seed plus `n`. `pipeline.retry` defaults to `1`, so evaluation permits at most 2 candidates by default. |

## BYOM model: endpoints, adapters, roles

Each `endpoints:` list entry declares `role`, `url`, wire `model`, and optional
`id`, `adapter`, `api_key_env`, and `timeout`. Roles are `vlm`, `llm`,
`image_edit`, `video_transfer`, `video_predict`, `image2video`, and `evaluator`.
`augmentation.model.name` resolves by endpoint `id`, then role, then the known
model-name mapping. Adapter contracts and full fields are in
[configuration-schema.md](references/configuration-schema.md).

## Supported Models

When the user hasn't specified a model, choose from their **input type and goal**:

| Input Type → Goal | `model.name` | Role / default adapter | Input → Output |
|-------------------|--------------|------------------------|----------------|
| Video — change scene attributes (weather, lighting, style) | `cosmos-transfer2.5` | `video_transfer` / `nim` | Video (+ controls) → Video |
| Video + precomputed WSM — follow world-state geometry/motion | A `video_transfer` endpoint id, e.g. `cosmos3-transfer-wsm` | `video_transfer` / `openai.video.async` | RGB video + WSM control + prompt → Video |
| Video + text — extend or predict continuation | `cosmos-predict` | `video_predict` / `nim` | Video+Text → Video |
| Text only — generate video from scratch | `cosmos-predict` (`inference_type: text2world`) | `video_predict` / `nim` | Text → Video |
| Image — edit specific attributes | `image-edit` | `image_edit` / `nim` (or `openai.chat.completions`, `openai.images.edits`) | Image → Image |
| Image — animate a first frame | `cosmos3-image2video` (or your Veo endpoint id) | `image2video` / `openai.video.sync` (Veo: `openai.video.async`) | Image + prompt → Video |

**Key rule:** use Cosmos Transfer for video scene changes, Cosmos Predict for
new/continued video, image edit for still edits, and image-to-video to animate a
frame. For Cosmos3 WSM, set `data[].inputs.controls.wsm`. Config selection
details are in [config-decision-tree.md](references/config-decision-tree.md).

All models run via remote HTTP through one `BaseExecutor`; there is no local `torchrun` and no `executor_type` field.

## Canonical Workflow

Follow this ordered decision tree; open the linked references only for the
selected mode's details.

### Config Correction Budget

`config_attempts` counts YAML versions, not validation or preflight calls. Keep
it across the whole workflow and never reset it.

1. Set it to `1` after initial authoring.
2. On a correctable error, if it is already `3`, **STOP** and report the error.
   Otherwise increment it once and make one correction pass containing all
   currently reported YAML fixes.
3. Validate and preflight that edited version without another increment. Repeat
   rule 2 only if a later check finds another correctable YAML error.

### STEP 0 — Mandatory Entry Gate

Execute this gate before every other step:

- **Scope gate:** if the request asks for training, fine-tuning, cluster/NIM
  deployment, or unrelated development, decline it and **STOP**.
- **Input gate:** resolve values using *Inputs*. If required values remain
  unresolved, request only those values and **STOP** until they are supplied.
- Proceed only when both gates pass and the input type and augmentation goal
  are known.

1. Select the model from *Supported Models* using that input type and goal.
2. Author the YAML in this order: `data`, `endpoints`, `captioning`,
   `augmentation`, `evaluators`, `pipeline`, then `data_processing`. Omit
   optional sections rather than creating placeholders. Set
   `config_attempts = 1` (**config attempt 1 of 3**).
3. Branch on the requested outcome:
   - **Configuration only or “do not execute”:** do not build, launch, run
     preflight, or infer. A trusted runtime is an already-running container
     whose recorded immutable image ID/digest, checkout commit, and config
     mounts satisfy step 4. If validation was explicitly requested and such a
     runtime exists, run only the validation command and apply step 5's failure
     rules. Return the config (mark it unvalidated if validation was not run)
     and **STOP**.
   - **Inference/run requested:** proceed to step 4.
4. Prepare the runtime in these verifiable sub-steps:
   a. Obtain `EXPECTED_RELEASE_REF` from the user as a release-owner-approved
      full commit or retrieve it from the verified release manifest.
   b. Verify the Git working tree is clean.
   c. Verify `HEAD` matches `EXPECTED_RELEASE_REF`. If a release tag selected
      the revision, verify the tag first, then compare its resolved full commit.
   d. Build the image and capture its immutable `sha256:` image ID, or obtain a
      verified registry digest.
   e. Create the `paidf` Docker bridge if needed and attach local services.
   f. Launch the container using the recorded image ID, the `paidf` network,
      required volume mounts, `--entrypoint /bin/bash`, and only the endpoint
      key variables named by `api_key_env` plus the scoped storage credential
      variables required for the configured remote media.
   If any sub-step fails, **STOP** and report the exact failure and relevant
   remediation: clean a dirty tree, supply the approved revision, resolve a
   revision mismatch, check the Docker daemon/build, fix network setup, or fix
   the container launch. Do not continue with a partial runtime.
5. Inside the container, run the validation-only command in Usage Step 2 and
   classify its result while preserving `config_attempts` (**current config
   attempt N of 3**):
   - **Success:** if no external Cosmos evaluator is enabled, proceed directly
     to step 7. If one is enabled, proceed to step 6.
   - **Editable YAML/schema error:** when required endpoint URL/model details are
     known, apply the *Config Correction Budget* rule, then rerun validation on
     that YAML version.
   - **Infrastructure/unknown input:** for missing endpoint details, credentials,
     DNS, connectivity, or unhealthy services, do not edit speculatively.
     **STOP** and report the error or ask for the missing endpoint details.
6. With external Cosmos evaluation enabled, preserve `config_attempts`
   (**current config attempt N of 3**) and run its hard preflight gate before
   captioning/generation by querying `/health`, `/checkers`, and
   `/dependency-graph`:
   - **Pass:** require `/health.status == "healthy"`; require every configured
     `checks[].name` to exactly equal one string in the top-level `checkers`
     array returned by `/checkers` and one key in the top-level `checkers`
     object returned by `/dependency-graph`. Do not use substring, alias, or
     nested-field matching. Then proceed to step 7.
   - **Config error:** only a 4xx response from `/checkers` or
     `/dependency-graph` whose body explicitly identifies a configured checker
     name or invalid config field qualifies. Apply the *Config Correction
     Budget* rule, then return to step 5; revalidation does not increment the
     counter.
   - **Infrastructure failure:** any timeout, network/DNS error, 5xx response,
     `/health` failure, unhealthy service, or missing deployed checker stops the
     workflow before generation. Create no candidate and report the root cause.
7. Enter inference only after step 5 succeeds and, when external evaluation is
   enabled, step 6 passes. Run a candidate loop bounded to
   **`pipeline.retry + 1` candidates** (`retry` defaults to `1`, so the default
   maximum is 2). Resolve the base seed once before the first candidate as
   defined in *Inputs*; retry `n` uses `base_seed + n`:
   - Generate a candidate and run its configured evaluators.
   - On pass, accept it and exit the loop.
   - On quality failure with attempts remaining, set the next seed to
     `base_seed + next_retry_number`. If `regenerate_caption_on_retry: true`,
     regenerate the caption next; then generate the replacement. If false,
     generate the replacement immediately.
   - On exhausted quality retries, fail evaluation and exit the loop.
   - On a service/contract failure, stop on the current candidate without
     generating another. For external Cosmos evaluation, this means a
     submission, polling, or discovery failure; a failed execution; a missing
     or skipped result; or an invalid gating contract without a literal
     top-level boolean `passed`. With `pipeline.evaluation.strict: true` (the
     default), apply `retain_failures`: its default `true` retains video, caption,
     metadata, and evaluation outputs for inspection; explicit `false` deletes
     them. With `strict: false`, retain the files and report the evaluator
     failure without failing the sample.
     - If provider metadata contains `error_code: ambiguous_post`, preserve the
       retained candidate, do not resubmit automatically, **STOP**, and defer
       to operator recovery. Follow the
       [ambiguous-POST recovery flow](references/evaluator-setup-guide.md#gating-and-failures).

## Usage

### Step 1: Verify, Build, and Launch the Docker Container

Set `EXPECTED_RELEASE_REF` to a release-owner-reviewed full 40-character commit,
never a tag or branch. If a signed release tag selects the revision, verify the
tag first and record its resolved full commit as `EXPECTED_RELEASE_REF`. Build
only from a clean matching checkout, then record the resulting immutable image
ID. For a prebuilt release, use its verified digest.

```bash
set -e

EXPECTED_RELEASE_REF="${EXPECTED_RELEASE_REF:?set a reviewed full commit ID}"
test "${#EXPECTED_RELEASE_REF}" -eq 40
case "$EXPECTED_RELEASE_REF" in *[!0-9a-fA-F]*) exit 1 ;; esac
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse "${EXPECTED_RELEASE_REF}^{commit}")"

DOCKER_BUILDKIT=0 docker build \
  -t paidf-augmentation:1.2.0 \
  -f docker/Dockerfile .
PAIDF_IMAGE_ID="$(docker image inspect --format '{{.Id}}' paidf-augmentation:1.2.0)"
case "$PAIDF_IMAGE_ID" in sha256:*) ;; *) exit 1 ;; esac
test "$(docker image inspect --format '{{.Id}}' paidf-augmentation:1.2.0)" = "$PAIDF_IMAGE_ID"

docker network inspect paidf >/dev/null 2>&1 || \
  docker network create paidf
# Attach each local model container once, for example:
# docker network connect paidf vlm

# Example only: replace these with exactly the api_key_env names declared by
# the selected config; use an empty array when every endpoint is unauthenticated.
PAIDF_ENDPOINT_KEY_ARGS=()
# Authenticated example:
# PAIDF_ENDPOINT_KEY_ARGS=(-e VLM_API_KEY -e LLM_API_KEY -e VEO_API_KEY)

# Forward only credential names required by the configured remote media. Give
# checker containers their own scoped credentials in their deployment.
PAIDF_STORAGE_CREDENTIAL_ARGS=()
# S3-compatible example (include only the names your storage config requires):
# PAIDF_STORAGE_CREDENTIAL_ARGS=(-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
#   -e AWS_DEFAULT_REGION -e AWS_ENDPOINT_URL)

docker run -it --rm \
  --network paidf \
  "${PAIDF_ENDPOINT_KEY_ARGS[@]}" \
  "${PAIDF_STORAGE_CREDENTIAL_ARGS[@]}" \
  -v "$(pwd)/modules:/workspace/modules" \
  -v "$(pwd)/configs:/workspace/configs" \
  -v "$(pwd)/data:/workspace/data" \
  --entrypoint /bin/bash \
  "$PAIDF_IMAGE_ID"
```

Persist the captured `sha256:` ID and use it directly for every later launch;
never re-resolve the mutable tag as the launch target. For registry releases,
verify the signed manifest and use `name:tag@sha256:<manifest-digest>`.

- **Networking:** augmentation is an outbound-only client and does not publish
  ports. Use the user-defined `paidf` bridge for the augmentation container and
  every local model, Arbitrator, and checker container. `cosmos-evaluator` may
  be a container/DNS hostname on that bridge; it is not a second bridge name.
  Change endpoint URLs from host `localhost` addresses to container DNS names,
  for example `http://vlm:8000/v1`. The default bridge is sufficient when every
  endpoint is remote.
- **API keys:** prefer a platform secrets manager that injects the required
  environment variables. Otherwise, export only the required keys and forward
  their names with `-e VAR_NAME`; never mount or load a broad credential file.
- **No GPU needed for remote inference** — add `--gpus` for `data_processing.alignment` and any **H.264** decode; pick a GPU not shared with a busy model server. Container runs as uid 10000; ensure `data/` is writable (or `--user "$(id -u):$(id -g)"`).

> **Warning:** Never enable Docker host-network mode on shared, multi-tenant, or
> production hosts. It is permitted only when all three conditions hold: (1) a
> required host-local service cannot be moved onto the `paidf` bridge, (2) the
> host is single-tenant and isolated with no other workloads, and (3) the user
> sends an explicit message in the current conversation choosing host networking
> after receiving a warning that it removes network namespace isolation. Silence,
> prior documentation, or an earlier unrelated approval is not acceptance. Do
> not select it automatically. Review
> [pipeline-operations.md](references/pipeline-operations.md#security-notes).

### Step 2: Validate, Then Run (Inside the Container)

```bash
# Schema and cross-section preflight only: no endpoint calls or inference.
uv run --no-sync modules/cli.py --config configs/<config_file>.yaml --validate-only

# Run only after validation succeeds.
uv run --no-sync modules/cli.py --config configs/<config_file>.yaml

# With OmegaConf CLI overrides (dot-list syntax)
uv run --no-sync modules/cli.py --config configs/<config_file>.yaml \
  data.0.inputs.rgb=/workspace/data/input.mp4 \
  augmentation.parameters.seed=42
```

Environment variables: generation/captioning keys resolve as the `api_key_env`
var → the role's default env var. External Cosmos evaluation reads only the
variable explicitly named by its endpoint and fails initialization when that
variable is unset. Leave `api_key_env` off for unauthenticated endpoints.
`LOG_LEVEL` sets logging.

## Configuration Schema

Configs are validated against `PipelineConfig` (`modules/aug_utils/schema/`) and have seven top-level sections. Author them in the canonical order above; YAML key order does not change runtime semantics. Full per-section YAML is in [configuration-schema.md](references/configuration-schema.md); runtime flow and common editing tasks are in [pipeline-operations.md](references/pipeline-operations.md).

## Troubleshooting

Run all inference and schema validation **inside the Docker container** for a consistent environment. For config-validation errors, runtime/endpoint errors, and typical per-stage timings, see [troubleshooting.md](references/troubleshooting.md).

## Limitations

- **Remote inference only.** All models run behind remote HTTP endpoints; no local weights, no `torchrun`, no `executor_type`, no Gradio executor.
- **GPU for configured local stages.** Remote inference needs no augmentation-container GPU. A CUDA GPU is required by `data_processing.alignment` (cupy) and by any local stage decoding H.264 because the image ships only the hardware `h264_cuvid` decoder (software AVC decode is off for licensing). VP9 decodes in software. Video **output** is VP9-only.
- **External-evaluator media must be remote.** When an external evaluator is enabled, schema validation rejects local `data[].inputs.rgb`, control media, generated `data[].output.video`, and configured companion media. Use checker-readable `s3://`, `msc://`, `gs://`, `az://`, or HTTP(S) URIs; the pipeline does not upload local files solely for evaluation.
- **Inference only.** This pipeline augments and generates media — it does not train or fine-tune models.
- **Auth varies by endpoint.** Hosted endpoints (e.g. Veo) need a key via `api_key_env`; local endpoints (e.g. vLLM) need none.

## Reference files

- [configuration-schema.md](references/configuration-schema.md) — full per-section YAML for every config section.
- [config-decision-tree.md](references/config-decision-tree.md) — which config to start from, model/captioning selection, alignment override rules.
- [pipeline-operations.md](references/pipeline-operations.md) — pipeline flow, worked example, common tasks, storage, security notes.
- [captioning-strategy-guide.md](references/captioning-strategy-guide.md) — all 6 captioning modes with complete YAML.
- [evaluator-setup-guide.md](references/evaluator-setup-guide.md) — built-in evaluator tuning plus external Cosmos discovery, gating, and failure semantics.
- [troubleshooting.md](references/troubleshooting.md) — validation/runtime errors and per-stage timings.
- [image-attribute-augmentation.md](references/image-attribute-augmentation.md) — Image Attribute Augmentation image-edit workflow and dataset packaging.
- [event-video-gen.md](references/event-video-gen.md) — smart-space image-to-video event generation.
