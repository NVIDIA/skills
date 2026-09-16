# Event Video Generation

Use this reference for Smart Spaces seed-image creation and event video
generation with Cosmos 3 Super. It covers text-to-image (T2I) seed creation,
image-to-video (I2V) event generation, distribution workflows, and VLM
verification.

Keep this as a reference under the augmentation skill. Do not create a separate
end-to-end Event Video Generation skill.

## Contents

- [Core Rules](#core-rules)
- [Supported Configs](#supported-configs)
- [Deterministic Docker CLI Workflow](#deterministic-docker-cli-workflow)
- [Create One Seed Image](#create-one-seed-image)
- [Create Multiple Seed Images](#create-multiple-seed-images)
- [Generate One Event Video](#generate-one-event-video)
- [Generate Multiple Event Videos](#generate-multiple-event-videos)
- [Prompt Requirements](#prompt-requirements)
- [VLM Verification](#vlm-verification)
- [Validation and Output Inspection](#validation-and-output-inspection)
- [Troubleshooting](#troubleshooting)

## Core Rules

- Use the checked-in Smart Spaces configs as templates. Make runtime copies for
  concrete endpoint URLs, input paths, output paths, environment descriptions,
  and anomaly distributions.
- Do not write API keys into YAML, shell history, logs, or generated configs.
  Pass them through the environment variables named by each endpoint's
  `api_key_env`.
- Preserve the seed image as the first frame of I2V output. Keep the camera,
  crop, perspective, layout, lighting, fixed objects, and background stable.
- Generate one explicit, physically plausible, non-graphic anomaly per clip.
  Prefer a visible setup, main action, and short aftermath.
- Keep runtime and distribution configs separate. Runtime configs execute one
  sample; distribution configs create multiple runtime configs.
- Use the generic `data[*].output.video` schema field for generated media. In
  the T2I config it intentionally points to a `.png`; in the I2V config it
  points to an `.mp4`.
- Use remote endpoints without local GPU flags. Add local GPU access only when
  a model endpoint or another processing stage runs inside the same container.
- Do not modify pipeline code to change environments, anomalies, endpoints,
  seeds, or output paths. These are config or CLI overrides.

## Supported Configs

| Workflow | Config | Purpose |
| --- | --- | --- |
| T2I runtime | `configs/cookbook/event-video-generation/config_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml` | Create one seed image from an environment description |
| T2I distribution | `configs/cookbook/event-video-generation/distribution_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml` | Create multiple T2I runtime configs |
| I2V runtime | `configs/cookbook/event-video-generation/config_event_video_gen_cosmos3_smart_spaces.yaml` | Create and verify one event video from one seed image |
| I2V distribution | `configs/cookbook/event-video-generation/distribution_event_video_gen_smart_spaces.yaml` | Create multiple I2V runtime configs from one or more seed images |

The runtime configs use endpoint-registry entries. T2I requires a prompt LLM
and Cosmos 3 Super Text2Image endpoint. I2V requires scene-understanding,
prompt-generation, question-generation, answering VLM/LLM endpoints, and a
Cosmos 3 Super Image2Video endpoint.

The OpenAI client appends `/chat/completions`, so configure chat endpoints with
a base URL ending in `/v1`, not with the full `/chat/completions` route. The
`openai.video.sync` adapter posts to `{url}/videos/sync`; keep the I2V endpoint
base URL ending in `/v1` when that is the service contract.

## Deterministic Docker CLI Workflow

Build the augmentation image from the repo root when a fresh local image is
required:

```bash
docker build -t paidf-augmentation:latest -f docker/Dockerfile .
```

For a prebuilt image, replace `paidf-augmentation:latest` with the exact tag or
digest being validated. Mount the current branch's configs and modules
read-only, and mount input and output directories separately:

```bash
HOST_INPUT="/absolute/path/to/smart_spaces_inputs"
HOST_OUTPUT="/absolute/path/to/smart_spaces_outputs"
mkdir -p "$HOST_OUTPUT"
docker network inspect paidf >/dev/null 2>&1 || \
  docker network create paidf

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --network paidf \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  -e VLM_API_KEY \
  -e LLM_API_KEY \
  -e BUILD_NVIDIA_API_KEY \
  -e VLM_ENDPOINT_MODEL \
  -e LLM_ENDPOINT_MODEL \
  -v "$(pwd)/modules:/workspace/modules:ro" \
  -v "$(pwd)/configs:/workspace/configs:ro" \
  -v "$HOST_INPUT:/workspace/data/in:ro" \
  -v "$HOST_OUTPUT:/workspace/data/out" \
  -w /workspace \
  --entrypoint /bin/bash \
  paidf-augmentation:latest
```

Mount the subdirectories individually, as above. Never mount over `/workspace`
itself (`-v "$(pwd):/workspace"`) — it hides the image's venv at `/workspace/.venv`.

Augmentation is an outbound client, so publish no ports with `-p`/`--publish`.
Keep the shared `paidf` bridge above for remote endpoints. When a model runs in
another container, attach it to the same bridge and use its container name in
the endpoint URL. Run host-local models in a container on the same bridge, or
use a remote endpoint.

> **Security:** Host networking is prohibited because it exposes host loopback
> services, internal interfaces, and metadata endpoints such as
> `169.254.169.254` to the container. Use only the isolated `paidf` bridge.

The commands below assume `/workspace` as the working directory inside the container.

## Create One Seed Image

Use the T2I runtime config to create one normal Smart Spaces seed image. Supply
an environment type, an optional detailed environment description, and unique
output paths:

```bash
uv run modules/cli.py \
  --config configs/cookbook/event-video-generation/config_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml \
  'captioning.llm.variables.env_type=[warehouse]' \
  'captioning.llm.variables.env_description=[warehouse receiving area with pallet racks, loading doors, floor markings, and 2-3 adult workers]' \
  data.0.output.video=/workspace/data/out/seeds/warehouse.png \
  data.0.output.caption=/workspace/data/out/seeds/warehouse_prompt.txt \
  data.0.output.metadata=/workspace/data/out/seeds/warehouse_metadata.json
```

Override both T2I endpoint URLs before running when the defaults still point to
`localhost:8000`. Keep the endpoint IDs unchanged unless the corresponding
consumer `endpoint_id` or model name is updated too.

Use `1024x576` unless the deployed Cosmos 3 Super T2I endpoint documents
another supported size. This size is known to satisfy the model's distributed
sequence-length constraints; some smaller 16:9 sizes do not.

## Create Multiple Seed Images

Make a runtime copy of the T2I distribution config and set these fields:

```yaml
workflow_type: seed_image_gen_t2i
env_type: warehouse
env_description: warehouse receiving area with pallet racks and 2-3 adult workers
n_augmentations: 5
example_augmentation_config: /workspace/configs/cookbook/event-video-generation/config_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml
config_output: /workspace/data/out/generated_configs/seed_image_gen_t2i
output_root: /workspace/data/out/generated_seed_images
output_basename: seed_image
seed_base: 42
```

Generate one runtime YAML per requested seed image:

```bash
uv run python \
  modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow /workspace/data/out/distribution_seed_image_gen.yaml
```

Paths in a distribution file are resolved relative to that file. Use absolute
container paths when the runtime copy is outside `/workspace/configs`.

For `n_augmentations: 5`, the generated configs use deterministic seeds from
`seed_base` through `seed_base + 4` and produce this layout:

```text
/workspace/data/out/generated_seed_images/
└── warehouse/
    ├── aug_0/
    │   ├── seed_image.png
    │   ├── seed_image_prompt.txt
    │   └── seed_image_metadata.json
    └── ...
```

Run each generated runtime config separately:

```bash
find /workspace/data/out/generated_configs/seed_image_gen_t2i \
  -type f -name '*.yaml' -print0 | sort -z | \
while IFS= read -r -d '' config; do
  uv run modules/cli.py --config "$config"
done
```

## Generate One Event Video

Use the I2V runtime config with an existing seed image. Override the input,
anomaly, environment, and every output path for a reproducible run:

```bash
uv run modules/cli.py \
  --config configs/cookbook/event-video-generation/config_event_video_gen_cosmos3_smart_spaces.yaml \
  data.0.inputs.rgb=/workspace/data/in/retail.png \
  'captioning.llm.variables.anomaly_type=[person_falling]' \
  'captioning.llm.variables.env_type=[retail]' \
  augmentation.parameters.seed=86000 \
  data.0.output.video=/workspace/data/out/videos/retail_person_falling.mp4 \
  data.0.output.caption=/workspace/data/out/videos/retail_person_falling_prompt.txt \
  data.0.output.metadata=/workspace/data/out/videos/retail_person_falling_metadata.json \
  data.0.output.evaluation=/workspace/data/out/videos/retail_person_falling_evaluation.json
```

Before running, replace the `cosmos3-super-image2video` endpoint's default
`localhost:8000` URL with the deployed I2V base URL, or add the equivalent
`endpoints.4.url=https://<i2v-host>/v1` CLI override for this exact template.
Retarget the four chat endpoints too when they are not served from the
checked-in NVIDIA API base URL.

Use anomaly values that describe a direct visible action. The checked-in
prompt supports these values explicitly:

```text
person_falling
person_climbing
person_running
smoking_or_vaping
person_fighting
fire_or_smoke
```

Other values are accepted as prompt variables, but verify that the prompt
generator and evaluator questions describe the intended event clearly. Update
evaluator `extra_questions` when a custom anomaly needs fixed, event-specific
questions.

## Generate Multiple Event Videos

Use the I2V distribution config instead of a shell loop when generating a
dataset. Put one or more supported seed images under `data_dir`, make a runtime
copy of the distribution YAML, and set concrete paths:

```yaml
workflow_type: event_video_gen_i2v
data_dir: /workspace/data/in/seeds
n_augmentations: 2
example_augmentation_config: /workspace/configs/cookbook/event-video-generation/config_event_video_gen_cosmos3_smart_spaces.yaml
config_output: /workspace/data/out/generated_configs/event_video_gen
output_root: /workspace/data/out/event_video_outputs
variables:
  anomaly_type:
    person_falling: 0.5
    fire_or_smoke: 0.5
```

Weights are relative positive probabilities. Use one value with weight `1.0`
when every generated clip must have the same anomaly.

Generate runtime configs:

```bash
uv run python \
  modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow /workspace/data/out/distribution_event_video_gen.yaml
```

For each seed image, the generator writes `n_augmentations` runtime configs and
uses this output layout:

```text
/workspace/data/out/event_video_outputs/
└── <seed_stem>/
    ├── aug_0/
    │   ├── output.mp4
    │   ├── output_prompt.txt
    │   ├── output_metadata.json
    │   └── output.mp4_evaluation.json
    └── ...
```

Run the generated configs in deterministic filename order:

```bash
find /workspace/data/out/generated_configs/event_video_gen \
  -type f -name '*.yaml' -print0 | sort -z | \
while IFS= read -r -d '' config; do
  uv run modules/cli.py --config "$config"
done
```

## Prompt Requirements

Build the final I2V prompt in this order:

```text
[Cinematography] + [Seed Preservation] + [Environment] + [Subject] +
[Action] + [Context] + [Style and Negative Constraints]
```

Require the prompt generator to:

- Use the seed image as the first frame.
- Preserve the exact crop, perspective, field of view, layout, lighting, object
  locations, color palette, and background.
- Lock the camera: no pan, tilt, zoom, dolly, tracking, handheld motion,
  reframing, cuts, montage, or time-lapse.
- Preserve existing people and objects without popping, disappearing,
  teleporting, morphing, duplication, or scale changes.
- Add only the minimum motion required for one anomaly.
- Show the action across multiple frames with realistic contact, weight,
  balance, and aftermath.
- Avoid graphic injury, visible camera hardware, text/UI overlays, cinematic
  effects, or unrelated side events.

## VLM Verification

The verifier decodes the generated MP4 in the augmentation container, samples
frames evenly across the complete clip, JPEG-encodes them, and sends them as
chronological `image_url` items in one VLM request per question.

Set `frames` to the total number of sampled images, not the source frame rate.
Cosmos 3 Super produces approximately 8 seconds at 24 fps, or about 192 source
frames. The Smart Spaces config samples 24 frames, which gives approximately 3
observations per second across the setup, action, and aftermath:

```yaml
evaluators:
  - attribute_verification:
      vlm_verification:
        endpoint_id: answering_vlm
        frames: 24
```

The schema default is one first frame, so set `frames` explicitly for events
that occur later in a generated video. More frames improve temporal coverage
but increase local decode time, request size, and VLM vision processing. Frame
sampling requires a working CPU video decoder in the augmentation container;
it does not require the CUDA `h264_cuvid` decoder.

## Validation and Output Inspection

Validate the shipped runtime configs and the targeted tests before running an
expensive endpoint job:

```bash
uv run pytest -q \
  tests/schema/test_schema.py \
  tests/config_distribution_generation/test_generate_augmentation_configs.py \
  tests/verification/test_verifier_adapter.py
```

For generated distribution configs, first inspect one YAML and confirm:

- `data[0].inputs.rgb` points to the intended seed image or is `null` for T2I.
- T2I output media ends in `.png`; I2V output media ends in `.mp4`.
- `captioning.llm.variables` contains the expected environment or anomaly.
- The selected endpoint IDs exist in `endpoints` with the correct roles.
- Output paths are writable inside the container.

After a run, inspect the files instead of relying only on process exit status:

```bash
test -s /workspace/data/out/videos/retail_person_falling.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate \
  -show_entries format=duration,size \
  -of json /workspace/data/out/videos/retail_person_falling.mp4 | jq .
jq '.attribute_verification // .evaluators // .' \
  /workspace/data/out/videos/retail_person_falling_metadata.json
jq . /workspace/data/out/videos/retail_person_falling_evaluation.json
```

Report generation and verification separately. A valid MP4 can exist even when
one or more evaluator questions fail.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `401` or `403` | Missing or incorrect endpoint key | Check the endpoint's `api_key_env` and pass that variable with Docker `-e`; never put the key in YAML |
| Chat endpoint returns `404` | Config contains the full `/chat/completions` path | Use the OpenAI-compatible base URL ending in `/v1` |
| Video endpoint returns `404` or `405` | Adapter and service route do not match | Use the adapter required by the service; `openai.video.sync` expects `{url}/videos/sync` |
| T2I fails for a small 16:9 resolution | Distributed sequence length is not divisible for that size | Retry with the verified `1024x576` setting |
| Generated distribution config cannot find its template | Relative path was resolved from the runtime workflow copy's directory | Set `example_augmentation_config` to an absolute container path |
| No distribution configs are generated | `data_dir` contains no supported media, or the mounted path is wrong | Inspect the mount and use image files for the I2V distribution input |
| MP4 exists but verification fails | Anomaly is unclear, evaluator questions do not match, or the answering response is unparseable | Inspect the prompt, questions, raw answers, and evaluation JSON; adjust config prompts/questions without changing generation code |
| Frame verification reports a decoder error | The container cannot decode the MP4 with its configured backend | Use a compatible CPU decoder and confirm it can read the generated H.264 MP4 |
| Output path is not writable | Container UID cannot write to the mounted host directory | Run with the host UID/GID or correct directory ownership; do not use world-writable permissions on shared systems |
