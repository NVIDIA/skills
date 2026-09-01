# Image Attribute Augmentation

Use this reference for Image Attribute Augmentation: clothing/accessory edits, Qwen image edit, distribution configs, attribute verification, single-ID grids, attribute-value tables, and augmented dataset creation.

This is an image-to-image workflow. Use the `image-edit` model path only; do not switch to Cosmos Predict or Cosmos Transfer to solve Image Attribute Augmentation image-edit requests.

## Core Rules

- Do not modify repo code unless the user explicitly asks. Runtime config copies, logs, and outputs can live under `/tmp` or the requested Image Attribute Augmentation dataset/output directory.
- Never write model keys to config files, logs, or scripts. Pass user-provided keys through environment variables named by each endpoint's `api_key_env` — e.g. `VLM_API_KEY`, `LLM_API_KEY`, `BUILD_NVIDIA_API_KEY`.
- **All inference is remote (BYOM).** The endpoint-only image bundles no local model weights. Image-edit endpoints are entries in the config's `endpoints:` list (role `image_edit`); choose the API contract with the endpoint's `adapter` field (`nim`, `openai.chat.completions`, or `openai.images.edits`). Captioning/verification use the `vlm` and `llm` role endpoints.
- **Safe Docker defaults.** Run the container as your own user (`--user "$(id -u):$(id -g)"`) and on the shared `paidf` bridge so outputs are not world-writable and host ports are not exposed. Run local model servers in containers on that bridge and use their container names in endpoint URLs. Never grant the augmentation container host-network access or make host directories world-writable on shared or multi-user machines.
- Keep checked-in Image Attribute Augmentation distribution configs as templates. Make runtime copies with concrete `data_dir`, `config_output`, `output_root`, and template paths.
- Validate generated configs and inspect expected `output.jpg` and `output_metadata.json` files. The CLI can exit successfully even when a specific sample did not produce an image.
- For remote OpenAI-compatible model endpoints, the augmentation container does not need GPU flags. A GPU is only required for the `data_processing.alignment` post-processor (cupy/CUDA).

## Deterministic Docker CLI Workflow

Build the endpoint-only augmentation image from the repo root when the user asks for a fresh local image (use legacy BuildKit per the repo's build convention):

```bash
DOCKER_BUILDKIT=0 docker build -t paidf-augmentation:1.1.0 -f docker/Dockerfile .
```

Launch the container from the repo root with the dataset and output mounted to stable `/workspace/data` paths:

```bash
HOST_DATA="/absolute/path/to/dataset_root"
HOST_OUT="/absolute/path/to/augmented_out"
mkdir -p "$HOST_OUT"
docker network inspect paidf >/dev/null 2>&1 || \
  docker network create paidf

docker run -it --rm \
  --user "$(id -u):$(id -g)" \
  --network paidf \
  -e VLM_API_KEY \
  -e LLM_API_KEY \
  -e BUILD_NVIDIA_API_KEY \
  -v "$(pwd)/modules:/workspace/modules:ro" \
  -v "$(pwd)/configs:/workspace/configs:ro" \
  -v "$HOST_DATA:/workspace/data/in:ro" \
  -v "$HOST_OUT:/workspace/data/out" \
  -w /workspace \
  --entrypoint /bin/bash \
  paidf-augmentation:1.1.0
```

> **Security & fallbacks.** Running as your own UID (`--user`) lets the container write `$HOST_OUT` without world-writable permissions, and the `:ro` mounts give read-only access to `modules/`/`configs/`, so no `chmod` is needed for them. The launch above uses the isolated `paidf` bridge. Attach local model containers to that bridge and address them by container name; host-network access is prohibited. If a non-default UID cannot use the image's prebuilt environment, drop `--user` and instead make `$HOST_OUT` writable by the container's non-root uid 10000 (e.g. `chown`), reserving world-writable permissions as a last resort on isolated single-user machines only. Prefer a secrets manager that injects the required environment variables; otherwise export only the required keys and forward their names with `-e VAR_NAME`. Never mount or load broad `.env`, SSH, cloud-credential, or token files.

Inside the container, all deterministic commands below assume `/workspace` as the working directory.

## Input Preparation

Preprocess Image Attribute Augmentation image folders into multi-panel images with:

```bash
uv run python modules/data_processing/combine_panes.py \
  /workspace/data/in \
  /workspace/data/out/panes
```

Use `combine_panes.py` because it accepts one or more images per ID and writes both `{person_key}.jpg` and `{person_key}.json` metadata needed for later splitting. The expected input layout is one subdirectory per person ID:

```text
/workspace/data/in/
├── person_0001/
│   ├── view_a.jpg
│   ├── view_b.jpg
│   └── view_c.jpg
├── person_0002/
│   ├── view_a.jpg
│   └── view_b.jpg
└── ...
```

The pane metadata sidecar records `image_order`, per-view `widths`, `original_resolutions`, `total_width`, and `height`; keep it beside the pane image so post-processing can split generated multi-pane outputs back to per-view crops.

When the user says to use only one image per ID, select a deterministic source image, usually the first sorted image in that ID folder, and document the chosen input path in the final response.

## Image Attribute Augmentation Distribution Workflow

1. Make a runtime copy of the requested distribution YAML, commonly `configs/cookbook/image-attribute-augmentation/attribute_distribution_1000_v1.yaml`.
2. Set concrete runtime paths in the copy:

```yaml
data_dir: /path/to/attribute_baseline/combined_imgs_<run_id>
config_output: /path/to/attribute_baseline/generated_configs_<run_id>
output_root: /path/to/attribute_baseline/output_<run_id>
example_augmentation_config: /path/to/image-edit template.yaml
```

3. Generate configs:

```bash
.venv/bin/python modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow /tmp/attribute_distribution_<run_id>.yaml
```

If the template named in the distribution config is missing, use the closest existing Image Attribute Augmentation image-edit config (e.g. `configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_chat_api.yaml`) as the template source, then apply the compatibility fixes below.

## Generated-Config Compatibility

Apply these fixes before running generated configs against the current CLI:

- Set `data[*].output.video` to the same path as `data[*].output.media`.
- If a legacy top-level `attribute_verification` block exists, add `evaluators: [{attribute_verification: ...}]`.
- In the evaluator copy, remove schema-forbidden legacy keys such as `retries` and `accept_if_mvc_passed`.
- Copy `captioning.verification_options` into `captioning.llm.verification_options` so verification questions get the full answer set.
- If the config still uses the old fixed-key `endpoints:` mapping, convert it to the BYOM **list** form (one entry per role with `role`/`url`/`model`/optional `adapter`). Remove any `augmentation.model.executor_type` (no longer a schema field).
- Set `pipeline.retry: 0` unless the user explicitly wants regeneration retries.
- For a VLM-verification endpoint that rejects `temperature: 0.0`, set `evaluators[].attribute_verification.vlm_verification.parameters.temperature: 1.0`.

Validate configs with `PipelineConfig` before launching a batch when the change touches schema-sensitive fields.

## Running Image Edit

Run image-edit augmentation inside `paidf-augmentation:1.1.0` (the endpoint-only image bundles the repo deps and reaches the model over a remote endpoint — no local weights). Run each generated config with `uv run modules/cli.py --config <cfg>` and log stdout/stderr per config. Do not pass API keys on the command line; export only the required keys and forward their names with Docker `-e VAR_NAME`.

Docker and UV notes:

- The image runs as a non-root user (uid 10000) that may not be able to write the host-owned repo mount. If `uv` reports permission errors under `/workspace/.venv` or `/workspace/uv.lock`, run Docker with the host UID/GID, set `HOME=/tmp`, set `UV_PROJECT_ENVIRONMENT=/tmp/augmentation-venv`, and use `uv run --frozen`.
- For NVIDIA-hosted OpenAI-compatible endpoints, configure the base URL to end at `/v1` (e.g. `https://integrate.api.nvidia.com/v1`), not the full `/chat/completions` URL — the OpenAI client appends `/chat/completions` internally.
- The CLI exits non-zero if any sample fails generation or a strict evaluator check, so exit status `0` means every sample completed. Failed samples are logged (`Sample failed: <path>`) and the run ends with a `N/M sample(s) failed` summary. Under `pipeline.evaluation.retain_failures` (default `true`) an evaluator-failed sample's output file is still written, so a file on disk is not by itself proof of success — check the exit code.

For a single smoke-test pane, use an Image Attribute Augmentation image-edit config directly (endpoints come from the config's `endpoints:` list):

```bash
mkdir -p /workspace/data/out/augmented_outputs/person_0001/aug_0

uv run modules/cli.py --config configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_chat_api.yaml \
  data.0.inputs.rgb=/workspace/data/out/panes/person_0001.jpg \
  data.0.output.video=/workspace/data/out/augmented_outputs/person_0001/aug_0/output.jpg \
  data.0.output.caption=/workspace/data/out/augmented_outputs/person_0001/aug_0/output.txt \
  data.0.output.metadata=/workspace/data/out/augmented_outputs/person_0001/aug_0/output_metadata.json
```

> To retarget an endpoint inline, override it by list index (e.g. `endpoints.0.url=http://localhost:8001/v1`); confirm which index holds the role you mean, or edit the config's `endpoints:` list directly.

For one augmentation per pane, preserve the standard output layout:

```bash
mkdir -p /workspace/data/out/augmented_outputs

for pane in /workspace/data/out/panes/*.jpg; do
  id="$(basename "${pane%.jpg}")"
  mkdir -p "/workspace/data/out/augmented_outputs/${id}/aug_0"

  uv run modules/cli.py --config configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_chat_api.yaml \
    data.0.inputs.rgb="$pane" \
    data.0.output.video="/workspace/data/out/augmented_outputs/${id}/aug_0/output.jpg" \
    data.0.output.caption="/workspace/data/out/augmented_outputs/${id}/aug_0/output.txt" \
    data.0.output.metadata="/workspace/data/out/augmented_outputs/${id}/aug_0/output_metadata.json"
done
```

To sweep a specific wardrobe value, add OmegaConf list overrides to a single call. Each call samples one combination from the lists:

```bash
mkdir -p /workspace/data/out/augmented_outputs/person_0001/aug_1

uv run modules/cli.py --config configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_chat_api.yaml \
  data.0.inputs.rgb=/workspace/data/out/panes/person_0001.jpg \
  data.0.output.video=/workspace/data/out/augmented_outputs/person_0001/aug_1/output.jpg \
  data.0.output.caption=/workspace/data/out/augmented_outputs/person_0001/aug_1/output.txt \
  data.0.output.metadata=/workspace/data/out/augmented_outputs/person_0001/aug_1/output_metadata.json \
  'captioning.llm.variables.top_outer_color=[red]' \
  'captioning.llm.variables.top_outer_type=[cropped jacket]' \
  'captioning.llm.variables.bottom_color=[blue]' \
  'captioning.llm.variables.bottom_type=[jeans]' \
  'captioning.llm.variables.shoe_color=[white]' \
  'captioning.llm.variables.shoe_type=[sneakers]'
```

`shoe_type` and `shoe_color` are inserted into the edit prompt but excluded from the default MCQ verification gate by `evaluators[0].attribute_verification.exclude_variables`.

## Output Inspection

Expected files usually have this shape:

```text
input:    <combined_dir>/<person_key>.jpg
output:   <output_root>/<person_key>/aug_<n>/output.jpg
prompt:   <output_root>/<person_key>/aug_<n>/output.txt
metadata: <output_root>/<person_key>/aug_<n>/output_metadata.json
```

Metadata contains `selections`, `attribute_verification.passed`, per-question results, and optional natural captions. A generated image can exist even when verification fails.

Use generated configs as the status source of truth:

```bash
.venv/bin/python -c '
from pathlib import Path
import yaml

config_dir = Path("/path/to/attribute_baseline/generated_configs_<run_id>")
total = done = with_image = with_metadata = metadata_without_image = 0

for cfg in sorted(config_dir.glob("*.yaml")):
    data = yaml.safe_load(cfg.read_text())
    sample = data["data"][0]
    output = Path(sample["output"]["video"])
    metadata = Path(sample["output"]["metadata"])
    image_exists = output.exists()
    metadata_exists = metadata.exists()

    total += 1
    with_image += int(image_exists)
    with_metadata += int(metadata_exists)
    metadata_without_image += int(metadata_exists and not image_exists)
    done += int(image_exists and metadata_exists)

pending = total - done
print(f"total={total} done={done} pending={pending} with_image={with_image} with_metadata={with_metadata} metadata_without_image={metadata_without_image}")
'
```

When the user asks for examples or current status, report concrete paths:

```text
input:    /path/to/attribute_baseline/combined_imgs_<run_id>/00000_RSTP.jpg
output:   /path/to/attribute_baseline/output_<run_id>/00000_RSTP/aug_0/output.jpg
metadata: /path/to/attribute_baseline/output_<run_id>/00000_RSTP/aug_0/output_metadata.json
```

## Augmented Dataset

After image-edit outputs are complete, split outputs back into an augmented Image Attribute Augmentation dataset with:

```bash
uv run --no-sync python modules/data_processing/create_attribute_augmented_dataset.py \
  --base-dir /workspace/data/out/panes \
  --augmented-folders /workspace/data/out/augmented_outputs \
  --output-dir /workspace/data/out/augmented_dataset \
  --output-json augmented_data.json
```

`--base-dir`, every value passed to `--augmented-folders`, and `--output-dir`
also accept MultiStorage paths. Configure the storage profile through the normal
MSC configuration environment, then use the same command with remote URIs:

```bash
uv run --no-sync python modules/data_processing/create_attribute_augmented_dataset.py \
  --base-dir msc://attribute-data/out/panes \
  --augmented-folders msc://attribute-data/out/augmented_outputs \
  --output-dir msc://attribute-data/out/augmented_dataset \
  --output-json augmented_data.json
```

The post-processing script does not require an original dataset JSON. It uses the pane metadata from preprocessing to split generated multi-pane images back into per-view crops, writes split images under `/workspace/data/out/augmented_dataset/augmented_imgs/<person_key>_aug<n>/`, and writes the dataset JSON to `/workspace/data/out/augmented_dataset/augmented_data.json`. The JSON includes selected attributes, generated queries, image paths, and attribute verification metadata.

The script also accepts a pane metadata JSON directly and Cosmos-style numeric run
directories directly. For example:

```bash
uv run --no-sync python modules/data_processing/create_attribute_augmented_dataset.py \
  --base-dir s3://bucket/run/preprocessing/person_0001/person_0001.json \
  --augmented-folders s3://bucket/run/cosmos/person_0001/0 \
  --output-dir /workspace/data/out/augmented_dataset
```

For discovery, `0` is treated like `aug_0`. If `output.jpg` is absent, the script
uses the only `.jpg` or `.jpeg` in the run directory. When
`output_metadata.json` is absent, images are still split, but selected attributes
and verification metadata are empty.

## Exploratory Grids And Tables

For user-facing examples, start with one ID and one source image unless the user asks for multi-view panels. Generate one attribute modification per call, reuse existing completed cells, and compose grids from the produced `output.jpg` files.

Grid and slide artifact practices:

- Keep one axis or block per attribute and label values directly above or beside the edited image.
- Group attributes visually when the user asks for a wide/horizontal slide artifact.
- Avoid an original column unless the user asks for it.
- Keep tiles close together and remove excess whitespace for slide-ready outputs.
- Save both PNG and JPG when useful for slides.
- For tables of attribute-value pairs, read the relevant Image Attribute Augmentation distribution/config YAML and output a Markdown or image table without running generation.
- For color vocabulary examples, prefer swatch/table artifacts when no generation is needed; use generated image grids only when the user asks to see edits on the person.
