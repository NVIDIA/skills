# Config Decision Tree

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../../configs/cookbook/README.md) for the folder layout.

## Step 1: Determine the Pipeline from Input Type

**Ask the user what their input is.** The input type determines which model to use:

```text
What is your input?
│
├── Video file (.mp4, .avi, etc.)
│   └── Do you want to change scene attributes (weather, lighting, colors, clothing)?
│       ├── Yes → Cosmos Transfer 2.5 (video → augmented video)
│       └── No, I want to predict/continue/extend the video
│           └── Cosmos Predict 2.5 (video + text → new video)
│
├── Image file (.png, .jpg, etc.)
│   ├── Edit the image (change attributes)?      → image edit (image → edited image)
│   ├── Animate it into a video?                 → image-to-video (Cosmos3 or Veo 3.1)
│   └── Generate a video from it + a text prompt → Cosmos Predict 2.5 (image + text → video)
│
├── Text only (no media input)
│   └── Cosmos Predict 2.5 (text → video, inference_type: text2world)
│
└── Not sure / multiple inputs
    └── Ask the user to clarify their input type and goal
```

## Step 2: Choose a Config

### Cosmos Transfer 2.5 (video → video)

```text
└── Scene-attribute transfer + hallucination/attribute verification → config_video_transfer_CT25_nim.yaml
```

### Cosmos Predict 2.5 (text/image/video → video)

No dedicated shipped config yet — create one and set `augmentation.model.name: cosmos-predict`, add a `video_predict`-role endpoint, and set `augmentation.parameters.inference_type` to `text2world` / `image2world` / `video2world`.

### image edit (image → image)

```text
├── Image Attribute Augmentation, chat.completions contract → config_image_edit_attribute_chat_api.yaml
├── Image Attribute Augmentation, images/edits contract     → config_image_edit_attribute_images_api.yaml
├── Image Attribute Augmentation, NIM /v1/infer contract    → config_image_edit_attribute_nim.yaml
├── Image Attribute Augmentation, hosted Gemma 4 llm role   → config_image_edit_attribute_gemma_llm.yaml
└── Defect Image Generation (align output back to input)
    ├── chat.completions + alignment                        → config_image_edit_defect_chat_api.yaml
    └── images/edits + resize + alignment                   → config_image_edit_defect_images_api.yaml
```

### image-to-video (first-frame image → video)

```text
├── Self-hosted Cosmos3 (sync /v1/videos/sync)       → config_image2video_cosmos3.yaml
└── Hosted Veo 3.1 (async create→poll→download)      → config_image2video_veo31.yaml
```

Both use a VLM→LLM caption chain (Qwen3-VL describes the seed → Qwen2.5 writes the motion prompt) plus `attribute_verification`. Cosmos3 uses the `openai.video.sync` adapter; Veo sets `adapter: openai.video.async` and needs a key via `api_key_env: VEO_API_KEY`.

For fixed-camera event video generation with Cosmos 3 Super, use `config_event_video_gen_cosmos3_smart_spaces.yaml`. It adds anomaly-specific prompting and sampled-frame VLM verification while using the `openai.video.sync` adapter.

### Batch Config Generation

```text
├── Generate N configs          → workflow_example.yaml  (Image Attribute Augmentation: attribute_distribution_1000_v1.yaml)
├── Generate N seed images      → distribution_seed_image_gen_cosmos3_super_t2i_smart_spaces.yaml
└── Generate N event videos     → distribution_event_video_gen_smart_spaces.yaml
```

### By Captioning Strategy

```text
What model are you using?
│
├── Cosmos Transfer / Predict / image-to-video (video output)
│   └── Default: VLM+LLM (VLM describes scene → LLM generates prompt with variables)
│       Config: config_video_transfer_CT25_nim.yaml, config_image2video_cosmos3.yaml
│
├── image edit (image editing)
│   └── Default: LLM-only (LLM generates editing instruction from variables)
│       Config: config_image_edit_attribute_chat_api.yaml
│
└── Fallbacks (any model):
    ├── Fixed text with variable substitution (no endpoints needed)
    │   └── Use captioning.llm.text: "change {color} to {value}"
    ├── Sample from a prompt file (no endpoints needed)
    │   └── Use captioning.llm.file_path: "/path/to/prompts.txt"
    └── VLM-only description (no LLM needed)
        └── Set captioning.vlm only (no captioning.llm)
```

## Endpoint Requirements by Configuration

All inference is remote — there is no `executor_type`. The generation model needs **one endpoint whose role matches the model**; its adapter defaults from the role (override with the endpoint's `adapter` field).

| `model.name` | Role | Default adapter | Endpoint needed |
|--------------|------|-----------------|-----------------|
| cosmos-transfer2.5 | video_transfer | nim | one `video_transfer` endpoint |
| cosmos-predict | video_predict | nim | one `video_predict` endpoint |
| image-edit | image_edit | nim | one `image_edit` endpoint (adapter `nim`, `openai.chat.completions`, or `openai.images.edits`) |
| cosmos3-image2video | image2video | openai.video.sync | one `image2video` endpoint |
| (Veo, any id) | image2video | — (set `openai.video.async`) | one `image2video` endpoint with `adapter: openai.video.async` + `api_key_env` |

Additionally, captioning and verification need:
- VLM captioning → a `vlm`-role endpoint
- LLM captioning → an `llm`-role endpoint (except text/file captioners, which need none)
- Attribute verification → a `vlm` endpoint (answering) + an `llm` endpoint (question generation)

If two endpoints share a role, disambiguate by giving them `id`s and pointing the consumer at one (`augmentation.model.name: <id>`, or `question_generation.endpoint_id` / `vlm_verification.endpoint_id`).

## Switching Models Without Changing Configs

Use OmegaConf CLI overrides. Endpoints are a list, so address entries by index:

```bash
# Point the image_edit endpoint at a different server / contract
uv run modules/cli.py --config configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_nim.yaml \
  augmentation.model.name=image-edit \
  endpoints.0.url=http://localhost:8001/v1 \
  endpoints.0.adapter=openai.chat.completions

# Re-target the video_transfer endpoint. In config_video_transfer_CT25_nim.yaml the
# list order is vlm(0), llm(1), video_transfer(2) — so it is index 2, NOT 0.
uv run modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml \
  endpoints.2.url=http://remote-server:8000
```

Index order is config-specific and fragile — **always confirm which entry holds the role you mean** (or just edit the YAML `endpoints:` list directly). Overriding the wrong index silently retargets a different service.

## Disabling Evaluators Inline

```bash
# Disable hallucination check but keep attribute verification
uv run modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml \
  evaluators.0.hallucination_check.enabled=false

# Disable all evaluators (quick generation test)
uv run modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml \
  evaluators=null
```

## Multi-Sample Batch Processing

### Inline (up to ~5 samples)

```bash
uv run modules/cli.py --config configs/cookbook/image-attribute-augmentation/config_image_edit_attribute_chat_api.yaml \
  data.0.inputs.rgb=/data/img1.png \
  data.0.output.video=/output/img1.jpg \
  data.0.output.caption=/output/img1.txt \
  data.0.output.metadata=/output/img1.json \
  data.1.inputs.rgb=/data/img2.png \
  data.1.output.video=/output/img2.jpg \
  data.1.output.caption=/output/img2.txt \
  data.1.output.metadata=/output/img2.json
```

### Batch Config Generation (10+ samples)

```bash
uv run modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow configs/cookbook/video-data-augmentation/workflow_example.yaml
```

The workflow config (`workflow_example.yaml`) specifies:
- `data_dir` — directory of input media files
- `n_augmentations` — number of augmented configs to generate per input
- `config_output` — where to write generated configs
- `example_augmentation_config` — base config template
- `variables` — independent probability distributions for sampled attributes
- `conditional_variables` — dependent attribute distributions keyed by a parent variable value

Use `conditional_variables` when attributes are not independent, for example:
- `road_condition` depends on `weather`
- `shoe_color` depends on `shoe_type`

## Control Modality Configuration (Cosmos Transfer Only)

Control modalities define structural guidance from input videos. Higher weights = more structural preservation from the control input; a weight of 0 (or omitting the key) means the modality is not used.

| Modality | Purpose / recommended use | Weight guidance |
|----------|---------------------------|-----------------|
| `edge` | Edge detection map — preserves structural outlines; the base structure signal in multi-control setups | 0.4–1.0 (use as the foundation) |
| `depth` | Depth estimation map — preserves spatial layout | 0.3–0.4 (moderate) |
| `seg` | Segmentation map — preserves object/semantic boundaries; pair with masks and `seg_control_prompt` | 0.2–0.4 (moderate); never use alone |
| `vis` | Visualization/appearance — preserves visual style, supplements edge/seg | 0.05–0.6 (low); avoid very high values |

### Weight Normalization Behavior

Control weights are **not** individually capped. The pipeline forwards each weight to the backend unchanged (as `{control_weight: <value>}`); the **Cosmos Transfer backend** is what normalizes, based on the **sum** of all active weights:

- **Sum ≤ 1.0**: Weights are applied **as-is** with no normalization. E.g., `{seg: 0.2, edge: 0.2}` stays unchanged.
- **Sum > 1.0**: Weights are **normalized proportionally** so the total equals 1.0. E.g., `{seg: 4.0, edge: 1.0}` (sum 5.0) becomes `{seg: 0.8, edge: 0.2}`.

**Best practices** (see [Cosmos Cookbook — Control Modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html)):
- Use multi-control combinations (e.g., edge + seg) for best results.
- Start with lower `vis` weights and increase as needed.
- Do not use `seg` alone — always pair with another modality.
- If the output doesn't incorporate prompt-described changes, increase `augmentation.parameters.guidance` (start at 3, boost to 5+ if needed).

### Control Inputs in Data Section

Control input files must match modalities configured in `augmentation.modalities`:

```yaml
data:
  - inputs:
      rgb: "/path/to/source.mp4"
      controls:
        edge: "/path/to/edge_map.mp4"     # Only needed if modalities.edge is set
        depth: "/path/to/depth_map.mp4"
    output:
      video: "/path/to/output.mp4"
      caption: "/path/to/prompt.txt"
      metadata: "/path/to/metadata.json"

augmentation:
  modalities:
    edge: 0.8
    depth: 0.4
    positive_prompt: "cinematic, photorealistic..."
    negative_prompt: "cartoon, low quality..."
```

If `controls` are set to `null` (e.g., `edge: null`), the Cosmos model extracts control signals automatically from the RGB input.

## Post-Processing: Data-Processing Alignment (image edit only)

`data_processing` is a top-level section (peer of `augmentation` and `evaluators`) for post-processors that mutate the augmentation output in place. Each sub-key is a separate processor; presence enables it.

`alignment` runs an MI-registration to warp+crop a generated image back into the reference frame of the input — useful when the model outputs at a different resolution (e.g. Qwen Image Edit upscales 158×114 → 1216×864). The aligned image overwrites `data.output.video`; recovered transform parameters land in `data.output.metadata` under `alignment`.

```yaml
data_processing:
  alignment:
    rot_range_deg: [-1.0, 1.0, 0.1]    # rotation search [start, stop, step] in degrees
    shift_step:    1                   # tx/ty grid step (px)
    bins:          64                  # MI histogram bins
    interp:        bilinear            # nearest | bilinear warp kernel
    no_resize:     true                # skip pre-resize of align→ref
    min_mi:        null                # null = no MI floor; else abort threshold
```

**Auto-derived parameters — do NOT set these in YAML.** Computed at runtime from actual image dimensions; the schema exposes them only as escape-hatch overrides for visibly-failing runs.

| Field | Auto rule |
|-------|-----------|
| `sx_range` | `[lo, hi, 0.1]` with `lo = ratio*0.9`, `hi = ratio*1.1`, `ratio = gen_w / ref_w` |
| `sy_range` | `[lo, hi, 0.1]` with `lo = ratio*0.9`, `hi = ratio*1.1`, `ratio = gen_h / ref_h` |
| `shift_range` | `max(ref_h, ref_w) // 4` |
| `pyr_levels` | `min(3, max_usable)` keeping smallest level ≥ 32 px |

**Agent guidance:** leave the four fields above unset when authoring an alignment config; the auto-derived value is more likely correct than any hand-set one. Override only when alignment is *visibly failing* on a real run (e.g. `mi_after` ≈ `mi_before`, scale recovery saturating at a range bound, or the aligned image looks shifted/scaled) — then copy the auto-logged values and adjust the bounds. To match an image-edit server's internal normalization for clean alignment, pair with `data_processing.preprocessing.resize` (`target_megapixels: 1.048576`, `grid: 32`).

**Requirements:** runs on GPU only (cupy + CUDA), in-container.
