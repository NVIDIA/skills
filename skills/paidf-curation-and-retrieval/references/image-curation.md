# Image Curation Pipeline

End-to-end reference for cosmos-curator's `annotate` (image) pipeline,
the image counterpart of the video split/dedup/shard pipelines.

> **Quick start:**
> ```bash
> make run_image_pipeline IMAGE_CONFIG_FILE=configs/image.yaml \
>     MODELS_DIR=/path/to/models DATA_DIR=/path/to/images
> ```

---

## When to Use

Use the image annotate pipeline when:

- Inputs are **still images** (jpg / png / webp) on local disk or S3,
  not videos. One task corresponds to one image and one image yields at
  most one output asset (no clip splitting).
- The desired outputs are some combination of: filtered images, image
  embeddings, and per-image VLM captions.
- You want a single pass that loads, filters, embeds, captions, and
  writes summary metadata.

Do **not** use it for video frames; use the video split pipeline
(`configs/split.yaml`) and consume `windows[]` captions instead.

---

## Pipeline Stages

The annotate pipeline is `load → (filter) → (embed) → (caption) → write`:

| Stage              | Purpose                                                                 | Toggle / Knob                                 |
|--------------------|-------------------------------------------------------------------------|-----------------------------------------------|
| Image load         | Discover and read images from the input path.                           | `num_ingest_workers_per_node`                 |
| Semantic filter    | VLM prompt-based reject/accept decision (optional).                     | `semantic_filter: enable`                     |
| Image classifier   | VLM classifier with allow/block taxonomy (optional).                    | `image_classifier: enable`                    |
| Embedding          | One vector per image: CLIP / Cosmos-Embed1 / InternVideo2 / OpenAI.     | `generate_embeddings`, `embedding_algorithm`  |
| Caption prep       | Decode + resize within pixel bounds, build model input.                 | `caption_prep_min_pixels`, `caption_prep_max_pixels` |
| Caption            | One caption per image: local vLLM / OpenAI-compatible / Gemini.         | `generate_captions`, `captioning_algorithm`   |
| Image writer       | Write images, embeddings, metadata, and `summary.json`.                 | `num_output_workers_per_node`                 |

Filtering, embedding, and captioning are independently toggleable. The
load and write stages always run.

---

## Output Layout

```text
{output_path}/
├── images/                           # passed-filter images
│   └── {output_id}.jpg
├── filtered_images/                  # rejected by semantic / classifier filtering
│   └── {output_id}.jpg
├── embeddings/
│   ├── clip/
│   ├── cosmos_embed1_<variant>/
│   ├── internvideo2/
│   └── openai/
│       └── {output_id}.npy
├── metas/
│   └── {output_id}.json              # per-image metadata
└── summary.json                      # aggregate run summary
```

### Per-image `metas/{output_id}.json`

| Field                              | Notes                                                                 |
|------------------------------------|-----------------------------------------------------------------------|
| `source_path`, `relative_path`     | Original input location.                                              |
| `width`, `height`                  | Post-prep dimensions when prep ran, else original.                    |
| `has_caption`, `is_filtered`       | Booleans; both can be false for embed-only runs.                      |
| `caption`                          | Present only when `has_caption` is true.                              |
| `caption_status`                   | `success`, `truncated`, `error`, or `null` (when captioning skipped). |
| `caption_failure_reason`           | Populated when `caption_status == "error"`.                           |
| `token_counts`                     | Per-model token usage map.                                            |
| `qwen_type_classification`         | Classifier labels (when classifier is enabled).                       |
| `qwen_rejection_stage` / `_reasons`| Which stage rejected the image and why.                               |
| `embedding_keys`                   | List of embedding backends that wrote a vector for this image.        |
| `errors`                           | Per-stage error payloads when something failed but the task survived. |

### `summary.json`

Aggregate counters: `num_input_images`, `num_output_tasks`,
`num_images_passed`, `num_images_filtered`, `num_images_with_caption`,
`num_images_with_embeddings`, plus `embedding_backend`,
`resize_min_pixels`, `resize_max_pixels`, and the
`images` / `filtered_images` / `captioned_images` ID lists.

### Resume Behavior

The pipeline skips already-processed images on rerun by reading
`summary.json` (preferred) or `metas/*.json` (fallback) at input
extraction time. Reusing the same `output_path` for incremental runs is
the supported pattern — there is no `--resume` flag.

---

## Invocation Modes

### 1. Makefile (recommended)

```bash
make run_image_pipeline IMAGE_CONFIG_FILE=configs/image.yaml \
    MODELS_DIR=/path/to/models DATA_DIR=/path/to/images
```

The `run_image_pipeline` target mounts `IMAGE_CONFIG_FILE` as
`/config/pipeline_config.yaml` and runs upstream
`python -m cosmos_curator.pipelines.image.run_pipeline`.

### 2. Docker directly

Do not hand-roll `docker run`. Use `make run_image_pipeline` with
`IMAGE_CONFIG_FILE`, `DATA_DIR`, and `DOCKER_NETWORK=bridge`. Keep Docker
network-namespace isolation. For a host-local endpoint, prefer
`EXTRA_DOCKER_ARGS` only as documented on the Makefile, and do not disable
isolation unless a human approves a documented trusted-internal exception.

### 3. Native upstream CLI (debugging)

Like the video CLI-debugging mode in `references/running-pipelines.md`, but with
the image `annotate` subcommand and its (~70) flags instead of a YAML:

```bash
pixi run -e default --as-is python -m cosmos_curator.pipelines.image.run_pipeline \
    annotate --input-image-path /data/images --output-path /data/out \
    --captioning-algorithm qwen --embedding-algorithm internvideo2
```

> Prefer config-file mode (`make run_image_pipeline`) for operator runs.
> Use the `annotate` subcommand only for debugging individual CLI flags.

---

## Algorithm Choices

### Captioning (`captioning_algorithm`)

| Algorithm           | Backend             | VRAM   | Notes                                  |
|---------------------|---------------------|--------|----------------------------------------|
| `qwen` (default)    | Local vLLM (Qwen2-VL-7B) | ~15 GB | Lightest local option.                  |
| `qwen3_5_27b`       | Local vLLM (Qwen3.5-27B) | ~20 GB | FP8.                                    |
| `qwen3_vl_30b`      | Local vLLM (Qwen3-VL-30B) | ~35 GB | FP16, best local quality.               |
| `qwen3_vl_30b_fp8`  | Local vLLM (Qwen3-VL-30B) | ~20 GB | FP8 quantized; recommended on a single H100/A100. |
| `qwen3_vl_235b{,_fp8}` | Local vLLM      | Multi-GPU | Largest local model.                |
| `nemotron`, `cosmos_r1`, `cosmos_r2` | Local vLLM (NVIDIA models) | varies | NVIDIA-specific captioners. |
| `gemini`            | Google Gemini API   | 0      | Requires API key; uses `gemini_caption_*`. |
| `openai`            | OpenAI-compatible API | 0    | Uses `openai_caption_*`; supports `openai_caption_raw_image`. |

Set `caption_prompt_text` to override the default `image` prompt with a
custom string. For domain-specific runs, write the prompt to a file and
inline it into the YAML.

### Embedding (`embedding_algorithm`)

| Algorithm           | Backend             | Vector dim | Notes                            |
|---------------------|---------------------|-----------|----------------------------------|
| `internvideo2` (default) | InternVideo2 ViT | 768      | Default; uses GPU.               |
| `clip`              | OpenAI CLIP ViT-L/14 | 768      | Standard CLIP embeddings.         |
| `cosmos-embed1-224p`| Cosmos-Embed1 224p  | 256       | Fastest Cosmos-Embed1 variant.    |
| `cosmos-embed1-336p`| Cosmos-Embed1 336p  | 768       | Higher quality, slower.           |
| `cosmos-embed1-448p`| Cosmos-Embed1 448p  | 768       | Highest quality, slowest.         |
| `openai`            | OpenAI-compatible embedding API | varies | Uses `openai_embedding_*`. |

Set `generate_embeddings: false` to skip the embedding stage entirely.

### Filtering Modes

- **Semantic filter** (`semantic_filter: enable`) — VLM is asked
  whether the image matches one or more rejection categories from
  `semantic_filter_categories` (or the default taxonomy). An image is
  rejected when the matched fraction exceeds
  `semantic_filter_rejection_threshold`. Set
  `semantic_filter_score_only: true` to annotate without rejecting.
- **Image classifier** (`image_classifier: enable`) — VLM is asked to
  classify the image; rejection is driven by `image_classifier_block`
  (or `image_classifier_block_file`) intersection. Set
  `image_classifier_use_custom_categories: true` when supplying a fully
  custom allow/block taxonomy rather than augmenting the default.

Both filter modes share the same backend variants
(`semantic_filter_model_variant`, `image_classifier_model_variant`)
as `captioning_algorithm`, plus their own
OpenAI / Gemini knobs.

---

## Configuration Decision Sketch

```text
Need still-image curation?
  YES -> use configs/image.yaml + make run_image_pipeline
  NO  -> use the video pipelines (configs/split.yaml etc.)

Need only embeddings?
  generate_captions: false
  generate_embeddings: true
  embedding_algorithm: <pick one>

Need only captions?
  generate_embeddings: false
  generate_captions: true
  captioning_algorithm: <pick one>

Need to filter junk before captioning?
  semantic_filter: enable          # VLM yes/no on rejection prompts
  semantic_filter_categories: "blurry,duplicate,nsfw"
  semantic_filter_rejection_threshold: 0.5

Need to bucket by image type and drop blocked classes?
  image_classifier: enable
  image_classifier_allow: ["car","person"]
  image_classifier_block: ["text","screenshot"]
```

---

## Known Gotchas

- Upstream image annotate supports config-file mode; prefer
  `make run_image_pipeline`. Bypass path:
  `python -m cosmos_curator.pipelines.image.run_pipeline /config/pipeline_config.yaml`.
- `caption_prep_min_pixels` / `caption_prep_max_pixels` default to the
  video-style values (`128*28*28`, `768*28*28`); leaving them as `null`
  in the YAML preserves those defaults via `fill_default_args`.
- Gemini and OpenAI backends require credentials in
  `~/.config/cosmos_curator/cosmos_curator.yaml` (not `HF_TOKEN`).
- The annotate pipeline is **resume-aware**: pointing two runs at the
  same `output_path` will skip images already represented under
  `summary.json` or `metas/`.
- Embeddings are written under `embeddings/<backend>/{output_id}.npy`.
  Multiple embedding backends per image are not supported in a single
  run; rerun with a different `embedding_algorithm` to add more.

---

## Cross-References

- `references/cosmos-curator.md` — framework architecture (Cosmos-Xenna
  / Ray, stage lifecycle, vLLM plugin catalog, shared captioning /
  embedding algorithm enum, Pixi environments, stage replay).
- `references/video-curation.md` — counterpart guide for the video
  split / dedup / shard pipelines.
- `references/configuration-decision-tree.md` — when to choose image
  curation vs. the video pipelines.
- `references/running-pipelines.md` — invocation patterns, SHM sizing, common Docker
  flags shared by all pipelines.
- `configs/image.yaml` — full reference YAML with every annotate flag
  documented inline.
- Upstream reference: `cosmos-curator/docs/curator/reference/IMAGE_PIPELINE.md`.
