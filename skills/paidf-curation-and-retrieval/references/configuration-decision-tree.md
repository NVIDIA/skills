# Configuration Decision Tree

How to configure cosmos-curator pipelines using flat YAML configs.

> **No KPI baseline available?** This decision tree assumes the
> agent already knows the dataset's resolution / camera type /
> domain / hardware / goal. If those are unknown, follow
> `calibration-config.md` FIRST -- it has a mandatory interview and
> a calibration-run template (`configs/split_calibration.yaml`) that
> produces the inputs this decision tree expects. Do not emit a
> full-run config from a one-line user request.

## Workflow Overview

```text
1. Gather requirements (dataset, domain, goal, hardware)
   -- if no KPI baseline, this is the calibration-config.md interview --
2. Choose pipeline and features (decision tree below)
3. Start from reference config or cookbook
4. Customize keys for your scenario
5. Validate and run
```

## Step 1: Gather Requirements

Ask these questions (skip any already answered). If the user has
provided no KPI samples and no prior pipeline output, this list is
the bare minimum -- expand it using `calibration-config.md` Phase 1
(Inputs A1-A5, Domain B1-B2, Goal C1-C3, Hardware D1-D3,
Calibration E1) before emitting anything.

1. **Dataset**: Where are the videos? Format/resolution/FPS? S3 or local?
2. **Domain**: Traffic safety? Warehouse? Construction? General video? Other?
3. **Goal**: Full captioning + filtering? Quick split only? Dedup? Training shards?
4. **Hardware**: GPU type/count (L40S, H100, A100)? Disk space?
5. **Scale**: How many videos? Total duration?
6. **Output**: Embeddings? Previews? Cosmos-Predict2 dataset?

## Step 2: Decision Tree

### Modality (image vs. video)

```text
Are the inputs still images (jpg / png / webp)?
  YES -> Use the image annotate pipeline.
         pipeline: "annotate"  in  configs/image.yaml
         Run: make run_image_pipeline IMAGE_CONFIG_FILE=configs/image.yaml
         See references/image-curation.md for stages, output layout,
         and algorithm choices. Skip the rest of this tree -- the
         video-only sections (splitting algorithm, motion filter,
         super-resolution, sharding, multi-cam) do not apply.
  NO  -> Continue with the video pipeline decision tree below
         (split / dedup / shard).
```

### Splitting Algorithm

```text
Is the footage from fixed-camera/dashcam with continuous recording?
  YES -> splitting_algorithm: "fixed-stride"
         fixed_stride_split_duration: 10-60  (scene-appropriate)
         fixed_stride_min_clip_length_s: 2-5
  NO  -> splitting_algorithm: "transnetv2"
         transnetv2_threshold: 0.4  (lower = more sensitive)
         transnetv2_min_length_s: 2.0
         transnetv2_max_length_s: 60.0
```

### Captioning Algorithm

```text
Do you need the best quality captions?
  YES -> Do you have 1+ GPU with ~20+ GB VRAM?
           YES -> captioning_algorithm: "qwen3_vl_30b_fp8" or "qwen3_5_27b"
           NO  -> captioning_algorithm: "qwen" or "nemotron"  (lighter local models)
  Need non-quantized Qwen3-VL?
           Use captioning_algorithm: "qwen3_vl_30b" only with ~35+ GB VRAM
  NO  -> Do you want API-based captioning?
           YES -> captioning_algorithm: "gemini"  (or "openai")
           NO  -> captioning_algorithm: "qwen"  (smaller, faster)

Need multi-GPU scaling?
  YES -> captioning_algorithm: "vllm_async"  (auto-configured in v2.0.0)
         qwen_num_gpus_per_worker: 2  (tensor parallelism)
         vllm_performance_mode: "throughput"
         # NOTE: explicit vllm_async_* engine knobs were removed in v2.0.0
```

### Captioning Prompt

```text
Is the domain traffic/fixed-camera video?
  YES -> captioning_prompt_variant: "default"
         enhance_captions_prompt_variant: "default"
Is it autonomous driving?
  YES -> captioning_prompt_variant: "av"
         enhance_captions_prompt_variant: "av"
General video?
  YES -> captioning_prompt_variant: "default"
         enhance_captions_prompt_variant: "default"
Custom domain?
  YES -> captioning_prompt_text: "<your full prompt string>"
```

### Filter Selection

```text
Want to remove static/near-static clips?
  YES -> motion_filter: "enable"
         motion_global_mean_threshold: <calibrated percentile>
         (derive from motion_score.global_mean; lower = more aggressive)

Want to remove low-quality/blurry clips?
  YES -> aesthetic_threshold: 4.5  (higher = stricter)
         aesthetic_reduction: "min"  (strictest) or "mean"

Want to filter by semantic content?
  YES -> vlm_filter: "enable"
         vlm_filter_rejection_threshold: 0.5

Want to classify clips by category?
  YES -> video_classifier: true
         video_classifier_use_custom_categories: true  (if custom lists)
         video_classifier_allow: ["category1", "category2"]

Want to remove clips with overlaid text/watermarks?
  YES -> artificial_text_filter: "enable"
```

### Embedding Algorithm

```text
Need embeddings for semantic dedup?
  YES -> generate_embeddings: true
         Which algorithm?
           Default/proven     -> embedding_algorithm: "internvideo2"
           Higher quality      -> embedding_algorithm: "cosmos-embed1-448p"
           API-based          -> embedding_algorithm: "openai"
  NO  -> generate_embeddings: false  (saves GPU memory)
```

### Super-Resolution

```text
Are source videos low-resolution (< 720p)?
  YES -> super_resolution: true
         sr_variant: "seedvr2_7b"  (best quality)
         sr_target_height: 720
         sr_target_width: 1280
  NO  -> super_resolution: false  (default)
```

## Step 3: Hardware Profiles

### Single L40S (48 GB)

```yaml
pipeline: "split"
captioning_algorithm: "qwen3_vl_30b_fp8"
generate_embeddings: false
```

Run with: `--gpus '"device=0"'`

### Dual L40S (2 x 48 GB)

```yaml
pipeline: "split"
captioning_algorithm: "qwen3_vl_30b_fp8"
generate_embeddings: true
embedding_algorithm: "internvideo2"
```

Run with: `--gpus '"device=0,1"'`

### H100 (80 GB)

```yaml
pipeline: "split"
captioning_algorithm: "qwen3_vl_30b_fp8"
generate_embeddings: true
super_resolution: true
```

Run with: `--gpus all`

### Multi-GPU (4+ GPUs)

```yaml
pipeline: "split"
captioning_algorithm: "vllm_async"   # auto-configured in v2.0.0
qwen_num_gpus_per_worker: 2          # tensor parallelism
vllm_performance_mode: "throughput"
generate_embeddings: true
```

### Quick Test (any GPU)

```yaml
pipeline: "split"
input_video_path: "/data/test"
output_clip_path: "/data/test-output"
model_weights_path: "/config/models"
limit: 3
dry_run: false
generate_embeddings: false
captioning_algorithm: "qwen3_vl_30b_fp8"
```

## Step 4: Config Key Reference

### Required Keys (split)

| Key | Description |
|-----|-------------|
| `pipeline` | Must be `"split"` |
| `input_video_path` | S3 or local path to input videos |
| `output_clip_path` | S3 or local path for output |
| `model_weights_path` | Path to downloaded model weights |

### Required Keys (dedup)

| Key | Description |
|-----|-------------|
| `pipeline` | Must be `"dedup"` |
| `input_embeddings_path` | Path to split output with embeddings |
| `output_path` | Path for dedup output |

### Required Keys (shard)

| Key | Description |
|-----|-------------|
| `pipeline` | Must be `"shard"` |
| `input_clip_path` | Path to split output with clips |
| `output_dataset_path` | Path for WebDataset tar archives |
| `captioning_algorithm` | Must match algorithm used in split |

### Feature Toggle Keys

| Key | Default | Effect |
|-----|---------|--------|
| `generate_captions` | `true` | VLM captioning stage |
| `generate_embeddings` | `true` | Embedding generation |
| `enhance_captions` | `false` upstream, `true` in some repo configs | LLM caption refinement / MCQ |
| `upload_clips` | `true` | Write transcoded clip files |
| `generate_previews` | `false` | Web preview thumbnails |
| `dry_run` | `false` | Metadata only, no clip output |
| `multi_cam` | `false` | Multi-camera synchronization |

For the complete key inventory (~200 split, ~20 dedup, ~25 shard), see
`configs/split.yaml`, `configs/dedup.yaml`, `configs/shard.yaml`. First-run
operator recipes are `cookbook/traffic-video-analytics/` and
`cookbook/warehouse-safety/`.

## Step 5: Validation Checklist

Before running:

- [ ] `pipeline` key is set and valid (`split`, `dedup`, or `shard`)
- [ ] Required I/O paths are set and accessible
- [ ] `model_weights_path` points to downloaded weights
- [ ] `captioning_algorithm` in shard config matches the split run
- [ ] `SHM_SIZE`/`--shm-size` fits host RAM (default 24gb; 24-64gb for typical multi-GPU)
- [ ] Docker `--gpus` matches intended GPU allocation
- [ ] For custom classification: `video_classifier_use_custom_categories: true`
- [ ] Test with `limit: 3` and `dry_run: true` before full run

## Quick Start

Copy and customize the closest cookbook YAML:

```bash
# Start from cookbook
cp cookbook/traffic-video-analytics/split.yaml my-split.yaml

# Edit I/O paths
# Edit: input_video_path, output_clip_path, model_weights_path

# Run
make run-pipeline CONFIG_FILE=my-split.yaml
```

Each cookbook scenario also ships an `input_config.json` with the same
overrides in compact JSON form. It is an agent-emitted manifest for
provenance and downstream tooling -- the cosmos-curator pipeline reads only
the YAML. When you author a new scenario, write both files so they stay in
lock-step.

The `input_config.json` format:

```json
{
  "cosmos_curator": {
    "pipeline": "split",
    "image": "cosmos-curator:2.3.0",
    "dataset_description": "...",
    "overrides": {
      "input_video_path": "/data/input",
      "splitting_algorithm": "fixed-stride",
      ...
    }
  },
  "video_stats": { ... }
}
```

Only `cosmos_curator.overrides` mirrors the YAML. Everything else
(`image`, `dataset_description`, `video_stats`) is metadata for
documentation and provenance tracking.

---

# Prompt Generation

## Prompt Variants

Cosmos-curate includes built-in prompt variants set via
`captioning_prompt_variant` and `enhance_captions_prompt_variant`
(`default`, `av`); for custom domains use
`captioning_prompt_text`. See `references/video-curation.md` §Video-Specific
Prompt Variants for the full variant table.

## The 8-Element Framework

Every production custom prompt follows this structure:

### 1. Role and Persona

Define the expert role. Be specific: not "video analyst" but "traffic safety
analyst specializing in collision detection and near-miss identification."

### 2. Event/Category Definitions with Visual Signatures

Define every event with concrete visual indicators (3+ per event), distinguishing
features, and confidence levels (HIGH/MEDIUM/LOW). Use 5-15 categories.
Source from `.agents/references/catalogs/{domain}.yaml`.

### 3. Default-Deny Baseline

Define "normal" explicitly (5-10 conditions). Include common false-positive
scenarios. Rule: when in doubt, classify as NORMAL.

### 4. Investigation Protocol

5-7 ordered steps: scene assessment, actor identification, event detection,
temporal analysis, classification. General context first, then specific.

### 5. Failure Modes and Edge Cases

10+ scenarios covering both false positives and false negatives. Each with:
what it looks like, why it is/isn't an event, key differentiator.

### 6. Visual-Grounding Constraints

Task-specific constraints preventing the downstream VLM from inferring beyond
visible evidence. They are not agent platform or hidden instructions.
"Only describe what you can see in the video frames."

### 7. Decision Checklist

Ordered binary questions the VLM must answer before classifying.

### 8. Output Format

Dense, factual, chronological narration. Include:
- Temporal ordering of events
- All visible actors and objects
- Spatial relationships and movements
- Event classification using an explicit allow-list such as
  `[event1, event2, ...]`

## Custom Prompt Integration

Set the custom prompt in your config YAML:

```yaml
captioning_prompt_text: |
  You are a warehouse safety analyst specializing in...
  [Full 8-element prompt here]
```

Or use the `--captioning-prompt-text` CLI flag when running in CLI mode.

## Domain Catalogs

Pre-built event taxonomies at `.agents/references/catalogs/` provide per-domain
`baseline_event`, `keywords`, severity-tiered `events`, `remap` aliases, and
`excluded` labels. See `references/context-understanding.md` §Domain Catalogs for
the authoritative field-by-field reference.
