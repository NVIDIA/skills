# Video Curation Pipelines

End-to-end reference for cosmos-curator's three **video** pipelines:
`split`, `dedup`, and `shard`. For framework-wide concepts (Pixi
environments, Ray/Xenna, vLLM plugins, captioning/embedding algorithm
enums, stage replay) see `references/cosmos-curator.md`. For still-image
curation see `references/image-curation.md`.

> **Quick start (Path A):** stage NGC sample clips into `cookbook/*/videos/`
> first (`docs/user-guide/samples-and-cookbooks.md`). Then:
> ```bash
> make run-pipeline \
>   CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml \
>   MODELS_DIR=/path/to/models \
>   DATA_DIR=$PWD/cookbook/traffic-video-analytics
> make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml \
>   MODELS_DIR=/path/to/models DATA_DIR=$PWD/cookbook/traffic-video-analytics
> make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/dedup.yaml \
>   MODELS_DIR=/path/to/models DATA_DIR=$PWD/cookbook/traffic-video-analytics
> make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/shard.yaml \
>   MODELS_DIR=/path/to/models DATA_DIR=$PWD/cookbook/traffic-video-analytics
> ```
> `configs/split.yaml` / `dedup.yaml` / `shard.yaml` remain the full flag
> reference and the Makefile default.

---

## Pipelines

### 1. Split-Annotate (`split`)

The main pipeline. Splits videos into clips, generates captions,
embeddings, and metadata.

**Logical stages:**

| Stage | What it does | Resource |
|-------|-------------|----------|
| Video Download | Read from disk or S3 into memory | CPU |
| Decode + Split | Decode frames, TransNetV2 or fixed-stride splitting | GPU (fractional) |
| Transcode | Encode each clip as H.264 MP4 | CPU or GPU (h264_nvenc) |
| Super-Resolution | Upscale clips via SeedVR2 (optional) | GPU |
| Motion Filter | Score/filter clips by optical flow magnitude | GPU (fractional) |
| Aesthetic Filter | Score/filter clips by CLIP aesthetic model | GPU (fractional) |
| Artificial Text Filter | Detect and filter overlaid text/watermarks | GPU (fractional) |
| VLM Filter | VLM-based content quality filter | GPU |
| Video Classifier | VLM-based category classification | GPU |
| Frame Extraction | Extract frames for embedding | CPU |
| Embedding | InternVideo2, Cosmos-Embed1, or OpenAI embedding | GPU (fractional) |
| VLM Captioning | Qwen3-VL, Gemini, OpenAI, Nemotron, Cosmos-R1/R2, or vLLM async | GPU |
| Caption Enhancement | LLM refinement of VLM captions (Qwen-LM, GPT-OSS-20B, OpenAI) | GPU |
| Clip Writer | Write clips + metadata to disk or S3 | CPU |
| SAM3 Tracking (optional) | Object tracking + per-event captions | GPU (separate Pixi env) |

### 2. Semantic Dedup (`dedup`)

Clusters clip embeddings and removes near-duplicates.

```text
Input: embedding parquets from split-annotate
       (iv2_embd_parquet/, ce1_embd_<variant>_parquet/, or openai_embd_parquet/)
Process: K-Means clustering -> cosine similarity within clusters -> prune duplicates
Output: dedup summary CSV + pruning tables per cluster
```

### 3. Shard-Dataset (`shard`)

Produces WebDataset archives for Cosmos-Predict2 fine-tuning.

```text
Input: clips + captions + metadata from split-annotate
Process: T5 text embedding -> sharding by resolution/aspect-ratio/frame-range
Output: tar archives organized by resolution/aspect-ratio/frame-range
```

---

## Output Directory Structure

### Split-Annotate Output

```text
{output_clip_path}/
├── clips/                          # transcoded clip MP4s
│   └── {clip-uuid}.mp4
├── metas/v0/                       # per-clip metadata JSON (primary data source)
│   └── {clip-uuid}.json
├── iv2_embd/                       # InternVideo2 embedding per clip (pickle)
│   └── {clip-uuid}.pickle
├── ce1_embd/                       # Cosmos-Embed1 embedding (if selected)
│   └── {clip-uuid}.pickle
├── openai_embd/                    # OpenAI-compatible embedding (if selected)
│   └── {clip-uuid}.pickle
├── iv2_embd_parquet/               # grouped embeddings for dedup pipeline
│   └── {video-uuid}_{chunk}.parquet
├── ce1_embd_parquet/               # grouped Cosmos-Embed1 embeddings
│   └── {video-uuid}_{chunk}.parquet
├── openai_embd_parquet/            # grouped OpenAI embeddings
│   └── {video-uuid}_{chunk}.parquet
├── metas_jsonl/v0/                 # chunked metadata (upload_clip_info_in_chunks)
│   └── {video-uuid}_{chunk}.jsonl
├── cosmos_video2world_dataset/     # Cosmos-Predict2 dataset (generate_cosmos_predict_dataset)
│   ├── metas/
│   ├── t5_xxl/
│   └── videos/
├── previews/                       # web preview thumbnails (generate_previews)
│   └── {clip-uuid}_{frame_range}.webp
├── processed_videos/               # per-input-video processing record
│   └── {input-video-relpath}.json
├── v0/all_window_captions.json     # aggregated captions for all clips
└── summary.json                    # pipeline run statistics
```

When SAM3 is enabled, additional sibling directories are written next
to the regular per-clip outputs:

```text
{output_clip_path}/
├── sam3_instances/                 # tracked object summaries
├── sam3_objects/                   # per-frame object boxes
├── sam3_tracked/                   # annotated videos (when requested or event captioning is enabled)
└── sam3_events/                    # per-event JSON (requires event_captioning: true)
```

### Per-Clip Metadata Schema (`metas/v0/*.json`)

```json
{
  "span_uuid": "unique clip identifier",
  "source_video": "original video filename",
  "duration_span": [0.0, 10.5],
  "width": 1920,
  "height": 1080,
  "framerate": 30.0,
  "num_frames": 315,
  "windows": [
    {
      "start_frame": 0,
      "end_frame": 255,
      "qwen3_vl_30b_fp8_caption": "A busy intersection with vehicles..."
    }
  ],
  "has_caption": true,
  "total_prompt_tokens": 1200,
  "total_output_tokens": 450,
  "motion_score": {
    "global_mean": 0.0023,
    "per_patch_min_256": 0.00001
  },
  "aesthetic_score": 5.2,
  "qwen_type_classification": ["traffic_congestion"]
}
```

Distribution analysis reads captions from the dynamic window key matching
the selected captioning algorithm, e.g. `windows[].qwen3_vl_30b_fp8_caption`.
Classifier labels live in `qwen_type_classification`. Motion scores are a
map; use `motion_score.global_mean` for scalar ranking.

### `all_window_captions.json` Format

```json
{
  "video.mp4": {
    "clip_uuid": {
      "0_255": "caption for frames 0-255",
      "256_511": "caption for frames 256-511"
    }
  }
}
```

### Dedup Output

```text
{output_path}/
├── clustering_results/
│   ├── kmeans_centroids.npy
│   └── embs_by_nearest_center/
│       └── nearest_cent={index}/
│           └── {sha}.parquet
└── extraction/
    ├── dedup_summary_{eps}.csv
    └── semdedup_pruning_tables/
        └── cluster_{index}.parquet
```

### Shard-Dataset Output

```text
{output_dataset_path}/
└── v0/
    └── resolution_720/
        └── aspect_ratio_16_9/
            └── frames_0_255/
                ├── metas/part_000000/000000.tar
                ├── t5_xxl/part_000000/000000.tar
                └── video/part_000000/000000.tar
```

---

## Video Config Key Reference

The cosmos-curator video pipeline accepts a flat YAML config file. Keys
are `snake_case` argparse `dest` names. See `configs/split.yaml`,
`configs/dedup.yaml`, `configs/shard.yaml` for the authoritative
full-key reference with inline comments.

Source modules:
- `cosmos_curator/pipelines/video/splitting_pipeline.py` (split)
- `cosmos_curator/pipelines/video/sharding_pipeline.py` (shard)
- `cosmos_curator/pipelines/video/dedup_pipeline.py` (dedup)
- `cosmos_curator/pipelines/common_pipeline_settings.py` (common, see
  `references/cosmos-curator.md` for the shared settings table)

Flag counts: split ~200 (plus ~30 SAM3/event-captioning flags), shard
~25, dedup ~20.

### Video-Specific Prompt Variants

| Variant | Domain | Config key |
|---------|--------|------------|
| `default` | General video | `captioning_prompt_variant: "default"` |
| `av` | Autonomous driving | `captioning_prompt_variant: "av"` |
| Custom | Any | `captioning_prompt_text: "..."` |

The image pipeline uses a separate `caption_prompt_variant: "image"`
default; see `references/image-curation.md`. For the custom-prompt structure,
see `references/configuration-decision-tree.md` §The 8-Element Framework.

### Caption Enhancement (MCQ / LLM Refinement)

Available only for video. Enable with `enhance_captions: true` and
select the LLM backend with `enhance_captions_lm_variant`.

| LM Variant | Backend | GPU | Notes |
|-----------|---------|-----|-------|
| `qwen_lm` | Qwen2.5-14B (local) | ~15 GB | Default |
| `gpt_oss_20b` | GPT-OSS-20B (local) | ~20 GB | Alternative |
| `openai` | OpenAI API | None | Remote |

```text
Video clip -> [VLM: scene caption] -> [LLM: enhanced caption / MCQ]
```

### Super-Resolution (SeedVR2)

Video-only. Enable with `super_resolution: true`.

| Variant | Quality | GPU | Notes |
|---------|---------|-----|-------|
| `seedvr2_3b` | Good | ~15 GB | Faster |
| `seedvr2_7b` | Best | ~35 GB | Default |
| `seedvr2_7b_sharp` | Best + sharpening | ~35 GB | Enhanced detail |

---

## SAM3 Tracking And Per-Event Verification

`origin/nvidia/main` adds optional SAM3 object tracking to the split
pipeline. Treat SAM3 as a second source of truth for curation decisions:
it should ground and verify candidate events for the VLM, not become a
broad annotation phase unless the user explicitly asks for annotation
output. Enable it only when track-level evidence is needed; it adds a
separate `sam3` Pixi environment and extra GPU/disk cost.

See `references/sam3-config.md` for the canonical SAM3 verification profile —
the full YAML key set, the `sam3:` vs `enable_sam3:` gotcha, and the tuning
rationale (FPS, clip duration, reconditioning, detection/association thresholds).

Outputs are written next to regular clip metadata (see the
Split-Annotate Output section above).

Per-event captioning requires SAM3 and writes `sam3_events/` JSON:

```yaml
# As with `sam3:`, the canonical key is `event_captioning:` (argparse
# `--event-captioning`); `enable_event_captioning:` silently does nothing.
event_captioning: true
event_caption_backend: "qwen"  # or "gemini"
event_caption_qwen_variant: "qwen"
```

For the traffic-video-analytics cookbook, set `event_caption_prompt_file` to
the shipped `event_caption_prompt.txt` after copying it under `DATA_DIR`.

### Verification-first event captioning

Use the event-captioning stage as a grounded verification pass:

1. Let the main captioner/classifier propose candidate events with high
   recall.
2. Use focused `sam3_prompts` for the actors needed to prove or disprove
   those events, such as cars, motorcycles, pedestrians, buses, trucks,
   signals, or scene-specific objects.
3. Ask the event captioner to verify whether the candidate event is
   `present`, `absent`, or `uncertain` using SAM3 tracks, instance IDs,
   motion continuity, contact/proximity, and temporal order.
4. Emit a keep/drop/review recommendation only when the reasoning is
   tied to visible evidence and tracked instances.

Preferred reasoning-friendly event-caption prompt contract:

```json
{
  "events": [
    {
      "event_id": "event_000000",
      "start_time": 2.4,
      "end_time": 6.8,
      "category": "vehicle_to_vehicle_collision",
      "instances": ["id_3", "id_7"],
      "event_caption": "Vehicle id_3 contacts vehicle id_7, followed by abrupt motion changes.",
      "verification": "present",
      "dominant_incident": true,
      "incident_objects": [
        {"id": "id_3", "class": "a car", "role": "struck vehicle", "evidence": "path intersects id_7"},
        {"id": "id_7", "class": "a car", "role": "striking vehicle", "evidence": "abrupt stop after contact"}
      ],
      "reasoning_summary": "id_3 and id_7 intersect, then both change motion, supporting a collision judgment.",
      "decision_basis": {
        "track_association": "id_3 and id_7 are the involved vehicles",
        "temporal_relation": "paths intersect near 3.1s before both slow or rotate",
        "visual_change": "abrupt post-contact motion change",
        "category_choice": "vehicle_to_vehicle_collision is more specific than collision_aftermath",
        "counter_check": "no clear occlusion at the interaction point"
      },
      "evidence": [
        "id_3 trajectory intersects id_7 near 3.1s",
        "both tracks show abrupt speed or orientation change after contact"
      ],
      "counter_evidence": [],
      "confidence": 0.87,
      "curation_decision": "keep"
    }
  ]
}
```

Do not ask the VLM to emit long free-form chain-of-thought. Instruct it
to reason internally, then output concise, auditable reasoning fields:
`reasoning_summary` capped at 25 words and `decision_basis` with track
association, temporal relation, visible change, category choice, and
counter-check. These fields make the dataset useful for reasoning and
review without storing verbose hidden reasoning traces.

For traffic-safety event labels, make the prompt dominant-incident-first:
the category is what happened to the directly involved actors, not the final
scene state or the presence of routine traffic. Require the VLM to do an
object audit from SAM3 ids, emit `incident_objects` for 1-6 directly involved
actors/obstructions, and exclude ambient vehicles. Use category precedence:
fire/smoke, vehicle-pedestrian collision, motorcycle/scooter/two-wheeler
crash, vehicle-to-vehicle collision, person on ground, obstruction/stall,
emergency response, unsafe turn/right-of-way conflict, signal/wrong-way,
collision aftermath, then normal traffic. Use `collision_aftermath` only when
post-crash evidence is visible but the specific crash type or actors cannot be
resolved; use `normal_traffic_flow` only when no safety event exists anywhere
in the clip.

For `person_on_ground_in_roadway`, add explicit posture and roadway-zone
checks. Require visible lying, sitting, slumped, or motionless posture in a
travel lane, crosswalk, or intersection. A pedestrian who is standing, walking,
or waiting on a shoulder, sidewalk, median, or roadside is counter-evidence for
this label; do not let a generic SAM3 `pedestrian` box trigger it by itself.

If the current upstream parser accepts only the base event fields, keep
the same verification content inside `event_caption` and the category
name, then validate with a small run before scaling. Do not mark a target
event as present just to avoid an empty output. If the pipeline requires
a nonempty event array, emit a clearly negative/background category such
as `normal_traffic_flow` with an `absent_for_target_event` explanation.

### Verification evidence checklist

For traffic-event curation, prefer evidence that ties the event to tracks:

- involved instance IDs and object classes
- start/end time of the observed behavior
- trajectory intersection, following distance, lane relation, or path crossing
- abrupt speed/orientation change after contact or avoidance
- visible counter-evidence, such as occlusion, no contact, or normal lane following
- final `keep`, `drop`, or `review` decision for the target curation task

---

## Multi-Camera Mode

For synchronized multi-angle video datasets:

```yaml
multi_cam: true
primary_camera_keyword: "front"
```

Groups videos by camera angle using filename keywords. The primary
camera's clips are used as the reference for synchronization. Image
pipeline does not support multi-camera mode.

---

## Presigned URL Support

For environments where S3 credentials cannot be configured:

- `input_presigned_s3_url`: HTTPS URL pointing to a ZIP of input videos.
- `output_presigned_s3_url`: HTTPS URL for output ZIP upload.

No AWS credentials needed when using presigned URLs. The image pipeline
does not support presigned URL mode today.

---

## Performance Tuning (Video-Specific)

For framework-wide concepts (GPU fractional allocation, vLLM modes,
Pixi envs) see `references/cosmos-curator.md`. The tables below list the
video pipeline's stage-specific defaults.

### Video Stage GPU Allocation

| Stage | Default GPUs | Typical VRAM |
|-------|-------------|-------------|
| TransNetV2 | 0.25 | ~2 GB |
| Motion filter | 0.5 | ~4 GB |
| Aesthetic filter | 0.25 | ~2 GB |
| Artificial text filter | 0.25 | ~2 GB |
| Embedding (IV2) | 0.25 | ~4 GB |
| Qwen captioning | 1.0 | ~20-35 GB |
| VLM filter | 1.0 (shared) | ~20-35 GB |
| Video classifier | 1.0 (shared) | ~20-35 GB |
| Super-resolution | 1.0+ | ~15-35 GB |
| SAM3 tracking | 1.0 | ~12-20 GB |

On a 48 GB GPU (L40S), TransNetV2 + aesthetic + embedding can share
one GPU while captioning uses another.

### Video Batch Sizes

| Parameter | Default | Tuning guidance |
|-----------|---------|----------------|
| `qwen_batch_size` | 8 | Increase for throughput, decrease if OOM |
| `vlm_filter_batch_size` | 16 | Higher = faster filtering |
| `embedding_batch_size` | 8 | Increase if GPU memory allows |
| `motion_score_batch_size` | 64 | CPU-bound, higher is usually fine |
| `enhance_captions_batch_size` | 32 | LLM batch size |
| `transcode_ffmpeg_batch_size` | 16 | CPU-bound |

### Video Worker Counts

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_download_workers_per_node` | 4 | Increase for S3 I/O bound workloads |
| `num_clip_writer_workers_per_node` | 8 | Increase for high clip counts |
| `vllm_prepare_num_cpus_per_worker` | 3.0 | CPU prep for VLM input |
| `transcode_cpus_per_worker` | 5.0 | FFmpeg encoding |

---

## Running The Video Pipelines

All invocation modes (Makefile, Docker direct, in-container, and native CLI
subcommand) are documented canonically in `references/running-pipelines.md`
(§Method 1–3 and the CLI-debugging note). The video module is
`cosmos_curator.pipelines.video.run_pipeline`: config-file mode takes a single
config path (swap `split.yaml` for `dedup.yaml` / `shard.yaml`), and CLI mode
(`split --input-video-path ...`) is for ad hoc flags.

For the image counterpart, see `references/image-curation.md` -- use
`make run_image_pipeline` (upstream image config-file mode).

---

## Resource Requirements (Video Components)

| Component | GPU VRAM | Notes |
|-----------|----------|-------|
| Qwen3-VL-30B-FP8 | ~20 GB | VLM captioning (FP8) |
| Qwen3-VL-30B | ~35 GB | VLM captioning (FP16) |
| Qwen2.5-14B (LM) | ~15 GB | LLM MCQ / caption enhancement |
| Nemotron-Nano-12B-v2-VL | ~15 GB | Lighter VLM alternative |
| Cosmos-Reason1/2-VL | ~35 GB | NVIDIA reasoning VLM |
| TransNetV2 | ~2 GB | Scene detection |
| InternVideo2 | ~4 GB | Embedding generation |
| Cosmos-Embed1 | ~4 GB | Alternative embedding |
| SeedVR2-7B | ~35 GB | Super-resolution |
| Motion filter | ~4 GB | Optical flow |
| Aesthetic filter | ~2 GB | CLIP aesthetic |
| SAM3 | ~12 GB | Object tracking when `sam3: true` (NOT `enable_sam3:`) |

Recommended GPUs: L40S (48 GB) or H100 (80 GB).

---

## Cross-References

- `references/cosmos-curator.md` -- framework architecture, Pixi envs,
  vLLM plugin catalog, captioning/embedding algorithm enums shared
  with the image pipeline, stage replay/compare, and generic
  performance-tuning concepts.
- `references/image-curation.md` -- still-image annotate pipeline.
- `references/configuration-decision-tree.md` -- when to choose video
  vs. image pipelines and how to pick algorithms.
- `references/running-pipelines.md` -- invocation patterns, SHM sizing, S3 setup,
  troubleshooting (shared by all pipelines).
- `configs/split.yaml`, `configs/dedup.yaml`, `configs/shard.yaml` --
  full-key reference YAMLs with inline comments.
- `cookbook/traffic-video-analytics/` -- Path A operator recipes
  (`split-minimal.yaml` is CE1-only first-run).
