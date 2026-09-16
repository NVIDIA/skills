# Cosmos-Curator Framework Reference

Shared architecture and framework concepts for the NVIDIA Cosmos-Curator
engine that powers all of this repo's pipelines. For pipeline-specific
detail, read the dedicated references:

- `references/video-curation.md` -- split / dedup / shard pipelines.
- `references/image-curation.md` -- image annotate pipeline.

---

## Image acquisition and verification

### Primary product path: pull

Use the pre-built Cosmos-Curator image selected by the repository
configuration:

```bash
make pull
make check-image
make check-setup
```

`make pull` reads the configured registry, image name, and tag from `.env` and
Make defaults, then creates the local tag used by the pipeline targets. Do not
copy registry/tag values into skill instructions; inspect the active
configuration when exact provenance is required.

`make check-image` verifies that the configured local image exists.
`make check-setup` verifies Docker access, the NVIDIA runtime, and the required
host FFmpeg sidecar. The sidecar must expose compatible binaries and libraries
through the repository's read-only `/opt/ffmpeg` mount.

### Optional developer-only source path

Source builds are for custom upstream development only. They are outside the
normal product/release path and are not release evidence.

```bash
make clone-curator
make build-dry-run
make build
```

The repository does not vendor Cosmos-Curator. `make clone-curator` creates an
ignored external checkout at the configured `COSMOS_REPO`; override that
variable to use an existing checkout. Run `make build-dry-run` first to validate
the checkout and source-build prerequisites without building the image. Run
`make build` only after the dry run succeeds.

Source-built image verification still starts with `make check-image`. FFmpeg
requirements depend on the resulting image; verify the configured sidecar
through `make check-setup`.

---

## Architecture

```text
paidf-curation-and-retrieval (this repo)
  ├── cookbook/                 Operator first-run recipes (split-minimal, split, dedup, shard)
  ├── configs/                  Full flag-reference YAML (Makefile CONFIG_FILE default)
  └── Makefile                  Pull, run, format targets

cosmos-curator (base Docker image)
  ├── cosmos_curator/pipelines/video/   split, dedup, shard pipelines
  ├── cosmos_curator/pipelines/image/   annotate pipeline (load -> filter -> embed -> caption -> write)
  ├── cosmos_curator/core/              stage/model/pipeline interfaces
  └── cosmos_curator/client/            CLI launcher (local, Slurm, NVCF)
```

Configs are flat YAML files consumed via upstream
`load_pipeline_config`. Each key is a `snake_case` argparse `dest`
name. Omitted keys get upstream parser defaults.

> **Image pipeline note:** upstream
> `cosmos_curator.pipelines.image.run_pipeline` accepts a YAML/JSON config
> with `pipeline: annotate` (same config-file UX as video). Prefer
> `make run_image_pipeline IMAGE_CONFIG_FILE=configs/image.yaml`. See
> `references/image-curation.md`.

---

## Pipelines (Overview)

The repo wires up four pipelines on top of the same engine. Read the
linked reference for stage detail, output schema, and runner usage.

| Pipeline | Modality | Purpose | Reference |
|----------|----------|---------|-----------|
| `split` | Video | Splits videos into clips, generates captions / embeddings / metadata. | `references/video-curation.md` |
| `dedup` | Video | K-Means semantic deduplication of split embeddings. | `references/video-curation.md` |
| `shard` | Video | Packages dedup output into WebDataset tars for Cosmos-Predict2 fine-tuning. | `references/video-curation.md` |
| `annotate` | Image | Loads still images, optionally filters, embeds, and captions them. | `references/image-curation.md` |

---

## Common Settings (All Pipelines)

These are mirrored by `CommonPipelineSettings` in
`cosmos_curator/pipelines/common_pipeline_settings.py` and surfaced as
CLI flags by `cosmos_curator/pipelines/pipeline_args.py:add_common_args`.
Every pipeline (video and image) accepts them.

| YAML key | CLI flag | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `input_s3_profile_name` | `--input-s3-profile-name` | str | `"default"` | AWS profile for input S3 |
| `output_s3_profile_name` | `--output-s3-profile-name` | str | `"default"` | AWS profile for output S3 |
| `execution_mode` | `--execution-mode` | str | `"AUTO"` | `AUTO`, `BATCH`, or `STREAMING` |
| `limit` | `--limit` | int | `0` | Max input items (0 = unlimited) |
| `verbose` | `--verbose` | bool | `false` | Verbose logging |
| `model_weights_path` | `--model-weights-path` | str | upstream default | Path to model weights (S3 or local) |
| `perf_profile` | `--perf-profile` / `--no-perf-profile` | bool | `true` | Lightweight performance profiling |
| `profile_tracing` | `--profile-tracing` | bool | `false` | OpenTelemetry distributed tracing |
| `profile_cpu` | `--profile-cpu` | bool | `false` | pyinstrument CPU profiling |
| `profile_memory` | `--profile-memory` | bool | `false` | memray memory profiling |
| `profile_gpu` | `--profile-gpu` | bool | `false` | torch.profiler GPU profiling |

---

## Captioning Algorithms (Shared Enum)

Both the video pipeline (`captioning_algorithm`) and the image pipeline
(`captioning_algorithm`, plus `semantic_filter_model_variant` and
`image_classifier_model_variant`) draw from the same upstream enum
defined at `cosmos_curator/pipelines/image/captioning/captioning_builders.py`
(`IMAGE_CAPTION_ALGOS`) and the parallel video registry. Pick the
backend appropriate to your modality:

| Algorithm | Backend | GPU requirement | Used by |
|-----------|---------|----------------|---------|
| `qwen` | Local vLLM (Qwen2-VL-7B) | ~15 GB | Video + Image (default for image) |
| `qwen3_5_27b` | Local vLLM (Qwen3.5-27B) | ~20 GB | Video + Image |
| `qwen3_6_27b` / `qwen3_6_27b_fp8` | Local vLLM (Qwen3.6-27B) | ~35 GB / ~20 GB | Video + Image (added in v2.0.0) |
| `qwen3_vl_30b` | Local vLLM (Qwen3-VL-30B) | ~35 GB | Video + Image |
| `qwen3_vl_30b_fp8` | Local vLLM (Qwen3-VL-30B) | ~20 GB | Video + Image (recommended for video) |
| `qwen3_vl_235b` / `qwen3_vl_235b_fp8` | Local vLLM (Qwen3-VL-235B) | Multi-GPU | Video + Image |
| `cosmos_r1`, `cosmos_r2` | Local vLLM (Cosmos-Reason-VL) | ~35 GB | Video + Image |
| `nemotron` | Local vLLM (Nemotron-Nano-12B-v2-VL) | ~15 GB | Video + Image |
| `gemini` | Gemini API | None (remote) | Video + Image (requires `GOOGLE_API_KEY`) |
| `openai` | OpenAI-compatible API | None (remote) | Video + Image |
| `vllm_async` | vLLM async engine (auto-configured in v2.0.0) | Configurable | **Video only** -- multi-GPU autoscale |

For video-only knobs (multi-GPU captioning via `qwen_num_gpus_per_worker`
+ `vllm_performance_mode`; the explicit `vllm_async_*` engine knobs were
removed in v2.0.0; prompt variants `default` / `av`, caption enhancement) see
`references/video-curation.md`. For image-only knobs (`caption_prep_*`,
`caption_prompt_variant: "image"`, semantic filter / classifier
variants) see `references/image-curation.md`.

---

## Embedding Algorithms (Shared Enum)

| Algorithm | Backend | GPU requirement | Used by |
|-----------|---------|----------------|---------|
| `internvideo2` | InternVideo2 ViT | ~4 GB | Video (default) + Image (default) |
| `cosmos-embed1-224p` | Cosmos-Embed1 224p | ~4 GB | Video + Image |
| `cosmos-embed1-336p` | Cosmos-Embed1 336p | ~4 GB | Video + Image |
| `cosmos-embed1-448p` | Cosmos-Embed1 448p | ~4 GB | Video + Image |
| `clip` | OpenAI CLIP ViT-L/14 | ~4 GB | **Image only** |
| `openai` | OpenAI-compatible embedding API | None (remote) | Video + Image |

The image pipeline writes embeddings under
`embeddings/<backend>/{output_id}.npy` (one backend per run); the video
pipeline writes per-clip pickles plus grouped parquets under
`<algo>_embd/` and `<algo>_embd_parquet/`. See the modality-specific
references for details.

---

## Pixi Environments

Cosmos-Curator uses Pixi (conda-based) environments for dependency
isolation. Stages declare `conda_env_name` to pin themselves to the
right env.

| Environment | Purpose | Key packages |
|-------------|---------|-------------|
| `default` | Release split / annotate environment | Python 3.12, Ray, release-supported stages |
| `transformers` | HuggingFace transformers stages | transformers, torch |
| `legacy-transformers` | Older transformer versions | Backward compat |
| `cuml` | RAPIDS cuML for clustering | cuML, cuDF (dedup pipeline) |
| `model-download` | Model weight downloading | HuggingFace Hub |
| `paddle-ocr` | OCR-based text detection | PaddleOCR |
| `seedvr2` | Super-resolution (video only) | SeedVR2 weights |
| `sam3` | SAM3 tracking (video only) | SAM3 weights |

Override which envs are baked into the image with the
`COSMOS_IMAGE_ENVS` Makefile variable.

> **FFmpeg:** distributable registry images expect a host sidecar mounted at
> `/opt/ffmpeg`. See `references/ffmpeg-sidecar.md`.

---

## Architecture Overview

Cosmos-Curator is built on **Cosmos-Xenna**, a GPU-accelerated streaming
pipeline framework using Ray.

### Core Concepts

- **PipelineTask**: Unit of work (one or more videos for the video
  pipelines, one image per task for the image annotate pipeline).
  Override `weight` for load balancing.
- **CuratorStage**: Processing step. Implements `resources` (CPU/GPU
  allocation), `conda_env_name`, and `process_data(task)`.
- **ModelInterface**: ML model wrapper. Implements `conda_env_name`,
  `model_id_names`, `setup()`. Registered in `all_models.py`.

### Stage Lifecycle

```text
stage_setup_on_node()    # once per node (download models, warm caches)
  -> stage_setup()       # once per worker (load model into GPU)
    -> process_data()    # called per batch (stage_batch_size, default 1)
```

### Execution Model

- One container per node (e.g., 200 CPU cores + 8 GPUs).
- Each stage has a pool of Ray actors (stateful workers).
- The orchestrator feeds tasks into stage 1 and moves completed tasks
  to the next stage.
- Auto-scaling adjusts worker pool sizes to balance throughput.
- Ray object store streams data between stages (pointer passing, not
  copies).
- Node affinity minimizes cross-node data movement.

`execution_mode: AUTO` picks STREAMING when the number of GPUs
requested by stages is at most the number of available GPUs, else
BATCH. STREAMING and BATCH force the respective mode.

---

## vLLM Interface

The `vllm_interface` module provides a plugin-based abstraction for
VLM captioning, used by both video and image pipelines.

### Available Model Plugins

| Plugin | Model | VRAM | Notes |
|--------|-------|------|-------|
| `VllmQwen7B` | Qwen2-VL-7B | ~15 GB | Default `qwen` algorithm |
| `VllmQwen3527B` | Qwen3.5-27B-FP8 | ~20 GB | Newer FP8 model |
| `VllmQwen3VL30B` | Qwen3-VL-30B | ~35 GB | FP16 |
| `VllmQwen3VL30BFP8` | Qwen3-VL-30B-FP8 | ~20 GB | FP8 quantized |
| `VllmQwen3VL235B` | Qwen3-VL-235B | Multi-GPU | Large model |
| `VllmNemotronNano12Bv2VL` | Nemotron-Nano-12B-v2-VL | ~15 GB | Lighter |
| `VllmCosmosReason1VL` | Cosmos-Reason1-VL | ~35 GB | NVIDIA reasoning |
| `VllmCosmosReason2VL` | Cosmos-Reason2-VL | ~35 GB | NVIDIA reasoning v2 |

The same plugins back the image pipeline's
`captioning_algorithm` / `semantic_filter_model_variant` /
`image_classifier_model_variant`; the wrapping stage modules differ
(`pipelines/video/captioning/` vs `pipelines/image/captioning/`).

### vLLM Async Engine (Video Only)

The `vllm_async` captioning algorithm provides advanced multi-GPU
scaling with autoscaling workers per node. It currently exists only on
the video side. As of cosmos-curator v2.0.0 the explicit `vllm_async_*`
engine knobs were removed: the engine now auto-configures, and multi-GPU
scaling is driven by the general `qwen_num_gpus_per_worker` (tensor
parallelism) and `vllm_performance_mode` knobs (see
`references/video-curation.md`).

---

## Stage Replay And Compare (Debugging)

Stage replay allows saving and replaying individual pipeline stages in
isolation. Stage compare validates replayed stage output against a
saved golden output. The mechanism is shared by all pipelines, though
in practice it has been most heavily exercised on the video side.

### Save Tasks

In your config YAML:

```yaml
stage_save: ["VllmCaptionStage", "MotionVectorDecodeStage"]
stage_save_sample_rate: 1.0
```

Tasks are saved as pickle files at
`{output_path}/tasks/{StageName}/{asset_name}_NNN.task.pkl`.

### Replay a Stage

```yaml
stage_replay: ["VllmCaptionStage"]
```

### Compare a Stage

```yaml
stage_compare: ["VllmCaptionStage"]
stage_compare_pass_threshold: 1.0
```

Use cases:
- Debug a specific stage without running the full pipeline.
- Iterate on stage logic with real production data.
- Profile stage performance in isolation.
- Test changes with the exact same inputs every time.

---

## Performance Tuning (Generic Concepts)

For modality-specific stage GPU/batch/worker tables, see:
- `references/video-curation.md` -- video stages (TransNetV2, motion,
  aesthetic, qwen captioning, super-resolution, SAM3, etc.).
- `references/image-curation.md` -- image stages (load, filter, embed,
  caption prep, caption, write).

### GPU Fractional Allocation

Most stages accept a `gpus_per_worker` (or stage-named equivalent)
parameter as a float, allowing multiple stages to share a single GPU.
For example, on a 48 GB L40S a video pipeline can run TransNetV2 +
aesthetic + embedding co-resident on one GPU while captioning uses
another.

### Batch Sizes

Each stage that wraps an ML model exposes a `*_batch_size` knob.
Increasing it raises throughput at the cost of GPU memory; decrease
when hitting OOM.

### Worker Counts

CPU-bound stages (download, transcode, FFmpeg, image load/write,
metadata writers) expose `num_*_workers_per_node` knobs. Tune up for
S3-bound or write-heavy workloads.

### vLLM Performance Modes

| Mode | Behavior |
|------|----------|
| `throughput` | Maximize tokens/second, larger batches |
| `balanced` | Balance latency and throughput |
| `interactivity` | Minimize per-request latency |

---

## Cross-References

- `references/video-curation.md` -- video pipeline stages, output
  schemas, video config key reference, prompt variants, caption
  enhancement, SeedVR2, SAM3, multi-camera mode, presigned URLs,
  video runner invocations, video resource VRAM table.
- `references/image-curation.md` -- image annotate pipeline stages,
  output schema, per-image metadata, resume behavior, image runner
  invocation via config-file mode.
- `references/configuration-decision-tree.md` -- when to choose video
  vs. image pipelines and how to pick algorithms.
- `references/running-pipelines.md` -- invocation patterns, SHM sizing, S3 setup,
  troubleshooting (shared by all pipelines).
- `references/ffmpeg-sidecar.md` -- host FFmpeg install, mount, verification.
