# Running the Pipeline

Operational guidance for config preflight, runtime preparation, secure
credentials, execution, monitoring, and troubleshooting.

## Choose the operational route

- **Advisory:** for sizing, monitoring, command planning, or troubleshooting,
  inspect the available config and run evidence, then report guidance. Do not
  prepare credentials or execute.
- **Execution:** only an explicit run request follows this order:
  prepare runtime → obtain credentials through approved runtime injection →
  validate config/runtime → confirm authorization → execute → validate output.
  Stop on any failed preflight.

## Prerequisites

1. Docker image available: `make pull` then `make check-image`.
2. **FFmpeg sidecar** installed: `make ffmpeg-install` (see
   [ffmpeg-sidecar.md](ffmpeg-sidecar.md)).
3. Models downloaded: `make download-models MODELS_DIR=/path/to/models`.
4. Config file prepared and reviewed (cookbook recipe or `configs/` reference).
5. `make check-setup` and, before a Curator GPU run,
   `make check-curator-runtime MODELS_DIR=/path/to/models`.
6. Environment variables set (see below).

## Mandatory config preflight

Validate every config before Docker starts:

1. Confirm the file exists, parses as YAML, uses
   `pipeline: split|dedup|shard|annotate`, and contains the required paths for
   that pipeline. Curator parameters may be flat or nested under `args`.
2. Reject `enable_sam3` and `enable_event_captioning`. They are deprecated and
   incorrect for upstream config-file mode; the canonical keys are `sam3` and
   `event_captioning`.
3. Confirm config paths, output permissions, model paths, selected GPU/SHM
   constraints, and pipeline-to-pipeline handoffs.
4. Do not execute until validation succeeds and the user has authorized the
   run.

Agents and operators must enforce this preflight before invoking Make or Docker.
Failures use ordinary human-readable Click errors with a nonzero exit;
automation must not assume a uniform JSON error envelope.

### Distributable images: use `pixi run --as-is`

Pre-built cosmos-curator Docker images ship pixi environments under
`/opt/cosmos-curator/.pixi/envs/`. Always pass **`--as-is`** so pixi uses
those envs without re-solving or rebuilding at runtime:

```bash
pixi run -e default --as-is python -m cosmos_curator.pipelines.video.run_pipeline ...
```

The Makefile and pipeline entrypoints follow this pattern. Omitting `--as-is` fails
on distributable images (permission errors / missing `pyproject.toml`).

## Environment Variables

Prepare these only for an explicit run request, never for advisory guidance.

### S3 Credentials

S3 operations require `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`S3_ENDPOINT_URL`. Inject them into the process environment at runtime through
an approved secret manager or operator deployment mechanism. Do not create,
source, copy, log, or share a credential-bearing file in this repository.

### API Keys (for Gemini/OpenAI backends)

Inject only the key required by the selected backend from an approved secret
manager into the process environment. Do not put credential values in command
arguments, committed files, examples, logs, or shell history. The supported
variable names are `NVIDIA_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY`.

### Model Weights Path

```bash
export MODELS_DIR="/path/to/models"
```

Or set `model_weights_path` in the config YAML.

### Verify Environment

```bash
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:+SET}"
echo "S3_ENDPOINT_URL: ${S3_ENDPOINT_URL:-NOT SET}"
echo "MODELS_DIR: ${MODELS_DIR:-NOT SET}"
```

Do not print or otherwise inspect secret values. Verify only whether the
required variables are present.

Environment variables are ephemeral -- re-source at the start of each session.

---

## Running the Pipeline

### Method 1: Makefile (recommended)

```bash
make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml \
  MODELS_DIR=/path/to/models DATA_DIR=$PWD/cookbook/traffic-video-analytics \
  DOCKER_NETWORK=bridge MODELS_MOUNT_MODE=rw \
  CURATOR_TMP=/path/to/large/scratch
```

Makefile default when `CONFIG_FILE` is omitted is `configs/split.yaml` (full
flag reference, placeholder I/O). Prefer a cookbook recipe for first run.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | `configs/split.yaml` | Pipeline YAML. First-run: `cookbook/traffic-video-analytics/split-minimal.yaml` |
| `MODELS_DIR` | `$HOME/models` | Host path to model weights |
| `DATA_DIR` | *(empty)* | Host path to input/output data |
| `FFMPEG_DIR` | `$HOME/cosmos-curator-ffmpeg` | Host FFmpeg sidecar (read-only at `/opt/ffmpeg`) |
| `DOCKER_NETWORK` | `bridge` | Prefer bridge; `host` can break Ray |
| `MODELS_MOUNT_MODE` | `rw` | `rw` or `ro` (+ HF cache under `/tmp`) |
| `CURATOR_TMP` | unset | Optional host scratch → `/tmp` + `/config/tmp` |
| `IMAGE_NAME` | `cosmos-curator` | Docker image name |
| `IMAGE_TAG` | `2.3.0` | Docker image tag |
| `GPUS` | `all` | GPU allocation |

### Method 2: Docker Direct

Do not hand-roll `docker run`. `make run-pipeline` already applies bridge
networking, the FFmpeg sidecar mount, model and data mounts, and
`IMAGE_NAME`/`IMAGE_TAG` (`cosmos-curator` / `2.3.0`). Override with
`EXTRA_DOCKER_ARGS` only when the Makefile documents that escape hatch.

### Method 3: Inside Container

```bash
make shell
# Inside the container:
pixi run -e default --as-is python -m cosmos_curator.pipelines.video.run_pipeline /config/pipeline_config.yaml
```

For native upstream CLI debugging, bypass YAML and call the subcommand
directly:

```bash
pixi run -e default --as-is python -m cosmos_curator.pipelines.video.run_pipeline \
  split --input-video-path /data/input --output-clip-path /data/output
```

---

## Multi-Stage Workflow

The three pipelines run as separate invocations. Each stage reads the
previous stage's output.

```bash
# Stage 1: Split videos into captioned clips (first-run: split-minimal.yaml)
make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml
make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml

# Stage 2: Deduplicate clips by embedding similarity
make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/dedup.yaml

# Stage 3: Package into WebDataset tar archives
make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/shard.yaml
```

Ensure I/O paths chain correctly:
- `dedup.input_embeddings_path` = `split.output_clip_path`
- `shard.input_clip_path` = `split.output_clip_path`
- `shard.input_semantic_dedup_path` = `dedup.output_path`

---

## Docker Volume Mounts

The Makefile `run-pipeline` target mounts:

| Host Path | Container Path | Mode | Purpose |
|-----------|---------------|------|---------|
| `$FFMPEG_DIR` | `/opt/ffmpeg` | `ro` | FFmpeg sidecar (required for distributable images) |
| `$MODELS_DIR` | `/config/models` | `rw` | Model weights (Make default; set `MODELS_MOUNT_MODE=ro` to override) |
| `$CONFIG_FILE` | `/config/pipeline_config.yaml` | `ro` | Config file |
| `$DATA_DIR` | Same as host | `rw` | Input/output data |

When using Make, additional host paths are mounted through `DATA_DIR` and
`MODELS_DIR`. Do not add extra bind mounts by editing `docker run` by hand.

---

## GPU Configuration

### Makefile

```bash
GPUS=all make run-pipeline CONFIG_FILE=...
GPUS='"device=0"' make run-pipeline CONFIG_FILE=...
GPUS='"device=0,1,2"' make run-pipeline CONFIG_FILE=...
```

### Docker Direct

Do not pass `--gpus` on a hand-rolled `docker run`. Use the Make `GPUS`
variable instead (see the Makefile block above).

### SHM Sizing

Docker shared memory is allocated from host RAM. The Makefile default is
`SHM_SIZE=24gb`. Use roughly `8gb-16gb` for single-GPU runs, `24gb-64gb`
for typical multi-GPU runs, and scale higher only on hosts with enough
system RAM.

---

## Key Makefile Targets

| Target | Description |
|--------|-------------|
| `make ffmpeg-install` | Install conda-forge LGPL FFmpeg into `FFMPEG_DIR` |
| `make check-setup` | Validate Docker, NVIDIA driver, FFmpeg sidecar |
| `make run-pipeline` | Run cosmos-curator pipeline in container |
| `make run_image_pipeline` | Run image annotate pipeline in container |
| `make shell` | Interactive shell in container |
| `make format` | Run ruff format |
| `make download-models` | Download model weights |

---

## Output Structure

### Split Output

```text
{output_clip_path}/
  ├── clips/                    transcoded clip MP4s
  ├── metas/v0/                 per-clip metadata JSON
  ├── iv2_embd_parquet/         InternVideo2 embeddings (for dedup)
  ├── ce1_embd_parquet/         Cosmos-Embed1 embeddings (if selected)
  ├── v0/all_window_captions.json
  └── summary.json
```

### Dedup Output

```text
{output_path}/
  ├── clustering_results/
  └── extraction/
      ├── dedup_summary_{eps}.csv
      └── semdedup_pruning_tables/
```

### Shard Output

The shard pipeline writes WebDataset tar archives under
`{output_dataset_path}/v0/` bucketed by resolution / aspect-ratio / frame-range
(`metas/`, `t5_xxl/`, `video/`). See `references/video-curation.md` §Shard-Dataset
Output for the authoritative directory tree.

---

## Monitoring Progress

Cosmos-curate logs to stdout:
```text
[INFO] Stage: VideoDownloader - Processing 50 videos...
[INFO] Stage: TransNetV2Splitter - Completed in 12m 34s
[INFO] Stage: QwenCaptioner - Processing 150 clips...
```

Monitor GPU utilization: `nvidia-smi -l 1`

---

## Scaling Guidelines

| Videos | Approach |
|--------|----------|
| 1-10 | Single run, single GPU |
| 10-100 | Single run, 2-4 GPUs |
| 100-1000 | Split into batches of 50-100, run sequentially |
| 1000+ | Multiple batch runs + dedup + shard |

---

## Troubleshooting

### CUDA Out of Memory (OOM)

Symptoms: `torch.cuda.OutOfMemoryError`, `Actor died unexpectedly`.

Fixes (in order of preference):
1. Use FP8 model: `captioning_algorithm: "qwen3_vl_30b_fp8"`
2. Disable embeddings: `generate_embeddings: false`
3. Disable filters: `motion_filter: "disable"`, `aesthetic_threshold: null`
4. Limit input: `limit: 10`
5. Use fewer GPUs to give more VRAM per worker

### Ray OutOfDiskError

Symptoms: `ray.exceptions.OutOfDiskError`, `No space left on device`.

Fix: inspect Ray temp usage and remove only stale run directories after
the failed pipeline is stopped. Do not blindly delete `/tmp/ray/*` on a
shared host.

```bash
du -sh /tmp/ray/* 2>/dev/null
```

Prefer mounting a large disk for Ray temp. Inside container, set
`RAY_TEMP_DIR=/path/to/large/disk`.

### SeedVR2 checkpoints missing (`ema_vae.pth` / embeds / `seedvr2_ema_*.pth`)

Curator SeedVR uses two layouts:

1. DiT/VAE under `/opt/cosmos-curator/SeedVR/ckpts` (host `$MODELS_DIR/seedvr2/`)
2. Fixed text embeds `pos_emb.pt` / `neg_emb.pt` under
   `/config/models/ByteDance-Seed/SeedVR2-{3B|7B}/`

PAIDF downloads all of these into `$MODELS_DIR/seedvr2/` and bind-mounts
the directory plus the two embed files onto the Curator paths at run time
(so a root-owned empty HF stub directory is not required).

HuggingFace sources (override with `HF_REPO_SEEDVR2_3B` / `_7B`):

| File | Repo |
|------|------|
| `ema_vae.pth` | `ByteDance-Seed/SeedVR2-7B` (fallback `…-3B`) |
| `seedvr2_ema_3b.pth` | `ByteDance-Seed/SeedVR2-3B` |
| `seedvr2_ema_7b.pth` | `ByteDance-Seed/SeedVR2-7B` |
| `pos_emb.pt` / `neg_emb.pt` | variant repo (`SeedVR2-3B` or `…-7B`) |

```bash
# Standalone download (DiT/VAE + text embeds)
make download-seedvr2 MODELS_DIR=/path/to/models SEEDVR_VARIANT=seedvr2_3b

# Or let preflight pull when CONFIG_FILE enables super_resolution
make run-pipeline CONFIG_FILE=... MODELS_DIR=... ENSURE_SEEDVR_CKPTS=auto
```

Set `HF_TOKEN` or `~/.config/cosmos_curator/hf_token.txt` if the HF repos are
gated. Missing SeedVR weights with SR enabled → SQA `BLOCKED`, not product FAIL.

### SHM Exhaustion

Symptoms: `Bus error`, `Unable to allocate shared memory`.

Fix: Increase SHM size on the Make invocation:
```bash
SHM_SIZE=32gb make run-pipeline CONFIG_FILE=<split-config>
```

Verify in container: `df -h /dev/shm`

### Docker Image Not Found

```text
docker: Error response from daemon: No such image: cosmos-curator:2.3.0
```

Fix:
```bash
make pull                                          # pull from the default NGC registry
make pull IMAGE_TAG=2.3.0                              # or pull a pinned tag
```

### FFmpeg Not Found in Container

Symptoms: `ffmpeg: command not found`, transcode stage fails immediately.

Fix:
```bash
make ffmpeg-install
make check-setup
```

See [ffmpeg-sidecar.md](ffmpeg-sidecar.md). Prefer `make run-pipeline` so the
read-only FFmpeg bind mount is included.

### Permission Denied on Volume Mounts

Create a private operator-owned directory and verify it is writable. Do not
change arbitrary host paths to root ownership or grant write access to every
local account. Use mode `0750` (owner and group only).

```bash
install -d -m 0750 /path/to/output
test -w /path/to/output
```

If the image requires a specific runtime UID/GID, use the documented
`EXTRA_DOCKER_ARGS=--user=<uid>:<gid>` mapping and grant only that identity the
minimum required access.

### Pipeline Running Slowly

1. Check GPU utilization: `nvidia-smi -l 1`
2. Common bottlenecks:
   - VLM captioning (GPU-bound): use `vllm_async` with more GPUs
   - Transcoding (CPU-bound): increase `transcode_cpus_per_worker`
   - I/O (disk-bound): use SSD for output, NVMe for Ray temp
3. Enable profiling: `profile_cpu: true`, `profile_gpu: true`

### NCCL Timeout / Ray Actor Hang

Symptoms: Pipeline hangs during model initialization, NCCL errors.

Fixes:
1. Reduce GPU count to isolate the issue
2. Check that all GPUs are visible: `nvidia-smi`
3. Increase SHM within available RAM, for example `--shm-size=32gb`
4. Inspect Docker disk usage with `docker system df`; prune only with explicit user approval.
5. Set `NCCL_DEBUG=WARN` environment variable for diagnostics

### Dedup Fails with "n_samples < n_clusters"

The dataset is too small for the requested cluster count.

Fix: Reduce `n_clusters` to be <= number of clips with embeddings.
