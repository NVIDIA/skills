# Build Reference: DS-SOP Image (`ds-sop:1.0.0`)

> This file is owned by `vss-build-vision-ai` — a **build reference**, NOT an invokable
> skill. It documents how to build the DS-SOP DeepStream inference image `ds-sop:1.0.0` from
> the NVIDIA sop-monitoring-blueprints source and smoke-test it standalone (`API_DUMMY_TEST`).
> Build `ds-sop:1.0.0` **if absent** before deploy (a locally-built image; see
> `deploy-ds-sop.md § Container Image`) — build-vision-ai's deploy does not yet
> auto-build local images, so this is an operator/pre-deploy step. The DS-SOP service (capability
> `sop-detection`) consumes the resulting image.

## Build & Standalone-Test the DS-SOP Image (`ds-sop:1.0.0`)

The build-vision-ai SOP profile (microservice **DS-SOP**, capability `sop-detection`)
assumes the image `ds-sop:1.0.0` already exists. This skill **builds it** from source and
**proves it works standalone** — including a model-free smoke test, so you can validate the
image without staging the 14.7 GB SOP models.

> Source of truth: the `NVIDIA/sop-monitoring-blueprints` repo (branch `main`), subtree
> `microservices/sop-inference-bp/` — cloned in **Step 0**. This source ships the **annotated RTSP
> output** (`:8554/ds-out/<stream-name>`, gated by `ENABLE_RTSP_OUTPUT`) that the SOP canonical flow
> relies on — DS-SOP re-streams it for VIOS to record (see `integrate-ds-sop.md`) — via the RTSP
> server code plus the GStreamer apt packages in `docker/Docker.build`. The only patch
> (`ddm_pytorch2.patch`, a PyTorch1→2 fix for the DDM/GEBD model) is applied by `docker/Docker.build`
> during the build.

## When to use
- Build `ds-sop:1.0.0` on a host that lacks it (the SOP profile needs it before deploy).
- Confirm a freshly built image is valid (imports + API server) without models/GPU-heavy load.
- Re-build after a source update.

## Prerequisites
| Item | Why |
|---|---|
| **Git** | clone the build source in Step 0 (`NVIDIA/sop-monitoring-blueprints`, branch `main`) |
| `docker login nvcr.io` | base images `nvcr.io/nvidia/blueprint/vss-engine:2.4.1` + `nvcr.io/nvidia/deepstream:8.0-triton-multiarch` are NGC-gated |
| **BuildKit / buildx** | `Docker.build` uses `RUN --mount=type=bind` (modern Docker has it on by default; else `DOCKER_BUILDKIT=1`) |
| **Internet** | build clones `github.com/MCG-NJU/DDM` @ `941e0fb595ab85dc86724a19ed0439ad6bc3632b` + `gst-plugin-pylon`, and auto-downloads the Basler Pylon SDK (~1.3 GB) |
| ~120 GB free disk | image is ~50 GB + base layers |
| GPU + NVIDIA container runtime | required for **both** steps — the smoke test runs `docker run --gpus all` and its CUDA-linked imports load at startup (a GPU-less host fails at container start); only the **real** run does heavy GPU *compute* (models/inference) |

## Step 0 — Clone the source
The image is **built from source** (no registry image to pull). Clone the repo once —
everything the later steps reference lives inside it:
```bash
git clone https://github.com/NVIDIA/sop-monitoring-blueprints.git
cd sop-monitoring-blueprints
git checkout 0dd472f     # pin — same commit as the VA-MCP patch / kibana dashboard (reproducible image)
cd microservices/sop-inference-bp
```

## Step 1 — Build
```bash
# Step 0 ran in a SEPARATE shell — its `cd` does not persist here. Re-enter the source dir
# (guard so a wrong cwd fails loudly instead of building from the wrong tree):
cd sop-monitoring-blueprints/microservices/sop-inference-bp || { echo "run Step 0 (clone) first" >&2; exit 1; }
mkdir -p binaries          # Docker.build bind-mounts ./binaries; Pylon SDK auto-downloads here at build time
NV_DS_SOP_IMAGE=ds-sop:1.0.0 docker compose -f deploy/compose.yaml build
# manual equivalent:  docker build . -f docker/Docker.build -t ds-sop:1.0.0
```
- Build **as-is** — `docker/Docker.build` already installs the RTSP GStreamer packages
  (`libgstrtspserver-1.0-dev`, `gir1.2-gst-rtsp-server-1.0`, `gstreamer1.0-plugins-ugly`,
  `gstreamer1.0-libav`) and the source carries the RTSP-output code, so **no extra patch** is needed
  for the annotated `:8554/ds-out` output. (The output is gated by `ENABLE_RTSP_OUTPUT`, default off
  — enable it at deploy for the DS-SOP→VIOS flow.)
- Image tag: **`ds-sop:1.0.0`** (set via `NV_DS_SOP_IMAGE`, as in the build command; override for any other name).
- The `pip` dependency-resolver `ERROR` lines (version conflict with the base image's
  `tensorrt-llm`) are **warnings, not build failures**.
- Verify: `docker images ds-sop:1.0.0 --format '{{.Repository}}:{{.Tag}} {{.Size}}'` (~50 GB).

## Step 2 — Standalone smoke test (NO models)
`API_DUMMY_TEST=true` skips `SOPProcessManager`/model load and boots only the API server —
ideal to prove the image is valid (all heavy imports OK + server serves) without any model.
```bash
set -eu
trap 'docker rm -f dssop-smoke >/dev/null 2>&1 || true' EXIT
docker run --rm -d --gpus all --network host \
  -e API_DUMMY_TEST=true -e API_SERVER_PORT=8300 \
  --name dssop-smoke ds-sop:1.0.0
# wait for the heavy imports (torch / vLLM / DeepStream / GstRtspServer) — poll /v1/ready (~30-60 s):
for i in $(seq 1 30); do curl -sf --max-time 5 http://localhost:8300/v1/ready >/dev/null 2>&1 && break; sleep 3; done
curl -sf --max-time 5 http://localhost:8300/v1/ready >/dev/null   # 200 REQUIRED — image boots + serves
for ep in live models metadata; do echo "/v1/$ep:"; curl -sf --max-time 5 http://localhost:8300/v1/$ep; echo; done
# (trap removes the container on exit — including when `set -e` aborts on a failed check)
```
**PASS criteria** (all must respond):
- `/v1/live` → `Service is live.`
- `/v1/ready` → `Dummy test mode,Service is ready.`  (note: no space after the comma — match loosely)
- `/v1/models` → `ds_sop_model`
- `/v1/metadata` → version `1.0.0` + model info (ddm + cosmos) + license

If the image were broken, it would crash on import (DeepStream/`pyds`/vLLM/`GstRtspServer`) before
serving. A clean smoke test = **build correct, container boots, API serves**.
> Note: this does NOT exercise real inference (DDM + Cosmos-Reason VLM are skipped) — that needs
> the real models (Step 3).

## Step 3 — Full standalone run (with models)
Models are NOT in the image — stage them on the host at runtime.

1. **Models — pre-stage them (only *verified* at startup, not auto-pulled):**
   - `/opt/models/vlm/checkpoint` — Cosmos-Reason VLM (must contain `config.json`).
   - `/opt/models/gbed_models/ddm/checkpoint.pth.tar` — DDM checkpoint (not auto-downloadable).
   - `/opt/sop/configs/{actions.json,vlm_prompts.txt}`.
   Obtain by **retraining via the SOP Training Blueprint**, OR
   from NGC `nv-metropolis-dev/vss-industrial/sop-data:1.0` (~14.7 GB; lays the VLM at
   `/opt/models/cosmos-reason1.1-7b/checkpoint`). Set `VLLM_MODEL_PATH` to wherever you stage the VLM.
2. **Test video — pull + transcode:**
   ```bash
   ngc registry resource download-version nvidia/tao/sop-server-fan-installation-data:1.0-260213
   cd sop-server-fan-installation-data_v1.0-260213
   tar -xzf sop-sample-training-data.tar.gz          # → server_fan/raw/Install_1.MP4
   ffmpeg -y -i server_fan/raw/Install_1.MP4 -c:v libx264 -r 30 -an \
     server_fan/raw/Install_1_h264_30fps.mp4         # → h264 1920x1080 30 fps
   ```
3. **Edit `deploy/.env`** REQUIRED paths (`MODEL_ROOT_DIR`, `VLLM_MODEL_PATH`, `DDM_MODEL_PATH` — the
   two model paths must live UNDER `MODEL_ROOT_DIR`, the only bind-mounted folder). For the canonical
   **DS-SOP→VIOS** flow also set **`ENABLE_RTSP_OUTPUT=true`** + `RTSP_PORT=8554` (+ `SW_ENCODER=true`)
   so DS-SOP re-streams the annotated result at `rtsp://<host>:8554/ds-out/<stream-name>`. Then run:
   ```bash
   docker compose -f deploy/compose.yaml up -d
   docker compose -f deploy/compose.yaml logs -f --tail=200 nvds-action-sop
   ```
   Ready when logs show `Uvicorn running on http://0.0.0.0:8300`; `GET /v1/ready` → 200.

## Notes
- This image is consumed by build-vision-ai's DS-SOP service (contract in `references/services/sop/`).
  Container workdir is `/opt/nvidia/nvds_sop`; key env: `API_SERVER_PORT=8300`,
  `DEFAULT_TOPIC=mdx-vlm-captions`, `SOP_MESSAGING_SCHEMA=JSON`, `VLLM_GPU_MEMORY_UTILIZATION=0.3`
  (raise to `0.6` on ≤48 GB GPUs), `ENABLE_MESSAGING=1`, and — for DS-SOP→VIOS — `ENABLE_RTSP_OUTPUT=true`
  + `RTSP_PORT=8554`. The service emits SOP records to Kafka **and** an annotated RTSP output VIOS records.
- Output: `ds-sop:1.0.0` (~50 GB); the Step 2 smoke test exercises all four endpoints.
