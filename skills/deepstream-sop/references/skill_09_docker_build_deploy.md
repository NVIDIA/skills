# § 9 — Docker Build and Deploy

> **Reference files to copy**: `Dockerfile_reference`, `compose_reference.yaml`,
> `copy_sources_reference.sh`, `package_file_list_reference.txt`, `start_server.sh`
> **Related sections**: § 5b (C++ plugin compilation), § 8 (Basler camera runtime usage),
> § 18 (RTSP streaming output — Docker packages, registry cache, compose env) — **optional, opt-in**

---

## Pre-build: Pull Base Image

The Dockerfile uses `nvcr.io/nvidia/blueprint/vss-engine:2.4.1` as the base image.
If it is not already cached locally, pull it before building:

```bash
# Check if base image exists locally
docker images | grep vss-engine

# Pull if missing (requires NGC credentials for nvcr.io)
docker pull nvcr.io/nvidia/blueprint/vss-engine:2.4.1

# Also needed: DeepStream image for header extraction (Stage 1 of Dockerfile)
docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch
```

---

## Pre-build: copy_sources.sh and package_file_list.txt

The Dockerfile copies application sources using `docker/copy_sources.sh`, which reads
a **manifest file** `docker/package_file_list.txt` — it does NOT use rsync or glob.
Each line in the manifest is a relative path that gets copied from the build context
into the container.

**CRITICAL**: Both files must exist and the manifest must list every file needed at
runtime. Copy from references:
- `references/copy_sources_reference.sh` → `docker/copy_sources.sh` (chmod +x)
- `references/package_file_list_reference.txt` → `docker/package_file_list.txt`

Then **update `package_file_list.txt`** to match your actual source files. Every
`.py`, `.cpp`, `Makefile`, config, and test file that the service needs must be
listed. Missing files will cause runtime `ModuleNotFoundError` or `FileNotFoundError`.

---

## RTSP Streaming Output: Build Packages (§ 18)

> **OPTIONAL (opt-in):** add these packages and the registry-cache clear **only when the user
> requested RTSP streaming output (§ 18)**. The default build does not need them; the reference
> `Dockerfile_reference` does **not** include them — skip this apt block and the post-build RTSP
> verification entirely unless RTSP output was requested.

The RTSP output branch needs extra GStreamer packages baked into the image. When RTSP output is
requested, **add** them to the apt block in `docker/Docker.build` (right after
`libgstreamer-plugins-base1.0-dev`):

```dockerfile
    libgstrtspserver-1.0-dev \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
```

followed by a registry-cache clear so the new encode/rtsp plugins are discovered at runtime:

```dockerfile
RUN rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin
```

Verify after build (inside the container):

```bash
python3 -c "import gi; gi.require_version('GstRtspServer','1.0'); from gi.repository import GstRtspServer; print('OK')"
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(bool(Gst.ElementFactory.find('x264enc')))"
```

See § 18 for the full post-build RTSP verification checklist.

---

## Compose env-passthrough (CRITICAL — `COMPOSE_ENV_PASSTHROUGH`)

`docker compose --env-file deploy/.env ...` does **NOT** inject `.env` into the container.
It only uses `.env` to **substitute** `${VAR}` references *inside* `compose.yaml` at parse
time. A variable that the service reads at runtime but is **not** listed under the service's
`environment:` block (and there is no `env_file:` directive) simply **never reaches the
container** — the process then falls back to whatever default is hard-coded in the Python
source, silently:

| Var omitted from `environment:` | In-code default used | Symptom |
|---|---|---|
| `SOP_MESSAGING_SCHEMA` | `NvProtoSchema` (`messager.py`) | Kafka messages are protobuf binary, not the flat `JSON` a Kibana/SOP dashboard expects → dashboard shows no captions |
| `DEFAULT_TOPIC` | `mdx-vlm-captions` | messages land on the wrong topic / index |
| `ENABLE_MESSAGING` | `false` | no Kafka output at all |
| `ENABLE_RTSP_OUTPUT` / `RTSP_PORT` (§ 18) | `true` / `8554` (api_server) | RTSP output config can't be tuned/disabled from `.env` |

**Rule:** every env var the service reads MUST be present in the `environment:` block as an
explicit `KEY: "${KEY:-default}"` passthrough. The `compose_reference.yaml` template already
lists the messaging four (`ENABLE_MESSAGING`, `KAFKA_BROKER`, `SOP_MESSAGING_SCHEMA`,
`DEFAULT_TOPIC`); § 18 adds the RTSP trio. **When `compose.yaml` is adapted from upstream /
canonical DeepStream-SOP source instead of copied from `compose_reference.yaml`, the upstream
file may be missing these — re-add them.** Alternatively add an `env_file: [deploy/.env]`
directive so the whole `.env` is injected.

Verify before deploy (the grep must print every var the service depends on):

```bash
grep -E "SOP_MESSAGING_SCHEMA|DEFAULT_TOPIC|ENABLE_MESSAGING|KAFKA_BROKER" deploy/compose.yaml
# (+ ENABLE_RTSP_OUTPUT|RTSP_PORT|SW_ENCODER when § 18 RTSP output was generated)
```

---

## DDM Source: Exact Git Checkout Required

The Dockerfile clones DDM and applies a PyTorch 2 compatibility patch. The **exact commit**
must be used — other commits may not be compatible with the patch:

```bash
git clone https://github.com/MCG-NJU/DDM.git
git checkout 941e0fb595ab85dc86724a19ed0439ad6bc3632b
git apply docker/ddm_pytorch2.patch
```

`docker/ddm_pytorch2.patch` must be present (copy from `references/ddm_pytorch2.patch`).

---

## Pre-build: Pylon SDK Binary

**CRITICAL**: The `binaries/` directory must exist in the build context and the Pylon
SDK binary must be a **real file** (not a symlink) inside it.

- Keep `binaries/` tracked with a placeholder (e.g. `binaries/.gitkeep`) and git-ignore
  the SDK tarball itself — otherwise a fresh clone has no `binaries/` dir and the Docker
  `--mount=type=bind,source=./binaries` fails with a cryptic **exit code 17** before any
  build step can print a message.
- Docker `--mount=type=bind` does not follow symlinks that point outside the build
  context; a symlink causes `exit code: 23` when the Dockerfile extracts the tarball.

1. **Download Pylon SDK 25.10.2** from https://www.baslerweb.com/en/downloads/software/1932603569/
   (requires free Basler registration) — or use the direct URL set in the Dockerfile ARG:
   `https://downloadbsl.blob.core.windows.net/software/pylon%2025.10.2/pylon-25.10.2_linux-x86_64_setup.tar.gz`

2. **Place the real file** (not a symlink) at `./binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz`

3. **Verify before building:**
   ```bash
   ls -lh binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz
   # Must show a real file (not -> symlink target), ~1.4 GB
   file binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz
   # Must show: gzip compressed data
   ```

If the file is missing, the Dockerfile **stops the build with an actionable message**
(the Pylon SDK is a commercial, license-gated download). To accept the Basler license
terms and let the build `curl` it from `PYLON_SDK_URL`, pass
`--build-arg ALLOW_PYLON_CDN_DOWNLOAD=1` (e.g.
`ALLOW_PYLON_CDN_DOWNLOAD=1 docker compose -f deploy/compose.yaml build`). The CDN
download still fails in air-gapped environments — pre-place the tarball there instead.

**gst-plugin-pylon build conflict**: During the Docker build, DeepStream shared libraries
conflict with the Meson/Ninja build of `gst-plugin-pylon`. The Dockerfile must temporarily
hide the DeepStream lib directory:

```bash
mv /opt/nvidia/deepstream/deepstream/lib /opt/nvidia/deepstream/deepstream/lib.tmp
meson setup builddir && ninja -C builddir && ninja -C builddir install
mv /opt/nvidia/deepstream/deepstream/lib.tmp /opt/nvidia/deepstream/deepstream/lib
```

Without this, the `gst-plugin-pylon` build fails with linker errors referencing the
DeepStream lib path. `gst-plugin-pylon v1.0.0` is cloned automatically from
https://github.com/basler/gst-plugin-pylon if not present in `binaries/gst-plugin-pylon/`.

---

## Build

```bash
# 1. Verify all pre-build steps above are complete, then:
docker compose -f deploy/compose.yaml build
# OR
docker build . -f docker/Docker.build -t nvds-sop:latest

# 2. For a clean rebuild:
docker compose -f deploy/compose.yaml build --no-cache
```

---

## Optional: DDM TensorRT engine (`DDM_TRT_OPTIMIZATION`)

The image installs `onnx==1.17.0 onnxscript==0.7.0 onnxruntime==1.19.2` (Dockerfile) and ships
`scripts/tensorrt/export_ddm_to_tensorrt.py` via `package_file_list.txt`. The TRT path is
**off by default**. Set `DDM_TRT_OPTIMIZATION=true` in `deploy/.env` to enable it: at Triton
init the backend loads a shape/batch-compatible cached engine from `DDM_TRT_ENGINE_OUTPUT_PATH`,
or **builds one on the fly** (~5–15 min the first time), falling back to PyTorch if the build
fails (§ 5). The default cache path is under `/tmp` and rebuilds per container — point
`DDM_TRT_ENGINE_OUTPUT_PATH` under the mounted `MODEL_ROOT_DIR` to persist it across restarts.

---

## Deploy

**IMPORTANT — GPU model paths are REQUIRED before launch.**
The user MUST configure `DDM_MODEL_PATH` and `VLLM_MODEL_PATH` in `deploy/.env`
before starting the service. The defaults point to paths that may not exist on the host.
Always ask the user to confirm these paths before proceeding.

| Variable | Default | Description |
|----------|---------|-------------|
| `DDM_MODEL_PATH` | `/models/gbed_models/ddm/checkpoint.pth.tar` | GEBD checkpoint path (default architecture: DDM); must exist inside `MODEL_ROOT_DIR` mount |
| `VLLM_MODEL_PATH` | _(none — required)_ | Cosmos Reason 1 or 2 — HuggingFace repo id or local path under `MODEL_ROOT_DIR`. No default; startup fails fast if empty |
| `MODEL_ROOT_DIR` | `/models` | Host directory mounted into container; must contain DDM checkpoint at the relative path |

```bash
# Create .env file — user MUST set DDM_MODEL_PATH and VLLM_MODEL_PATH
cat > deploy/.env << 'EOF'
NV_DS_SOP_IMAGE=nvds-sop:latest

# === GPU Model Paths (REQUIRED — no defaults; service fails fast if unset) ===
DDM_MODEL_PATH=/models/gbed_models/ddm/checkpoint.pth.tar
# A fine-tuned checkpoint dir under MODEL_ROOT_DIR, or a Hugging Face repo id.
VLLM_MODEL_PATH=/models/<your-finetuned-vlm-checkpoint>
MODEL_ROOT_DIR=/models

HOST_CACHE=$HOME/.cache/ds_sop
MEDIA_STORAGE_DIR=/tmp/nvds_sop_storage
NVIDIA_VISIBLE_DEVICES=0
EOF

# Verify model paths before launch
ls -lh /models/gbed_models/ddm/checkpoint.pth.tar  # DDM checkpoint must exist

# Start services
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
```

### Container privilege posture & optional env

`deploy/compose.yaml` runs the service **unprivileged** by default. It uses `network_mode: host`
and `ipc: host` for low-latency RTSP/Basler streaming and Pylon's GenICam shared-memory transport
(GPU access comes from `runtime: nvidia`); override `NETWORK_MODE` / `IPC_MODE` to tighten those.
Enabling `privileged` mode (e.g. in a `compose.override.yaml`) can resolve extra device / file /
shared-memory access issues, but use it with caution.

Optional `.env` additions:
```bash
# DDM TensorRT (optional — see § 5)
DDM_TRT_OPTIMIZATION=false
DDM_TRT_ENGINE_OUTPUT_PATH=/tmp/trt_opt/ddm.engine   # put under MODEL_ROOT_DIR to persist
DDM_TRT_PRECISION=fp32                                # fp32 | fp16 | bf16

# Basler camera / emulation (optional — see § 8, § 11)
PYLON_CAMEMU=1
CAMERA_EMULATION_DIR=/path/to/emulation/assets       # mounted to /opt/nvidia/nvds_sop/streams/simulation
CAMERA_NUM_BUFFERS=                                   # EOS after N frames (emulation only)
```

---

## Waiting for Service Ready

**Model loading takes several minutes on first run** — VLM model and DDM initialises,
and a warmup inference runs. Do NOT assume the service is ready after a fixed timeout.

Poll `/v1/ready` every 10 seconds and watch `docker compose logs` in parallel:

```bash
for i in $(seq 1 60); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8300/v1/ready)
  echo "[$(date +%H:%M:%S)] attempt $i: HTTP $STATUS"
  [ "$STATUS" = "200" ] && echo "Service is ready." && break
  docker compose -f deploy/compose.yaml logs --tail=5 --no-log-prefix nvds-action-sop 2>&1
  sleep 10
done
```
