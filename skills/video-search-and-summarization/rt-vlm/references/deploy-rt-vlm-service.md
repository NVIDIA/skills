# Deploy RT-VLM Service

## 1. Overview

**Service**: `rtvi-vlm` (container name `rtvi-vlm`)
**Image (x86 / Jetson-Tegra)**: `nvcr.io/nvidia/vss-core/vss-rt-vlm:3.1.0` (multiarch)
**Image (SBSA / DGX Spark / Grace)**: `nvcr.io/nvidia/vss-core/vss-rt-vlm:3.1.0-sbsa`
**Primary port**: `${RTVI_VLM_PORT}` → container `8000` (FastAPI REST, `/v1`)
**Validated GPUs**: H100 · RTX PRO 6000 Blackwell · L40S · DGX SPARK · IGX Thor · AGX Thor

Real-Time VLM is VSS's streaming vision-language inference service: RTSP decode →
segmentation → VLM inference (vLLM) → Kafka publication (NvSchema protobuf).
In this compose, rtvi-vlm is wired by default to call a **sibling NIM**
(`cosmos-reason1-7b`, `cosmos-reason2-8b`, or `qwen3-vl-8b-instruct`) over
OpenAI-compat HTTP (`VLM_MODEL_TO_USE=openai-compat`). **Kafka lives on the
host**, not in-compose (`KAFKA_BOOTSTRAP_SERVERS=${HOST_IP}:9092`).

## 2. Related Skill

The top-level `skills/rt-vlm/SKILL.md` file covers the VSS 3.1 API
(`/v1/generate_captions_alerts`, `/v1/files`, `/v1/streams/add`,
`/v1/chat/completions`, Kafka topics, and the four standard workflows). This
reference answers "how do I deploy / debug rtvi-vlm?"; the top-level skill
answers "how do I call rtvi-vlm?". Hit `http://localhost:${RTVI_VLM_PORT}/docs`
(FastAPI auto-docs) or `GET /openapi.json` on the running service for the
live-authoritative schema — see §16.

## 3. Prerequisites

- **Docker Engine 28.2+** + Compose plugin **2.36+** (this compose uses
  `${VAR:+:path}` conditional-bind syntax that older Compose rejects)
- **NVIDIA Driver 580+** + NVIDIA Container Toolkit
  (`docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` must succeed)
- **Git LFS** (HF-backed models)
- **≥ 50 GB disk** for image + 20–80 GB for model weights on first run
- **Kafka on host** reachable at `${HOST_IP}:9092` (compose does NOT bundle Kafka)
- **Sibling NIM compose** providing the VLM backend: rtvi-vlm `depends_on`
  `cosmos-reason1-7b` / `cosmos-reason2-8b` / `qwen3-vl-8b-instruct`, all
  `required: false`. Launch one of those first.
- **`MDX_DATA_DIR`** host path — compose bind-mounts
  `${MDX_DATA_DIR}/data_log/vst/clip_storage` with no default → mount breaks if unset
- **Free port**: `${RTVI_VLM_PORT}` (whatever you pick)
- **Outbound**: `nvcr.io`, `huggingface.co`, any remote NIM/OpenAI endpoints

> ⚠️ **Profiles are mandatory.** Service declares **5 blueprint profiles**
> (§12). Plain `docker compose up` starts **nothing** — pass `--profile <name>`.

## 4. NGC / Registry Preflight

```bash
# Obtain an NGC key: https://ngc.nvidia.com/setup/api-key
export NGC_CLI_API_KEY="<YOUR_NGC_KEY>"
echo "$NGC_CLI_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Verify pull for the exact production image
docker pull "nvcr.io/nvidia/vss-core/vss-rt-vlm:3.1.0"
```

> ⚠️ **`docker compose pull` fails on standalone deployments** (recent Docker
> Compose): the compose file's `depends_on` references sibling NIM services
> that are not defined in this single-file project. Compose rejects this as
> `invalid compose project` at project-load time even when every reference is
> `required: false`, and `--no-deps` does NOT bypass project validation. Use
> `docker pull` directly (above) to warm the image cache instead.

## 5. Required Secrets & Credentials

All values are `${VAR:-}` placeholders — nothing hardcoded. Use a gitignored `.env`.

Host-side vars in this compose use the `RTVI_VLM_*` / `RTVI_VLLM_*` prefix and
rewrite to canonical container-side names at the compose boundary.

| Host env var | → Container env | Purpose | Where to get |
|---|---|---|---|
| `NGC_CLI_API_KEY` or `RTVI_VLM_API_KEY` | `NGC_API_KEY` + `VIA_VLM_API_KEY` | Image pull + NIM backend auth | <https://ngc.nvidia.com/setup/api-key> |
| `HF_TOKEN` | `HF_TOKEN` | Gated HF models (Qwen3-VL) | <https://huggingface.co/settings/tokens> |
| `NVIDIA_API_KEY` | `NVIDIA_API_KEY` | Generic NVIDIA API (defaults to `NOAPIKEYSET`) | NVIDIA dev portal |
| `OPENAI_API_KEY` | `OPENAI_API_KEY` | OpenAI directly when `openai-compat` (defaults `NOAPIKEYSET`) | <https://platform.openai.com/api-keys> |
| `OPENAI_API_VERSION` | `OPENAI_API_VERSION` | Azure OpenAI version pin | — |
| `RTVI_VLM_ENDPOINT` | `VIA_VLM_ENDPOINT` | Custom OpenAI-compat endpoint URL | Your backend |
| `RTVI_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` | `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` | Remote model name | Your backend |
| `REDIS_PASSWORD` | `REDIS_PASSWORD` | Only when `ENABLE_REDIS_ERROR_MESSAGES=true` | Your Redis |

> ⚠️ **Minimum to boot**: `NGC_CLI_API_KEY` + whatever the sibling NIM needs.

Use the `.env` block in §12 as the starting point.

## 6. Required Volume Mounts

| Compose line | Spec | Stateful? | `down -v` destroys? |
|---|---|---|---|
| 108 | `${ASSET_STORAGE_DIR:-/dummy}${ASSET_STORAGE_DIR:+:/tmp/assets}` (optional bind over tmpfs) | yes (if set) | yes (host bind) |
| 109 | `${RTVI_VLM_HF_CACHE:-rtvi-hf-cache}:/tmp/huggingface` (named by default, multi-GB) | **yes** | **YES — multi-GB re-download** |
| 110 | `${MDX_DATA_DIR}/data_log/vst/clip_storage:/home/vst/vst_release/streamer_videos` — **no default → required** | yes | yes (host bind) |
| 111 | `${NGC_MODEL_CACHE:-rtvi-ngc-model-cache}:/opt/nvidia/rtvi/.rtvi/ngc_model_cache` (named) | **yes** | **YES — re-download weights** |
| 112 | `${RTVI_VLM_LOG_DIR:-/dummy}${RTVI_VLM_LOG_DIR:+:/opt/nvidia/rtvi/log/rtvi/}` (optional bind) | no | no |

**Required host-path setup** — `MDX_DATA_DIR` is not optional:

```bash
# Container runs as UID 1001; host dir must be writable by that UID
export MDX_DATA_DIR=/abs/path/to/mdx-data
mkdir -p "$MDX_DATA_DIR/data_log/vst/clip_storage"
sudo chown -R 1001:1001 "$MDX_DATA_DIR/data_log/vst/clip_storage"
# No sudo? Grant UID 1001 via ACL:
#   sudo setfacl -R -m u:1001:rwx "$MDX_DATA_DIR/data_log/vst/clip_storage"
# Avoid `chmod 777` — exposes clip storage to every user on the host.
```

Optional host-path overrides:

```bash
mkdir -p ./rtvi-assets && sudo chown 1001:1001 ./rtvi-assets
# .env: ASSET_STORAGE_DIR=$(pwd)/rtvi-assets

mkdir -p ./rtvi-logs && sudo chown 1001:1001 ./rtvi-logs
# .env: RTVI_VLM_LOG_DIR=$(pwd)/rtvi-logs
```

> ⚠️ `docker compose down -v` wipes `rtvi-hf-cache` + `rtvi-ngc-model-cache` →
> **20–80 GB re-download** on next up.

## 7. Required Environment Variables

| Host var | Required | Compose default | Notes |
|---|---|---|---|
| `RTVI_VLM_PORT` | **YES** (`${RTVI_VLM_PORT?}` strict) | — | Host REST API port |
| `HOST_IP` | **YES (effectively)** | — | Interpolated into `KAFKA_BOOTSTRAP_SERVERS=${HOST_IP}:9092`; no fallback |
| `MDX_DATA_DIR` | **YES (effectively)** | — | Interpolated into VST clip-storage bind mount; no fallback |
| `NGC_CLI_API_KEY` | **YES** | — | Image pull + NIM auth |
| `VLM_MODEL_TO_USE` (via `RTVI_VLM_MODEL_TO_USE`) | effectively required | `openai-compat` | `cosmos-reason1` / `cosmos-reason2` / `openai-compat` / `custom` |
| `MODEL_PATH` (via `RTVI_VLM_MODEL_PATH`) | conditional | `ngc:nim/nvidia/cosmos-reason2-8b:hf-1208` | Needed when not `openai-compat`. **Override to `:1208-fp8-static-kv8`** — this is the tag VSS docs and NIM sibling composes serve; the compose default `:hf-1208` is a different quant variant. See §20. |

The most important host-side variables use the `RTVI_VLM_*` or `RTVI_VLLM_*`
prefix and are rewritten to canonical container-side names by compose.

## 8. Optional / Feature-Flag Environment Variables

- **vLLM tuning** (compose defaults): `VLLM_MAX_NUM_SEQS=256`,
  `VLLM_MAX_NUM_BATCHED_TOKENS=5120`, `VLM_MAX_MODEL_LEN=32768`,
  `VLLM_NUM_SCHEDULER_STEPS=8`, `VLLM_ENABLE_PREFIX_CACHING=true`,
  `VLLM_GPU_MEMORY_UTILIZATION=""` (auto-tuned)
- **Feature toggles**: `ENABLE_OTEL_MONITORING=false`,
  `INSTALL_PROPRIETARY_CODECS=false`, `FORCE_SW_AV1_DECODER=""`,
  `VSS_SKIP_INPUT_MEDIA_VERIFICATION=""`, `ENABLE_REDIS_ERROR_MESSAGES=false`,
  `RTVI_ADD_TIMESTAMP_TO_VLM_PROMPT=""`
- **Auto-tuned by entrypoint** (override only when needed):
  `VLM_BATCH_SIZE`, `NUM_GPUS`, `VLLM_GPU_MEMORY_UTILIZATION`
  (auto-set to `0.7` when VRAM ≤ 50 GB)

## 9. GPU Selection & Hardware

```yaml
# compose line 40
device_ids: ["${RT_VLM_DEVICE_ID:-0}"]
```

> **Note:** `RT_VLM_DEVICE_ID` breaks the `RTVI_VLM_*` pattern because this name
> is fixed by the upstream `met-blueprints` compose — don't rename locally.

Plus `NVIDIA_VISIBLE_DEVICES=${RTVI_VLM_NVIDIA_VISIBLE_DEVICES:-all}`.

```bash
RT_VLM_DEVICE_ID=0                   # by index
RT_VLM_DEVICE_ID=GPU-abc123...       # by UUID (from `nvidia-smi -L`)
```

**Jetson Thor / DGX Spark caveat**: docs note instability at 8+ vision tokens
concurrent — cap at ≤2 streams or drop input resolution.

## 10. Port Conflict Map

| Container port | Host port | Collision risk |
|---|---|---|
| `8000` | `${RTVI_VLM_PORT}` | Many NVIDIA NIMs also bind 8000 — pick an unused port in `.env` |

Kafka and Redis are **not bundled** — expected on host or in a sibling compose.

## 11. Models Used & Swap Guide

Set `RTVI_VLM_MODEL_TO_USE` in `.env` to select the backend. After any change:

```bash
sudo docker compose -f rtvi-vlm-docker-compose.yml \
  --profile bp_developer_alerts_2d_vlm up -d --force-recreate rtvi-vlm
```

Verify what loaded: `curl -s "http://localhost:${RTVI_VLM_PORT}/v1/models" | jq`

---

### Option A — Remote NIM endpoint (openai-compat)

Point rtvi-vlm at an already-running NIM (sibling container, remote host, or
NVIDIA API Catalog):

```bash
# .env:
RTVI_VLM_MODEL_TO_USE=openai-compat
RTVI_VLM_ENDPOINT=http://<nim-host>:8000/v1
RTVI_VLM_OPENAI_MODEL_DEPLOYMENT_NAME=cosmos-reason2-8b   # model name the NIM exposes
RTVI_VLM_API_KEY=<ngc-or-nim-token>
```

---

### Option B — OpenAI / Azure OpenAI

```bash
# .env:
RTVI_VLM_MODEL_TO_USE=openai-compat
RTVI_VLM_ENDPOINT=https://api.openai.com/v1           # or Azure endpoint
RTVI_VLM_OPENAI_MODEL_DEPLOYMENT_NAME=gpt-4o          # or Azure deployment name
RTVI_VLM_API_KEY=sk-...                               # OpenAI key
OPENAI_API_KEY=sk-...                                 # some code paths read this directly
# Azure only:
# OPENAI_API_VERSION=2024-02-01
```

---

### Option C — Self-hosted NGC NIM (cosmos-reason1 or cosmos-reason2)

Model is downloaded and served by vLLM inside the container. Requires ~16–20 GB
VRAM for the 8B models.

```bash
# .env for cosmos-reason2 (recommended tag — matches VSS docs / NIM siblings):
RTVI_VLM_MODEL_TO_USE=cosmos-reason2
RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8
NGC_CLI_API_KEY=<ngc-key>

# .env for cosmos-reason1:
RTVI_VLM_MODEL_TO_USE=cosmos-reason1
RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason1-7b:hf-1208
NGC_CLI_API_KEY=<ngc-key>
```

---

### Option D — HuggingFace model (vLLM-compatible)

`VLM_MODEL_TO_USE=vllm-compatible` is the correct value for any HF-hosted or
locally-served vLLM-compatible model. Reference:
https://docs.nvidia.com/vss/latest/real-time-vlm.html#hugging-face-models-locally

```bash
# .env — authenticate via HF_TOKEN env var:
RTVI_VLM_MODEL_TO_USE=vllm-compatible
RTVI_VLM_MODEL_PATH=git:https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
HF_TOKEN=hf_...
```

Avoid embedding HF tokens directly in model URLs; keep them in `HF_TOKEN` so
resolved config and logs do not contain credentials.

Validated model: `Qwen/Qwen3-VL-30B-A3B-Instruct`. Other Qwen3-VL sizes work
but are not officially validated.

---

### Option E — Custom NGC artifact or local vLLM-compatible model

For a custom NGC artifact, use `cosmos-reason2` (same NGC NIM loader):

```bash
RTVI_VLM_MODEL_TO_USE=cosmos-reason2
RTVI_VLM_MODEL_PATH=ngc:<org>/<team>/<model>:<version>
NGC_CLI_API_KEY=<ngc-key>
```

For a local directory containing a vLLM-compatible model, use `vllm-compatible`
and mount the host path into the container:

```bash
# .env:
RTVI_VLM_MODEL_TO_USE=vllm-compatible
RTVI_VLM_MODEL_PATH=/opt/models/my-vlm          # path inside the container
```

> Note: `RTVI_VLM_MODEL_IMPLEMENTATION_PATH` (`MODEL_IMPLEMENTATION_PATH` inside
> the container) is present in the compose env mapping but its behavior for
> custom local models is not documented — omit it unless you have confirmed it
> works for your use case.

Add the bind mount to the compose `volumes:` section:
```yaml
volumes:
  - /host/path/to/models:/opt/models:ro
```

## 12. Deploy

This compose declares **5 blueprint profiles**. Service will NOT start under
plain `docker compose up` — `--profile <name>` is required.

| Profile | Intended use |
|---|---|
| `bp_developer_alerts_2d_vlm` | Alerts blueprint (2D, VLM-only) |
| `bp_developer_alerts_2d_cv_IGX-THOR` | Alerts (2D + CV) on IGX Thor |
| `bp_developer_base_2d_IGX-THOR` | Base 2D on IGX Thor |
| `bp_developer_alerts_2d_cv_AGX-THOR` | Alerts (2D + CV) on AGX Thor |
| `bp_developer_base_2d_AGX-THOR` | Base 2D on AGX Thor |

Generic VLM workflow → `bp_developer_alerts_2d_vlm`.

```bash
# 0. Fetch compose (if not in a met-blueprints checkout)
mkdir -p /work/rtvi_deploy && cd /work/rtvi_deploy
wget -q -O rtvi-vlm-docker-compose.yml \
  "https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/video-search-and-summarization/refs/heads/3.1.0/deployments/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml"

# 0a. Detect platform → select correct image tag
#     x86_64 and Tegra-based Jetson/AGX/IGX Thor use the multiarch image.
#     SBSA server-ARM (DGX Spark, Grace Hopper) requires the -sbsa variant.
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  VLM_TAG="3.1.0"
elif [ "$ARCH" = "aarch64" ]; then
  if grep -qi tegra /proc/cpuinfo 2>/dev/null || [ -f /etc/nv_tegra_release ]; then
    VLM_TAG="3.1.0"         # Jetson / AGX Thor / IGX Thor (Tegra)
  else
    VLM_TAG="3.1.0-sbsa"    # DGX Spark / Grace Hopper (SBSA server-ARM)
  fi
else
  echo "Unsupported architecture: $ARCH" && exit 1
fi
echo "Platform: $ARCH → image tag: $VLM_TAG"

# 0b. Standalone fix — recent Docker Compose rejects `depends_on` references to
#     sibling NIMs that aren't defined in this single-file project, even with
#     `required: false`. Strip the depends_on block for standalone deploys.
#     Use yq if available (handles YAML correctly), otherwise fall back to a
#     small stdlib-only Python edit of this known compose file:
if command -v yq >/dev/null; then
  yq -i 'del(.services.rtvi-vlm.depends_on)' rtvi-vlm-docker-compose.yml
else
  python3 - <<'PY'
from pathlib import Path

p = Path("rtvi-vlm-docker-compose.yml")
out = []
skip = False
base_indent = 4
for line in p.read_text().splitlines():
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if not skip and line.startswith("    depends_on:"):
        skip = True
        continue
    if skip:
        if stripped and indent <= base_indent:
            skip = False
            out.append(line)
        continue
    out.append(line)
p.write_text("\n".join(out) + "\n")
PY
fi
#     Verify it's gone (should print 0):
grep -c 'depends_on' rtvi-vlm-docker-compose.yml

# 1. Config — set model vars per §11 (Options A–E)
cat > .env <<EOF
NGC_CLI_API_KEY=<your-ngc-key>
RTVI_VLM_PORT=8100
HOST_IP=<host-ip>
MDX_DATA_DIR=/work/rtvi_deploy/mdx-data
RTVI_VLM_IMAGE_TAG=${VLM_TAG}
RT_VLM_DEVICE_ID=0
# Model config (choose one option from §11):
RTVI_VLM_MODEL_TO_USE=cosmos-reason2
RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8
EOF

# 2. Prepare VST clip-storage host dir (required)
mkdir -p "$MDX_DATA_DIR/data_log/vst/clip_storage"
sudo chown -R 1001:1001 "$MDX_DATA_DIR/data_log/vst/clip_storage"

# 3. NGC auth — if running docker via sudo, pass the key inline (sudo drops env vars)
echo "<your-ngc-key>" | sudo docker login nvcr.io -u '$oauthtoken' --password-stdin
# Or preserve env: sudo --preserve-env=NGC_CLI_API_KEY bash -c \
#   'echo "$NGC_CLI_API_KEY" | docker login nvcr.io -u $oauthtoken --password-stdin'

# 4. Pull image directly (docker compose pull fails on standalone — see §4)
sudo docker pull "nvcr.io/nvidia/vss-core/vss-rt-vlm:${VLM_TAG}"

# 5. Bring up — plain `up` (no profile) starts nothing
sudo --preserve-env=NGC_CLI_API_KEY \
  docker compose -f rtvi-vlm-docker-compose.yml \
  --profile bp_developer_alerts_2d_vlm up -d

# 6. Wait for healthy — start_period is 1200s (20 MIN) on first boot.
#    Model weight download + vLLM warmup can take the full window.
#    Do NOT kill as "stuck" before 20 minutes have elapsed.
until [ "$(sudo docker compose -f rtvi-vlm-docker-compose.yml ps --format json rtvi-vlm \
  | jq -r '[.[].Health] | all(. == "healthy")')" = "true" ]; do
  echo "waiting for rtvi-vlm… (up to 20 minutes on first run)"
  sleep 15
done

# 7. Verify
curl -f "http://localhost:${RTVI_VLM_PORT}/v1/health/ready"
```

## 13. Dry Run

```bash
cd deployments/rtvi/rtvi-vlm

# Resolved compose (audit; --no-interpolate keeps ${VAR} literal — no secrets leaked)
docker compose -f rtvi-vlm-docker-compose.yml \
  --profile bp_developer_alerts_2d_vlm config --no-interpolate

# Validation only
docker compose -f rtvi-vlm-docker-compose.yml \
  --profile bp_developer_alerts_2d_vlm config --quiet && echo "compose valid"

# Create containers + pull + volumes, but don't start
docker compose -f rtvi-vlm-docker-compose.yml \
  --profile bp_developer_alerts_2d_vlm up --no-start

# Cleanup
docker compose -f rtvi-vlm-docker-compose.yml down
```

> Note: compose uses `${VAR:+:path}` conditional-bind on `ASSET_STORAGE_DIR` and
> `RTVI_VLM_LOG_DIR`. Older Compose (<2.36) rejects `config` with "too many
> colons". `up` works regardless; only `config` fails. Upgrade Compose.

## 14. Verify Deployment

```bash
# Health
curl -f "http://localhost:${RTVI_VLM_PORT}/v1/health/ready"

# Loaded model
curl -s "http://localhost:${RTVI_VLM_PORT}/v1/models" | jq

# OpenAPI spec (FastAPI auto-docs)
curl -s "http://localhost:${RTVI_VLM_PORT}/openapi.json" | jq '.paths | keys'
```

Healthy log signatures (`docker logs rtvi-vlm`):
- `Auto-selecting VLM Batch Size to <N>`
- `Free GPU memory is <N> MiB`
- `Using <VLM_MODEL_TO_USE>`
- `RTVI Server loaded`
- `Backend is running at http://0.0.0.0:<port>`

## 15. Logs & Status

```bash
docker compose -f rtvi-vlm-docker-compose.yml ps

# By container name (compose sets container_name: rtvi-vlm)
docker logs -f rtvi-vlm

# Or by service via compose
docker compose -f rtvi-vlm-docker-compose.yml logs -f rtvi-vlm
docker compose -f rtvi-vlm-docker-compose.yml logs --tail 200 --since 10m rtvi-vlm

docker stats rtvi-vlm
nvidia-smi dmon -s u
```

Verbosity: set `RTVI_VLM_LOG_LEVEL=DEBUG` (DEBUG/INFO/WARNING/ERROR) and
`up -d --force-recreate rtvi-vlm`. Host-persisted logs at `${RTVI_VLM_LOG_DIR}`
when set.

## 16. API Usage (from real-time-vlm-api.html)

**Base URL**: `http://<host>:${RTVI_VLM_PORT}/v1`
**Auth**: `Authorization: Bearer <token>` (when token gating is enabled)

Documented endpoint categories (full schemas via `/openapi.json` or `/docs`
once the service is up):

| Category | Purpose |
|---|---|
| Health Check | `/v1/health/ready` — readiness probe; used by Docker healthcheck |
| Captions | Generate VLM captions and alerts for videos and live streams |
| Files | Upload and manage video/image files |
| Live Stream | Add, list, and manage RTSP live streams |
| Models | `/v1/models` — list available VLM models |
| Metrics | Prometheus metrics endpoint |
| Metadata | Service metadata and version info |
| NIM Compatible | OpenAI-compatible endpoints for interop |

> ⚠️ **Docs API page is a landing page only** — concrete paths, request/response
> schemas, and error codes were not retrievable from the upstream HTML.
> `GET /openapi.json` on the running service is authoritative for specifics.

## 17. Debugging Common Failures

| Symptom | Root cause | Fix |
|---|---|---|
| `docker compose up` starts nothing | `--profile` not specified | Add `--profile bp_developer_alerts_2d_vlm` (§12) |
| `Exited (1)` immediately, logs mention `RTVI_VLM_PORT` | Strict sentinel fired | Set `RTVI_VLM_PORT` in `.env` |
| Container starts but Kafka errors `:9092 connection refused` | `HOST_IP` unset → `KAFKA_BOOTSTRAP_SERVERS=:9092` | Set `HOST_IP` to an address reachable from the container. Non-fatal for API/inference — Kafka publishing is just disabled. |
| Volume mount error mentioning `data_log/vst/clip_storage` | `MDX_DATA_DIR` unset → malformed mount | Set `MDX_DATA_DIR`; pre-create the `data_log/vst/clip_storage` subtree |
| `service "X" depends on undefined service "Y": invalid compose project` | Recent Docker Compose rejects `depends_on` refs to sibling NIM services not defined in this single-file project — even with `required: false`. `--no-deps` does NOT bypass this validation. | Remove the `depends_on` block from the local compose copy (§12 step 0b). Only needed for standalone deploys without the full met-blueprints project. |
| `docker compose pull` → `invalid compose project` | Same `depends_on` validation runs before pull | Use `docker pull nvcr.io/nvidia/vss-core/vss-rt-vlm:<tag>` directly (§4) |
| `password is empty` on `sudo docker login` | `sudo` drops the user's environment — `$NGC_CLI_API_KEY` is not set in the sudo shell | Pass the key inline: `echo "<key>" \| sudo docker login nvcr.io -u '$oauthtoken' --password-stdin`, or use `sudo --preserve-env=NGC_CLI_API_KEY` |
| `unauthorized` on `docker compose pull` | Missing NGC auth or no org access | `docker login nvcr.io` with a key that has `nvidia/vss-core` access |
| `Exited (1)` "Error: No GPUs were found" | Container can't see GPUs | Install NVIDIA Container Toolkit; `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` must work |
| `Exited (137)` OOM | VRAM pressure | Lower `RTVI_VLLM_GPU_MEMORY_UTILIZATION`; drop `RTVI_VLLM_MAX_NUM_SEQS` below 256; bigger GPU via `RT_VLM_DEVICE_ID`; drop `RTVI_VLM_MAX_MODEL_LEN` |
| First `up` hangs 10+ min | Model weight download + vLLM warmup | Expected: `start_period: 1200s`. Watch `docker logs` for NIM progress; don't kill before 20 min. |
| Device reboot on Jetson Thor / DGX Spark at 8+ vision tokens | Known issue (docs) | Cap at ≤2 concurrent streams or drop resolution |
| Stream deletion lags under heavy load | VLM inference exceeds chunk duration (docs — expected) | Reduce concurrent streams |

## 18. Upgrade & Rollback

**Forward**:
```bash
# .env: RTVI_VLM_IMAGE_TAG=<new-tag>
docker compose -f rtvi-vlm-docker-compose.yml --profile <p> pull rtvi-vlm
docker compose -f rtvi-vlm-docker-compose.yml --profile <p> up -d --force-recreate rtvi-vlm
```

**Rollback**:
```bash
# Record current tag first: `docker compose -f ... images rtvi-vlm`
# .env: RTVI_VLM_IMAGE_TAG=<prior-tag>
docker compose -f rtvi-vlm-docker-compose.yml --profile <p> pull rtvi-vlm
docker compose -f rtvi-vlm-docker-compose.yml --profile <p> up -d --force-recreate rtvi-vlm
```

Named volumes survive both. Re-download only if `MODEL_PATH` changes.

## 19. Tear Down

```bash
cd deployments/rtvi/rtvi-vlm

# Keep named volumes (model caches preserved)
docker compose -f rtvi-vlm-docker-compose.yml --profile bp_developer_alerts_2d_vlm down

# WIPES model caches (20–80 GB re-download)
docker compose -f rtvi-vlm-docker-compose.yml --profile bp_developer_alerts_2d_vlm down -v

# Remove locally-pulled image
docker compose -f rtvi-vlm-docker-compose.yml down --rmi local

# Optional host-side (do NOT rm $MDX_DATA_DIR — shared with other services)
# rm -rf ./rtvi-assets ./rtvi-logs
```

## 20. Gotchas & Known Issues

- **🟢 Docs list `/v1/ready` for health, but the real endpoint is `/v1/health/ready`** — which is what the compose healthcheck already uses. Trust the compose, not the docs.
- **🟢 Healthcheck tuning divergence**: docs show `start_period: 300s`,
  `retries: 3`; compose sets `1200s` / `5`. The compose values are
  deliberately more lenient for model-download-on-first-boot. Not a bug.
- **🔴 Default MODEL_PATH tag divergence — always override**: compose default
  is `ngc:nim/nvidia/cosmos-reason2-8b:hf-1208`, but the tag that matches VSS
  docs and sibling NIM composes is **`:1208-fp8-static-kv8`** (FP8-static
  weights + KV8 cache). Use that tag unless you have a specific reason to run
  the `hf-1208` variant. The two are **not interchangeable** — different quant
  schemes produce different `torch_aot_compile` cache hashes, so swapping tags
  on a live cache volume will emit the `_Missing has no attribute _modules`
  warning and force a full vLLM recompile on first boot.
- **Profiles are mandatory**: `docker compose up` without `--profile` starts
  nothing. 5 profiles available — §12.
- **`container_name: rtvi-vlm` hardcoded** (line 22) — can't run two instances
  on the same host without editing. Second `up` fails with
  `Conflict. The container name "/rtvi-vlm" is already in use`.
- **Long `start_period` (1200s = 20 min)**: first boot downloads model weights
  and warms vLLM. Pre-warn operators not to kill as stuck.
- **`depends_on.required: false` is NOT enough on recent Docker Compose**: Compose
  validates all `depends_on` service references at project load time and rejects
  them with `invalid compose project` if the services aren't defined — regardless
  of `required: false`. `--no-deps` does not bypass this. For standalone
  deployments (no full met-blueprints project), strip the `depends_on` block
  from the local compose copy (§12 step 0b). The `required: false` behavior
  works correctly only when running under the full met-blueprints multi-file
  project where all sibling services are defined.
- **`sudo docker` drops environment variables**: `NGC_CLI_API_KEY` and other
  vars set in the user shell are invisible to `sudo docker`. Pass secrets inline
  (`echo "<key>" | sudo docker login ...`) or use `sudo --preserve-env=VAR_NAME`.
- **External Kafka required**: `KAFKA_BOOTSTRAP_SERVERS=${HOST_IP}:9092` — if
  `HOST_IP` isn't set, the container tries `:9092` and fails.
  `host.docker.internal` is wired via `extra_hosts` as an alternative value.
- **`MDX_DATA_DIR` required**: no default on the bind mount. Without it the
  mount spec expands to garbage.
- **Host-var rewrite convention**: most host-side vars use `RTVI_VLM_*` or
  `RTVI_VLLM_*` and rewrite to canonical names inside the container.
- **`VLM_MODEL_TO_USE=openai-compat` by default**: this stack expects a sibling
  NIM on the same network, not a self-hosted vLLM. Standalone operation
  requires `RTVI_VLM_ENDPOINT` or switching to `cosmos-reason2` + `MODEL_PATH`.
- **Parser volume-split warnings**: the compose's `${VAR:-default}:path` mount
  syntax trips the pyyaml-fallback parser's colon-splitting heuristic. Re-read
  the raw compose (§6 cites the raw text). `up` is unaffected.
- **Docs gaps**: VSS docs cover Deploy + Troubleshoot but NOT tear-down,
  rollback, or backup/restore. §18–19 derive from Compose conventions.

## 21. References

- **Deploy docs**: <https://docs.nvidia.com/vss/latest/real-time-vlm.html>
- **API docs**: <https://docs.nvidia.com/vss/latest/real-time-vlm-api.html>
  (landing page only — see `/openapi.json` on the running service for specifics)
- **Compose (met-blueprints checkout)**: `deployments/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
- **Compose (raw, 3.1.0)**: `https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/video-search-and-summarization/refs/heads/3.1.0/deployments/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
