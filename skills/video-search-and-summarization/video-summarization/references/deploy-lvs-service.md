# Deploy LVS Service - Long Video Summarization Blueprint

## Contents

- 1. Overview
- 2. Related Skill Entry Point
- 3. Prerequisites
- 4. NGC / Registry Preflight
- 5. Required Secrets & Credentials
- 6. Required Volume Mounts
- 7. Required Environment Variables
- 8. GPU Selection & Hardware
- 9. Port Map
- 10. Models & Swap Guide
- 11. Deploy
- 12. Dry Run
- 13. Verify Deployment
- 14. Logs & Status
- 15. Debugging Common Failures
- 16. Upgrade & Rollback
- 17. Tear Down
- 18. Gotchas & Known Issues
- 19. Discrepancies Between This Compose And The Public Docs
- 20. References

## 1. Overview

> **Service**: `lvs-server` (container name `vss-lvs`)
> **Image**: `nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f`
> **Compose profile**: `bp_developer_lvs_2d` (required — nothing starts without it)
> **Health endpoint**: `http://<HOST_IP>:${BACKEND_PORT:-38111}/v1/ready`
> **Primary ports** (host network): `38111` REST API, `38112` MCP/SSE, `38113` frontend
> **Hardware**: ≥1 NVIDIA GPU. VRAM ≤50 GB triggers `VLLM_GPU_MEMORY_UTILIZATION=0.7`
> auto-tune in the entrypoint.

LVS orchestrates VLM-based caption generation and LLM-based summarization over
long videos, backed by Elasticsearch (default) or Milvus for event storage. This
compose ships the **single `lvs-server` container** plus a catalog of 16
`depends_on` NIM sidecar services (LLM + VLM), all declared `required: false`
— they only start when **their own profiles** are also activated. In this
blueprint you point LVS at pre-running external LLM/VLM NIMs via
`LVS_LLM_BASE_URL` / `VLM_BASE_URL` or their `HOST_IP:<port>` equivalents.

## 2. Related Skill Entry Point

The `video-summarization` skill owns the user-facing summarization workflow.
Its [`lvs-api.md`](lvs-api.md) reference owns direct LVS API usage: summarize
calls, model listing, health probes, recommended config, and metrics. This
reference only deploys, operates, and debugs the LVS service container.

## 3. Prerequisites

- Docker Engine ≥ 20.10 with Compose plugin (`docker compose version` to verify)
- NVIDIA Container Toolkit installed and wired to the docker daemon
  *(scaffolded — GPU is requested via `deploy.resources.reservations.devices`)*
- ~20 GB free disk for the container image + more for model cache (depends on
  whether you pre-mount `MODEL_ROOT_DIR`)
- **Free host ports**: 38111, 38112, 38113 (compose uses `network_mode: host`,
  so the container binds these directly — port remapping is NOT possible
  without switching off host networking)
- Outbound network access to `nvcr.io` (image pull) and whichever LLM/VLM NIM
  endpoints you are pointing at
- **External services already reachable**: Elasticsearch (or Milvus), an LLM
  NIM, a VLM NIM — LVS does not provision these; this blueprint only runs the
  summarization app itself
- A **git checkout of `met-blueprints`** since the compose reads
  `$MDX_SAMPLE_APPS_DIR/lvs/.env` and `$MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml`

## 4. NGC / Registry Preflight

> **(scaffolded by microservice-runbook-generator — not literally in your compose;
> audit before following)** — inferred because the image namespace is
> `nvcr.io/nvstaging/vss-core/...`.

```bash
# Get an NGC API key from https://ngc.nvidia.com/setup/api-key
export NGC_API_KEY="nvapi-..."
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

# Verify your key has pull access to the nvstaging org (this is a staging build —
# most users need an internal NGC org; if you get "unauthorized" try the
# public 3.1.0 tag from the docs instead, see Discrepancies in §19).
docker pull nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f

# Warm the image cache (the `lvs-server` image is the only one pulled by
# this compose; sidecar LLM/VLM NIMs only pull if their profiles are on).
docker compose --profile bp_developer_lvs_2d -f ~/met-blueprints/deployments/lvs/compose.yml pull
```

## 5. Required Secrets & Credentials

See [`lvs.env.example`](lvs.env.example) for the full template. Never commit
your populated `.env`.

| Env var | Purpose | Where to get | Notes |
|---|---|---|---|
| `NGC_API_KEY` | Pull `nvcr.io/nvstaging/*` image | <https://ngc.nvidia.com/setup/api-key> | scaffolded — used for `docker login`, not passed into the container |
| `NVIDIA_API_KEY` | LLM API key (fallback when `OPENAI_API_KEY` is unset) | <https://build.nvidia.com> | `nvapi-…`; effectively required — `LVS_LLM_API_KEY` chains `${OPENAI_API_KEY:-${NVIDIA_API_KEY}}` |
| `OPENAI_API_KEY` | Preferred LLM / VLM API key | OpenAI or matching provider | If set, overrides `NVIDIA_API_KEY` for both LLM and VLM |
| `VIA_VLM_API_KEY` | VLM API key (fallback if `OPENAI_API_KEY` unset) | your VLM provider | Defaults to the literal string `not-used` — only required when the VLM endpoint actually enforces auth |

> The compose file itself does **not** contain any secret literals. One
> plaintext-secret audit done.

## 6. Required Volume Mounts

Two bind mounts, both defined in `compose.yml` (`volumes:` block):

| Source (host) | Target (container) | Purpose | Stateful? | `down -v` safe? |
|---|---|---|---|---|
| `$MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml` | `/app/config.yaml` (ro) | CA-RAG pipeline config (tools + summarization function) | no — read-only | yes (bind mount unaffected) |
| `${MODEL_ROOT_DIR:-/tmp/model_cache}` | same path inside container | VLM weight download cache (identical mount on both sides — set `MODEL_PATH` inside this dir) | yes | yes (bind mount) |

**Pre-create before first `up`:**

```bash
# 1) met-blueprints checkout must be present; set MDX_SAMPLE_APPS_DIR so the
#    env_file and config.yaml mount resolve. The compose references
#    $MDX_SAMPLE_APPS_DIR/lvs/.env AND $MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml
export MDX_SAMPLE_APPS_DIR=~/met-blueprints/deployments
ls "$MDX_SAMPLE_APPS_DIR/lvs/.env" "$MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml"

# 2) Model cache on the host (bind-mount target). Keep off /tmp if you don't
#    want it wiped on reboot.
export MODEL_ROOT_DIR=/opt/models
sudo mkdir -p "$MODEL_ROOT_DIR"
sudo chown -R "$(id -u):$(id -g)" "$MODEL_ROOT_DIR"
```

> **Named-volume warning**: this compose declares no named volumes, only
> bind mounts. `docker compose down -v` will not wipe `$MODEL_ROOT_DIR` —
> but it also won't clean it, so you must delete the host path manually.

## 7. Required Environment Variables

All loaded via the `env_file: $MDX_SAMPLE_APPS_DIR/lvs/.env` reference. The
compose's `environment:` block re-exports values under canonical names the
container expects.

| Var | Required | Default (compose) | Provenance | Notes |
|---|---|---|---|---|
| `MDX_SAMPLE_APPS_DIR` | **yes** (effectively) | — | from-compose | interpolated into `env_file` and volume `source`; unset → the compose resolves to literal `/lvs/...` and fails |
| `CONTAINER_IMAGE` | no | `nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f` | from-compose | pin a different tag by setting in `.env` |
| `GPU_DEVICES` | no | `0` | from-compose | comma-separated GPU device IDs (e.g. `"2,3"`) |
| `HOST_IP` | **yes** (effectively) | — | from-compose | interpolated into `LLM_BASE_URL` and `VLM_BASE_URL` when those are not explicitly set |
| `LVS_LLM_MODEL_NAME` | **yes** | — | from-compose + docs | docs example: `openai/gpt-oss-120b` or `meta/llama-3.1-70b-instruct` |
| `LLM_BASE_URL` **or** (`HOST_IP` + `LLM_PORT`) | **yes** | — | from-compose | one or the other must resolve; LVS builds `LVS_LLM_BASE_URL` from `${LLM_BASE_URL:-http://${HOST_IP}:${LLM_PORT}}/v1` |
| `VLM_BASE_URL` **or** (`HOST_IP` + `VLM_PORT`) | **yes** | — | from-compose | same pattern as LLM; LVS builds `VIA_VLM_ENDPOINT=${VLM_BASE_URL:-http://${HOST_IP}:${VLM_PORT}}/v1/` |
| `NVIDIA_API_KEY` | **yes** (unless `OPENAI_API_KEY` set) | — | from-compose + docs | see §5 |
| `LVS_DATABASE_BACKEND` | no | `elasticsearch_db` | from-compose | or `vector_db` for Milvus |
| `ES_HOST` + `ES_PORT` | **yes** when `LVS_DATABASE_BACKEND=elasticsearch_db` | — | from-compose | docs default `ES_PORT=9202`; the sample `.env` in met-blueprints uses `9200` — see §19 Discrepancies |
| `MILVUS_DB_HOST` + `MILVUS_DB_GRPC_PORT` | **yes** when `LVS_DATABASE_BACKEND=vector_db` | — | from-compose | default Milvus gRPC port `19530` |
| `LVS_EMB_ENABLE` | **yes** | — | from-compose | set `false` to skip the embedding-NIM requirement entirely (simplest standalone setup); `true` requires the next three |
| `LVS_EMB_MODEL_NAME` | required when `LVS_EMB_ENABLE=true` | — | from-compose | e.g. `nvidia/nv-embedqa-e5-v5` |
| `LVS_EMB_BASE_URL` | required when `LVS_EMB_ENABLE=true` | — | from-compose | e.g. `http://${HOST_IP}:9232/v1` |

See [`lvs-environment-variables.md`](lvs-environment-variables.md) for **optional / feature-flag** vars
(OTEL, log level, RTVI-VLM integration, VLM input dimensions, audio/Riva,
etc.) and the full docs ↔ compose alignment table.

## 8. GPU Selection & Hardware

The compose requests a single GPU slot via the modern `deploy.resources` API
(it does **not** use the `runtime: nvidia` legacy flag). The specific device
is bound by `GPU_DEVICES`:

```yaml
# from compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['${GPU_DEVICES:-0}']
          capabilities: [gpu]
```

Patterns:

| Goal | Set in `.env` |
|---|---|
| Default GPU 0 | `GPU_DEVICES=0` (or leave unset) |
| Specific GPU | `GPU_DEVICES=2` |
| Multiple GPUs | `GPU_DEVICES=0,1` |
| Pin by UUID | `GPU_DEVICES=GPU-abc123...` (from `nvidia-smi -L`) |

The entrypoint (`start_via.sh`) auto-tunes `VLM_BATCH_SIZE` from detected VRAM:
- ≤46 GB → batch 1–3
- 46–80 GB → batch 2–16
- \>80 GB → batch 16–128 (int4 path)

Override with `VLM_BATCH_SIZE=<n>` in `.env` if you want a specific size.
Override `VLLM_GPU_MEMORY_UTILIZATION` if you are co-tenanting another model.

SM 10.x GPUs have an automatic `TRT_LLM_MODE=fp16` override in the entrypoint
(int4_awq is skipped). SM 12.1 also sets `TRT_LLM_ATTN_BACKEND=FLASHINFER`.

## 9. Port Map

This compose uses **`network_mode: host`** — the container shares the host's
network namespace and binds directly. The compose has no `ports:` block
because host networking ignores it.

| Port | Service | Notes |
|---|---|---|
| 38111 | REST API (`BACKEND_PORT`) | `curl http://<host-ip>:38111/v1/ready` |
| 38112 | MCP server (`LVS_MCP_PORT`) | SSE transport at `/sse`; gated by `LVS_ENABLE_MCP=true` |
| 38113 | Frontend (`FRONTEND_PORT`) | served directly from the backend |

**Remapping** is NOT possible without switching off host networking. If you
must co-deploy a second LVS on one host, change `BACKEND_PORT` /
`LVS_MCP_PORT` / `FRONTEND_PORT` in `.env` so the two instances don't collide
— *and* remove or change `container_name: vss-lvs` (see §19).

Common co-tenant collisions: Elasticsearch (9200), Milvus (19530), NIM
containers (most default to 8000 — unlikely to touch these 38xxx ports).

## 10. Models & Swap Guide

LVS is **client-only** for VLM/LLM: it calls remote OpenAI-compatible
endpoints (LLM NIM, VLM NIM, or third-party APIs). It does NOT host the
models itself — model weights that the docs reference (`Qwen3-VL-8B`, etc.)
live in whatever NIM you point `VLM_BASE_URL` at.

### LLM (summarization)

- **Controlled by**: `LVS_LLM_MODEL_NAME`, `LVS_LLM_BASE_URL`, `LVS_LLM_API_KEY`
- **Docs default**: `openai/gpt-oss-120b` via a co-deployed LLM NIM
- **Swap example (external LLM NIM)**:
  ```bash
  echo 'LVS_LLM_MODEL_NAME=meta/llama-3.1-70b-instruct' >> .env
  echo 'LLM_BASE_URL=http://llm-nim.internal:8002'      >> .env
  echo 'NVIDIA_API_KEY=nvapi-...'                       >> .env
  docker compose --profile bp_developer_lvs_2d up -d --force-recreate
  ```
- **Swap example (build.nvidia.com API)**:
  ```bash
  echo 'LVS_LLM_MODEL_NAME=meta/llama-3.1-70b-instruct' >> .env
  echo 'LLM_BASE_URL=https://integrate.api.nvidia.com'  >> .env
  echo 'NVIDIA_API_KEY=nvapi-...'                       >> .env
  ```

### VLM (frame captioning)

- **Controlled by**: `VLM_BASE_URL` (or `HOST_IP`+`VLM_PORT`), `VIA_VLM_API_KEY`,
  `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME`, `VLM_MODEL_TO_USE`
- **Docs default**: `Qwen3-VL-8B-Instruct` via a vLLM-compatible NIM
  (docs default `VLM_MODEL_TO_USE=vllm-compatible`; the blueprint's sample
  `.env` uses `openai-compat` — either works, both are remote endpoints)
- **Swap to RTVI-VLM sidecar**:
  ```bash
  echo 'USE_RTVI_VLM=true'                              >> .env
  echo 'RTVI_VLM_URL=http://rtvi-vlm.internal:9191'    >> .env
  # optional: echo 'RTVI_VLM_URL_PASSTHROUGH=true'     >> .env
  ```

### Embedding model (optional)

Gated by `LVS_EMB_ENABLE`. Set `false` to turn off embedding entirely; set
`true` plus `LVS_EMB_MODEL_NAME` + `LVS_EMB_BASE_URL`.

## 11. Deploy

```bash
# 1) Blueprints checkout must exist; export its deployments dir
export MDX_SAMPLE_APPS_DIR=~/met-blueprints/deployments
ls "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml"  # sanity-check

# 2) Populate .env — use the example shipped with video-summarization as a
#    starting point, OR edit met-blueprints/deployments/lvs/.env directly.
export VIDEO_SUMMARIZATION_SKILL_DIR=/path/to/skills/video-summarization
cp "$VIDEO_SUMMARIZATION_SKILL_DIR/references/lvs.env.example" "$MDX_SAMPLE_APPS_DIR/lvs/.env"
$EDITOR "$MDX_SAMPLE_APPS_DIR/lvs/.env"    # fill every REQUIRED field

# 3) Model cache bind mount (skip if you already have one)
export MODEL_ROOT_DIR=/opt/models
sudo mkdir -p "$MODEL_ROOT_DIR"
sudo chown -R "$(id -u):$(id -g)" "$MODEL_ROOT_DIR"

# 4) NGC login (image is on nvcr.io/nvstaging/...)
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

# 5) Warm the image cache — the profile flag is REQUIRED or nothing resolves
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  pull

# 6) Bring up. The healthcheck's start_period is 120s — expect up to ~2
#    minutes before readiness, longer if the VLM NIM you point at is
#    downloading weights.
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  up -d

# 7) Wait for healthy
until [ "$(docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" --profile bp_developer_lvs_2d ps --format json | jq -r '[.[].Health] | all(. == "healthy")')" = "true" ]; do
  echo "waiting for health..."
  sleep 5
done

# 8) Verify. Host networking → use the host's IP (or localhost).
curl -f http://localhost:${BACKEND_PORT:-38111}/v1/ready
```

**Profiles in this compose** (only one is defined):

| Profile | Enables |
|---|---|
| `bp_developer_lvs_2d` | `lvs-server` (the summarization container) |

Every `docker compose` invocation below must include
`--profile bp_developer_lvs_2d` or nothing matches.

## 12. Dry Run

```bash
# Print the fully-resolved compose (env substitution applied, anchors
# expanded) for human audit. Pass the profile or services disappear.
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  config

# Exit-code-only validation
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  config --quiet && echo "compose is valid"

# Create containers and check volume mounts, without starting them
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  up --no-start
```

## 13. Verify Deployment

```bash
# Health
curl -f http://localhost:38111/v1/ready
# Expected: 200 OK (body format not documented; service-readiness signal)

# List available models (see lvs-api.md for endpoint details)
curl -s http://localhost:38111/v1/models | jq

# Primary path - summarize a test video (see the video-summarization skill for
# the full summarize flow)
```

Healthy log signatures (grep `docker compose logs vss-lvs`):

- `Starting VIA server in release mode`
- `VIA Server loaded`
- `Backend is running at http://0.0.0.0:38111`
- `Auto-selecting VLM Batch Size to <N>` or `Using VLM Batch Size <N>`

## 14. Logs & Status

```bash
# Snapshot
docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d ps

# Live tail (compose service name is lvs-server; container name is vss-lvs)
docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d logs -f lvs-server
# — or directly by container name, no profile flag needed:
docker logs -f vss-lvs

# History, bounded
docker logs --tail 200 --since 10m vss-lvs

# Resource usage
docker stats vss-lvs
```

Log verbosity: set `VSS_LOG_LEVEL=DEBUG` (default in the sample `.env`),
`INFO`, `WARN`, or `ERROR`. Logs inside the container live at
`/tmp/via-logs/` (set `VIA_LOG_DIR` to redirect; nothing is bind-mounted, so
they disappear on `down`).

## 15. Debugging Common Failures

See [`lvs-debugging.md`](lvs-debugging.md) for the long-form table. Quick reference:

| Symptom | Root cause | Fix |
|---|---|---|
| `services.lvs-server: 'profiles' configured ... but nothing matched` | You forgot `--profile bp_developer_lvs_2d` | Add the flag to every `docker compose` call |
| `invalid mount config for type "bind": bind source path does not exist: /lvs/.env` | `MDX_SAMPLE_APPS_DIR` unset | `export MDX_SAMPLE_APPS_DIR=~/met-blueprints/deployments` before compose runs |
| `Exited (1)` immediately, logs say `unauthorized` | NGC login missing or no access to `nvstaging` org | `docker login nvcr.io`; if still blocked, this is a staging tag — try the public docs tag `nvcr.io/nvidia/vss-core/vss-long-video-summarization:3.1.0` (see §19 Discrepancies) |
| `Exited (137)` (OOM) | VRAM exhausted | Lower `VLM_BATCH_SIZE`, lower `VLLM_GPU_MEMORY_UTILIZATION`, or set `GPU_DEVICES` to a larger GPU |
| `bind: address already in use` for 38111/38112/38113 | Another process on host networking binds same port | Stop the other process or set `BACKEND_PORT` / `LVS_MCP_PORT` / `FRONTEND_PORT` in `.env` |
| Healthcheck keeps failing past 120s `start_period` | Upstream LLM/VLM NIM unreachable; LVS blocks on first request | `curl` `LVS_LLM_BASE_URL/models` and the VLM endpoint from the host to verify; check `docker logs vss-lvs` for the exact HTTP error |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit not installed / daemon not restarted | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| API returns 503 "Another video is being processed" | LVS processes one video at a time | Wait, or scale by running a second LVS on a different host / different ports + container_name |
| VLM returns incomplete JSON | Model couldn't satisfy schema | Simplify events list, increase frames-per-chunk, retry; docs' Known Issue |
| MCP server unreachable | `LVS_ENABLE_MCP` not `true`, or port 38112 blocked | Check env; check host firewall |

## 16. Upgrade & Rollback

```bash
# Bump image tag: edit CONTAINER_IMAGE in .env or export inline
export CONTAINER_IMAGE=nvcr.io/nvstaging/vss-core/vss-video-summarization:<new-tag>

docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d pull
docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d up -d
```

**Rollback**: set `CONTAINER_IMAGE` back to the previous tag and re-run `pull`
+ `up -d --force-recreate`. The bind-mounted model cache at `MODEL_ROOT_DIR`
survives — no re-download unless the new image needs a different model.

## 17. Tear Down

```bash
# Stop + remove container; bind mounts (config.yaml, MODEL_ROOT_DIR) untouched
docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d down

# -v is a no-op here since the compose declares no named volumes. Bind mounts
# are host-owned; delete them manually if you want to reclaim the space:
# sudo rm -rf "$MODEL_ROOT_DIR"

# Remove the pulled image too (optional)
docker rmi nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f
```

> There are **no named volumes** in this compose, so `down -v` has no
> destructive effect here. The only stateful data is whatever lives under
> `$MODEL_ROOT_DIR` on the host.

## 18. Gotchas & Known Issues

- **`profiles: ["bp_developer_lvs_2d"]` on `lvs-server` (compose.yml:21)** —
  plain `docker compose up` is a no-op. Every command must pass
  `--profile bp_developer_lvs_2d`. *(from compose)*
- **`container_name: vss-lvs` is hardcoded (compose.yml:19)** — you cannot
  run two LVS instances on one host without editing the compose; the second
  `up` fails with a name conflict. *(from compose)*
- **`network_mode: host` on compose.yml:32** — port remapping via `ports:` is
  impossible; change the `BACKEND_PORT` / `LVS_MCP_PORT` / `FRONTEND_PORT`
  env vars to avoid host-port collisions instead. *(from compose)*
- **16 × `depends_on` with `required: false` (compose.yml:102–150)** — all
  the LLM/VLM NIM sidecars (`nvidia-nemotron-*`, `gpt-oss-*`, `qwen3-vl-*`,
  `cosmos-reason*`, etc.) are defined in sibling blueprint composes. With
  `required: false`, LVS starts even when those services aren't running or
  their profiles aren't on — meaning the stack will happily come up
  "healthy" (container process up) while the upstream LLM/VLM you configured
  is down. Always verify the LLM endpoint independently. *(from compose)*
- **`start_period: 120s`** — the healthcheck grace window. If the VLM/LLM
  endpoint is slow to warm up on its end, LVS may still be unhealthy after
  120s; that's expected, not a bug. *(from compose)*
- **Single-video serialization** — VIA processes one video at a time; parallel
  requests get HTTP 503. Docs' Known Issue. *(from docs)*
- **Incomplete JSON from VLM** — simplify event lists, bump frames-per-chunk,
  retry. Docs' Known Issue. *(from docs)*
- **Entrypoint VRAM auto-tune** — if VRAM ≤50 GB, entrypoint forces
  `VLLM_GPU_MEMORY_UTILIZATION=0.7` and a smaller `VLM_BATCH_SIZE`. Look for
  the `Auto-selecting VLM Batch Size to <N>` log line. *(from entrypoint)*
- **SM 10.x → fp16 override** — entrypoint skips int4_awq on Blackwell-class
  GPUs and forces fp16, regardless of `TRT_LLM_MODE`. *(from entrypoint)*
- **Docs ↔ compose drift** — compose uses `nvcr.io/nvstaging/...:3.2.0-rc1-...`
  (staging RC build), while the public docs describe
  `nvcr.io/nvidia/vss-core/vss-long-video-summarization:3.1.0`. The repo names
  also differ (`vss-video-summarization` vs `vss-long-video-summarization`).
  If you lack `nvstaging` org access, pin `CONTAINER_IMAGE` to the public
  tag. See §19 Discrepancies.

## 19. Discrepancies between this compose and the public docs

| Field | Compose value | Docs value | Impact |
|---|---|---|---|
| Image registry + repo | `nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f` | `nvcr.io/nvidia/vss-core/vss-long-video-summarization:3.1.0` | Different NGC org + repo name. Users without `nvstaging` access will hit `unauthorized` — fall back to the public tag. |
| `ES_PORT` default | no compose default; sample `.env` uses `9200` | `9202` | Align with whatever Elasticsearch you're pointing at. |
| `ES_TRANSPORT_PORT` | sample `.env` uses `9300` | `9302` | Only matters if you run ES on the same host. |
| `VLM_MODEL_TO_USE` default | `openai-compat` (sample `.env`) | `vllm-compatible` | Both are client-side selectors; match to what your VLM NIM actually is. |
| AWS creds (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`) | not passed in compose `environment:` | listed in docs for S3 video URLs | Add to `.env` if you want S3-hosted inputs (they'll be picked up via `env_file`). |

## 20. References

- **Docs**: https://docs.nvidia.com/vss/latest/long-video-summarization.html
- **Compose source**: `~/met-blueprints/deployments/lvs/compose.yml`
- **Dockerfile source**: `~/long-video-summarization/docker/Dockerfile`
- **Entrypoint source**: `~/long-video-summarization/start_via.sh`
- **CA-RAG config sample**: `~/met-blueprints/deployments/lvs/configs/config.yaml`
- **Sample `.env`**: `~/met-blueprints/deployments/lvs/.env`
- Generated by the `microservice-runbook-generator` skill.
