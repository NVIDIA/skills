# Warehouse Blueprint Reference

Blueprint: VSS Warehouse — RT-DETR (2D) / Sparse4D (3D) / MV3DT (multi-view 3D tracking with BEV Fusion) perception + behavior analytics over multi-camera warehouse streams. Distinct from the core VSS profiles (`base`, `alerts`, `lvs`, `search`): it lives under `<repo>/deploy/docker/industry-profiles/warehouse-operations/` and is deployed from `<repo>/deploy/docker/` using `industry-profiles/warehouse-operations/.env`.

The compose files ship **in-tree** in the `video-search-and-summarization` repo — no NGC compose bundle to download. App data (videos and models) is the only artifact you may need to acquire; see [App Data](#app-data).

Work through **one path** under [Choose your path](#choose-your-path). Reference tables (variants, services, GPU layout, endpoints, artifacts) are in the top half; operational phases are in the bottom half.

---

## Profile Variants

| Profile Name | MODE | BP_PROFILE | SAMPLE_VIDEO_DATASET | NUM_STREAMS | LLM | RTVI VLM |
|---|---|---|---|---|---|---|
| 2D Vision AI Profile | `2d` | `bp_wh_kafka` or `bp_wh_redis` | `warehouse-loading-dock-3cams-synthetic` | 3 | none | none |
| 2D Vision AI with Agents Profile | `2d` | `bp_wh` | `nv-warehouse-4cams` | 4 | `local` / `local_shared` / `remote` / `none` | **always local** |
| 3D Vision AI Profile | `3d` | `bp_wh_kafka` or `bp_wh_redis` | `warehouse-4cams-20mx20m-synthetic` | 4 | none | none |
| MV3DT Vision AI Profile | `mv3dt` | `bp_wh_kafka` or `bp_wh_redis` | `warehouse-4cams-20mx20m-synthetic` | 4 | none | none |
| Warehouse Auto-Calibration | `2d` / `3d` / `mv3dt` | `bp_wh_auto_calib` | (same as mode default) | (same as mode default) | none | none |
| Standalone Auto-Calibration | any | `auto_calib` | n/a | n/a | none | none |

`COMPOSE_PROFILES` is computed automatically: `${BP_PROFILE}_${MODE},llm_${LLM_MODE}_${LLM_NAME_SLUG}`. No `vlm_*` slice — `vss-rtvi-vlm` is always deployed for `bp_wh` and there is no VLM NIM.

## Minimal vs Extended Profile

Applies to `bp_wh_kafka` and `bp_wh_redis` only (all modes: 2d, 3d, mv3dt).

| Feature | Minimal (`MINIMAL_PROFILE="true"`) | Extended (`MINIMAL_PROFILE=""`) |
|---|---|---|
| Perception (RT-DETR 2D / Sparse4D 3D) | ✅ | ✅ |
| Behavior Analytics | ✅ | ✅ |
| VST / NvStreamer | ✅ | ✅ |
| Auto-Calibration | ✅ | ✅ |
| ELK (Elasticsearch/Logstash/Kibana) | ❌ | ✅ |
| Video Analytics API (`vss-video-analytics-api`, `MDX_PORT` 8081) | ❌ | ✅ |
| Monitoring | ❌ | ✅ |
| Bounding box overlays in VST | ❌ | ✅ (requires Elasticsearch) |

## Services Deployed

The warehouse blueprint boots the **full VSS stack** (agent + UI + VST + RTVI behind HAProxy) on top of the warehouse CV pipeline. Service set varies by `BP_PROFILE` and `MODE`. Perception, behavior analytics, nvstreamer, and most other services use the **same container names** in 2D and 3D — no `-2d` / `-3d` suffix. MV3DT uses a **`-mv3dt` suffix** on all its containers (e.g. `vss-vios-nvstreamer-mv3dt`, `vss-behavior-analytics-mv3dt`, `vss-rtvi-cv-mv3dt`, `vss-configurator-mv3dt`, `vss-video-analytics-api-mv3dt`).

### Warehouse CV core (2D and 3D profiles)

| Container | Purpose |
|---|---|
| `vss-vios-nvstreamer` | Streams sample video files via RTSP |
| VST stack: `vss-vios-postgres`, `-sensor`, `-streamprocessing`, `-sdr`, `-mcp`, `-ingress`, `-envoy` | Video ingestion, recording, stream management |
| `vss-rtvi-cv` | DeepStream perception (RT-DETR for 2D, Sparse4D for 3D) |
| `vss-rtvi-cv-sdr` | Stream data router — manages DeepStream lifecycle |
| `vss-rtvi-cv-config-adaptor` | DeepStream config adaptor (3D only) |
| `vss-configurator` | Blueprint configurator — stream and hardware configs |
| `vss-behavior-analytics` | Behavior analytics — ROI, tripwire, proximity events |
| `kafka` or `redis` (`STREAM_TYPE`) | Message broker for CV metadata and control bus |
| `vss-broker-health-check` | Waits for broker readiness before starting dependent services |

### MV3DT CV core (`bp_wh_kafka_mv3dt` / `bp_wh_redis_mv3dt`)

MV3DT adds MQTT-based cross-camera messaging and BEV Fusion on top of per-camera DeepStream perception. All MV3DT containers carry a `-mv3dt` suffix.

| Container | Purpose |
|---|---|
| `vss-vios-nvstreamer-mv3dt` | Streams sample video files via RTSP |
| VST stack: `vss-vios-postgres`, `sensor-ms-mv3dt`, `-streamprocessing`, `-sdr`, `-mcp`, `-ingress`, `-envoy` | Video ingestion, recording, stream management |
| `vss-rtvi-cv-mv3dt` | DeepStream perception (per-camera) |
| `vss-rtvi-cv-bev-fusion` | BEV Fusion — fuses per-camera detections into a unified 3D BEV frame |
| `mosquitto` | MQTT broker for cross-camera messaging between perception and BEV fusion |
| `vss-configurator-mv3dt` | Blueprint configurator — stream and hardware configs |
| `vss-behavior-analytics-mv3dt` | Behavior analytics — 3D spatial analytics |
| `kafka` or `redis` (`STREAM_TYPE`) | Message broker for CV metadata and control bus |
| `vss-broker-health-check` | Waits for broker readiness before starting dependent services |

### Warehouse Auto-Calibration (`bp_wh_auto_calib`)

Deploys only the minimum services needed for camera calibration — no perception, no behavior analytics, no agent stack. Available for all modes (`bp_wh_auto_calib_2d`, `bp_wh_auto_calib_3d`, `bp_wh_auto_calib_mv3dt`). Skips broker health check. These are the only warehouse profiles that start `vss-auto-calibration` and `vss-auto-calibration-ui`; regular `bp_wh`, `bp_wh_kafka`, and `bp_wh_redis` profiles do not.

| Container | Purpose |
|---|---|
| `vss-vios-nvstreamer` / `vss-vios-nvstreamer-mv3dt` | Streams sample video files via RTSP |
| `vss-configurator` / `vss-configurator-mv3dt` | Blueprint configurator |
| `vss-auto-calibration` (+ `vss-auto-calibration-ui`) | Camera auto-calibration |
| VST stack (subset) | Stream management for calibration |

### Agent + UI + ingress (`bp_wh` only)

| Container | Port |
|---|---|
| `vss-haproxy-ingress` | `HAPROXY_PORT` (default `7777`) |
| `vss-agent-ui` (Next.js) | 3000 |
| `vss-agent` | `VSS_AGENT_PORT` (default `8000`) |
| `vss-va-mcp` | `VSS_VA_MCP_PORT` (default `9901`) |
| `phoenix` (telemetry) | 6006 |

### Storage / observability (conditional)

| Container | Port | Deployed when |
|---|---|---|
| `elasticsearch` | `VSS_ES_PORT` (default `9200`) | `BP_PROFILE=bp_wh` (always — vss-agent storage), **or** kafka/redis with `MINIMAL_PROFILE=""` (extended, any mode — for `mdx-bev`, ELK, overlays, analytics API) |
| `kibana` / `logstash` / `vss-video-analytics-api` | various | Same condition as `elasticsearch` (MV3DT uses `vss-video-analytics-api-mv3dt`) |

> **3D / MV3DT `mdx-bev` index requires Elasticsearch — and ES is only deployed for kafka/redis in extended mode** (`MINIMAL_PROFILE=""`). In minimal mode, the BEV-sync check cannot run because the index is never persisted.

### LLM + RTVI VLM (`bp_wh` only)

| Container | Port | When |
|---|---|---|
| LLM NIM — container name = `LLM_NAME_SLUG` (e.g. `nvidia-nemotron-nano-9b-v2`) | `LLM_PORT` (default `30081`) | `LLM_MODE=local` or `local_shared` |
| `vss-rtvi-vlm` (real-time VLM) | 8018 | **Always** deployed for `bp_wh` — hardcoded in compose profile `bp_wh_2d` |
| `vss-alert-bridge` | `ALERT_BRIDGE_PORT` (default `9080`) | Always deployed for `bp_wh` |

> **No VLM NIM container.** VSS has two VLM paths: a standalone **VLM NIM** (controlled by `VLM_MODE` / `VLM_NAME_SLUG`, used by base/alerts/lvs/search profiles) and an integrated **RTVI VLM** (`vss-rtvi-vlm`). The warehouse blueprint uses **RTVI VLM only** — `vss-rtvi-vlm` is always deployed via the hardcoded compose profile `bp_wh_2d`, and `vss-agent` connects to it directly. Because warehouse does not use the standalone VLM NIM path, `VLM_MODE=none` and `VLM_NAME_SLUG=none` in the warehouse `.env`. There is no `vlm_*` slice in `COMPOSE_PROFILES`, so VLM NIM containers (e.g. `cosmos-reason2-8b` on port 30082) are never deployed.

## Perception Model

- **2D model:** RT-DETR with EfficientViT/L2 backbone
- **3D model:** Sparse4D (depth-aware perception, requires 4-camera dataset)
- **MV3DT model:** Per-camera DeepStream perception + BEV Fusion (multi-view 3D tracking, fuses detections from multiple cameras into a unified BEV frame via MQTT)
- **Detects:** People, humanoid robots, forklifts, autonomous vehicles, warehouse equipment
- **Output:** 2D bounding boxes (or 3D BEV frames) with tracked object IDs via Kafka/Redis `mdx-raw` topic; 3D / MV3DT BEV frames also land in the `mdx-bev` Elasticsearch index

## GPU Layout

| Role | Device | Used by |
|---|---|---|
| RT-CV perception (DeepStream — RT-DETR for 2D, Sparse4D for 3D, MV3DT for mv3dt) — always local | `RT_CV_DEVICE_ID` (default: `0`) | All warehouse profiles |
| RTVI VLM — always local | `RT_VLM_DEVICE_ID` (default: `1`) | `bp_wh` only |
| LLM NIM (dedicated) | `LLM_DEVICE_ID` (default: `2`) | `bp_wh` with `LLM_MODE=local` |
| LLM NIM sharing the RTVI VLM device | `SHARED_LLM_VLM_DEVICE_ID` (default: `2`) | `bp_wh` with `LLM_MODE=local_shared` |

`LLM_MODE` accepts `local`, `local_shared`, `remote`, or `none`:
- `local` — LLM NIM on its own GPU (`LLM_DEVICE_ID`)
- `local_shared` — LLM NIM colocated with RTVI VLM on `SHARED_LLM_VLM_DEVICE_ID` (use when GPU count is limited)
- `remote` — point at an external LLM endpoint via `LLM_BASE_URL` (no LLM NIM deployed)
- `none` — no LLM, for `bp_wh_kafka` / `bp_wh_redis` / `bp_wh_auto_calib`

RTVI VLM has no equivalent mode setting — it is always deployed locally on `RT_VLM_DEVICE_ID` for `bp_wh`. `VLM_MODE` in the warehouse `.env` is set to `none` because warehouse uses RTVI VLM instead of the standalone VLM NIM path.

## Access Points

**Prefer the HAProxy ingress (port `7777`)** — it gives a single browser-reachable origin and rewrites paths to internal services. Direct ports are only useful for diagnostics from the host. Routes confirmed against `deploy/docker/services/infra/haproxy/haproxy.cfg.template`.

### Via HAProxy ingress (`http://<EXTERNAL_IP>:<HAPROXY_PORT>` — default `<EXTERNAL_IP>:7777`)

| Path | Backend | Profile |
|---|---|---|
| `/` | `vss-agent-ui` (Next.js) | `bp_wh` (returns 503 in `bp_wh_kafka`/`bp_wh_redis` — no UI backend) |
| `/vst`, `/vst/...` | `vss-vios-ingress` (VST / VIOS UI) | All |
| `/storage`, `/storage/...` | `vst-storage` (compat → `/vst/storage/...`) | All |
| `/kibana`, `/kibana/...` | `kibana` | `bp_wh`, or kafka/redis extended (2D or 3D) |
| `/video-analytics-api`, `.../...` | `vss-video-analytics-api` | `bp_wh`, or kafka/redis extended |
| `/behavior-analytics`, `.../...` | `vss-behavior-analytics` | All |
| `/perception-sdr`, `.../...` | `vss-rtvi-cv-sdr` | All |
| `/alert-bridge`, `.../...` | `vss-alert-bridge` | `bp_wh` only |
| `/phoenix`, `.../...` | `phoenix` | `bp_wh` only |
| `/va-mcp`, `.../...` | `vss-va-mcp` | `bp_wh` only |
| `/api`, `/api/...` | `vss-agent` | `bp_wh` only |
| `/api/chat`, `.../...` | `vss-agent-ui` | `bp_wh` only |
| `/chat`, `/static`, `/websocket` | `vss-agent` | `bp_wh` only |

### Direct ports (no HAProxy route — diagnostics only)

| Service | URL | Profile |
|---|---|---|
| NvStreamer UI | `http://<HOST_IP>:31000` | All |
| Auto-Calibration UI | `http://<HOST_IP>:5000` | `auto_calib`, `bp_wh_auto_calib_2d`, `bp_wh_auto_calib_3d`, `bp_wh_auto_calib_mv3dt` |
| Elasticsearch API | `http://<HOST_IP>:9200` | `bp_wh`, or kafka/redis extended |
| VSS Agent API (direct) | `http://<HOST_IP>:8000` | `bp_wh` only (prefer `/api` via HAProxy) |
| VST MCP (direct) | `http://<HOST_IP>:8001` | All |
| Phoenix (direct) | `http://<HOST_IP>:6006` | `bp_wh` only (prefer `/phoenix` via HAProxy) |
| Kibana (direct) | `http://<HOST_IP>:5601` | Prefer `/kibana` via HAProxy |
| Video Analytics API (direct) | `http://<HOST_IP>:8081` (`MDX_PORT`) | Prefer `/video-analytics-api` via HAProxy |
| VST UI (direct) | `http://<HOST_IP>:30888/vst` | Prefer `/vst` via HAProxy |

`EXTERNAL_IP` defaults to `${HOST_IP}` but should be set to the browser-reachable hostname/IP. On Brev, follow the same secure-link pattern as the other VSS profiles (`SKILL.md` Step 1c). The HAProxy `h_main` ACL only routes when the `Host:` header matches `${VSS_PUBLIC_HOST}`, `${EXTERNAL_IP}`, `${HOST_IP}`, `localhost`, or `127.0.0.1` (with or without `:${HAPROXY_PORT}`) — wrong Host headers get a 404 from haproxy.

## Compose File Structure

Deployed from `<repo>/deploy/docker/` (the repo's compose root) using:
- `industry-profiles/warehouse-operations/.env` — all configuration
- `compose.yml` — root top-level include (foundational, monitoring, vst, industry-profiles, etc.)
  - `industry-profiles/compose.yml` — industry sub-include
    - `industry-profiles/warehouse-operations/compose.yml` — warehouse sub-include
      - `industry-profiles/warehouse-operations/warehouse-2d-app/warehouse-2d-app.yml` — 2D app services
      - `industry-profiles/warehouse-operations/warehouse-3d-app/warehouse-3d-app.yml` — 3D app services
      - `industry-profiles/warehouse-operations/warehouse-mv3dt-app/warehouse-mv3dt-app.yml` — MV3DT app services

## App Data

App data (sample videos, perception models) is **not** bundled with the repo. Pick one source:

| Source | When to use | `VSS_DATA_DIR` |
|---|---|---|
| `<repo>/data` | Quick start — drop assets into the repo's `data/` directory | `<repo>/data` |
| Custom local path | Existing dataset on a non-repo path (e.g. `/mnt/warehouse-data`) | user-provided path |
| NGC app-data resource | Reproducing the official sample-video deployment | extracted path of `nvidia/vss-warehouse/vss-warehouse-app-data:<version>` **or** `nvstaging/vss-warehouse/vss-warehouse-app-data:<version>` (staging keys land here) |

Ask the user which source they want and whether they already have the assets on disk. Only run the NGC download (next subsection) when they explicitly choose the NGC source.

### NGC app-data download (optional)

| Artifact | NGC Resource | Local directory after extract |
|---|---|---|
| App data (videos, models) | `nvidia/vss-warehouse/vss-warehouse-app-data:<version>` **or** `nvstaging/vss-warehouse/vss-warehouse-app-data:<version>` | `vss-warehouse-app-data_v<version>/` |

> **Org may be `nvidia` or `nvstaging`.** Production keys access the canonical `nvidia/...` path; staging / NVIDIAN keys typically only see `nvstaging/...`. If you get `403 Access Denied` on one, retry with the other before assuming the resource is missing. Confirm by running `ngc org list` to see which orgs the current key belongs to.

## Known Limitations

- Bounding box overlays do not appear in VST in the minimal profile — Elasticsearch is required for overlay rendering. Metadata is available from the live Kafka/Redis stream only.
- Perception model for `warehouse-loading-dock-3cams-synthetic` is trained on synthetic data — accuracy may vary on custom real-world scenes.
- `nv-warehouse-4cams` dataset is only valid with `BP_PROFILE=bp_wh` and `MODE=2d`.
- `warehouse-4cams-20mx20m-synthetic` dataset is valid with `MODE=3d` or `MODE=mv3dt`.
- MV3DT mode (`MODE=mv3dt`) does not support `bp_wh` (agents) — only `bp_wh_kafka`, `bp_wh_redis`, and `bp_wh_auto_calib`.
- `bp_wh` profile in 2D mode is not supported on IGX-THOR or DGX-SPARK.

---

## Choose your path

| Goal | Where to start |
|------|----------------|
| **New machine / first install** | [Full deploy (Phases 1-9)](#full-deploy-phases-1-9). Run phases in order; each must pass before the next. |
| **Redeploy** (`.env` change, clean restart, broken stack) | [Redeploy](#redeploy). Skips Phases 1–4 — host is already set up and artifacts exist. |
| **Tear down only** (stop and remove containers/volumes; keep files on disk) | [Lifecycle: Tear down](#lifecycle-tear-down). |

**`<repo>`** — path to your `video-search-and-summarization` checkout. All compose commands run from `<repo>/deploy/docker/`, with `--env-file industry-profiles/warehouse-operations/.env`. If you don't know the repo path, **ask explicitly** before running shell commands.

---

## Lifecycle (shared)

Use these sections for **redeploy**, **Phase 8–9**, and **tear down**. Default log file for bring up and monitor:

```bash
LOG=${LOG:-/tmp/warehouse-blueprint.log}
```

### Lifecycle: Tear down

Hard teardown — removes all containers, the project network, and all volume belonging to this stack.

```bash
cd <repo>/deploy/docker

# Hard teardown — `-v` ensures named volumes are also removed.
# Containers + network + project's named volumes all go.
docker compose -f compose.yml --env-file industry-profiles/warehouse-operations/.env down -v

# Sweep any leftover anonymous/dangling volumes from prior partial runs.
docker volume prune -f

# Reclaim disk: stopped containers, dangling images, unused networks.
docker system prune -f

# Wipe bind-mounted state under $VSS_DATA_DIR/data_log/* AND revert
# blueprint-configurator backups. Resolves VSS_DATA_DIR from the env file,
# so pass the SAME env you used with `docker compose --env-file ...`.
bash ./scripts/cleanup_all_datalog.sh -e industry-profiles/warehouse-operations/.env
```

### Lifecycle: Bring up

Pulls images and builds the perception container (~10–15 min first run). If `docker compose` fails to pull from `nvcr.io`, confirm `NGC_CLI_API_KEY` is set and retry `docker login` as shown.

```bash
LOG=${LOG:-/tmp/warehouse-blueprint.log}
cd <repo>/deploy/docker

docker login --username '$oauthtoken' --password "${NGC_CLI_API_KEY}" nvcr.io

nohup docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  up --detach --pull always --force-recreate --build \
  > "$LOG" 2>&1 &
echo "Compose PID $! — logging to $LOG"
```

### Lifecycle: Monitor

Poll every ~60s:

```bash
LOG=${LOG:-/tmp/warehouse-blueprint.log}
tail -20 "$LOG"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

**Stack is ready when these show `Up`** (same container names in 2D and 3D; MV3DT uses `-mv3dt` suffix):

- 2D / 3D profiles: `vss-vios-nvstreamer`, `vss-rtvi-cv`, `vss-configurator`, `vss-behavior-analytics`, broker (`kafka` / `redis`), `vss-broker-health-check`, plus the `vss-vios-*` VST stack
- 3D extra: `vss-rtvi-cv-config-adaptor`
- MV3DT profiles: `vss-vios-nvstreamer-mv3dt`, `vss-rtvi-cv-mv3dt`, `vss-rtvi-cv-bev-fusion`, `mosquitto`, `vss-configurator-mv3dt`, `vss-behavior-analytics-mv3dt`, broker (`kafka` / `redis`), `vss-broker-health-check`, plus VST stack
- `bp_wh` extra: `vss-rtvi-vlm`, `vss-alert-bridge`, `vss-agent`, `vss-agent-ui`, `vss-va-mcp`, `vss-haproxy-ingress`, `phoenix`, plus the LLM NIM container (named after `LLM_NAME_SLUG`) when `LLM_MODE=local` / `local_shared`
- Extended extra (kafka/redis, any mode): `vss-haproxy-ingress`, `logstash`, `kibana`, `vss-video-analytics-api` (MV3DT uses `vss-video-analytics-api-mv3dt`)
- `elasticsearch`: `BP_PROFILE=bp_wh` (always), **or** kafka/redis with `MINIMAL_PROFILE=""` (extended, any mode)
- `bp_wh_auto_calib`: only nvstreamer, configurator, auto-calibration, and VST subset

Check FPS (same container for 2D/3D; use `vss-rtvi-cv-mv3dt` for MV3DT):

```bash
# 2D / 3D:
docker logs -f vss-rtvi-cv 2>&1 | grep -i fps | head -5
# MV3DT:
docker logs -f vss-rtvi-cv-mv3dt 2>&1 | grep -i fps | head -5
```

---

## Redeploy

**When to use:** The machine already satisfies [Phase 2](#phase-2-system-prerequisites); the repo is checked out and `VSS_DATA_DIR` is populated. You edited the warehouse `.env`, need a clean restart, or are recovering a bad state.

**Do not** re-run NGC CLI install, driver install, or NGC app-data download unless something is actually missing or broken.

1. Obtain **`<repo>`** path (ask if unknown — see [Choose your path](#choose-your-path)).
2. Run **[Lifecycle: Tear down](#lifecycle-tear-down)**.
3. Run **[Lifecycle: Bring up](#lifecycle-bring-up)** (same `LOG` as monitor).
4. Run **[Lifecycle: Monitor](#lifecycle-monitor)**.

---

## Full deploy (Phases 1-9)

Work through phases in order; each must pass before moving to the next.

### Phase 1: NGC CLI

#### 1.1 Check

```bash
ngc --version
echo "NGC_CLI_API_KEY: ${NGC_CLI_API_KEY:+SET}${NGC_CLI_API_KEY:-NOT SET}"
ngc config current 2>/dev/null | grep -q "apikey" && echo "NGC config: key present" || echo "NGC config: no key"
```

Both set → skip to Phase 2.

#### 1.2 Install (NGC CLI 4.10.0+)

See [`ngc.md` § Install NGC CLI](ngc.md#install-ngc-cli-if-missing) for the
AMD64 / ARM64 install commands. They are kept in `ngc.md` as the single
canonical reference.

#### 1.3 Configure API Key

If no key: go to https://ngc.nvidia.com → **Setup → API Keys → Generate Personal Key** (set **NGC Catalog** permission). Copy immediately.

> **Important:** NGC API keys may look like base64. Use the key exactly as provided — **do not base64-decode it.**

```bash
export NGC_CLI_API_KEY='<key>'
echo "export NGC_CLI_API_KEY='<key>'" >> ~/.bashrc
```

Or configure interactively: `ngc config set`

> Never commit the NGC API key to version control.

#### 1.4 Verify NGC Access

Image paths in `deploy/docker/` reference **both** `nvcr.io/nvidia/vss-core/...` (public org) and `nvcr.io/nvstaging/vss-core/...` (staging org). Which one your key resolves depends on your NGC org membership — list both teams and the warehouse resources visible to it. Confirm the actual paths against `<repo>/deploy/docker/services/**/compose*.{yml,yaml}` and the warehouse `.env` (e.g. `PERCEPTION_IMAGE`, `BEV_FUSION_MV3DT_IMAGE`).

```bash
# Probe both orgs — at least one should succeed for warehouse to deploy
ngc registry image list "nvidia/vss-core/*"     2>&1 | head -10
ngc registry image list "nvstaging/vss-core/*"  2>&1 | head -10
```

**`Missing org` error** → run `ngc config set` (or write `~/.ngc/config` directly) and match the org to the one used when generating the key. Run `ngc org list` to see which orgs the current key has access to before guessing.

---

### Phase 2: System Prerequisites

Run each check in order. **If a check fails, automatically install and re-verify — do not wait for the user.** Only stop if a requirement cannot be met automatically (unsupported hardware, insufficient RAM/CPU).

#### Supported Hardware

`HARDWARE_PROFILE` is a **blueprint setting**, not a string that `nvidia-smi` always prints verbatim. For **discrete GPUs**, match the GPU model from `nvidia-smi` / `lspci` to a row below. **IGX-THOR** and **DGX-SPARK** are **whole-system platforms** (kits/boards): set the profile from product/SKU or vendor docs if you already know the machine type; `nvidia-smi` shows the **on-board NVIDIA GPU name** (e.g. a Thor-class or Spark system GPU), not the text `IGX-THOR` or `DGX-SPARK`. On **DGX Spark**, unified memory can make some `nvidia-smi` memory fields show **Not Supported**; driver and device listing should still be checked per [DGX Spark user guide](https://docs.nvidia.com/dgx/dgx-spark/).

Valid values: `H100, L40, L40S, L4, A6000, RTXA6000, RTXA6000ADA, RTXPRO6000BW, IGX-THOR, DGX-SPARK`. All profiles include tuned `max_streams_supported` for 2D, 3D, and MV3DT modes.

| Discrete GPU (typical `nvidia-smi` name) | HARDWARE_PROFILE |
|---|---|
| RTX PRO 6000 Blackwell | `RTXPRO6000BW` |
| RTX 4500 Blackwell | `RTX4500` — 32 GB; see [alerts.md § RTX 4500](alerts.md#rtx-4500-32-gb) for the required `LLM_MODE=remote` + FP8 VLM model overrides |
| H100 (NVL, SXM HBM3) | `H100` |
| RTX A6000 Ada Generation | `RTXA6000ADA` |
| RTX A6000 (Ampere) | `RTXA6000` |
| A6000 (generic alias) | `A6000` |
| L40S | `L40S` |
| L40 | `L40` |
| L4 | `L4` |
| Platform: NVIDIA IGX Thor (kit / board) | `IGX-THOR` |
| Platform: NVIDIA DGX Spark | `DGX-SPARK` |

> **Do NOT use a higher profile on lower-profile hardware** (e.g. `H100` on an `L4`) — the env file warns against this directly.

**GPUs not in the list above:** the warehouse blueprint may not have a tuned profile. Pick the closest match from the table or treat the deployment as unsupported on that GPU until the upstream list adds it.

#### 2.1 GPU Detection and NVIDIA Driver

**Detect GPUs and driver:**

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

Use the **`name`** column to pick **`HARDWARE_PROFILE`** from the [Supported Hardware](#supported-hardware) list. For **IGX-THOR** or **DGX-SPARK**, set `HARDWARE_PROFILE` to that value when the deployment target is that platform, even though `name` will be a GPU part name, not `IGX-THOR` / `DGX-SPARK`. The blueprint does not currently accept custom/free-form profile strings — if the host's GPU is not in the table, the deployment is unsupported on that hardware.

**Required driver versions (match the platform):**

| Platform | Driver version |
|---|---|
| x86 Ubuntu 24.04 | **580.105.08** (required) |
| DGX-SPARK | `580.95.05` |
| IGX-THOR | `580.00` |

##### Install NVIDIA Driver (Ubuntu 24.04)

On **Ubuntu 24.04**, install **NVIDIA Driver 580.105.08**. Do not substitute an unpinned `nvidia-driver-580` unless it resolves to that exact build.

- **Download (580.105.08):** https://www.nvidia.com/en-us/drivers/details/257738/
- **Installation guide:** https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html
- **Driver search by GPU/platform:** https://www.nvidia.com/Download/index.aspx

If `nvidia-smi` fails → driver missing or wrong version. Detect hardware automatically — **do not ask the user what GPU they have**:

```bash
lspci | grep -i nvidia
```

Install matching kernel headers, then install the driver per the guides above (runfile or repository pin to **580.105.08** on Ubuntu 24.04). Example prep for apt-based installs:

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)
```

After installation, load the module if needed and verify:

```bash
sudo modprobe nvidia
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

If `modprobe` exits non-zero, retry `nvidia-smi` anyway — modules may already be loaded. If `nvidia-smi` still fails, check loaded modules and retry:

```bash
lsmod | grep nvidia
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

If it still fails → reboot (`sudo reboot`), then re-run the `nvidia-smi` query above.

**Verify:** `nvidia-smi` must report driver version **580.105.08** on Ubuntu 24.04 and list the GPU(s) correctly.

##### NVIDIA Fabric Manager (when required)

> **Single-GPU systems: SKIP THIS SECTION ENTIRELY.** Fabric Manager is not needed and `nvidia-fabricmanager-580` may even fail to install because it depends on `nvidia-kernel-common-580-server-*` (the server variant of the driver), which conflicts with the standard `nvidia-driver-580` you just installed. If you have one GPU and aren't on an NVLink/NVSwitch system, do not install Fabric Manager.

Fabric Manager is required only on systems where multiple GPUs are connected via **NVLink** or **NVSwitch** (e.g. DGX multi-GPU, HGX baseboards, NVSwitch servers, multi-GPU NVLink topologies, datacenter GPUs in NVLink layouts). It is **not** required for single-GPU systems or multi-GPU **PCIe-only** setups without NVLink/NVSwitch.

Docs: https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html

On **Ubuntu 24.04**, use Fabric Manager **580.105.08** to match the driver (package version typically tracks the driver):

```bash
sudo apt-get update
sudo apt-get install -y nvidia-fabricmanager-580=580.105.08-1
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```

If that exact apt version is unavailable, use the NVIDIA archive for 580.105.08: https://developer.download.nvidia.com/compute/nvidia-driver/redist/fabricmanager/linux-x86_64/fabricmanager-linux-x86_64-580.105.08-archive.tar.xz

#### 2.2 Docker

Reference versions: **Docker Engine 28.3.3** and **Docker Compose plugin v2.39.1+**. If Docker Engine is already **28.3.3** and the Compose plugin is **v2.39.1 or newer**, proceed to §2.3.

```bash
docker --version        # need 28.3.3
docker compose version  # need v2.39.1+
docker ps               # must run without sudo
```

**Install / pin Docker (Ubuntu 24.04):**

The pinned Docker CE packages come from Docker's official apt repository. If `apt` says `docker-ce` or `containerd.io` is unavailable, the Docker apt source is missing; add it first, then install the pinned versions.

```bash
# Remove conflicting distro packages if present. It is okay if apt says none are installed.
sudo apt-get remove -y docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc || true

# Add Docker's official apt repository.
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update

# Optional sanity check: these should print available Docker repo versions.
apt-cache madison docker-ce | grep '28.3.3'
apt-cache madison docker-compose-plugin | grep '2.39.1'
apt-cache madison docker-ce-rootless-extras | grep '28.3.3'

# Install or downgrade to the known-good reference versions.
sudo systemctl stop docker docker.socket 2>/dev/null || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
  docker-ce=5:28.3.3-1~ubuntu.24.04~noble \
  docker-ce-cli=5:28.3.3-1~ubuntu.24.04~noble \
  containerd.io=2.2.2-1~ubuntu.24.04~noble \
  docker-buildx-plugin \
  docker-compose-plugin=2.39.1-1~ubuntu.24.04~noble \
  docker-ce-rootless-extras=5:28.3.3-1~ubuntu.24.04~noble
sudo systemctl enable --now docker

# Optional: hold so unattended-upgrades doesn't move them back
sudo apt-mark hold docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-ce-rootless-extras

docker version --format '{{.Server.Version}}'   # -> 28.3.3
docker compose version --short                  # -> 2.39.1+
```

##### When to pin to Docker 28.3.3 / Compose v2.39.1+

Pin Docker if you hit this specific failure during `docker compose up --pull always`:

```
error from registry: Incorrect Repository Format
```

Then re-run `docker compose up --pull always` after the pinned install succeeds.

**Non-root Docker:**
```bash
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl restart docker
```

**cgroupfs driver** — `/etc/docker/daemon.json` must contain `"exec-opts": ["native.cgroupdriver=cgroupfs"]`. If missing:
```bash
sudo bash -c 'cat > /etc/docker/daemon.json << EOF
{
    "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
EOF'
sudo systemctl daemon-reload && sudo systemctl restart docker
```

#### 2.3 NVIDIA Container Toolkit

Canonical install + verify lives in [`prerequisites.md` § 3 NVIDIA Container Toolkit](prerequisites.md#3-nvidia-container-toolkit). Run that block and re-verify with `docker run --rm --gpus all ubuntu:24.04 nvidia-smi` before continuing.

#### 2.4 Linux Kernel Settings

```bash
sysctl net.ipv6.conf.all.disable_ipv6
sysctl net.core.rmem_max
```

If not set:
```bash
sudo mkdir -p /etc/sysctl.d
sudo bash -c "printf '%s\n' \
  'net.ipv6.conf.all.disable_ipv6 = 1' \
  'net.ipv6.conf.default.disable_ipv6 = 1' \
  'net.ipv6.conf.lo.disable_ipv6 = 1' \
  'net.core.rmem_max = 5242880' \
  'net.core.wmem_max = 5242880' \
  'net.ipv4.tcp_rmem = 4096 87380 16777216' \
  'net.ipv4.tcp_wmem = 4096 65536 16777216' \
  > /etc/sysctl.d/99-vss.conf"
sudo sysctl --system
```

**DGX-SPARK / IGX-THOR / AGX-THOR only** — system cache cleaner and (IGX-Thor) VIC clock boost. These are platform prerequisites that apply to every profile on edge hardware, not just warehouse. Canonical install + verify block lives in [`edge.md` § Cache cleaner (every edge deploy)](edge.md#cache-cleaner-every-edge-deploy).

#### 2.5 IPv6 Localhost Entry

Both `/etc/hosts` and `/etc/cloud/templates/hosts.debian.tmpl` must use `localhost6` for the `::1` entry.

```bash
grep "^::1" /etc/hosts
grep "^::1" /etc/cloud/templates/hosts.debian.tmpl 2>/dev/null || echo "(template not present)"
```

Expected: `::1 localhost6 ip6-localhost ip6-loopback`

If it reads `::1 localhost ip6-localhost ip6-loopback`:
```bash
sudo sed -i 's/^::1 localhost ip6-localhost ip6-loopback/::1 localhost6 ip6-localhost ip6-loopback/' /etc/hosts
if [ -f /etc/cloud/templates/hosts.debian.tmpl ]; then
  sudo sed -i 's/^::1 localhost ip6-localhost ip6-loopback/::1 localhost6 ip6-localhost ip6-loopback/' \
    /etc/cloud/templates/hosts.debian.tmpl
fi
```

#### 2.6 Minimum System Resources

```bash
nproc    # 10+ cores (x86)
free -h  # 64 GB+ RAM
df -h /  # 500 GB+ SSD
```

---

### Phase 3: Interactive Configuration

**Ask these four questions before touching `.env`.**

#### Q1 — Deployment Mode

> "Which mode?
> - **2d** — 2D detection/tracking with **RT-DETR**, no depth
> - **3d** — 3D perception with depth using **Sparse4D**, requires 4-camera dataset
> - **mv3dt** — Multi-View 3D Tracking: per-camera DeepStream perception + **BEV Fusion** across cameras via MQTT, requires 4-camera dataset"

#### Q2 — Blueprint Profile

Refer to the [Profile Variants table](#profile-variants) above for the
profile / mode / dataset matrix instead of restating it here. The question is
just "which profile from that table?".

#### Q3 — Stream Type

Skip for `bp_wh` and `bp_wh_auto_calib`. For `bp_wh_kafka` / `bp_wh_redis`:

> "Which broker — **kafka** or **redis**?"

Variable combinations — pick one row matching the user's Vision-AI variant
and stream type:

| Vision AI | Stream type | `BP_PROFILE` | `STREAM_TYPE` | `SAMPLE_VIDEO_DATASET` | `NUM_STREAMS` |
|---|---|---|---|---|---|
| 2D Vision AI | kafka | `bp_wh_kafka` | `kafka` | `warehouse-loading-dock-3cams-synthetic` | 3 |
| 2D Vision AI | redis | `bp_wh_redis` | `redis` | `warehouse-loading-dock-3cams-synthetic` | 3 |
| 2D Vision AI with Agents | n/a | `bp_wh` | — | `nv-warehouse-4cams` | 4 (also set `LLM_MODE=local`; RTVI VLM is always local) |
| 3D Vision AI | kafka | `bp_wh_kafka` | `kafka` | `warehouse-4cams-20mx20m-synthetic` | 4 |
| 3D Vision AI | redis | `bp_wh_redis` | `redis` | `warehouse-4cams-20mx20m-synthetic` | 4 |
| MV3DT Vision AI | kafka | `bp_wh_kafka` | `kafka` | `warehouse-4cams-20mx20m-synthetic` | 4 |
| MV3DT Vision AI | redis | `bp_wh_redis` | `redis` | `warehouse-4cams-20mx20m-synthetic` | 4 |
| Warehouse Auto-Calibration | n/a | `bp_wh_auto_calib` | — | mode-specific default | mode-specific default (also set `LLM_MODE=none`) |

`3D Vision AI` and `MV3DT Vision AI` intentionally share the same dataset and
stream counts — they differ only at the perception layer (`Sparse4D` vs
per-camera DeepStream + BEV Fusion).

#### Q4 — Deployment Profile

Skip for `bp_wh` and `bp_wh_auto_calib`. For `bp_wh_kafka` / `bp_wh_redis` (any mode):

> "Which profile?
> - **minimal** — excludes ELK, Video Analytics API, monitoring. Recommended for IGX-THOR.
> - **extended** — full deployment."

```bash
MINIMAL_PROFILE="true"   # minimal
MINIMAL_PROFILE=""       # extended
```

#### Q5 — Data Source & Calibration

> "Are you using the **sample dataset** or your **own data** (custom videos / live RTSP streams)?"

**Sample dataset** — calibration files ship with the app data. No extra step needed; proceed to Phase 4.

**Own data** — you need a calibration file before the analytics pipeline can produce meaningful results.

> "Do you already have a calibration JSON file, or do you need to generate one first?"

- **Already have a calibration file** — proceed to Phase 4. You'll mount it in Phase 5 (`.env` config).
- **Need to generate a calibration file** — pick a calibration path based on your video source:

  | You have… | Profile to deploy | What it does |
  |---|---|---|
  | **Video files on disk** | `auto_calib` | Standalone auto-calibration. Upload videos directly to the calibration UI — no nvstreamer, no VST stack needed. |
  | **Live RTSP streams** (or want to use nvstreamer) | `bp_wh_auto_calib_2d` / `bp_wh_auto_calib_3d` / `bp_wh_auto_calib_mv3dt` | Warehouse auto-calibration. Calibrate against RTSP streams served by nvstreamer + VST stack. |

  Deploy the chosen calibration profile first, generate the calibration JSON via the Auto-Calibration UI (`http://<HOST_IP>:5000`).

  > **Note:** Post-calibration cleanup depends on mode. In 2D, Auto-Calibration adds blank `group` and `region` fields to `calibration.json`; they are not required for 2D and should be removed. For 3D / MV3DT, calibration files require camera clustering to populate `sensors[].group` — see [Calibration Generation](#calibration-generation).

  Once the calibration file is ready, redeploy with the full warehouse profile.

---

### Phase 4: Acquire App Data (first run only)

Compose files ship in the repo — **nothing to download for compose**. Only app data may need to be acquired, and only for the source the user chose in [App Data](#app-data).

**Option A — `<repo>/data`:** ensure assets are present at `<repo>/data` and proceed to Phase 5 (`VSS_DATA_DIR=<repo>/data`).

**Option B — custom local path:** confirm the path exists and has the expected `models/` and `videos/` subdirs, then set `VSS_DATA_DIR=<that path>` in Phase 5.

**Option C — NGC `vss-warehouse-app-data`:**

```bash
export NGC_CLI_API_KEY='<your-ngc-api-key>'

ngc registry resource download-version "nvidia/vss-warehouse/vss-warehouse-app-data:<APP_DATA_VERSION>"
cd vss-warehouse-app-data_v<APP_DATA_VERSION>
tar -xvf vss-warehouse-app-data.tar.gz

sudo chmod -R 777 /path/to/vss-warehouse-app-data
```

See [App Data → NGC app-data download](#ngc-app-data-download-optional) for the current version pin.

---

### Phase 5: Configure the warehouse .env

Edit `<repo>/deploy/docker/industry-profiles/warehouse-operations/.env`. Keys below match the actual file — only the values listed need editing for a typical deploy; the rest have working defaults.

```bash
# --- Deployment selectors (Phase 3 answers go here) ---
MODE=<2d|3d|mv3dt>
BP_PROFILE=<bp_wh|bp_wh_kafka|bp_wh_redis|bp_wh_auto_calib>
STREAM_TYPE=<kafka|redis>           # ignored by bp_wh and bp_wh_auto_calib; set for bp_wh_kafka / bp_wh_redis
MINIMAL_PROFILE="true"              # or "" for extended (bp_wh_kafka / bp_wh_redis only)

SAMPLE_VIDEO_DATASET="<dataset-name>"
NUM_STREAMS=<3|4>

# --- Hardware ---
# Valid: H100, L40, L40S, L4, A6000, RTXA6000, RTXA6000ADA, RTXPRO6000BW, IGX-THOR, DGX-SPARK
HARDWARE_PROFILE=H100

# GPU device IDs (defaults shown — change only if you need a non-default layout)
RT_CV_DEVICE_ID='0'                 # perception (always local)
RT_VLM_DEVICE_ID='1'                # RTVI VLM, bp_wh only (always local)
LLM_DEVICE_ID='2'                   # bp_wh + LLM_MODE=local
SHARED_LLM_VLM_DEVICE_ID='2'        # bp_wh + LLM_MODE=local_shared (LLM colocated with RTVI VLM)

# --- LLM (bp_wh only; set LLM_MODE=none for bp_wh_kafka / bp_wh_redis / bp_wh_auto_calib) ---
# RTVI VLM has no mode — it is always deployed locally for bp_wh.
LLM_MODE=local                      # local | local_shared | remote | none
LLM_NAME=nvidia/nvidia-nemotron-nano-9b-v2
LLM_NAME_SLUG=nvidia-nemotron-nano-9b-v2
# LLM_BASE_URL — only when LLM_MODE=remote

# --- RTVI VLM (bp_wh; always local — these are image/model selectors, not a mode toggle) ---
# vss-rtvi-vlm is always deployed for bp_wh (hardcoded in compose profile bp_wh_2d).
VLM_NAME=nim_nvidia_cosmos-reason2-8b_0303-fp8-dynamic-kv8
RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:0303-fp8-dynamic-kv8
RTVI_VLM_MODEL_TO_USE=cosmos-reason2

# --- MQTT (mv3dt only — cross-camera messaging for BEV Fusion) ---
MQTT_HOST=localhost
MQTT_PORT=1883

# --- Paths ---
VSS_APPS_DIR="<repo>/deploy/docker"
# One of: <repo>/data, a custom local path, or extracted NGC app-data dir (see Phase 4)
VSS_DATA_DIR="<repo>/data"

# --- Networking ---
HOST_IP='<HOST_IP>'
EXTERNAL_IP="${HOST_IP}"             # browser-reachable hostname/IP (Brev: secure-link domain)
HAPROXY_PORT=7777                    # ingress for VSS UI

# --- Credentials ---
NGC_CLI_API_KEY='<your-ngc-api-key>'           # required for local NIMs + image pulls
NVIDIA_API_KEY=''                              # required for build.nvidia.com remote endpoints
OPENAI_API_KEY=''                              # required for OpenAI remote endpoints
```

> **DGX-SPARK (SBSA):** swap to the `-sbsa`-tagged image variants. Comment the default `PERCEPTION_TAG="3.2.0-26.05.1"` and uncomment `PERCEPTION_TAG="3.2.0-sbsa-26.05.1"`. Apply the same pattern to `BEV_FUSION_MV3DT_TAG` (mv3dt only), `RTVI_VLM_IMAGE_TAG`, `VST_*_IMAGE_TAG`, and `NVSTREAMER_IMAGE_TAG`.

---

### Phase 6: Pre-flight Check

**Do not proceed if any check fails. Never use `sudo` with `docker` — fix non-root setup (2.2) first.**

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
docker info 2>/dev/null | grep -i "runtimes"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi 2>&1 | head -5
echo "NGC_CLI_API_KEY: ${NGC_CLI_API_KEY:+SET}${NGC_CLI_API_KEY:-NOT SET}"
ngc config current 2>/dev/null | grep -q "apikey" && echo "NGC config: key present" || echo "NGC config: no key"
```

---

### Phase 7: Dry-Run

```bash
cd <repo>/deploy/docker
docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  config | grep "container_name"
```

Show container list to the user, then ask: **"Looks good — deploy now?"**

---

### Phase 8: Deploy

From `<repo>/deploy/docker`, run **[Lifecycle: Bring up](#lifecycle-bring-up)** after the user confirms Phase 7.

---

### Phase 9: Monitor Progress

Run **[Lifecycle: Monitor](#lifecycle-monitor)** using the same `LOG` as Phase 8.

---

## After deploy

See [Access Points](#access-points) for service URLs.

---

## Calibration Generation

Two paths are available to generate calibration files depending on your video source:

| Path | Profile | When to use |
|---|---|---|
| **Standalone Auto-Calibration** (`auto_calib`) | `auto_calib` | You have video files on disk and want to upload them directly to the calibration UI. No nvstreamer or VST stack needed. |
| **Warehouse Auto-Calibration** (`bp_wh_auto_calib`) | `bp_wh_auto_calib_2d` / `bp_wh_auto_calib_3d` / `bp_wh_auto_calib_mv3dt` | You want to calibrate against live RTSP streams served by nvstreamer (using the warehouse dataset and VST stack). |

Both paths deploy `vss-auto-calibration` + `vss-auto-calibration-ui` and produce calibration JSON files consumable by behavior-analytics.

### 2D calibration cleanup

In 2D, Auto-Calibration adds blank `group` and `region` fields to the generated `calibration.json`. These fields are not required for 2D calibration and should be removed before redeploying the full warehouse profile.

### Camera Clustering (3D / MV3DT only)

After calibration is generated via Auto-Calibration, run camera clustering before redeploying the full warehouse profile. For 3D/MV3DT, the required field lives directly on each camera sensor as `sensors[].group`. The warehouse blueprint docker compose setup uses one BEV group, so run the clustering tool with `--n_clusters 1` and then verify the group field is present.

If you have a plan-view image, include it when running clustering so region metadata is derived in the same pass. For multiple independent BEV groups, tune `--n_clusters` and `--max_camera_per_group`; otherwise keep `--n_clusters 1`. Docs: https://docs.nvidia.com/vss/3.1.0/warehouse-docs/3D-profile.html#camera-clustering

### MV3DT-specific configuration updates

When adding new cameras to the MV3DT profile, run the MV3DT utility scripts under `tools/rtvi-cv-mv3dt-utils` after calibration and camera clustering are complete, and before redeploying the full warehouse profile. These scripts generate the MV3DT-specific files consumed by the per-camera tracker and MQTT communication layer:

1. **Camera information files** (`camInfo/<sensor_id>.yml`) — each camera requires a `camInfo` file containing the 3x4 projection matrix and per-class object model dimensions, generated from `calibration.json`.
2. **MQTT publish/subscribe configuration** (`pub_sub_info_config.yml`) — defines the inter-camera communication graph for MV3DT by generating a vision-neighbor graph from camera calibration data.
3. **Tracker configuration** (`ds-mv3dt-tracker-config.yml`) — ensure the `ObjectModelProjection.cameraModelFilepath` section maps each sensor ID to its corresponding `camInfo` file.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ngc: command not found` | Run Phase 1.2 |
| `Missing org` NGC error | Run `ngc config set`, match org to API key |
| NGC auth / `docker login nvcr.io` fails | Re-export `NGC_CLI_API_KEY` and retry |
| `unknown or invalid runtime name: nvidia` | Install NVIDIA Container Toolkit — Phase 2.3 |
| Streams not appearing in VST | `docker logs vss-vios-nvstreamer` |
| Perception not starting | `docker logs vss-rtvi-cv` (2D/3D) or `docker logs vss-rtvi-cv-mv3dt` (MV3DT) — verify models in `$VSS_DATA_DIR/models/` |
| `vss-configurator` health check failing | Wait 60s and recheck (60s start period) |
| Low FPS | GPU oversaturated — reduce `NUM_STREAMS` and redeploy |
| Dataset/mode mismatch | `nv-warehouse-4cams` → `bp_wh` + `MODE=2d`; `warehouse-4cams-20mx20m-synthetic` → `MODE=3d` or `MODE=mv3dt` |
| Redeploy / reset without reinstall | [Redeploy](#redeploy) |