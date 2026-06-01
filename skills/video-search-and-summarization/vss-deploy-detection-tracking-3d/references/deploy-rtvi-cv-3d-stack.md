# Deploy the RTVI-CV-3D (MV3DT) stack

The actual `docker compose up` recipe. Parent: [`../SKILL.md`](../SKILL.md). Run this **after** Q0/Q1/Q2/Q3 in SKILL.md resolved, calibration is on disk (either ship-with-repo for sample, or landed by [`calibration-workflow.md`](calibration-workflow.md), or user-supplied), and [`configure-cameras.md`](configure-cameras.md) has synced `NUM_STREAMS` to the calibration file count.

## What this brings up

`MODE=mv3dt` + `BP_PROFILE=bp_wh_kafka` (or `_redis`) resolves the compose profile to `bp_wh_kafka_mv3dt` (or `bp_wh_redis_mv3dt`). `MINIMAL_PROFILE` then toggles the `_extended` services on top:

### Always deployed (under either profile)

| Container | Image | Role |
|---|---|---|
| `vss-rtvi-cv-mv3dt` | `nvcr.io/nvstaging/vss-core/vss-rt-cv:${PERCEPTION_TAG}` | Per-camera DeepStream perception |
| `vss-rtvi-cv-bev-fusion` | `nvcr.io/nvstaging/vss-core/vss-rt-cv-mv3dt-bev-fusion:${BEV_FUSION_MV3DT_TAG}` | BEV Fusion — fuses per-camera detections to a single BEV frame |
| `mosquitto` | `eclipse-mosquitto:2` | MQTT bus between perception and fusion |
| `kafka` *or* `redis` | (per `STREAM_TYPE`) | Carries `mdx-raw` (input) and `mdx-bev` (output) |
| `broker-health-check` | (built locally) | Validates broker + creates topics |
| `vss-vios-sensor` (`sensor-ms-mv3dt`) | VST sensor image | VST sensor microservice |
| `centralizedb` (PostgreSQL) | postgres | Backing store for VST sensor-ms |
| `vss-configurator-mv3dt` (+ `*-init`) | `nvcr.io/nvstaging/vss-core/vss-configurator` | Sensor registration, DeepStream config materialization |
| `vss-vios-nvstreamer-mv3dt` | nvstreamer | RTSP server for sample/videos data |
| `vss-auto-calibration` (+ `-ui`) | AMC images | Calibration UI on port 5000 (always under `bp_wh_*_mv3dt`) |
| **`vss-behavior-analytics-mv3dt`** | analytics | 3D spatial analytics — always under `bp_wh_*_mv3dt`, **not** gated by `MINIMAL_PROFILE` |

### Extra under extended (`MINIMAL_PROFILE=""`) — needed for VST overlays

| Container | Why |
|---|---|
| `elasticsearch` + `elasticsearch-init-container` | Backing store for the `mdx-bev` index; VST renders overlays only when this is populated |
| `logstash` | Pipes broker metadata → Elasticsearch |
| `kibana` + `vss-kibana-init-mv3dt` | Dashboards (also needed for overlay rendering) |
| `vss-video-analytics-api-mv3dt` | Serves overlay data to VST |
| `vss-import-calibration-output-mv3dt` | Imports the `calibration.json` into Elasticsearch |

These services share a single `${MINIMAL_PROFILE:+_extended}` gate — they come up together as a unit, not individually selectable.

**Recommendation: default to extended** for any user who wants a complete e2e experience including overlays. Drop to minimal only when explicitly asked for the smallest footprint (edge / Thor / "just give me the topic data").

## Step 0 — Pre-deploy host-path checks

Don't trust `docker compose config` to catch missing bind-mount sources — it doesn't validate host paths. Run these first:

```bash
ENV_FILE="${VSS_APPS_DIR}/industry-profiles/warehouse-operations/.env"

# Re-source key vars from .env so we can check them
set -a; . "${ENV_FILE}"; set +a

# 1. App-data layout
for sub in videos models data_log; do
  test -d "${VSS_DATA_DIR}/${sub}" || { echo "ERROR: ${VSS_DATA_DIR}/${sub} missing — VSS_DATA_DIR is not pointing at extracted vss-warehouse-app-data"; exit 1; }
done

# 2. Dataset videos
test -d "${VSS_DATA_DIR}/videos/${SAMPLE_VIDEO_DATASET}" \
  || { echo "ERROR: ${VSS_DATA_DIR}/videos/${SAMPLE_VIDEO_DATASET} missing"; exit 1; }
VIDEO_COUNT=$(ls "${VSS_DATA_DIR}/videos/${SAMPLE_VIDEO_DATASET}/"*.mp4 2>/dev/null | wc -l)
echo "Found ${VIDEO_COUNT} videos under ${VSS_DATA_DIR}/videos/${SAMPLE_VIDEO_DATASET}/"

# 3. Calibration mount
CAL_DIR="${VSS_APPS_DIR}/industry-profiles/warehouse-operations/warehouse-mv3dt-app/calibration/sample-data/${SAMPLE_VIDEO_DATASET}"
test -f "${CAL_DIR}/calibration.json" || { echo "ERROR: ${CAL_DIR}/calibration.json missing"; exit 1; }
CAM_COUNT=$(ls "${CAL_DIR}/camInfo/"*.{yml,yaml} 2>/dev/null | wc -l)
echo "Found ${CAM_COUNT} calibration files under ${CAL_DIR}/camInfo/"

# 4. The configurator enforces min(NUM_STREAMS, HARDWARE_PROFILE.max_streams_supported)
#    and trims excess videos to match. See SKILL.md Prerequisites §3.
echo "NUM_STREAMS=${NUM_STREAMS}, HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "If max_streams_supported for ${HARDWARE_PROFILE}.mv3dt is < ${NUM_STREAMS},"
echo "the configurator will trim videos to that cap at deploy time."
```

If videos < camera count and `HARDWARE_PROFILE.mv3dt.max_streams_supported` < camera count, the deploy will appear to succeed but you'll only get a subset of streams. Fix one of: source missing videos, raise `HARDWARE_PROFILE`-supported cap, or lower expectations.

### Step 0b — Patch hardcoded `streamprocessing` mounts (custom datasets only)

`services/vios/streamprocessing/docker-compose.yaml` hardcodes two bind-mount sources to `sample-data/warehouse-4cams-20mx20m-synthetic/` regardless of `SAMPLE_VIDEO_DATASET`:

```yaml
# Under the `streamprocessing-ms-mv3dt:` service block — `streamprocessing-ms-3d:` mirrors the same pattern for MODE=3d.
- ${VSS_APPS_DIR}/.../calibration/sample-data/warehouse-4cams-20mx20m-synthetic/calibration.json:/home/vst/vst_release/configs/calibration.json
- ${VSS_APPS_DIR}/.../calibration/sample-data/warehouse-4cams-20mx20m-synthetic/images/Top.png:/home/vst/vst_release/configs/Top.png
```

VST reads from `/home/vst/vst_release/configs/calibration.json` when rendering 3D bbox overlays on each camera stream — so for any `SAMPLE_VIDEO_DATASET` other than `warehouse-4cams-20mx20m-synthetic`, **VST overlays project with the sample warehouse's `cameraMatrix` instead of yours**, even though every other consumer (perception, behavior-analytics, video-analytics-api) correctly uses your dataset's calibration. Symptom: bbox positions wildly off on the VST video wall, top-view widget shows the sample warehouse layout instead of yours, AMC/Kibana overlays look fine.

Idempotent patch — no-op when slug is already the literal sample, no-op after a prior patch:

```bash
COMPOSE_SP="${VSS_APPS_DIR}/services/vios/streamprocessing/docker-compose.yaml"

if grep -q 'sample-data/warehouse-4cams-20mx20m-synthetic/calibration\.json' "${COMPOSE_SP}"; then
  sed -i 's|sample-data/warehouse-4cams-20mx20m-synthetic/calibration\.json|sample-data/${SAMPLE_VIDEO_DATASET}/calibration.json|g' "${COMPOSE_SP}"
  sed -i 's|sample-data/warehouse-4cams-20mx20m-synthetic/images/Top\.png|sample-data/${SAMPLE_VIDEO_DATASET}/images/Top.png|g' "${COMPOSE_SP}"
  echo "Patched streamprocessing compose: sample-data path now resolves via \${SAMPLE_VIDEO_DATASET}"
else
  echo "streamprocessing compose already patched (or sample dataset in use) — no change"
fi
```

If the stack is **already running** when you discover this (Step 5 in [`verify-and-view.md`](verify-and-view.md) is showing the sample warehouse layout), apply the patch and recreate the affected container in place — no need to bring the full stack down:

```bash
cd "${VSS_APPS_DIR}"
docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  up -d --no-deps --force-recreate streamprocessing-ms-mv3dt

# VST's per-tab session caches the sensorIds, which change on streamprocessing recreate
# → hard-refresh the VST tab (Ctrl+Shift+R) so the cached streamId is dropped.
```

This is an upstream-bug workaround. When the compose source is fixed (`${SAMPLE_VIDEO_DATASET}` instead of the literal), this step becomes a true no-op and can be dropped from the skill.

## Step 1 — Env recipe

Edit `${VSS_APPS_DIR}/industry-profiles/warehouse-operations/.env`. The shipped `.env` defaults to **2D** (`MODE=2d`, `BP_PROFILE=bp_wh`, `HARDWARE_PROFILE=H100`, paths as placeholders, `NGC_CLI_API_KEY=''`) — you must change at least `MODE`, `BP_PROFILE`, paths, `HOST_IP`, and `NGC_CLI_API_KEY` for MV3DT. Confirm every key below:

> **Also set `LLM_MODE=none`.** Some shipped `.env` variants default `LLM_MODE=local`, which adds `llm_local_<slug>` to `COMPOSE_PROFILES` and pulls up the local LLM NIM stack — unwanted for MV3DT-only and a heavy GPU/model download. MV3DT needs no LLM/VLM, so set both `LLM_MODE=none` and `VLM_MODE=none`.

```bash
# All keys below live in industry-profiles/warehouse-operations/.env — locate by name (line numbers drift across releases).
# Deployment selectors
MODE=mv3dt
BP_PROFILE=bp_wh_kafka                      # or bp_wh_redis
STREAM_TYPE=kafka                           # match BP_PROFILE
MINIMAL_PROFILE=""                          # EXTENDED (default for overlays)
# MINIMAL_PROFILE="true"                    # uncomment for minimal (no overlays)

# Dataset + stream count
SAMPLE_VIDEO_DATASET="<your-dataset-slug>"  # see "Slug" note below
NUM_STREAMS=4                               # must equal camInfo count

# Hardware — use the slug from SKILL.md Prerequisites §3 (canonical keys live in blueprint_config.yml)
HARDWARE_PROFILE=H100                       # see SKILL.md Prerequisites §3 table
RT_CV_DEVICE_ID='0'                         # GPU for perception
LLM_MODE=none                               # no LLM/VLM for MV3DT
VLM_MODE=none

# Paths (REQUIRED)
VSS_APPS_DIR="<repo>/deploy/docker"         # your checkout's deploy/docker
VSS_DATA_DIR="<extracted-vss-warehouse-app-data>"  # NOT the repo path
HOST_IP='<browser-reachable-IP>'            # not localhost
EXTERNAL_IP="${HOST_IP}"

# MQTT (mv3dt only)
MQTT_HOST=localhost
MQTT_PORT=1883

# NGC credential for image pulls
NGC_CLI_API_KEY='<your-ngc-key>'
```

`COMPOSE_PROFILES` is computed automatically by the .env (search for `^COMPOSE_PROFILES=`): `${BP_PROFILE}_${MODE},llm_${LLM_MODE}_${LLM_NAME_SLUG}` → for MV3DT this resolves to `bp_wh_kafka_mv3dt,llm_none_none`.

### `VSS_DATA_DIR` — what to point it at

This is the directory containing the **extracted** `vss-warehouse-app-data` tarball — **separate from the repo**. Expected layout:

```
<extracted-dir>/
├── videos/<dataset>/        Camera*.mp4 or cam_*.mp4
├── models/mv3dt/BodyPose3DNet/   TRT/onnx weights
├── data_log/                 broker / VST log dir (created at deploy)
└── auto-calib/vggt/          optional VGGT model
```

If you haven't extracted it yet, discover the latest tag rather than relying on a pinned one — release cuts and staging snapshots get re-published over time, and the most recent tag is rarely the one any doc still references:

```bash
export NGC_CLI_API_KEY='<your-key>'

# Discover what's actually published for your key. Try both orgs — most
# keys see one or the other (not both). Release tags follow <maj>.<min>.<patch>;
# staging tags are dated (e.g. v3.2.0-MMDDYYYY). Pick the most recent
# UPLOAD_COMPLETE row that matches the perception/fusion image tag base
# in .env (PERCEPTION_TAG=<base>-...). Mismatching app-data and image
# versions is a common silent-deploy bug.
NGC_CLI_ORG=nvidia    ngc registry resource list "nvidia/vss-warehouse/vss-warehouse-app-data:*"    --format_type ascii | head -10
NGC_CLI_ORG=nvstaging ngc registry resource list "nvstaging/vss-warehouse/vss-warehouse-app-data:*" --format_type ascii | head -10

ORG=<nvidia-or-nvstaging>
TAG=<picked-tag>
NGC_CLI_ORG="$ORG" ngc registry resource download-version "${ORG}/vss-warehouse/vss-warehouse-app-data:${TAG}"

# The tarball extracts into a nested vss-warehouse-app-data/ directory — flatten it.
cd "vss-warehouse-app-data_v${TAG#v}" || cd "vss-warehouse-app-data_${TAG}"
tar -xvf vss-warehouse-app-data.tar.gz
sudo chmod -R a+rX /path/to/vss-warehouse-app-data
# Then point VSS_DATA_DIR at /path/to/vss-warehouse-app-data
```

After extraction, run the `mkdir -p` + `chmod -R 777 $VSS_DATA_DIR/data_log` step from [`../SKILL.md`](../SKILL.md) Prerequisites §4 before deploy — kafka / elasticsearch / redis won't start without it.

> Always verify the video count before deploy — the pre-flight check above prints it. If the count is lower than the dataset name implies (e.g. fewer than the four cameras in `warehouse-4cams-20mx20m-synthetic/`), the GPU's `mv3dt` cap (SKILL.md Prerequisites §3) determines whether this affects you: if the cap is at or below the present video count, the configurator's `keep_count` op uses what's there; if the cap is higher, source the additional cams separately before deploying.

### `SAMPLE_VIDEO_DATASET` slug

Drives the calibration mount path:

```
${VSS_APPS_DIR}/industry-profiles/warehouse-operations/warehouse-mv3dt-app/calibration/sample-data/${SAMPLE_VIDEO_DATASET}/
├── calibration.json
├── camInfo/(Camera*|cam_*).{yml|yaml}
└── images/
```

| User path | Slug to set |
|---|---|
| Sample dataset | `warehouse-4cams-20mx20m-synthetic` (ship-with-repo) |
| User videos (after AMC) | Whatever the user chose in Q3 (e.g. `customer-aisle-4cams`) — [`calibration-workflow.md`](calibration-workflow.md) lands files there |
| User RTSP (after AMC) | Same — Q3 slug |

### SBSA (DGX-SPARK / ARM64) note

Swap to `-sbsa` image tags. From the shipped `.env`:

```bash
# In .env, comment the multi-arch tag and uncomment the -sbsa variant next to it:
# PERCEPTION_TAG="3.2.0-sbsa-26.05.1"
# BEV_FUSION_MV3DT_TAG="3.2.0-26.05.3-sbsa"
```

Apply the same pattern to `RTVI_VLM_IMAGE_TAG`, `VST_*_IMAGE_TAG`, and `NVSTREAMER_IMAGE_TAG` if those keys are set in your `.env`. Per-key list lives in `vss-deploy-profile/references/warehouse.md` (search for "SBSA").

## Step 2 — Dry-run

```bash
cd "${VSS_APPS_DIR}"
docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  config | grep -E '(container_name|profiles:)' | head -80
```

**Extended** (`MINIMAL_PROFILE=""`) — expect ~18–22 `container_name:` entries. Confirm these are present in addition to the always-deployed core:

- `elasticsearch` + `elasticsearch-init-container`
- `logstash`
- `kibana` + `vss-kibana-init-mv3dt`
- `vss-video-analytics-api-mv3dt`
- `vss-import-calibration-output-mv3dt`

**Minimal** (`MINIMAL_PROFILE="true"`) — expect ~12–15 entries; the above five are absent.

In both modes, sanity check these MV3DT-core containers are present:

- `vss-rtvi-cv-mv3dt`
- `vss-rtvi-cv-bev-fusion`
- `mosquitto`
- `kafka` *or* `redis`
- `vss-vios-sensor`
- `vss-configurator-mv3dt`
- `vss-vios-nvstreamer-mv3dt`
- `vss-behavior-analytics-mv3dt` (always under `bp_wh_*_mv3dt`)

If any of the core are missing, `COMPOSE_PROFILES` is wrong — re-check `MODE` + `BP_PROFILE` + `STREAM_TYPE`.

## Step 3 — Deploy

> **Redeploying after a dataset / camera-set change?** Reset the MV3DT named volumes first so VST sensor records re-initialize from the new calibration:
>
> ```bash
> cd "${VSS_APPS_DIR}"
> docker compose -f compose.yml \
>   --env-file industry-profiles/warehouse-operations/.env down -v
> ```
>
> Plain `docker compose down` (without `-v`) preserves the Kafka log and the VST Postgres DB by design — useful for restarting against the same dataset, less useful when the camera names or count have changed. Full discussion: [`teardown.md`](teardown.md). For first-time deploys on a clean host, skip this and go straight to the commands below.

```bash
cd "${VSS_APPS_DIR}"

# NGC login (first time on this host)
docker login --username '$oauthtoken' --password "${NGC_CLI_API_KEY}" nvcr.io

# Bring up (~10–15 min first run — PERCEPTION image pull + BodyPose3DNet TRT engine build)
LOG=${LOG:-/tmp/mv3dt-deploy.log}
nohup docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  up --detach --pull always --force-recreate --build \
  > "$LOG" 2>&1 &
echo "Compose PID $! — logging to $LOG"
```

## Step 4 — Watch the bring-up

Poll every ~60s:

```bash
tail -20 "$LOG"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'mv3dt|mosquitto|kafka|redis|elasticsearch|logstash|kibana|vios|centralizedb|configurator|behavior'
```

Expected first-run timing:

- `vss-rtvi-cv-mv3dt` sits in `(starting)` for 5–10 min while DeepStream builds the BodyPose3DNet TensorRT engine. Tail `docker logs -f vss-rtvi-cv-mv3dt` for `Build engine successfully` lines.
- `vss-rtvi-cv-bev-fusion` reports unhealthy until `/tmp/fusion_ready` is created — the health check probes that sentinel file.
- `broker-health-check` reaches `Exit 0` once the broker is up and topics are seeded. If it stays running, the broker is still booting.
- Under extended: `elasticsearch-init-container`, `vss-kibana-init-mv3dt`, and `vss-import-calibration-output-mv3dt` are one-shot init containers and reach `Exit 0` after completing — leave them alone.

Once perception logs an FPS line and `/tmp/fusion_ready` exists (check via `docker inspect`), continue to [`verify-and-view.md`](verify-and-view.md).

## When deploy fails

- Image pull 401 / 403 → re-run `docker login nvcr.io`; verify `ngc registry image list "nvstaging/vss-core/*"` (or `nvidia/vss-core/*`) returns results.
- `error from registry: Incorrect Repository Format` mid-pull → Docker/Compose version incompatibility with the bare-tag local-build services in `services/infra/compose.yml`. See [`troubleshooting.md`](troubleshooting.md) — "`error from registry: Incorrect Repository Format` during compose pull" for a version-independent pre-build workaround and the Docker-pin alternative.
- `unknown or invalid runtime name: nvidia` → install NVIDIA Container Toolkit (`vss-deploy-profile/references/prerequisites.md` §2.3).
- `redis ... Can't open the log file: Permission denied`, `kafka ... /tmp/kafka-data/cluster_id: Permission denied`, or elasticsearch `AccessDeniedException` → `$VSS_DATA_DIR/data_log` perms weren't opened up. Run the `mkdir -p` + `chmod -R 777` step from [`../SKILL.md`](../SKILL.md) Prerequisites §4 and redeploy. Don't recursive-chown.
- `vss-configurator-mv3dt` exits 1 immediately → almost always `VSS_DATA_DIR` pointing at the repo instead of the extracted app-data directory. See Step 0 checks.
- Containers in `Created` state forever → almost always the same `VSS_DATA_DIR` issue. Stop everything, fix `.env`, redeploy.
- Profile mismatch (e.g. expected containers not in `docker compose config`) → confirm `MODE=mv3dt`, `BP_PROFILE` is one of `bp_wh_kafka` / `bp_wh_redis`. Other failure modes → [`troubleshooting.md`](troubleshooting.md).

When you need to start clean: [`teardown.md`](teardown.md).
