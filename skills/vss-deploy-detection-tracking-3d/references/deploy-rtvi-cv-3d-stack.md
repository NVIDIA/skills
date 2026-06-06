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
| `vss-broker-health-check` | (built locally) | Validates broker + creates topics (one-shot, exits 0) |
| `vss-vios-sensor` | VST sensor image | VST sensor microservice |
| `vss-vios-ingress` | VST | VST ingress (healthy) |
| `vss-vios-streamprocessing` | VST | Records streams; serves the VST video wall |
| `vss-haproxy-ingress` | haproxy | Ingress — **present under MV3DT** (services are still reached on their direct ports) |
| `vss-vios-postgres` (PostgreSQL) | postgres | Backing store for VST sensor-ms |
| `sdr-controller` | (built locally) | SDR + Envoy consolidation (registers streamprocessing) |
| `vss-configurator-mv3dt` (+ `*-init`) | `nvcr.io/nvstaging/vss-core/vss-configurator` | Sensor registration, DeepStream config materialization |
| `vss-vios-nvstreamer-mv3dt` | nvstreamer | RTSP server for sample/videos data |
| **`vss-behavior-analytics-mv3dt`** | analytics | 3D spatial analytics — always under `bp_wh_*_mv3dt`, **not** gated by `MINIMAL_PROFILE` |

> **Auto-calibration is not part of this deploy.** AMC (`vss-auto-calibration` / `-ui`) is **not** in the `bp_wh_kafka_mv3dt` / `bp_wh_redis_mv3dt` profile — those differ from the `bp_wh_auto_calib_mv3dt` / `auto_calib` profiles that enable AMC. When calibration is missing, the [`calibration-workflow.md`](calibration-workflow.md) chain brings AMC up separately (the `auto_calib` profile) and tears it down before this deploy. If you see `vss-auto-calibration` running alongside MV3DT, it's from that separate flow, not this one.

### Extra under extended (`MINIMAL_PROFILE=""`) — needed for VST overlays

| Container | Why |
|---|---|
| `elasticsearch` + `vss-elasticsearch-init` | Backing store for the `mdx-bev` index; VST renders overlays only when this is populated |
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

### Step 0a — Detect stale state from a prior deploy (redeploys only)

A prior deploy leaves two kinds of stale state that get silently reused and break the next `up`. On a fresh host both checks are no-ops. On a redeploy, run them **before** `up`.

**(i) Stale `mdx_*` named volumes.** MV3DT's `kafka` / `elastic` / `postgres` data live in Docker **named volumes** (`mdx_mdx-kafka`, `mdx_vios_pg_data`, …) that bind to a host path baked in **at volume-creation time**. If `VSS_DATA_DIR` has changed since the last deploy, the next `up` fails with `failed to mount local volume: … no such file or directory`. This is detectable with nothing running:

```bash
CUR="${VSS_DATA_DIR%/}"
STALE_VOL=0
if [ -z "${CUR}" ]; then
  echo "VSS_DATA_DIR is not set — source the .env (Step 0) before running this check."
else
  for v in $(docker volume ls -q | grep -E '^mdx_'); do
    dev=$(docker volume inspect "$v" --format '{{.Options.device}}' 2>/dev/null)
    case "$dev" in
      "${CUR}"/*|"") ;;                               # current path or non-bind — fine
      *) echo "STALE volume ${v} -> ${dev}"; STALE_VOL=1 ;;
    esac
  done
  [ "$STALE_VOL" = 1 ] && echo "Stale mdx_* volumes point outside VSS_DATA_DIR=${CUR} — reset with 'down -v' below."
fi
```

> **A passing path-check does *not* mean the volumes are state-free.** This check only flags volumes whose baked path points *outside* the current `VSS_DATA_DIR`. On a same-host redeploy with the **same** `VSS_DATA_DIR`, the `mdx_*` volumes pass silently yet still carry the prior deploy's VST Postgres sensor records (`mdx_vios_pg_data`) and Kafka offsets (`mdx_mdx-kafka`) — which is a common cause of `Active sources : 0` after an otherwise clean-looking redeploy. So treat this check as "will the volume mount," not "is it empty." For any **clean-redeploy intent** (new dataset, changed camera set/names, or any "stuck at 0 sources" reset), reset the volumes with `down -v` regardless of the path result — see (ii) below and the clean-redeploy callout before Step 3.

**(ii) Stale VST sensor records.** A prior deploy's VST Postgres DB and configurator state survive a plain `docker compose down`, so old sensor records (a different dataset, a removed camera, or empty/offline entries) get reused and perception stalls at `Active sources : 0` while containers still look healthy. Only checkable when VST is already up:

```bash
VST_HOST="${HOST_IP:-localhost}"; VST_PORT="${VST_PORT:-30888}"
CAL_DIR="${VSS_APPS_DIR}/industry-profiles/warehouse-operations/warehouse-mv3dt-app/calibration/sample-data/${SAMPLE_VIDEO_DATASET}"

if docker ps --format '{{.Names}}' | grep -q '^vss-vios-sensor$'; then
  EXISTING=$(curl -sf "http://${VST_HOST}:${VST_PORT}/vst/api/v1/sensor/list" 2>/dev/null \
    | jq -r '.[].name' 2>/dev/null | sort)
  EXPECTED=$(jq -r '.sensors[].id' "${CAL_DIR}/calibration.json" 2>/dev/null | sort)
  echo "VST already running."
  echo "Registered sensors:"; echo "${EXISTING:-(none)}"
  echo "Expected for ${SAMPLE_VIDEO_DATASET}:"; echo "${EXPECTED:-(unknown)}"
  if [ -z "${EXPECTED}" ]; then
    # calibration.json wasn't readable — skip the comparison rather than flag a
    # false-positive that would recommend a destructive down -v. Fix CAL_DIR /
    # SAMPLE_VIDEO_DATASET first (these come from the Step 0 .env sourcing).
    echo "Could not read expected sensors from ${CAL_DIR}/calibration.json — skipping stale-sensor check."
  elif [ "${EXISTING}" != "${EXPECTED}" ]; then
    echo "STALE / MISMATCHED VST state — the registered sensors do not match this dataset."
    echo "A scoped reset is recommended before deploying (resets VST Postgres + named volumes):"
    echo "  docker compose -f compose.yml --env-file industry-profiles/warehouse-operations/.env down -v"
    echo "  bash scripts/cleanup_all_datalog.sh -e industry-profiles/warehouse-operations/.env --skip-revert-from-oldest-backup"
  else
    echo "VST sensor set matches the expected dataset — no reset needed."
  fi
fi
```

`down -v` is destructive (drops the VST DB and broker volumes), so **confirm with the user via `AskUserQuestion` before running it.** Full discussion of `down -v` semantics is in [`teardown.md`](teardown.md); the targeted sensor-trim alternative is in [`configure-cameras.md`](configure-cameras.md) Step 5.

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

After extraction, run the `mkdir -p` + scoped-ACL `data_log` permission step from [`../SKILL.md`](../SKILL.md) Prerequisites §4 before deploy — kafka / elasticsearch / redis won't start without it.

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

### SBSA note (DGX-SPARK only)

The only platform that needs an `-sbsa` image tag is **DGX-SPARK**, and only for the **Perception** image. Every other platform uses the shipped non-SBSA tags — including **AGX-THOR / IGX-THOR** (ARM64, but confirmed **not** to need SBSA), GB200, and all x86 GPUs. Do not infer SBSA from the platform being ARM64.

On DGX-SPARK, switch `PERCEPTION_TAG` to its `-sbsa` variant — comment the default and uncomment the `-sbsa` line shipped beside it in `.env`:

```bash
# PERCEPTION_TAG ships an SBSA variant for DGX-SPARK — comment the default, uncomment the -sbsa line:
# PERCEPTION_TAG="3.2.0-26.05.1"
PERCEPTION_TAG="3.2.0-sbsa-26.05.1"
```

The `blueprint-configurator` enforces this: on `HARDWARE_PROFILE=DGX-SPARK` it validates that `PERCEPTION_TAG` contains `sbsa`.

**BEV Fusion needs no SBSA build.** `BEV_FUSION_MV3DT_TAG` is a single image that runs on all platforms including DGX-SPARK — leave it at its shipped tag. There is no `-sbsa` variant for it; don't hand-construct one (the pull would fail).

Treat the shipped `.env` as the source of truth — swap only keys that carry a commented `-sbsa` line (currently `PERCEPTION_TAG`). The per-key list also lives in `vss-deploy-profile/references/warehouse.md` (search for "SBSA").

## Step 2 — Dry-run

```bash
cd "${VSS_APPS_DIR}"
docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env \
  config | grep -E '(container_name|profiles:)' | head -80
```

> **Filtering compose noise.** `docker compose config`/`up` prints a `level=warning msg="The \"VAR\" variable is not set. Defaulting to a blank string."` line for every variable that belongs to a profile you're **not** deploying (`EVAL_*`, `LVS_*`, `MILVUS_*`, `GF_*`, `VST_MCP_URL`, …). For MV3DT these are **expected and benign** — they are not a problem. To see only the lines that matter, drop them:
>
> ```bash
> docker compose -f compose.yml --env-file industry-profiles/warehouse-operations/.env config 2>&1 >/dev/null \
>   | grep -v 'variable is not set'
> # Empty output = no real errors. Anything that still prints here is actionable —
> # e.g. "couldn't find env file: ..." means a path in .env is wrong; fix before deploying.
> ```

**Extended** (`MINIMAL_PROFILE=""`) — expect ~18–22 `container_name:` entries. Confirm these are present in addition to the always-deployed core:

- `elasticsearch` + `vss-elasticsearch-init`
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

> **Redeploying? `down -v` alone is not a clean reset.** It resets the named volumes (Kafka log, VST Postgres), but host-side runtime state under `${VSS_DATA_DIR}/data_log` (VST / SDRC / configurator / broker state) is left in place and gets reused — which can leave MV3DT at `Active sources : 0` even though every container is healthy. For a truly fresh redeploy (new dataset, changed camera set/names, or any "stuck at 0 sources" reset), clear **both**:
>
> ```bash
> cd "${VSS_APPS_DIR}"
> # 1. Reset containers + named volumes
> docker compose -f compose.yml --env-file industry-profiles/warehouse-operations/.env down -v
>
> # 2. Clear host-side data_log — rotate it (non-destructive, keeps a backup):
> ts=$(date +%Y%m%d_%H%M%S)
> mv "${VSS_DATA_DIR}/data_log" "${VSS_DATA_DIR}/data_log.bak.${ts}"
> #    ...or delete in place with the bundled script:
> #    bash scripts/cleanup_all_datalog.sh -e industry-profiles/warehouse-operations/.env --skip-revert-from-oldest-backup
>
> # 3. Recreate the data_log subdirs and re-apply the scoped ACLs — see SKILL.md Prerequisites §4
> #    (mkdir the subdirs, then setfacl for UIDs 70/999/1000 — NOT chmod 777).
> ```
>
> Then redeploy (below) and confirm with the **readiness gate** in [`verify-and-view.md`](verify-and-view.md) (Step 4b) — `Active sources == NUM_STREAMS` and growing `mdx-raw`/`mdx-bev` offsets — not just container health. Plain `docker compose down` (no `-v`, no `data_log` clear) is only for restarting against the **same** dataset. Full teardown discussion: [`teardown.md`](teardown.md). For first-time deploys on a clean host, skip this and go straight to the commands below.

```bash
cd "${VSS_APPS_DIR}"

# NGC login (first time on this host)
docker login --username '$oauthtoken' --password "${NGC_CLI_API_KEY}" nvcr.io

# Fail fast: confirm the key can access the gated vss-core images BEFORE the long background up.
# Refs come from the resolved compose, so this tracks PERCEPTION_TAG / BEV_FUSION_MV3DT_TAG
# (the -sbsa swap, and any PERCEPTION_IMAGE / BEV_FUSION_MV3DT_IMAGE org override) automatically.
# manifest inspect checks registry access only — no layer download — so it stays fast even though
# the perception image is multi-GB (the real pull happens in the backgrounded `up --pull always`).
VSS_CORE_IMAGES=$(docker compose -f compose.yml \
  --env-file industry-profiles/warehouse-operations/.env config --images \
  | grep -E 'nvcr\.io/.*/vss-core/' | sort -u)
if [ -z "$VSS_CORE_IMAGES" ]; then
  echo "No vss-core images in the resolved compose — confirm MODE=mv3dt and COMPOSE_PROFILES resolved to bp_wh_kafka_mv3dt before continuing."
  exit 1
fi
for img in $VSS_CORE_IMAGES; do
  echo "Checking access: $img"
  if ! docker manifest inspect "$img" >/dev/null 2>&1; then
    echo
    echo "NGC login succeeded, but this key does not have access to the required MV3DT image:"
    echo "  $img"
    echo "vss-core is published under both nvidia/ and nvstaging/, and your key may only see one."
    echo "Either point PERCEPTION_IMAGE / BEV_FUSION_MV3DT_IMAGE in .env at the org your key can pull,"
    echo "or provide an NGC key with vss-core access, then retry."
    exit 1
  fi
done

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
- `vss-broker-health-check` reaches `Exit 0` once the broker is up and topics are seeded. If it stays running, the broker is still booting.
- Under extended: `vss-elasticsearch-init`, `vss-kibana-init-mv3dt`, and `vss-import-calibration-output-mv3dt` are one-shot init containers and reach `Exit 0` after completing — leave them alone.

Once perception logs an FPS line and `/tmp/fusion_ready` exists (check via `docker inspect`), continue to [`verify-and-view.md`](verify-and-view.md).

## When deploy fails

- Image pull 401 / 403 → the Step 3 access check should have caught this before bring-up; if it slips through, re-run `docker login nvcr.io` and verify `ngc registry image list "nvstaging/vss-core/*"` (or `nvidia/vss-core/*`) returns results. If only one org resolves, point `PERCEPTION_IMAGE` / `BEV_FUSION_MV3DT_IMAGE` in `.env` at that org.
- `error from registry: Incorrect Repository Format` mid-pull → Docker/Compose version incompatibility with the bare-tag local-build services in `services/infra/compose.yml`. See [`troubleshooting.md`](troubleshooting.md) — "`error from registry: Incorrect Repository Format` during compose pull" for a version-independent pre-build workaround and the Docker-pin alternative.
- `unknown or invalid runtime name: nvidia` → install NVIDIA Container Toolkit (`vss-deploy-profile/references/prerequisites.md` §2.3).
- `redis ... Can't open the log file: Permission denied`, `kafka ... /tmp/kafka-data/cluster_id: Permission denied`, or elasticsearch `AccessDeniedException` → `$VSS_DATA_DIR/data_log` isn't writable by the container UIDs. Run the `mkdir -p` + scoped-ACL permission step from [`../SKILL.md`](../SKILL.md) Prerequisites §4 and redeploy. Don't recursive-chown.
- `vss-configurator-mv3dt` exits 1 immediately → almost always `VSS_DATA_DIR` pointing at the repo instead of the extracted app-data directory. See Step 0 checks.
- Containers in `Created` state forever → almost always the same `VSS_DATA_DIR` issue. Stop everything, fix `.env`, redeploy.
- Profile mismatch (e.g. expected containers not in `docker compose config`) → confirm `MODE=mv3dt`, `BP_PROFILE` is one of `bp_wh_kafka` / `bp_wh_redis`. Other failure modes → [`troubleshooting.md`](troubleshooting.md).

When you need to start clean: [`teardown.md`](teardown.md).
