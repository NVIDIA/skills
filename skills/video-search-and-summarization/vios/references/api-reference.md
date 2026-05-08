# VIOS REST API Reference

## Sample data bootstrap

VIOS stores videos uploaded by the user. For requests that reference a
**"sample"** video by friendly name (e.g. *"the sample warehouse
video"*, *"sample-warehouse-ladder"*, *"warehouse_safety_0001"*) the
expected file is one of the 8 mp4s shipped in NGC bundle
`nvidia/vss-developer/dev-profile-sample-data:3.1.0`. Before any
upload-style request, ensure the bundle is extracted locally:

```bash
SAMPLE_DIR="/tmp/vss-sample-data/dev-profile-sample-data"

if [ ! -d "$SAMPLE_DIR" ]; then
  mkdir -p /tmp/vss-sample-data
  cd /tmp/vss-sample-data

  # NGC CLI required (export NGC_CLI_API_KEY first if not already set).
  ngc registry resource download-version \
    nvidia/vss-developer/dev-profile-sample-data:3.1.0 \
    --org nvidia --team vss-developer

  # Bundle ships as a single tar.gz inside dev-profile-sample-data_v3.1.0/.
  tar -xzf dev-profile-sample-data_v3.1.0/dev-profile-sample-data.tar.gz
fi

ls "$SAMPLE_DIR"/  # verify expected mp4s present
```

Bundle contents (use these filenames verbatim when asked for *"the
&lt;name&gt; video"*):

| Friendly name in user query | Local filename |
|---|---|
| sample warehouse video | `warehouse_sample.mp4` |
| sample-warehouse-ladder | `sample-warehouse-ladder.mp4` |
| warehouse safety 1 / 2 | `warehouse_safety_0001.mp4` / `warehouse_safety_0002.mp4` |
| sample-sim-traffic | `sample-sim-traffic.mp4` |
| sample-sim-jaywalking | `sample-sim-jaywalking.mp4` |
| sample-sim-box-conveyor | `sample-sim-box-conveyor.mp4` |
| sample-drone-bridge | `sample-drone-bridge.mp4` |

If the user names a video that isn't in this list (e.g. *"airport
video"*, *"neon-pink monster truck"*), do **not** substitute a
similar-sounding bundle file — list the available names back to the
user and ask which one they meant. Don't invent paths or fabricate
upload responses.

`NGC_CLI_API_KEY` must be set in the environment for `ngc registry`
calls to authenticate. The variable is provided by the deploy/eval
harness; if it's missing, fail with the actionable error rather than
trying to proceed.

---


## Operations

### 1. Version / Health Check

Lightweight endpoint to verify the VST backend is reachable. Used as the availability check before any other API call.

```bash
curl -sf --connect-timeout 5 "http://<VST_ENDPOINT>/vst/api/v1/sensor/version" | jq .
```
Response: version metadata for the running VST service.

---

### 2. Sensor List

**List all sensors:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/list" | jq .
```
Response: array of sensor objects. Key fields: `sensorId`, `name`, `location`, `state` (online/offline/removed), `sensorIp`, `hardwareId`, `tags`, `type`, `isTimelinePresent`, `isRemoteSensor`.

**Get single sensor info:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/<sensorId>/info" | jq .
```
Response: hardware metadata — `sensorId`, `name`, `sensorIp`, `location`, `manufacturer`, `hardware`, `hardwareId`, `firmwareVersion`, `serialNumber`, `tags`, `isRemoteSensor`, `position`. Does **not** include `state` or `type` — use `GET /sensor/status` for state, `GET /sensor/list` for type.

**Get sensor status (all):**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/status" | jq .
```
Response: object keyed by `sensorId`, each with `{name, state, errorCode, errorMessage}`.

**Get status of a single sensor:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/<sensorId>/status" | jq .
```
Response: `{name, state, errorCode, errorMessage}`.

**Get streams for a sensor** (returns `streamId` values needed for clip/snapshot calls):
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/<sensorId>/streams" | jq .
```
Response fields per stream: `streamId`, `isMain`, `url`, `vodUrl`, `name`, metadata with `bitrate`, `codec`, `framerate`, `resolution`.

**Get all streams across all sensors** (grouped by sensorId):
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/sensor/streams" | jq .
```

**Get all active live streams:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/live/streams" | jq .
```

**Get all streams available for replay:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/replay/streams" | jq .
```

---

### 3. Timelines & Storage Size

Always use the `/storage` service for timelines.

**Get timeline for a specific stream:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/<streamId>/timelines" | jq .
```

**Get timelines for all streams:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/timelines" | jq .
```

**Get timelines filtered to specific streams:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/timelines?streams=<streamId1>&streams=<streamId2>" | jq .
```

Response: object mapping `streamId` -> array of `{startTime, endTime}` (ISO 8601).

**Get storage usage (per-stream and totals):**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/size" | jq .
```
Response: object keyed by `streamId`, each with `{sizeInMegabytes, state}`, plus a `total` key with `{sizeInMegabytes, totalDiskCapacity, totalAvailableStorageSize, remainingStorageDays}`.

---

### 4. Video Clip Extraction

> **startTime / endTime:** Use values provided by the user. If not provided, first run:
> ```bash
> curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/<streamId>/timelines" | jq .
> ```
> Pick `startTime` and `endTime` from within a valid recorded range returned by that response.

**Download clip as binary (TS container by default):**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/file/<streamId>?startTime=<startTime>&endTime=<endTime>&disableAudio=true" \
  -o clip.ts
```

**Download clip as MP4:**
```bash
curl -s "http://<VST_ENDPOINT>/vst/api/v1/storage/file/<streamId>?startTime=<startTime>&endTime=<endTime>&container=mp4&disableAudio=true" \
  -o clip.mp4
```

**Get a temporary URL for the clip** (returns a URL instead of streaming bytes — preferred for large clips):
```bash
