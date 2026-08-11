---
name: "amc-run-rtsp-calibration"
description: "Calibrate a new dataset from live RTSP camera streams via the AutoMagicCalib REST API. Use when the user provides RTSP URLs or asks to calibrate live cameras; VIOS records clips, AMC ingests them, then runs calibration."
owner: "NVIDIA CORPORATION"
service: "auto-magic-calib"
version: "1.0.0"
reviewed: "2026-06-15"
license: "Apache-2.0"
metadata:
  author: "Shubham Agrawal <shuagrawal@nvidia.com>"
  tags: [amc, calibration, rtsp, vios, rest-api, camera, python]
---

# Skill: Calibrate from RTSP Streams

## When to Use This Skill

Activate this skill when the user wants to calibrate from live RTSP camera streams. Typical prompts:

- "calibrate RTSP streams" / "calibrate from live cameras"
- "run AMC on RTSP"
- The user provides one or more `rtsp://...` URLs

VIOS records fixed-duration clips from each stream, the AMC microservice ingests those clips into a project, then the workflow follows the same verification, calibration, polling, and results path as pre-recorded MP4 calibration.

Do not use this skill for local MP4 files already on disk; route those requests to `skills/amc-run-video-calibration/SKILL.md`. Do not use it for the bundled sample dataset; route that to `skills/amc-run-sample-calibration/SKILL.md`.

Never reuse files from the bundled sample dataset, extracted sample zip, `assets/`, or previous projects for RTSP calibration unless the user explicitly provides those paths for this RTSP scene. Similar camera names, stream counts, or `cam_00`/`cam_01` ordering are not evidence that sample alignment, layout, GT, or detector settings apply.

## Prerequisites

- [ ] AMC microservice and UI running (follow `skills/amc-setup-calibration-stack/SKILL.md` if needed).
- [ ] VIOS is running and reachable from the AMC microservice.
- [ ] `VIOS_BASE_URL` is configured in the AMC microservice environment before capture starts.
- [ ] RTSP URLs are reachable from the VIOS host.
- [ ] Camera streams have enough moving people/objects for calibration; record at least 2-3 minutes when possible.
- [ ] Python 3 with `requests` installed when using the bundled script.

## Data Privacy

RTSP URLs may contain usernames, passwords, hostnames, or network topology. Do not print full RTSP URLs if credentials are embedded. Pass VIOS tokens through environment variables or secure host prompts; do not echo tokens in chat, logs, or final answers.

## What to Ask the User

### Required

1. RTSP URLs, one per camera.
2. Camera names, one per stream. Use `cam_00`, `cam_01`, ... if the user does not provide names.
3. Recording duration in seconds. Minimum is `60`; prefer `120`-`180` or more when the scene has sparse motion.
4. Microservice URL, for example `http://<HOST_IP>:8000` or `http://<HOST_IP>:8000/v1`.
5. Project name.
6. Calibration asset source for this RTSP scene:
   - a local directory to scan, such as `/data/my_rtsp_calib/`;
   - explicit paths to settings, alignment, layout, and optional GT files; or
   - confirmation that the user will upload/tune settings and alignment in the AMC UI.

If the user does not provide a local asset source, stop and ask whether they want to provide a path or use UI upload. Give the UI link as `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>`; the default UI port is `5000`.

### Auto-Detected or Asked

RTSP clips are recorded by VIOS, so there is no local videos directory to anchor file discovery. Only scan a directory the user explicitly provided for this RTSP scene. If the user provides a settings file path, use that file's directory as the scan directory. If the user provides a calibration asset directory, scan only that directory. Otherwise ask this question before planning uploads or calibration:

> Do you have a local calibration asset directory or settings file for these RTSP streams, or should you upload/tune settings and alignment in the AMC UI at `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>`?

| File | Candidate filenames | UI fallback |
|---|---|---|
| Calibration settings | Explicit user path, or `settings.json`, `config.json`, or `calibration_config.json` in the user-provided asset directory | UI Step 3: Parameters |
| Alignment JSON | Explicit user path, or `alignment_data.json` in the user-provided asset directory/settings directory | UI Step 4: Alignment |
| Layout PNG | Explicit user path, or `layout.png` in the user-provided asset directory/settings directory | UI Step 4: Alignment |
| Ground truth zip | Optional explicit user path, or `GT.zip`/`gt.zip` in the user-provided asset directory | Omit metrics |

Posting the settings file replaces UI Step 3 and may pin `detector` or `detector_type`. If it pins `resnet` or `transformer`, pass that same detector to `/calibrate`. If no settings file pins a detector, ask the user which detector to use; do not silently default to `resnet`.

### Optional

6. `sensor_id` per stream if the cameras are already registered in VIOS. Leave unset for auto-registration.
7. Ground truth zip (`GT.zip`) for evaluation metrics.
8. Focal lengths, one per camera.
9. VIOS bearer token, if the VIOS deployment requires one.
10. Whether to run VGGT refinement after AMC completes, only when the project reports `vggt_state == "READY"`.

## Instructions

The bundled script in [scripts/run_rtsp_calibration.py](scripts/run_rtsp_calibration.py) implements this sequence end to end. Use the prose below for decisions, UI fallback, and troubleshooting.

### Step 0 - Verify AMC and VIOS

Confirm the AMC microservice is reachable:

```bash
curl -sf http://<HOST_IP>:<MS_PORT>/v1/ready
```

Confirm VIOS is reachable before starting capture. Probe in this order and stop at the first working URL:

```bash
: "${REPO_ROOT:?set REPO_ROOT to the auto-magic-calib checkout. Run amc-setup-calibration-stack Step 0b first.}"
grep -q "AutoMagicCalib" "$REPO_ROOT/README.md" 2>/dev/null && grep -q "auto-magic-calib-ms" "$REPO_ROOT/compose/ms/compose.yml" 2>/dev/null || { echo "ERROR: REPO_ROOT is not an auto-magic-calib checkout: $REPO_ROOT" >&2; exit 1; }
VIOS_BASE_URL=""

# Default local VIOS port.
if curl -sf http://localhost:30888/vst/api/v1/sensor/list >/dev/null 2>&1; then
  HOST_IP=$(grep ^HOST_IP "$REPO_ROOT/compose/.env" 2>/dev/null | cut -d= -f2)
  VIOS_BASE_URL="http://${HOST_IP:-localhost}:30888"
  echo "VIOS detected at $VIOS_BASE_URL"
fi

# Running AMC microservice container environment.
if [ -z "$VIOS_BASE_URL" ]; then
  VIOS_BASE_URL=$(docker exec auto-magic-calib-ms-1 printenv VIOS_BASE_URL 2>/dev/null)
fi

# Compose environment file.
if [ -z "$VIOS_BASE_URL" ]; then
  VIOS_BASE_URL=$(grep ^VIOS_BASE_URL "$REPO_ROOT/compose/.env" 2>/dev/null | cut -d= -f2-)
fi

if [ -n "$VIOS_BASE_URL" ]; then
  curl -sf "${VIOS_BASE_URL}/vst/api/v1/sensor/list" >/dev/null \
    && echo "VIOS up at $VIOS_BASE_URL" \
    || { echo "VIOS_BASE_URL=$VIOS_BASE_URL is set but not responding"; VIOS_BASE_URL=""; }
fi
```

If VIOS is not reachable, ask the user to deploy VIOS and provide the base URL. Do not start RTSP capture until `${VIOS_BASE_URL}/vst/api/v1/sensor/list` returns 200.

If VIOS is reachable but the AMC microservice is missing `VIOS_BASE_URL`, do not edit checked-in compose files. Export the variable and relaunch the microservice with a temporary compose override:

```bash
cd "$REPO_ROOT/compose"
export VIOS_BASE_URL="http://<VIOS_HOST>:30888"
OVERRIDE_FILE="${TMPDIR:-/tmp}/amc-vios.override.yml"
cat > "$OVERRIDE_FILE" <<'YAML'
services:
  auto-magic-calib-ms:
    environment:
      - VIOS_BASE_URL=${VIOS_BASE_URL}
YAML

docker compose -f compose.yml -f "$OVERRIDE_FILE" up -d auto-magic-calib-ms
docker exec auto-magic-calib-ms-1 printenv VIOS_BASE_URL
```

A host-shell export alone is not enough after the container is already running; the microservice process must be restarted with `VIOS_BASE_URL` in its environment.

### Step 1 - Create Project

`POST /v1/create_project` with form field `project_name`. Save the returned `project_id`.

### Step 2 - Start RTSP Capture

```
POST /v1/rtsp/capture/<project_id>
Content-Type: application/json

{
  "streams": [
    {"rtsp_url": "rtsp://...", "camera_name": "cam_00", "sensor_id": null},
    {"rtsp_url": "rtsp://...", "camera_name": "cam_01", "sensor_id": null}
  ],
  "duration_seconds": 180,
  "vios_token": null,
  "ssl_verify": false
}
```

The response can nest session fields under `session`:

```
{"code": 0, "message": "...", "session": {"session_id": "...", "status": "STARTING"}}
```

Save `session.session_id`.

### Step 3 - Poll Capture, Then Ingest

Poll every 10 seconds:

```
GET /v1/rtsp/capture/<project_id>/<session_id>
```

Session lifecycle:

```
STARTING -> RECORDING -> COMPLETED -> INGESTING -> INGESTED
                       -> ERROR
RECORDING -> CANCELLED
```

When capture reaches `COMPLETED`, ingest the recorded clips into the AMC project:

```
POST /v1/rtsp/capture/<project_id>/<session_id>/ingest
```

After ingest succeeds, the project has video files attached and the rest of the workflow matches the MP4 upload path.

Need to stop early: `POST /v1/rtsp/capture/<project_id>/<session_id>/stop`. A partial clip can still be ingested if VIOS produced one.

Other session endpoints:

- `GET /v1/rtsp/sessions/<project_id>` - list sessions for a project.
- `DELETE /v1/rtsp/session/<project_id>/<session_id>` - delete a session record.

### Step 4 - Upload Settings, Alignment, Layout, and Optional Files

Resolve local files using the anchor-file pattern above. Upload resolved files:

| File | Endpoint | Notes |
|---|---|---|
| Calibration settings | `POST /v1/config/<project_id>` | JSON body posted as-is; replaces UI Step 3 |
| Alignment JSON | `POST /v1/upload_alignment/<project_id>` | Multipart `alignment_file` |
| Layout PNG | `POST /v1/upload_layout/<project_id>` | Multipart `layout_file` |
| Ground truth zip | `POST /v1/upload_gt_file/<project_id>` | Optional |
| Focal lengths | `POST /v1/upload_focal_length/<project_id>` | Optional repeated `focal_length` values |

Use only files from explicit user-provided paths or a user-provided calibration asset directory. Do not extract or scan sample data to find fallback settings, alignment, layout, or GT.

If settings are missing, direct the user to UI Step 3: Parameters at `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>`, then ask which detector to use (`resnet` or `transformer`) before calibration. If alignment or layout is missing, direct the user to UI Step 4: Alignment for this project. For RTSP projects, videos are already ingested; do not re-upload videos in the UI fallback.

Before continuing after UI Step 4, verify:

```bash
PROJECT_ID=<project_id>
: "${REPO_ROOT:?set REPO_ROOT to the auto-magic-calib checkout. Run amc-setup-calibration-stack Step 0b first.}"
grep -q "AutoMagicCalib" "$REPO_ROOT/README.md" 2>/dev/null && grep -q "auto-magic-calib-ms" "$REPO_ROOT/compose/ms/compose.yml" 2>/dev/null || { echo "ERROR: REPO_ROOT is not an auto-magic-calib checkout: $REPO_ROOT" >&2; exit 1; }
PROJECT_DIR_REL=$(grep ^PROJECT_DIR "$REPO_ROOT/compose/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
HOST_PROJECTS=$(cd "$REPO_ROOT/compose" && realpath "${PROJECT_DIR_REL:-../../projects}")
ls "$HOST_PROJECTS/project_${PROJECT_ID}/manual_adjustment/"
# Expected: alignment_data.json, layout.png
```

### Step 5 - Verify, Calibrate, Poll, and Fetch Results

Verify:

```
POST /v1/verify_project/<project_id>
```

The project must return `project_state == "READY"`.

Confirm the plan before calibrating. Summarize:

- Stream count and recording duration.
- Detector: `resnet` or `transformer`.
- Settings source: explicit uploaded settings file, user-provided asset directory, or UI Step 3.
- Alignment/layout source: explicit uploaded files, user-provided asset directory, or UI manual adjustment.
- Optional GT and focal-length overrides.

Start calibration:

```
POST /v1/calibrate/<project_id>
Content-Type: application/json

{"detector_type": "<resnet-or-transformer>"}
```

Poll:

```
GET /v1/get_project_info/<project_id>
```

Stop on `COMPLETED` or `ERROR`. On error, fetch `GET /v1/amc/calibrate/<project_id>/log`.

Fetch results:

```
GET /v1/result/<project_id>/evaluation_statistics
```

Only expect evaluation statistics when GT was uploaded.

### Step 6 - Optional VGGT Refinement

After AMC calibration completes, read `project_info.vggt_state` from `GET /v1/get_project_info/<project_id>`.

- If `vggt_state == "READY"`, ask whether to run VGGT refinement.
- If confirmed, call `POST /v1/vggt/calibrate/<project_id>`, poll `vggt_state`, then fetch `GET /v1/vggt_results/<project_id>/evaluation_statistics`.
- If VGGT is not ready, skip it and explain that AMC calibration is complete.

## Complete Python Script

Use the bundled script from the `amc-run-rtsp-calibration` skill package, not from the `auto-magic-calib` repo root. If the user points the agent at this skill folder directly instead of installing it, set `AMC_RTSP_SKILL_DIR` to the directory containing this `SKILL.md`, or run the command from that directory.

Common environment variables:

```bash
export BASE_URL=http://<HOST_IP>:8000
export PROJECT_NAME=rtsp_calibration_run
export RTSP_URLS='rtsp://user:pass@cam0/stream,rtsp://user:pass@cam1/stream'
export CAMERA_NAMES='cam_00,cam_01'
export DURATION_SECONDS=180
export VIOS_BASE_URL=http://<VIOS_HOST>:30888
export CALIB_ASSET_DIR=/path/to/rtsp-calibration-assets
# Or provide explicit CONFIG_FILE, ALIGNMENT_JSON, LAYOUT_PNG, and optional GT_ZIP.
export DETECTOR_TYPE=transformer  # Required when settings do not set detector/detector_type.
export AMC_UI_URL=http://<HOST_IP>:5000
export RUN_VGGT=false

# Optional but recommended: REPO_ROOT points to the auto-magic-calib checkout.
# PROJECTS_DIR can be set explicitly when project outputs live elsewhere.
if [ -z "${DEEPSTREAM_REPO_ROOT:-}" ] && [ -n "${REPO_ROOT:-}" ] && [ -d "$REPO_ROOT/../../skills/amc-run-rtsp-calibration" ]; then
  DEEPSTREAM_REPO_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
fi

SCRIPT_PATH=""
for candidate in \
  "${AMC_RTSP_SKILL_DIR:+$AMC_RTSP_SKILL_DIR/scripts/run_rtsp_calibration.py}" \
  "$PWD/scripts/run_rtsp_calibration.py" \
  "${DEEPSTREAM_REPO_ROOT:+$DEEPSTREAM_REPO_ROOT/skills/amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py}" \
  "$PWD/skills/amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py" \
  "$HOME/.claude/skills/amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py" \
  "$HOME/.codex/skills/amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py" \
  "$HOME/.cursor/skills/amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py"; do
  if [ -f "$candidate" ]; then
    SCRIPT_PATH="$candidate"
    break
  fi
done

[ -n "$SCRIPT_PATH" ] || {
  echo "ERROR: could not find amc-run-rtsp-calibration/scripts/run_rtsp_calibration.py" >&2
  echo "Set AMC_RTSP_SKILL_DIR to the amc-run-rtsp-calibration skill directory, or run this block from that directory." >&2
  exit 1
}

python3 "$SCRIPT_PATH"
```

Alternative stream input:

```bash
export STREAMS_JSON='[
  {"rtsp_url":"rtsp://cam0/stream","camera_name":"cam_00","sensor_id":null},
  {"rtsp_url":"rtsp://cam1/stream","camera_name":"cam_01","sensor_id":null}
]'
```

Optional env vars are `CALIB_ASSET_DIR`, `CONFIG_FILE`, `ALIGNMENT_JSON`, `LAYOUT_PNG`, `GT_ZIP`, `FOCAL_LENGTHS`, `DETECTOR_TYPE`, `AMC_UI_URL`, `VIOS_TOKEN`, `SSL_VERIFY`, `RUN_VGGT`, `REPO_ROOT`, and `PROJECTS_DIR`.

## Success Criteria

- VIOS health probe returns 200.
- Capture session reaches `COMPLETED`.
- Ingest returns success and project info shows the expected video files.
- `verify_project` returns `READY`.
- AMC calibration reaches `project_state == "COMPLETED"`.
- If GT was uploaded, evaluation statistics are returned.
- No RTSP credentials, bearer tokens, NGC keys, or HuggingFace tokens are printed or persisted by the agent.

## Key Output Files

Results persist on the AMC server under:

```
projects/project_<project_id>/
|-- manual_adjustment/
|   |-- alignment_data.json
|   `-- layout.png
|-- output/
|   |-- single_view_results/cam_XX/
|   |   |-- camInfo_hyper_XX.yaml
|   |   `-- trajDump_Stream_0_3d.txt
|   `-- multi_view_results/BA_output/results_ba/
|       |-- initial/camInfo_XX.yaml
|       `-- refined/camInfo_XX.yaml
`-- calibration.log
```

## Troubleshooting

| Issue | Fix |
|---|---|
| VIOS `/vst/api/v1/sensor/list` returns connection refused | VIOS is not running or not reachable from this host. Ask the user to deploy VIOS or provide the reachable base URL. |
| Capture endpoint returns 503 or "VIOS not configured" | Export `VIOS_BASE_URL`, relaunch the microservice with the temporary compose override from Step 0, then retry capture. |
| Session stuck in `STARTING` | VIOS accepted the request but sensors may not be online. Check `${VIOS_BASE_URL}/vst/api/v1/sensor/list` and wait 20-30 seconds after sensor restarts. |
| Session stuck in `RECORDING` past `duration_seconds` | Call `POST /v1/rtsp/capture/<project_id>/<session_id>/stop`, then ingest the partial clip if available. |
| Ingest fails with "No clip available" | The recording window may not overlap the VIOS timeline. Wait for sensors to become online, then start a new capture. |
| 400 "empty streams" | Pass at least one stream object with `rtsp_url` and `camera_name`. |
| 400 "duration too short" | Use `duration_seconds >= 60`. |
| 404 on `/v1/rtsp/capture/<project_id>` | Create the project first with `/v1/create_project`. |
| `verify_project` is not `READY` after ingest | Check project info and confirm expected videos, alignment, and layout are attached. |
| Calibration reaches `ERROR` | Fetch `GET /v1/amc/calibrate/<project_id>/log`; common causes are insufficient tracklets, static scenes, or incorrect alignment. |

## Related Skills

- `skills/amc-setup-calibration-stack/SKILL.md` - start AMC microservice and UI first.
- `skills/amc-run-video-calibration/SKILL.md` - calibrate from local pre-recorded MP4 files.
- `skills/amc-run-sample-calibration/SKILL.md` - verify the stack with the bundled sample dataset.

<!-- signing marker -->
