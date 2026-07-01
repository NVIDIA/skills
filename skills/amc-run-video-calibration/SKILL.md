---
name: "amc-run-video-calibration"
description: "Calibrate a new dataset from pre-recorded video files via the AutoMagicCalib REST API. Use when user has local MP4s and says 'calibrate my videos', 'run AMC on these videos', or similar."
owner: "NVIDIA CORPORATION"
service: "auto-magic-calib"
version: "1.0.0"
reviewed: "2026-04-28"
license: "Apache-2.0"
data_classification: public
metadata:
  tags: [amc, calibration, rest-api, camera, python]
---

# Skill: Calibrate from Video Files

## When to Use This Skill

Activate this skill when the user has pre-recorded MP4 files and wants to calibrate them via the AMC REST API. Typical prompts:

- "calibrate my videos" / "run AMC on these videos"
- "calibrate from video files"

Drives calibration through the REST API on user-supplied **pre-recorded MP4 files** — no CLI scripts or Docker bind-mounts required, just a running microservice and your files.

## Prerequisites

- [ ] AMC microservice **and** UI running (follow `skills/amc-setup-calibration-stack/SKILL.md`)
- [ ] You know the microservice URL (e.g. `http://<HOST_IP>:<MS_PORT>`) and UI URL
- [ ] Video files locally as `cam_00.mp4`, `cam_01.mp4`, … time-synchronized, ~1920×1080
- [ ] Python 3 with `requests`

## Data Privacy

Video files uploaded via this skill are transmitted to the AutoMagicCalib backend (REST endpoint). Only use this skill when the backend is deployed on a trusted platform / network.

## What to Ask the User

### Required
(Video-file naming and the microservice URL are specified under Prerequisites above — collect the inputs below.)
1. **Videos directory** — the folder the skill globs for `cam_*.mp4`, uploaded sorted alphabetically.
2. **Microservice URL**
3. **Project name** — short descriptive string

### Auto-Detected (ask only if not found)

The script searches the videos dir, its first-level subdirectories, and its parent. If exactly one match is found, it is used; otherwise the script prints the searched locations and continues to explicit path or UI fallback:

| File | Candidate filenames | UI fallback |
|---|---|---|
| Calibration settings | `settings.json`, `config.json`, `calibration_config.json` | UI Step 3: Parameters |
| Alignment JSON | `alignment_data.json` | UI Step 4: Alignment |
| Layout PNG | `layout.png` | UI Step 4: Alignment |

Posting the settings file replaces UI Step 3 and may pin the detector (`resnet`/`transformer`), which is passed to `/calibrate` separately — see Step 4.

### Optional
4. **Ground truth zip** — `GT.zip` with `_World_Cameras_Camera_XX/` folders (enables evaluation metrics)
5. **Focal lengths** — one per camera, e.g. `1269.0, 1099.5, 1099.5`
6. **Detector type** — `resnet` (default, fast) or `transformer` (slower, better under occlusion)
7. **Run VGGT refinement?** — if VGGT is ready after AMC completes, ask the user whether to run refinement (see setup skill)

See root `README.md` "Custom Dataset" section for input-video guidelines and ground-truth format.

---

## Instructions

All endpoints below are implemented end-to-end in the [Complete Python Script](#complete-python-script) — the prose is the workflow plus the decisions the agent must make; the script is the authoritative runnable.

### Step 1 — Create Project

`POST /v1/create_project` (form field `project_name`) → save the returned `project_id`.

### Step 2 — Upload Videos (required)

`POST /v1/upload_video_files/<project_id>` (multipart `files`). **Upload sorted alphabetically** — the server assigns camera indices by upload order.

### Step 3 — Resolve Local Files (Auto-Scan, Ask, or UI)

For each of calibration-settings, alignment, and layout, run this resolution:

1. **Auto-scan** `VIDEO_DIR`, one level of subdirectories under `VIDEO_DIR`, and `VIDEO_DIR.parent` for the candidate filenames (table above).
2. If **exactly one match**, use it and print what was found.
3. If **zero or multiple matches**, print the searched locations, then ask the user for an explicit path using the host's question mechanism; if none is available, ask in chat and wait. If they don't have the file, mark it for UI fallback.
4. **UI fallback**: tell the user to complete the corresponding UI step; wait for confirmation; for alignment/layout also verify files landed in `projects/project_<id>/manual_adjustment/`.

### Step 4 — Upload Resolved Files

Upload each file resolved locally:

| File | Endpoint | Notes |
|---|---|---|
| Calibration settings | `POST /v1/config/<project_id>` (JSON, posted as-is) | Replaces UI Step 3 (rectification, bundle-adjustment, evaluation, detector, …). Non-2xx is surfaced — never silently fall back. Skip on the UI-fallback path. |
| Alignment | `POST /v1/upload_alignment/<project_id>` (`alignment_data.json`) | |
| Layout | `POST /v1/upload_layout/<project_id>` (`layout.png`) | |
| Ground truth (optional) | `POST /v1/upload_gt_file/<project_id>` (`GT.zip`) | Enables evaluation metrics |
| Focal lengths (optional) | `POST /v1/upload_focal_length/<project_id>` (repeated `focal_length=`) | Overrides GeoCalib estimates |

After a successful settings POST, parse the file for `"detector"` / `"detector_type"` — if it's `"resnet"` or `"transformer"`, use that value for the `/calibrate` call in Step 7 (detector is a separate API parameter, not consumed by `/config`).

### Step 5 — UI Fallback (only for files the user doesn't have locally)

If any of settings / alignment / layout was not resolved in Step 3, direct the user to the appropriate UI step:

- **Settings missing** → "Open UI project `<project_id>`, go to **Step 3: Parameters**, tune via the settings dialog (or accept defaults), click Save." **Also**: before the `/calibrate` call, ask the user which detector to use (`resnet` or `transformer`) using the host's question mechanism; if none is available, ask in chat and wait. UI Step 3 does not cover detector choice.
- **Alignment or layout missing** → "Open UI project `<project_id>`, go to **Step 4: Alignment**, upload layout, mark correspondence points, click Save."

Wait for user confirmation. For non-interactive script runs, provide the needed files up front; the script exits with a clear message rather than waiting on input. For alignment/layout, verify on disk before continuing:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
# Resolve PROJECT_DIR from compose/.env (default: projects/ at repo root).
PROJECT_DIR_REL=$(grep ^PROJECT_DIR "$REPO_ROOT/compose/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
HOST_PROJECTS=$(cd "$REPO_ROOT/compose" && realpath "${PROJECT_DIR_REL:-../../projects}")

ls "$HOST_PROJECTS/project_<project_id>/manual_adjustment/"
# Expected: alignment_data.json, layout.png
```

### Step 6 — Verify Project

`POST /v1/verify_project/<project_id>` → must return `{"project_state": "READY"}` before calibrating.

### Step 7 — Start Calibration

**Confirm the plan before calibrating.** Whether the settings file and detector were auto-detected or asked, present a short summary and get explicit user confirmation before `POST /calibrate` using the host's question mechanism; if none is available, ask in chat and wait. The resolved values are the defaults, so confirming is one click, but the user can switch the detector or skip an auto-detected settings file. The standalone Python script prints the same plan and prompts only when stdin is interactive. Summarize:

- **Detector** — `resnet` or `transformer` (the value to be sent).
- **Calibration settings** — the file being applied (path), or "defaults" if none.
- **Optional overrides** — ground-truth zip and focal lengths, if any.

```
POST /v1/calibrate/<project_id>
Content-Type: application/json

{"detector_type": "resnet"}
```

### Step 8 — Poll for Completion

`GET /v1/get_project_info/<project_id>` every 10 s — `project_info.project_state` goes `RUNNING` → `COMPLETED` (or `ERROR`, pull the log). Typical time: **10–60 min** depending on video length and detector.

### Step 9 — Get Results

`GET /v1/result/<project_id>/evaluation_statistics` (only if GT was uploaded; includes `Average L2 distance(m)` and `Average reprojection error 0(px)`), and `GET /v1/amc/calibrate/<project_id>/log` for the calibration log.

### Status Fields from `get_project_info`

`project_info.project_state` is the AMC calibration lifecycle for the project: `RUNNING` → `COMPLETED` (or `ERROR`).

`project_info.vggt_state` is also **per-project**, a project-scoped VGGT refinement lifecycle rather than a direct global service or model-load status. A newly created project can report `vggt_state: "INIT"` even when the VGGT model is present and mounted. The expected VGGT lifecycle is `INIT` → `READY` after AMC calibration completes → `RUNNING` while VGGT refinement runs → `COMPLETED` (or `ERROR`).

Use `vggt_state == "READY"` only as the gate for optional VGGT refinement in Step 10. Interpret `INIT` on a new or uncalibrated project as normal project state. If AMC calibration is complete and the project remains in a non-ready VGGT state, confirm VGGT setup and model availability with the setup skill checks and service logs.

### Step 10 — (Optional) VGGT Refinement

After AMC calibration completes, read `vggt_state` from `GET /v1/get_project_info/<project_id>`.

- If the project reports `vggt_state == "READY"`, ask the user whether to run VGGT refinement using the host's question mechanism; if none is available, ask in chat and wait.
- If the user confirms, `POST /v1/vggt/calibrate/<project_id>`, poll `vggt_state` via `get_project_info`, then `GET /v1/vggt_results/<project_id>/evaluation_statistics`.
- If VGGT is not ready, skip refinement and explain that the user can set up VGGT with `amc-setup-calibration-stack` and rerun this optional step later.

The standalone Python script prompts only when stdin is interactive. In non-interactive runs, set `RUN_VGGT = True` to opt in; otherwise the script prints that VGGT is ready and continues without blocking.

---

## Complete Python Script

Use `scripts/run_video_calibration.py` for the runnable implementation. Set `BASE_URL`, `PROJECT_NAME`, and `VIDEO_DIR`; optional env vars are `CONFIG_FILE`, `ALIGNMENT_JSON`, `LAYOUT_PNG`, `GT_ZIP`, `FOCAL_LENGTHS`, `DETECTOR_TYPE`, `RUN_VGGT`, `REPO_ROOT`, and `PROJECTS_DIR`. The script implements UI fallback, plan confirmation, VGGT prompt/opt-in behavior, polling, and refined statistics retrieval.

## Success Criteria

- `project_state == "COMPLETED"` after polling.
- If manual alignment was used: `projects/project_<id>/manual_adjustment/` contains `alignment_data.json` + `layout.png`.
- If GT was uploaded: evaluation returns typical thresholds:
  - `Average L2 distance(m)` < 1.5
  - `Average reprojection error 0(px)` < 5
- No `ERROR` state.

## Key Output Files (on server)

```
projects/project_<project_id>/
├── manual_adjustment/
│   ├── alignment_data.json
│   └── layout.png
├── output/
│   ├── single_view_results/cam_XX/
│   │   ├── camInfo_hyper_XX.yaml
│   │   └── trajDump_Stream_0_3d.txt
│   └── multi_view_results/BA_output/results_ba/
│       ├── initial/camInfo_XX.yaml
│       └── refined/camInfo_XX.yaml          # ← final calibration
└── calibration.log
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `verify_project` state not `READY` | Confirm videos uploaded and alignment + layout are present (either via API or via UI manual alignment) |
| Manual alignment files missing after UI step | User didn't click Save; also verify `projects/project_<id>/manual_adjustment/` exists |
| Calibration stuck `RUNNING` > 90 min | `GET /v1/amc/calibrate/<id>/log` — usually insufficient tracklets (scene too static). See "Custom Dataset" guidelines in root README. |
| Immediate `ERROR` state | Check video naming: must be `cam_00.mp4`, `cam_01.mp4`, … contiguous |
| Low L2 but high reprojection | Provide explicit `focal_length` override via Step 3 |
| VGGT stays non-ready after AMC completes | `INIT` is expected for a new project. After AMC calibration reaches `COMPLETED`, the project should transition to `READY` before optional VGGT refinement when VGGT is configured. If refinement is required and the state remains `INIT` or otherwise non-ready, confirm VGGT setup and model availability with setup skill Step 2 and MS logs. |
| Upload timeout | Large videos — bump `timeout=300` to e.g. `600` in the script |

## For Downstream Skills — MV3DT Export

A downstream Multi-View 3D Tracking skill fetches the MV3DT-format calibration directly from the microservice (this skill does **not** download it; it returns the `project_id`). After this skill reports `COMPLETED`:

- `GET /v1/result/{project_id}/mv3dt_result?result_type=amc` → `mv3dt_output.zip` (contains `transforms.yml`).
- If VGGT ran to `COMPLETED` (Step 10): `?result_type=vggt` → `vggt_mv3dt_output.zip`.

## Related Skills

- `skills/amc-setup-calibration-stack/SKILL.md` — start MS + UI first.
- `skills/amc-run-sample-calibration/SKILL.md` — verify the stack with the bundled sample before trying your own.

Root `README.md` "Custom Dataset" and "Calibration Workflow (UI)" sections document input-video guidelines and the UI-driven alternative to this API flow.

<!-- signing marker -->
