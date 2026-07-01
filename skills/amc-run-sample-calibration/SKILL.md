---
name: "amc-run-sample-calibration"
description: "Run end-to-end calibration on the shipped sample dataset (sdg_08_2_sample_data_010926.zip) against a running AMC microservice. Use when user says 'test sample dataset', 'run sample calibration', 'verify AMC install', or 'launch and test'."
owner: "NVIDIA CORPORATION"
service: "auto-magic-calib"
version: "1.0.0"
reviewed: "2026-04-28"
license: "Apache-2.0"
data_classification: public
metadata:
  tags: [amc, calibration, sample, rest-api, validation, python]
---

# Skill: Calibrate Sample Dataset

## When to Use This Skill

Activate this skill when the user wants to sanity-check a running AMC stack with the bundled sample dataset. Typical prompts:

- "test the sample dataset" / "run sample calibration"
- "verify AMC install"
- "launch and test" (chain with `amc-setup-calibration-stack` if the MS isn't already running)

**Do NOT use this skill when:**

- The user references their own video paths (e.g. `/data/videos/`, `cam_*.mp4` not from the bundled zip) — route to `amc-run-video-calibration`. This skill is exclusively for `assets/sdg_08_2_sample_data_010926.zip`.

Prerequisite: AMC microservice running on a port in 8000-8009. If no backend is detected, delegate to `amc-setup-calibration-stack` first.

If execution cannot proceed in the current environment (no backend, missing sample data, etc.), surface the blocker AND describe the expected workflow + API sequence concisely so the user understands what will run once prerequisites are met. Do not fabricate calibration outputs, evaluation metrics, or trajectories.

## Overview

Run a full calibration on the bundled sample dataset (`sdg_08_2_sample_data_010926.zip`, 4 synthetic warehouse cameras with ground truth) against a running AutoMagicCalib microservice. Useful for verifying that a freshly-launched stack works end-to-end before throwing real data at it.

The sample includes GT, so the run produces evaluation metrics (L2 distance, reprojection error) — no calibration parameter tuning needed.

## Prerequisites

- [ ] AMC microservice running (follow `skills/amc-setup-calibration-stack/SKILL.md` if not)
- [ ] Sample zip present at `assets/sdg_08_2_sample_data_010926.zip`
- [ ] Python 3 with `requests` available, or use the Swagger UI path below
  - The bundled script self-heals: if `requests` is missing it creates a throwaway venv under `${TMPDIR:-/tmp}/amc-sample-test-venv` (nothing written to the repo)
  - If `python3 -m venv` itself fails with `ensurepip not available`: `sudo apt install -y python3-venv python3-pip`

## Instructions

**"launch AMC and test sample dataset" (or similar):**

1. Run `skills/amc-setup-calibration-stack/SKILL.md` first.
2. Wait for `/v1/ready` to return OK.
3. Extract sample data (snippet below) — idempotent, safe to re-run.
4. Run the bundled script in [Run Script](#run-script).
5. Report final metrics + UI URL for manual inspection.
6. VGGT refinement is attempted by default when the project reports `vggt_state: READY`; otherwise the script explains that VGGT setup is optional and can be enabled later for refinement.

**"test sample dataset" (MS already running):**

1. Detect backend: scan ports 8000–8009 for a `/v1/ready` response.
2. If none → point to the setup skill.
3. Extract sample data if not already cached.
4. Run the bundled script.
5. Report metrics.

### Detect Running Backend

```bash
MS_PORT=""
for port in {8000..8009}; do
  if curl -s "http://localhost:$port/v1/ready" | grep -q '"code":0'; then
    MS_PORT=$port; break
  fi
done
[ -z "$MS_PORT" ] && { echo "No running backend. Run amc-setup-calibration-stack skill first."; exit 1; }
echo "Backend on port $MS_PORT"
```

### Locate + Extract Sample Data (idempotent)

```bash
export REPO_ROOT=$(git rev-parse --show-toplevel)

SAMPLE_ZIP="$REPO_ROOT/assets/sdg_08_2_sample_data_010926.zip"
[ -f "$SAMPLE_ZIP" ] || { echo "Sample zip not found at $SAMPLE_ZIP"; exit 1; }

# Cache directory next to the zip.
SAMPLE_DIR="$(dirname "$SAMPLE_ZIP")/.cache/sdg_08_2_sample_data_010926"

if [ ! -d "$SAMPLE_DIR" ]; then
  mkdir -p "$SAMPLE_DIR"
  unzip -q "$SAMPLE_ZIP" -d "$SAMPLE_DIR"
fi
ls "$SAMPLE_DIR"
# Expected (possibly inside a wrapper folder): alignment_data/  GT.zip  videos/
```

## Run Script

Run `scripts/run_sample_calibration.py` from the `auto-magic-calib` repo root, or set `REPO_ROOT=/path/to/auto-magic-calib`. The script reads `compose/.env` for the backend port, accepts `BASE_URL`, `MS_PORT`, `SAMPLE_DIR`, and `RUN_VGGT` overrides, creates a fresh project each run, attempts VGGT when ready, and prints the NGC warehouse dataset note at the end.

## Alternative: Swagger UI Walkthrough

> **Agent shortcut**: if the user explicitly requested a Swagger UI walkthrough (or said "no Python"), emit the table below and stop — do not invoke shell tooling, read other sections, or run the bundled Python script.

The microservice exposes an interactive OpenAPI UI at **`http://<HOST_IP>:<MS_PORT>/docs`**. If you prefer clicking through the API by hand:

1. Open `http://<HOST_IP>:<MS_PORT>/docs` in a browser.
2. Unzip `sdg_08_2_sample_data_010926.zip` into a cache directory next to it.
3. Execute these endpoints **in order**, copying the `project_id` from step 1 into subsequent paths:

   | # | Endpoint | Body / Files |
   |---|---|---|
   | 1 | `POST /v1/create_project` | `project_name`: any string |
   | 2 | `POST /v1/upload_video_files/{project_id}` | `files`: upload all 4 `videos/cam_0*.mp4` **sorted by name** |
   | 3 | `POST /v1/upload_alignment/{project_id}` | `alignment_file`: `alignment_data/alignment_data.json` |
   | 4 | `POST /v1/upload_layout/{project_id}` | `layout_file`: `alignment_data/layout.png` |
   | 5 | `POST /v1/upload_gt_file/{project_id}` | `gt_file`: `GT.zip` |
   | 6 | `POST /v1/verify_project/{project_id}` | — (expect `project_state: READY`) |
   | 7 | `POST /v1/calibrate/{project_id}` | JSON: `{"detector_type": "resnet"}` |
   | 8 | `GET /v1/get_project_info/{project_id}` | Refresh every ~10 s until `project_state` = `COMPLETED` |
   | 9 | `GET /v1/result/{project_id}/evaluation_statistics` | Read L2 distance + reprojection error |
   | 10 optional | `POST /v1/vggt/calibrate/{project_id}` then `GET /v1/vggt_results/{project_id}/evaluation_statistics` | Run only when `vggt_state` is `READY`; poll `vggt_state` until `COMPLETED` |

This is the same sequence the bundled Python script runs, just executed manually. Step 10 is attempted by default when `vggt_state` is `READY`; otherwise it is skipped with setup guidance.

### Status Fields from `get_project_info`

`project_info.project_state` is the AMC calibration lifecycle for the project. Poll it until it reaches `COMPLETED` (or stop on `ERROR`).

`project_info.vggt_state` is a **per-project** VGGT refinement lifecycle, a project-scoped status rather than a direct global service or model-load status. A newly created project can report `vggt_state: "INIT"` even when the VGGT model is present and mounted. The expected lifecycle is `INIT` → `READY` after AMC calibration completes → `RUNNING` while VGGT refinement runs → `COMPLETED` (or `ERROR`). Interpret `INIT` on a new or uncalibrated project as normal project state. If AMC calibration is complete and the project remains in a non-ready VGGT state, confirm VGGT setup and model availability with the setup skill checks and service logs.

## Success Criteria

- Project reaches `project_state == "COMPLETED"` within ~30 min.
- `/v1/result/{id}/evaluation_statistics` returns non-empty `statistics` (GT was uploaded).
- VGGT either runs to `vggt_state == "COMPLETED"` and reports `/v1/vggt_results/{id}/evaluation_statistics`, or is skipped with setup guidance because the project is not `READY` for VGGT.
- No `ERROR` state encountered.

Representative metrics for the sample (yours should be similar):

```
Average L2 distance(m)               : < 1.5
Average reprojection error 0(px)     : < 10
```

## Key Output Files (on the server)

Results persist under `$REPO_ROOT/projects/project_<project_id>/`:

```
projects/project_<project_id>/
├── output/
│   ├── single_view_results/cam_XX/
│   │   ├── camInfo_hyper_XX.yaml
│   │   └── trajDump_Stream_0_3d.txt
│   └── multi_view_results/BA_output/results_ba/refined/
│       └── camInfo_XX.yaml          # ← final calibration (use this)
└── calibration.log
```

## Monitoring Progress

```bash
PROJECT_ID=<id_from_step_1>
REPO_ROOT=$(git rev-parse --show-toplevel)
tail -F --retry "$REPO_ROOT/projects/project_${PROJECT_ID}/calibration.log"
```

Or stream MS logs:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
docker compose -f "$REPO_ROOT/compose/compose.yml" logs -f auto-magic-calib-ms
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `requests` not installed | Inside a venv: `python3 -m venv venv && ./venv/bin/pip install requests`. If `python3 -m venv` fails: `sudo apt install -y python3-venv python3-pip` first |
| `[2] Uploaded N videos` where N >> 4 | `SAMPLE_DIR` resolved to the repo root (or another over-broad path) and `rglob("cam_*.mp4")` swept stale videos from `.cache/`, `projects/`, etc. Stop the run (`POST /v1/stop_calibration/{id}`), delete the project (`DELETE /v1/delete_project/{id}`), set `SAMPLE_DIR` explicitly to the extracted sample dir, re-run. The script anchors on `videos/` and asserts `len(videos) <= 16` to fail loud |
| `verify_project` returns state `!= READY` | Confirm all 4 videos + alignment + layout + GT uploaded; inspect `GET /v1/get_project_info/{id}` response |
| Sample not extracted | `unzip <repo_root>/assets/sdg_08_2_sample_data_010926.zip -d <repo_root>/assets/.cache/sdg_08_2_sample_data_010926/` |
| `cam_*.mp4` glob finds 0 files | Check wrapper-folder depth: `find <sample_dir> -name "cam_*.mp4"` |
| Calibration times out (>60 min) | Check `calibration.log` for "insufficient tracklets"; see root `README.md` guidelines on input videos |
| Upload returns 413 | Raise server upload limit, or split files (sample files are <200 MB total so this is unusual) |
| Port scan finds no backend | Backend not running — run `amc-setup-calibration-stack` skill |

## Additional Sample Dataset

The root `README.md` also documents `nv_warehouse_032326.zip`, a real-world warehouse dataset available from NGC. Download it with `ngc registry resource download-version "nvidia/amc-nv-warehouse"`; then use `amc-run-video-calibration`, upload `nv_warehouse_config.json` in the config step, and run with the `transformer` detector. It does not include ground-truth data.

## Related Skills

- `skills/amc-setup-calibration-stack/SKILL.md` — launch MS + UI (prerequisite).
- `skills/amc-run-video-calibration/SKILL.md` — run calibration on your own pre-recorded MP4s.

Root `README.md` "Sample Data Setup" and "Calibration Workflow (UI)" sections cover the human-oriented path through the same sample.

<!-- signing marker -->
