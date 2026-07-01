#!/usr/bin/env python3
"""Run AMC calibration for user-provided pre-recorded videos."""

import os
import sys
import time
from pathlib import Path

import requests

def _env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


def _env_float_list(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


# Required env vars: BASE_URL, PROJECT_NAME, VIDEO_DIR. Optional file paths and
# knobs use CONFIG_FILE, ALIGNMENT_JSON, LAYOUT_PNG, GT_ZIP, FOCAL_LENGTHS,
# DETECTOR_TYPE, RUN_VGGT, REPO_ROOT, and PROJECTS_DIR.
BASE_URL       = os.environ.get("BASE_URL", "http://<HOST_IP>:<MS_PORT>/v1")
PROJECT_NAME   = os.environ.get("PROJECT_NAME", "my_calibration_run")
VIDEO_DIR      = Path(os.environ.get("VIDEO_DIR", "/path/to/videos"))
CONFIG_FILE    = _env_path("CONFIG_FILE")
ALIGNMENT_JSON = _env_path("ALIGNMENT_JSON")
LAYOUT_PNG     = _env_path("LAYOUT_PNG")
GT_ZIP         = _env_path("GT_ZIP")
FOCAL_LENGTHS  = _env_float_list("FOCAL_LENGTHS")
DETECTOR_TYPE  = os.environ.get("DETECTOR_TYPE", "resnet")
RUN_VGGT       = _env_bool("RUN_VGGT", False)

# Host projects dir (for verifying manual alignment output); override via PROJECTS_DIR env var.
REPO_ROOT    = Path(os.environ.get("REPO_ROOT", Path.cwd()))
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", REPO_ROOT / "projects"))
IS_INTERACTIVE = sys.stdin.isatty()

VIDEO_FILES = sorted(VIDEO_DIR.glob("cam_*.mp4"))
assert VIDEO_FILES, f"No cam_*.mp4 files under {VIDEO_DIR}"

# --- Auto-scan helper ---
def _resolve_local(override, candidate_names, scan_dirs, label):
    """Path if found locally (override or single scan hit), else None (-> ask user / UI fallback)."""
    if override and Path(override).exists():
        return Path(override)
    hits = []
    for d in scan_dirs:
        for name in candidate_names:
            p = d / name
            if p.exists():
                hits.append(p)
    if len(hits) == 1:
        print(f"    auto-detected {label}: {hits[0]}")
        return hits[0]
    searched = [str(d) for d in scan_dirs]
    if len(hits) > 1:
        print(f"    multiple {label} candidates found in {searched}: {hits}; use an explicit path or UI fallback")
    else:
        print(f"    no {label} file auto-detected in {searched}; use an explicit path or UI fallback")
    return None

_first_level_dirs = sorted((p for p in VIDEO_DIR.iterdir() if p.is_dir()), key=lambda p: p.name)
_scan_dirs = [VIDEO_DIR, *_first_level_dirs, VIDEO_DIR.parent]
CONFIG_FILE    = _resolve_local(CONFIG_FILE,    ["settings.json", "config.json", "calibration_config.json"], _scan_dirs, "config")
ALIGNMENT_JSON = _resolve_local(ALIGNMENT_JSON, ["alignment_data.json"],                                       _scan_dirs, "alignment")
LAYOUT_PNG     = _resolve_local(LAYOUT_PNG,     ["layout.png"],                                                _scan_dirs, "layout")

s = requests.Session()

# Step 1 -- Create project
r = s.post(f"{BASE_URL}/create_project", data={"project_name": PROJECT_NAME})
r.raise_for_status()
project_id = r.json()["project_id"]
print(f"[1] Created project: {project_id}")

# Step 2 -- Upload videos (sorted)
files, handles = [], []
for v in VIDEO_FILES:
    f = open(v, "rb"); handles.append(f)
    files.append(("files", (v.name, f, "video/mp4")))
r = s.post(f"{BASE_URL}/upload_video_files/{project_id}", files=files, timeout=300)
for f in handles: f.close()
r.raise_for_status()
print(f"[2] Uploaded {len(VIDEO_FILES)} videos")

# Step 3/4 -- Upload resolved files
if CONFIG_FILE and CONFIG_FILE.exists():
    r = s.post(f"{BASE_URL}/config/{project_id}",
               data=CONFIG_FILE.read_bytes(),
               headers={"Content-Type": "application/json"})
    r.raise_for_status()
    print(f"[3] Applied calibration config from {CONFIG_FILE.name} (replaces UI Step 3)")
    # Detector is consumed via the separate /calibrate parameter, so extract it for Step 7.
    try:
        import json as _json
        _cfg = _json.loads(CONFIG_FILE.read_text())
        _det = _cfg.get("detector") or _cfg.get("detector_type")
        if _det in ("resnet", "transformer"):
            DETECTOR_TYPE = _det
            print(f"    Detector overridden from config: {DETECTOR_TYPE}")
    except Exception:
        pass  # non-JSON config or no detector field -- keep DETECTOR_TYPE as-is

if ALIGNMENT_JSON and ALIGNMENT_JSON.exists():
    with open(ALIGNMENT_JSON, "rb") as f:
        s.post(f"{BASE_URL}/upload_alignment/{project_id}",
               files={"alignment_file": (ALIGNMENT_JSON.name, f, "application/json")}).raise_for_status()
    print(f"[3] Uploaded alignment: {ALIGNMENT_JSON.name}")

if LAYOUT_PNG and LAYOUT_PNG.exists():
    with open(LAYOUT_PNG, "rb") as f:
        s.post(f"{BASE_URL}/upload_layout/{project_id}",
               files={"layout_file": (LAYOUT_PNG.name, f, "image/png")}).raise_for_status()
    print(f"[3] Uploaded layout: {LAYOUT_PNG.name}")

if GT_ZIP and GT_ZIP.exists():
    with open(GT_ZIP, "rb") as f:
        s.post(f"{BASE_URL}/upload_gt_file/{project_id}",
               files={"gt_file": (GT_ZIP.name, f, "application/zip")}, timeout=120).raise_for_status()
    print(f"[3] Uploaded GT zip")

if FOCAL_LENGTHS:
    s.post(f"{BASE_URL}/upload_focal_length/{project_id}",
           data={"focal_length": FOCAL_LENGTHS}).raise_for_status()
    print(f"[3] Uploaded focal lengths: {FOCAL_LENGTHS}")

# Step 5 -- UI fallback for anything not resolved
ui_tasks = []
if not CONFIG_FILE:
    ui_tasks.append("Step 3 (Parameters): tune settings or accept defaults, then Save.")
if not ALIGNMENT_JSON or not LAYOUT_PNG:
    ui_tasks.append("Step 4 (Alignment): upload layout, mark correspondence points, then Save.")
if ui_tasks:
    print(f"\n[5] UI action required for project {project_id}:")
    for t in ui_tasks:
        print(f"    - {t}")
    if not IS_INTERACTIVE:
        raise RuntimeError(
            "UI action is required before continuing. Run interactively, or provide "
            "CONFIG_FILE, ALIGNMENT_JSON, and LAYOUT_PNG so the script can run unattended."
        )
    input("    Press Enter when done...")
    # Verify alignment files if the UI fallback was used for alignment
    if not ALIGNMENT_JSON or not LAYOUT_PNG:
        manual_dir = PROJECTS_DIR / f"project_{project_id}" / "manual_adjustment"
        assert (manual_dir / "alignment_data.json").exists() and (manual_dir / "layout.png").exists(), (
            f"Alignment files missing under {manual_dir}. Re-check UI Step 4 and click Save."
        )
        print(f"    Alignment files verified at {manual_dir}")

# Step 6 -- Verify
r = s.post(f"{BASE_URL}/verify_project/{project_id}")
r.raise_for_status()
state = r.json()["project_state"]
print(f"[6] Project state: {state}")
assert state == "READY", f"Expected READY, got {state}"

# Step 7 -- Confirm plan and calibrate
print("\n[7] Calibration plan:")
print(f"    Detector:             {DETECTOR_TYPE}")
print(f"    Calibration settings: {CONFIG_FILE if CONFIG_FILE else 'UI Step 3 settings/defaults'}")
print(f"    Alignment JSON:       {ALIGNMENT_JSON if ALIGNMENT_JSON else 'UI Step 4/manual_adjustment'}")
print(f"    Layout PNG:           {LAYOUT_PNG if LAYOUT_PNG else 'UI Step 4/manual_adjustment'}")
print(f"    Ground truth zip:     {GT_ZIP if GT_ZIP else 'not provided'}")
print(f"    Focal lengths:        {FOCAL_LENGTHS if FOCAL_LENGTHS else 'not provided'}")
if IS_INTERACTIVE:
    answer = input("    Start calibration? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Stopped before calibration.")
else:
    print("    Non-interactive stdin detected; starting with the plan above.")

s.post(f"{BASE_URL}/calibrate/{project_id}",
       json={"detector_type": DETECTOR_TYPE}).raise_for_status()
print(f"[7] Calibration started (detector={DETECTOR_TYPE})")

# Step 8 -- Poll. Print on every state change, plus a heartbeat at least once a
# minute so a long RUNNING state still shows progress.
print(f"[8] Polling (10-60 min typical)...")
start, last_state, last_beat = time.time(), "", 0.0
while time.time() - start < 5400:
    info = s.get(f"{BASE_URL}/get_project_info/{project_id}").json()
    st = info["project_info"]["project_state"]
    mins, secs = divmod(int(time.time() - start), 60)
    if st != last_state or time.time() - last_beat >= 60:
        print(f"    [{mins:>3}m {secs:02d}s] {st}", flush=True)
        last_state, last_beat = st, time.time()
    if st == "COMPLETED":
        print(f"[8] Done in {mins}m {secs:02d}s"); break
    if st == "ERROR":
        # Surface the tail of the calibration log so the failure is actionable.
        try:
            log_lines = s.get(f"{BASE_URL}/amc/calibrate/{project_id}/log").text.splitlines()
            print("    --- last calibration log lines ---")
            for line in log_lines[-20:]:
                print(f"    {line}")
        except Exception:
            pass
        raise RuntimeError(f"ERROR state -- full log: GET {BASE_URL}/amc/calibrate/{project_id}/log")
    time.sleep(10)
else:
    raise RuntimeError(
        f"Calibration still running after {int((time.time() - start) // 60)} min -- "
        f"inspect GET {BASE_URL}/amc/calibrate/{project_id}/log"
    )

# Step 9 -- Results
print(f"\n[9] Results:")
r = s.get(f"{BASE_URL}/result/{project_id}/evaluation_statistics")
if r.status_code == 200:
    for k, v in (r.json().get("statistics") or r.json()).items():
        print(f"    {k}: {v}")
else:
    print("    No GT provided -- skipping evaluation_statistics")

# Step 10 -- VGGT (optional)
info = s.get(f"{BASE_URL}/get_project_info/{project_id}").json()
vggt_state = info.get("project_info", {}).get("vggt_state", "INIT")
if vggt_state == "READY" and not RUN_VGGT:
    if IS_INTERACTIVE:
        answer = input("\n[10] VGGT is READY. Run VGGT refinement now? [y/N]: ").strip().lower()
        RUN_VGGT = answer in ("y", "yes")
    else:
        print("\n[10] VGGT is READY. Set RUN_VGGT=True to run VGGT refinement in non-interactive runs.")
elif vggt_state != "READY":
    print(f"\n[10] VGGT not ready (state={vggt_state}) -- skipping")
    print("     To run VGGT refinement later, set up the VGGT model with amc-setup-calibration-stack and rerun this optional step.")

if RUN_VGGT and vggt_state == "READY":
    s.post(f"{BASE_URL}/vggt/calibrate/{project_id}").raise_for_status()
    print("\n[10] VGGT started")
    t0 = time.time()
    while time.time() - t0 < 900:
        vs = s.get(f"{BASE_URL}/get_project_info/{project_id}").json() \
            .get("project_info", {}).get("vggt_state", "INIT")
        if vs == "COMPLETED":
            print("     VGGT done")
            r = s.get(f"{BASE_URL}/vggt_results/{project_id}/evaluation_statistics")
            if r.status_code == 200:
                print("     VGGT evaluation statistics:")
                for k, v in (r.json().get("statistics") or r.json()).items():
                    print(f"        {k}: {v}")
            else:
                print(f"     VGGT evaluation statistics unavailable (HTTP {r.status_code})")
            break
        if vs == "ERROR":
            raise RuntimeError("VGGT failed")
        time.sleep(10)

print(f"\nProject: {project_id}")
print("Review the calibration:")
print(f"    UI:                open project {project_id} in the AMC web UI, then the Results page to view the overlay")
print(f"    Final camera parameters: projects/project_{project_id}/output/multi_view_results/BA_output/results_ba/refined/camInfo_XX.yaml")