#!/usr/bin/env python3
"""Run AMC calibration against the bundled synthetic sample dataset."""

import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ModuleNotFoundError:
    venv = Path(os.environ.get("TMPDIR", "/tmp")) / "amc-sample-test-venv"
    python = venv / "bin" / "python3"
    pip = venv / "bin" / "pip"
    if not python.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    subprocess.check_call([str(pip), "install", "--quiet", "requests"])
    os.execv(str(python), [str(python), __file__, *sys.argv[1:]])

def _read_env_key(path: Path, key: str):
    try:
        for line in path.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


# Run from the auto-magic-calib repo root, or set REPO_ROOT explicitly.
REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path.cwd())
MS_PORT = (
    os.environ.get("MS_PORT")
    or _read_env_key(REPO_ROOT / "compose" / ".env", "AUTO_MAGIC_CALIB_MS_PORT")
    or "8000"
)
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{MS_PORT}/v1")
RUN_VGGT = _env_bool("RUN_VGGT", True)

# Sample zip lives in assets/.
def _find_sample_dir() -> Path:
    candidate = REPO_ROOT / "assets" / ".cache" / "sdg_08_2_sample_data_010926"
    if candidate.exists():
        return candidate
    sys.exit(
        "Sample data not extracted. Run the extraction snippet from this skill first, "
        "or pass SAMPLE_DIR= explicitly."
    )

# NOTE: do NOT write `Path(os.environ.get("SAMPLE_DIR", "")) or _find_sample_dir()`
# -- Path("") evaluates to Path('.') which is truthy, so the `or` never falls
# through and the script silently picks `.` (typically the repo root). Rglobbing
# `cam_*.mp4` from there can sweep dozens of stale videos from prior test runs.
_env_sample = os.environ.get("SAMPLE_DIR")
SAMPLE_DIR = Path(_env_sample).resolve() if _env_sample else _find_sample_dir()

# Locate sample files (handle an optional wrapper folder from unzip)
def _find(path: Path, name: str) -> Path:
    hits = list(path.rglob(name))
    if not hits:
        sys.exit(f"Could not find {name} under {path}")
    return hits[0]

# Anchor video discovery on the canonical `videos/` directory if present
# (non-recursive). Only fall back to rglob if no `videos/` folder exists,
# and assert a sane upper bound so a misconfigured SAMPLE_DIR fails loud
# instead of uploading every cam_*.mp4 in the tree.
videos_dirs = list(SAMPLE_DIR.rglob("videos"))
videos_dir = next((d for d in videos_dirs if d.is_dir()), None)
if videos_dir is not None:
    videos = sorted(videos_dir.glob("cam_*.mp4"))
else:
    videos = sorted(SAMPLE_DIR.rglob("cam_*.mp4"))

alignment = _find(SAMPLE_DIR, "alignment_data.json")
layout = _find(SAMPLE_DIR, "layout.png")
gt_zip = _find(SAMPLE_DIR, "GT.zip")

assert len(videos) >= 2, f"Need >=2 cam_XX.mp4 under {SAMPLE_DIR}, found {len(videos)}"
# Sample dataset has 4 cameras -- bail if SAMPLE_DIR is so wide we'd upload
# unrelated videos. Override SAMPLE_DIR explicitly if you need a different one.
assert len(videos) <= 16, (
    f"Found {len(videos)} cam_*.mp4 under {SAMPLE_DIR} -- looks like SAMPLE_DIR "
    "is too broad (probably picked up stale test caches). Set SAMPLE_DIR to the "
    "extracted sample folder explicitly and re-run."
)
print(f"Base URL:   {BASE_URL}")
print(f"Sample dir: {SAMPLE_DIR}")
print(f"Videos:     {[v.name for v in videos]}")

s = requests.Session()
DEFAULT_TIMEOUT = (10, 120)  # (connect, read) -- keeps hung MS calls from blocking the 3600s budget

# Step 1 -- Create project
project_name = f"sample_test_{int(time.time())}"
r = s.post(f"{BASE_URL}/create_project", data={"project_name": project_name}, timeout=DEFAULT_TIMEOUT)
r.raise_for_status()
project_id = r.json()["project_id"]
print(f"[1] Created project {project_name} -> {project_id}")

# Step 2 -- Upload videos (sorted alphabetically; upload order defines camera indices)
files, handles = [], []
for v in videos:
    f = open(v, "rb"); handles.append(f)
    files.append(("files", (v.name, f, "video/mp4")))
r = s.post(f"{BASE_URL}/upload_video_files/{project_id}", files=files, timeout=300)
for f in handles: f.close()
r.raise_for_status()
print(f"[2] Uploaded {len(videos)} videos")

# Step 3 -- Upload alignment JSON
with open(alignment, "rb") as f:
    r = s.post(f"{BASE_URL}/upload_alignment/{project_id}",
               files={"alignment_file": (alignment.name, f, "application/json")},
               timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
print(f"[3] Uploaded alignment JSON")

# Step 4 -- Upload layout PNG
with open(layout, "rb") as f:
    r = s.post(f"{BASE_URL}/upload_layout/{project_id}",
               files={"layout_file": (layout.name, f, "image/png")},
               timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
print(f"[4] Uploaded layout PNG")

# Step 5 -- Upload GT zip (enables evaluation metrics)
with open(gt_zip, "rb") as f:
    r = s.post(f"{BASE_URL}/upload_gt_file/{project_id}",
               files={"gt_file": (gt_zip.name, f, "application/zip")}, timeout=120)
    r.raise_for_status()
print(f"[5] Uploaded GT zip")

# Step 6 -- Verify project
r = s.post(f"{BASE_URL}/verify_project/{project_id}", timeout=DEFAULT_TIMEOUT)
r.raise_for_status()
state = r.json()["project_state"]
print(f"[6] verify_project -> {state}")
if state != "READY":
    raise RuntimeError(f"Expected READY, got {state}")

# Step 7 -- Start calibration (defaults work for this dataset)
r = s.post(f"{BASE_URL}/calibrate/{project_id}", json={"detector_type": "resnet"}, timeout=DEFAULT_TIMEOUT)
r.raise_for_status()
print(f"[7] Calibration started (detector=resnet)")

# Step 8 -- Poll for completion (~10-30 min for sample). Print on every state
# change, plus a heartbeat at least once a minute so a long RUNNING state still
# shows progress.
print(f"[8] Polling (expect 10-30 min)...")
start, last_state, last_beat = time.time(), "", 0.0
while time.time() - start < 3600:
    r = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    st = r.json()["project_info"]["project_state"]
    mins, secs = divmod(int(time.time() - start), 60)
    if st != last_state or time.time() - last_beat >= 60:
        print(f"    [{mins:>3}m {secs:02d}s] {st}", flush=True)
        last_state, last_beat = st, time.time()
    if st == "COMPLETED":
        print(f"[8] Completed in {mins}m {secs:02d}s")
        break
    if st == "ERROR":
        # Surface the tail of the calibration log so the failure is actionable.
        try:
            log_lines = s.get(f"{BASE_URL}/amc/calibrate/{project_id}/log", timeout=DEFAULT_TIMEOUT).text.splitlines()
            print("    --- last calibration log lines ---")
            for line in log_lines[-20:]:
                print(f"    {line}")
        except Exception:
            pass
        sys.exit(f"Calibration failed. Full log: GET {BASE_URL}/amc/calibrate/{project_id}/log")
    time.sleep(10)
else:
    sys.exit("Timed out after 60 min")

# Step 9 -- Evaluation statistics (GT was uploaded, so this should return metrics)
r = s.get(f"{BASE_URL}/result/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
if r.status_code == 200:
    stats = r.json().get("statistics", r.json())
    print(f"\n[9] Evaluation statistics:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
else:
    print(f"\n[9] evaluation_statistics returned {r.status_code}: {r.text[:200]}")

# Step 10 -- Optional VGGT refinement
if RUN_VGGT:
    info = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=DEFAULT_TIMEOUT).json()
    vggt_state = info.get("project_info", {}).get("vggt_state", "INIT")
    if vggt_state == "READY":
        r = s.post(f"{BASE_URL}/vggt/calibrate/{project_id}", timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        print(f"\n[10] VGGT refinement started")
        t0 = time.time()
        while time.time() - t0 < 900:
            info = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=DEFAULT_TIMEOUT).json()
            vs = info.get("project_info", {}).get("vggt_state", "INIT")
            if vs == "COMPLETED":
                print("[10] VGGT refinement completed")
                r = s.get(f"{BASE_URL}/vggt_results/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
                if r.status_code == 200:
                    print("\n[10] VGGT evaluation statistics:")
                    for k, v in (r.json().get("statistics") or r.json()).items():
                        print(f"    {k}: {v}")
                else:
                    print(f"\n[10] VGGT evaluation_statistics returned {r.status_code}: {r.text[:200]}")
                break
            if vs == "ERROR":
                raise RuntimeError("VGGT refinement failed")
            time.sleep(10)
        else:
            raise RuntimeError("VGGT refinement still running after 15 min")
    else:
        print(f"\n[10] VGGT skipped: project vggt_state={vggt_state}")
        print("     To run VGGT refinement later, set up the VGGT model with amc-setup-calibration-stack and rerun this optional step.")

print(f"\nProject ID: {project_id}")
print("Inspect in UI: open the project in the web UI to view results and overlay videos")
print("Additional dataset: nv_warehouse_032326.zip is available from NGC; download with: ngc registry resource download-version \"nvidia/amc-nv-warehouse\"")
print("Use amc-run-video-calibration with nv_warehouse_config.json and detector=transformer.")