#!/usr/bin/env python3
"""Run AMC calibration from live RTSP streams recorded through VIOS."""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests


def _env_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float_list(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv(name):
    value = os.environ.get(name, "")
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_base_url(value):
    base = value.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _redact_rtsp_url(url):
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"<redacted>@{host}{port}", parts.path, parts.query, parts.fragment))
    except Exception:
        return "<redacted-rtsp-url>"


def _parse_streams():
    streams_json = os.environ.get("STREAMS_JSON")
    if streams_json:
        streams = json.loads(streams_json)
        if not isinstance(streams, list):
            raise SystemExit("STREAMS_JSON must be a JSON list of stream objects.")
    else:
        urls = _csv("RTSP_URLS")
        if not urls:
            raise SystemExit("Set RTSP_URLS or STREAMS_JSON before running.")
        names = _csv("CAMERA_NAMES") or [f"cam_{i:02d}" for i in range(len(urls))]
        sensor_ids = _csv("SENSOR_IDS")
        if len(names) != len(urls):
            raise SystemExit("CAMERA_NAMES must have the same count as RTSP_URLS.")
        if sensor_ids and len(sensor_ids) != len(urls):
            raise SystemExit("SENSOR_IDS must have the same count as RTSP_URLS when set.")
        streams = []
        for idx, url in enumerate(urls):
            sensor_id = sensor_ids[idx] if sensor_ids else None
            streams.append({"rtsp_url": url, "camera_name": names[idx], "sensor_id": sensor_id or None})

    normalized = []
    for idx, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise SystemExit(f"Stream entry {idx} must be an object.")
        rtsp_url = stream.get("rtsp_url")
        camera_name = stream.get("camera_name") or f"cam_{idx:02d}"
        if not rtsp_url or not str(rtsp_url).startswith("rtsp://"):
            raise SystemExit(f"Stream entry {idx} must include an rtsp:// URL.")
        normalized.append({
            "rtsp_url": rtsp_url,
            "camera_name": camera_name,
            "sensor_id": stream.get("sensor_id"),
        })
    return normalized


def _resolve_local(override, candidate_names, scan_dir, label):
    if override and Path(override).exists():
        return Path(override)
    if scan_dir is None:
        return None
    hits = [scan_dir / name for name in candidate_names if (scan_dir / name).exists()]
    if len(hits) == 1:
        print(f"    auto-detected {label}: {hits[0]}")
        return hits[0]
    if len(hits) > 1:
        print(f"    multiple {label} candidates in {scan_dir}: {hits}; use an explicit path or UI fallback")
    return None


def _json_response(response, label):
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise RuntimeError(f"{label} returned code={code}: {payload}")
    return payload


def _session_from_payload(payload):
    session = payload.get("session") or payload
    if "session_id" not in session:
        raise RuntimeError(f"RTSP capture response did not include session_id: {payload}")
    return session


def _check_amc_ready(session, base_url):
    try:
        response = session.get(f"{base_url}/ready", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"AMC microservice is not reachable at {base_url}/ready: {exc}") from exc


def _check_vios_ready(base_url):
    response = requests.get(f"{base_url.rstrip('/')}/vst/api/v1/sensor/list", timeout=10)
    response.raise_for_status()


def _default_ui_url(base_url):
    explicit = os.environ.get("AMC_UI_URL") or os.environ.get("UI_URL")
    if explicit:
        return explicit.rstrip("/")
    try:
        parts = urlsplit(base_url.rstrip("/"))
        host = parts.hostname or "<HOST_IP>"
        return f"{parts.scheme or 'http'}://{host}:5000"
    except Exception:
        return "http://<HOST_IP>:5000"


def _detector_from_config(path):
    if not path or not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text())
    except Exception:
        return None
    det = cfg.get("detector") or cfg.get("detector_type")
    return det if det in ("resnet", "transformer") else None


def _require_detector(current):
    if current in ("resnet", "transformer"):
        return current
    if not IS_INTERACTIVE:
        raise RuntimeError(
            "Detector type is required before calibration. Set DETECTOR_TYPE=resnet or "
            "DETECTOR_TYPE=transformer, or provide CONFIG_FILE with detector/detector_type."
        )
    while True:
        answer = input("    Detector type for calibration [resnet/transformer]: ").strip().lower()
        if answer in ("resnet", "transformer"):
            return answer
        print("    Enter 'resnet' or 'transformer'.")


# Required env vars: RTSP_URLS or STREAMS_JSON. BASE_URL and PROJECT_NAME have
# defaults for easy editing but should normally be set by the caller.
BASE_URL = _normalize_base_url(os.environ.get("BASE_URL", "http://<HOST_IP>:<MS_PORT>/v1"))
PROJECT_NAME = os.environ.get("PROJECT_NAME", "rtsp_calibration_run")
STREAMS = _parse_streams()
DURATION_SECONDS = int(os.environ.get("DURATION_SECONDS", "180"))
if DURATION_SECONDS < 60:
    raise SystemExit("DURATION_SECONDS must be at least 60.")

CONFIG_FILE = _env_path("CONFIG_FILE")
CALIB_ASSET_DIR = _env_path("CALIB_ASSET_DIR")
ALIGNMENT_JSON = _env_path("ALIGNMENT_JSON")
LAYOUT_PNG = _env_path("LAYOUT_PNG")
GT_ZIP = _env_path("GT_ZIP")
FOCAL_LENGTHS = _env_float_list("FOCAL_LENGTHS")
DETECTOR_TYPE = os.environ.get("DETECTOR_TYPE")
RUN_VGGT = _env_bool("RUN_VGGT", False)
VIOS_TOKEN = os.environ.get("VIOS_TOKEN") or None
SSL_VERIFY = _env_bool("SSL_VERIFY", False)
VIOS_BASE_URL = os.environ.get("VIOS_BASE_URL", "").rstrip("/")
REQUIRE_VIOS_HEALTH = _env_bool("REQUIRE_VIOS_HEALTH", True)

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path.cwd()))
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", REPO_ROOT / "projects"))
IS_INTERACTIVE = sys.stdin.isatty()
AMC_UI_URL = _default_ui_url(BASE_URL)

for label, path in (
    ("CONFIG_FILE", CONFIG_FILE),
    ("CALIB_ASSET_DIR", CALIB_ASSET_DIR),
    ("ALIGNMENT_JSON", ALIGNMENT_JSON),
    ("LAYOUT_PNG", LAYOUT_PNG),
    ("GT_ZIP", GT_ZIP),
):
    if path and not path.exists():
        raise SystemExit(f"{label} set but path not found: {path}")
if CALIB_ASSET_DIR and not CALIB_ASSET_DIR.is_dir():
    raise SystemExit(f"CALIB_ASSET_DIR must be a directory: {CALIB_ASSET_DIR}")

CONFIG_FILE = _resolve_local(
    CONFIG_FILE,
    ["settings.json", "config.json", "calibration_config.json"],
    CALIB_ASSET_DIR,
    "calibration settings",
)
_scan_dir = CALIB_ASSET_DIR or (CONFIG_FILE.parent if CONFIG_FILE else None)
ALIGNMENT_JSON = _resolve_local(ALIGNMENT_JSON, ["alignment_data.json"], _scan_dir, "alignment")
LAYOUT_PNG = _resolve_local(LAYOUT_PNG, ["layout.png"], _scan_dir, "layout")
GT_ZIP = _resolve_local(GT_ZIP, ["GT.zip", "gt.zip"], _scan_dir, "ground truth")

config_detector = _detector_from_config(CONFIG_FILE)
if config_detector:
    DETECTOR_TYPE = config_detector

local_asset_source = any([CONFIG_FILE, ALIGNMENT_JSON, LAYOUT_PNG])
if not local_asset_source:
    print(
        "[0] No local calibration asset source was provided for these RTSP streams.\n"
        "    Provide CALIB_ASSET_DIR or explicit CONFIG_FILE/ALIGNMENT_JSON/LAYOUT_PNG,\n"
        f"    or use AMC UI upload/tuning at {AMC_UI_URL}.\n"
        "    Do not reuse sample dataset assets unless they are explicitly intended for this RTSP scene."
    )
    if not IS_INTERACTIVE:
        raise SystemExit(
            "No local calibration asset source was provided. Run interactively for UI upload, "
            "or set CALIB_ASSET_DIR / CONFIG_FILE / ALIGNMENT_JSON / LAYOUT_PNG."
        )
    answer = input("    Continue and complete settings/alignment in the AMC UI after capture? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Stopped before capture; provide a local asset path or use the AMC UI.")

s = requests.Session()

print("[0] Checking AMC microservice")
_check_amc_ready(s, BASE_URL)

print("[0] Checking VIOS")
if VIOS_BASE_URL:
    try:
        _check_vios_ready(VIOS_BASE_URL)
        print(f"    VIOS reachable at {VIOS_BASE_URL}")
    except requests.RequestException as exc:
        raise SystemExit(f"VIOS_BASE_URL is set but not reachable: {exc}") from exc
else:
    try:
        _check_vios_ready("http://localhost:30888")
        VIOS_BASE_URL = "http://localhost:30888"
        print("    VIOS reachable at http://localhost:30888")
    except requests.RequestException as exc:
        if REQUIRE_VIOS_HEALTH:
            raise SystemExit(
                "VIOS health check failed. Set VIOS_BASE_URL to a reachable VIOS endpoint, "
                "or set REQUIRE_VIOS_HEALTH=false only if the AMC microservice has already "
                "been verified with VIOS_BASE_URL configured."
            ) from exc
        print("    VIOS health check skipped; ensure the AMC microservice has VIOS_BASE_URL configured.")

print("\n[1] RTSP capture plan:")
print(f"    Project:              {PROJECT_NAME}")
print(f"    Stream count:         {len(STREAMS)}")
print(f"    Duration seconds:     {DURATION_SECONDS}")
print(f"    AMC UI:               {AMC_UI_URL}")
print(f"    Calibration assets:   {CALIB_ASSET_DIR if CALIB_ASSET_DIR else 'explicit files or UI'}")
print(f"    Calibration settings: {CONFIG_FILE if CONFIG_FILE else 'UI Step 3 settings/defaults'}")
print(f"    Alignment JSON:       {ALIGNMENT_JSON if ALIGNMENT_JSON else 'UI Step 4/manual_adjustment'}")
print(f"    Layout PNG:           {LAYOUT_PNG if LAYOUT_PNG else 'UI Step 4/manual_adjustment'}")
print(f"    Detector:             {DETECTOR_TYPE if DETECTOR_TYPE else 'ask before calibration'}")
for idx, stream in enumerate(STREAMS):
    sensor_state = "provided" if stream.get("sensor_id") else "auto"
    print(f"    Stream {idx}:           {stream['camera_name']} {sensor_state} {_redact_rtsp_url(stream['rtsp_url'])}")
if IS_INTERACTIVE:
    answer = input("    Start RTSP capture? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Stopped before capture.")
else:
    print("    Non-interactive stdin detected; starting with the plan above.")

# Step 1 -- Create project.
r = s.post(f"{BASE_URL}/create_project", data={"project_name": PROJECT_NAME})
project_id = _json_response(r, "create_project")["project_id"]
print(f"\n[2] Created project: {project_id}")

# Step 2 -- Start RTSP capture.
capture_body = {
    "streams": STREAMS,
    "duration_seconds": DURATION_SECONDS,
    "vios_token": VIOS_TOKEN,
    "ssl_verify": SSL_VERIFY,
}
r = s.post(f"{BASE_URL}/rtsp/capture/{project_id}", json=capture_body, timeout=30)
payload = _json_response(r, "rtsp capture")
session_info = _session_from_payload(payload)
session_id = session_info["session_id"]
print(f"[3] Capture session: {session_id}")

# Step 3 -- Poll capture until the recording is complete, then ingest.
print(f"[4] Polling capture status (timeout {DURATION_SECONDS + 600}s)")
start = time.time()
last_state = ""
while time.time() - start < DURATION_SECONDS + 600:
    r = s.get(f"{BASE_URL}/rtsp/capture/{project_id}/{session_id}", timeout=30)
    payload = _json_response(r, "rtsp capture status")
    sess = payload.get("session") or payload
    state = (sess.get("status") or sess.get("state") or "").upper()
    elapsed = int(time.time() - start)
    if state != last_state:
        print(f"    [{elapsed:>4}s] {state}", flush=True)
        last_state = state
    if state in ("COMPLETED", "INGESTED"):
        break
    if state in ("ERROR", "CANCELLED"):
        raise RuntimeError(f"Capture {state}: {payload}")
    time.sleep(10)
else:
    raise RuntimeError("RTSP capture polling timed out.")

if last_state != "INGESTED":
    r = s.post(f"{BASE_URL}/rtsp/capture/{project_id}/{session_id}/ingest", timeout=120)
    _json_response(r, "rtsp ingest")
    print("[4] Ingested recorded clips into the project")
else:
    print("[4] Capture session was already ingested")

# Step 4 -- Upload settings, alignment, layout, and optional files.
if CONFIG_FILE and CONFIG_FILE.exists():
    r = s.post(
        f"{BASE_URL}/config/{project_id}",
        data=CONFIG_FILE.read_bytes(),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    print(f"[5] Applied calibration config from {CONFIG_FILE.name} (replaces UI Step 3)")
    det = _detector_from_config(CONFIG_FILE)
    if det:
        DETECTOR_TYPE = det
        print(f"    Detector overridden from config: {DETECTOR_TYPE}")

if ALIGNMENT_JSON and ALIGNMENT_JSON.exists():
    with open(ALIGNMENT_JSON, "rb") as f:
        s.post(
            f"{BASE_URL}/upload_alignment/{project_id}",
            files={"alignment_file": (ALIGNMENT_JSON.name, f, "application/json")},
            timeout=60,
        ).raise_for_status()
    print(f"[5] Uploaded alignment: {ALIGNMENT_JSON.name}")

if LAYOUT_PNG and LAYOUT_PNG.exists():
    with open(LAYOUT_PNG, "rb") as f:
        s.post(
            f"{BASE_URL}/upload_layout/{project_id}",
            files={"layout_file": (LAYOUT_PNG.name, f, "image/png")},
            timeout=60,
        ).raise_for_status()
    print(f"[5] Uploaded layout: {LAYOUT_PNG.name}")

if GT_ZIP and GT_ZIP.exists():
    with open(GT_ZIP, "rb") as f:
        s.post(
            f"{BASE_URL}/upload_gt_file/{project_id}",
            files={"gt_file": (GT_ZIP.name, f, "application/zip")},
            timeout=120,
        ).raise_for_status()
    print("[5] Uploaded GT zip")

if FOCAL_LENGTHS:
    s.post(
        f"{BASE_URL}/upload_focal_length/{project_id}",
        data={"focal_length": FOCAL_LENGTHS},
        timeout=60,
    ).raise_for_status()
    print(f"[5] Uploaded focal lengths: {FOCAL_LENGTHS}")

ui_tasks = []
if not CONFIG_FILE:
    ui_tasks.append("Step 3 (Parameters): tune settings or accept defaults, then Save.")
if not ALIGNMENT_JSON or not LAYOUT_PNG:
    ui_tasks.append("Step 4 (Alignment): upload layout, mark correspondence points, then Save. Do not re-upload videos.")
if ui_tasks:
    print(f"\n[5] UI action required for project {project_id}:")
    print(f"    Open AMC UI: {AMC_UI_URL}")
    for task in ui_tasks:
        print(f"    - {task}")
    if not IS_INTERACTIVE:
        raise RuntimeError(
            "UI action is required before continuing. Run interactively, or provide "
            "CONFIG_FILE, ALIGNMENT_JSON, and LAYOUT_PNG so the script can run unattended."
        )
    input("    Press Enter when done...")
    if not ALIGNMENT_JSON or not LAYOUT_PNG:
        manual_dir = PROJECTS_DIR / f"project_{project_id}" / "manual_adjustment"
        assert (manual_dir / "alignment_data.json").exists() and (manual_dir / "layout.png").exists(), (
            f"Alignment files missing under {manual_dir}. Re-check UI Step 4 and click Save."
        )
        print(f"    Alignment files verified at {manual_dir}")

# Step 5 -- Verify, confirm, calibrate, poll, and fetch results.
DETECTOR_TYPE = _require_detector(DETECTOR_TYPE)

r = s.post(f"{BASE_URL}/verify_project/{project_id}", timeout=60)
r.raise_for_status()
verify_payload = r.json()
state = verify_payload.get("project_state") or verify_payload.get("project_info", {}).get("project_state")
print(f"\n[6] Project state: {state}")
assert state == "READY", f"Expected READY, got {state}"

print("\n[7] Calibration plan:")
print(f"    Detector:             {DETECTOR_TYPE}")
print(f"    Stream count:         {len(STREAMS)}")
print(f"    Duration seconds:     {DURATION_SECONDS}")
print(f"    AMC UI:               {AMC_UI_URL}")
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

s.post(f"{BASE_URL}/calibrate/{project_id}", json={"detector_type": DETECTOR_TYPE}, timeout=60).raise_for_status()
print(f"[7] Calibration started (detector={DETECTOR_TYPE})")

print("[8] Polling calibration (10-60 min typical)")
start, last_state, last_beat = time.time(), "", 0.0
while time.time() - start < 5400:
    info = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=30).json()
    st = info["project_info"]["project_state"]
    mins, secs = divmod(int(time.time() - start), 60)
    if st != last_state or time.time() - last_beat >= 60:
        print(f"    [{mins:>3}m {secs:02d}s] {st}", flush=True)
        last_state, last_beat = st, time.time()
    if st == "COMPLETED":
        print(f"[8] Done in {mins}m {secs:02d}s")
        break
    if st == "ERROR":
        try:
            log_lines = s.get(f"{BASE_URL}/amc/calibrate/{project_id}/log", timeout=30).text.splitlines()
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

print("\n[9] Results:")
r = s.get(f"{BASE_URL}/result/{project_id}/evaluation_statistics", timeout=30)
if r.status_code == 200:
    for key, value in (r.json().get("statistics") or r.json()).items():
        print(f"    {key}: {value}")
else:
    print("    No GT provided -- skipping evaluation_statistics")

info = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=30).json()
vggt_state = info.get("project_info", {}).get("vggt_state", "INIT")
if vggt_state == "READY" and not RUN_VGGT:
    if IS_INTERACTIVE:
        answer = input("\n[10] VGGT is READY. Run VGGT refinement now? [y/N]: ").strip().lower()
        RUN_VGGT = answer in ("y", "yes")
    else:
        print("\n[10] VGGT is READY. Set RUN_VGGT=true to run VGGT refinement in non-interactive runs.")
elif vggt_state != "READY":
    print(f"\n[10] VGGT not ready (state={vggt_state}) -- skipping")

if RUN_VGGT and vggt_state == "READY":
    s.post(f"{BASE_URL}/vggt/calibrate/{project_id}", timeout=60).raise_for_status()
    print("\n[10] VGGT started")
    t0 = time.time()
    while time.time() - t0 < 900:
        vs = s.get(f"{BASE_URL}/get_project_info/{project_id}", timeout=30).json() \
            .get("project_info", {}).get("vggt_state", "INIT")
        if vs == "COMPLETED":
            print("     VGGT done")
            r = s.get(f"{BASE_URL}/vggt_results/{project_id}/evaluation_statistics", timeout=30)
            if r.status_code == 200:
                print("     VGGT evaluation statistics:")
                for key, value in (r.json().get("statistics") or r.json()).items():
                    print(f"        {key}: {value}")
            else:
                print(f"     VGGT evaluation statistics unavailable (HTTP {r.status_code})")
            break
        if vs == "ERROR":
            raise RuntimeError("VGGT failed")
        time.sleep(10)

print(f"\nProject: {project_id}")
print("Review the calibration:")
print(f"    UI:                open project {project_id} in the AMC web UI, then the Results page")
print(f"    Final parameters:  projects/project_{project_id}/output/multi_view_results/BA_output/results_ba/refined/camInfo_XX.yaml")
