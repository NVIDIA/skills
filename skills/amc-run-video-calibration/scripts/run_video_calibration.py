#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run AMC calibration for user-provided pre-recorded videos.

Usage:
    Set BASE_URL, PROJECT_NAME, and VIDEO_DIR, then run this module directly.

Environment arguments:
    Required: BASE_URL, PROJECT_NAME, VIDEO_DIR
    Optional: CONFIG_FILE, ALIGNMENT_JSON, LAYOUT_PNG, GT_ZIP, FOCAL_LENGTHS,
        DETECTOR_TYPE, RUN_VGGT, REPO_ROOT, PROJECTS_DIR,
        CONFIRM_CALIBRATION, CALIBRATION_TIMEOUT_SECONDS,
        VGGT_TIMEOUT_SECONDS

Output:
    Prints upload/progress status, project ID, evaluation metrics, and the
    server-side result locations to review in AMC.

Exit codes:
    0 for success, 1 for handled workflow failures, and 2 for unexpected
    runtime failures.
"""

import ipaddress
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires Python 3 with the 'requests' package installed. "
        "Install it first, for example: python3 -m pip install requests"
    ) from exc


DEFAULT_TIMEOUT = (10, 120)
VIDEO_UPLOAD_TIMEOUT_SECONDS = 300
GT_UPLOAD_TIMEOUT_SECONDS = 120


def _env_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


def _env_float_list(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _is_loopback_host(hostname):
    if not hostname:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _normalize_base_url(value):
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            "BASE_URL must be an absolute http:// or https:// URL, for example "
            "https://amc.example.com/v1 or http://localhost:8000/v1"
        )
    path = parsed.path.rstrip("/")
    normalized_path = path if path.endswith("/v1") else f"{path}/v1" if path else "/v1"
    if (
        parsed.scheme == "http"
        and not _is_loopback_host(parsed.hostname)
        and not _env_bool("ALLOW_INSECURE_HTTP", False)
    ):
        raise ValueError(
            "BASE_URL uses cleartext HTTP for a non-loopback host. Use HTTPS, "
            "tunnel the service to localhost, or set ALLOW_INSECURE_HTTP=true "
            "only for trusted development setups."
        )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, normalized_path, parsed.query, parsed.fragment)
    )


def _response_detail(response):
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        return text[:400] if text else "<empty response>"


def _response_json_object(response, label):
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{label} did not return JSON: {_response_detail(response)}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} returned an unexpected JSON payload: {payload!r}")
    return payload


def _require_field(payload, field_name, label):
    if field_name not in payload or payload[field_name] in (None, ""):
        raise RuntimeError(f"{label} response missing {field_name!r}: {payload}")
    return payload[field_name]


def _require_object_field(payload, field_name, label):
    value = _require_field(payload, field_name, label)
    if not isinstance(value, dict):
        raise TypeError(
            f"{label} field {field_name!r} must be a JSON object, got {value!r}"
        )
    return value


def _require_string_field(payload, field_name, label):
    value = _require_field(payload, field_name, label)
    if not isinstance(value, str):
        raise TypeError(
            f"{label} field {field_name!r} must be a string, got {value!r}"
        )
    return value


def _raise_for_status(response, label):
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"{label} failed: HTTP {response.status_code}: {_response_detail(response)}"
        ) from exc


def _check_amc_ready(session, base_url):
    response = session.get(f"{base_url}/ready", timeout=DEFAULT_TIMEOUT)
    _raise_for_status(response, "AMC readiness probe")
    payload = _response_json_object(response, "AMC readiness probe")
    if payload.get("code") not in (None, 0, "0"):
        raise RuntimeError(f"AMC readiness probe returned non-zero code: {payload}")


def _resolve_local(override, candidate_names, scan_dirs, label):
    """Return a local file path if there is a single unambiguous match."""
    if override and override.exists():
        return override

    hits = []
    for scan_dir in scan_dirs:
        for name in candidate_names:
            candidate = scan_dir / name
            if candidate.exists():
                hits.append(candidate)

    if len(hits) == 1:
        print(f"    auto-detected {label}: {hits[0]}")
        return hits[0]

    searched = [str(scan_dir) for scan_dir in scan_dirs]
    if len(hits) > 1:
        print(
            f"    multiple {label} candidates found in {searched}: {hits}; "
            "use an explicit path or UI fallback"
        )
    else:
        print(
            f"    no {label} file auto-detected in {searched}; "
            "use an explicit path or UI fallback"
        )
    return None


def _parse_detector_from_config(config_file, current_detector):
    try:
        config_payload = json.loads(config_file.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"    warning: could not parse detector_type from {config_file}: {exc}; "
            f"continuing with DETECTOR_TYPE={current_detector}"
        )
        return current_detector

    if not isinstance(config_payload, dict):
        print(
            f"    warning: config file {config_file} did not parse to a JSON object; "
            f"continuing with DETECTOR_TYPE={current_detector}"
        )
        return current_detector

    detector = config_payload.get("detector") or config_payload.get("detector_type")
    if detector in ("resnet", "transformer"):
        print(f"    Detector overridden from config: {detector}")
        return detector
    if detector is not None:
        print(
            f"    warning: unsupported detector {detector!r} in {config_file}; "
            f"continuing with DETECTOR_TYPE={current_detector}"
        )
    return current_detector


def _get_project_info(session, base_url, project_id):
    response = session.get(
        f"{base_url}/get_project_info/{project_id}",
        timeout=DEFAULT_TIMEOUT,
    )
    _raise_for_status(response, "get_project_info")
    return _response_json_object(response, "get_project_info")


def _project_state(payload, label):
    project_info = _require_object_field(payload, "project_info", label)
    return _require_string_field(project_info, "project_state", f"{label}.project_info")


def _vggt_state(payload, label):
    project_info = _require_object_field(payload, "project_info", label)
    value = project_info.get("vggt_state", "INIT")
    if not isinstance(value, str):
        raise TypeError(
            f"{label}.project_info field 'vggt_state' must be a string, got {value!r}"
        )
    return value


def _statistics_payload(response, label):
    payload = _response_json_object(response, label)
    statistics = payload.get("statistics", payload)
    if not isinstance(statistics, dict):
        raise TypeError(f"{label} returned unexpected statistics payload: {payload!r}")
    return statistics


def _validate_video_files(video_files, video_dir):
    if not video_files:
        raise RuntimeError(
            f"No cam_*.mp4 files were found under {video_dir}. "
            "Provide a directory with contiguous cam_00.mp4, cam_01.mp4, ... files."
        )

    actual_names = [path.name for path in video_files]
    expected_names = [f"cam_{index:02d}.mp4" for index in range(len(video_files))]
    if actual_names != expected_names:
        raise RuntimeError(
            "VIDEO_DIR must contain a contiguous camera sequence named "
            "cam_00.mp4, cam_01.mp4, ... with no gaps. "
            f"Found {actual_names} under {video_dir}."
        )


def main():
    # Required env vars: BASE_URL, PROJECT_NAME, VIDEO_DIR. Optional file paths and
    # knobs use CONFIG_FILE, ALIGNMENT_JSON, LAYOUT_PNG, GT_ZIP, FOCAL_LENGTHS,
    # DETECTOR_TYPE, RUN_VGGT, REPO_ROOT, PROJECTS_DIR, CONFIRM_CALIBRATION,
    # CALIBRATION_TIMEOUT_SECONDS, and VGGT_TIMEOUT_SECONDS.
    base_url = _normalize_base_url(
        os.environ.get("BASE_URL", "http://<HOST_IP>:<MS_PORT>/v1")
    )
    project_name = os.environ.get("PROJECT_NAME", "my_calibration_run")
    video_dir = Path(os.environ.get("VIDEO_DIR", "/path/to/videos")).expanduser()
    config_file = _env_path("CONFIG_FILE")
    alignment_json = _env_path("ALIGNMENT_JSON")
    layout_png = _env_path("LAYOUT_PNG")
    gt_zip = _env_path("GT_ZIP")
    focal_lengths = _env_float_list("FOCAL_LENGTHS")
    detector_type = os.environ.get("DETECTOR_TYPE", "resnet")
    run_vggt = _env_bool("RUN_VGGT", False)
    confirm_calibration = _env_bool("CONFIRM_CALIBRATION", False)
    calibration_timeout_seconds = _env_int("CALIBRATION_TIMEOUT_SECONDS", 5400)
    vggt_timeout_seconds = _env_int("VGGT_TIMEOUT_SECONDS", 900)

    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd())).expanduser()
    projects_dir = Path(os.environ.get("PROJECTS_DIR", repo_root / "projects")).expanduser()
    is_interactive = sys.stdin.isatty()

    for label, path in (
        ("VIDEO_DIR", video_dir),
        ("CONFIG_FILE", config_file),
        ("ALIGNMENT_JSON", alignment_json),
        ("LAYOUT_PNG", layout_png),
        ("GT_ZIP", gt_zip),
    ):
        if path and not path.exists():
            raise RuntimeError(f"{label} set but path not found: {path}")

    if not video_dir.is_dir():
        raise RuntimeError(f"VIDEO_DIR must point to a directory: {video_dir}")

    video_files = sorted(video_dir.glob("cam_*.mp4"))
    _validate_video_files(video_files, video_dir)

    first_level_dirs = sorted(
        (path for path in video_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    scan_dirs = [video_dir, *first_level_dirs, video_dir.parent]
    config_file = _resolve_local(
        config_file,
        ["settings.json", "config.json", "calibration_config.json"],
        scan_dirs,
        "config",
    )
    alignment_json = _resolve_local(
        alignment_json,
        ["alignment_data.json"],
        scan_dirs,
        "alignment",
    )
    layout_png = _resolve_local(
        layout_png,
        ["layout.png"],
        scan_dirs,
        "layout",
    )

    session = requests.Session()
    _check_amc_ready(session, base_url)

    response = session.post(
        f"{base_url}/create_project",
        data={"project_name": project_name},
        timeout=DEFAULT_TIMEOUT,
    )
    _raise_for_status(response, "create_project")
    project_id = _require_field(
        _response_json_object(response, "create_project"),
        "project_id",
        "create_project",
    )
    print(f"[1] Created project: {project_id}")

    with ExitStack() as stack:
        files = []
        for video in video_files:
            handle = stack.enter_context(open(video, "rb"))
            files.append(("files", (video.name, handle, "video/mp4")))
        response = session.post(
            f"{base_url}/upload_video_files/{project_id}",
            files=files,
            timeout=VIDEO_UPLOAD_TIMEOUT_SECONDS,
        )
    _raise_for_status(response, "upload_video_files")
    print(f"[2] Uploaded {len(video_files)} videos")

    if config_file:
        response = session.post(
            f"{base_url}/config/{project_id}",
            data=config_file.read_bytes(),
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_status(response, "upload config")
        print(f"[3] Applied calibration config from {config_file.name} (replaces UI Step 3)")
        detector_type = _parse_detector_from_config(config_file, detector_type)

    if alignment_json:
        with open(alignment_json, "rb") as handle:
            response = session.post(
                f"{base_url}/upload_alignment/{project_id}",
                files={"alignment_file": (alignment_json.name, handle, "application/json")},
                timeout=DEFAULT_TIMEOUT,
            )
        _raise_for_status(response, "upload_alignment")
        print(f"[3] Uploaded alignment: {alignment_json.name}")

    if layout_png:
        with open(layout_png, "rb") as handle:
            response = session.post(
                f"{base_url}/upload_layout/{project_id}",
                files={"layout_file": (layout_png.name, handle, "image/png")},
                timeout=DEFAULT_TIMEOUT,
            )
        _raise_for_status(response, "upload_layout")
        print(f"[3] Uploaded layout: {layout_png.name}")

    if gt_zip:
        with open(gt_zip, "rb") as handle:
            response = session.post(
                f"{base_url}/upload_gt_file/{project_id}",
                files={"gt_file": (gt_zip.name, handle, "application/zip")},
                timeout=GT_UPLOAD_TIMEOUT_SECONDS,
            )
        _raise_for_status(response, "upload_gt_file")
        print("[3] Uploaded GT zip")

    if focal_lengths:
        response = session.post(
            f"{base_url}/upload_focal_length/{project_id}",
            data={"focal_length": focal_lengths},
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_status(response, "upload_focal_length")
        print(f"[3] Uploaded focal lengths: {focal_lengths}")

    ui_tasks = []
    if not config_file:
        ui_tasks.append("Step 3 (Parameters): tune settings or accept defaults, then Save.")
    if not alignment_json or not layout_png:
        ui_tasks.append("Step 4 (Alignment): upload layout, mark correspondence points, then Save.")

    if ui_tasks:
        print(f"\n[5] UI action required for project {project_id}:")
        for task in ui_tasks:
            print(f"    - {task}")
        if not is_interactive:
            raise RuntimeError(
                "UI action is required before continuing. Run interactively, or provide "
                "CONFIG_FILE, ALIGNMENT_JSON, and LAYOUT_PNG so the script can run unattended."
            )
        input("    Press Enter when done...")
        if not alignment_json or not layout_png:
            print("    Continuing with verify_project to confirm the UI uploads.")

    response = session.post(
        f"{base_url}/verify_project/{project_id}",
        timeout=DEFAULT_TIMEOUT,
    )
    _raise_for_status(response, "verify_project")
    verify_payload = _response_json_object(response, "verify_project")
    project_state = _require_string_field(
        verify_payload,
        "project_state",
        "verify_project",
    )
    print(f"[6] Project state: {project_state}")
    if project_state != "READY":
        raise RuntimeError(
            f"verify_project returned {project_state!r}; expected 'READY'. "
            "Confirm videos, alignment, and layout were uploaded successfully."
        )

    print("\n[7] Calibration plan:")
    print(f"    Detector:             {detector_type}")
    print(
        f"    Calibration settings: "
        f"{config_file if config_file else 'UI Step 3 settings/defaults'}"
    )
    print(
        f"    Alignment JSON:       "
        f"{alignment_json if alignment_json else 'UI Step 4/manual_adjustment'}"
    )
    print(
        f"    Layout PNG:           "
        f"{layout_png if layout_png else 'UI Step 4/manual_adjustment'}"
    )
    print(f"    Ground truth zip:     {gt_zip if gt_zip else 'not provided'}")
    print(f"    Focal lengths:        {focal_lengths if focal_lengths else 'not provided'}")

    if is_interactive:
        answer = input("    Start calibration? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("Stopped before calibration.")
    else:
        if not confirm_calibration:
            raise RuntimeError(
                "Non-interactive mode requires explicit confirmation. "
                "Set CONFIRM_CALIBRATION=true to proceed, or run interactively."
            )
        print(
            "    Non-interactive stdin detected; CONFIRM_CALIBRATION=true set; "
            "starting with the plan above."
        )

    response = session.post(
        f"{base_url}/calibrate/{project_id}",
        json={"detector_type": detector_type},
        timeout=DEFAULT_TIMEOUT,
    )
    _raise_for_status(response, "calibrate")
    print(f"[7] Calibration started (detector={detector_type})")

    print("[8] Polling (10-60 min typical)...")
    start_time = time.time()
    last_state = ""
    last_heartbeat = 0.0
    while time.time() - start_time < calibration_timeout_seconds:
        info = _get_project_info(session, base_url, project_id)
        project_state = _project_state(info, "get_project_info")
        minutes, seconds = divmod(int(time.time() - start_time), 60)
        if project_state != last_state or time.time() - last_heartbeat >= 60:
            print(f"    [{minutes:>3}m {seconds:02d}s] {project_state}", flush=True)
            last_state = project_state
            last_heartbeat = time.time()
        if project_state == "COMPLETED":
            print(f"[8] Done in {minutes}m {seconds:02d}s")
            break
        if project_state == "ERROR":
            try:
                response = session.get(
                    f"{base_url}/amc/calibrate/{project_id}/log",
                    timeout=DEFAULT_TIMEOUT,
                )
                _raise_for_status(response, "fetch calibration log")
                log_lines = response.text.splitlines()
                print("    --- last calibration log lines ---")
                for line in log_lines[-20:]:
                    print(f"    {line}")
            except RuntimeError as exc:
                print(f"    warning: {exc}")
            raise RuntimeError(
                f"Calibration entered ERROR state. Full log: "
                f"GET {base_url}/amc/calibrate/{project_id}/log"
            )
        time.sleep(10)
    else:
        raise RuntimeError(
            f"Calibration still running after {int((time.time() - start_time) // 60)} min. "
            "Increase CALIBRATION_TIMEOUT_SECONDS if needed, or inspect "
            f"GET {base_url}/amc/calibrate/{project_id}/log"
        )

    print("\n[9] Results:")
    response = session.get(
        f"{base_url}/result/{project_id}/evaluation_statistics",
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code == 200:
        for key, value in _statistics_payload(response, "evaluation_statistics").items():
            print(f"    {key}: {value}")
    elif gt_zip is None:
        print(
            f"    Evaluation statistics unavailable (HTTP {response.status_code}). "
            "Ground truth was not uploaded, so this is expected."
        )
    else:
        raise RuntimeError(
            f"evaluation_statistics failed after GT upload: HTTP {response.status_code}: "
            f"{_response_detail(response)}"
        )

    info = _get_project_info(session, base_url, project_id)
    vggt_state = _vggt_state(info, "get_project_info")
    if vggt_state == "READY" and not run_vggt:
        if is_interactive:
            answer = input("\n[10] VGGT is READY. Run VGGT refinement now? [y/N]: ").strip().lower()
            run_vggt = answer in ("y", "yes")
        else:
            print("\n[10] VGGT is READY. Set RUN_VGGT=true to run VGGT refinement in non-interactive runs.")
    elif vggt_state != "READY":
        print(f"\n[10] VGGT not ready (state={vggt_state}) -- skipping")
        print(
            "     To run VGGT refinement later, set up the VGGT model with "
            "amc-setup-calibration-stack and rerun this optional step."
        )

    if run_vggt and vggt_state == "READY":
        response = session.post(
            f"{base_url}/vggt/calibrate/{project_id}",
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_status(response, "vggt/calibrate")
        print("\n[10] VGGT started")
        vggt_start_time = time.time()
        while time.time() - vggt_start_time < vggt_timeout_seconds:
            info = _get_project_info(session, base_url, project_id)
            vggt_state = _vggt_state(info, "get_project_info")
            if vggt_state == "COMPLETED":
                print("     VGGT done")
                response = session.get(
                    f"{base_url}/vggt_results/{project_id}/evaluation_statistics",
                    timeout=DEFAULT_TIMEOUT,
                )
                if response.status_code == 200:
                    print("     VGGT evaluation statistics:")
                    for key, value in _statistics_payload(
                        response, "vggt evaluation_statistics"
                    ).items():
                        print(f"        {key}: {value}")
                else:
                    raise RuntimeError(
                        f"VGGT evaluation_statistics failed: HTTP {response.status_code}: "
                        f"{_response_detail(response)}"
                    )
                break
            if vggt_state == "ERROR":
                raise RuntimeError("VGGT refinement entered ERROR state")
            time.sleep(10)
        else:
            raise RuntimeError(
                f"VGGT refinement still running after {vggt_timeout_seconds // 60} min. "
                "Increase VGGT_TIMEOUT_SECONDS if needed and rerun the optional step."
            )

    print(f"\nProject: {project_id}")
    print("Review the calibration:")
    print(
        f"    UI:                open project {project_id} in the AMC web UI, "
        "then the Results page to view the overlay"
    )
    print(
        f"    Final camera parameters: "
        f"{projects_dir}/project_{project_id}/output/multi_view_results/BA_output/"
        "results_ba/refined/camInfo_XX.yaml"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"ERROR: network request failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        raise SystemExit(1) from None
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
