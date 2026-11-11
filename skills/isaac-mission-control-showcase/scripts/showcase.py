#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helper commands for the default small warehouse Carter showcase."""

import argparse
import hashlib
import http.client
import json
import math
import os
import signal
import subprocess
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import run_state

ROBOT_NAME = "carter01"
MAP_BASENAME = "carter_warehouse_navigation"
MAP_ID = f"{MAP_BASENAME}.png"
EXPECTED_MAP_SHA256 = (
    "dd2f5e382a5f331866becaeaffb391a7e46b595873bf25c9cbb4e280ec261b8e"
)
EXPECTED_START_X = -3.3
EXPECTED_START_Y = 1.3
START_POSE_TOLERANCE_M = 0.5
MIN_MOTION_TRANSLATION_M = 0.25
MIN_MOTION_HEADING_RAD = 0.1
KNOWN_GOOD_ROUTE = [
    {"x": -5.1, "y": 1.4},
    {"x": -2.625, "y": 1.2},
    {"x": -0.2, "y": -2.075},
    {"x": -2.625, "y": -5.35},
    {"x": -5.1, "y": 1.4},
]
TRANSIENT_HTTP_ERRORS = (
    OSError,
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.HTTPException,
    TimeoutError,
    json.JSONDecodeError,
)


def require_http_url(url):
    """Reject any URL whose scheme is not HTTP(S) before it reaches urlopen."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"refusing non-HTTP(S) URL: {url}")
    return url


def fetch_json(url, timeout_s=5):
    """Fetch and decode one JSON document from an HTTP URL."""
    require_http_url(url)
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # nosec B310
        return json.loads(response.read().decode())


def post_json(url, payload, timeout_s=120):
    """POST a JSON payload and return the decoded response text."""
    require_http_url(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310
        return response.read().decode()


def robot_url(database_controller_port, robot_name):
    """Return the Mission Database robot endpoint URL."""
    encoded = urllib.parse.quote(robot_name, safe="")
    return f"http://127.0.0.1:{database_controller_port}/robot/{encoded}"

def mission_url(database_controller_port, mission_id):
    """Return the Mission Database mission endpoint URL."""
    encoded = urllib.parse.quote(mission_id, safe="")
    return f"http://127.0.0.1:{database_controller_port}/mission/{encoded}"



def robot_summary(data):
    """Return a compact, stable robot state summary for logs."""
    status = data.get("status", {})
    pose = status.get("pose", {})
    return {
        "name": data.get("name"),
        "lifecycle": data.get("lifecycle"),
        "state": status.get("state"),
        "online": status.get("online"),
        "position_initialized": status.get("position_initialized"),
        "pose": {key: pose.get(key) for key in ("x", "y", "theta")},
        "errors": status.get("errors"),
    }



def mission_summary(data):
    """Return a compact, stable mission state summary for logs."""
    status = data.get("status", {})
    return {
        "name": data.get("name"),
        "state": status.get("state"),
        "current_node": status.get("current_node"),
        "start_timestamp": status.get("start_timestamp"),
        "end_timestamp": status.get("end_timestamp"),
        "failure_reason": status.get("failure_reason"),
        "failure_category": status.get("failure_category"),
    }


def numeric_pose(data):
    """Return an x/y/theta pose when Mission Database has initialized it."""
    pose = ((data or {}).get("status") or {}).get("pose") or {}
    if not all(
        isinstance(pose.get(key), (int, float))
        for key in ("x", "y", "theta")
    ):
        return None
    return {key: float(pose[key]) for key in ("x", "y", "theta")}


def pose_delta(initial, current):
    """Return translation and wrapped heading change between two poses."""
    translation = math.hypot(
        current["x"] - initial["x"], current["y"] - initial["y"]
    )
    heading = abs(
        (current["theta"] - initial["theta"] + math.pi)
        % (2 * math.pi)
        - math.pi
    )
    return translation, heading

def file_sha256(path):
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_grayscale(path):
    """Validate that a PNG contains only opaque grayscale pixels."""
    try:
        from PIL import Image
    except ImportError:
        print("WARN: PIL unavailable; skipped grayscale class validation")
        return

    image = Image.open(path).convert("RGBA")
    for pixel in image.getdata():
        red, green, blue, alpha = pixel
        if alpha != 255 or red != green or green != blue:
            print(
                f"ERROR: map is not strict grayscale RGBA; "
                f"first bad pixel: {pixel}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    print(f"Map validated: {path} size={image.size} mode=RGBA grayscale")


def validate_maps(args):
    """Validate the bundled, Mission Control, and Carter map resources."""
    paths = [
        Path(args.source),
        Path(args.mission_control),
        Path(args.carter),
    ]
    for path in paths:
        if not path.is_file():
            print(f"ERROR: missing map: {path}", file=sys.stderr)
            return 1
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            print(f"ERROR: unreadable map {path}: {exc}", file=sys.stderr)
            return 1

    hashes = {file_sha256(path) for path in paths}
    if len(hashes) != 1:
        print(
            "ERROR: skill/Mission Control/Carter map PNG hashes differ",
            file=sys.stderr,
        )
        for path in paths:
            print(f"{file_sha256(path)}  {path}", file=sys.stderr)
        return 1

    map_hash = next(iter(hashes))
    if map_hash != EXPECTED_MAP_SHA256:
        print(
            f"ERROR: unexpected bundled map SHA-256: {map_hash}",
            file=sys.stderr,
        )
        return 1

    yaml_paths = [
        Path(args.source_yaml),
        Path(args.mission_control_yaml),
        Path(args.carter_yaml),
    ]
    for path in yaml_paths:
        if not path.is_file():
            print(f"ERROR: missing map metadata: {path}", file=sys.stderr)
            return 1
    yaml_hashes = {file_sha256(path) for path in yaml_paths}
    if len(yaml_hashes) != 1:
        print(
            "ERROR: skill/Mission Control/Carter map metadata differs",
            file=sys.stderr,
        )
        return 1

    validate_grayscale(paths[0])
    print(
        f"map_sha256={map_hash} "
        f"metadata_sha256={next(iter(yaml_hashes))}"
    )
    return 0


def validate_defaults(args):
    """Validate Mission Control defaults point at the map and robot."""
    defaults = Path(args.defaults)
    text = defaults.read_text()
    container_config_dir = args.container_config_dir.rstrip("/")
    expected_map_png = f"{container_config_dir}/maps/carter_warehouse_navigation.png"
    expected_map_yaml = f"{container_config_dir}/maps/carter_warehouse_navigation.yaml"
    if expected_map_png not in text:
        print(
            "ERROR: defaults.yaml is not configured for "
            "carter_warehouse_navigation.png",
            file=sys.stderr,
        )
        print(
            "Run the change-map skill or restore the small warehouse defaults "
            "before this showcase.",
            file=sys.stderr,
        )
        return 1
    if expected_map_yaml not in text:
        print(
            "ERROR: defaults.yaml is not configured for "
            "carter_warehouse_navigation.yaml",
            file=sys.stderr,
        )
        return 1
    if f'name: "{args.robot_name}"' not in text:
        print(
            f"ERROR: defaults.yaml does not define robot {args.robot_name!r}",
            file=sys.stderr,
        )
        return 1
    print(f"defaults_valid robot_name={args.robot_name}")
    return 0


def validate_wpg(args):
    """Require WPG to expose a non-empty graph for the configured map."""
    encoded_map_id = urllib.parse.quote(args.map_id, safe="")
    url = (
        f"http://127.0.0.1:{args.wpg_port}/v1/graph"
        f"?map_id={encoded_map_id}"
    )
    try:
        graph = fetch_json(url, timeout_s=30)
    except TRANSIENT_HTTP_ERRORS as exc:
        print(f"ERROR: WPG map lookup failed: {exc}", file=sys.stderr)
        return 1
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list) or not nodes:
        print(
            f"ERROR: WPG has no graph nodes for map_id={args.map_id}",
            file=sys.stderr,
        )
        return 1
    print(f"wpg_map_id={args.map_id} nodes={len(nodes)}")
    return 0


def wait_robot(args):
    """Wait for Carter to be online, idle, error-free, and at its start pose."""
    deadline = time.time() + args.timeout
    url = robot_url(args.database_controller_port, args.robot_name)
    last = None
    while time.time() < deadline:
        try:
            data = fetch_json(url)
            status = data.get("status", {})
            pose = status.get("pose", {})
            pose_ready = (
                status.get("position_initialized")
                and isinstance(pose.get("x"), (int, float))
                and isinstance(pose.get("y"), (int, float))
                and abs(pose["x"] - EXPECTED_START_X) <= START_POSE_TOLERANCE_M
                and abs(pose["y"] - EXPECTED_START_Y) <= START_POSE_TOLERANCE_M
            )
            last = robot_summary(data)
            if (
                data.get("name") == args.robot_name
                and data.get("lifecycle") == "ALIVE"
                and status.get("online")
                and status.get("state") == "IDLE"
                and not status.get("errors")
                and pose_ready
            ):
                print(json.dumps(last))
                return 0
        except TRANSIENT_HTTP_ERRORS as exc:
            last = f"transient HTTP error: {type(exc).__name__}: {exc}"
        time.sleep(2)

    print(
        f"ERROR: {args.robot_name} was not online/IDLE/error-free "
        "at the expected start pose before timeout",
        file=sys.stderr,
    )
    print(last, file=sys.stderr)
    return 1


def submit_mission(args):
    """Submit the known-good route-only mission to Mission Control."""
    payload = {
        "route": KNOWN_GOOD_ROUTE,
        "timeout": 900,
        "solver": "NVIDIA_CUOPT",
    }
    query = urllib.parse.urlencode({
        "mission_id": args.mission_id,
        "mandatory_robot_name": args.robot_name,
    })
    url = (
        f"http://127.0.0.1:{args.mc_api_port}/api/v1/mission/"
        f"submit_mission?{query}"
    )
    print(post_json(url, payload))
    print(f"mission_id={args.mission_id}")
    print(f"robot_name={args.robot_name}")
    print(f"route={json.dumps(KNOWN_GOOD_ROUTE, separators=(',', ':'))}")
    return 0


def robot_is_healthy(robot, expected_name):
    status = (robot or {}).get("status") or {}
    return bool(
        robot and robot.get("name") == expected_name
        and robot.get("lifecycle") == "ALIVE"
        and status.get("online") is True
        and status.get("position_initialized") is True
        and status.get("errors") in ({}, [])
    )


def process_running(isaac):
    """Reject exited/zombie processes and a reused PID."""
    try:
        fields = Path(f"/proc/{int(isaac['pid'])}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == isaac["start_ticks"]
    except (OSError, KeyError, ValueError, IndexError):
        return False


def record_isaac(args):
    fields = Path(f"/proc/{args.pid}/stat").read_text().rsplit(")", 1)[1].split()
    run_state.update(args.work_dir, isaac={
        "pid": args.pid, "start_ticks": fields[19], "running": fields[0] != "Z",
    })
    return 0


# Executed inside Carter; its cgroup contains the launch process and children.
CARTER_HEALTH_SCRIPT = r'''
import json
import subprocess
from pathlib import Path
required = [
    "nav2_map_server/map_server", "nav2_amcl/amcl",
    "nav2_controller/controller_server", "nav2_planner/planner_server",
    "nav2_bt_navigator/bt_navigator",
    "isaac_ros_vda5050_nav2_client/vda5050_nav2_client",
]
processes = {
    name: subprocess.run(
        ["pgrep", "-f", "^/opt/ros/humble/lib/" + name + " "],
        stdout=subprocess.DEVNULL, check=False,
    ).returncode == 0 for name in required
}
events = dict(line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines())
print(json.dumps({"processes": processes, "oom_events": {
    key: int(events[key]) for key in ("oom", "oom_kill")
}}))
'''


def runtime_health(manifest):
    """Collect bounded, explicit observations; unavailable checks cannot pass."""
    health = {
        "isaac_running": None, "processes_healthy": None, "oom": None,
    }
    try:
        container = manifest["carter_container"]
        state = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", "{{json .State}}", container],
            text=True, timeout=10,
        ))
        if state.get("OOMKilled") is True:
            health["oom"] = True
        if state.get("Running") is not True:
            health["processes_healthy"] = False
            return health
        observed = json.loads(subprocess.check_output(
            ["docker", "exec", container, "python3", "-c", CARTER_HEALTH_SCRIPT],
            text=True, timeout=10,
        ))
        health["processes"] = observed["processes"]
        health["processes_healthy"] = bool(observed["processes"]) and all(
            value is True for value in observed["processes"].values()
        )
        health["oom_events"] = observed["oom_events"]
        counts = [observed["oom_events"][key] for key in ("oom", "oom_kill")]
        if all(type(value) is int and value >= 0 for value in counts):
            if state.get("OOMKilled") is False:
                health["oom"] = any(value > 0 for value in counts)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, TypeError) as exc:
        health["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        health["isaac_running"] = process_running(manifest.get("isaac") or {})
    return health


def check_health(args):
    health = runtime_health(run_state.load(args.work_dir))
    good = (health["isaac_running"] is True
            and health["processes_healthy"] is True and health["oom"] is False)
    if not good:
        print(json.dumps({"runtime_health": health}), file=sys.stderr)
    return 0 if good else 1


def stop_recorded(args):
    """Stop only resources matching this manifest's recorded ownership."""
    manifest = run_state.load(args.work_dir)
    if not manifest:
        raise ValueError("no run manifest for targeted cleanup")
    project = manifest["compose_project"]
    containers = sorted(manifest.get("containers", []),
                        key=lambda name: name != manifest.get("carter_container"))
    for container in containers:
        inspected = subprocess.run(
            ["docker", "inspect", container], capture_output=True, text=True, timeout=10,
        )
        if inspected.returncode:
            if "No such object" in inspected.stderr:
                continue
            raise RuntimeError(inspected.stderr)
        info = json.loads(inspected.stdout)[0]
        labels = info.get("Config", {}).get("Labels") or {}
        if not (labels.get("com.docker.compose.project") == project
                or labels.get("com.nvidia.mission-control-showcase.run") == project):
            raise ValueError(f"container ownership mismatch: {container}")
        if info["State"]["Running"]:
            print(f"stop_container={container}", flush=True)
            if not args.dry_run:
                subprocess.run(["docker", "stop", container], check=True, timeout=60)
    isaac = manifest.get("isaac") or {}
    if process_running(isaac):
        print(f"stop_isaac_pid={isaac['pid']}", flush=True)
        if not args.dry_run:
            os.kill(int(isaac["pid"]), signal.SIGTERM)
            deadline = time.monotonic() + 30
            while process_running(isaac) and time.monotonic() < deadline:
                time.sleep(0.2)
            if process_running(isaac):
                raise RuntimeError("Isaac Sim did not exit after SIGTERM")
    if not args.dry_run:
        run_state.update(args.work_dir, state="stopped", isaac={"running": False})
    return 0


def finalize(args):
    """Write one terminal result; cleanup cannot replace an earlier verdict."""
    manifest = run_state.load(args.work_dir)
    if manifest.get("state") not in run_state.VERDICT_STATES:
        if args.exit_code:
            run_state.update(args.work_dir, state="failed", acceptance={
                "accepted": False, "reasons": [f"{args.phase}: {args.failed_command}"],
            }, failure={"phase": args.phase, "command": args.failed_command,
                        "exit_code": args.exit_code})
        elif args.demo:
            health = {}
            mission, robot = None, None
            try:
                port = manifest["ports"]["database_controller_port"]
                mission = fetch_json(mission_url(port, manifest["mission_id"]))
                robot = fetch_json(robot_url(port, manifest["robot"]))
            except (*TRANSIENT_HTTP_ERRORS, KeyError) as exc:
                health["database_error"] = f"{type(exc).__name__}: {exc}"
            health.update(runtime_health(manifest))
            run_state.update(args.work_dir, isaac={"running": health["isaac_running"]},
                             health=health)
            identity_matches = (mission or {}).get("name") == manifest.get("mission_id")
            run_state.accept(
                args.work_dir,
                mission_state=((mission or {}).get("status") or {}).get("state")
                if identity_matches else None,
                robot_state=((robot or {}).get("status") or {}).get("state"),
                robot_healthy=robot_is_healthy(robot, manifest["robot"]),
                oom=health["oom"], processes_healthy=health["processes_healthy"],
                isaac_running=health["isaac_running"],
            )
    path = run_state.write_result(args.work_dir)
    print(f"run_result={path}")
    return 1 if run_state.load(args.work_dir).get("state") == "failed" else 0


def wait_demo_complete(args):
    """Observe canonical mission fate and independent pose-motion evidence."""
    deadline = time.time() + args.timeout
    mission_endpoint = mission_url(args.database_controller_port, args.mission_id)
    robot_endpoint = robot_url(args.database_controller_port, args.robot_name)
    last = None
    initial_pose = None
    moving_pose = None
    max_translation = 0.0
    max_heading = 0.0

    while time.time() < deadline:
        mission = None
        robot = None
        try:
            mission = fetch_json(mission_endpoint)
        except TRANSIENT_HTTP_ERRORS as exc:
            last = {"mission": f"transient HTTP error: {type(exc).__name__}: {exc}"}

        try:
            robot = fetch_json(robot_endpoint)
        except TRANSIENT_HTTP_ERRORS as exc:
            if last is None:
                last = {}
            last["robot"] = f"transient HTTP error: {type(exc).__name__}: {exc}"

        if mission is None:
            print(json.dumps(last), flush=True)
            time.sleep(2)
            continue

        mission_state = (mission.get("status") or {}).get("state")
        robot_status = (robot or {}).get("status", {})
        current_pose = numeric_pose(robot) if robot_is_healthy(robot, args.robot_name) else None
        if current_pose is not None:
            if initial_pose is None:
                initial_pose = current_pose
            else:
                translation, heading = pose_delta(initial_pose, current_pose)
                max_translation = max(max_translation, translation)
                max_heading = max(max_heading, heading)
                if moving_pose is None and (
                    translation >= MIN_MOTION_TRANSLATION_M
                    or heading >= MIN_MOTION_HEADING_RAD
                ):
                    moving_pose = current_pose
        if getattr(args, "work_dir", None) and initial_pose is not None:
            run_state.record_motion(args.work_dir, {
                "initial_pose": initial_pose, "moving_pose": moving_pose,
                "final_pose": current_pose, "max_translation_m": round(max_translation, 4),
                "max_heading_rad": round(max_heading, 4),
                "motion_confirmed": moving_pose is not None,
            })
        last = {"mission": mission_summary(mission)}
        if robot is not None:
            last["robot"] = robot_summary(robot)
        print(json.dumps(last), flush=True)

        if mission_state == "COMPLETED":
            if robot is None:
                time.sleep(2)
                continue
            if (robot_status.get("state") == "IDLE"
                    and robot_is_healthy(robot, args.robot_name)):
                motion = {
                    "initial_pose": initial_pose,
                    "moving_pose": moving_pose,
                    "final_pose": current_pose,
                    "max_translation_m": round(max_translation, 4),
                    "max_heading_rad": round(max_heading, 4),
                    "motion_confirmed": moving_pose is not None,
                }
                print(json.dumps({"motion_evidence": motion}), flush=True)
                if getattr(args, "work_dir", None):
                    run_state.record_motion(args.work_dir, motion)
                if moving_pose is not None:
                    return 0
                print(
                    "ERROR: mission completed, but Mission Database poses did not confirm motion",
                    file=sys.stderr,
                )
                return 2
        if mission_state in {"FAILED", "CANCELED", "CANCELLED"}:
            print("ERROR: mission finished unsuccessfully", file=sys.stderr)
            return 1
        time.sleep(2)

    print("ERROR: demo mission did not complete before timeout", file=sys.stderr)
    print(last, file=sys.stderr)
    return 1

def run_status(args):
    """Report recorded run state, preferring historical evidence over a sample.

    A stationary sample taken after the mission ended is not proof the robot
    never moved. When the manifest already records confirmed motion, that
    verdict stands and is reported as historical.
    """
    manifest = run_state.load(args.work_dir)
    if not manifest:
        print(
            f"ERROR: no run manifest under {args.work_dir}", file=sys.stderr
        )
        return 1

    sampled = None
    if args.sample_pose:
        try:
            robot = fetch_json(
                robot_url(args.database_controller_port, manifest.get("robot") or "")
            )
            pose = numeric_pose(robot)
            stored = (manifest.get("motion") or {}).get("initial_pose")
            if pose is not None and stored is not None:
                translation, heading = pose_delta(stored, pose)
                sampled = {
                    "max_translation_m": round(translation, 4),
                    "max_heading_rad": round(heading, 4),
                    "motion_confirmed": translation >= MIN_MOTION_TRANSLATION_M
                    or heading >= MIN_MOTION_HEADING_RAD,
                }
        except TRANSIENT_HTTP_ERRORS as exc:
            sampled = {"error": f"{type(exc).__name__}: {exc}"}

    status = {
        "run_id": manifest.get("run_id"),
        "state": manifest.get("state"),
        "mission_id": manifest.get("mission_id"),
        "compose_project": manifest.get("compose_project"),
        "containers": manifest.get("containers"),
        "isaac": manifest.get("isaac"),
        "evidence": manifest.get("evidence"),
        "acceptance": manifest.get("acceptance"),
        "motion": run_state.motion_status(args.work_dir, sampled),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def build_parser():
    """Build the command-line parser for showcase helper subcommands."""
    parser = argparse.ArgumentParser(
        description="Small warehouse Carter showcase helper commands"
    )
    subparsers = parser.add_subparsers(required=True)

    stop_parser = subparsers.add_parser("stop-recorded")
    stop_parser.add_argument("--work-dir", required=True)
    stop_parser.add_argument("--dry-run", action="store_true")
    stop_parser.set_defaults(func=stop_recorded)
    record_parser = subparsers.add_parser("record-isaac")
    record_parser.add_argument("--work-dir", required=True)
    record_parser.add_argument("--pid", type=int, required=True)
    record_parser.set_defaults(func=record_isaac)
    health_parser = subparsers.add_parser("check-health")
    health_parser.add_argument("--work-dir", required=True)
    health_parser.set_defaults(func=check_health)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--work-dir", required=True)
    finalize_parser.add_argument("--exit-code", type=int, required=True)
    finalize_parser.add_argument("--phase", required=True)
    finalize_parser.add_argument("--failed-command", default="")
    finalize_parser.add_argument("--demo", action="store_true")
    finalize_parser.set_defaults(func=finalize)

    validate_maps_parser = subparsers.add_parser("validate-maps")
    validate_maps_parser.add_argument("--source", required=True)
    validate_maps_parser.add_argument("--source-yaml", required=True)
    validate_maps_parser.add_argument("--mission-control", required=True)
    validate_maps_parser.add_argument("--mission-control-yaml", required=True)
    validate_maps_parser.add_argument("--carter", required=True)
    validate_maps_parser.add_argument("--carter-yaml", required=True)
    validate_maps_parser.set_defaults(func=validate_maps)

    validate_defaults_parser = subparsers.add_parser("validate-defaults")
    validate_defaults_parser.add_argument("--defaults", required=True)
    validate_defaults_parser.add_argument("--container-config-dir", required=True)
    validate_defaults_parser.add_argument("--robot-name", default=ROBOT_NAME)
    validate_defaults_parser.set_defaults(func=validate_defaults)

    validate_wpg_parser = subparsers.add_parser("validate-wpg")
    validate_wpg_parser.add_argument("--wpg-port", type=int, default=8000)
    validate_wpg_parser.add_argument("--map-id", default=MAP_ID)
    validate_wpg_parser.set_defaults(func=validate_wpg)

    wait_robot_parser = subparsers.add_parser("wait-robot")
    wait_robot_parser.add_argument(
        "--database-controller-port", type=int, required=True
    )
    wait_robot_parser.add_argument("--timeout", type=int, required=True)
    wait_robot_parser.add_argument("--robot-name", default=ROBOT_NAME)
    wait_robot_parser.set_defaults(func=wait_robot)

    submit_mission_parser = subparsers.add_parser("submit-mission")
    submit_mission_parser.add_argument("--mission-id", required=True)
    submit_mission_parser.add_argument(
        "--mc-api-port", type=int, required=True)
    submit_mission_parser.add_argument("--robot-name", default=ROBOT_NAME)
    submit_mission_parser.set_defaults(func=submit_mission)

    run_status_parser = subparsers.add_parser("run-status")
    run_status_parser.add_argument("--work-dir", required=True)
    run_status_parser.add_argument("--database-controller-port", type=int, default=5000)
    run_status_parser.add_argument("--sample-pose", action="store_true")
    run_status_parser.set_defaults(func=run_status)
    wait_demo_parser = subparsers.add_parser("wait-demo-complete")
    wait_demo_parser.add_argument("--work-dir")
    wait_demo_parser.add_argument("--mission-id", required=True)
    wait_demo_parser.add_argument(
        "--database-controller-port", type=int, required=True
    )
    wait_demo_parser.add_argument("--timeout", type=int, required=True)
    wait_demo_parser.add_argument("--robot-name", default=ROBOT_NAME)
    wait_demo_parser.set_defaults(func=wait_demo_complete)

    return parser


def main(argv=None):
    """Run one showcase helper subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
