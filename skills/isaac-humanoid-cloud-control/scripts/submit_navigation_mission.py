#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview or submit a Unitree G1 route through Mission Control release 4.6.

Usage:
    python3 submit_navigation_mission.py --host HOST [ROUTE] [--submit]

Arguments:
    --host and route coordinates are required; robot, ports, IDs, and timeout are
    optional. Actual submission also requires --validated-free-space --submit after
    explicit operator approval of the previewed payload.

Output:
    Prints the exact endpoint and payload, readiness evidence, and API response.

Exit codes:
    0 means preview or acceptance succeeded; 1 means validation or submission failed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import ipaddress
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


from http_transport import open_http


NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        required=True,
        help="Mission Control host without a URL scheme",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--database-controller-port", type=int, default=5003)
    parser.add_argument("--scheme", choices=("http", "https"), default="http")
    parser.add_argument("--robot", default="g1_0")
    parser.add_argument(
        "--mission-id",
        help="Unique mission ID; defaults to a UTC timestamped ID",
    )
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument(
        "--waypoint",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA"),
        help="Route waypoint; repeat for a multi-waypoint route",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Submit the supplied loop-shaped route; include a validated final "
            "waypoint near, but not equal to, the start"
        ),
    )
    parser.add_argument("--map-id", default="map")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Mission timeout in seconds",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Mission Control solver/HTTP timeout in seconds",
    )
    parser.add_argument("--confirm-free-space", action="store_true")
    parser.add_argument(
        "--validated-free-space",
        action="store_true",
        help="Assert that occupancy, clearance, and route reachability were checked",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Perform the explicitly operator-approved POST; default is preview only",
    )
    args = parser.parse_args()

    # Keep authority syntax out of the host argument; format IPv6 explicitly.
    try:
        address = ipaddress.ip_address(args.host.strip("[]"))
        args.host = f"[{address}]" if address.version == 6 else str(address)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", args.host):
            parser.error("--host must be a hostname or IP without URL syntax")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    if not NAME_RE.fullmatch(args.robot):
        parser.error(
            "--robot may contain only letters, digits, underscores, and dashes"
        )
    mission_id = args.mission_id or (
        f"{args.robot}-nav-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if not NAME_RE.fullmatch(mission_id):
        parser.error(
            "--mission-id may contain only letters, digits, underscores, and dashes"
        )
    if not NAME_RE.fullmatch(args.map_id):
        parser.error(
            "--map-id may contain only letters, digits, underscores, and dashes"
        )
    if args.waypoint and (args.x is not None or args.y is not None):
        parser.error("use either repeatable --waypoint or --x/--y, not both")
    if args.waypoint:
        coordinates = [tuple(waypoint) for waypoint in args.waypoint]
    else:
        if args.x is None or args.y is None:
            parser.error("provide --x and --y or at least one --waypoint")
        coordinates = [(args.x, args.y, args.theta)]
    if args.loop:
        if len(coordinates) < 3:
            parser.error("--loop requires at least three supplied waypoints")
        first = coordinates[0]
        last = coordinates[-1]
        if math.hypot(last[0] - first[0], last[1] - first[1]) <= 0.11:
            parser.error(
                "--loop final waypoint must be more than 0.11 m from the first; "
                "use a validated near-start endpoint so Nav2 cannot immediately "
                "satisfy a closed route at its initial pose"
            )
    if not all(
        math.isfinite(value)
        for waypoint in coordinates
        for value in waypoint
    ):
        parser.error("mission coordinates must be finite")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.database_controller_port <= 65535:
        parser.error("--database-controller-port must be between 1 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    # Mission Control release 4.6 models an explicit route as Point2D values.
    # Theta and map ID remain inputs to the pre-submission Nav2/map checks, but
    # are not fields in the MissionData route schema.
    waypoints = [{"x": x, "y": y} for x, y, _theta in coordinates]
    payload = {
        "route": waypoints,
        "timeout": args.timeout,
        "solver": "NVIDIA_CUOPT",
    }
    query = urllib.parse.urlencode(
        {"mission_id": mission_id, "mandatory_robot_name": args.robot}
    )
    endpoint = (
        f"{args.scheme}://{args.host}:{args.port}"
        f"/api/v1/mission/submit_mission?{query}"
    )
    robot_endpoint = (
        f"{args.scheme}://{args.host}:{args.database_controller_port}"
        f"/robot/{urllib.parse.quote(args.robot, safe='')}"
    )
    mission_record_endpoint = (
        f"{args.scheme}://{args.host}:{args.database_controller_port}"
        f"/mission/{urllib.parse.quote(mission_id, safe='')}"
    )
    print(f"Mission ID: {mission_id}")
    print(f"Endpoint: {endpoint}")
    print(json.dumps(payload, indent=2))

    if not args.submit:
        print(
            "Preview only. Validate occupancy, footprint clearance, and route "
            "reachability, obtain explicit operator approval of this payload, then add "
            "--validated-free-space --submit with the same --mission-id."
        )
        return 0
    if not (args.confirm_free_space or args.validated_free_space):
        sys.stdout.flush()
        print(
            "Refusing submission: --validated-free-space or "
            "--confirm-free-space is required.",
            file=sys.stderr,
        )
        return 2

    try:
        with open_http(
            robot_endpoint, timeout=args.request_timeout
        ) as response:
            robot = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Robot readiness check failed: {exc}", file=sys.stderr)
        return 1
    status = robot.get("status", {})
    if not (
        robot.get("name") == args.robot
        and robot.get("lifecycle") == "ALIVE"
        and status.get("online") is True
        and status.get("state") == "IDLE"
        and status.get("position_initialized") is True
        and not status.get("errors")
    ):
        print(
            "Refusing submission: robot is not ALIVE, online, IDLE, "
            "position-initialized, and error-free.",
            file=sys.stderr,
        )
        print(json.dumps(robot, indent=2), file=sys.stderr)
        return 1
    print("Robot readiness: ALIVE, online, IDLE, position-initialized, error-free")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with open_http(
            request, timeout=args.request_timeout
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"Mission Control response: HTTP {response.status}")
            print(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            f"Mission submission failed: HTTP {exc.code}: {body}",
            file=sys.stderr,
        )
        return 1
    except TimeoutError:
        # Mission Control can finish route solving after the HTTP client has
        # timed out. Resolve the ambiguous POST through the canonical database
        # before failing; never blindly retry and risk a duplicate mission.
        try:
            with open_http(
                mission_record_endpoint, timeout=args.request_timeout
            ) as response:
                mission = json.loads(response.read().decode("utf-8"))
            state = mission.get("status", {}).get("state")
            if mission.get("name") == mission_id and state in {
                "PENDING",
                "RUNNING",
                "COMPLETED",
            }:
                print(
                    "Mission Control response timed out, but Mission Database "
                    f"confirms mission state {state}."
                )
            else:
                raise RuntimeError("canonical mission record is not active")
        except Exception as exc:  # noqa: BLE001
            print(
                "Mission submission timed out and canonical acceptance could "
                f"not be established: {exc}",
                file=sys.stderr,
            )
            return 1
    except urllib.error.URLError as exc:
        print(f"Mission submission failed: {exc.reason}", file=sys.stderr)
        return 1

    print(
        f"Mission {mission_id} accepted. Read submit-and-monitor-mission from the dependency manifest and verify this ID without resubmitting; "
        "acceptance is not completion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
