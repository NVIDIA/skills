#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check ROS readiness for the Isaac Sim Unitree G1 cloud-control stack.

Usage:
    python3 check_runtime.py [--base-frame FRAME] [--timeout SECONDS]

Arguments:
    --base-frame selects the robot TF frame; --timeout bounds each ROS query.

Output:
    Writes passed observations to stdout and blocking failures to stderr.

Exit codes:
    0 means ready; 1 means a readiness check or runtime operation failed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time


def run(command: list[str], timeout: float) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode()
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode()
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return 124, (stdout + stderr).strip()


def clock_value(output: str) -> int | None:
    sec = re.search(r"(?m)^\s*sec:\s*(\d+)\s*$", output)
    nanosec = re.search(r"(?m)^\s*nanosec:\s*(\d+)\s*$", output)
    if not sec or not nanosec:
        return None
    return int(sec.group(1)) * 1_000_000_000 + int(nanosec.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-frame", default="pelvis")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    if not shutil.which("ros2"):
        print(
            "FAIL: ros2 is unavailable; run inside the sourced Isaac ROS environment",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    notes: list[str] = []

    code, output = run(["ros2", "topic", "list"], args.timeout)
    if code:
        failures.append(f"topic discovery failed: {output}")
        topics: set[str] = set()
    else:
        topics = set(output.splitlines())

    for topic in ("/clock", "/cmd_vel", "/tf", "/tf_static"):
        if topic not in topics:
            failures.append(f"required topic is missing: {topic}")
    if not any("joint_states" in topic for topic in topics):
        failures.append("no joint-state topic was discovered")
    if not any("imu" in topic.lower() for topic in topics):
        failures.append("no IMU topic was discovered")

    samples: list[int] = []
    for _ in range(2):
        code, output = run(
            ["ros2", "topic", "echo", "/clock", "--once"], args.timeout
        )
        value = clock_value(output) if code == 0 else None
        if value is None:
            failures.append(f"could not read /clock: {output}")
            break
        samples.append(value)
        time.sleep(0.2)
    if len(samples) == 2:
        if samples[1] <= samples[0]:
            failures.append(
                f"/clock did not advance: {samples[0]} -> {samples[1]} ns"
            )
        else:
            notes.append(
                f"OK: /clock advanced by {samples[1] - samples[0]} ns"
            )

    code, output = run(["ros2", "lifecycle", "nodes"], args.timeout)
    lifecycle_nodes = output.splitlines() if code == 0 else []
    for suffix in ("planner_server", "controller_server", "bt_navigator"):
        matches = [
            node.strip()
            for node in lifecycle_nodes
            if node.strip().endswith(suffix)
        ]
        if not matches:
            failures.append(f"Nav2 lifecycle node is missing: *{suffix}")
            continue
        node = matches[0]
        state_code, state = run(
            ["ros2", "lifecycle", "get", node], args.timeout
        )
        if state_code or "active" not in state.lower():
            failures.append(f"Nav2 node is not active: {node}: {state}")
        else:
            notes.append(f"OK: {node} is active")

    _, output = run(
        ["ros2", "run", "tf2_ros", "tf2_echo", "map", args.base_frame],
        min(args.timeout, 4.0),
    )
    if "Translation:" not in output and "At time" not in output:
        failures.append(
            f"TF map -> {args.base_frame} did not resolve: {output}"
        )
    else:
        notes.append(f"OK: TF map -> {args.base_frame} resolved")

    for note in notes:
        print(note)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        print(
            f"Runtime readiness failed with {len(failures)} issue(s).",
            file=sys.stderr,
        )
        return 1
    print(
        "Runtime readiness passed. Verify cloud robot state and stop any "
        "manual /cmd_vel publisher before submission."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
