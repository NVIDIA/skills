#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enable AGILE and release the Isaac Sim gantry after measured stabilization.

Usage:
    python3 guarded_gantry_release.py --allow-simulation-gantry-release [OPTIONS]

Arguments:
    The authorization flag is mandatory; timing and attitude limits are optional.

Output:
    Writes gate, stability, release, and re-engagement evidence to the terminal.

Exit codes:
    0 means guarded release passed; 1 means authorization or a safety gate failed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import rclpy
from controller_manager_msgs.srv import ListHardwareComponents
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import SetBool


SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR / "shared"))
from simulation_gate import require_local_domain, require_simulation  # noqa: E402


def verify_simulation(node, domain):
    """Collect live, read-only evidence before any controller or gantry mutation."""
    # Keep spinning ROS while the separate scene probe runs: the zero-command
    # timer and fresh sensor subscriptions must not stall during revalidation.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            subprocess.run,
            [sys.executable, str(SKILL_DIR / "references/isaac-sim-remote/scripts/run.py"),
             "--host", "127.0.0.1", "--port", "8226", "--timeout", "15", "--raw",
             "--file", str(SKILL_DIR / "scripts/simulation_identity.py")],
            capture_output=True, text=True, check=True, timeout=25,
        )
        spin_until(node, pending.done, 30.0, "live simulation identity")
        completed = pending.result()
    envelope = json.loads(completed.stdout)
    if envelope.get("status") != "ok":
        raise RuntimeError("live Isaac Sim identity probe failed")
    identity = json.loads(envelope["output"])
    hardware_client = node.create_client(
        ListHardwareComponents, "/controller_manager/list_hardware_components")
    time_client = node.create_client(GetParameters, "/controller_manager/get_parameters")
    try:
        hardware = call(node, hardware_client, ListHardwareComponents.Request(),
                        15.0, "hardware identity")
        clock = call(node, time_client, GetParameters.Request(names=["use_sim_time"]),
                     15.0, "controller clock identity")
        providers = []
        for name, namespace in node.get_node_names_and_namespaces():
            services = node.get_service_names_and_types_by_node(name, namespace)
            if any(service == "/launcher/gantry/set_enabled" for service, _ in services):
                providers.append((name, namespace))
        use_sim_time = (len(clock.values) == 1
                        and clock.values[0].type == ParameterType.PARAMETER_BOOL
                        and clock.values[0].bool_value)
        require_simulation(identity, domain,
                           [(item.name, item.plugin_name) for item in hardware.component],
                           providers, use_sim_time)
    finally:
        node.destroy_client(hardware_client)
        node.destroy_client(time_client)
    print(f"Simulation interlock passed on isolated domain {domain}", flush=True)


def quaternion_roll_pitch(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    """Return roll and pitch in radians for a normalized quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError("zero-length IMU orientation quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(sin_pitch)


def spin_until(node, predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    raise RuntimeError(f"timed out waiting for {description}")


def call(node, client, request, timeout: float, description: str):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError(f"service unavailable: {description}")
    future = client.call_async(request)
    spin_until(node, future.done, timeout, f"{description} response")
    response = future.result()
    if response is None:
        raise RuntimeError(f"empty response from {description}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-simulation-gantry-release",
        action="store_true",
        help="Assert explicit operator approval for this particular Isaac Sim gantry release attempt",
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--stabilization-seconds", type=float, default=5.0)
    parser.add_argument("--post-release-seconds", type=float, default=5.0)
    parser.add_argument(
        "--freshness-seconds",
        type=float,
        default=1.0,
        help="Maximum wall-clock sensor age; streamed simulation may run below real time",
    )
    parser.add_argument("--max-tilt-radians", type=float, default=0.25)
    parser.add_argument("--max-angular-rate", type=float, default=1.0)
    args = parser.parse_args()

    if not args.allow_simulation_gantry_release:
        parser.error("--allow-simulation-gantry-release is required")
    for name in (
        "startup_timeout",
        "stabilization_seconds",
        "post_release_seconds",
        "freshness_seconds",
        "max_tilt_radians",
        "max_angular_rate",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    try:
        domain = require_local_domain(os.environ)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    rclpy.init()
    node = rclpy.create_node("guarded_gantry_release")
    zero_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    set_params = node.create_client(SetParameters, "/safety_controller/set_parameters")
    get_params = node.create_client(GetParameters, "/safety_controller/get_parameters")
    gantry = node.create_client(SetBool, "/launcher/gantry/set_enabled")

    latest: dict[str, object] = {"imu": None, "imu_time": 0.0, "joint_time": 0.0}

    def on_imu(message: Imu) -> None:
        latest["imu"] = message
        latest["imu_time"] = time.monotonic()

    def on_joint(_message: JointState) -> None:
        latest["joint_time"] = time.monotonic()

    node.create_subscription(Imu, "/imu_sensor_broadcaster/imu", on_imu, 10)
    node.create_subscription(JointState, "/joint_states", on_joint, 10)

    try:
        deadline = time.monotonic() + args.startup_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if set_params.service_is_ready() and get_params.service_is_ready() and gantry.service_is_ready():
                now = time.monotonic()
                if latest["imu"] is not None and now - float(latest["joint_time"]) <= args.freshness_seconds:
                    break
        else:
            raise RuntimeError("controller, gantry service, and fresh sensors did not become ready")

        verify_simulation(node, domain)

        set_request = SetParameters.Request(
            parameters=[
                Parameter(
                    name="blend_ratio",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=1.0,
                    ),
                )
            ]
        )
        set_response = call(
            node, set_params, set_request, 15.0, "safety-controller parameter service"
        )
        if not set_response.results or not set_response.results[0].successful:
            reason = set_response.results[0].reason if set_response.results else "no result"
            raise RuntimeError(f"failed to enable AGILE blend ratio: {reason}")

        get_response = call(
            node,
            get_params,
            GetParameters.Request(names=["blend_ratio"]),
            15.0,
            "safety-controller readback service",
        )
        if not get_response.values or not math.isclose(
            get_response.values[0].double_value, 1.0, abs_tol=1e-6
        ):
            raise RuntimeError("AGILE blend-ratio readback was not 1.0")
        print("AGILE blend_ratio read back as 1.0", flush=True)

        zero = Twist()
        zero_timer = node.create_timer(0.02, lambda: zero_pub.publish(zero))

        def sample_status() -> tuple[bool, str]:
            now = time.monotonic()
            imu = latest["imu"]
            if imu is None:
                return False, "no IMU sample"
            imu_age = now - float(latest["imu_time"])
            joint_age = now - float(latest["joint_time"])
            if imu_age > args.freshness_seconds:
                return False, f"IMU sample stale ({imu_age:.3f}s)"
            if joint_age > args.freshness_seconds:
                return False, f"joint-state sample stale ({joint_age:.3f}s)"
            roll, pitch = quaternion_roll_pitch(
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w,
            )
            angular_rate = math.sqrt(
                imu.angular_velocity.x**2
                + imu.angular_velocity.y**2
                + imu.angular_velocity.z**2
            )
            safe = (
                abs(roll) <= args.max_tilt_radians
                and abs(pitch) <= args.max_tilt_radians
                and angular_rate <= args.max_angular_rate
            )
            return safe, (
                f"roll={roll:.3f}rad pitch={pitch:.3f}rad "
                f"angular_rate={angular_rate:.3f}rad/s "
                f"(limits {args.max_tilt_radians:.3f}rad, "
                f"{args.max_angular_rate:.3f}rad/s)"
            )

        stable_since: float | None = None
        deadline = time.monotonic() + args.startup_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            safe, _status = sample_status()
            if safe:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= args.stabilization_seconds:
                    break
            else:
                stable_since = None
        else:
            raise RuntimeError("robot did not satisfy the continuous pre-release balance window")
        print(
            f"Balance limits held for {args.stabilization_seconds:.1f}s; releasing gantry",
            flush=True,
        )

        verify_simulation(node, domain)
        safe, status = sample_status()
        if not safe:
            raise RuntimeError(f"balance changed during identity check: {status}")
        release = call(
            node,
            gantry,
            SetBool.Request(data=False),
            15.0,
            "Isaac Sim gantry service",
        )
        if not release.success:
            raise RuntimeError(f"gantry release rejected: {release.message}")

        deadline = time.monotonic() + args.post_release_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            safe, status = sample_status()
            if not safe:
                recovery = call(
                    node,
                    gantry,
                    SetBool.Request(data=True),
                    15.0,
                    "emergency gantry re-engagement",
                )
                if not recovery.success:
                    raise RuntimeError("gantry re-engagement rejected; no retry authorized")
                raise RuntimeError(
                    f"post-release balance limit failed ({status}); gantry re-engaged"
                )

        print(
            f"Gantry disabled; unsupported balance held for {args.post_release_seconds:.1f}s",
            flush=True,
        )
        zero_timer.cancel()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
