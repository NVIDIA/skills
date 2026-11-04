#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only host preflight for the Unitree G1 Isaac Sim cloud-control demo.

Usage:
    python3 preflight.py --map MAP_YAML [PATH_OVERRIDES] [--check-*]

Arguments:
    --map is required; installation paths, hosts, ports, and live checks are optional.

Output:
    Writes discovered values and pass/fail observations without changing the host.

Exit codes:
    0 means all selected checks passed; 1 means a blocking check failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import re
import shlex
import shutil
import socket
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-path",
        default=os.environ.get("ISAAC_PATH"),
        help="Isaac Sim root; auto-discovered when omitted",
    )
    parser.add_argument(
        "--isaac-ros-ws",
        default=os.environ.get("ISAAC_ROS_WS"),
        help="Isaac ROS workspace; auto-discovered when omitted",
    )
    parser.add_argument("--map", dest="map_yaml", required=True)
    parser.add_argument("--cloud-host", default="127.0.0.1")
    parser.add_argument(
        "--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT_TCP", "1883"))
    )
    parser.add_argument("--sim-host", default="127.0.0.1")
    parser.add_argument(
        "--viewer-host",
        default=os.environ.get("WEB_VIEWER_HOST"),
        help=(
            "Browser-reachable simulator host; when omitted, use the SSH server "
            "address for remote sessions or --sim-host otherwise"
        ),
    )
    parser.add_argument(
        "--web-viewer-context", default=os.environ.get("WEB_VIEWER_CONTEXT")
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="Require explicit paths instead of searching standard local locations",
    )
    parser.add_argument(
        "--emit-discovered-env",
        action="store_true",
        help="Print discovered POSIX export lines and exit without running host checks",
    )
    parser.add_argument("--webrtc-signal-port", type=int, default=49100)
    parser.add_argument("--viewer-port", type=int, default=8210)
    parser.add_argument("--scene-server-host", default="127.0.0.1")
    parser.add_argument("--scene-server-port", type=int, default=8226)
    parser.add_argument(
        "--check-scene-server",
        action="store_true",
        help="Check the loopback Isaac Sim Python server after app startup",
    )
    parser.add_argument(
        "--check-streaming",
        action="store_true",
        help="Check the browser viewer and WebRTC TCP signaling after startup",
    )
    parser.add_argument(
        "--dispatch-port",
        type=int,
        default=int(os.environ.get("DATABASE_CONTROLLER_PORT", "5003")),
        help=(
            "Mission Database controller port used by Mission Dispatch "
            "(kept under its legacy option name for compatibility)"
        ),
    )
    parser.add_argument(
        "--mission-control-port", type=int, default=int(os.environ.get("MC_PORT", "8050"))
    )
    parser.add_argument("--check-cloud", action="store_true")
    parser.add_argument("--timeout", type=float, default=1.5)
    return parser


def map_image(map_yaml: Path) -> Path | None:
    try:
        text = map_yaml.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"""(?m)^\s*image\s*:\s*['"]?([^'"#\r\n]+)""", text)
    if not match:
        return None
    image = Path(match.group(1).strip()).expanduser()
    return image if image.is_absolute() else map_yaml.parent / image


def first_matching_path(candidates: list[Path], marker: Path) -> Path | None:
    """Return the first candidate containing marker, preserving order."""
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / marker).exists():
            return candidate
    return None


def standard_isaac_sim_roots() -> list[Path]:
    """List bounded, conventional Isaac Sim install locations."""
    roots = [
        Path(value)
        for name in ("ISAAC_PATH", "ISAAC_SIM_PATH", "ISAACSIM_PATH")
        if (value := os.environ.get(name))
    ]
    roots.extend(
        [
            Path("/isaac-sim"),
            Path("/opt/isaac-sim"),
            Path("/opt/isaacsim"),
            Path("/workspace/isaac-sim"),
            Path("/workspaces/isaac-sim"),
            Path.home() / "isaac-sim",
            Path.home() / "isaacsim",
        ]
    )
    for parent in (Path("/opt"), Path.home() / ".local/share/ov/pkg"):
        if parent.is_dir():
            roots.extend(sorted(parent.glob("isaac-sim*"), reverse=True))
            roots.extend(sorted(parent.glob("isaacsim*"), reverse=True))
    return roots


def standard_isaac_ros_workspaces() -> list[Path]:
    """List bounded, conventional Isaac ROS workspace locations."""
    roots = [
        Path(value)
        for name in ("ISAAC_ROS_WS", "ISAAC_ROS_WORKSPACE")
        if (value := os.environ.get(name))
    ]
    roots.extend(
        [
            Path.home() / "isaac_ros-dev",
            Path.home() / "isaac_ros_ws",
            Path("/workspace/isaac_ros-dev"),
            Path("/workspaces/isaac_ros-dev"),
            Path("/opt/isaac_ros-dev"),
        ]
    )
    if Path.home().is_dir():
        roots.extend(sorted(Path.home().glob("isaac_ros*")))
    return roots


def standard_web_viewer_contexts(isaac_path: str | None) -> list[Path]:
    """List official web-viewer locations paired with local Isaac Sim installs."""
    contexts = [
        Path(value)
        for name in ("WEB_VIEWER_CONTEXT", "ISAAC_SIM_WEB_VIEWER_CONTEXT")
        if (value := os.environ.get(name))
    ]
    if isaac_path:
        sim_root = Path(isaac_path).expanduser()
        contexts.append(sim_root / "tools/docker/web-viewer")
        if sim_root.parent.is_dir():
            for source_root in sorted(sim_root.parent.glob("IsaacSim-source-*"), reverse=True):
                contexts.append(source_root / "tools/docker/web-viewer")
    return contexts


def discover_paths(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    """Resolve explicit values first, then known local installation layouts."""
    isaac_path = args.isaac_path
    isaac_ros_ws = args.isaac_ros_ws
    viewer_context = args.web_viewer_context
    if args.no_discover:
        return isaac_path, isaac_ros_ws, viewer_context

    if not isaac_path:
        discovered = first_matching_path(standard_isaac_sim_roots(), Path("isaac-sim.sh"))
        isaac_path = str(discovered) if discovered else None
    if not isaac_ros_ws:
        extension = Path(
            "src/isaac_ros_deploy/isaac_deploy/"
            "isaac_ros_deploy_isaac_sim_extension"
        )
        discovered = first_matching_path(standard_isaac_ros_workspaces(), extension)
        isaac_ros_ws = str(discovered) if discovered else None
    if not viewer_context:
        discovered = first_matching_path(
            standard_web_viewer_contexts(isaac_path), Path("Dockerfile")
        )
        viewer_context = str(discovered) if discovered else None
    return isaac_path, isaac_ros_ws, viewer_context


def discover_viewer_host(args: argparse.Namespace) -> str:
    """Resolve the address a remote user's browser can route to."""
    if args.viewer_host:
        return args.viewer_host
    if args.sim_host not in {"127.0.0.1", "localhost", "::1"}:
        return args.sim_host
    ssh_connection = os.environ.get("SSH_CONNECTION", "").split()
    if len(ssh_connection) == 4:
        return ssh_connection[2]
    return args.sim_host


def url_host(host: str) -> str:
    """Bracket IPv6 literals for use in an HTTP URL."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def find_ldconfig() -> str | None:
    """Find ldconfig even when /sbin is absent from a non-login PATH."""
    if shutil.which("ldconfig"):
        return shutil.which("ldconfig")
    for path in (Path("/sbin/ldconfig"), Path("/usr/sbin/ldconfig")):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def nvenc_library_available() -> bool:
    """Check the linker cache, then standard NVIDIA driver library paths."""
    ldconfig = find_ldconfig()
    if ldconfig:
        libraries = subprocess.run(
            [ldconfig, "-p"], check=False, capture_output=True, text=True
        )
        if "libnvidia-encode.so" in libraries.stdout:
            return True
    return any(
        directory.is_dir() and any(directory.glob("libnvidia-encode.so*"))
        for directory in (
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib64"),
            Path("/usr/lib/wsl/lib"),
        )
    )


def command_output(command: list[str]) -> tuple[bool, str]:
    """Run a short probe and return a compact diagnostic on failure."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, result.stdout.strip()
    detail = (result.stderr or result.stdout).strip().replace("\n", " ")
    return False, detail[:400] or f"exit status {result.returncode}"


def main() -> int:
    args = build_parser().parse_args()
    args.isaac_path, args.isaac_ros_ws, args.web_viewer_context = discover_paths(args)
    viewer_host = discover_viewer_host(args)
    if args.emit_discovered_env:
        unresolved = []
        for name, value in (
            ("ISAAC_PATH", args.isaac_path),
            ("ISAAC_ROS_WS", args.isaac_ros_ws),
            ("WEB_VIEWER_CONTEXT", args.web_viewer_context),
        ):
            if value:
                print(f"export {name}={shlex.quote(str(Path(value).resolve()))}")
            else:
                unresolved.append(name)
        if unresolved:
            print(
                "# Could not discover: " + ", ".join(unresolved),
                file=sys.stderr,
            )
            return 1
        print(f"export WEB_VIEWER_HOST={shlex.quote(viewer_host)}")
        return 0
    failures: list[str] = []
    passed: list[str] = []

    def require(ok: bool, message: str) -> None:
        (passed if ok else failures).append(("OK: " if ok else "FAIL: ") + message)

    require(
        platform.machine() in {"x86_64", "amd64"},
        f"x86_64 host detected ({platform.machine()})",
    )
    nvidia_smi = shutil.which("nvidia-smi")
    require(nvidia_smi is not None, "nvidia-smi is available")
    if nvidia_smi:
        gpu_ready, gpu_detail = command_output(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        require(
            gpu_ready,
            "NVIDIA driver exposes a GPU"
            + (f" ({gpu_detail.splitlines()[0]})" if gpu_ready and gpu_detail else "")
            + (f": {gpu_detail}" if not gpu_ready else ""),
        )
    require(
        shutil.which("isaac-ros") is not None,
        "isaac-ros CLI is available; if missing, follow "
        "references/isaac-ros-setup.md to install it and configure Docker mode, "
        "then rerun preflight",
    )
    require(
        importlib.util.find_spec("PIL") is not None,
        "Pillow is available for occupancy-aware scene suggestions",
    )
    docker = shutil.which("docker")
    require(docker is not None, "docker CLI is available")
    if docker is not None:
        daemon_ready, daemon_detail = command_output(
            [docker, "info", "--format", "{{.ServerVersion}}"]
        )
        require(
            daemon_ready,
            "Docker daemon is reachable"
            + (f" (server {daemon_detail})" if daemon_ready and daemon_detail else "")
            + (f": {daemon_detail}" if not daemon_ready else ""),
        )
        compose = subprocess.run(
            [docker, "compose", "version"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(compose.returncode == 0, "docker compose is available for the web viewer")
    require(nvenc_library_available(), "NVENC runtime library is available")

    if not args.web_viewer_context:
        failures.append(
            "FAIL: WEB_VIEWER_CONTEXT was not discovered; install Isaac Sim with "
            "tools/docker/web-viewer, set WEB_VIEWER_CONTEXT, or pass "
            "--web-viewer-context"
        )
    else:
        viewer_context = Path(args.web_viewer_context).expanduser().resolve()
        require(
            (viewer_context / "Dockerfile").is_file(),
            f"official web-viewer Dockerfile exists at {viewer_context}",
        )

    if not args.isaac_path:
        failures.append(
            "FAIL: ISAAC_PATH was not discovered; install Isaac Sim 6.0 in a "
            "standard location, set ISAAC_PATH, or pass --isaac-path"
        )
    else:
        isaac_path = Path(args.isaac_path).expanduser().resolve()
        require(
            (isaac_path / "isaac-sim.sh").is_file(),
            f"Isaac Sim launcher exists at {isaac_path / 'isaac-sim.sh'}",
        )
        require(
            (isaac_path / "isaac-sim.streaming.sh").is_file(),
            f"Isaac Sim streaming launcher exists at {isaac_path / 'isaac-sim.streaming.sh'}",
        )
        require(
            (isaac_path / "exts/isaacsim.ros2.core/jazzy/lib").is_dir(),
            "Isaac Sim Jazzy ROS 2 bridge libraries exist",
        )
        require(
            (
                isaac_path
                / "exts/isaacsim.code_editor.python_server/config/extension.toml"
            ).is_file(),
            "Isaac Sim remote Python server extension exists",
        )

    if not args.isaac_ros_ws:
        failures.append(
            "FAIL: ISAAC_ROS_WS with the Deploy extension was not discovered; "
            "set ISAAC_ROS_WS or pass --isaac-ros-ws"
        )
    else:
        workspace = Path(args.isaac_ros_ws).expanduser().resolve()
        extension = (
            workspace
            / "src/isaac_ros_deploy/isaac_deploy/"
            "isaac_ros_deploy_isaac_sim_extension"
        )
        require(workspace.is_dir(), f"Isaac ROS workspace exists at {workspace}")
        require(
            extension.is_dir(),
            f"Isaac ROS Deploy extension source exists at {extension}",
        )

    map_yaml = Path(args.map_yaml).expanduser().resolve()
    require(map_yaml.is_file(), f"Nav2 map YAML exists at {map_yaml}")
    image = map_image(map_yaml) if map_yaml.is_file() else None
    require(image is not None, "Nav2 map YAML has an image field")
    if image is not None:
        require(image.is_file(), f"Nav2 map image exists at {image.resolve()}")

    if args.check_cloud:
        for port, service in (
            (args.mqtt_port, "MQTT"),
            (args.dispatch_port, "Mission Database controller (Mission Dispatch dependency)"),
            (args.mission_control_port, "Mission Control"),
        ):
            try:
                with socket.create_connection(
                    (args.cloud_host, port), timeout=args.timeout
                ):
                    passed.append(
                        f"OK: {service} is reachable at {args.cloud_host}:{port}"
                    )
            except OSError as exc:
                failures.append(
                    f"FAIL: {service} is not reachable at "
                    f"{args.cloud_host}:{port}: {exc}"
                )
    else:
        passed.append(
            "NOTE: cloud connectivity was not checked; "
            "add --check-cloud after stack startup"
        )

    if args.check_streaming:
        for port, service in (
            (args.webrtc_signal_port, "WebRTC signaling"),
            (args.viewer_port, "browser viewer"),
        ):
            try:
                with socket.create_connection(
                    (args.sim_host, port), timeout=args.timeout
                ):
                    passed.append(
                        f"OK: {service} is reachable at {args.sim_host}:{port}"
                    )
            except OSError as exc:
                failures.append(
                    f"FAIL: {service} is not reachable at "
                    f"{args.sim_host}:{port}: {exc}"
                )
        passed.append(
            "NOTE: WebRTC media uses UDP 47998; verify it through the live browser view"
        )
        passed.append(
            f"LIVE VIEWER URL: http://{url_host(viewer_host)}:{args.viewer_port}/"
        )
    else:
        passed.append(
            "NOTE: streaming was not checked; add --check-streaming after "
            "the headless app and browser viewer start"
        )

    if args.check_scene_server:
        if args.scene_server_host not in {"127.0.0.1", "localhost", "::1"}:
            failures.append(
                "FAIL: scene server must remain loopback-only; use 127.0.0.1, "
                "localhost, or ::1"
            )
        else:
            try:
                with socket.create_connection(
                    (args.scene_server_host, args.scene_server_port),
                    timeout=args.timeout,
                ):
                    passed.append(
                        "OK: Isaac Sim Python scene server is reachable on "
                        f"loopback at {args.scene_server_host}:{args.scene_server_port}"
                    )
            except OSError as exc:
                failures.append(
                    "FAIL: Isaac Sim Python scene server is not reachable at "
                    f"{args.scene_server_host}:{args.scene_server_port}: {exc}"
                )
    else:
        passed.append(
            "NOTE: live scene inventory was not checked; add "
            "--check-scene-server after Isaac Sim reports app ready"
        )

    for line in passed + failures:
        print(line)
    if failures:
        print(
            f"Preflight failed with {len(failures)} blocking issue(s).",
            file=sys.stderr,
        )
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
