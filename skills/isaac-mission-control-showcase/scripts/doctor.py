#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for the Mission Control showcase: dependencies, then host.

Two phases, run in order, with different failure meanings.

  dependencies  Resolve the upstream skills this showcase orchestrates. They
                are owned by their own repositories and never copied into this
                package. This finds where they actually live, verifies the
                pinned commit and the entrypoints the runner needs, and writes
                the manifest the runner and the adapters consume. Exits 3 when
                something blocks, because nothing downstream can proceed.

  host          Read-only checks of this machine: Isaac Sim, GPU and driver,
                memory, Docker, ports, images, and the map assets. Changes no
                containers, images, files, or ports. Exits 1 on failure.

The dependency phase must succeed first: the host phase is handed the resolved
upstream entrypoints and cannot check what has not been resolved.

Resolution precedence, highest first:

  1. per-skill directory override      e.g. CHANGE_MAP_SKILL_DIR
  2. per-upstream checkout root        e.g. MISSION_CONTROL_UPSTREAM_ROOT
  3. normalized checkout root          $MISSION_CONTROL_SHOWCASE_UPSTREAM_ROOT
                                       or ~/.mission-control-showcase/upstreams

Nothing else is searched. There is no scan of developer directories, no
fallback to a vendored copy, and no silent substitution of a different copy of
a skill. A path that does not resolve becomes a blocker naming the skill, its
upstream, the entrypoint, and the accepted override.

Resolution and verification are read-only. `--prepare` is the only mode that
may create or update a managed checkout, and it refuses to touch a directory
supplied through an override.
"""

import argparse
import json
import os
import re
import shlex
import subprocess  # nosec B404
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
DEPENDENCIES = SKILL_DIR / "upstream-versions.lock.json"
sys.path.insert(0, str(SKILL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_state  # noqa: E402

HOME_ENV = "MISSION_CONTROL_SHOWCASE_HOME"
ROOT_ENV = "MISSION_CONTROL_SHOWCASE_UPSTREAM_ROOT"
MANIFEST_ENV = "MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST"
MANIFEST_SCHEMA = "mission-control-showcase/deps-manifest@1"
MANIFEST_NAME = "showcase-deps.json"

EXIT_BLOCKED = 3
EXIT_CONFIG = 2


def atomic_write(path: Path, text: str) -> None:
    """Write via a temporary sibling plus os.replace so no reader sees a
    partial manifest, and a crash leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def showcase_home() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".mission-control-showcase")


def default_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or showcase_home() / "upstreams")


def default_manifest_path() -> Path:
    explicit = os.environ.get(MANIFEST_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return showcase_home() / "state" / MANIFEST_NAME


def _git(args, cwd):
    return subprocess.run(  # nosec B603
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def head_commit(path: Path):
    try:
        return _git(["rev-parse", "HEAD"], path)
    except (subprocess.CalledProcessError, OSError):
        return None


def worktree_dirty(path: Path):
    """True when tracked files differ from HEAD, so runtime code != the commit."""
    try:
        return bool(_git(["status", "--porcelain", "--untracked-files=no"], path))
    except (subprocess.CalledProcessError, OSError):
        return None


def resolve_upstream_root(name, spec, root):
    """Return (path, source) for an upstream checkout without searching."""
    override = os.environ.get(spec["root_env"])
    if override:
        return Path(override).expanduser(), f"env:{spec['root_env']}"
    return root / spec["checkout"], "normalized-root"


def resolve(dependencies, root, *, strict_commit=True):
    """Resolve every dependency. Read-only; never mutates anything."""
    upstreams, skills, blockers = {}, {}, []

    for name, spec in dependencies["upstreams"].items():
        path, source = resolve_upstream_root(name, spec, root)
        commit = head_commit(path) if path.is_dir() else None
        dirty = worktree_dirty(path) if path.is_dir() else None
        entry = {
            "url": spec["url"],
            "expected_ref": spec["ref"],
            "expected_commit": spec["commit"],
            "root": str(path),
            "source": source,
            "present": path.is_dir(),
            "commit": commit,
            "dirty": dirty,
        }
        if not path.is_dir():
            entry["status"] = "missing"
        elif commit is None:
            entry["status"] = "not-a-git-checkout"
        elif strict_commit and commit != spec["commit"]:
            entry["status"] = "commit-mismatch"
        elif dirty and strict_commit:
            entry["status"] = "dirty-worktree"
        else:
            entry["status"] = "ready"
            entry["pinned"] = bool(
                commit == spec["commit"] and not dirty
            )
        upstreams[name] = entry

    for name, spec in dependencies["skills"].items():
        up = spec["upstream"]
        up_entry = upstreams[up]
        up_spec = dependencies["upstreams"][up]
        override = os.environ.get(spec["path_env"])
        if override:
            skill_dir = Path(override).expanduser()
            source = f"env:{spec['path_env']}"
        else:
            skill_dir = Path(up_entry["root"]) / up_spec["skills_path"] / name
            source = up_entry["source"]

        entrypoints, missing = {}, []
        for key, rel in spec["entrypoints"].items():
            target = skill_dir / rel
            entrypoints[key] = str(target)
            if not target.is_file():
                missing.append(rel)

        resources, missing_resources = {}, []
        for rel in spec.get("resources", []):
            target = skill_dir / rel
            resources[rel] = str(target)
            if not target.exists():
                missing_resources.append(rel)

        has_skill_md = (skill_dir / "SKILL.md").is_file()
        entry = {
            "upstream": up,
            "skill_dir": str(skill_dir),
            "source": source,
            "present": skill_dir.is_dir(),
            "skill_md": str(skill_dir / "SKILL.md") if has_skill_md else None,
            "entrypoints": entrypoints,
            "resources": resources,
            "adapter": spec.get("adapter"),
            "runner_env": spec.get("runner_env", {}),
            "agent_only": bool(spec.get("agent_only")),
        }
        # A per-skill override is trusted only as far as it can be verified. It
        # is never assumed to sit at the pinned commit.
        if override:
            override_commit = head_commit(skill_dir)
            override_dirty = worktree_dirty(skill_dir)
            pinned = bool(
                override_commit == up_spec["commit"] and not override_dirty
            )
            entry["commit"] = override_commit
            entry["dirty"] = override_dirty
            entry["pinned"] = pinned
            if not pinned:
                entry["unpinned_reason"] = (
                    "per-skill override is not at the pinned commit"
                    if override_commit != up_spec["commit"]
                    else "per-skill override has local modifications"
                )
        else:
            entry["pinned"] = bool(up_entry.get("pinned"))
            if not entry["pinned"]:
                entry["unpinned_reason"] = f"upstream {up} is {up_entry['status']}"

        if not skill_dir.is_dir():
            entry["status"] = "missing"
            blockers.append(
                f"{name}: not found at {skill_dir}. It is owned by {up_spec['url']} "
                f"at ref {up_spec['ref']}. Set {spec['path_env']}=<skill dir>, or "
                f"set {up_spec['root_env']}=<checkout>, or run: "
                f"doctor.py dependencies --prepare"
            )
        else:
            problems = []
            if missing:
                problems.append(
                    f"{name}: missing entrypoint(s) {', '.join(missing)} under "
                    f"{skill_dir}. Expected from {up_spec['url']} at "
                    f"{up_spec['ref']}. Override with {spec['path_env']}=<skill dir> "
                    f"or re-run --prepare"
                )
            if missing_resources:
                problems.append(
                    f"{name}: missing declared resource(s) "
                    f"{', '.join(missing_resources)} under {skill_dir}. "
                    f"Expected from {up_spec['url']} at {up_spec['ref']}."
                )
            if not has_skill_md:
                problems.append(
                    f"{name}: {skill_dir} has no SKILL.md, so its own contract "
                    f"cannot be read. This does not look like the upstream skill. "
                    f"Point {spec['path_env']} at the real skill directory."
                )
            if problems:
                entry["status"] = "incomplete"
                blockers.extend(problems)
            elif override is None and up_entry["status"] != "ready":
                entry["status"] = "upstream-" + up_entry["status"]
                if up_entry["status"] == "dirty-worktree":
                    blockers.append(
                        f"{name}: upstream {up} at {up_entry['root']} has local "
                        f"modifications, so runtime code differs from "
                        f"{up_spec['commit'][:12]}. Commit, stash, or re-run "
                        f"--prepare; or accept it with --allow-commit-drift."
                    )
                elif up_entry["status"] == "commit-mismatch":
                    blockers.append(
                        f"{name}: upstream {up} requires {up_spec['ref']} "
                        f"({up_spec['commit'][:12]}), found "
                        f"{(up_entry['commit'] or '?')[:12]} at {up_entry['root']}"
                    )
                else:
                    blockers.append(
                        f"{name}: upstream {up} is {up_entry['status']} at "
                        f"{up_entry['root']}, so its commit cannot be checked "
                        f"against {up_spec['commit'][:12]}. Expected "
                        f"{up_spec['url']} at {up_spec['ref']}. Set "
                        f"{spec['path_env']}=<skill dir>, set "
                        f"{up_spec['root_env']}=<checkout>, or run: "
                        f"{Path(__file__).name} dependencies --prepare"
                    )
            else:
                entry["status"] = "ready"
        skills[name] = entry

    return upstreams, skills, blockers


def build_manifest(dependencies, root, upstreams, skills, blockers, drift_allowed=False):
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream_root": str(root),
        "ready": not blockers,
        "pinned": all(u.get("pinned") for u in upstreams.values()),
        "drift_allowed": drift_allowed,
        "blockers": blockers,
        "upstreams": upstreams,
        "skills": skills,
    }


def env_lines(manifest):
    """Shell assignments the runner and references source. Paths only."""
    lines = [
        f"export {MANIFEST_ENV}="
        f"{shlex.quote(str(manifest.get('manifest_path', '')))}",
        "export MISSION_CONTROL_SHOWCASE_REQUIRE_PREFLIGHT=1",
        f"export {ROOT_ENV}={shlex.quote(manifest['upstream_root'])}",
    ]
    for name, entry in sorted(manifest["upstreams"].items()):
        var = name.upper() + "_UPSTREAM_ROOT"
        lines.append(f"export {var}={shlex.quote(entry['root'])}")
    for name, entry in sorted(manifest["skills"].items()):
        var = name.upper().replace("-", "_") + "_SKILL_DIR"
        lines.append(f"export {var}={shlex.quote(entry['skill_dir'])}")
        for key, path in sorted(entry["entrypoints"].items()):
            runner = (entry.get("runner_env") or {}).get(key)
            if runner:
                lines.append(f"export {runner}={shlex.quote(path)}")
    return "\n".join(lines) + "\n"


def prepare(dependencies, root, upstreams, only=None):
    """Create or update managed checkouts. Never touches an overridden path."""
    changed = []
    for name, spec in dependencies["upstreams"].items():
        if only and name not in only:
            continue
        entry = upstreams[name]
        if entry["source"].startswith("env:"):
            print(f"  skip {name}: supplied through {entry['source']}, not managed here")
            continue
        path = Path(entry["root"])
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").is_dir():
            print(f"  init {name} at {path}")
            _git(["init", "--quiet", "."], path)
            _git(["remote", "add", "origin", spec["url"]], path)
        try:
            _git(["fetch", "--quiet", "--depth", "1", "origin", spec["commit"]], path)
        except subprocess.CalledProcessError:
            _git(["fetch", "--quiet", "origin", spec["ref"]], path)
        _git(["checkout", "--quiet", spec["commit"]], path)
        print(f"  ready {name} at {path} ({spec['commit'][:12]})")
        changed.append(name)
    return changed




MEMINFO_PATH = Path("/proc/meminfo")
MIN_DRIVER_MAJOR = 590
MIN_SYSTEM_MEMORY_BYTES = 64_000_000_000
MIN_AVAILABLE_MEMORY_BYTES = 48_000_000_000


def run_cmd(argv, timeout_s=15):
    """Run a command without shell expansion."""
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


class Doctor:
    """Emit redacted findings and track severity."""

    def __init__(self, strict_security=False):
        """Initialize failure/warning counters and strictness."""
        self.strict_security = strict_security
        self.failures = 0
        self.warnings = 0

    def _emit(self, level, message):
        """Print one aligned status line."""
        print(f"{level:<5} {message}")

    def ok(self, message):
        """Record a successful check."""
        self._emit("OK", message)

    def info(self, message):
        """Print informational context without changing status."""
        self._emit("INFO", message)

    def warn(self, message):
        """Record a warning check result."""
        self.warnings += 1
        self._emit("WARN", message)

    def fail(self, message):
        """Record a failing check result."""
        self.failures += 1
        self._emit("FAIL", message)

    def security(self, message):
        """Record a security finding according to strictness."""
        if self.strict_security:
            self.fail(message)
        else:
            self.warn(message)


def check_command(report, argv, label):
    """Require a command to succeed."""
    rc, stdout, stderr = run_cmd(argv)
    if rc == 0:
        report.ok(label)
        return True
    detail = (stderr or stdout).strip().splitlines()
    report.fail(f"{label}: {detail[-1] if detail else 'command failed'}")
    return False


def check_packaged_files(report, args):
    """Require every cross-skill entrypoint and bundled asset."""
    skills = Path(args.skills_dir)
    assets = Path(args.asset_dir)
    required = (
        Path(args.bring_up_cloud),
        Path(args.change_map),
        Path(args.change_fleet),
        Path(args.isaac_send),
        assets / "carter_warehouse_navigation.png",
        assets / "carter_warehouse_navigation.yaml",
    )
    for path in required:
        if path.is_file():
            try:
                display = path.relative_to(skills.parent)
            except ValueError:
                display = path
            report.ok(f"packaged dependency: {display}")
        else:
            report.fail(f"missing packaged dependency: {path}")


def check_clean_docker(report, ignored_containers):
    """Require a reachable daemon with no non-ignored running containers."""
    if not check_command(report, ["docker", "info"], "Docker daemon reachable"):
        return
    check_command(
        report,
        ["docker", "compose", "version"],
        "Docker Compose available",
    )
    rc, stdout, _ = run_cmd(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"]
    )
    running = [line for line in stdout.splitlines() if line.strip()]
    ignored_names = set(ignored_containers)
    ignored = []
    blocking = []
    for line in running:
        name = line.split("\t", 1)[0]
        if name in ignored_names:
            ignored.append(line)
        else:
            blocking.append(line)

    for line in ignored:
        report.info(f"ignored running container: {line}")
    if blocking:
        report.fail(
            "non-ignored Docker containers must be stopped before startup"
        )
        for line in blocking:
            report.info(f"running: {line}")
    else:
        report.ok("no non-ignored Docker containers are running")

    rc, stdout, _ = run_cmd(
        ["docker", "info", "--format", "{{json .Runtimes}}"]
    )
    if rc == 0 and "nvidia" in stdout:
        report.ok("NVIDIA container runtime registered")
    else:
        report.fail("NVIDIA container runtime is not registered")


def check_gpu(report):
    """Require the driver line validated with the canonical image set."""
    rc, stdout, _ = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if rc != 0 or not stdout.strip():
        report.fail("nvidia-smi did not report an available GPU")
        return
    for line in stdout.splitlines():
        report.ok(f"GPU: {line.strip()}")
        fields = [field.strip() for field in line.split(",")]
        match = re.match(r"(\d+)", fields[1] if len(fields) > 1 else "")
        if not match:
            report.fail(f"could not parse NVIDIA driver version from: {line}")
        elif int(match.group(1)) < MIN_DRIVER_MAJOR:
            report.fail(
                f"NVIDIA driver {fields[1]} is too old; the canonical "
                f"showcase requires R{MIN_DRIVER_MAJOR} or newer"
            )
        else:
            report.ok(
                f"NVIDIA driver satisfies the R{MIN_DRIVER_MAJOR}+ requirement"
            )


def check_memory(report):
    """Require enough effective memory; treat swap as a fallback, not a floor."""
    try:
        values = {}
        with MEMINFO_PATH.open(encoding="utf-8") as stream:
            for line in stream:
                name, _, raw = line.partition(":")
                if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    values[name] = int(raw.split()[0]) * 1024
    except (OSError, ValueError, IndexError) as exc:
        report.fail(f"could not inspect host memory: {exc}")
        return

    missing = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"} - values.keys()
    if missing:
        report.fail("host memory report is missing: " + ", ".join(sorted(missing)))
        return

    def gb(value):
        return f"{value / 1_000_000_000:.1f} GB"

    report.info(
        "host memory: "
        f"total={gb(values['MemTotal'])} "
        f"available={gb(values['MemAvailable'])} "
        f"swap_total={gb(values['SwapTotal'])} "
        f"swap_free={gb(values['SwapFree'])}"
    )

    if values["MemTotal"] < MIN_SYSTEM_MEMORY_BYTES:
        report.fail(
            f"system memory is {gb(values['MemTotal'])}; "
            f"the showcase requires {gb(MIN_SYSTEM_MEMORY_BYTES)}"
        )

    verdict = run_state.memory_policy(
        values["MemTotal"],
        values["MemAvailable"],
        values["SwapFree"],
        MIN_AVAILABLE_MEMORY_BYTES,
        MIN_SYSTEM_MEMORY_BYTES,
    )
    message = (
        f"effective memory {gb(verdict['effective_bytes'])} "
        f"(available {gb(values['MemAvailable'])} + free swap "
        f"{gb(values['SwapFree'])}); "
        f"requirement {gb(MIN_AVAILABLE_MEMORY_BYTES)}. {verdict['detail']}"
    )
    if verdict["status"] == "ok":
        report.ok(message)
    elif verdict["status"] == "warn":
        report.warn(message)
    else:
        report.fail(message)


def check_ros_bridge_executables(report, isaac_sim_dir):
    """Require the ROS 2 bridge helpers Isaac invokes to be executable.

    An archive that loses a Unix executable bit installs a helper as 0664. Isaac
    starts, then ROS startup dies with PermissionError deep in a launch. Catch it
    here, name the exact path, and recommend the narrowest repair.
    """
    root = Path(isaac_sim_dir) / "exts" / "isaacsim.ros2.core"
    if not root.is_dir():
        report.info(f"no ROS 2 bridge extension at {root}; skipping executable checks")
        return

    # Shared libraries in the same directory are correctly non-executable; only
    # helpers Isaac invokes as programs need the bit. Select by file type rather
    # than by a list of names, which would go stale as helpers are added.
    def is_library(path):
        return ".so" in path.suffixes or ".so" in path.name.split(".", 1)[-1]

    candidates = sorted(
        p for p in (root / "bin").glob("*") if p.is_file() and not is_library(p)
    )
    if not candidates:
        report.warn(f"ROS 2 bridge has no invocable helpers under {root / 'bin'}")
        return

    bad = [p for p in candidates if not os.access(p, os.X_OK)]
    for path in bad:
        mode = oct(path.stat().st_mode & 0o777)
        report.fail(
            f"ROS 2 bridge helper is not executable: {path} (mode {mode}). "
            f"Isaac Sim will fail at ROS startup with PermissionError. "
            f"Repair exactly this file with: chmod +x {path}"
        )
    if not bad:
        report.ok(
            f"{len(candidates)} ROS 2 bridge helper(s) under {root / 'bin'} are executable"
        )


def check_asset_uri(report, uri):
    """Require the selected canonical USD URI to be remotely reachable."""
    if not uri.startswith(("https://", "http://")):
        report.fail("warehouse USD must be an explicit HTTP(S) URI")
        return
    request = urllib.request.Request(uri, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # nosec B310
            status = response.status
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 405}:
            report.fail(f"warehouse USD URI returned HTTP {exc.code}")
            return
        request = urllib.request.Request(
            uri, headers={"Range": "bytes=0-0"}, method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # nosec B310
                status = response.status
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as retry:
            report.fail(f"warehouse USD URI is unreachable: {retry}")
            return
    except (OSError, urllib.error.URLError) as exc:
        report.fail(f"warehouse USD URI is unreachable: {exc}")
        return
    if 200 <= status < 400:
        report.ok(f"warehouse USD URI reachable: {uri}")
    else:
        report.fail(f"warehouse USD URI returned HTTP {status}")


def configured_ports(args):
    """Return protocol, port, and label tuples that must start unused."""
    ports = [
        ("tcp", 1883, "MQTT"),
        ("tcp", 9001, "MQTT websocket"),
        ("tcp", 5002, "Mission Database API"),
        ("tcp", 5003, "Mission Database controller"),
        ("tcp", 5432, "Postgres"),
        ("tcp", 5050, "cuOpt"),
        ("tcp", 8050, "Mission Control"),
        ("tcp", args.isaac_python_port, "Isaac Python server"),
    ]
    return ports


def check_ports(report, args):
    """Require selected ports to be unused and identify conflicting listeners."""
    tcp_rc, tcp_stdout, _ = run_cmd(["ss", "-H", "-ltnp"])
    udp_rc, udp_stdout, _ = run_cmd(["ss", "-H", "-lunp"])
    if tcp_rc != 0 or udp_rc != 0:
        report.fail("could not inspect host TCP/UDP listeners with ss")
        return
    listeners = {
        "tcp": tcp_stdout.splitlines(),
        "udp": udp_stdout.splitlines(),
    }
    for protocol, port, label in configured_ports(args):
        pattern = re.compile(rf"(?<!\d):{port}(?!\d)")
        conflicts = [line for line in listeners[protocol] if pattern.search(line)]
        if conflicts:
            report.fail(
                f"{label} {protocol.upper()} port {port} is already in use"
            )
            for line in conflicts:
                report.info(f"listener: {line.strip()}")
        else:
            report.ok(f"{label} {protocol.upper()} port {port} is available")


def check_isaac_install(report, args):
    """Require a usable local Isaac Sim installation for the GUI launch."""
    root = Path(args.isaac_sim_dir)
    launcher = root / "isaac-sim.sh"
    if not root.is_dir():
        report.fail(f"ISAAC_SIM_DIR is not a directory: {root}")
        return
    report.ok(f"Isaac Sim installation: {root}")
    if launcher.is_file() and os.access(launcher, os.X_OK):
        report.ok(f"Isaac Sim launcher is executable: {launcher.name}")
    else:
        report.fail(f"missing or non-executable Isaac Sim launcher: {launcher}")

    ros_lib = root / "exts" / "isaacsim.ros2.core" / "humble" / "lib"
    if ros_lib.is_dir():
        report.ok("bundled ROS 2 Humble bridge libraries are present")
    else:
        report.fail(f"ROS 2 bridge libraries are missing: {ros_lib}")

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        report.ok(f"display available for the Isaac Sim window: DISPLAY={os.environ.get('DISPLAY', '')}")
    else:
        report.fail(
            "no DISPLAY or WAYLAND_DISPLAY is set; the Isaac Sim GUI window "
            "cannot open. Use a local session or an X/VNC display."
        )


# Cloud image variables the runner dereferences in required_runtime_images().
# Keep in sync with run.sh; an unset value here is what makes prefetch fail.
CLOUD_IMAGE_VARS = (
    "MOSQUITTO_IMAGE",
    "POSTGRES_IMAGE",
    "MISSION_DATABASE_IMAGE",
    "MISSION_DISPATCH_IMAGE",
    "WPG_IMAGE",
    "CUOPT_IMAGE",
    "MISSION_CONTROL_IMAGE",
)


def resolve_cloud_file(bring_up_cloud, relative):
    """Resolve a bring-up-cloud-stack resource the way up.sh and run.sh do.

    Workspace override first, then the shipped resources/ next to up.sh.
    Returns None when neither exists."""
    workspace = os.environ.get("MC_WORKSPACE") or str(Path.cwd() / "skills-workspace")
    candidates = (
        Path(workspace) / relative,
        Path(bring_up_cloud).resolve().parent.parent / "resources" / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_env_file(path):
    """Read KEY=VALUE lines from a shell-sourceable file.

    Deliberately not a shell parser: these files are flat assignments, and
    doctor must never execute what it inspects."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.isidentifier():
            values[key] = value.strip().strip('"').strip("'")
    return values


def resolve_cloud_images(args):
    """Return {var: (value, source)} for the cloud image variables.

    Environment wins, matching what the runner will actually dereference.
    Falling back to a file means the caller has not sourced it, which is the
    condition that makes prefetch fail with an unbound variable."""
    resolved = {}
    file_values = {}
    sources = {}
    for relative in ("docker-compose/.env", "docker-compose/images.conf"):
        path = resolve_cloud_file(args.bring_up_cloud, relative)
        if path is None:
            continue
        # images.conf is read last so it wins, matching up.sh's source order.
        for key, value in parse_env_file(path).items():
            file_values[key] = value
            sources[key] = path
    for name in CLOUD_IMAGE_VARS:
        env_value = os.environ.get(name)
        if env_value:
            resolved[name] = (env_value, "environment")
        elif file_values.get(name):
            resolved[name] = (file_values[name], str(sources[name]))
        else:
            resolved[name] = (None, None)
    return resolved


def check_cloud_images(report, args):
    """Verify every cloud image reference the runner will dereference.

    Without this the preflight can pass while run.sh dies in
    prefetch_runtime_images: required_runtime_images() runs inside a process
    substitution, so set -u kills only that subshell and the runner continues
    with an empty image list before failing on `docker pull ""`."""
    for relative in ("docker-compose/.env", "docker-compose/images.conf"):
        path = resolve_cloud_file(args.bring_up_cloud, relative)
        if path is None:
            report.fail(f"cloud config not found: {relative}")
        else:
            report.ok(f"cloud config: {path}")

    resolved = resolve_cloud_images(args)
    missing = [name for name, (value, _) in resolved.items() if not value]
    if missing:
        report.fail(
            "cloud image variables unresolved: " + ", ".join(missing)
        )
        report.info(
            "these come from bring-up-cloud-stack resources/docker-compose/"
            "images.conf; the runner dereferences them in "
            "required_runtime_images()"
        )
        return

    from_file = sorted(
        name for name, (_, source) in resolved.items() if source != "environment"
    )
    if from_file:
        report.fail(
            "cloud image variables resolved from file, not the environment: "
            + ", ".join(from_file)
        )
        report.info(
            "run.sh sources images.conf before invoking preflight, so values "
            "reaching doctor only from disk mean the runner will dereference "
            "them unset in required_runtime_images()"
        )
    else:
        report.ok(
            f"all {len(CLOUD_IMAGE_VARS)} cloud image variables resolved "
            "from the environment"
        )
    for name in CLOUD_IMAGE_VARS:
        report.info(f"  {name}={resolved[name][0]}")


def check_images(report, args):
    """Report local image state without pulling or starting anything."""
    images = [args.nova_carter_image]
    images.extend(
        value
        for value, _ in resolve_cloud_images(args).values()
        if value
    )
    seen = set()
    for image in images:
        if image in seen:
            continue
        seen.add(image)
        rc, stdout, _ = run_cmd(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image]
        )
        if rc == 0:
            report.ok(f"image local: {image} ({stdout.strip()[:27]})")
        else:
            report.info(f"image is not local yet: {image}")


def validate_inputs(report, args):
    """Validate identity and the LKG/custom version contract."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.robot_name):
        report.ok(f"robot identity valid: {args.robot_name}")
    else:
        report.fail(f"invalid robot identity: {args.robot_name!r}")
    for name in args.ignore_containers:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            report.fail(f"invalid ignored container name: {name!r}")
    report.info(
        "a local Isaac Sim installation supplies the runtime; its version and "
        "the warehouse asset URI are the caller's contract"
    )


def doctor(args):
    """Run the read-only preflight."""
    report = Doctor()
    print("Mission Control showcase preflight")
    print("Read-only: no containers, images, files, or ports are changed")
    print("")
    validate_inputs(report, args)
    check_packaged_files(report, args)
    check_isaac_install(report, args)
    check_ros_bridge_executables(report, args.isaac_sim_dir)
    check_clean_docker(report, args.ignore_containers)
    check_gpu(report)
    check_memory(report)
    check_ports(report, args)
    check_asset_uri(report, args.warehouse_usd_uri)
    check_cloud_images(report, args)
    check_images(report, args)
    print("")
    if report.failures:
        print(
            f"Preflight result: FAIL "
            f"({report.failures} failures, {report.warnings} warnings)"
        )
        return 1
    print(f"Preflight result: OK ({report.warnings} warnings)")
    return 0


def run_dependencies(args):
    """Resolve the upstream skills and write the manifest the runner consumes."""
    if not DEPENDENCIES.is_file():
        print(f"error: {DEPENDENCIES} not found", file=sys.stderr)
        return EXIT_CONFIG
    dependencies = json.loads(DEPENDENCIES.read_text())
    root = Path(args.upstream_root).expanduser() if args.upstream_root else default_root()

    upstreams, skills, blockers = resolve(
        dependencies, root, strict_commit=not args.allow_commit_drift
    )

    if args.prepare:
        if args.check_only:
            print("error: --prepare and --check-only are mutually exclusive", file=sys.stderr)
            return EXIT_CONFIG
        print(f"Preparing managed checkouts under {root}")
        prepare(dependencies, root, upstreams)
        upstreams, skills, blockers = resolve(
            dependencies, root, strict_commit=not args.allow_commit_drift
        )

    manifest = build_manifest(
        dependencies, root, upstreams, skills, blockers, args.allow_commit_drift
    )

    manifest_path = Path(args.manifest).expanduser() if args.manifest else default_manifest_path()
    manifest["manifest_path"] = str(manifest_path)

    if not args.check_only:
        atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.env_file:
            atomic_write(Path(args.env_file).expanduser(), env_lines(manifest))

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"upstream root: {root}")
        for name, entry in sorted(skills.items()):
            print(f"  {entry['status']:<22} {name:<26} {entry['skill_dir']}")
        if not args.check_only:
            print(f"manifest: {manifest_path}")
        if blockers:
            print("\nBlockers:", file=sys.stderr)
            for line in blockers:
                print(f"  - {line}", file=sys.stderr)

    return EXIT_BLOCKED if blockers else 0


def build_parser():
    """Build the two-phase preflight CLI."""
    parser = argparse.ArgumentParser(
        prog="doctor.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    phases = parser.add_subparsers(dest="phase", required=True, metavar="PHASE")

    deps = phases.add_parser(
        "dependencies",
        aliases=["deps"],
        help="resolve the upstream skills and write the dependency manifest",
        description=(
            "Resolve the upstream skills this showcase orchestrates and write "
            "the manifest the runner and the adapters consume. Exits 3 when a "
            "dependency is missing, unpinned, or incomplete."
        ),
    )
    deps.add_argument("--manifest", help="Where to write the dependency manifest.")
    deps.add_argument("--env-file", help="Also write sourceable shell assignments here.")
    deps.add_argument("--upstream-root", help="Normalized checkout root override.")
    deps.add_argument("--prepare", action="store_true",
                      help="Create or update managed checkouts. Not read-only.")
    deps.add_argument("--check-only", action="store_true",
                      help="Resolve and report; write nothing.")
    deps.add_argument(
        "--allow-commit-drift",
        action="store_true",
        help=(
            "Accept a checkout whose commit differs from the pin or whose "
            "worktree is dirty. The manifest is marked unpinned and "
            "drift_allowed, and the runtime refuses it when "
            "MISSION_CONTROL_SHOWCASE_REQUIRE_PREFLIGHT is set."
        ),
    )
    deps.add_argument("--json", action="store_true", help="Print the manifest to stdout.")
    deps.set_defaults(run=run_dependencies)

    host = phases.add_parser(
        "host",
        help="read-only checks of this machine",
        description=(
            "Read-only checks of Isaac Sim, GPU and driver, memory, Docker, "
            "ports, images, and the map assets. Changes nothing. Exits 1 on "
            "failure. Requires the entrypoints the dependency phase resolved."
        ),
    )
    host.add_argument("--skills-dir", required=True)
    host.add_argument("--asset-dir", required=True)
    host.add_argument("--bring-up-cloud", required=True)
    host.add_argument("--change-map", required=True)
    host.add_argument("--change-fleet", required=True)
    host.add_argument("--isaac-send", required=True)
    host.add_argument("--isaac-sim-dir", required=True)
    host.add_argument("--warehouse-usd-uri", required=True)
    host.add_argument("--nova-carter-image", required=True)
    host.add_argument("--robot-name", required=True)
    host.add_argument("--isaac-python-port", type=int, default=8226)
    host.add_argument(
        "--ignore-container",
        dest="ignore_containers",
        action="append",
        default=[],
        metavar="NAME",
        help="allow one running container by exact name; repeat as needed",
    )
    host.set_defaults(run=doctor)

    return parser


def main(argv=None):
    """Run one preflight phase."""
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
