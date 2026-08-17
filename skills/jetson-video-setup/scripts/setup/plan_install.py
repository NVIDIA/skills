#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan from environment 1.2 and execute APT under a canonical re-plan digest."""
# pylint: disable=missing-function-docstring, too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches, too-many-locals, too-many-return-statements
# pylint: disable=too-many-statements,wrong-import-position

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

_SETUP_DIR = Path(__file__).resolve().parent
if str(_SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(_SETUP_DIR))

from setup_contract import (FULL_PROFILE, SMOKE_PROFILE,  # noqa: E402
    VERIFICATION_PROFILES, authenticated_candidate_binding, canonical_json_sha256,
    cuda_at_least_13, cuda_build_package, durable_venv_path, environment_contract_errors,
    file_identity, native_sdk_release, normalized_cuda_evidence, normalized_native_surface,
    normalized_pynvc_surface, normalized_python_bootstrap, parse_apt_sources, profile_dependencies,
    public_candidate_binding, pycuda_bootstrap_packages, read_json, require_isolated,
    require_selected_surfaces_live, run_command, sha256_bytes,
    system_executable, transaction_sibling_path, utc_now, write_new_bytes, write_new_json)
SCHEMA_VERSION, APT_SCHEMA_VERSION, VALIDATION_SCHEMA = "1.5", "1.0", "1.0"
PLAN_KIND, ENVIRONMENT_KIND, ENVIRONMENT_MAJOR = "nvcodec-install-plan", "nvcodec-environment", "1"
COMPONENTS, SURFACES = ("native-sdk", "pynvc"), {"native-sdk": "native", "pynvc": "pynvc"}
REQUEST_INTENTS, TRANSACTION_ACTIONS = ("plan-only", "setup-install"), ("refresh", "preview",
                                                                        "apply")
NVIDIA_PACKAGE, NATIVE_RELEASE_FAMILY = "nvidia-video-codec-sdk", "13.0"
CUDA_PACKAGE, VENV_PACKAGE, DEV_PACKAGE = "cuda-toolkit", "python3-venv", "python3-dev"
BUILD_ENV_NAMES = ("PATH", "CPATH", "LIBRARY_PATH")
CUDA_BOOTSTRAP_ROUTE = "cuda-minimal-build-curand-bootstrap-then-replan"
PYNVC_VERSION, PYNVC_SPEC = "2.1.0", "PyNvVideoCodec==2.1.0"
RUNTIME_SPECS = {"numpy": "numpy>=1.24", "pycuda": "pycuda==2026.1", "torch": "torch==2.9.1+cu130"}
SUPPORT_SPECS = ("setuptools>=68", "wheel>=0.41", "pytools>=2011.2", "platformdirs>=2.2.0", "mako")
DEFAULT_PYNVC_VENV = Path.home() / ".venvs" / "nvcodec"
TRANSACTION_SCRIPT, PIP_SCRIPT = str(Path(__file__).resolve()), str(_SETUP_DIR /
                                                                    "lock_pip_reports.py")
TRANSACTION_PREFIX = (sys.executable, "-B", "-I", TRANSACTION_SCRIPT)
PLAN_ARTIFACT_NAME, REFRESH_RECEIPT_NAME = ("nvcodec-install-plan.json",
                                            "nvcodec-apt-refresh-receipt.json")
REFRESH_PREFLIGHT_ID, CLEAN_VENV_REPORT = "refresh-apt-metadata", "nvcodec-clean-venv-creation.json"
BINARY_REPORT, TORCH_REPORT = "nvcodec-pip-binary-resolve.json", "nvcodec-pip-torch-resolve.json"
INSECURE_APT_PATTERNS = tuple(rf"{name}\s+\"?(?:1|true|yes)\"?" for name in (
    "Acquire::AllowInsecureRepositories", "Acquire::AllowDowngradeToInsecureRepositories",
    "Acquire::AllowWeakRepositories", "APT::Get::AllowUnauthenticated"))
POLICY_FACTS = ("status", "package", "expected_candidate", "observed_candidate", "expected_origin",
                "bound_origin", "candidate_origins", "configured_sources",
                "configuration_bypasses", "verified")
APPROVAL_BATCH_REASONS = {
    "apt-metadata-refresh": "Refresh APT metadata once, then discard and regenerate the plan.",
    "native-sdk-install": "Apply the reviewed candidate-pinned native SDK APT transaction.",
    "pynvc-bootstrap": "Create the new isolated Python environment.",
    "pynvc-resolution": "Resolve every required Python package without applying anything.",
    "pynvc-install": "Apply the reviewed Python package set after each resolver report passes."}
_PUBLIC_ORIGIN_TEXT = ("the stock signature-authenticated public NVIDIA Jetson origin"
                       " (repo.download.nvidia.com/jetson/{{common,som}}, rNN.N/main)")
_UNBOUND = " is not bound to " + _PUBLIC_ORIGIN_TEXT + "; no install command was generated."
BLOCKED_ROUTES = {"interpreter": ("canonical-interpreter-required",
                                  "Setup requires the one canonical system interpreter: {}"),
                  "native-surface": ("native-surface-evidence-required",
                                     "The native surface is malformed, so it cannot be"
                                     " planned: {}"),
                  "native-cuda": ("cuda-toolkit-minimum-required",
                                  "The native surface requires an observed CUDA Toolkit 13.0+;"
                                  " saw {!r}."),
                  "native-installed": ("native-13.0-release-required",
                                       "Installed native SDK {!r} is outside the supported"
                                       " 13.0.x family."),
                  "native-candidate": (
                      "release-apt-candidate-required",
                      "No nvidia-video-codec-sdk candidate is in local APT metadata; verify the"
                      " image's release repository and never invent one."),
                  "native-release": ("native-13.0-release-required",
                                     "Observed candidate {!r} is not a supported 13.0.x release."),
                  "pynvc-surface": ("pynvc-surface-evidence-required",
                                    "The Python surface is malformed, so it cannot be planned: {}"),
                  "pynvc-fresh-venv": (
                      "fresh-venv-target-required",
                      "A fresh or reinstall PyNvVideoCodec setup requires an explicit unique"
                      " --venv target; none was supplied."),
                  "pynvc-reuse": (
                      "existing-environment-reuse-required",
                      "A complete PyNvVideoCodec environment is already at {}; reuse it, or pass"
                      " --fresh-setup with a new --venv to authorize one."),
                  "pynvc-new-venv": (
                      "new-venv-target-required",
                      "The requested venv is the observed environment at {}; choose a new path"
                      " because environments are never repaired or reused in place."),
                  "pynvc-venv-exists": (
                      "fresh-venv-target-exists",
                      "The --venv target already exists at {}; setup never deletes, repairs, or"
                      " reuses an environment. Choose a new unique path."),
                  "pynvc-venv-location": (
                      "durable-absolute-venv-required",
                      "The --venv target {!r} must be an absolute durable path outside the current"
                      " working directory and transient trees."),
                  "apt-architecture": (
                      "target-architecture-candidate-required",
                      "The {} candidate resolves to APT architecture {!r}, not the Jetson target;"
                      " no install command was generated."),
                  "cuda-evidence": (
                      "cuda-build-environment-required",
                      "PyCUDA needs the target CUDA build environment, but the"
                      " installation.cuda_toolkit block is malformed: {}"),
                  "build-tools": (
                      "pycuda-system-build-tools-required",
                      "PyCUDA needs canonical /usr/bin gcc, g++, and make executables: {}"),
                  "cuda-minimum": (
                      "cuda-toolkit-minimum-required",
                      "Observed CUDA candidate {!r} is not the matching stable CUDA 13.0+"
                      " release."),
                  "python-evidence": (
                      "python-bootstrap-evidence-required",
                      "A new isolated environment needs the installation.python bootstrap"
                      " evidence, which is malformed: {}"),
                  "apt-candidate": (
                      "authenticated-apt-candidate-required",
                      "No {} candidate exists in current authenticated APT metadata; verify the"
                      " image's repository, never add or change one."),
                  "base-apt-origin": (
                      "authenticated-apt-origin-required",
                      "The {} candidate {!r} is not bound to a configured signature-authenticated"
                      " APT origin; no install command was generated."),
                  "apt-origin": (
                      "authenticated-public-origin-required", "The {} candidate {!r}" + _UNBOUND),
                  "jetson-linux": (
                      "jetson-linux-incompatible",
                      "No installation command is generated until a live Jetson Linux 38.5+"
                      " environment is observed: {}")}
SETUP_HELPERS = ("plan_install.py", "probe_nvcodec.py", "lock_pip_reports.py", "setup_contract.py",
                 "verify_native.py", "verify_pynvc_sample.py")
UNKNOWNS = ("External network reachability was not tested.",
            "Operational readiness requires the post-install official sample artifacts.",
            "Jetson Linux 38.5+ is required; exact L4T and JetPack revisions are diagnostic only.")
NATIVE_PROBE, PYNVC_PROBE = "nvcodec-environment-after-native.json", "nvcodec-env-after-python.json"
PYNVC_VERIFICATION = "nvcodec-pynvc-verify.json"
VALIDATION_STEPS = {
    "native-sdk": ((str(_SETUP_DIR / "probe_nvcodec.py"), "--runtime", "native", "--output",
                   NATIVE_PROBE),
                   (str(_SETUP_DIR / "verify_native.py"), "--environment", NATIVE_PROBE,
                    "--sdk-root", "/opt/nvidia/video-codec-sdk", "--build-dir",
                    "nvcodec-native-build", "--work-dir", "nvcodec-native-smoke", "--build",
                    "--run-encode", "--run-decode", "--output", "nvcodec-native-verify.json")),
    "pynvc": ((str(_SETUP_DIR / "probe_nvcodec.py"), "--runtime", "pynvc", "--output", PYNVC_PROBE,
               "--setup-candidate"),
              (str(_SETUP_DIR / "verify_pynvc_sample.py"), "--environment", PYNVC_PROBE,
               "--work-dir", "nvcodec-pynvc-smoke", "--output", PYNVC_VERIFICATION,
               "--register-current", "--profile", "@profile"))}
POST_INSTALL_STEPS = {"native-sdk": ("produce-final-native-probe", "verify-native-once"),
                      "pynvc": ("produce-final-pynvc-probe", "verify-pynvc-once-and-register")}


class PlanContractError(ValueError):
    """No executable plan can be produced."""


def _dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True)


def _field(value: Any, path: tuple[str, ...]) -> Any:
    for name in path:
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def _emit(path: Path, document: dict[str, Any], code: int) -> int:
    write_new_json(path, document)
    print(_dumps(document))
    return code


def _run(argv: list[str]) -> dict[str, Any]:
    started, monotonic = utc_now(), time.monotonic()
    completed = run_command(argv)
    return {"argv": argv, "cwd": str(Path.cwd().resolve()), "started_at": started,
            "ended_at": utc_now(), "duration_seconds": time.monotonic() - monotonic,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.decode("utf-8", errors="replace"),
            "stderr": completed.stderr.decode("utf-8", errors="replace"),
            "stdout_bytes": completed.stdout, "stderr_bytes": completed.stderr}


def _observed(evidence: dict[str, Any], prefix: Path, stem: str = "") -> dict[str, Any]:
    infix = f".{stem}" if stem else ""
    streams = {}
    for name in ("stdout", "stderr"):
        raw = evidence[f"{name}_bytes"]
        written = write_new_bytes(prefix.with_name(f"{prefix.name}{infix}.{name}.log"), raw)
        streams[name] = {"path": str(written), "size_bytes": len(raw), "sha256": sha256_bytes(raw)}
    return {key: evidence[key] for key in ("argv", "cwd", "started_at", "ended_at",
                                           "duration_seconds", "exit_code")} | streams


def _missing(node: Any, paths: Any, label: str) -> list[str]:
    return [f"{label}.{'.'.join(path)} is required"
            for path in paths if not str(_field(node, path) or "").strip()]


def _cuda_errors(cuda: Any) -> list[str]:
    if not isinstance(cuda, dict):
        return ["cuda must be an object"]
    if cuda.get("status") not in {"installed", "absent"}:
        return [f"cuda.status must be installed or absent, saw {cuda.get('status')!r}"]
    if cuda["status"] == "absent":
        return []
    prefixes = cuda.get("environment_prefixes")
    bound = isinstance(prefixes, dict) and all(
        isinstance(prefixes.get(name), list) and prefixes[name] for name in BUILD_ENV_NAMES)
    return _missing(cuda, (("version",), ("root",)), "cuda") + ([] if bound else [
        f"cuda.environment_prefixes must list {', '.join(BUILD_ENV_NAMES)} when installed"])


def _command(phase: str, argv: list[str], *, mutates: bool, network: bool = False,
             privilege: str = "user", group: str | None = None) -> dict[str, Any]:
    protected = mutates or network
    if protected != (group in APPROVAL_BATCH_REASONS):
        raise ValueError(f"exactly the protected commands carry an approval group: {phase}")
    return {"phase": phase, "argv": argv, "authorization_status": "unassigned",
            "new_approval_required": None, "execution_group": group, "mutates": mutates,
            "requires_network": network, "privilege": privilege}


def _validation_commands(component: str, interpreter: str,
                         profile: str = SMOKE_PROFILE) -> list[list[str]]:
    """Bind the selected profile into the post-install verification argv the plan publishes."""
    return [[interpreter, "-I", *(profile if item == "@profile" else item for item in step)]
            for step in VALIDATION_STEPS[component]]


def _plan_commands(preflight: Any, plans: Any) -> list[tuple[str, dict[str, Any]]]:
    located = [(f"preflight_commands[{index}]", item)
               for index, item in enumerate(preflight if isinstance(preflight, list) else [])
               if isinstance(item, dict)]
    located.extend((f"{name}.commands[{index}]", item)
                   for name, component in (plans.items() if isinstance(plans, dict) else [])
                   if isinstance(component, dict)
                   for index, item in enumerate(component.get("commands") or [])
                   if isinstance(item, dict))
    return located


def _execution_batches(preflight: list[dict[str, Any]], plans: dict[str, dict[str, Any]],
                       request_intent: str) -> list[dict[str, Any]]:
    if request_intent not in REQUEST_INTENTS:
        raise ValueError(f"unsupported request intent: {request_intent}")
    preauthorized = request_intent == "setup-install"
    batches: dict[str, dict[str, Any]] = {}
    for location, item in _plan_commands(preflight, plans):
        if not (item["mutates"] or item["requires_network"]):
            item.update({"authorization_status": "not-required", "new_approval_required": False})
            continue
        item["authorization_status"] = "preauthorized" if preauthorized else "requires-confirmation"
        item["new_approval_required"] = not preauthorized
        group = item["execution_group"]
        batch = batches.setdefault(group, {
            "id": group, "authorization_status": item["authorization_status"], "phases": [],
            "new_approval_required": item["new_approval_required"], "execution": "sequential",
            "reason": APPROVAL_BATCH_REASONS[group], "command_locations": []})
        batch["command_locations"].append(location)
        batch["phases"].append(item["phase"])
    return list(batches.values())


def _component(name: str, observed: str) -> dict[str, Any]:
    return {"name": name, "observed_status": observed, "plan_status": "blocked",
            "selected_route": None, "artifact": {}, "blocking_reasons": [], "commands": [],
            "validation_commands": []}


def _blocked(result: dict[str, Any], route_id: str, *detail: Any) -> dict[str, Any]:
    route, reason = BLOCKED_ROUTES[route_id]
    result.update({"plan_status": "blocked", "selected_route": route, "commands": [],
                   "blocking_reasons": [reason.format(*detail)], "validation_commands": []})
    return result


def _apt_install(result: dict[str, Any], apt: dict[str, Any], package: str, *, phase: str,
                 group: str, refresh_done: bool, reinstall: bool = False,
                 missing_id: str = "apt-candidate",
                 nvidia_release: bool = False) -> tuple[list[dict[str, Any]], str] | None:
    record = _field(apt, ("candidates", package))
    record = record if isinstance(record, dict) else {}
    candidate = record.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        if refresh_done or not apt.get("source_files"):
            _blocked(result, missing_id, package)
        else:
            result.update({"plan_status": "ready_to_review", "blocking_reasons": [],
                           "commands": [], "validation_commands": [],
                           "selected_route": "apt-metadata-refresh-then-replan",
                           "required_preflight_actions": [REFRESH_PREFLIGHT_ID]})
            result["artifact"]["apt_metadata"] = {"source_files": apt.get("source_files", []),
                                                  "candidate_state": "unverified_until_refresh"}
        return None
    argv, apt_cache = record.get("query_argv"), apt.get("apt_cache")
    origin = None if (record.get("query_status") not in {"ok", "success"}
                      or record.get("query_exit_code") != 0 or not isinstance(apt_cache, str)
                      or argv != [apt_cache, "policy", package]) else (
            public_candidate_binding(record) if nvidia_release
            else authenticated_candidate_binding(record))
    if origin is None:
        _blocked(result, "apt-origin" if nvidia_release else "base-apt-origin",
                 package, candidate)
        return None
    if origin.get("architecture") not in ("aarch64", "arm64"):
        _blocked(result, "apt-architecture", package, origin.get("architecture"))
        return None
    result["artifact"].setdefault("apt_installs", {})[package] = {
        "package": package, "candidate": candidate, "selected_origin": origin,
        "candidate_origin_evidence": {key: record.get(key) for key in (
            "query_status", "query_argv", "query_exit_code", "candidate_origins")}}
    request = ["--package-spec", f"{package}={candidate}", *(["--reinstall"] if reinstall else [])]
    receipt = f"{phase}-apt-simulation.json"
    return [_command(f"{phase}-preview", [*TRANSACTION_PREFIX, "preview", *request,
                                          "--receipt", receipt],
                     mutates=False),
            _command(
                phase,
                [*TRANSACTION_PREFIX, "apply", *request, "--receipt", receipt,
                 "--output", f"{phase}-apt-apply.json"],
                mutates=True, network=True, privilege="root", group=group,
            )], candidate


def _plan_native(environment: dict[str, Any], apt: dict[str, Any], *, refresh_done: bool,
                 fresh_setup: bool) -> dict[str, Any]:
    surface, defects = normalized_native_surface(environment)
    installed = surface["installed"]
    package = _field(surface, ("package",))
    package = package if isinstance(package, dict) else {}
    package_installed = (package.get("name") == NVIDIA_PACKAGE
                         and package.get("status") == "installed")
    result = _component("native-sdk", "installed" if package_installed else "missing")
    if defects:
        return _blocked(result, "native-surface", "; ".join(defects))
    try:
        interpreter = system_executable("python3")
    except (FileNotFoundError, OSError) as exc:
        return _blocked(result, "interpreter", exc)
    if installed:
        cuda_version = str(_field(surface, ("cuda", "version")) or "")
        result["artifact"] = {
            "package": package.get("name"), "package_version": package.get("version"),
            "sdk_root": surface.get("sdk_root"), "cuda": surface.get("cuda"),
            "expected_release_family": NATIVE_RELEASE_FAMILY,
            "build_prerequisites": surface.get("build_prerequisites")}
        if not cuda_at_least_13(cuda_version):
            return _blocked(result, "native-cuda", cuda_version)
        if not native_sdk_release(package.get("version")):
            return _blocked(result, "native-installed", package.get("version"))
        if not fresh_setup:
            result.update({"plan_status": "installed",
                           "selected_route": "existing-release-apt-installation",
                           "validation_commands": _validation_commands("native-sdk", interpreter)})
            return result
    commands: list[dict[str, Any]] = []
    if not isinstance(_field(surface, ("tools", "pkg_config")), dict):
        selected = _apt_install(result, apt, "pkg-config", phase="install-native-pkg-config",
                                group="native-sdk-install", refresh_done=refresh_done)
        if selected is None:
            return result
        commands.extend(selected[0])
    selected = _apt_install(result, apt, NVIDIA_PACKAGE, phase="install-native-sdk",
                            group="native-sdk-install", refresh_done=refresh_done,
                            reinstall=package_installed, missing_id="native-candidate",
                            nvidia_release=True)
    if selected is None:
        return result
    sdk_commands, candidate = selected
    if not native_sdk_release(candidate):
        return _blocked(result, "native-release", candidate)
    result["artifact"].update({"package": NVIDIA_PACKAGE, "candidate": candidate,
                               "expected_release_family": NATIVE_RELEASE_FAMILY})
    commands.extend(sdk_commands)
    result.update({"plan_status": "ready_to_review", "commands": commands,
                   "selected_route": ("configured-release-apt-fresh-reinstall" if fresh_setup
                                      and package_installed else "configured-release-apt-repair"
                                      if package_installed else "configured-release-apt"),
                   "validation_commands": _validation_commands("native-sdk", interpreter)})
    return result


def _plan_pynvc(environment: dict[str, Any], apt: dict[str, Any], venv: Path | None, *,
                refresh_done: bool, fresh_setup: bool, profile: str) -> dict[str, Any]:
    surface, defects = normalized_pynvc_surface(environment)
    installed = surface["installed"]
    result = _component("pynvc", "installed" if installed else "missing")
    if defects:
        return _blocked(result, "pynvc-surface", "; ".join(defects))
    required = profile_dependencies(profile)
    ready = bool(installed and surface.get("version") == PYNVC_VERSION and all(
        _field(surface, ("dependencies", name, "ready")) is True for name in required))
    observed = str(surface.get("sys_prefix") or "").rstrip("/")
    result["artifact"] = {"requested_spec": PYNVC_SPEC, "verification_profile": profile,
                          "runtime_specs": {name: RUNTIME_SPECS[name] for name in required},
                          "observed_version": surface.get("version"), "observed_ready": ready,
                          "observed_prefix": surface.get("sys_prefix")}
    if venv is None and fresh_setup:
        return _blocked(result, "pynvc-fresh-venv")
    if venv is None:
        venv = Path(observed) if ready and observed else DEFAULT_PYNVC_VENV
    resolved_venv = durable_venv_path(venv)
    if resolved_venv is None:
        return _blocked(result, "pynvc-venv-location", str(venv))
    target = PurePosixPath(resolved_venv.as_posix())
    planned, target_python = str(target), str(target / "bin" / "python")
    same_target = bool(observed and observed == planned.rstrip("/"))
    if ready and not fresh_setup:
        if not same_target:
            return _blocked(result, "pynvc-reuse", observed)
        result.update({"plan_status": "installed", "selected_route":
                       "existing-isolated-environment", "validation_commands":
                       _validation_commands("pynvc", str(surface["interpreter"]), profile)})
        return result
    if installed and same_target:
        return _blocked(result, "pynvc-new-venv", observed)
    if os.path.lexists(planned):
        return _blocked(result, "pynvc-venv-exists", planned)
    cuda, python = normalized_cuda_evidence(environment), normalized_python_bootstrap(environment)
    python_errors = ["python must be an object"] if not isinstance(python, dict) else [
        f"python.{name} must be ok or missing, saw {python.get(name)!r}"
        for name in ("venv_module", "development_headers")
        if python.get(name) not in {"ok", "missing"}]
    for errors, route in ((_cuda_errors(cuda), "cuda-evidence"),
                          (python_errors, "python-evidence")):
        if errors:
            return _blocked(result, route, "; ".join(errors))
    if cuda["status"] == "absent":
        toolkit_candidate = _field(apt, ("candidates", CUDA_PACKAGE, "candidate"))
        packages = pycuda_bootstrap_packages(toolkit_candidate)
        if not packages:
            return _blocked(result, "cuda-minimum", toolkit_candidate)
        commands: list[dict[str, Any]] = []
        for package in packages:
            phase = ("install-pynvc-cuda-curand-dev" if package.startswith("libcurand")
                     else "install-pynvc-cuda-minimal-build")
            selected = _apt_install(result, apt, package, phase=phase, group="pynvc-bootstrap",
                                    refresh_done=refresh_done, nvidia_release=True)
            if selected is None:
                return result
            planned, candidate = selected
            if package.startswith("cuda-minimal-build-") and (
                    cuda_build_package(candidate) != package):
                return _blocked(result, "cuda-minimum", candidate)
            commands.extend(planned)
        result.update({"plan_status": "ready_to_review", "selected_route": CUDA_BOOTSTRAP_ROUTE,
                       "requires_reprobe_and_replan": True, "commands": commands,
                       "validation_commands": []})
        return result
    try:
        base = system_executable("python3")
        identity = file_identity(base, label="base interpreter")
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _blocked(result, "interpreter", exc)
    system_bin = Path("/usr/bin")
    try:
        for name in ("gcc", "g++", "make"):
            if (system_bin / name).resolve(strict=True) != Path(system_executable(name)):
                raise FileNotFoundError(f"{system_bin / name} is not the canonical {name}")
    except (FileNotFoundError, OSError) as exc:
        return _blocked(result, "build-tools", exc)
    prefixes = {name: list(cuda["environment_prefixes"][name]) for name in BUILD_ENV_NAMES}
    prefixes["PATH"] = list(dict.fromkeys([*prefixes["PATH"], str(system_bin)]))
    build_env = [f"{name}={':'.join(str(value) for value in prefixes[name])}"
                 for name in BUILD_ENV_NAMES]
    result["artifact"].update({
        "cuda_build_environment": {"version": cuda["version"], "root": cuda["root"],
                                   "environment": build_env, "scope": "PyCUDA build only"},
        "fresh_venv": {
            "base_executable": base, "base_interpreter_identity": identity, "target": planned,
            "target_python": target_python, "creation_report": CLEAN_VENV_REPORT,
            "target_must_not_exist": True, "reuse_allowed": False, "delete_existing": False,
            "fresh_setup_required": fresh_setup}})
    bootstrap: list[dict[str, Any]] = []
    for package, state, phase in ((VENV_PACKAGE, python["venv_module"], "install-python-venv"),
                                  (DEV_PACKAGE, python["development_headers"],
                                   "install-python-development-headers")):
        if state != "missing":
            continue
        selected = _apt_install(result, apt, package, phase=phase, group="pynvc-bootstrap",
                                refresh_done=refresh_done)
        if selected is None:
            return result
        bootstrap.extend(selected[0])
        result["artifact"].setdefault("repair_reasons", []).append(
            f"{package} is absent, so the plan installs its exact authenticated candidate before"
            " creating the isolated environment.")
    specs = [PYNVC_SPEC, RUNTIME_SPECS["numpy"], *SUPPORT_SPECS]
    resolve = [target_python, "-I", "-m", "pip", "--isolated", "install", "--no-input",
               "--disable-pip-version-check", "--no-cache-dir", "--ignore-installed", "--dry-run",
               "--report"]
    lock = [target_python, "-I", PIP_SCRIPT, "materialize-apply", "--report", BINARY_REPORT,
            *(["--report", TORCH_REPORT] if "torch" in required else [])]
    for requirement in [*specs, *(RUNTIME_SPECS[name] for name in required if name != "numpy")]:
        lock.extend(["--require", requirement])
    lock.extend(["--source-require", RUNTIME_SPECS["pycuda"], "--lock",
                 "nvcodec-pip.requirements.lock", "--manifest", "nvcodec-pip-lock-manifest.json",
                 "--install-report", "nvcodec-pip-locked-install.json"])
    for assignment in build_env:
        lock.extend(["--build-env", assignment])
    result["commands"] = [*bootstrap,
                          _command("create-venv",
                                   [base, "-I", PIP_SCRIPT, "create-venv", "--target", planned,
                                    "--output", CLEAN_VENV_REPORT], mutates=True,
                                   group="pynvc-bootstrap"),
                          _command("resolve-public-binary-wheels",
                                   [*resolve, BINARY_REPORT, "--only-binary=:all:", *specs],
                                   mutates=False,
                                   network=True, group="pynvc-resolution"),
                          *([_command("resolve-cuda-pytorch-wheel",
                                      [*resolve, TORCH_REPORT, "--only-binary=:all:",
                                       "--extra-index-url",
                                       "https://download.pytorch.org/whl/cu130",
                                       RUNTIME_SPECS["torch"]],
                                      mutates=False, network=True, group="pynvc-resolution")]
                            if "torch" in required else []),
                          _command("install-locked-python-artifacts", lock, mutates=True,
                                   network=True,
                                   group="pynvc-install")]
    result.update({"plan_status": "ready_to_review",
                   "selected_route": ("apt-bootstrap-then-fresh-isolated-environment" if bootstrap
                                      else "fresh-isolated-environment"),
                   "validation_commands": _validation_commands("pynvc", target_python, profile)})
    return result


def _receipt_binding(receipt_path: Path, environment: dict[str, Any],
                     source_path: Path) -> dict[str, Any]:
    def moment(value: Any) -> datetime | None:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    resolved = receipt_path.expanduser().resolve()
    errors, identity, receipt = [], {}, {}
    if resolved.parent != source_path.parent or resolved.name != REFRESH_RECEIPT_NAME:
        errors.append("refresh receipt must be the exact sibling of the refreshed probe artifact")
    try:
        identity = file_identity(resolved, label="refresh receipt")
        parsed = read_json(resolved)
        receipt = parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError) as exc:
        errors.append(f"refresh receipt could not be read: {exc}")
    errors.extend(f"refresh receipt {name} is not {expected!r}"
                  for name, expected in (("kind", "nvcodec-apt-refresh-receipt"),
                                         ("schema_version", APT_SCHEMA_VERSION), ("exit_code", 0))
                  if receipt.get(name) != expected)
    ended, generated = moment(receipt.get("ended_at")), moment(environment.get("generated_at"))
    if ended is None:
        errors.append("refresh receipt ended_at is invalid")
    elif generated is not None and ended > generated:
        errors.append("refresh receipt must precede the refreshed environment probe")
    return {"path": str(resolved), "sha256": identity.get("sha256"), "identity": identity,
            "verified": not errors, "errors": errors}


def _target_contract(environment: dict[str, Any]) -> dict[str, Any]:
    jetson = _field(environment, ("platform", "jetson_linux"))
    record = _field(jetson, ("compatibility",))
    record = record if isinstance(record, dict) else {}
    return {"compatible": record.get("status") == "compatible", "status": record.get("status"),
            "minimum_version": record.get("minimum_version"),
            "observed_version": _field(jetson, ("version",)),
            "reason": record.get("reason") or "no live Jetson Linux release was observed"}


def _overall(states: list[Any]) -> str:
    if "blocked" in states:
        return "blocked"
    if states and all(item == "installed" for item in states):
        return "installed"
    return "ready_to_review"


def build_plan(source_path: Path, components: list[str] | None = None, *, venv: Path | None = None,
               request_intent: str = "plan-only", fresh_setup: bool = False,
               profile: str = SMOKE_PROFILE,
               apt_refresh_receipt: Path | None = None) -> dict[str, Any]:
    resolved = source_path.expanduser().resolve()
    environment = read_json(resolved)
    if profile not in VERIFICATION_PROFILES:
        raise PlanContractError(f"verification profile must be one of"
                                f" {list(VERIFICATION_PROFILES)}; saw {profile!r}")
    if not isinstance(environment, dict) or environment.get("kind") != ENVIRONMENT_KIND:
        raise ValueError(f"input is not an {ENVIRONMENT_KIND} artifact")
    if environment.get("mode") != "live":
        raise ValueError(f"source artifact must be a live {ENVIRONMENT_KIND}")
    if (errors := environment_contract_errors(
            environment, include_surface_values=False)):
        raise PlanContractError(f"source artifact is not a live {ENVIRONMENT_KIND} "
                                f"{ENVIRONMENT_MAJOR}.x: " + "; ".join(errors))
    receipt = (_receipt_binding(apt_refresh_receipt, environment, resolved)
               if apt_refresh_receipt is not None else None)
    if receipt is not None and not receipt["verified"]:
        raise PlanContractError("APT refresh receipt validation failed: "
                                + "; ".join(receipt["errors"]))
    selected = components or [name for name in COMPONENTS
                              if environment["requested_runtime"] in (SURFACES[name], "both")]
    ordered = [name for name in COMPONENTS if name in set(selected)]
    candidates = _field(environment, ("installation", "apt"))
    apt = candidates if isinstance(candidates, dict) else {}
    done, plans = receipt is not None, {}
    contract, live = _target_contract(environment), {}
    if not contract["compatible"]:
        plans = {name: _blocked(_component(name, "unknown"), "jetson-linux", contract["reason"])
                 for name in ordered}
    elif "native-sdk" in ordered:
        live["native-sdk"] = _plan_native(environment, apt, refresh_done=done,
                                          fresh_setup=fresh_setup)
    if contract["compatible"] and "pynvc" in ordered:
        live["pynvc"] = _plan_pynvc(environment, apt, venv, refresh_done=done,
                                    fresh_setup=fresh_setup, profile=profile)
    plans.update(live)
    for name, component in plans.items():
        if component["plan_status"] == "blocked" and component["commands"]:
            raise ValueError(f"{name} is blocked but still contains executable commands")
    parked = [name for name, value in plans.items() if value.get("required_preflight_actions")]
    preflight = [{**_command("refresh-metadata",
                             [*TRANSACTION_PREFIX,
                              "refresh", "--receipt", REFRESH_RECEIPT_NAME],
                             mutates=True, network=True, privilege="root",
                             group="apt-metadata-refresh"),
                  "id": REFRESH_PREFLIGHT_ID, "affected_components": parked,
                  "next_step": "Discard this plan, re-run the live probe, and regenerate the plan"
                               " before installing packages."}] if parked else []
    batches = _execution_batches(preflight, plans, request_intent)
    preauthorized = request_intent == "setup-install"
    terminal = [name for name in COMPONENTS
                if plans.get(name, {}).get("plan_status") not in {None, "blocked"}
                and not plans.get(name, {}).get("required_preflight_actions")
                and plans.get(name, {}).get("selected_route") != CUDA_BOOTSTRAP_ROUTE]
    plan = {"schema_version": SCHEMA_VERSION, "kind": PLAN_KIND, "mutated": False,
            "generated_from": {**file_identity(resolved, label="source artifact"),
                               "kind": environment["kind"], "mode": environment["mode"],
                               "schema_version": environment["schema_version"],
                               "canonical_sha256": canonical_json_sha256(environment)},
            "plan_inputs": {
                "environment": str(resolved), "components": ordered,
                "request_intent": request_intent,
                "venv": str(venv) if venv is not None else None, "fresh_setup": fresh_setup,
                "verification_profile": profile,
                "apt_refresh_receipt": (str(apt_refresh_receipt.expanduser().resolve())
                                        if apt_refresh_receipt is not None else None)},
            "setup_helpers": {name: file_identity(_SETUP_DIR / name, label=name)
                              for name in SETUP_HELPERS},
            "target_contract": contract, "unknowns": list(UNKNOWNS),
            "verification_profile": profile,
            "platform": environment["platform"], "selected_gpu": environment["selected_gpu"],
            "requested_components": ordered, "execution_order": ordered, "fresh_setup": fresh_setup,
            "request_intent": request_intent, "apt_refresh_completed": done,
            "overall_status": _overall([value["plan_status"] for value in plans.values()]),
            "new_approval_required": any(item["new_approval_required"] for item in batches),
            "authorization_policy": {
                "mode": "intent-scoped", "normal_plan_preauthorized": preauthorized,
                "renew_on_drift": ["argv", "candidate", "dependency-set", "version", "origin",
                                   "archive-hash", "setup-mode", "verification-profile"]},
            "execution_batches": batches, "apt_refresh_receipt": receipt,
            "preflight_commands": preflight, "components": plans,
            "post_install_validation_sequence": [step for name in terminal
                                                 for step in POST_INSTALL_STEPS[name]]}
    plan["plan_digest"] = canonical_json_sha256(plan)
    return plan


def _require_canonical_plan(reviewed: Any) -> dict[str, Any]:
    if not isinstance(reviewed, dict) or reviewed.get("kind") != PLAN_KIND:
        raise ValueError("reviewed artifact is not an nvcodec install plan")
    digest = reviewed.get("plan_digest")
    if canonical_json_sha256({key: value for key, value in reviewed.items()
                              if key != "plan_digest"}) != digest:
        raise ValueError("reviewed plan digest does not cover its own content")
    inputs = reviewed.get("plan_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("reviewed plan carries no recorded planner inputs")
    require_selected_surfaces_live(read_json(Path(str(inputs["environment"]))),
                                   list(inputs["components"]))
    receipt = inputs.get("apt_refresh_receipt")
    fresh = build_plan(Path(str(inputs["environment"])), list(inputs["components"]),
                       venv=Path(str(inputs["venv"])) if inputs.get("venv") else None,
                       request_intent=str(inputs["request_intent"]),
                       fresh_setup=bool(inputs["fresh_setup"]),
                       profile=str(inputs.get("verification_profile") or FULL_PROFILE),
                       apt_refresh_receipt=Path(str(receipt)) if receipt else None)
    if fresh["plan_digest"] != digest:
        raise ValueError("reviewed plan is stale: the canonical in-process re-plan produced a"
                         " different digest. Re-probe, regenerate the plan, and review it again.")
    return fresh


def _authorizing_plan(invocation: list[str]) -> dict[str, Any]:
    directory = Path.cwd().resolve()
    plan_path = directory / PLAN_ARTIFACT_NAME
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"transaction actions require the reviewed sibling plan {plan_path}")
    plan = _require_canonical_plan(read_json(plan_path))
    receipt_path, binding = directory / REFRESH_RECEIPT_NAME, plan.get("apt_refresh_receipt")
    if receipt_path.is_file() and invocation[0] != "refresh" and (
            not isinstance(binding, dict) or binding.get("path") != str(receipt_path)
            or binding.get("sha256") != file_identity(receipt_path)["sha256"]):
        raise ValueError("an APT refresh completed in this working directory; re-probe to a new"
                         " artifact, re-plan, and review the new plan before applying")
    prefix = list(TRANSACTION_PREFIX)
    matches = [item for _, item in _plan_commands(plan["preflight_commands"], plan["components"])
               if (argv := list(item.get("argv", [])))[:len(prefix)] == prefix
               and argv[len(prefix):] == invocation]
    if len(matches) != 1:
        raise ValueError("this invocation is not exactly one command of the reviewed plan")
    if matches[0]["mutates"] or matches[0]["requires_network"]:
        if plan.get("request_intent") != "setup-install":
            raise ValueError("plan-only intent does not authorize a mutating transaction")
        if matches[0]["authorization_status"] != "preauthorized":
            raise ValueError("this transaction is not preauthorized by the reviewed plan")
    return plan


def _planned_origin(plan: dict[str, Any], package: str) -> dict[str, Any]:
    for component in plan["components"].values():
        artifact = component.get("artifact", {})
        installs = artifact.get("apt_installs") if isinstance(artifact, dict) else None
        for record in (artifact, *(installs.values() if isinstance(installs, dict) else ())):
            if (isinstance(record, dict) and record.get("package") == package
                    and isinstance(record.get("selected_origin"), dict)):
                return record["selected_origin"]
    raise ValueError(f"the reviewed plan binds no authenticated public origin for {package}")


def _transaction_context(args: argparse.Namespace,
                         invocation: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = bool(args.expected_origin_uri or args.expected_suite_component
                    or args.source_sha256 or args.source_entry)
    if supplied and not (args.expected_origin_uri and args.expected_suite_component
                         and args.source_sha256):
        raise ValueError("APT policy binding requires origin, suite, and source hashes")
    if len(args.source_sha256) != len(args.source_entry):
        raise ValueError("every --source-sha256 requires one paired --source-entry")
    sources = []
    for value, entry in zip(args.source_sha256, args.source_entry, strict=True):
        path, separator, digest = value.rpartition("=")
        source = Path(path).expanduser()
        if (not separator or not source.is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("--source-sha256 requires absolute-path=64-lowercase-hex")
        if source.is_symlink() or not source.is_file():
            raise ValueError("--source-sha256 path must be a regular non-symlink source file")
        sources.append({"path": str(source.resolve()), "sha256": digest,
                        "entry": json.loads(entry)})
    plan = _authorizing_plan(invocation)
    origin = _planned_origin(plan, args.package_spec.split("=", 1)[0])

    def key(item: dict[str, Any]) -> tuple[str, str]:
        return str(item.get("path")), _dumps(item.get("entry"))
    if supplied and (args.expected_origin_uri != origin.get("uri")
                     or args.expected_suite_component != origin.get("suite_component")
                     or sorted(sources, key=key) != sorted(origin.get("configured_sources", []),
                                                           key=key)):
        raise ValueError("APT origin/source assertions differ from the canonical reviewed plan")
    return plan, origin


def _apt_argv(action: str, package_spec: str = "", reinstall: bool = False) -> list[str]:
    apt_get, sudo = system_executable("apt-get"), [system_executable("sudo"), "-n"]
    if action == "refresh":
        return [*sudo, apt_get, "-o", "APT::Update::Error-Mode=any", "update"]
    extra = ["--reinstall"] if reinstall else []
    if action == "simulate":
        return [apt_get, "--simulate", "--no-remove", "install", *extra, package_spec]
    return [*sudo, apt_get, "--no-remove", "install", "-y", *extra, package_spec]


def _transaction_summary(stdout: str, stderr: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, str]]] = {"installed": [], "removed": [], "configured": []}
    patterns = (("Inst", "installed"), ("Unpacking", "installed"), ("Remv", "removed"),
                ("Removing", "removed"), ("Conf", "configured"), ("Setting up", "configured"))
    for line in stdout.splitlines():
        for keyword, bucket in patterns:
            if (match := re.match(rf"^{keyword}\s+(\S+)\s*(.*)$", line)):
                buckets[bucket].append({"package": match.group(1),
                                        "detail": match.group(2).strip()})
                break
    combined = f"{stdout}\n{stderr}"
    if re.search(r"The following packages will be REMOVED", combined, re.I):
        buckets["removed"].append({"package": "<apt-removal-set>", "detail": "removal summary"})
    return {**buckets, "downgrade_detected": bool(
        re.search(r"\bDOWNGRADED\b|\bdowngrad(?:e|ed|ing)\b", combined, re.I))}


def _policy_origins(stdout: str, candidate: str) -> list[dict[str, Any]]:
    lines = stdout.splitlines()
    header = re.compile(r"^\s*(?:\*{3}\s+)?\S+\s+\d+\s*$")
    row = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+Packages\s*$")
    selected = rf"^\s*(?:\*{{3}}\s+)?{re.escape(candidate)}\s+\d+\s*$"
    start = next((index for index, line in enumerate(lines)
                  if re.match(selected, line)), None) if candidate else None
    origins: list[dict[str, Any]] = []
    for line in lines[start + 1:] if start is not None else []:
        if header.match(line):
            break
        if (match := row.match(line)):
            origins.append({"priority": int(match.group(1)), "uri": match.group(2).rstrip("/"),
                            "suite_component": match.group(3), "architecture": match.group(4)})
    return origins


def _policy_evidence(package: str, expected: str, origin: dict[str, Any],
                     prefix: Path) -> dict[str, Any]:
    policy, config = _run([system_executable("apt-cache"), "policy", package]), _run(
        [system_executable("apt-config"), "dump"])
    match = re.search(r"^\s*Candidate:\s*(\S+)", policy["stdout"], re.MULTILINE)
    observed = match.group(1) if match and match.group(1) != "(none)" else None
    sources = []
    for item in origin.get("configured_sources", []):
        raw = Path(str(item.get("path"))).read_bytes()
        entries = parse_apt_sources(item.get("path"), raw.decode("utf-8", errors="replace"))
        sources.append({"path": item.get("path"), "expected_sha256": item.get("sha256"),
                        "sha256": sha256_bytes(raw), "expected_entry": item.get("entry"),
                        "matches": sha256_bytes(raw) == item.get("sha256"),
                        "entry_match_count": sum(1 for e in entries if e == item.get("entry")),
                        "trust_bypass": any(e.get("trust_bypass") is True for e in entries)})
    bypasses = [name for name in INSECURE_APT_PATTERNS if re.search(name, config["stdout"], re.I)]
    clean = bool(sources and not bypasses and all(
        item["matches"] and item["entry_match_count"] == 1 and not item["trust_bypass"]
        for item in sources))
    live = _policy_origins(policy["stdout"], observed or "")
    binder = (public_candidate_binding if package in {NVIDIA_PACKAGE, CUDA_PACKAGE}
              or package.startswith(("cuda-minimal-build-", "libcurand-dev-"))
              else authenticated_candidate_binding)
    bound = binder({
        "query_status": "ok", "candidate": observed, "candidate_origins": [
            {**item, "authentication": "apt-signature-chain",
             "configured_source_sha256": [entry["sha256"] for entry in sources],
             "configured_sources": origin.get("configured_sources", [])}
            for item in live]}) if clean and observed else None
    verified = bool(policy["exit_code"] == 0 and config["exit_code"] == 0 and observed == expected
                    and bound is not None and bound.get("uri") == origin.get("uri")
                    and bound.get("suite_component") == origin.get("suite_component"))
    keys = ("uri", "suite_component")
    return {"status": "verified" if verified else "failed", "package": package,
            "expected_candidate": expected, "observed_candidate": observed, "verified": verified,
            "expected_origin": {key: origin.get(key) for key in keys},
            "bound_origin": {key: bound.get(key) for key in keys} if bound else None,
            "candidate_origins": live, "configured_sources": sources,
            "configuration_bypasses": bypasses,
            "policy_command": _observed(policy, prefix, "policy"),
            "config_command": _observed(config, prefix, "apt-config")}


def _simulation(package_spec: str, reinstall: bool, prefix: Path,
                origin: dict[str, Any]) -> tuple[dict[str, Any], int]:
    package, _, candidate = package_spec.partition("=")
    policy = _policy_evidence(package, candidate, origin, prefix)
    evidence = _run(_apt_argv("simulate", package_spec, reinstall))
    transaction = _transaction_summary(evidence["stdout"], evidence["stderr"])
    safe = bool(policy["verified"] and evidence["exit_code"] == 0 and not transaction["removed"]
                and not transaction["downgrade_detected"])
    return {"schema_version": APT_SCHEMA_VERSION, "kind": "nvcodec-apt-simulation-receipt",
            "package_spec": package_spec, "reinstall": reinstall, "policy": policy,
            **_observed(evidence, prefix), "simulation_argv": evidence["argv"],
            "transaction": transaction, "safe_to_apply": safe}, 0 if safe else 2


def refresh(receipt_path: Path) -> int:
    receipt_path = transaction_sibling_path(receipt_path, "refresh receipt", ("",))
    _authorizing_plan(sys.argv[1:])
    evidence = _run(_apt_argv("refresh"))
    return _emit(receipt_path, {
        "schema_version": APT_SCHEMA_VERSION, "kind": "nvcodec-apt-refresh-receipt",
        **_observed(evidence, receipt_path),
        "next_step": "The candidate set may have changed. Discard this plan, re-run"
                     " probe_nvcodec.py to a new sibling artifact, re-plan with"
                     " --apt-refresh-receipt, and review the new plan before installing."},
                 0 if evidence["exit_code"] == 0 else 2)


def preview(args: argparse.Namespace, invocation: list[str]) -> int:
    receipt_path = transaction_sibling_path(
        args.receipt, "simulation receipt", ("policy", "apt-config", ""))
    _, origin = _transaction_context(args, invocation)
    receipt, code = _simulation(args.package_spec, args.reinstall, receipt_path, origin)
    return _emit(receipt_path, receipt, code)


def apply(args: argparse.Namespace, invocation: list[str]) -> int:
    receipt_path = transaction_sibling_path(args.receipt, "receipt")
    output = transaction_sibling_path(
        args.output, "report", ("immediate-simulation.policy",
                                "immediate-simulation.apt-config",
                                "immediate-simulation", ""))
    if output == receipt_path:
        raise ValueError("APT apply report and reviewed receipt must be distinct sibling files")
    plan, origin = _transaction_context(args, invocation)
    reviewed_identity = file_identity(receipt_path, label="reviewed receipt")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("kind") != "nvcodec-apt-simulation-receipt":
        raise ValueError("the reviewed receipt is not an nvcodec APT simulation receipt")
    if (receipt.get("package_spec") != args.package_spec
            or receipt.get("reinstall") is not args.reinstall
            or receipt.get("exit_code") != 0 or receipt.get("safe_to_apply") is not True):
        raise ValueError("reviewed APT simulation receipt does not match this safe transaction")
    repeated, code = _simulation(args.package_spec, args.reinstall,
                                 output.with_name(output.name + ".immediate-simulation"), origin)
    if code != 0:
        raise ValueError("immediate pre-apply APT simulation is unsafe")
    for name in ("package_spec", "reinstall", "simulation_argv", "transaction"):
        if repeated[name] != receipt.get(name):
            raise ValueError(f"APT transaction drifted after reviewed simulation: {name}")
    if any(repeated["policy"][name] != receipt.get("policy", {}).get(name)
           for name in POLICY_FACTS):
        raise ValueError("APT candidate/origin/source policy drifted after review")
    evidence = _run(_apt_argv("apply", args.package_spec, args.reinstall))
    transaction = _transaction_summary(evidence["stdout"], evidence["stderr"])
    return _emit(output, {"schema_version": APT_SCHEMA_VERSION, "kind": "nvcodec-apt-apply-report",
                          "package_spec": args.package_spec, "reinstall": args.reinstall,
                          "authorizing_plan_digest": plan["plan_digest"],
                          "reviewed_receipt": reviewed_identity,
                          "immediate_simulation": repeated, "transaction": transaction,
                          "safe": (safe := bool(
                              evidence["exit_code"] == 0 and not transaction["removed"]
                              and not transaction["downgrade_detected"])),
                          **_observed(evidence, output)}, 0 if safe else 2)


def validate_plan(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["install plan must be an object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        return [f"schema_version {SCHEMA_VERSION} is required; {data.get('schema_version')!r} is"
                " historical-review-only and cannot validate for execution"]
    components = data.get("components") if isinstance(data.get("components"), dict) else {}
    renew = data.get("request_intent") != "setup-install"
    states = [value.get("plan_status") for value in components.values() if isinstance(value, dict)]
    errors = [message for ok, message in (
        (data.get("kind") == PLAN_KIND, f"kind must be {PLAN_KIND}"),
        (data.get("request_intent") in REQUEST_INTENTS,
         "request_intent must be plan-only or setup-install"),
        (isinstance(data.get("components"), dict), "components must be an object"),
        (len(states) == len(components), "every component must be an object"),
        (not states or data.get("overall_status") == _overall(states),
         f"overall_status should be {_overall(states)} for component states {states}"),
        (canonical_json_sha256({key: value for key, value in data.items()
                                if key != "plan_digest"}) == data.get("plan_digest"),
         "plan_digest does not cover its own content")) if not ok]
    errors.extend(f"{name} is blocked but still contains executable commands"
                  for name, value in components.items()
                  if isinstance(value, dict) and value.get("plan_status") == "blocked"
                  and value.get("commands"))
    for location, item in _plan_commands(data.get("preflight_commands"), components):
        expected = "requires-confirmation" if renew else "preauthorized"
        if (item.get("mutates") is True or item.get("requires_network") is True) and (
                item.get("authorization_status") != expected
                or item.get("new_approval_required") is not renew):
            errors.append(f"{location} must be {expected} with new_approval_required={renew}")
    try:
        _require_canonical_plan(data)
    except (OSError, ValueError) as exc:
        errors.append(f"canonical re-plan rejected this plan: {exc}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, epilog="'validate PLAN' checks"
                                     " a plan; refresh/preview/apply run APT beside the plan.")
    parser.add_argument("environment", type=Path,
                        help=f"{ENVIRONMENT_KIND} {ENVIRONMENT_MAJOR}.x JSON from probe_nvcodec.py")
    parser.add_argument("--component", action="append", choices=COMPONENTS, dest="components",
                        help="surface to plan; repeatable. Defaults to requested_runtime.")
    parser.add_argument("--venv", type=Path, default=None, help="Isolated Python environment"
                        " target. Omitted, a complete observed environment is reused, else the"
                        " default new target. A different path needs --fresh-setup and a new one.")
    parser.add_argument("--request-intent", choices=REQUEST_INTENTS, default="plan-only",
                        help="Use setup-install only when installation was explicitly requested")
    parser.add_argument("--apt-refresh-completed", action="store_true",
                        help="Compatibility assertion; requires --apt-refresh-receipt")
    parser.add_argument("--apt-refresh-receipt", type=Path, help="Successful nvcodec-apt-refresh"
                        "-receipt JSON; must be the sibling of the post-refresh probe artifact")
    parser.add_argument("--fresh-setup", action="store_true", help="Require each surface's fresh"
                        " action, without uninstalling working base packages")
    parser.add_argument("--profile", choices=VERIFICATION_PROFILES, default=SMOKE_PROFILE,
                        help="Official-sample verification profile the Python environment is"
                             " provisioned for. The default installs no Torch; full-samples adds"
                             " it for the sample routes that import it.")
    parser.add_argument("--output", type=Path,
                        help="Also write the plan here; an existing path is never overwritten")
    return parser


def plan_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.apt_refresh_completed and args.apt_refresh_receipt is None:
        raise PlanContractError("--apt-refresh-completed requires --apt-refresh-receipt from the"
                                " successful refresh action and cannot stand alone")
    plan = build_plan(args.environment, args.components, venv=args.venv,
                      request_intent=args.request_intent, fresh_setup=args.fresh_setup,
                      apt_refresh_receipt=args.apt_refresh_receipt, profile=args.profile)
    if args.output:
        write_new_json(args.output, plan)
    print(_dumps(plan))
    return 2 if plan["overall_status"] == "blocked" else 0


def validate_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} validate",
        description=("Validate one reviewed nvcodec-install-plan. Non-mutating, but it re-plans"
                     " canonically, so the probe artifact named in plan_inputs must be readable."))
    parser.add_argument("plan", type=Path, help="nvcodec-install-plan JSON to validate")
    data = read_json(parser.parse_args(argv).plan)
    errors = validate_plan(data)
    print(_dumps({"schema_version": VALIDATION_SCHEMA, "errors": errors, "valid": not errors,
                  "kind": "nvcodec-install-plan-validation", "overall_status": (
                      data.get("overall_status") if isinstance(data, dict) else None)}))
    return 0 if not errors else 2


def transaction_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name,
                                     description="Execute one planned APT transaction.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("refresh").add_argument("--receipt", type=Path, required=True)
    for name in ("preview", "apply"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--package-spec", required=True)
        sub.add_argument("--reinstall", action="store_true")
        sub.add_argument("--expected-origin-uri")
        sub.add_argument("--expected-suite-component")
        sub.add_argument("--source-sha256", action="append", default=[])
        sub.add_argument("--source-entry", action="append", default=[])
        sub.add_argument("--receipt", type=Path, required=True)
        if name == "apply":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "refresh":
        return refresh(args.receipt)
    if not re.fullmatch(r"[^=\s]+=[^=\s]+", args.package_spec):
        raise ValueError("--package-spec must be exact package=version")
    if args.action == "preview":
        return preview(args, argv)
    return apply(args, argv)


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    validating, status, code = action == "validate", "invalid", 2
    try:
        require_isolated()
        if validating:
            return validate_main(sys.argv[2:])
        if action in TRANSACTION_ACTIONS:
            return transaction_main(sys.argv[1:])
        return plan_main(sys.argv[1:])
    except PlanContractError as exc:
        error = exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error, status, code = exc, "error", 3
    bare = action in TRANSACTION_ACTIONS  # the APT actions keep exactly two keys, never a superset
    if validating or bare:  # both always terminate at rc 3; only planning reports invalid/rc 2
        status, code = "error", 3
    print(_dumps({"status": status, "error": str(error)} if bare else {
        "schema_version": VALIDATION_SCHEMA if validating else SCHEMA_VERSION, "status": status,
        "kind": f"{PLAN_KIND}-validation-error" if validating else f"{PLAN_KIND}-error",
        "error": str(error)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
