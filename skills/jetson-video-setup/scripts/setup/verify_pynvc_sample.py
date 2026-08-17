#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate and run the wheel-owned NVIDIA PyNvVideoCodec encode/decode samples.

This setup operation authority scores real sample output and authenticates every imported module,
native extension, sample, helper, and config against wheel ``RECORD``. It is not a capability
query, owns its command line, uses only the standard library, and imports no sibling skill.
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-lines,too-many-locals

from __future__ import annotations

import argparse
import base64
import csv
import importlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SETUP_DIR = Path(__file__).resolve().parent
if str(_SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(_SETUP_DIR))

# pylint: disable=wrong-import-position
from setup_contract import (  # noqa: E402
    AUXILIARY_MODULE, AUXILIARY_ROLE, AUXILIARY_STEM, FULL_PROFILE, SMOKE_PROFILE, _read_regular,
    _symlink_free, VERIFICATION_PROFILES, associated_extensions, canonical_json_sha256, read_json,
    extension_stem, file_identity, parse_jetson_release, profile_dependencies, run_command,
    require_isolated, sha256_bytes, utc_now, write_new_bytes, write_new_json,
)
# pylint: enable=wrong-import-position

SCHEMA_VERSION = "1.3"
DIST_NAME = "PyNvVideoCodec"
EXPECTED_VERSION = "2.1.0"
RESULT_KIND = "nvcodec-pynvc-sample-verification"
ERROR_KIND = "nvcodec-pynvc-sample-verification-error"
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_MAJOR = "1"
KNOWN_STALE_RECORD = "known_upstream_stale_record"
MINIMUM_JETSON_LINUX = (38, 5)
_REGISTRY_PATH = Path.home() / ".local/state/jetson-videosdk/current-pynvc.json"
_REGISTRY_SCHEMA = "jetson-videosdk/current-pynvc/1"
_PYNVC_FIELDS = ("interpreter", "interpreter_identity", "sys_prefix", "version", "extension")
# The bounded smoke proves this tuple; full-samples retains the caller's positive geometry/count.
SMOKE_WIDTH, SMOKE_HEIGHT, SMOKE_FRAMES = 640, 360, 1
# Public PyNvVideoCodec 2.1.0 has one evidenced stale RECORD row for its native extension.
# No other member may miss its hash, and the exact distribution version must also match.
_NATIVE_EXTENSION_MEMBER = re.compile(
    r"PyNvVideoCodec/_?PyNvVideoCodec(?:_(?:121|130))?\.[^/]*\.so")
_AUXILIARY_FIELDS = ("role", "path", "loaded_path", "sha256")
_DEPENDENCY_RULES: dict[str, dict[str, Any]] = {
    "numpy": {"minimum": (1, 24)}, "pycuda": {"version": "2026.1"},
    "torch": {"version": "2.9.1+cu130", "cuda_build": "13.0", "cuda_available": True}}
# Every wheel-owned row the official samples read at run time; "config" is resolved per install.
# Both decode routes stay authenticated under either profile, but exactly one is executed.
SAMPLE_PATHS = {"encode": "basic/encode.py", "decode": "advanced/decode.py",
                "decode_perf": "advanced/decode_perf.py",
                # ``from utils.<x> import`` executes this initializer, so it is authenticated code.
                "__init__.py": "utils/__init__.py",
                "Utils.py": "utils/Utils.py", "frame_utils.py": "utils/frame_utils.py",
                "encode_parser.py": "utils/encode_parser.py",
                "decode_parser.py": "utils/decode_parser.py"}
PROFILE_DECODE_SAMPLE = {SMOKE_PROFILE: "decode_perf", FULL_PROFILE: "decode"}
# One decoded-output invariant, owned by verify() and merely restated by the CLI.
_SMOKE_DECODED_CONFLICT = ("a decoded-output path is valid only with --profile full-samples; the"
                           " pynvc-smoke decode route (advanced/decode_perf.py) writes no raw"
                           " output")
_SMOKE_GEOMETRY_CONFLICT = (
    f"the pynvc-smoke readiness proof is fixed at {SMOKE_WIDTH}x{SMOKE_HEIGHT}"
)


def _validate_smoke_request(profile: str, width: int, height: int, frames: int,
                            decoded_path: Path | None) -> None:
    if profile != SMOKE_PROFILE:
        return
    if frames != SMOKE_FRAMES:
        raise ValueError(_SMOKE_FRAMES_CONFLICT)
    if (width, height) != (SMOKE_WIDTH, SMOKE_HEIGHT):
        raise ValueError(_SMOKE_GEOMETRY_CONFLICT)
    if decoded_path is not None:
        raise ValueError(_SMOKE_DECODED_CONFLICT)


_SMOKE_FRAMES_CONFLICT = ("--profile pynvc-smoke verifies exactly one bounded frame; use"
                          " --frames 1 or provision --profile full-samples for a multi-frame"
                          " official-sample proof")
ENCODE_SUCCESS_RE = re.compile(r"Completed encoding\s+(\d+)\s+frames using CPU buffers")
DECODE_SUCCESS_RE = re.compile(
    r"^Successfully decoded requested\s+(\d+)\s+frames to\s+(.+?)\s*$", re.MULTILINE)
# The bounded one-frame decode_perf.py smoke. Public 2.1.0 emits, on the ``-m thread`` path this
# argv pins, "Thread <name>: Successfully decoded requested <n> frames" (line 124) and the
# unprefixed summary "Total frames decoded: <n>" (line 274). Both are mandatory and each must
# appear exactly once: line 205 prints its worker error from inside an ``except`` block, so the
# child can exit 0 after a failed decode and a clean exit alone is never a proof. Each marker gets
# its own anchored pattern rather than one loose "Successfully decoded" pattern, because line 134
# also emits the distinct "Successfully decoded all <n> frames" that a loose pattern double-counts.
ENCODE_FAILURE_PATTERN = re.compile(
    r"(?im)^\s*(?:CreateEncoder failure:|An unexpected error occurred:"
    r"|Traceback \(most recent call last\)|Preset [^\r\n]* not found\. Using default\."
    r"|Invalid RC mode given\. Using cbr as default"
    r"|Can't parse bitrate string\. Using default value)")
_DECODE_FAILURES = (
    r"Operation or configuration not supported:|An unexpected error occurred:"
    r"|CUDA error \(context/stream/memory\):|Traceback \(most recent call last\)"
    r"|Decode Error occurred for picture\b"
    r"|NvDecoder::~NvDecoder\(\): exception during cleanup \(suppressed\)")
DECODE_FAILURE_PATTERN = re.compile(rf"(?im)^\s*(?:{_DECODE_FAILURES})")
# The perf route additionally rejects every swallowed worker failure and every degraded-run
# warning, either of which can accompany a zero exit code: "Thread/Process <name> error:" (line
# 205) and "Thread <name>: Warning: Video ended before reaching requested frame count" (line 132).
# Match only the sample's exact degraded-run warning; unrelated library warnings are diagnostic.
PERF_FAILURE_PATTERN = re.compile(
    rf"(?im)(?:^\s*(?:{_DECODE_FAILURES})|\b(?:Thread|Process)\b[^:\r\n]*\berror:"
    r"|^\s*Thread\b[^:\r\n]*:\s*Warning:\s*Video ended before reaching requested frame count\b)")


class _NotReady(ValueError):
    """A valid surface cannot run the official sample yet; a completed negative."""


def _render(value: Any) -> str:
    """Strict stdout serialization; allow_nan=False rejects non-finite floats at the source."""
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True)


def _result(kind: str, status: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "status": status, **fields}


def _version_at_least(value: Any, minimum: tuple[int, ...]) -> bool:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)*)\s*", value) if isinstance(value, str) else None
    if not match:
        return False
    observed = tuple(int(part) for part in match.group(1).split("."))
    width = max(len(observed), len(minimum))
    return observed + (0,) * (width - len(observed)) >= minimum + (0,) * (width - len(minimum))


def _require_fields(value: Any, fields: tuple[str, ...], *, label: str) -> dict[str, Any]:
    """Required-subset validation: present and non-null. Unknown keys are always permitted."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = [name for name in fields if value.get(name) is None]
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    return value


def _same_path(first: Any, second: Any) -> bool:
    return Path(str(first)).resolve() == Path(str(second)).resolve()


# --- Wheel RECORD authentication of the files this process actually executes --

def _wheel_owned_identity(distribution: Any, path: Any, *, label: str) -> dict[str, Any]:
    """Prove one executed file is a wheel-RECORD member inside sys.prefix and hash it locally.

    Ownership is the assertion: the executed path resolves to exactly one installed distribution
    entry and exactly one ``RECORD`` row, both inside the running ``sys.prefix``. The recorded
    digest is re-derived here and published as evidence rather than silently trusted.
    """
    identity = file_identity(path, label=label)
    resolved = Path(identity["path"])
    if not resolved.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"{DIST_NAME} {label} {resolved} is outside sys.prefix {sys.prefix}")
    entries = [entry for entry in distribution.files or []
               if Path(distribution.locate_file(entry)).resolve() == resolved]
    if len(entries) != 1:
        raise ValueError(
            f"expected one {DIST_NAME} distribution entry for {resolved}, found {len(entries)}")
    text = distribution.read_text("RECORD")
    if not text:
        raise ValueError(f"installed {DIST_NAME} distribution has no readable RECORD")
    member = str(entries[0]).replace("\\", "/")
    rows = [row for row in csv.reader(io.StringIO(text))
            if row and row[0].replace("\\", "/") == member]
    if len(rows) != 1:
        raise ValueError(f"expected one wheel RECORD row for {member}, found {len(rows)}")
    encoded_hash, recorded_size = (rows[0] + ["", ""])[1:3]
    record: dict[str, Any] = {
        "record_path": member, "hash": encoded_hash or None,
        "size_bytes": int(recorded_size) if str(recorded_size).isdigit() else None}
    if not encoded_hash or "=" not in encoded_hash:
        record["consistency"] = "record_hash_absent"
    else:
        algorithm, expected = encoded_hash.split("=", 1)
        if algorithm.lower() != "sha256":
            raise ValueError(f"{label} wheel RECORD hash algorithm must be sha256, got {algorithm}")
        digest = base64.urlsafe_b64encode(
            bytes.fromhex(identity["sha256"])).rstrip(b"=").decode("ascii")
        record["consistency"] = (
            "matches_record" if digest == expected
            and record["size_bytes"] in (None, identity["size_bytes"]) else "upstream_mismatch")
    return {**identity, "wheel_member": member, "record": record}


def _require_record_consistency(identity: dict[str, Any], *, label: str, version: str,
                                allow_native_stale: bool = False) -> None:
    """Fail on a RECORD mismatch, permitting only the one evidenced native-extension row.

    Scoped by exact distribution version AND exact ``RECORD`` member: the public PyNvVideoCodec
    2.1.0 native extension and nothing else. Ownership of the executed file is still proven
    independently, so the exception narrows to the recorded digest alone and is published."""
    record, member = identity["record"], identity["wheel_member"]
    if record["consistency"] == "matches_record":
        return
    if (allow_native_stale and record["consistency"] == "upstream_mismatch"
            and version == EXPECTED_VERSION
            and _NATIVE_EXTENSION_MEMBER.fullmatch(member) is not None):
        record["consistency"] = KNOWN_STALE_RECORD
        record["exception"] = (
            f"known upstream {DIST_NAME} {EXPECTED_VERSION} stale RECORD row for its native"
            " extension; wheel ownership of the executed file is still proven locally")
        return
    raise ValueError(
        f"{DIST_NAME} {label} does not match its wheel RECORD ({record['consistency']}): {member}")


def _authenticate_auxiliaries(distribution: Any, module_identity: dict[str, Any],
                              extension_identity: dict[str, Any]) -> list[dict[str, Any]]:
    """Authenticate zero or one exact ``PyNvVideoCodec.VersionCheck`` helper beside the primary.

    The helper is held to the ORDINARY wheel-RECORD rule: ``_require_record_consistency`` is
    called with the default ``allow_native_stale=False``, so the primary's version- and
    member-scoped public-2.1.0 stale-row exception can never widen to cover it. It must live in
    the same package directory under the same ``sys.prefix``, and every other associated
    compiled path -- a second helper, a foreign extension, an unexpected submodule -- is
    rejected outright rather than ignored."""
    package_dir = os.path.dirname(module_identity["path"])
    prefix = str(Path(sys.prefix).resolve())
    associated, symlinked = associated_extensions(package_dir)
    if symlinked:
        raise ValueError(f"loaded PyNvVideoCodec extensions must not be symlinks: {symlinked}")
    candidates = [path for path, names in associated.items() if AUXILIARY_MODULE in names]
    if len(candidates) > 1:
        raise ValueError(f"{AUXILIARY_MODULE} must resolve to one loaded path; saw {candidates}")
    rows: list[dict[str, Any]] = []
    for resolved in candidates:
        if (extension_stem(resolved) != AUXILIARY_STEM
                or os.path.dirname(resolved) != package_dir
                or not resolved.startswith(prefix + os.sep)):
            raise ValueError(f"{AUXILIARY_MODULE} must be a readable {AUXILIARY_STEM} extension"
                             f" directly in {package_dir} under {prefix}; saw {resolved}")
        identity = _wheel_owned_identity(distribution, resolved, label="VersionCheck auxiliary")
        _require_record_consistency(identity, label="VersionCheck auxiliary",
                                    version=str(distribution.version))
        rows.append({**identity, "role": AUXILIARY_ROLE, "loaded_path": resolved,
                     "module_names": associated[resolved]})
    expected = {extension_identity["path"], *candidates}
    if unexpected := sorted(set(associated) - expected):
        raise ValueError(f"only the selected primary and one {AUXILIARY_MODULE} auxiliary may be"
                         f" loaded; also loaded {[(p, associated[p]) for p in unexpected]}")
    return rows


def _authenticated_pynvc(module: Any, distribution: Any) -> dict[str, Any]:
    """Authenticate the imported module and the extension this interpreter actually loaded.

    ``loaded_path`` comes from the imported ``_PyNvVideoCodec.__file__``, never from metadata, so
    it proves which binary really ran rather than which one merely exists on disk."""
    distribution_version = str(distribution.version)
    module_version = str(getattr(module, "__version__", "") or "")
    if distribution_version != EXPECTED_VERSION or module_version != EXPECTED_VERSION:
        raise _NotReady(
            f"{DIST_NAME} distribution and imported module must both be exact public"
            f" {EXPECTED_VERSION}; observed {distribution_version!r} and {module_version!r}")
    module_file = getattr(module, "__file__", None)
    extension_file = getattr(getattr(module, "_PyNvVideoCodec", None), "__file__", None)
    if not module_file or not extension_file:
        raise ValueError(f"imported {DIST_NAME} package/extension paths are unavailable")
    module_identity = _wheel_owned_identity(distribution, module_file, label="module")
    _require_record_consistency(module_identity, label="module", version=distribution_version)
    extension_identity = _wheel_owned_identity(distribution, extension_file, label="extension")
    _require_record_consistency(extension_identity, label="extension",
                                version=distribution_version, allow_native_stale=True)
    auxiliaries = _authenticate_auxiliaries(distribution, module_identity, extension_identity)
    return {
        "status": "verified", "authentication": "local_wheel_record_ownership",
        # Lexical: exactly what was invoked. Resolved identity: what actually ran. A venv's
        # bin/python is a symlink, so the identity is always taken from the resolved target.
        "interpreter": sys.executable, "sys_prefix": str(Path(sys.prefix).resolve()),
        "interpreter_identity": file_identity(Path(sys.executable).resolve(strict=True),
                                              label="verification interpreter"),
        "version": distribution_version,
        "module": {**module_identity, "loaded_path": str(module_file), "version": module_version},
        "extension": {**extension_identity, "loaded_path": str(extension_file)},
        "auxiliary_extensions": auxiliaries}


# --- nvcodec-environment 1.2 consumption (required subset only) ---------------

def _require_dependencies(dependencies: Any, profile: str) -> None:
    """Require exactly the recorded dependency rows the selected profile's samples need.

    The smoke pair -- basic/encode.py in ``-m cpu`` mode and advanced/decode_perf.py -- never
    reaches the unguarded module-level ``import torch`` in samples/utils/Utils.py, so a missing
    Torch is a complete ready state under ``pynvc-smoke`` and stays a hard block under
    ``full-samples``, whose advanced/decode.py route imports that helper directly."""
    if not isinstance(dependencies, dict):
        raise ValueError("pynvc.dependencies must be a JSON object")
    for name in profile_dependencies(profile):
        rule, row = _DEPENDENCY_RULES[name], dependencies.get(name)
        if (not isinstance(row, dict) or row.get("status") != "installed"
                or row.get("ready") is not True):
            raise _NotReady(f"environment does not show {profile} official-sample dependency"
                            f" {name} installed and ready")
        for field, expected in rule.items():
            if field == "minimum" and not _version_at_least(row.get("version"), expected):
                raise _NotReady(f"official sample requires {name} >= "
                                + ".".join(str(part) for part in expected))
            if field != "minimum" and row.get(field) != expected:
                raise _NotReady(f"official sample requires {name} {field}={expected!r},"
                                f" environment reports {row.get(field)!r}")


def _platform_compatibility(jetson_linux: Any) -> dict[str, Any]:
    """Recompute the documented Jetson Linux minimum from the environment release line."""
    fields = _require_fields(jetson_linux, ("version", "release_line"),
                             label="platform.jetson_linux")
    minimum = ".".join(str(part) for part in MINIMUM_JETSON_LINUX)
    release = parse_jetson_release(str(fields["release_line"]))
    if not _version_at_least(release["version"], MINIMUM_JETSON_LINUX):
        raise _NotReady(
            f"official {DIST_NAME} sample verification requires Jetson Linux {minimum} or newer;"
            f" the environment release line resolves to {release['version']!r}")
    return {"minimum_version": minimum, "observed_version": release["version"], "supported": True,
            "release_line": release["release_line"], "jetson_linux": str(fields["version"]),
            "evidence": "recomputed_from_environment_release_line"}


def _validate_environment(data: Any, *, gpu: int, profile: str = FULL_PROFILE) -> dict[str, Any]:
    """Validate a live schema-1.x environment and normalize its established PyNv identity.

    Only the pynvc surface is inspected: a malformed or absent native surface can never block this
    one, because the two surfaces remain structurally independent."""
    if not isinstance(data, dict) or data.get("kind") != ENVIRONMENT_KIND:
        raise ValueError(f"a {ENVIRONMENT_KIND} artifact is required")
    schema_version = str(data.get("schema_version") or "")
    if schema_version.partition(".")[0] != ENVIRONMENT_MAJOR:
        raise ValueError(f"unsupported {ENVIRONMENT_KIND} schema major {schema_version!r}; this"
                         f" verifier consumes schema {ENVIRONMENT_MAJOR}.x only")
    if data.get("mode") != "live":
        raise ValueError(f"a live {ENVIRONMENT_KIND} artifact is required")
    if data.get("selected_gpu") != gpu:
        raise ValueError("environment selected_gpu does not match the requested smoke GPU")
    pynvc = data.get("pynvc")
    if not isinstance(pynvc, dict) or pynvc.get("imported") is not True:
        raise _NotReady(f"environment does not show an imported {DIST_NAME} surface")
    identity = _require_fields(
        pynvc.get("identity"),
        ("status", "interpreter", "sys_prefix", "module", "extension"),
        label="pynvc.identity")
    if identity["status"] != "verified":
        raise _NotReady("environment does not carry a verified PyNvVideoCodec identity")
    module = _require_fields(identity["module"], ("version", "path"),
                             label="pynvc.identity.module")
    extension = _require_fields(identity["extension"], ("path",),
                                label="pynvc.identity.extension")
    record = extension.get("record") if isinstance(extension.get("record"), dict) else {}
    extension = {**extension, "sha256": extension.get("sha256") or record.get("sha256"),
                 "loaded_path": extension.get("loaded_path") or extension.get("path")}
    _require_fields(extension, ("path", "sha256", "loaded_path"),
                    label="pynvc.identity.extension")
    python = data.get("installation", {}).get("python", {})
    identities = python.get("interpreter_identities", {}) if isinstance(python, dict) else {}
    interpreter_identity = identity.get("interpreter_identity")
    if not isinstance(interpreter_identity, dict):
        interpreter_identity = identities.get("running") if isinstance(identities, dict) else None
    _require_fields(interpreter_identity, ("path", "sha256"),
                    label="installation.python.interpreter_identities.running")
    packages = python.get("packages", {}) if isinstance(python, dict) else {}
    dependencies = {}
    for name in profile_dependencies(profile):
        row = packages.get(name, {}) if isinstance(packages, dict) else {}
        dependencies[name] = {**row, "ready": bool(
            row.get("status") == "installed" and row.get("requirement_satisfied") is True)}
    _require_dependencies(dependencies, profile)
    return {
        "version": identity.get("version") or pynvc.get("distribution_version"),
        "interpreter": identity["interpreter"], "sys_prefix": identity["sys_prefix"],
        "interpreter_identity": interpreter_identity, "module": module, "extension": extension,
        "auxiliary_extensions": pynvc.get("auxiliary_extensions"),
        "dependencies": dependencies}


def _auxiliary_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    """Normalize a recorded auxiliary list. Absent means "none observed", which keeps controlled
    fixtures usable; an explicitly present value must be a list carrying at most one complete
    ``driver_version_check`` row, so a helper can never be smuggled in as an unnamed role."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 1:
        raise ValueError(f"{label} must be a list with at most one {AUXILIARY_ROLE} row")
    rows = [_require_fields(row, _AUXILIARY_FIELDS, label=f"{label}[0]") for row in value]
    if any(row["role"] != AUXILIARY_ROLE for row in rows):
        raise ValueError(f"{label} may carry only the {AUXILIARY_ROLE} role")
    return rows


def _require_surface_binding(surface: dict[str, Any], live: dict[str, Any],
                             *, require_running: bool = True) -> None:
    """Require exactly one loaded extension matching ``extension.loaded_path`` and
    ``module.{version,path}`` matching the imported module. ``loaded_path != path`` fails closed.

    The auxiliary lists are compared the same way, and a count disagreement fails: if either the
    environment or this process observed a compiled helper the other did not, neither side's
    evidence describes what actually ran."""
    extension, module = surface["extension"], surface["module"]
    recorded, interpreter = surface["interpreter_identity"], live["interpreter_identity"]
    recorded_aux = surface.get("auxiliary_extensions")
    aux = _auxiliary_rows(recorded_aux, label="pynvc.auxiliary_extensions")
    live_aux = live["auxiliary_extensions"]
    auxiliary_checks = ((len(aux) == len(live_aux),
                         "auxiliary_extensions disagree: one side observed a compiled helper the"
                         " other did not"),
                        (all(row["role"] == item["role"] and row["sha256"] == item["sha256"]
                             and _same_path(row["path"], item["path"])
                             and _same_path(row["loaded_path"], item["loaded_path"])
                             for row, item in zip(aux, live_aux)),
                         "auxiliary_extensions do not match the helper this interpreter actually"
                         " loaded")) if recorded_aux is not None else ()
    running_checks = (((_same_path(surface["interpreter"], sys.executable),
                       "interpreter is not the interpreter running this verification"),)
                      if require_running else ())
    failed = [message for passed, message in (
        *auxiliary_checks,
        (_same_path(extension["loaded_path"], extension["path"]),
         "extension.loaded_path does not equal extension.path"),
        (_same_path(extension["loaded_path"], live["extension"]["loaded_path"]),
         "extension.loaded_path is not the extension this interpreter loaded"),
        (extension["sha256"] == live["extension"]["sha256"],
         "extension.sha256 does not match the loaded extension"),
        (str(module["version"]) == live["module"]["version"],
         "module.version does not match the imported module"),
        (_same_path(module["path"], live["module"]["loaded_path"]),
         "module.path does not match the imported module"),
        (str(surface["version"]) == live["version"],
         "version does not match the installed distribution"),
        (str(surface["sys_prefix"]) == live["sys_prefix"], "sys_prefix does not match"),
        *running_checks,
        (recorded.get("path") == interpreter["path"]
         and recorded.get("sha256") == interpreter["sha256"],
         "interpreter_identity does not match this interpreter on disk")) if not passed]
    if failed:
        raise ValueError("environment pynvc mismatch: " + "; ".join(failed))


# --- Validated PyNvVideoCodec venv registry -----------------------------------

class RegistryNotReady(Exception):
    """The validated-venv registry is absent, stale, or fails to re-authenticate."""

    def __init__(self, reason: str, next_action: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.next_action = next_action


def _json_snapshot(path: Any, *, label: str) -> tuple[dict[str, Any], Any]:
    """Parse the exact bytes hashed as identity, then require the path to retain that identity."""
    raw, size_bytes, resolved = _read_regular(path, label=label)
    identity = {"path": str(resolved), "size_bytes": size_bytes, "sha256": sha256_bytes(raw)}
    data = read_json(raw)
    if file_identity(resolved, label=label) != identity:
        raise ValueError(f"{label} changed while it was authenticated")
    return identity, data


def _recorded_profile(data: Any, *, label: str) -> str:
    """Read one recorded verification profile, treating absent evidence as ``full-samples``.

    Evidence written before the profile split records no profile at all, and reading that silence
    as the Torch-requiring full profile is the fail-closed direction: it can only refuse a
    registry a stricter proof would have accepted, never admit a weaker one."""
    profile = str(data.get("verification_profile") or FULL_PROFILE
                  ) if isinstance(data, dict) else FULL_PROFILE
    if profile not in VERIFICATION_PROFILES:
        raise RegistryNotReady("profile_unrecognized",
                               f"{label} records verification profile {profile!r}, which this"
                               f" helper does not know; re-run setup and re-register")
    return profile


def _verified_pynvc(path: Path, *, expected: Any = None) -> tuple[
        dict[str, Any], dict[str, Any], str]:
    """Re-authenticate verification -> environment -> PyNv surface before registry use."""
    try:
        identity, data = _json_snapshot(path, label="pynvc verification artifact")
    except (OSError, ValueError) as exc:
        raise RegistryNotReady("artifact_unreadable", f"re-run setup: {exc}") from exc
    if expected is not None and identity != expected:
        raise RegistryNotReady("artifact_identity_changed", "re-run and re-register setup")
    profile = _recorded_profile(data, label="the pynvc verification artifact")
    try:
        surface = _require_fields(data.get("pynvc") if isinstance(data, dict) else None,
                                  _PYNVC_FIELDS, label="pynvc verification surface")
        operations = _require_fields(data.get("hardware_operations"),
                                     ("encode", "decode"), label="hardware_operations")
        if (data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != RESULT_KIND
                or data.get("ready") is not True or data.get("status") != "operation_verified"
                or data.get("software_fallback") is not False
                or any(not isinstance(operations[name], dict)
                       or operations[name].get("status") != "operation_verified"
                       for name in ("encode", "decode"))):
            raise ValueError("verification result is not a live no-fallback encode/decode proof")
        bound = _require_fields(
            data.get("environment"),
            ("path", "sha256", "canonical_sha256", "kind", "schema_version", "mode",
             "selected_gpu", "interpreter", "official_sample_dependencies"),
            label="bound environment")
        if not Path(str(bound["path"])).is_absolute():
            raise ValueError("bound environment path must be absolute")
        environment_identity, environment = _json_snapshot(
            bound["path"], label="bound environment")
        if not isinstance(environment, dict):
            raise ValueError("bound environment must be a JSON object")
        if (environment_identity["path"] != bound["path"]
                or environment_identity["sha256"] != bound["sha256"]
                or canonical_json_sha256(environment) != bound["canonical_sha256"]):
            raise ValueError("bound environment file identity or canonical hash changed")
        gpu = environment.get("selected_gpu") if isinstance(environment, dict) else None
        if (bound["kind"] != ENVIRONMENT_KIND or environment.get("kind") != ENVIRONMENT_KIND
                or bound["schema_version"] != "1.2"
                or environment.get("schema_version") != "1.2"
                or bound["mode"] != "live" or environment.get("mode") != "live"
                or not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0
                or bound["selected_gpu"] != gpu):
            raise ValueError("bound environment kind/schema/mode/GPU is not the verified live run")
        if _recorded_profile(bound, label="the bound environment record") != profile:
            raise RegistryNotReady(
                "profile_binding_mismatch",
                "the verification artifact and its bound environment record different"
                f" verification profiles; re-run verify_pynvc_sample.py --profile {profile}"
                " --register-current against a freshly probed environment")
        environment_surface = _validate_environment(environment, gpu=gpu, profile=profile)
        _require_surface_binding(environment_surface, surface, require_running=False)
        python = _require_fields(
            environment.get("installation", {}).get("python"),
            ("executable", "interpreter_identities"), label="environment Python installation")
        identities = _require_fields(
            python["interpreter_identities"], ("running",),
            label="environment Python interpreter identities")
        legacy_venv = python.get("venv") if isinstance(python.get("venv"), dict) else {}
        prefix = python.get("sys_prefix") or legacy_venv.get("prefix")
        python_path, prefix_path = Path(str(python["executable"])), Path(str(prefix))
        # The recorded dependency set is compared inside the profile that selected it, and its own
        # failure is separated out: an artifact proven under one profile is not defective, it is
        # simply not the environment another profile asks for, so the caller is told which venv to
        # provision instead of a generic "not verified".
        if bound["official_sample_dependencies"] != environment_surface["dependencies"]:
            raise RegistryNotReady(
                "profile_dependency_mismatch",
                f"the registered environment does not carry the {profile} dependency set"
                f" {sorted(profile_dependencies(profile))}; provision a new {profile} venv"
                " (plan_install.py --profile) and re-run verify_pynvc_sample.py"
                f" --profile {profile} --register-current")
        if (not isinstance(prefix, str) or not Path(prefix).is_absolute()
                or not python_path.is_absolute() or python_path.parent.parent != prefix_path
                or Path(str(environment_surface["interpreter"])).parent.parent != prefix_path
                or not _same_path(python["executable"], environment_surface["interpreter"])
                or not _same_path(prefix, environment_surface["sys_prefix"])
                or identities["running"] != environment_surface["interpreter_identity"]):
            raise ValueError("environment Python/PyNv binding disagrees with verification")
        if (not Path(str(bound["interpreter"])).is_absolute()
                or not _same_path(bound["interpreter"], surface["interpreter"])):
            raise ValueError("bound environment interpreter disagrees with verification")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RegistryNotReady(
            "artifact_not_verified", "re-run verification until encode and decode both pass"
        ) from exc
    _live_interpreter(surface)
    return identity, surface, profile


def _live_interpreter(surface: dict[str, Any]) -> None:
    """Re-hash the exact interpreter bound inside its recorded environment prefix."""
    recorded = surface["interpreter_identity"]
    try:
        interpreter, prefix = Path(str(surface["interpreter"])), Path(str(surface["sys_prefix"]))
        lexical = interpreter.resolve(strict=True)
        live = file_identity(recorded["path"], label="registered interpreter")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RegistryNotReady("interpreter_missing", f"re-run setup: {exc}") from exc
    if (not isinstance(recorded, dict) or not interpreter.is_absolute()
            or not prefix.is_absolute() or interpreter.parent.parent != prefix
            or str(lexical) != recorded.get("path") or live != recorded):
        raise RegistryNotReady("interpreter_changed", "re-run and re-register setup")


def publish_pynvc_registry(verification_artifact: Any) -> Path:
    """Atomically bind one fully re-authenticated operation proof to the fixed registry."""
    identity, surface, profile = _verified_pynvc(Path(verification_artifact))
    directory = _symlink_free(_REGISTRY_PATH.parent, label="pynvc registry directory")
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    staged, target = (directory / f".tmp-current-pynvc-{os.getpid()}.json",
                      directory / _REGISTRY_PATH.name)
    staged.unlink(missing_ok=True)
    try:
        os.replace(write_new_json(staged, {
            "schema": _REGISTRY_SCHEMA, "verification_artifact": identity, "pynvc": surface,
            "verification_profile": profile, "published_utc": utc_now()}), target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def load_pynvc_registry() -> dict[str, Any]:
    """Re-authenticate registry, proof, environment, surface and interpreter; never scan."""
    try:
        _identity, registry = _json_snapshot(_REGISTRY_PATH, label="pynvc registry")
    except FileNotFoundError as exc:
        raise RegistryNotReady("registry_absent", "run setup and register the validated venv"
                               " (verify_pynvc_sample.py --register-current)") from exc
    except (OSError, ValueError) as exc:
        raise RegistryNotReady("registry_unreadable", f"re-register setup: {exc}") from exc
    bound = registry.get("verification_artifact") if isinstance(registry, dict) else None
    if (not isinstance(bound, dict) or registry.get("schema") != _REGISTRY_SCHEMA
            or not isinstance(bound.get("path"), str) or not Path(bound["path"]).is_absolute()):
        raise RegistryNotReady("registry_binding_missing", "re-register setup")
    identity, surface, profile = _verified_pynvc(Path(bound["path"]), expected=bound)
    recorded = surface["interpreter_identity"]
    if (surface != registry.get("pynvc") or not isinstance(recorded, dict)
            or _recorded_profile(registry, label="the pynvc registry") != profile
            or not isinstance(recorded.get("path"), str)):
        raise RegistryNotReady("registry_surface_stale", "re-run and re-register setup")
    # The published profile travels with the surface so a consumer routes on the proof that was
    # actually run: the smoke pair proves frame production, not the byte-exact raw decode.
    return {"verification_artifact": identity, "verification_profile": profile, **surface}


# --- Wheel-owned official sample payload --------------------------------------

def _sample_files(distribution: Any) -> tuple[Path, dict[str, Path]]:
    """Locate the official encode/decode pair, its helpers and its config in this install."""
    prefix = Path(sys.prefix).resolve()
    for entry in distribution.files or []:
        if not str(entry).replace("\\", "/").endswith("samples/basic/encode.py"):
            continue
        located = Path(distribution.locate_file(entry))
        if located.is_symlink() or not located.is_file():
            continue
        root = located.resolve().parents[1]
        if not root.is_relative_to(prefix) or not all(
                (root / relative).is_file() for relative in SAMPLE_PATHS.values()):
            continue
        files = {name: (root / relative).resolve() for name, relative in SAMPLE_PATHS.items()}
        for parent in ("basic", "advanced"):
            config = root / parent / "encode_config.json"
            if config.is_file() and not config.is_symlink():
                return root, files | {"config": config.resolve()}
        raise FileNotFoundError("the installed sample payload has no encode_config.json")
    raise FileNotFoundError(
        f"installed {DIST_NAME} distribution does not contain the official encode/decode samples")


def _authenticated_samples(distribution: Any, files: dict[str, Path]) -> dict[str, Any]:
    """Every sample, helper and config a child reads needs its own verified RECORD row.

    ``allow_native_stale`` is deliberately absent: the stale-row exception covers the native
    extension only and can never excuse a sample payload mismatch."""
    records = {}
    for name, path in sorted(files.items()):
        identity = _wheel_owned_identity(distribution, path, label=f"official sample {name}")
        _require_record_consistency(identity, label=f"official sample {name}",
                                    version=str(distribution.version))
        records[name] = identity
    return records


def _run_child(argv: list[str], *, cwd: Path, timeout: int, prefix: Path) -> dict[str, Any]:
    started_at, monotonic_start = utc_now(), time.monotonic()
    try:
        completed = run_command(argv, cwd=cwd, timeout=timeout)
        streams = (completed.stdout, completed.stderr)
        exit_code, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired as exc:
        streams, exit_code, timed_out = (exc.stdout, exc.stderr), None, True
    raw = [value if isinstance(value, bytes) else str(value or "").encode() for value in streams]
    return {
        "argv": argv, "cwd": str(cwd), "exit_code": exit_code, "timed_out": timed_out,
        "stdout": raw[0].decode("utf-8", errors="replace"),
        "stderr": raw[1].decode("utf-8", errors="replace"),
        "started_at": started_at, "ended_at": utc_now(),
        "duration_seconds": time.monotonic() - monotonic_start,
        "streams": {name: {"path": str(write_new_bytes(
            prefix.with_name(f"{prefix.name}.{name}.log"), data)), "size_bytes": len(data),
            "sha256": sha256_bytes(data)} for name, data in zip(("stdout", "stderr"), raw)}}


def _score(child: dict[str, Any], *, label: str, frames: int, timeout: int, matches: list[Any],
           reported_frames: int | None, failure_pattern: re.Pattern[str],
           checks: list[tuple[bool, str]], reasons: list[str]) -> tuple[dict[str, Any], bool]:
    """Score one official operation. A clean child exit alone is explicitly insufficient: the exact
    frame marker, the operation-specific proofs and the absence of any known failure marker are
    each mandatory, and every failed proof is reported verbatim."""
    marker = failure_pattern.search(child["stdout"] + "\n" + child["stderr"])
    proofs = [(len(matches) == 1 and reported_frames == frames,
               f"official {label} sample did not report exactly {frames} {label}d frames"),
              *checks,
              (marker is None, "official {} sample reported failure marker: {}".format(
                  label, marker.group(0).strip() if marker else None))]
    if child["timed_out"]:
        reasons.append(f"official {label} sample timed out after {timeout} seconds")
    elif child["exit_code"] != 0:
        reasons.append(f"official {label} sample exited {child['exit_code']}")
    reasons.extend(reason for passed, reason in proofs if not passed)
    success = not child["timed_out"] and child["exit_code"] == 0 and all(p for p, _ in proofs)
    return {"status": "operation_verified" if success else "operation_failed",
            "command": child["argv"], "cwd": child["cwd"], "exit_code": child["exit_code"],
            "timed_out": child["timed_out"], "reported_frames": reported_frames,
            "frame_marker_matches": len(matches), "stdout_tail": child["stdout"][-4000:],
            "stderr_tail": child["stderr"][-4000:], "streams": child["streams"]}, success


def _output_identity(path: Path, *, label: str) -> dict[str, Any] | None:
    return file_identity(path, label=label) if path.is_file() and not path.is_symlink() else None


def _perf_decode_argv(sample: Path, bitstream: Path, *, gpu: int, frames: int) -> list[str]:
    """Exactly the public 2.1.0 decode_perf.py options its own ``--help`` publishes, and no other.

    ``-d 1`` pins the wheel's current ``use_device_memory`` default explicitly so the smoke stays
    deterministic if upstream changes it, and ``-n 1``/``-m thread`` pin the single in-process
    worker the marker contract is written against. The sample has no ``-o``: it writes no raw
    output at all, so none is passed."""
    return [sys.executable, "-I", str(sample), "-i", str(bitstream), "-d", "1",
            "-f", str(frames), "-n", "1", "-m", "thread", "-g", str(gpu)]


def _perf_markers(frames: int) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Bind both anchored decode_perf.py markers to the exact requested frame count.

    The count comes from the ``-f`` operand rather than being any positive integer, and the
    per-worker marker matches only the "decoded requested" literal."""
    return (re.compile(rf"^(?:Thread|Process) .*: Successfully decoded requested {frames}"
                       r" frames\s*$", re.MULTILINE),
            re.compile(rf"^Total frames decoded:\s*{frames}\s*$", re.MULTILINE))


def _perf_decode_evidence(child: dict[str, Any], *, frames: int, timeout: int,
                          reasons: list[str]) -> tuple[dict[str, Any], bool]:
    """Score the bounded decode_perf.py smoke: frame production, proven by two exact markers.

    This route proves FRAME PRODUCTION ONLY. decode_perf.py writes no raw frames, so no decoded
    artifact, size or path proof is claimed here; the byte-exact raw-output proof stays with
    advanced/decode.py under full-samples. Reported FPS and elapsed time are deliberately not
    asserted: the sample subtracts session-initialization time, so both can go negative here."""
    worker_re, total_re = _perf_markers(frames)
    text = child["stdout"] + "\n" + child["stderr"]
    workers, totals = worker_re.findall(text), total_re.findall(child["stdout"])
    operation, success = _score(
        child, label="decode", frames=frames, timeout=timeout, matches=totals,
        reported_frames=frames if len(totals) == 1 else None,
        failure_pattern=PERF_FAILURE_PATTERN, reasons=reasons,
        checks=[(len(workers) == 1, "official decode_perf sample did not report exactly one"
                 f" worker that successfully decoded the requested {frames} frames")])
    operation.update({"sample": SAMPLE_PATHS["decode_perf"], "worker_marker_matches": len(workers),
                      "decoded_output": None, "proves": "frame_production_only",
                      "raw_output_proof": "not_applicable: decode_perf.py writes no raw frames"})
    return operation, success


def _encode_evidence(child: dict[str, Any], *, frames: int, bitstream: Path, timeout: int,
                     reasons: list[str]) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    matches = ENCODE_SUCCESS_RE.findall(child["stdout"])
    identity = _output_identity(bitstream, label="H.264 bitstream output")
    size = identity["size_bytes"] if identity else 0
    operation, success = _score(
        child, label="encode", frames=frames, timeout=timeout, matches=matches,
        reported_frames=int(matches[0]) if len(matches) == 1 else None,
        failure_pattern=ENCODE_FAILURE_PATTERN, reasons=reasons,
        checks=[(size > 0, "official encode sample did not produce a non-empty H.264 bitstream")])
    operation["bitstream"] = {"path": str(bitstream), "size_bytes": size,
                              "sha256": identity["sha256"] if identity else None}
    return operation, success, identity


def _decode_evidence(child: dict[str, Any], *, frames: int, decoded: Path, expected_size: int,
                     timeout: int, reasons: list[str]) -> tuple[dict[str, Any], bool]:
    matches = DECODE_SUCCESS_RE.findall(child["stdout"])
    marker_path = matches[0][1] if len(matches) == 1 else None
    named = bool(marker_path) and decoded.is_file() and _same_path(marker_path, decoded)
    identity = _output_identity(decoded, label="decoded NV12 output")
    size = identity["size_bytes"] if identity else 0
    operation, success = _score(
        child, label="decode", frames=frames, timeout=timeout, matches=matches,
        reported_frames=int(matches[0][0]) if len(matches) == 1 else None,
        failure_pattern=DECODE_FAILURE_PATTERN, reasons=reasons,
        checks=[(named, "official decode success marker did not name the requested output"),
                (size == expected_size,
                 f"official decode sample produced {size} bytes; expected {expected_size}")])
    operation.update({"reported_output_path": marker_path, "reported_output_matches": named,
                      "decoded_output": {"path": str(decoded), "format": "NV12", "size_bytes": size,
                                         "expected_size_bytes": expected_size,
                                         "sha256": identity["sha256"] if identity else None}})
    return operation, success


def _fresh_path(path: Any, *, label: str) -> Path:
    """Resolve one attempt-owned path; a pre-existing path is refused so stale output can't pass."""
    requested = Path(path).expanduser().absolute()
    if any(ord(character) < 32 or ord(character) == 127 for character in str(requested)):
        raise ValueError(f"{label} path must not contain control characters")
    if not requested.parent.is_dir():
        raise FileNotFoundError(f"{label} parent directory does not exist: {requested.parent}")
    if os.path.lexists(requested):
        raise FileExistsError(
            f"{label} already exists; every attempt and retry requires a fresh path: {requested}")
    return requested.parent.resolve(strict=True) / requested.name


def _prepare_paths(*, work_dir: Path, input_path: Path | None, bitstream_path: Path | None,
                   decoded_path: Path | None, gpu: int, width: int, height: int, frames: int,
                   timeout: int) -> dict[str, Any]:
    """Validate the request tuple, then create the fresh directory and paths this attempt owns."""
    for name, value in (("GPU index", gpu), ("frame count", frames), ("timeout", timeout)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if frames <= 0 or timeout <= 0:
        raise ValueError("frame count and timeout must be positive")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("NV12 smoke dimensions must be positive and even")
    work = _fresh_path(work_dir, label="verification work directory")
    work.mkdir(mode=0o700)
    source = (write_new_bytes(work / f"pynvc-smoke-{width}x{height}.nv12",
                              (bytes([16]) * (width * height)
                               + bytes([128]) * (width * height // 2)) * frames)
              if input_path is None else Path(input_path).expanduser().absolute())
    return {"work": work, "source": source, "generated_input": input_path is None,
            "frame_bytes": width * height * 3 // 2,
            "bitstream": _fresh_path(bitstream_path or work / f"pynvc-sample-{width}x{height}.h264",
                                     label="bitstream output"),
            "decoded": _fresh_path(decoded_path or work / f"pynvc-decode-{width}x{height}.nv12",
                                   label="decoded output")}


def verify(*, environment_path: Path, work_dir: Path, gpu: int, width: int, height: int,
           frames: int, input_path: Path | None, bitstream_path: Path | None,
           decoded_path: Path | None, timeout: int,
           profile: str = SMOKE_PROFILE) -> dict[str, Any]:
    """Run the official encode/decode pair the selected profile owns and return its result.

    The encode half is the same wheel-owned basic/encode.py ``-m cpu`` proof in either profile;
    the decode half is the profile's own route -- the bounded one-frame advanced/decode_perf.py
    frame-production smoke, or advanced/decode.py's byte-exact raw-output proof."""
    if profile not in VERIFICATION_PROFILES:
        raise ValueError(f"verification profile must be one of {list(VERIFICATION_PROFILES)}")
    _validate_smoke_request(profile, width, height, frames, decoded_path)
    require_isolated()
    distribution = importlib.metadata.distribution(DIST_NAME)
    live = _authenticated_pynvc(importlib.import_module(DIST_NAME), distribution)
    identity = file_identity(environment_path, label="environment artifact")
    # read_json() parses a str/bytes source as the document itself, so the resolved path this was
    # just hashed from must be handed over as a Path to be read as a file.
    environment = read_json(Path(identity["path"]))
    surface = _validate_environment(environment, gpu=gpu, profile=profile)
    _require_surface_binding(surface, live)
    compatibility = _platform_compatibility(_require_fields(
        environment.get("platform"), ("jetson_linux",), label="platform")["jetson_linux"])
    samples, files = _sample_files(distribution)
    records = _authenticated_samples(distribution, files)
    paths = _prepare_paths(work_dir=work_dir, input_path=input_path, bitstream_path=bitstream_path,
                           decoded_path=decoded_path, gpu=gpu, width=width, height=height,
                           frames=frames, timeout=timeout)
    expected_size = paths["frame_bytes"] * frames
    source = file_identity(paths["source"], label="raw NV12 input")
    if source["size_bytes"] < expected_size:
        raise ValueError("raw NV12 input does not contain the requested number of complete frames")

    reasons: list[str] = []
    encode_operation, encode_success, bitstream_identity = _encode_evidence(
        _run_child([sys.executable, "-I", str(files["encode"]), "-i", source["path"], "-o",
                    str(paths["bitstream"]), "-s", f"{width}x{height}", "-m", "cpu", "-if", "NV12",
                    "-c", "h264", "-f", str(frames), "-g", str(gpu), "-json", str(files["config"])],
                   cwd=samples, timeout=timeout, prefix=paths["work"] / "official-encode"),
        frames=frames, bitstream=paths["bitstream"], timeout=timeout, reasons=reasons)
    decode_operation: dict[str, Any] = {"status": "not_run",
                                        "reason": "official encode operation did not verify"}
    decode_success = False
    decode_sample = files[PROFILE_DECODE_SAMPLE[profile]]
    if encode_success and profile == SMOKE_PROFILE:
        decode_operation, decode_success = _perf_decode_evidence(
            _run_child(_perf_decode_argv(decode_sample, paths["bitstream"], gpu=gpu,
                                         frames=frames),
                       cwd=samples, timeout=timeout, prefix=paths["work"] / "official-decode"),
            frames=frames, timeout=timeout, reasons=reasons)
    elif encode_success:
        decode_operation, decode_success = _decode_evidence(
            _run_child([sys.executable, "-I", str(decode_sample), "-i", str(paths["bitstream"]),
                        "-o", str(paths["decoded"]), "-d", "1", "-g", str(gpu), "-f", str(frames)],
                       cwd=samples, timeout=timeout, prefix=paths["work"] / "official-decode"),
            frames=frames, decoded=paths["decoded"], expected_size=expected_size, timeout=timeout,
            reasons=reasons)

    # Terminal re-authentication covering both child launches: a sample payload, raw input,
    # environment, wheel or bitstream substitution anywhere in the window fails closed.
    for label, before, after in (
            ("official sample payload", records, _authenticated_samples(distribution, files)),
            ("raw NV12 input", source, file_identity(paths["source"], label="raw NV12 input")),
            ("environment artifact", identity,
             file_identity(identity["path"], label="environment artifact")),
            (f"{DIST_NAME} wheel identity", live, _authenticated_pynvc(
                sys.modules[DIST_NAME], importlib.metadata.distribution(DIST_NAME))),
            ("H.264 bitstream output", bitstream_identity,
             _output_identity(paths["bitstream"], label="H.264 bitstream output")
             if encode_success else bitstream_identity)):
        if before != after:
            raise ValueError(f"{label} changed during authenticated sample execution")

    success = encode_success and decode_success
    return _result(
        RESULT_KIND, "operation_verified" if success else "operation_failed",
        generated_at=utc_now(), ready=success, software_fallback=False, pynvc=live,
        platform_compatibility=compatibility, verification_profile=profile,
        distribution={"name": DIST_NAME, "version": live["version"], "python": sys.executable},
        environment={"path": identity["path"], "sha256": identity["sha256"],
                     "canonical_sha256": canonical_json_sha256(environment),
                     "kind": environment.get("kind"), "mode": environment.get("mode"),
                     "schema_version": environment.get("schema_version"), "selected_gpu": gpu,
                     "interpreter": sys.executable, "verification_profile": profile,
                     "official_sample_dependencies": {
                         name: surface["dependencies"].get(name)
                         for name in profile_dependencies(profile)}},
        official_samples={"root": str(samples), "rows": records, "executed_decode":
                          SAMPLE_PATHS[PROFILE_DECODE_SAMPLE[profile]],
                          "provenance": f"{DIST_NAME} {EXPECTED_VERSION} wheel-owned"
                                        f" {profile} encode/decode samples"},
        input={"path": source["path"], "generated_fixture": paths["generated_input"],
               "format": "NV12", "width": width, "height": height, "frames": frames,
               "size_bytes": source["size_bytes"], "sha256": source["sha256"]},
        hardware_operation=encode_operation,
        hardware_operations={"encode": encode_operation, "decode": decode_operation},
        reasons=reasons)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run version-matched NVIDIA PyNvVideoCodec encode and decode samples.")
    parser.add_argument("--environment", type=Path, required=True,
                        help="Validated live nvcodec-environment 1.2 JSON")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd() / "nvcodec-pynvc-smoke",
                        help="Fresh, nonexistent operation directory; every retry needs a new one")
    for name, default in (("gpu", 0), ("width", SMOKE_WIDTH), ("height", SMOKE_HEIGHT),
                          ("frames", SMOKE_FRAMES), ("timeout", 120)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, description in (("input", "Optional existing raw NV12 input"),
                              ("bitstream", "H.264 bitstream output path"),
                              ("decoded", "Decoded NV12 output path"),
                              ("output", "Fresh, nonexistent JSON report path; a failed attempt"
                                         " consumes it and every retry requires a new one")):
        parser.add_argument(f"--{name}", type=Path, help=description)
    parser.add_argument("--profile", choices=VERIFICATION_PROFILES, default=SMOKE_PROFILE,
                        help="Official-sample verification profile. The default proves"
                             " basic/encode.py -m cpu plus a bounded one-frame"
                             " advanced/decode_perf.py and needs no Torch; full-samples runs"
                             " advanced/decode.py's byte-exact raw proof and does need Torch.")
    parser.add_argument("--register-current", action="store_true",
                        help="After a safely written, operation_verified live result, atomically"
                             " publish the fixed validated-venv registry"
                             " ($HOME/.local/state/jetson-videosdk/current-pynvc.json). Requires"
                             " --output. A non-ready result leaves any prior registry unchanged.")
    args = parser.parse_args()
    if args.register_current and args.output is None:
        parser.error("--register-current requires --output")
    try:
        _validate_smoke_request(
            args.profile, args.width, args.height, args.frames, args.decoded
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is not None and os.path.lexists(args.output):
        print(_render(_result(ERROR_KIND, "error", error=(
            "verification report output already exists; every attempt and retry requires a fresh"
            f" --output path: {args.output}"))))
        return 3

    try:
        result = verify(environment_path=args.environment, work_dir=args.work_dir, gpu=args.gpu,
                        width=args.width, height=args.height, frames=args.frames,
                        input_path=args.input, bitstream_path=args.bitstream,
                        decoded_path=args.decoded, timeout=args.timeout, profile=args.profile)
        exit_code = 0 if result["ready"] else 2
    # A valid-but-not-yet-ready surface blocks only pynvc and leaves native actionable. Import
    # failure and bad input are helper errors at rc 3, preserving the established result contract.
    except (_NotReady, subprocess.TimeoutExpired) as exc:
        result, exit_code = _result(RESULT_KIND, "unknown", ready=False, software_fallback=False,
                                    reasons=[str(exc)]), 2
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result, exit_code = _result(ERROR_KIND, "error", error=str(exc)), 3

    if args.output is not None:
        try:
            write_new_json(args.output, result)
        except (OSError, ValueError) as exc:
            print(_render(_result(ERROR_KIND, "error", error=(
                f"verification report output could not be written safely: {exc}"))))
            return 3
    print(_render(result))
    if args.register_current and result.get("ready") is True:
        # Publication is a separate step AFTER the verification evidence is safely written. Its
        # failure preserves the written evidence and leaves any prior registry unchanged.
        try:
            publish_pynvc_registry(args.output)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"registration_failed: {exc}", file=sys.stderr)
            return 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
