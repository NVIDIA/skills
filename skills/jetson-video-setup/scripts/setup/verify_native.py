#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run the package-installed NVIDIA Video Codec SDK samples as an operation proof.

This helper is the *operation* authority for the native surface: it builds the official
``AppEncCuda`` and ``AppDec`` samples from the package-owned ``Samples`` tree, encodes one fresh
NV12 frame, and -- only after AppEncCuda's own success marker has been proven by this run --
decodes that fresh bitstream with AppDec. An API query is never a substitute for either step.

Only ``installation.native_sdk`` of the ``nvcodec-environment`` 1.2 artifact is read, and only
its required subset is validated, so a malformed or absent ``pynvc`` block can neither block nor
be blocked by this proof. Invoked directly as ``python3 -I .../verify_native.py``: it owns its
own CLI, never activates a virtual environment, and never imports another skill.
"""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_SETUP_DIR = Path(__file__).resolve().parent
if str(_SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(_SETUP_DIR))

# pylint: disable=wrong-import-position
from setup_contract import (  # noqa: E402
    canonical_json_sha256, file_identity, parse_jetson_release, read_json, require_isolated,
    run_command, sha256_bytes, system_executable, utc_now, write_new_bytes, write_new_json,
)

# pylint: enable=wrong-import-position

SCHEMA_VERSION = "1.5"
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_MAJOR = "1"
NATIVE_PACKAGE = "nvidia-video-codec-sdk"

# The official, non-negotiable operation markers. Each is anchored to a whole physical line so
# no embedded or trailing text can satisfy a count: "...encoded: 11" and "...encoded: 1x" are
# both rejected as one encoded frame.
_ENCODED_RE = re.compile(r"^[^\S\r\n]*Total frames encoded:[^\S\r\n]*(\d+)[^\S\r\n]*\r?$",
                         re.MULTILINE)
_BITSTREAM_RE = re.compile(r"^Bitstream saved in file\s+(.+?)\s*$", re.MULTILINE)
_DECODED_RE = re.compile(r"^[^\S\r\n]*Total frame decoded:[^\S\r\n]*(\d+)[^\S\r\n]*\r?$",
                         re.MULTILINE)
# The smoke is one fixed 640x360 NV12 frame. --width/--height carry the requested geometry and
# default to it; 345,600 stays the authoritative payload size for the fixture and the decoded
# output, so a request outside the fixed smoke geometry is refused rather than silently resized.
_SMOKE_GEOMETRY = (640, 360)
_SMOKE_NV12_BYTES = 345600
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_RE = re.compile(r"\d+(?:\.\d+)*(?:[.+~-][0-9A-Za-z.+~-]*)?")
# Private replacement for the removed public-release-family predicate: stable Debian revisions
# of the public 13.0.x family only, never prerelease or "really" family redirects.
_PUBLIC_SDK_FAMILY = "13.0"
_PUBLIC_SDK_RE = re.compile(r"13\.0(?:\.\d+)*(?:\+[0-9A-Za-z.]+)?(?:-[0-9A-Za-z.+]+)?")
_FAILURE_PATTERNS = (
    ("nvenc_error", re.compile(r"(?im)^[^\S\r\n]*(?:\[[^]\r\n]*\][^\S\r\n]*)*"
                               r"(?:NV_ENC_ERR_[A-Z0-9_]+\b|NVENC error\b)")),
    ("cuda_error", re.compile(r"(?im)^[^\S\r\n]*(?:\[[^]\r\n]*\][^\S\r\n]*)*"
                              r"(?:CUDA_ERROR_[A-Z0-9_]+\b|CudaCheckError\(\) failed\b|"
                              r"CUDA (?:driver |runtime )?(?:API )?error\b)")),
    ("error", re.compile(r"(?im)^\s*(?:Traceback \(most recent call last\)|\[(?:FATAL|ERROR)\s*\]|"
                         r"(?:uncaught )?exception\b|error\b|failed\b|failure\b|"
                         r"[a-z_][a-z0-9_.:~]*(?:\([^\r\n]*\))?\s+returned error\b)")),
)
# installation.native_sdk.tools carries exactly these five entries; tools.generator.name is the
# CMake generator name and nothing outside _GENERATOR_NAMES is accepted.
_REQUIRED_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
_GENERATOR_NAMES = ("Ninja", "Unix Makefiles")
_SAMPLES = {"encoder": ("AppEncode", "AppEncCuda"), "decoder": ("AppDecode", "AppDec")}
_DRIVER_LIBRARIES = {"encoder": ("libcuda.so.1", "libnvidia-encode.so.1"),
                     "decoder": ("libcuda.so.1", "libnvcuvid.so.1")}
# Static AppDec module -> Debian package repair table: a rendering aid for the probe's own
# unresolved_modules, never a live query. The deterministic repair hint is the payload.
_APPDEC_PACKAGES = {"libavcodec": "libavcodec-dev", "libavformat": "libavformat-dev",
                    "libavutil": "libavutil-dev", "libswresample": "libswresample-dev"}
# The two official aggregate reports nvcodec-native-verification 1.5 publishes beside the
# operation proof. They are RAW sample-reported rows: never an API support claim, never operation
# evidence, and never a readiness gate. Each runs between two identity checks of its own binary.
_AGGREGATE_ROUTES = {"encoder": ("AppEncCuda", "-ec"), "decoder": ("AppDec", "-dc")}
_ENCODER_CAPS_RE = re.compile(r"(?m)^Encoder Capability Summary[ \t]*\r?$")
_CAPS_GPU_RE = re.compile(r"(?m)^GPU\s+(\d+)\s+-\s+(.+?)[ \t]*\r?$")
_CAPS_CODEC_RE = re.compile(r"(?m)^[ \t]*(H264|HEVC|AV1):[ \t]*(yes|no)[ \t]*\r?$")
_ENCODER_CAPS_HINT = "For detailed information about all capabilities, use -ec-detail"
_DECODER_CAPS_RE = re.compile(r"(?m)^GPU Decoder Capabilities[ \t]*\r?$")
_DECODER_SUMMARY_RE = re.compile(r"(?m)^Codec Support Summary[ \t]*\r?$")
_DECODER_CAPS_HINT = "For detailed information about all capabilities, use -dc-detail"
_ENCODER_SECTIONS = ("=== CODEC SUPPORT SUMMARY ===", "=== ENCODER CAPABILITIES SUMMARY TABLE ===")
_AGGREGATE_SCOPE = {
    "encoder": ("raw sample-reported values only; nvEncGetEncodeCaps status is unchecked by the"
                " sample, so rows are neither authoritative unsupported nor operation evidence"),
    "decoder": ("raw sample-reported values only; this report is neither authoritative"
                " cuvidGetDecoderCaps evidence nor operation evidence")}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value and Path(value).is_absolute(),
             f"{label} must be an absolute path, got {value!r}")
    return Path(value)


def _sha256_path(path: Path) -> str:
    """Hash content through any symlink; file_identity deliberately refuses a symlinked leaf."""
    return sha256_bytes(path.read_bytes())


def _identity(path: Path, label: str) -> dict[str, Any] | None:
    return file_identity(path, label=label) if path.is_file() and not path.is_symlink() else None


def _fresh_directory(path: Path, *, label: str) -> Path:
    requested = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    _require(all(31 < ord(character) != 127 for character in str(requested)),
             f"{label} must not contain control characters")
    _require(requested.parent.is_dir(), f"{label} parent does not exist: {requested.parent}")
    resolved = requested.parent.resolve(strict=True) / requested.name
    _require(not os.path.lexists(requested) and not os.path.lexists(resolved),
             f"{label} must not already exist: {resolved}")
    resolved.mkdir(mode=0o700)
    return resolved


def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 300) -> dict[str, Any]:
    """Run one bounded child; a timeout becomes failure evidence rather than an exception."""
    try:
        done = run_command(argv, cwd=cwd, timeout=timeout)
        code, out, err, expired = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, out, err, expired = None, exc.stdout, exc.stderr, True
    return {"argv": list(argv), "cwd": str(cwd) if cwd else str(Path.cwd()), "exit_code": code,
            "timed_out": expired, "stdout": bytes(out or b"").decode("utf-8", errors="replace"),
            "stderr": bytes(err or b"").decode("utf-8", errors="replace")}


def _failure_markers(evidence: dict[str, Any]) -> list[str]:
    """Return stable labels for explicit sample or runtime failures in either stream."""
    combined = evidence["stdout"] + "\n" + evidence["stderr"]
    return [label for label, pattern in _FAILURE_PATTERNS if pattern.search(combined)]


def _tail(evidence: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    return {"argv": evidence["argv"], "exit_code": evidence["exit_code"],
            "timed_out": evidence["timed_out"], "stdout_tail": evidence["stdout"][-limit:],
            "stderr_tail": evidence["stderr"][-limit:]}


# --- nvcodec-environment 1.2: required subset of installation.native_sdk only -

def _version_token(value: Any, *, label: str) -> str:
    """Validate -- never merely read -- a version token."""
    text = value.strip() if isinstance(value, str) else ""
    _require(text and _VERSION_RE.fullmatch(text), f"{label} must be a version token: {value!r}")
    return text


def _public_sdk_release(value: Any) -> bool:
    normalized = value.strip().split(":", 1)[-1] if isinstance(value, str) else ""
    if not normalized or "~" in normalized or "really" in normalized.lower():
        return False
    return _PUBLIC_SDK_RE.fullmatch(normalized) is not None


def _tool(tools: Any, name: str) -> dict[str, Any]:
    """Validate one required tool entry and re-authenticate it live against its recorded hash.

    An *absent* tool -- no entry, or a path that is not an executable regular file -- returns an
    ``unavailable`` record naming the defect so the run still emits the complete ready=false
    result document. A present-but-corrupt entry is a malformed artifact and still raises.
    """
    label = f"installation.native_sdk.tools.{name}"
    record = tools.get(name) if isinstance(tools, dict) else None
    if not isinstance(record, dict):
        return {"status": "unavailable", "name": name,
                "reason": f"required build tool {name} is absent: {label} is missing"}
    path = _absolute(record.get("path"), label=f"{label}.path")
    digest = record.get("sha256")
    _require(isinstance(digest, str) and _SHA256_RE.fullmatch(digest),
             f"{label}.sha256 must be a sha-256 digest")
    entry = {"status": "verified", "path": str(path), "sha256": digest,
             "version": _version_token(record.get("version"), label=f"{label}.version")}
    if name == "generator":
        _require(record.get("name") in _GENERATOR_NAMES,
                 f"{label}.name must be the CMake generator name, one of "
                 f"{list(_GENERATOR_NAMES)}, got {record.get('name')!r}")
        entry["name"] = record["name"]
    if not (path.is_file() and os.access(path, os.X_OK)):
        return {"status": "unavailable", "name": name,
                "reason": f"required build tool {name} is absent: {path} is not an executable"
                          " regular file"}
    _require(_sha256_path(path) == digest,
             f"{label} changed on disk after the environment was produced")
    return entry


def _prerequisite_reason(build: Any) -> str:
    """Render deterministic repair guidance for incomplete AppDec build prerequisites."""
    unresolved = build.get("unresolved_modules") if isinstance(build, dict) else None
    modules = sorted(str(item) for item in unresolved) if isinstance(unresolved, list) else []
    packages = [_APPDEC_PACKAGES[name] for name in modules if name in _APPDEC_PACKAGES]
    repair = (f"install {', '.join(packages)} before CMake" if len(packages) == len(modules)
              else "read the CMake configure stderr tail for the unmapped module(s)")
    return ("installation.native_sdk.build_prerequisites are not complete"
            + (f": AppDec needs {', '.join(modules)}; {repair}." if modules else ""))


def _native_surface(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the required subset of ``installation.native_sdk``.

    Additive keys are ignored and no set equality is applied. ``pynvc`` is never read.
    """
    installation = data.get("installation")
    native = installation.get("native_sdk") if isinstance(installation, dict) else None
    _require(isinstance(native, dict), "environment has no installation.native_sdk block")
    _require(native.get("installed") is True, "installation.native_sdk.installed is not true")
    package = native.get("package") if isinstance(native.get("package"), dict) else {}
    _require(package.get("name") == NATIVE_PACKAGE and package.get("status") == "installed",
             f"installation.native_sdk.package does not show {NATIVE_PACKAGE} installed")
    version = _version_token(package.get("version"),
                             label="installation.native_sdk.package.version")
    _require(_public_sdk_release(version), f"installation.native_sdk.package.version {version!r}"
             f" is outside the public {_PUBLIC_SDK_FAMILY}.x Video Codec SDK release family")
    cuda = native.get("cuda") if isinstance(native.get("cuda"), dict) else {}
    _require(cuda.get("status") == "installed", "installation.native_sdk.cuda.status must be"
             f" 'installed', got {cuda.get('status')!r}")
    build = native.get("build_prerequisites")
    _require(isinstance(build, dict) and build.get("status") == "complete"
             and build.get("unresolved_modules") == [], _prerequisite_reason(build))
    return {"package": {"name": NATIVE_PACKAGE, "status": "installed", "version": version},
            "sdk_root": str(_absolute(native.get("sdk_root"),
                                      label="installation.native_sdk.sdk_root")),
            "cuda": {"status": "installed",
                     "version": _version_token(cuda.get("version"),
                                               label="installation.native_sdk.cuda.version"),
                     "root": str(_absolute(cuda.get("root"),
                                           label="installation.native_sdk.cuda.root"))},
            "build_prerequisites": {"status": "complete", "unresolved_modules": []},
            "tools": {name: _tool(native.get("tools"), name) for name in _REQUIRED_TOOLS}}


def _platform(data: dict[str, Any]) -> dict[str, Any]:
    """Cross-check platform release evidence. Minimum-version gating belongs to the probe."""
    block = data.get("platform") if isinstance(data.get("platform"), dict) else {}
    jetson = block.get("jetson_linux") if isinstance(block.get("jetson_linux"), dict) else {}
    line = jetson.get("release_line")
    _require(isinstance(line, str) and line and "\n" not in line and "\r" not in line,
             "platform.jetson_linux.release_line must be exactly one non-empty physical line")
    parsed = parse_jetson_release(line)
    _require(parsed["release_major"] is not None,
             "platform.jetson_linux.release_line carries no Jetson Linux release token")
    _require(parsed["version"] == jetson.get("version"),
             "platform.jetson_linux.version disagrees with its own release_line")
    return {"jetson_linux": parsed["version"], "machine": block.get("machine"),
            "release": parsed}


def _load_environment(path: Path, gpu: int) -> dict[str, Any]:
    identity = file_identity(path, label="nvcodec-environment")
    raw = Path(identity["path"]).read_bytes()
    _require(sha256_bytes(raw) == identity["sha256"],
             "nvcodec-environment changed while it was being read")
    data = read_json(raw)
    _require(isinstance(data, dict) and data.get("kind") == ENVIRONMENT_KIND,
             "input is not an nvcodec-environment artifact")
    schema = data.get("schema_version")
    _require(isinstance(schema, str) and schema, "environment schema_version is missing")
    _require(schema.split(".", 1)[0] == ENVIRONMENT_MAJOR,
             f"unsupported nvcodec-environment major version {schema!r}: native verification"
             f" consumes schema {ENVIRONMENT_MAJOR}.x only")
    _require(data.get("mode") == "live", "environment mode must be 'live'")
    _require(data.get("selected_gpu") == gpu,
             "environment selected_gpu does not match the requested --gpu")
    return {"path": identity["path"], "identity": identity, "schema_version": schema,
            "canonical_sha256": canonical_json_sha256(data), "selected_gpu": gpu,
            "platform": _platform(data), "native": _native_surface(data)}


# --- Live package-ownership re-authentication --------------------------------

def _package_ownership(sdk_root: Path, declared: str, timeout: int) -> dict[str, Any]:
    """Re-authenticate ownership of ``sdk_root`` live, never from a recorded root inventory.

    ``dpkg-query -W`` must report exactly the declared version, ``dpkg-query -L`` must own
    ``sdk_root`` as a directory plus at least one regular file beneath it, and ``dpkg --verify``
    must be silent.
    """
    query, dpkg = system_executable("dpkg-query"), system_executable("dpkg")
    status = _run([query, "-W", "-f=${db:Status-Abbrev}\t${Version}", NATIVE_PACKAGE],
                  timeout=timeout)
    fields = status["stdout"].strip().split("\t")
    live = fields[1].strip() if len(fields) == 2 and fields[0].startswith("ii") else None
    listing = _run([query, "-L", NATIVE_PACKAGE], timeout=timeout)
    owned = [Path(item.strip()) for item in listing["stdout"].splitlines()
             if item.strip().startswith("/")]
    files = [item for item in owned if item != sdk_root and item.is_relative_to(sdk_root)
             and item.is_file() and not item.is_symlink()]
    integrity = _run([dpkg, "--verify", NATIVE_PACKAGE], timeout=timeout)
    root_owned = bool(listing["exit_code"] == 0 and sdk_root in owned and sdk_root.is_dir()
                      and not sdk_root.is_symlink())
    silent = integrity["exit_code"] == 0 and not integrity["stdout"].strip()
    reasons = [message for ok, message in (
        (status["exit_code"] == 0,
         f"dpkg-query -W exited {status['exit_code']} for {NATIVE_PACKAGE}"),
        (live == declared,
         f"live dpkg version {live!r} is not the declared {NATIVE_PACKAGE} version {declared!r}"),
        (_public_sdk_release(live), f"live dpkg version {live!r} is outside the public"
                                    f" {_PUBLIC_SDK_FAMILY}.x release family"),
        (root_owned, f"{sdk_root} is not a package-owned directory of {NATIVE_PACKAGE}"),
        (bool(files), f"{NATIVE_PACKAGE} owns no regular file beneath {sdk_root}"),
        (silent, f"dpkg --verify reported a modified {NATIVE_PACKAGE} payload")) if not ok]
    return {"status": "verified" if not reasons else "failed", "package": NATIVE_PACKAGE,
            "declared_version": declared, "live_version": live, "sdk_root": str(sdk_root),
            "sdk_root_owned": root_owned, "owned_file_count": len(files), "verify_silent": silent,
            "queries": [_tail(status, 2000), _tail(listing, 2000), _tail(integrity)],
            "reasons": reasons}


# --- Configure and build the two official targets ----------------------------

def _configure_and_build(environment: dict[str, Any], build_dir: Path, timeout: int,
                         reasons: list[str]) -> tuple[Path, dict[str, Any]]:
    """Configure and build only the official AppEncCuda and AppDec targets in a fresh tree."""
    native = environment["native"]
    tools = native["tools"]
    samples = Path(native["sdk_root"]) / "Samples"
    _require(samples.is_dir() and not samples.is_symlink(),
             f"package-owned Samples tree is missing: {samples}")
    build_root = _fresh_directory(build_dir, label="native build directory")
    arguments = ["-S", str(samples), "-B", str(build_root), "-G", tools["generator"]["name"],
                 f"-DCMAKE_MAKE_PROGRAM={tools['generator']['path']}",
                 "-DCMAKE_BUILD_TYPE=Release",
                 f"-DCMAKE_CXX_COMPILER={tools['cxx']['path']}",
                 f"-DCUDAToolkit_ROOT={native['cuda']['root']}",
                 f"-DCUDAToolkit_NVCC_EXECUTABLE={tools['nvcc']['path']}",
                 f"-DCMAKE_CUDA_COMPILER={tools['nvcc']['path']}",
                 f"-DPKG_CONFIG_EXECUTABLE={tools['pkg_config']['path']}"]
    cmake = tools["cmake"]["path"]
    commands = [_run([cmake, *arguments], timeout=timeout)]
    if commands[0]["exit_code"] != 0:
        reasons.append("official sample configure failed")
    for target in ("AppEncCuda", "AppDec") if not reasons else ():
        commands.append(_run([cmake, "--build", str(build_root), "--target", target,
                              "--parallel", "2"], timeout=timeout))
        if commands[-1]["exit_code"] != 0:
            reasons.append(f"official {target} sample build failed")
    return build_root, {"status": "failed" if reasons else "built", "root": str(build_root),
                        "generator": tools["generator"]["name"], "arguments": arguments,
                        "commands": [_tail(item) for item in commands]}


# --- Built binaries and real driver-library linkage --------------------------

def _sample_binary(build_root: Path | None, role: str) -> Path | None:
    """Return only CMake's deterministic target output, confined to the fresh build tree."""
    parts = _SAMPLES[role]
    path = build_root.joinpath(*parts, parts[-1]) if build_root is not None else None
    if path is None or not path.is_file() or path.is_symlink():
        return None
    resolved = path.resolve(strict=True)
    return resolved if resolved.is_relative_to(build_root.resolve()) else None


def _resolved_library(line: str | None) -> tuple[str, str | None]:
    """Classify one ldd row as a real resolution, a missing library, or an SDK link stub."""
    value = line.split("=>", 1)[1].strip().split(" ", 1)[0] if line and "=>" in line else ""
    try:
        real = str(Path(value).resolve(strict=True)) if value.startswith("/") else ""
    except OSError:
        real = ""
    if not real or not Path(real).is_file():
        return "missing", value or None
    return ("stub" if "/stubs/" in value or "/stubs/" in real else "resolved"), real


def _linkage(binary: Path, required: tuple[str, ...], timeout: int) -> dict[str, Any]:
    """Reject stub-linked builds: every required driver library must resolve for real."""
    try:
        evidence = _run([system_executable("ldd"), str(binary)], timeout=timeout)
    except FileNotFoundError:
        return {"status": "unknown", "reason": "ldd is unavailable", "required": list(required)}
    rows = (evidence["stdout"] + "\n" + evidence["stderr"]).splitlines()
    libraries: dict[str, Any] = {}
    for library in required:
        line = next((item.strip() for item in rows if item.strip().startswith(library)), None)
        state, real = _resolved_library(line)
        libraries[library] = {"status": state, "real_path": real, "line": line}
    failures = sorted(f"{name} runtime resolution is {item['status']}"
                      for name, item in libraries.items() if item["status"] != "resolved")
    return {"status": "verified" if not failures and evidence["exit_code"] == 0 else "failed",
            "command": evidence["argv"], "exit_code": evidence["exit_code"],
            "libraries": libraries, "failures": failures,
            "stderr_tail": evidence["stderr"][-2000:]}


def _built_samples(build_root: Path | None, timeout: int,
                   reasons: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Locate, hash, and link-check both official sample binaries."""
    binaries = {role: _sample_binary(build_root, role) for role in _SAMPLES}
    records: dict[str, Any] = {}
    for role, path in binaries.items():
        label = _SAMPLES[role][-1]
        if path is None:
            reasons.append(f"official {label} binary is not built")
            records[role] = {"binary": None, "identity": None, "linkage": {
                "status": "unknown", "reason": f"{label} binary is not built"}}
            continue
        linkage = _linkage(path, _DRIVER_LIBRARIES[role], timeout)
        if linkage["status"] != "verified":
            reasons.append(f"official {label} runtime driver-library linkage is not verified")
        records[role] = {"binary": str(path), "linkage": linkage,
                         "identity": file_identity(path, label=f"{label} binary")}
    return binaries, records


# --- The official aggregate capability reports (schema 1.5) -------------------

def _classification(role: str) -> dict[str, Any]:
    """The frozen official_sample_report / sample_reported classification for one route."""
    sample, option = _AGGREGATE_ROUTES[role]
    return {"evidence_source_type": "official_sample_report", "official_sample": True,
            "sample_name": sample, "report_option": option, "scope": "sample_reported"}


def _encoder_gpus(stdout: str) -> list[dict[str, Any]]:
    """Split the AppEncCuda -ec report per GPU and keep its rows verbatim, never as a verdict."""
    matches = list(_CAPS_GPU_RE.finditer(stdout))
    records = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stdout)
        block = stdout[match.end():end]
        rows = _CAPS_CODEC_RE.findall(block)
        records.append({
            "gpu": int(match.group(1)), "name": match.group(2).strip(),
            "sample_reported_codec_rows": [{"codec": codec.lower(), "value": value}
                                           for codec, value in rows],
            "required_sections_present": bool(
                all(block.count(section) == 1 for section in _ENCODER_SECTIONS)
                and sorted(codec for codec, _ in rows) == ["AV1", "H264", "HEVC"])})
    return records


def _aggregate(role: str, binary: Path, identity: dict[str, Any],
               request: dict[str, int]) -> dict[str, Any]:
    """Run one official aggregate report without promoting its rows to a support claim.

    The sample binary is re-authenticated immediately before and after the report, so a binary
    substituted around the report window fails closed rather than producing quiet evidence.
    """
    sample, option = _AGGREGATE_ROUTES[role]
    initial = file_identity(binary, label=f"{sample} before {option}")
    _require(initial == identity, f"official {sample} changed before its {option} report")
    evidence = _run([str(binary), option], timeout=request["timeout"])
    terminal = file_identity(binary, label=f"{sample} after {option}")
    _require(terminal == identity, f"official {sample} changed while its {option} report ran")
    stdout, gpu = evidence["stdout"], request["gpu"]
    records = (_encoder_gpus(stdout) if role == "encoder" else
               [{"gpu": int(item.group(1)), "name": item.group(2).strip()}
                for item in _CAPS_GPU_RE.finditer(stdout)])
    ordinals = [item["gpu"] for item in records]
    failures = _failure_markers(evidence)
    shape = (bool(len(_ENCODER_CAPS_RE.findall(stdout)) == 1
                  and stdout.count(_ENCODER_CAPS_HINT) == 1
                  and all(item["required_sections_present"] for item in records))
             if role == "encoder" else
             bool(len(_DECODER_CAPS_RE.findall(stdout)) == len(ordinals)
                  and len(_DECODER_SUMMARY_RE.findall(stdout)) == len(ordinals)
                  and stdout.count(_DECODER_CAPS_HINT) == 1))
    valid = bool(evidence["exit_code"] == 0 and not evidence["timed_out"] and not failures
                 and ordinals and len(ordinals) == len(set(ordinals)) and gpu in ordinals and shape)
    return {"status": "completed" if valid else "unknown",
            "evidence_classification": _classification(role),
            "authority": f"official {sample} aggregate report",
            "claim_scope": _AGGREGATE_SCOPE[role], "command": evidence["argv"],
            "cwd": evidence["cwd"], "exit_code": evidence["exit_code"],
            "timed_out": evidence["timed_out"], "selected_gpu": gpu,
            "selected_gpu_present": gpu in ordinals, "reported_gpus": records,
            "failure_markers": failures, "stdout_tail": stdout[-8000:],
            "stderr_tail": evidence["stderr"][-4000:],
            "binary_integrity": {"initial": initial, "terminal": terminal}}


def _aggregates(binaries: dict[str, Any], samples: dict[str, Any], request: dict[str, int],
                reasons: list[str]) -> dict[str, Any]:
    """Capture both aggregate reports. They never add a reason: readiness is the operation proof.

    This CLI always requires --run-encode and --run-decode, so the baseline ``not_run`` state is
    unreachable here; an unbuilt or stub-linked sample yields ``unknown`` and the run still emits
    the complete artifact shape.
    """
    records: dict[str, Any] = {}
    for role, (sample, _) in _AGGREGATE_ROUTES.items():
        record = samples.get(role) or {}
        if reasons or binaries.get(role) is None or record.get("identity") is None or (
                record.get("linkage") or {}).get("status") != "verified":
            records[role] = {
                "status": "unknown", "evidence_classification": _classification(role),
                "reason": f"authenticated {sample} build and runtime linkage are unavailable"}
            continue
        records[role] = _aggregate(role, binaries[role], record["identity"], request)
    return records


# --- The official encode -> independent decode operation proof ---------------

def _operation(command: list[str], evidence: dict[str, Any], success: bool,
               extra: dict[str, Any]) -> dict[str, Any]:
    return {"status": "operation_verified" if success else "operation_failed",
            "command": command, "cwd": evidence["cwd"], "exit_code": evidence["exit_code"],
            "timed_out": evidence["timed_out"], "stdout_tail": evidence["stdout"][-4000:],
            "stderr_tail": evidence["stderr"][-4000:], **extra}


def _reason(sample: str, evidence: dict[str, Any], failures: list[str], detail: str) -> str:
    if evidence["timed_out"]:
        return f"official {sample} timed out"
    if failures:
        return f"official {sample} emitted explicit failure marker(s): " + ", ".join(failures)
    return detail


def _encode(binary: Path, work: Path, request: dict[str, int],
            reasons: list[str]) -> dict[str, Any]:
    """Encode exactly one fresh NV12 frame with the official AppEncCuda sample."""
    width, height = request["width"], request["height"]
    source = work / f"native-smoke-{width}x{height}.nv12"
    output = work / f"native-appenc-{width}x{height}.h264"
    _require(not os.path.lexists(source) and not os.path.lexists(output),
             "a native encode input or bitstream already exists; use a fresh work directory")
    write_new_bytes(source, bytes([16]) * (width * height) + bytes([128]) * (width * height // 2))
    fixture = file_identity(source, label="native NV12 input")
    _require(fixture["size_bytes"] == _SMOKE_NV12_BYTES,
             f"the one-frame NV12 fixture must be exactly {_SMOKE_NV12_BYTES} bytes,"
             f" got {fixture['size_bytes']}")
    command = [str(binary), "-i", str(source), "-s", f"{width}x{height}", "-if", "nv12",
               "-gpu", str(request["gpu"]), "-codec", "h264", "-o", str(output)]
    evidence = _run(command, cwd=work, timeout=request["timeout"])
    failures = _failure_markers(evidence)
    frames = _ENCODED_RE.findall(evidence["stdout"])
    markers = [item.strip() for item in _BITSTREAM_RE.findall(evidence["stdout"])]
    matched = bool(len(markers) == 1
                   and Path(markers[0]).expanduser().resolve() == output.resolve())
    identity = _identity(output, "native H.264 bitstream")
    success = bool(evidence["exit_code"] == 0 and not evidence["timed_out"] and not failures
                   and frames == ["1"] and matched and identity and identity["size_bytes"] > 0)
    if not success:
        reasons.append(_reason("AppEncCuda", evidence, failures,
                               "official AppEncCuda did not report exactly one encoded frame, its"
                               " exact requested output marker, and a non-empty bitstream"))
    return _operation(command, evidence, success, {
        "encoded_frames": int(frames[0]) if len(frames) == 1 else None,
        "frame_marker_matches": len(frames), "failure_markers": failures,
        "output_marker": markers[0] if len(markers) == 1 else None,
        "output_marker_matches_requested": matched, "bitstream": identity,
        "input": {"path": str(source), "format": "NV12", "width": width, "height": height,
                  "frames": 1, "sha256": fixture["sha256"],
                  "size_bytes": fixture["size_bytes"]}})


def _encode_marker_accepted(encode: dict[str, Any]) -> bool:
    """The encode-before-decode gate.

    AppDec is launched only after AppEncCuda's own success marker has been proven by this run: a
    verified operation status, exactly one reported encoded frame, the exact requested
    ``Bitstream saved in file`` marker, and a non-empty authenticated bitstream. This is an
    ordering guarantee over a real operation; it must never be weakened to a capability query.
    """
    bitstream = encode.get("bitstream")
    return bool(encode.get("status") == "operation_verified"
                and encode.get("encoded_frames") == 1
                and encode.get("output_marker_matches_requested") is True
                and isinstance(bitstream, dict) and isinstance(bitstream.get("sha256"), str)
                and isinstance(bitstream.get("size_bytes"), int) and bitstream["size_bytes"] > 0)


def _fresh_bitstream(encode: dict[str, Any]) -> dict[str, Any] | None:
    """Re-authenticate the encoded bitstream on disk BEFORE AppDec is launched.

    Freshness and non-emptiness are preconditions of the decode launch, never postconditions. A
    stale, replaced, truncated, or empty bitstream stops the run before the decoder starts, so
    AppDec is never pointed at a file this run did not just produce.
    """
    recorded = encode.get("bitstream")
    if not isinstance(recorded, dict) or not isinstance(recorded.get("path"), str):
        return None
    try:
        live = _identity(Path(recorded["path"]), "AppDec input bitstream")
    except (OSError, ValueError):
        return None
    return live if live == recorded and live["size_bytes"] > 0 else None


def _decode(binary: Path, work: Path, bitstream: dict[str, Any], request: dict[str, int],
            reasons: list[str]) -> dict[str, Any]:
    """Decode the fresh AppEncCuda bitstream with the official AppDec sample."""
    width, height = request["width"], request["height"]
    source, expected = Path(bitstream["path"]), _SMOKE_NV12_BYTES
    output, legacy = work / f"native-appdec-{width}x{height}.nv12", work / "out.native"
    _require(not os.path.lexists(output) and not os.path.lexists(legacy),
             "native decoded output appeared before AppDec launch; use a fresh work directory")
    command = [str(binary), "-i", str(source), "-o", str(output), "-gpu", str(request["gpu"])]
    evidence = _run(command, cwd=work, timeout=request["timeout"])
    _require(file_identity(source, label="AppDec input bitstream") == bitstream,
             "the fresh AppEncCuda bitstream changed while AppDec was running")
    failures = _failure_markers(evidence)
    frames = _DECODED_RE.findall(evidence["stdout"])
    stray = os.path.lexists(legacy)
    identity = _identity(output, "decoded NV12 output")
    success = bool(evidence["exit_code"] == 0 and not evidence["timed_out"] and not failures
                   and frames == ["1"] and not stray and identity
                   and identity["size_bytes"] == expected)
    if not success:
        reasons.append("official AppDec created an unexpected out.native output" if stray else
                       _reason("AppDec", evidence, failures,
                               "official AppDec did not report exactly one decoded frame and"
                               f" produce the exact {expected}-byte one-frame NV12 output"))
    return _operation(command, evidence, success, {
        "decoded_frames": int(frames[0]) if len(frames) == 1 else None,
        "frame_marker_matches": len(frames), "failure_markers": failures,
        "unexpected_out_native_appeared": stray, "input_bitstream": bitstream,
        "decoded_output": {"path": str(output), "format": "NV12", "width": width,
                           "height": height, "frames": 1, "expected_size_bytes": expected,
                           **(identity or {})}})


def _operations(binaries: dict[str, Any], work_dir: Path, request: dict[str, int],
                reasons: list[str]) -> dict[str, Any]:
    """Run the official encode, then -- only on a proven encode marker -- the official decode."""
    decode = {"status": "not_run", "reason": "official AppDec is launched only after a proven"
              " AppEncCuda success marker from this run"}
    if reasons or not binaries["encoder"] or not binaries["decoder"]:
        return {"decode": decode, "encode": {
            "status": "not_run", "reason": "official AppEncCuda was not launched: "
                                           + ("; ".join(reasons) or "no authenticated binary")}}
    work = _fresh_directory(work_dir, label="native verification work directory")
    encode = _encode(binaries["encoder"], work, request, reasons)
    if not _encode_marker_accepted(encode):
        return {"encode": encode, "decode": decode}
    bitstream = _fresh_bitstream(encode)
    if bitstream is None:
        reasons.append("the AppEncCuda bitstream was not fresh, unchanged, and non-empty at the"
                       " AppDec launch point")
        return {"encode": encode, "decode": dict(decode, reason=reasons[-1])}
    return {"encode": encode,
            "decode": _decode(binaries["decoder"], work, bitstream, request, reasons)}


def _validate_request(request: dict[str, int]) -> None:
    for name in ("gpu", "width", "height", "timeout"):
        _require(isinstance(request[name], int) and not isinstance(request[name], bool),
                 f"--{name} must be an integer")
    _require(request["gpu"] >= 0, "--gpu must be non-negative")
    _require(request["timeout"] > 0, "--timeout must be positive")
    _require((request["width"], request["height"]) == _SMOKE_GEOMETRY,
             f"the native smoke is fixed at {_SMOKE_GEOMETRY[0]}x{_SMOKE_GEOMETRY[1]}, got"
             f" {request['width']}x{request['height']}")
    _require(request["width"] * request["height"] * 3 // 2 == _SMOKE_NV12_BYTES,
             f"the one-frame NV12 payload must be exactly {_SMOKE_NV12_BYTES} bytes")


def _select_sdk_root(requested: Path, recorded: str) -> Path:
    """Accept the recorded root or its unversioned parent with one 13.0.x child."""
    selected = Path(recorded)
    root = requested.expanduser().absolute()
    if root == selected:
        return selected
    _require(root.is_dir() and not root.is_symlink(),
             f"--sdk-root {requested} is not a regular SDK directory")
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and _public_sdk_release(path.name)
    )
    _require(candidates == [selected],
             f"--sdk-root {requested} does not contain only the canonical"
             f" installation.native_sdk.sdk_root {recorded}")
    return selected


def verify(paths: dict[str, Path], request: dict[str, int]) -> dict[str, Any]:
    """Run the full official native build plus encode -> decode proof and return its artifact."""
    _validate_request(request)
    environment = _load_environment(paths["environment"], request["gpu"])
    native = environment["native"]
    sdk_root = _select_sdk_root(paths["sdk_root"], native["sdk_root"])
    # An absent required build tool blocks the run but still yields the complete result shape.
    reasons: list[str] = sorted(str(item["reason"]) for item in native["tools"].values()
                                if item.get("status") != "verified")
    ownership = _package_ownership(sdk_root, native["package"]["version"],
                                   min(request["timeout"], 60))
    reasons.extend(ownership["reasons"])
    build_root, build = (
        (None, {"status": "blocked", "root": None, "commands": [],
                "reason": "native configure was blocked before CMake: " + "; ".join(reasons)})
        if reasons else
        _configure_and_build(environment, paths["build_dir"], request["timeout"], reasons))
    binaries, samples = _built_samples(build_root, min(request["timeout"], 60), reasons)
    aggregates = _aggregates(binaries, samples, request, reasons)
    operations = _operations(binaries, paths["work_dir"], request, reasons)
    states = [operations[name].get("status") for name in ("encode", "decode")]
    ready = bool(states == ["operation_verified"] * 2 and not reasons)
    return {
        "schema_version": SCHEMA_VERSION, "kind": "nvcodec-native-verification",
        "generated_at": utc_now(), "ready": ready, "software_fallback": False,
        "status": ("operation_verified" if ready else
                   "operation_failed" if "operation_failed" in states else "unknown"),
        "command_provenance": ("package-installed Video Codec SDK Samples/AppEncode/AppEncCuda ->"
                               " Samples/AppDecode/AppDec"),
        "environment": environment, "package_ownership": ownership,
        "native_build_prerequisites": native["build_prerequisites"],
        "build": build, "samples": samples,
        "aggregate_encoder_capabilities": aggregates["encoder"],
        "aggregate_decoder_capabilities": aggregates["decoder"],
        "hardware_operations": operations, "reasons": reasons,
    }


def _emit_error(exc: BaseException, output: Path | None, code: int) -> int:
    document = {"schema_version": SCHEMA_VERSION, "kind": "nvcodec-native-verification-error",
                "status": "error", "generated_at": utc_now(), "error": str(exc)}
    # allow_nan=False keeps stdout strict even on the error path.
    print(json.dumps(document, allow_nan=False, indent=2, sort_keys=True))
    if output is not None:
        try:
            write_new_json(output, document)
        except (OSError, ValueError):
            pass
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", type=Path, required=True,
                        help="Validated live nvcodec-environment 1.2 JSON")
    parser.add_argument("--sdk-root", type=Path, default=Path("/opt/nvidia/video-codec-sdk"),
                        help="Canonical SDK root or its unversioned parent")
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "nvcodec-native-build")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd() / "nvcodec-native-smoke")
    for flag, note in (("build", "Configure and build official AppEncCuda and AppDec"),
                       ("run-encode", "Run the built official AppEncCuda"),
                       ("run-decode", "Run official AppDec on the fresh AppEncCuda bitstream")):
        parser.add_argument(f"--{flag}", action="store_true", help=note)
    for name, default in (("gpu", 0), ("width", _SMOKE_GEOMETRY[0]),
                          ("height", _SMOKE_GEOMETRY[1]), ("timeout", 300)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output", type=Path,
                        help="Write the JSON artifact here; an existing path is never overwritten")
    args = parser.parse_args()
    if not (args.build and args.run_encode and args.run_decode):
        parser.error("native verification requires --build, --run-encode, and --run-decode")
    try:
        require_isolated()
        if args.output is None:
            raise ValueError("native verification requires --output")
        if os.path.lexists(args.output.expanduser()):
            raise FileExistsError(f"native verification output already exists: {args.output}")
        result = verify({"environment": args.environment, "sdk_root": args.sdk_root,
                         "build_dir": args.build_dir, "work_dir": args.work_dir},
                        {"gpu": args.gpu, "width": args.width, "height": args.height,
                         "timeout": args.timeout})
        print(write_new_json(args.output, result).read_bytes().decode("utf-8"), end="")
        return 0 if result["ready"] else 2
    except subprocess.TimeoutExpired as exc:
        return _emit_error(exc, args.output, 2)
    except (FileNotFoundError, ValueError) as exc:
        return _emit_error(exc, args.output, 3)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _emit_error(exc, args.output, 3)


if __name__ == "__main__":
    raise SystemExit(main())
