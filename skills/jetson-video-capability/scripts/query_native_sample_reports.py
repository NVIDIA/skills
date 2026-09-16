#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query raw native AppEncCuda/AppDec capability reports from verified binaries."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-lines

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).absolute().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# pylint: disable=wrong-import-position
from _capability_contract import (  # noqa: E402
    add_setup_remediation,
    ENVIRONMENT_SCHEMA_VERSION as ENVIRONMENT_SCHEMA,
    exclusive_write_bytes,
)

KIND = "nvcodec-native-sample-capability-report"
SCHEMA = "1.0"
VERIFY_KIND = "nvcodec-native-verification"
VERIFY_SCHEMA = "1.5"
PACKAGE = "nvidia-video-codec-sdk"
MAX_JSON = 16 * 1024 * 1024
MAX_STREAM = 8 * 1024 * 1024
MAX_BINARY = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
SDK_RE = re.compile(r"13\.0(?:\.\d+)*(?:\+[0-9A-Za-z.]+)?(?:-[0-9A-Za-z.+]+)?")
NVCC_RE = re.compile(r"\brelease\s+(13\.\d+)(?:\D|$)")
GPU_RE = re.compile(r"(?m)^GPU[ \t]+(\d+)[ \t]+-[ \t]+([^\r\n]+?)[ \t]*\r?$")
GPU_IN_USE_RE = re.compile(r"(?m)^GPU in use:[ \t]+([^\r\n]+?)[ \t]*\r?$")
CODEC_RE = re.compile(r"(?m)^[ \t]*(H264|HEVC|AV1):[ \t]*(yes|no)[ \t]*\r?$")
DECODER_ROW_RE = re.compile(
    r"^Codec[ \t]+(?P<codec>[A-Z0-9]+)[ \t]+BitDepth[ \t]+(?P<bit>\d+)"
    r"[ \t]+ChromaFormat[ \t]+(?P<chroma>\d+:\d+:\d+)[ \t]+Supported[ \t]+"
    r"(?P<value>[01])[ \t]+MaxWidth[ \t]+(?P<maxw>\d+)[ \t]+MaxHeight[ \t]+"
    r"(?P<maxh>\d+)[ \t]+MaxMBCount[ \t]+(?P<maxmb>\d+)[ \t]+MinWidth[ \t]+"
    r"(?P<minw>\d+)[ \t]+MinHeight[ \t]+(?P<minh>\d+)[ \t]+SurfaceFormat"
    r"[ \t]+(?P<formats>[A-Z0-9]+(?:[ \t]+[A-Z0-9]+)*)[ \t]*\r?$"
)
FAILURES = (
    ("nvenc_error", re.compile(r"(?im)^\s*(?:NV_ENC_ERR_[A-Z0-9_]+\b|NVENC error\b)")),
    ("cuda_error", re.compile(r"(?im)^\s*(?:CUDA_ERROR_[A-Z0-9_]+\b|CUDA .*error\b)")),
    ("error", re.compile(
        r"(?im)^\s*(?:Traceback \(most recent call last\)|\[(?:FATAL|ERROR)\s*\]|"
        r"(?:uncaught )?exception\b|error\b|failed\b|failure\b)"
    )),
)
REQUIRED_LIBS = {
    "encoder": ("libcuda.so.1", "libnvidia-encode.so.1"),
    "decoder": ("libcuda.so.1", "libnvcuvid.so.1"),
}
SAMPLES = {"encoder": ("AppEncCuda", "-ec"), "decoder": ("AppDec", "-dc")}
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
BUILD_TOOLS = {
    "cmake": (Path("/usr/bin/cmake"),),
    "cxx": (Path("/usr/bin/g++"),),
    "pkg_config": (Path("/usr/bin/pkg-config"),),
    "nvcc": (
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/usr/local/cuda-13.0/bin/nvcc"),
    ),
}
GENERATORS = (
    ("Ninja", Path("/usr/bin/ninja")),
    ("Unix Makefiles", Path("/usr/bin/make")),
)
APPDEC_MODULES = ("libavcodec", "libavformat", "libavutil", "libswresample")


class _Unavailable(ValueError):
    """A well-formed native authority is unavailable."""


def _public_sdk_release(value: Any) -> bool:
    """Accept the same stable Debian 13.0 family as native setup verification."""
    normalized = value.strip().split(":", 1)[-1] if isinstance(value, str) else ""
    if not normalized or "~" in normalized or "really" in normalized.lower():
        return False
    return SDK_RE.fullmatch(normalized) is not None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-RFC JSON numeric constant is forbidden: {value}")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _loads(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )


def _dumps(value: Any, *, compact: bool = False) -> bytes:
    options = {"sort_keys": True, "allow_nan": False}
    text = json.dumps(
        value, separators=(",", ":") if compact else None,
        indent=None if compact else 2, **options
    )
    return (text + ("" if compact else "\n")).encode("utf-8")


def _snapshot(path: Path, *, label: str, limit: int = MAX_JSON,
              executable: bool = False) -> tuple[bytes, dict[str, Any]]:
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {requested}")
    flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(requested, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"{label} must be a regular file no larger than {limit} bytes")
        chunks, total = [], 0
        while chunk := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError(f"{label} exceeds {limit} bytes")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(requested, follow_symlinks=False)
    before_token = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_token = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if before_token != after_token or after_token != current_token:
        raise ValueError(f"{label} changed while it was read")
    if executable and not after.st_mode & 0o111:
        raise ValueError(f"{label} is not executable")
    raw = b"".join(chunks)
    return raw, {"path": str(requested.resolve(strict=True)), "size_bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path, *, label: str) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    raw, identity = _snapshot(path, label=label)
    value = _loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return raw, identity, value


def _same_identity(observed: dict[str, Any], expected: Any, *, label: str) -> None:
    if not isinstance(expected, dict) or any(
        expected.get(key) != observed[key] for key in ("path", "size_bytes", "sha256")
    ):
        raise ValueError(f"{label} does not match its recorded identity")


def _run(argv: list[str], timeout: int) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    try:
        done = subprocess.run(argv, capture_output=True, check=False, timeout=timeout,
                              env=SAFE_ENV)
        code, out, err, expired = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, out, err, expired = None, exc.stdout or b"", exc.stderr or b"", True
    out = out.encode() if isinstance(out, str) else out
    err = err.encode() if isinstance(err, str) else err
    ended_at = datetime.now(timezone.utc)
    return {"argv": argv, "cwd": str(Path.cwd().resolve()), "exit_code": code,
            "timed_out": expired, "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round(time.monotonic() - started_clock, 6),
            "stdout_bytes": out, "stderr_bytes": err}


def _run_build(argv: list[str], timeout: int) -> dict[str, Any]:
    """Run a build command and reap its whole process group on timeout."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=SAFE_ENV,
        start_new_session=True,
    )
    expired = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = process.communicate()
    ended_at = datetime.now(timezone.utc)
    return {
        "argv": argv,
        "cwd": str(Path.cwd().resolve()),
        "exit_code": None if expired else process.returncode,
        "timed_out": expired,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(time.monotonic() - started_clock, 6),
        "stdout_bytes": out,
        "stderr_bytes": err,
    }


def _command_record(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "argv": evidence["argv"],
        "cwd": evidence["cwd"],
        "exit_code": evidence["exit_code"],
        "timed_out": evidence["timed_out"],
        "started_at": evidence["started_at"],
        "ended_at": evidence["ended_at"],
        "duration_seconds": evidence["duration_seconds"],
        "stdout": _stream(evidence["stdout_bytes"]),
        "stderr": _stream(evidence["stderr_bytes"]),
    }


def _fresh_workspace(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if os.path.lexists(requested):
        raise ValueError(f"native report workspace must be fresh: {requested}")
    parent = requested.parent.resolve(strict=True)
    if requested.parent != parent or requested.parent.is_symlink():
        raise ValueError("native report workspace parent must be canonical and non-symlink")
    requested.mkdir(mode=0o700)
    return requested.resolve(strict=True)


def _executable(candidates: tuple[Path, ...], *, label: str) -> tuple[Path, dict[str, Any]]:
    """Select the first executable from one bounded preference list."""
    for candidate in candidates:
        try:
            path = candidate.resolve(strict=True)
        except OSError:
            continue
        if path.is_file() and path.stat().st_mode & 0o111:
            return path, _snapshot(
                path,
                label=f"native report tool {label}",
                executable=True,
                limit=MAX_BINARY,
            )[1]
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise _Unavailable(
        f"required native report tool {label} is unavailable at fixed candidate paths: "
        f"{locations}"
    )


def _local_tools(timeout: int, reports: tuple[str, ...]) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for name, candidates in BUILD_TOOLS.items():
        path, identity = _executable(candidates, label=name)
        tools[name] = {"path": path, "identity": identity}
    available_generators = []
    for name, candidate in GENERATORS:
        try:
            path = candidate.resolve(strict=True)
        except OSError:
            continue
        if path.is_file() and path.stat().st_mode & 0o111:
            available_generators.append((name, path))
    if not available_generators:
        raise _Unavailable("no supported CMake generator is installed")
    generator_name, generator_path = available_generators[0]
    tools["generator"] = {
        "name": generator_name,
        "path": generator_path,
        "identity": _snapshot(
            generator_path,
            label="native report build generator",
            executable=True,
            limit=MAX_BINARY,
        )[1],
    }
    nvcc_version = _run([str(tools["nvcc"]["path"]), "--version"], min(timeout, 30))
    nvcc_text = (nvcc_version["stdout_bytes"] + nvcc_version["stderr_bytes"]).decode(
        "utf-8", errors="replace"
    )
    if nvcc_version["exit_code"] != 0 or NVCC_RE.search(nvcc_text) is None:
        raise _Unavailable("native report build requires a CUDA 13.x nvcc toolchain")
    tools["nvcc_version_command"] = _command_record(nvcc_version)
    if "decoder" in reports:
        missing = []
        for module in APPDEC_MODULES:
            query = _run(
                [str(tools["pkg_config"]["path"]), "--exists", module], min(timeout, 30)
            )
            if query["exit_code"] != 0:
                missing.append(module)
        if missing:
            raise _Unavailable(
                "native AppDec report build prerequisites are missing: "
                + ", ".join(missing)
            )
    return tools


def _stream(raw: bytes) -> dict[str, Any]:
    within = len(raw) <= MAX_STREAM
    return {"capture_status": "complete" if within else "limit_exceeded",
            "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "content": raw.decode("utf-8", errors="replace") if within else None,
            "tail": raw[-4000:].decode("utf-8", errors="replace")}


def _markers(stdout: str, stderr: str) -> list[str]:
    combined = stdout + "\n" + stderr
    return [name for name, pattern in FAILURES if pattern.search(combined)]


def _classification(role: str) -> dict[str, Any]:
    sample, option = SAMPLES[role]
    return {"evidence_source_type": "official_sample_report", "official_sample": True,
            "sample_name": sample, "report_option": option, "scope": "sample_reported"}


def _line_count(lines: list[str], value: str) -> int:
    return sum(line.rstrip(" \t\r") == value for line in lines)


def _encoder_parse(
    stdout: str, gpu: int
) -> tuple[str | None, list[dict[str, Any]], bool, str | None]:
    lines = stdout.splitlines()
    legacy = _line_count(lines, "Encoder Capability Summary")
    compact = _line_count(lines, "Encoder Capability")
    if (legacy, compact) not in ((1, 0), (0, 1)):
        return None, [], False, "encoder output has no single recognized header"
    matches = list(GPU_RE.finditer(stdout))
    records = []
    for index, match in enumerate(matches):
        block = stdout[match.end():matches[index + 1].start() if index + 1 < len(matches)
                       else len(stdout)]
        rows = CODEC_RE.findall(block)
        valid_rows = sorted(codec for codec, _value in rows) == ["AV1", "H264", "HEVC"]
        valid_sections = True
        if legacy:
            valid_sections = (
                _line_count(block.splitlines(), "=== CODEC SUPPORT SUMMARY ===") == 1
                and _line_count(
                    block.splitlines(), "=== ENCODER CAPABILITIES SUMMARY TABLE ==="
                ) == 1
            )
        records.append({"gpu": int(match.group(1)), "name": match.group(2).strip(),
                        "sample_reported_codec_rows": [
                            {"codec": name.lower(), "value": value} for name, value in rows],
                        "required_sections_present": valid_sections,
                        "required_rows_present": valid_rows})
    ordinals = [item["gpu"] for item in records]
    hint_ok = _line_count(
        lines, "For detailed information about all capabilities, use -ec-detail"
    ) == 1
    valid = bool(records and len(ordinals) == len(set(ordinals)) and gpu in ordinals
                 and all(item["required_sections_present"] and item["required_rows_present"]
                         for item in records) and (not legacy or hint_ok))
    return ("legacy_summary" if legacy else "sdk_13_compact"), records, gpu in ordinals, (
        None if valid else "encoder report grammar or selected GPU is incomplete")


def _decoder_parse(  # pylint: disable=too-many-branches
    stdout: str, gpu: int
) -> tuple[str | None, list[dict[str, Any]], bool, str | None]:
    lines = stdout.splitlines()
    legacy_headers = _line_count(lines, "GPU Decoder Capabilities")
    compact_headers = _line_count(lines, "Decoder Capability")
    if legacy_headers and not compact_headers:
        records = [{"gpu": int(match.group(1)), "name": match.group(2).strip()}
                   for match in GPU_RE.finditer(stdout)]
        ordinals = [item["gpu"] for item in records]
        valid = bool(records and len(ordinals) == len(set(ordinals)) and gpu in ordinals
                     and legacy_headers == len(records)
                     and _line_count(lines, "Codec Support Summary") == len(records)
                     and _line_count(
                         lines,
                         "For detailed information about all capabilities, use -dc-detail",
                     ) == 1)
        return "legacy_summary", records, gpu in ordinals, (
            None if valid else "legacy decoder report grammar or selected GPU is incomplete")
    if compact_headers != 1 or legacy_headers:
        return None, [], False, "decoder output has no single recognized header"
    names = GPU_IN_USE_RE.findall(stdout)
    codec_lines = [line for line in lines if line.lstrip().startswith("Codec")]
    matches = [DECODER_ROW_RE.fullmatch(line) for line in codec_lines]
    rows = []
    for line, match in zip(codec_lines, matches):
        if match is None:
            continue
        rows.append({"raw_line": line, "codec": match["codec"].lower(),
                     "bit_depth": int(match["bit"]), "chroma_format": match["chroma"],
                     "supported_field_value": int(match["value"]),
                     "max_width": int(match["maxw"]), "max_height": int(match["maxh"]),
                     "max_mb_count": int(match["maxmb"]), "min_width": int(match["minw"]),
                     "min_height": int(match["minh"]),
                     "surface_formats": match["formats"].split()})
    valid = bool(gpu == 0 and len(names) == 1 and codec_lines and all(matches))
    records = [{"gpu": None, "name": names[0].strip() if len(names) == 1 else None,
                "sample_reported_codec_rows": rows}]
    return "sdk_13_compact", records, valid, (
        None if valid else "compact decoder report requires GPU 0, one GPU name, and strict rows")


def _live_linkage(binary: Path, role: str, timeout: int) -> dict[str, Any]:
    evidence = _run(["/usr/bin/ldd", str(binary)], timeout)
    text = (evidence["stdout_bytes"] + b"\n" + evidence["stderr_bytes"]).decode(
        "utf-8", errors="replace")
    libraries = {}
    for name in REQUIRED_LIBS[role]:
        line = next((item.strip() for item in text.splitlines()
                     if item.strip().startswith(name)), None)
        value = line.split("=>", 1)[1].strip().split()[0] if line and "=>" in line else ""
        try:
            real = str(Path(value).resolve(strict=True)) if value.startswith("/") else None
        except OSError:
            real = None
        state = "resolved" if real and "/stubs/" not in value and "/stubs/" not in real else (
            "stub" if real else "missing")
        libraries[name] = {"status": state, "real_path": real, "line": line}
    if evidence["exit_code"] != 0 or any(item["status"] != "resolved"
                                         for item in libraries.values()):
        raise ValueError(
            f"{SAMPLES[role][0]} live driver linkage is not verified"
        )
    return {"status": "verified", "libraries": libraries, "command": evidence["argv"]}


def _local_package(timeout: int) -> dict[str, Any]:
    query = Path("/usr/bin/dpkg-query")
    dpkg = Path("/usr/bin/dpkg")
    if not query.is_file() or not dpkg.is_file():
        raise _Unavailable("dpkg package authority is unavailable")
    status = _run(
        [str(query), "-W", "-f=${db:Status-Abbrev}\t${Version}", PACKAGE],
        min(timeout, 60),
    )
    fields = status["stdout_bytes"].decode("utf-8", errors="replace").strip().split("\t")
    if status["exit_code"] != 0 or len(fields) != 2 or not fields[0].startswith("ii"):
        raise _Unavailable(f"{PACKAGE} is not installed")
    version = fields[1]
    if not _public_sdk_release(version):
        raise ValueError(f"installed {PACKAGE} version is not public SDK 13.0.x: {version}")
    listing = _run([str(query), "-L", PACKAGE], min(timeout, 60))
    if listing["exit_code"] != 0:
        raise ValueError(f"cannot enumerate package-owned {PACKAGE} files")
    owned = {
        Path(line.strip())
        for line in listing["stdout_bytes"].decode("utf-8", errors="replace").splitlines()
        if line.strip().startswith("/")
    }
    roots = sorted(
        {
            path.parent.parent
            for path in owned
            if path.name == "CMakeLists.txt"
            and path.parent.name == "Samples"
            and path.is_file()
            and not path.is_symlink()
        }
    )
    if len(roots) != 1:
        raise ValueError(
            f"{PACKAGE} must own exactly one complete SDK Samples root, found {len(roots)}"
        )
    sdk_root = roots[0]
    samples = sdk_root / "Samples"
    if (
        not sdk_root.is_dir()
        or sdk_root.is_symlink()
        or not samples.is_dir()
        or samples.is_symlink()
        or samples / "CMakeLists.txt" not in owned
    ):
        raise ValueError("package-owned SDK Samples root is malformed")
    unexpected = [
        path
        for path in samples.rglob("*")
        if (path.is_file() or path.is_symlink()) and path not in owned
    ]
    if unexpected:
        raise ValueError(
            f"package-owned SDK Samples tree contains {len(unexpected)} unowned file(s)"
        )
    source_records = []
    for path in sorted(item for item in owned if item.is_relative_to(samples)):
        if path.is_symlink():
            raise ValueError(f"package-owned SDK source must not be a symlink: {path}")
        if path.is_file():
            identity = _snapshot(path, label="package-owned SDK source")[1]
            source_records.append(
                (str(path.relative_to(samples)), identity["size_bytes"], identity["sha256"])
            )
    if not source_records:
        raise ValueError(f"{PACKAGE} owns no regular source below its Samples tree")
    source_manifest = hashlib.sha256(
        _dumps(source_records, compact=True)
    ).hexdigest()
    integrity = _run([str(dpkg), "--verify", PACKAGE], min(timeout, 60))
    if integrity["exit_code"] != 0 or integrity["stdout_bytes"].strip():
        raise ValueError(f"dpkg verification reported a modified {PACKAGE} payload")
    return {
        "name": PACKAGE,
        "version": version,
        "sdk_root": str(sdk_root.resolve(strict=True)),
        "owned_path_count": len(owned),
        "source_tree": {
            "regular_file_count": len(source_records),
            "manifest_sha256": source_manifest,
        },
        "commands": [_command_record(item) for item in (status, listing, integrity)],
    }


def _verify_tool_identities(tools: dict[str, Any]) -> None:
    for name in ("cmake", "cxx", "pkg_config", "nvcc", "generator"):
        record = tools[name]
        observed = _snapshot(
            Path(record["path"]),
            label=f"native report tool {name}",
            executable=True,
            limit=MAX_BINARY,
        )[1]
        _same_identity(observed, record["identity"], label=f"native report tool {name}")


def _local_build(
    package: dict[str, Any],
    tools: dict[str, Any],
    workspace: Path,
    reports: tuple[str, ...],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = Path(package["sdk_root"]) / "Samples"
    build_root = workspace / "build"
    cuda_root = Path(tools["nvcc"]["path"]).parent.parent.resolve(strict=True)
    configure_argv = [
        str(tools["cmake"]["path"]),
        "-S",
        str(samples),
        "-B",
        str(build_root),
        "-G",
        str(tools["generator"]["name"]),
        f"-DCMAKE_MAKE_PROGRAM={tools['generator']['path']}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={tools['cxx']['path']}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        f"-DCUDAToolkit_NVCC_EXECUTABLE={tools['nvcc']['path']}",
        f"-DCMAKE_CUDA_COMPILER={tools['nvcc']['path']}",
        f"-DPKG_CONFIG_EXECUTABLE={tools['pkg_config']['path']}",
    ]
    commands = [_run_build(configure_argv, timeout)]
    failures: dict[str, str] = {}
    configure_failed = (
        commands[-1]["exit_code"] != 0 or commands[-1]["timed_out"]
    )
    if configure_failed:
        failures.update(
            {role: "official native sample configure failed" for role in reports}
        )
    for role in () if configure_failed else reports:
        target = SAMPLES[role][0]
        commands.append(
            _run_build(
                [
                    str(tools["cmake"]["path"]),
                    "--build",
                    str(build_root),
                    "--target",
                    target,
                    "--parallel",
                    "2",
                ],
                timeout,
            )
        )
        if commands[-1]["exit_code"] != 0 or commands[-1]["timed_out"]:
            failures[role] = f"official {target} report target build failed"
    _verify_tool_identities(tools)
    binaries: dict[str, Any] = {}
    locations = {
        "encoder": build_root / "AppEncode" / "AppEncCuda" / "AppEncCuda",
        "decoder": build_root / "AppDecode" / "AppDec" / "AppDec",
    }
    for role in reports:
        if role in failures:
            continue
        binary = locations[role]
        resolved = binary.resolve(strict=True)
        if (
            not resolved.is_relative_to(workspace)
            or not resolved.is_file()
            or resolved.stat().st_mode & 0o111 == 0
        ):
            raise ValueError(f"official {SAMPLES[role][0]} build output is invalid")
        identity = _snapshot(
            resolved,
            label=SAMPLES[role][0],
            executable=True,
            limit=MAX_BINARY,
        )[1]
        try:
            linkage = _live_linkage(resolved, role, min(timeout, 60))
        except ValueError as exc:
            failures[role] = str(exc)
            continue
        binaries[role] = {
            "path": resolved,
            "identity": identity,
            "linkage": linkage,
        }
    evidence_tools = {
        name: {
            "path": str(record["path"]),
            "identity": record["identity"],
            **({"name": record["name"]} if name == "generator" else {}),
        }
        for name, record in tools.items()
        if name in {"cmake", "cxx", "pkg_config", "nvcc", "generator"}
    }
    return binaries, {
        "workspace": str(workspace),
        "build_root": str(build_root.resolve(strict=True)),
        "requested_targets": [SAMPLES[role][0] for role in reports],
        "target_failures": failures,
        "tools": evidence_tools,
        "nvcc_version_command": tools["nvcc_version_command"],
        "commands": [_command_record(item) for item in commands],
    }


def _authenticate_local(
    workspace_path: Path,
    output_path: Path,
    gpu: int,
    reports: tuple[str, ...],
    timeout: int,
) -> dict[str, Any]:
    package = _local_package(timeout)
    sdk_root = Path(package["sdk_root"])
    workspace_requested = workspace_path.expanduser().absolute()
    output_requested = output_path.expanduser().absolute()
    if workspace_requested == sdk_root or workspace_requested.is_relative_to(sdk_root):
        raise ValueError("native report workspace must be outside the installed SDK tree")
    output_parent = output_requested.parent.resolve(strict=True)
    if (output_parent / output_requested.name).is_relative_to(sdk_root):
        raise ValueError("native report output must be outside the installed SDK tree")
    tools = _local_tools(timeout, reports)
    workspace = _fresh_workspace(workspace_path)
    binaries, build = _local_build(package, tools, workspace, reports, timeout)
    return {
        "source": "local_package_build",
        "gpu": gpu,
        "package": package,
        "build": build,
        "binaries": binaries,
        "tools": tools,
    }


def _authenticate(  # pylint: disable=too-many-branches,too-many-statements
    path: Path, gpu: int, reports: tuple[str, ...], timeout: int
) -> dict[str, Any]:
    raw, identity, data = _load(path, label="native verification")
    if data.get("kind") != VERIFY_KIND or data.get("schema_version") != VERIFY_SCHEMA:
        raise ValueError(f"input must be {VERIFY_KIND} schema {VERIFY_SCHEMA}")
    if not isinstance(data.get("ready"), bool) or data.get("status") not in (
        "operation_verified", "operation_failed", "unknown"
    ):
        raise ValueError("native verification readiness/status fields are malformed")
    if data.get("ready") is not True or data.get("status") != "operation_verified":
        raise _Unavailable("native verification is not operation_verified")
    if data.get("software_fallback") is not False:
        raise ValueError("ready native verification must prove software_fallback=false")
    operations = data.get("hardware_operations")
    if not isinstance(operations, dict) or any(
        not isinstance(operations.get(role), dict)
        or operations[role].get("status") != "operation_verified"
        for role in ("encode", "decode")
    ):
        raise ValueError("ready native verification lacks both official operation proofs")
    environment = data.get("environment")
    if not isinstance(environment, dict) or environment.get("selected_gpu") != gpu:
        raise ValueError("native verification selected GPU does not match --gpu")
    env_raw, env_identity, env_data = _load(
        Path(environment.get("identity", {}).get("path", "")), label="environment")
    _same_identity(env_identity, environment.get("identity"), label="environment")
    if (
        hashlib.sha256(_dumps(env_data, compact=True)).hexdigest()
        != environment.get("canonical_sha256")
    ):
        raise ValueError("environment canonical JSON digest does not match")
    if (
        env_data.get("kind") != "nvcodec-environment"
        or env_data.get("schema_version") != ENVIRONMENT_SCHEMA
        or env_data.get("mode") != "live"
        or env_data.get("selected_gpu") != gpu
    ):
        raise ValueError(
            f"bound environment is not the selected live schema-{ENVIRONMENT_SCHEMA} target"
        )
    native = environment.get("native")
    package = native.get("package") if isinstance(native, dict) else None
    version = package.get("version") if isinstance(package, dict) else None
    if (
        not _public_sdk_release(version)
        or package.get("name") != PACKAGE
        or package.get("status") != "installed"
    ):
        raise ValueError("native verification package is not public nvidia-video-codec-sdk 13.0.x")
    ownership = data.get("package_ownership")
    expected_ownership = {
        "status": "verified", "package": PACKAGE, "declared_version": version,
        "live_version": version, "sdk_root_owned": True, "verify_silent": True,
    }
    if not isinstance(ownership, dict) or any(
        ownership.get(key) != value for key, value in expected_ownership.items()
    ):
        raise ValueError("native package ownership is not verified")
    query = _run(["/usr/bin/dpkg-query", "-W", "-f=${db:Status-Abbrev}\\t${Version}",
                  PACKAGE], min(timeout, 60))
    fields = query["stdout_bytes"].decode(errors="replace").strip().split("\t")
    if (
        query["exit_code"] != 0 or len(fields) != 2
        or not fields[0].startswith("ii") or fields[1] != version
    ):
        raise ValueError("live native package version does not match verification")
    verify = _run(["/usr/bin/dpkg", "--verify", PACKAGE], min(timeout, 60))
    if verify["exit_code"] != 0 or verify["stdout_bytes"].strip():
        raise ValueError("live dpkg verification is not silent")
    samples, binaries = data.get("samples"), {}
    if not isinstance(samples, dict):
        raise ValueError("native verification has no samples block")
    for role in reports:
        record = samples.get(role)
        recorded_linkage = record.get("linkage") if isinstance(record, dict) else None
        recorded_libraries = (
            recorded_linkage.get("libraries") if isinstance(recorded_linkage, dict) else None
        )
        if (
            not isinstance(recorded_libraries, dict)
            or recorded_linkage.get("status") != "verified"
            or any(
                not isinstance(recorded_libraries.get(name), dict)
                or recorded_libraries[name].get("status") != "resolved"
                for name in REQUIRED_LIBS[role]
            )
        ):
            raise ValueError(f"recorded {SAMPLES[role][0]} linkage is not verified")
        expected = record.get("identity")
        binary = Path(record.get("binary", ""))
        _same_identity(_snapshot(binary, label=SAMPLES[role][0], executable=True,
                                 limit=MAX_BINARY)[1], expected,
                       label=SAMPLES[role][0])
        if expected.get("path") != str(binary.expanduser().absolute().resolve(strict=True)):
            raise ValueError(f"{SAMPLES[role][0]} binary and identity paths differ")
        _live_linkage(binary, role, min(timeout, 60))
        binaries[role] = {"path": binary, "identity": expected}
    return {
        "source": "setup_verification",
        "raw": raw,
        "identity": identity,
        "data": data,
        "environment_raw": env_raw,
        "environment_identity": env_identity,
        "environment_data": env_data,
        "version": version,
        "binaries": binaries,
    }


def _report(role: str, binary: dict[str, Any], gpu: int, timeout: int) -> dict[str, Any]:
    path, expected = binary["path"], binary["identity"]
    initial = _snapshot(path, label=SAMPLES[role][0], executable=True,
                        limit=MAX_BINARY)[1]
    _same_identity(initial, expected, label=SAMPLES[role][0])
    evidence = _run([str(path), SAMPLES[role][1]], timeout)
    terminal = _snapshot(path, label=SAMPLES[role][0], executable=True,
                         limit=MAX_BINARY)[1]
    _same_identity(terminal, initial, label=SAMPLES[role][0])
    streams = {"stdout": _stream(evidence["stdout_bytes"]),
               "stderr": _stream(evidence["stderr_bytes"])}
    stdout, stderr = (streams[name]["content"] or "" for name in ("stdout", "stderr"))
    variant, records, selected, parse_reason = (
        _encoder_parse(stdout, gpu) if role == "encoder" else _decoder_parse(stdout, gpu))
    markers = _markers(stdout, stderr)
    reasons = []
    if evidence["timed_out"]:
        reasons.append("official sample report timed out")
    if evidence["exit_code"] != 0:
        reasons.append(f"official sample exited {evidence['exit_code']}")
    if markers:
        reasons.append("explicit failure marker(s): " + ", ".join(markers))
    if any(item["capture_status"] != "complete" for item in streams.values()):
        reasons.append(f"captured stream exceeded {MAX_STREAM} bytes")
    if parse_reason:
        reasons.append(parse_reason)
    return {"status": "unknown" if reasons else "completed", "format_variant": variant,
            "evidence_classification": _classification(role),
            "authority": f"official {SAMPLES[role][0]} aggregate report",
            "claim_scope": (
                "raw sample-reported values only; neither API/product support nor"
                " operation evidence"
            ),
            "command": evidence["argv"], "cwd": evidence["cwd"],
            "exit_code": evidence["exit_code"], "timed_out": evidence["timed_out"],
            "started_at": evidence["started_at"], "ended_at": evidence["ended_at"],
            "duration_seconds": evidence["duration_seconds"],
            "selected_gpu": gpu, "selected_gpu_present": selected,
            "reported_gpus": records, "failure_markers": markers,
            "reason": "; ".join(reasons) if reasons else None, "streams": streams,
            "stdout_tail": streams["stdout"]["tail"], "stderr_tail": streams["stderr"]["tail"],
            "binary_integrity": {"initial": initial, "terminal": terminal}}


def _empty(role: str, status: str, reason: str | None = None) -> dict[str, Any]:
    return {"status": status, "format_variant": None,
            "evidence_classification": _classification(role), "reason": reason}


def _document(  # pylint: disable=too-many-branches,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    path: Path | None,
    workspace: Path | None,
    output: Path,
    gpu: int,
    reports: tuple[str, ...],
    timeout: int,
) -> tuple[dict[str, Any], int]:
    if path is None:
        if workspace is None:
            raise ValueError(
                "--workspace is required when --native-verification is omitted"
            )
        try:
            auth = _authenticate_local(workspace, output, gpu, reports, timeout)
            reason = None
        except _Unavailable as exc:
            auth, reason = None, str(exc)
    elif not os.path.lexists(path.expanduser()):
        reason = f"native verification is unavailable: {path}"
        auth = None
    else:
        try:
            auth = _authenticate(path, gpu, reports, timeout)
            reason = None
        except _Unavailable as exc:
            auth, reason = None, str(exc)
    aggregates = {role: _empty(role, "not_requested") for role in SAMPLES}
    if auth is None:
        for role in reports:
            aggregates[role] = _empty(role, "unknown", reason)
        native = {"initial_identity": None, "terminal_identity": None,
                  "kind": VERIFY_KIND, "schema_version": VERIFY_SCHEMA,
                  "ready": False, "status": "unknown"}
    elif auth["source"] == "setup_verification":
        for role in reports:
            aggregates[role] = _report(role, auth["binaries"][role], gpu, timeout)
        terminal_raw, terminal_identity, _ = _load(path, label="native verification")
        env_raw, env_terminal, _ = _load(
            Path(auth["environment_identity"]["path"]), label="environment")
        if terminal_raw != auth["raw"] or env_raw != auth["environment_raw"]:
            raise ValueError("native verification or environment changed during reports")
        native = {"initial_identity": auth["identity"], "terminal_identity": terminal_identity,
                  "canonical_sha256": hashlib.sha256(
                      _dumps(auth["data"], compact=True)).hexdigest(),
                  "kind": VERIFY_KIND, "schema_version": VERIFY_SCHEMA, "ready": True,
                  "status": "operation_verified",
                  "environment_identity": env_terminal,
                  "package": {"name": PACKAGE, "version": auth["version"]},
                  "samples": {
                      role: {
                          "binary": str(auth["binaries"][role]["path"]),
                          "identity": auth["binaries"][role]["identity"],
                      }
                      for role in reports
                  }}
    else:
        for role in reports:
            if role in auth["binaries"]:
                aggregates[role] = _report(role, auth["binaries"][role], gpu, timeout)
            else:
                aggregates[role] = _empty(
                    role,
                    "unknown",
                    auth["build"]["target_failures"].get(
                        role, f"{SAMPLES[role][0]} report binary is unavailable"
                    ),
                )
        terminal_package = _local_package(timeout)
        if any(
            terminal_package.get(field) != auth["package"].get(field)
            for field in ("name", "version", "sdk_root", "owned_path_count", "source_tree")
        ):
            raise ValueError("native package identity changed during reports")
        _verify_tool_identities(auth["tools"])
        native = {
            "initial_identity": None,
            "terminal_identity": None,
            "kind": VERIFY_KIND,
            "schema_version": VERIFY_SCHEMA,
            "ready": False,
            "status": "not_supplied",
        }
    counts = {name: sum(item["status"] == name for item in aggregates.values())
              for name in ("completed", "unknown", "not_requested")}
    status = ("complete" if counts["completed"] == len(reports) else
              "partial" if counts["completed"] else "unknown")
    result = {"kind": KIND, "schema_version": SCHEMA,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": status, "authority": "native_official_samples", "gpu": gpu,
              "requested_reports": list(reports), "native_verification": native,
              "aggregate_encoder_capabilities": aggregates["encoder"],
              "aggregate_decoder_capabilities": aggregates["decoder"], "summary": counts}
    if auth is not None and auth["source"] == "local_package_build":
        result["local_authentication"] = {
            "status": "authenticated",
            "package": auth["package"],
            "build": auth["build"],
            "samples": {
                role: {
                    "binary": str(auth["binaries"][role]["path"]),
                    "identity": auth["binaries"][role]["identity"],
                    "linkage": auth["binaries"][role]["linkage"],
                }
                for role in reports
                if role in auth["binaries"]
            },
        }
        if auth["build"]["target_failures"]:
            add_setup_remediation(
                result,
                surface="native",
                reason="; ".join(
                    auth["build"]["target_failures"][role]
                    for role in reports
                    if role in auth["build"]["target_failures"]
                ),
            )
    elif auth is None:
        add_setup_remediation(result, surface="native", reason=reason)
    return result, 0 if status == "complete" else 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(3)


def main() -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument(
        "--native-verification",
        type=Path,
        help=(
            "Optional setup-produced schema-1.5 verification; when omitted, "
            "authenticate the installed package and build requested report targets"
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Fresh explicit build workspace required without --native-verification",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--report", action="append", choices=tuple(SAMPLES))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not sys.flags.ignore_environment or not sys.flags.no_user_site:
            raise ValueError("invoke this producer with isolated Python: python3 -I")
        if args.gpu < 0 or args.timeout <= 0:
            raise ValueError("--gpu must be non-negative and --timeout must be positive")
        if os.path.lexists(args.output.expanduser()):
            raise FileExistsError(f"output already exists: {args.output}")
        reports = tuple(dict.fromkeys(args.report or SAMPLES))
        if (args.native_verification is None) == (args.workspace is None):
            raise ValueError(
                "provide --native-verification or --workspace, but not both"
            )
        result, code = _document(
            args.native_verification,
            args.workspace,
            args.output,
            args.gpu,
            reports,
            args.timeout,
        )
        rendered = _dumps(result)
        exclusive_write_bytes(args.output, rendered)
        print(rendered.decode("utf-8"), end="")
        return code
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result = {"kind": KIND + "-error", "schema_version": SCHEMA, "status": "error",
                  "generated_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
        rendered = _dumps(result)
        print(rendered.decode("utf-8"), end="")
        if not os.path.lexists(args.output.expanduser()):
            try:
                exclusive_write_bytes(args.output, rendered)
            except (OSError, ValueError):
                pass
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
