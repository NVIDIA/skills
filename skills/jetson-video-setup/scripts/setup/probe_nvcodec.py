#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the frozen read-only ``nvcodec-environment`` 1.2 artifact. Native/PyNv remain
independent; wheel RECORD attestation stays in verify_pynvc_sample.py."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
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
    AUXILIARY_MODULE, AUXILIARY_ROLE, AUXILIARY_STEM, MINIMUM_JETSON_LINUX,
    MINIMUM_VERSION_TEXT, PRIMARY_STEMS, PYNVC_PACKAGE,
    associated_extensions, environment_contract_errors, extension_stem,
    file_identity,
    owned_extension, parse_apt_sources, parse_jetson_release, public_candidate_binding,
    pycuda_bootstrap_packages,
    query_pynvc_capabilities, read_json, require_isolated, root_evidence, run_command, sha256_bytes,
    surface_contract_errors, system_executable, utc_now, write_new_json,
)
from verify_pynvc_sample import RegistryNotReady, load_pynvc_registry  # noqa: E402

# pylint: enable=wrong-import-position

KIND, SCHEMA_VERSION, SURFACES = "nvcodec-environment", "1.2", ("native", "pynvc")
ERROR_KIND = "nvcodec-environment-error"
VALIDATION_VERSION, VALIDATION_KIND = "1.0", "nvcodec-environment-validation"
VALIDATION_ERROR_KIND = "nvcodec-environment-validation-error"
NATIVE_PACKAGE, NATIVE_SDK_ROOT = "nvidia-video-codec-sdk", "/opt/nvidia/video-codec-sdk"
_TOOL_ALIASES = {"cxx": "g++", "pkg_config": "pkg-config"}
CMAKE_GENERATORS = (("ninja", "Ninja"), ("make", "Unix Makefiles"))
CODECS = ("h264", "hevc", "av1")
APPDEC_CONSUMER = "Samples/AppDecode/AppDec"
APPDEC_MODULES = ("libavcodec", "libavformat", "libavutil", "libswresample")
PYCUDA_VERSION, NUMPY_MINIMUM = "2026.1", (1, 24)
TORCH_VERSION, TORCH_CUDA_BUILD = "2.9.1+cu130", "13.0"

_RELEASE_SOURCE = Path("/etc/nv_tegra_release")
_APT_SOURCES_LIST = Path("/etc/apt/sources.list")
_APT_SOURCES_DIR = Path("/etc/apt/sources.list.d")
_APT_BYPASS = re.compile(
    r"(?:Acquire::Allow(?:Insecure|Weak|DowngradeToInsecure)Repositories"
    r"|APT::Get::AllowUnauthenticated)\s+\"?(?:1|true|yes)\"?", re.I)
_POLICY_FIELD = re.compile(r"^\s*(Installed|Candidate):[ \t]*(\S+)[ \t]*$", re.M)
_POLICY_ORIGIN = re.compile(r"^\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+Packages\s*$")
_POLICY_VERSION = re.compile(r"^\s+(?:\*\*\*\s+)?(\S+)\s+\d+\s*$")
_NVCC_RELEASE = re.compile(r"\brelease\s+(\d+(?:\.\d+)+)")
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+")
_DIGITS = re.compile(r"\d+")
_CUDA_PREFIXES = {"PATH": ("bin",), "CPATH": ("include",), "LIBRARY_PATH": ("lib64", "lib")}
_COMMAND_EVIDENCE: list[dict[str, Any]] = []
_DEPENDENCIES: dict[str, Any] = {}
_PROBE_WARNING = ("this probe executes no encode or decode operation; API capability fields"
                  " are not operational verification")
_PYTHON_PROBE = """
import json, os, sysconfig
try:
    import ensurepip, venv; venv.EnvBuilder(with_pip=True); usable = True
except BaseException:
    usable = False
include = os.path.join(sysconfig.get_paths()["include"], "Python.h")
print(json.dumps({"venv": usable, "headers": os.path.isfile(include)}))
"""
_COMMAND_TIMEOUT = 30
_DELEGATE_TIMEOUT = 900
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


class _DuplicateKeyError(ValueError):
    """A strict-JSON validation defect that keeps the legacy negative-result envelope."""


def _run(argv: list[str], timeout: int = _COMMAND_TIMEOUT) -> dict[str, Any]:
    """Run one bounded read-only command, recording argv and exit status as command evidence."""
    try:
        done = run_command(argv, timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "error", "returncode": None, "stdout": "", "stderr": str(exc)[:512]}
    else:
        result = {"status": "ok" if done.returncode == 0 else "error",
                  "returncode": done.returncode,
                  "stdout": done.stdout.decode("utf-8", "replace"),
                  "stderr": done.stderr.decode("utf-8", "replace")[:512]}
    _COMMAND_EVIDENCE.append({"argv": list(argv), "status": result["status"],
                              "returncode": result["returncode"], "mutating": False})
    return result


def _tool(name: str) -> str | None:
    """Return the one canonical system path for a tool, or None when there is not one."""
    try:
        return system_executable(name)
    except FileNotFoundError:
        return None


def _identity(path: Any) -> dict[str, Any] | None:
    """Return {path, size_bytes, sha256} for one real file, or None when unreadable."""
    if path is None:
        return None
    try:
        return file_identity(os.path.realpath(os.fspath(path)), label="file")
    except (OSError, TypeError, ValueError):
        return None


def _read_text(path: Any, limit: int = 4096) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _member(value: Any, key: str) -> dict[str, Any]:
    """Read one nested object, tolerating an absent or malformed member."""
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, dict) else {}


def _absolute(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def _command_version(path: Any, pattern: re.Pattern[str] | None = None) -> str | None:
    """Read one tool's own reported version; absent output yields None, never a guess."""
    if path is None:
        return None
    result = _run([str(path), "--version"], timeout=15)
    output = result["stdout"] or result["stderr"]
    if result["status"] != "ok" or not output:
        return None
    if pattern is not None:
        match = pattern.search(output)
        return match.group(1) if match else None
    found = _VERSION_TOKEN.findall(output.splitlines()[0])
    return found[-1] if found else None


def _release_number(value: Any) -> list[int]:
    return [int(part) for part in _DIGITS.findall(str(value))]


def _release_gate(record: dict[str, Any]) -> dict[str, Any]:
    """Own the 38.5+ policy on the artifact that declares it: shared ``parse_jetson_release``
    returns structure only, so the decision is published once rather than re-derived."""
    major, minor = record.get("release_major"), record.get("release_minor")
    if not isinstance(major, int) or not isinstance(minor, int):
        status, reason = "unknown", (f"no parseable Jetson Linux release line in"
                                     f" {_RELEASE_SOURCE}; the requirement was not evaluated")
    elif (major, minor) >= MINIMUM_JETSON_LINUX:
        status, reason = "compatible", f"Jetson Linux {record['version']} satisfies the minimum"
    else:
        status, reason = "unsupported", (f"Jetson Linux {record['version']} is older than the"
                                         " documented minimum")
    return {"status": status, "minimum_version": MINIMUM_VERSION_TEXT,
            "reason": f"{reason} ({MINIMUM_VERSION_TEXT}+).", "revision": record.get("revision")}


def _nvidia_smi(gpu: int) -> dict[str, Any]:
    """Observe the canonical nvidia-smi; every ``gpus`` row carries its own driver ``index``, so
    a consumer selects the requested GPU by that field and never by list position."""
    executable = _tool("nvidia-smi")
    record: dict[str, Any] = {"status": "unavailable", "executable": executable, "gpus": [],
                              "selected_gpu": gpu, "gpu_name": None, "driver_version": None,
                              "exit_code": None}
    if executable is None:
        return record
    result = _run([executable, "--query-gpu=index,name,driver_version",
                   "--format=csv,noheader,nounits"])
    record["exit_code"] = result["returncode"]
    record["status"] = "observed" if result["status"] == "ok" else "error"
    for line in result["stdout"].splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        row = {"index": int(fields[0]), "name": fields[1] or None,
               "driver_version": fields[2] or None}
        record["gpus"].append(row)
        if row["index"] == gpu:
            record["gpu_name"], record["driver_version"] = row["name"], row["driver_version"]
    return record


def _platform_record() -> dict[str, Any]:
    """Publish the 1.2 ``platform`` identity and release gate: ``jetson`` is the observed
    presence of the release file and ``machine`` the kernel's own reported architecture."""
    record = parse_jetson_release(_read_text(_RELEASE_SOURCE, 1024))
    uname = os.uname()
    return {
        "system": uname.sysname, "release": uname.release, "machine": uname.machine,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "jetson": _RELEASE_SOURCE.exists(),
        "jetson_linux": {"status": "observed" if record.get("release_line") else "unknown",
                         **record, "release_source": str(_RELEASE_SOURCE),
                         "compatibility": _release_gate(record)},
    }


def _apt_sources() -> list[dict[str, Any]]:
    """Parse every configured binary APT source together with its exact file hash."""
    paths = [_APT_SOURCES_LIST]
    if _APT_SOURCES_DIR.is_dir():
        paths.extend(sorted(path for path in _APT_SOURCES_DIR.iterdir()
                            if path.suffix in {".list", ".sources"}))
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        records.append({"path": str(path), "sha256": sha256_bytes(raw),
                        "entries": parse_apt_sources(path, raw.decode("utf-8", "replace"))})
    return records


def _signature_enforced(sources: list[dict[str, Any]]) -> bool:
    """Require a readable apt configuration with no authentication bypass anywhere."""
    executable = _tool("apt-config")
    if executable is None:
        return False
    result = _run([executable, "dump"])
    return bool(result["status"] == "ok" and _APT_BYPASS.search(result["stdout"]) is None
                and not any(entry.get("trust_bypass") for item in sources
                            for entry in item["entries"]))


def _bind_origin(origin: dict[str, Any], sources: list[dict[str, Any]],
                 enforced: bool) -> dict[str, Any]:
    """Bind one candidate origin to the configured source files that declare it. The digest list
    is deliberately not deduplicated: the core requires digests and records to be one multiset."""
    suite, _, component = str(origin["suite_component"]).partition("/")
    expected = (str(origin["uri"]), suite, component, False)
    matches = [{"path": item["path"], "sha256": item["sha256"], "entry": entry}
               for item in sources for entry in item["entries"]
               if (entry.get("uri"), entry.get("suite"), entry.get("component"),
                   entry.get("trust_bypass")) == expected]
    if enforced and matches:
        origin["authentication"] = "apt-signature-chain"
        origin["configured_source_sha256"] = sorted(item["sha256"] for item in matches)
        origin["configured_sources"] = matches
    return origin


def _policy_origins(stdout: str, candidate: str, sources: list[dict[str, Any]],
                    enforced: bool) -> list[dict[str, Any]]:
    """Read the apt-cache version table and return only the candidate version's origins."""
    origins: list[dict[str, Any]] = []
    active = False
    for line in stdout.splitlines():
        match = _POLICY_ORIGIN.match(line)
        if match is None:
            version = _POLICY_VERSION.match(line)
            active = version is not None and version.group(1) == candidate
        elif active:
            origins.append(_bind_origin(
                {"priority": int(match.group(1)), "uri": match.group(2).rstrip("/"),
                 "suite_component": match.group(3), "architecture": match.group(4),
                 "authentication": "unverified"}, sources, enforced))
    return origins


def _apt_candidate(apt_cache: str | None, package: str, sources: list[dict[str, Any]],
                   enforced: bool) -> dict[str, Any]:
    """Observe one package's installed/candidate versions and its bound public origin; the exact
    argv and exit code are recorded so the planner can re-check where the binding came from."""
    record: dict[str, Any] = {"package": package, "query_status": "unavailable",
                              "query_argv": None, "query_exit_code": None, "installed": None,
                              "candidate": None, "candidate_origins": []}
    if apt_cache is not None:
        argv = [apt_cache, "policy", package]
        result = _run(argv)
        found = {name: None if value == "(none)" else value
                 for name, value in _POLICY_FIELD.findall(result["stdout"])}
        candidate = found.get("Candidate")
        record.update({"query_status": result["status"], "query_argv": argv,
                       "query_exit_code": result["returncode"],
                       "installed": found.get("Installed"), "candidate": candidate,
                       "candidate_origins": _policy_origins(
                           result["stdout"], candidate, sources, enforced) if candidate else []})
    # The single APT trust decision lives in the shared core: an internal mirror, an
    # unsigned or trust-bypassed source, or a suite outside exact rNN.N/main yields None.
    record["public_origin"] = public_candidate_binding(record)
    return record


def _apt_probe() -> dict[str, Any]:
    """Observe local APT metadata only; the probe never refreshes an index or hits the network."""
    tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    packages = (NATIVE_PACKAGE, "cuda-toolkit", "pkg-config", "python3-venv", f"{tag}-venv",
                "python3-dev", f"{tag}-dev")
    sources = _apt_sources()
    enforced = _signature_enforced(sources)
    apt_cache = _tool("apt-cache")
    candidates = {
        name: _apt_candidate(apt_cache, name, sources, enforced) for name in packages
    }
    for build_package in pycuda_bootstrap_packages(candidates["cuda-toolkit"].get("candidate")):
        candidates[build_package] = _apt_candidate(
            apt_cache, build_package, sources, enforced
        )
    return {
        "apt_cache": apt_cache, "signature_enforced": enforced,
        "source_files": [item["path"] for item in sources],
        "sources": [{"path": item["path"], "sha256": item["sha256"]} for item in sources],
        "candidates": candidates,
    }


def _dpkg_package(name: str) -> dict[str, Any]:
    """Report one Debian package as {name, status, version} from dpkg-query alone."""
    executable = _tool("dpkg-query")
    if executable is None:
        return {"name": name, "status": "unknown", "version": None}
    result = _run([executable, "-W", "-f=${db:Status-Abbrev}\t${Version}", name])
    fields = result["stdout"].split("\t")
    installed = (result["status"] == "ok" and len(fields) == 2
                 and fields[0].startswith("ii") and bool(fields[1].strip()))
    return {"name": name, "status": "installed" if installed else "missing",
            "version": fields[1].strip() if installed else None}


def _cuda_root() -> Path | None:
    """Return the realpath of the selected CUDA root: the /usr/local/cuda link, else newest."""
    versioned = sorted(Path("/usr/local").glob("cuda-*"),
                       key=lambda path: _release_number(path.name), reverse=True)
    for candidate in (Path("/usr/local/cuda"), *versioned):
        if (candidate / "bin" / "nvcc").is_file():
            return Path(os.path.realpath(candidate))
    return None


def _cuda_marker_version(root: Path) -> str | None:
    try:
        data = read_json(root / "version.json")
    except (OSError, ValueError):
        data = None
    entry = data.get("cuda") if isinstance(data, dict) else None
    version = entry.get("version") if isinstance(entry, dict) else None
    if isinstance(version, str) and version:
        return version
    match = _VERSION_TOKEN.search(_read_text(root / "version.txt", 256))
    return match.group(0) if match else None


def _nvcc_discovery(root: Any) -> dict[str, Any]:
    return {"root": str(root) if root else None,
            "nvcc": str(root / "bin" / "nvcc") if root else None,
            "provenance": "the /usr/local/cuda link, else the newest /usr/local/cuda-* with nvcc"}


def _absent_cuda(root: Any = None, version: Any = None) -> dict[str, Any]:
    return {"status": "absent", "version": version, "root": str(root) if root else None,
            "nvcc_discovery": _nvcc_discovery(root),
            "environment_prefixes": {name: [] for name in _CUDA_PREFIXES}}


def _cuda() -> dict[str, Any]:
    """Publish the one CUDA observation; rule 9 requires status and version. The vocabulary is
    installed/absent because every native consumer and the planner fail closed on anything else.
    ``cuda_toolkit`` and the native surface carry this one observation, so they cannot disagree;
    the prefixes are the plan-time PyCUDA source-build environment."""
    root = _cuda_root()
    if root is None:
        return _absent_cuda()
    version = (
        _dpkg_package("cuda-toolkit")["version"]
        or _cuda_marker_version(root)
        or _command_version(root / "bin" / "nvcc", _NVCC_RELEASE)
    )
    if not version:
        return _absent_cuda(root)
    return {"status": "installed", "version": version, "root": str(root),
            "nvcc_discovery": _nvcc_discovery(root),
            "environment_prefixes": {
                name: [str(root / part) for part in parts if (root / part).is_dir()]
                for name, parts in _CUDA_PREFIXES.items()}}


def _cuda_probe() -> dict[str, Any]:
    """Observe CUDA exactly once; a failure here can never reach either surface."""
    try:
        return _cuda()
    except Exception:  # pylint: disable=broad-exception-caught
        return _absent_cuda()


def _tool_record(path: Any, pattern: re.Pattern[str] | None = None) -> dict[str, Any] | None:
    identity = _identity(path)
    version = _command_version(identity["path"] if identity else None, pattern)
    if identity is None or version is None:
        return None
    return {"path": identity["path"], "version": version, "sha256": identity["sha256"]}


def _native_tools(cuda_root: Any) -> dict[str, Any]:
    """Emit the five required native tools; `generator.name` is the CMake generator name (R4)."""
    nvcc = str(Path(str(cuda_root)) / "bin" / "nvcc") if _absolute(cuda_root) else None
    records: dict[str, Any] = {
        "cmake": _tool_record(_tool("cmake")), "cxx": _tool_record(_tool("c++")),
        "nvcc": _tool_record(nvcc, _NVCC_RELEASE), "generator": None,
        "pkg_config": _tool_record(_tool("pkg-config"))}
    for executable, name in CMAKE_GENERATORS:
        record = _tool_record(_tool(executable))
        if record is not None:
            records["generator"] = {**record, "name": name, "executable": executable}
            break
    return {name: record for name, record in records.items() if record is not None}


def _pkg_config_module(pkg_config: str | None, name: str) -> dict[str, Any]:
    if pkg_config is None:
        return {"status": "unknown", "version": None}
    result = _run([pkg_config, "--modversion", name], timeout=15)
    version = result["stdout"].strip().splitlines()[0].strip() if result["stdout"].strip() else ""
    ready = result["status"] == "ok" and bool(version)
    return {"status": "available" if ready else "missing", "version": version or None}


def _build_prerequisites() -> dict[str, Any]:
    """Preflight the exact pkg-config modules that gate the official AppDec target."""
    pkg_config = _tool("pkg-config")
    modules = {name: _pkg_config_module(pkg_config, name) for name in APPDEC_MODULES}
    unresolved = [name for name in APPDEC_MODULES if modules[name]["status"] != "available"]
    return {"status": "complete" if not unresolved else "incomplete",
            "unresolved_modules": unresolved, "consumer": APPDEC_CONSUMER,
            "required_modules": list(APPDEC_MODULES), "modules": modules}


def _native_surface(cuda: dict[str, Any]) -> dict[str, Any]:
    """Produce and independently validate the native surface published as ``native_sdk``."""
    base = Path(NATIVE_SDK_ROOT)
    roots: list[str] = []
    versioned: list[str] = []
    if base.is_dir() and not base.is_symlink():
        roots.append(os.path.realpath(base))
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.is_symlink():
                resolved_child = os.path.realpath(child)
                roots.append(resolved_child)
                if re.fullmatch(r"13\.0(?:\.\d+)*", child.name):
                    versioned.append(resolved_child)
    package = _dpkg_package(NATIVE_PACKAGE)
    evidence = [root_evidence(root) for root in versioned]
    complete_roots = [item["root"] for item in evidence if item["status"] == "complete"]
    resolved = complete_roots[0] if len(versioned) == len(complete_roots) == 1 else None
    complete = resolved is not None and package["status"] == "installed"
    status = ("installed" if complete else "partial"
              if package["status"] == "installed" or roots else "missing")
    surface = {
        "installed": False, "status": status,
        "package": package, "sdk_root": resolved,
        "roots": roots, "root_evidence": evidence,
        "headers": sorted(name for item in evidence for name in item["required_files"][:3]
                          if os.path.isfile(name)),
        "sample_roots": sorted(f"{root}/Samples" for root in roots
                               if Path(root, "Samples").is_dir()),
        "complete_roots": complete_roots,
        "reason": ("one coherent versioned SDK root carries every required member" if complete
                   else "no single complete versioned Video Codec SDK root was observed"),
        "build_prerequisites": _build_prerequisites(), "cuda": cuda,
        "tools": _native_tools(cuda.get("root")),
    }
    return _sealed(surface, surface_contract_errors("native", surface))


def _dependency(name: str, ready: Any = None) -> dict[str, Any]:
    try:
        version: str | None = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    satisfied = version is not None and (bool(ready(version)) if callable(ready) else True)
    return {"status": "installed" if version is not None else "missing", "version": version,
            "ready": satisfied, "requirement_satisfied": satisfied}


def _torch_dependency() -> dict[str, Any]:
    """Report the CUDA-enabled PyTorch the official decode sample imports."""
    record = _dependency("torch")
    if record["status"] != "installed":
        return {**record, "cuda_build": None, "cuda_available": None,
                "sample_readiness": "not_ready"}
    try:
        module = importlib.import_module("torch")
        build = getattr(getattr(module, "version", None), "cuda", None)
        available = bool(module.cuda.is_available())
    except Exception:  # pylint: disable=broad-exception-caught
        build, available = None, None
    ready = record["version"] == TORCH_VERSION and build == TORCH_CUDA_BUILD and available is True
    return {**record, "cuda_build": build, "cuda_available": available, "ready": ready,
            "requirement_satisfied": ready,
            "sample_readiness": "ready" if ready else "not_ready"}


def _dependencies() -> dict[str, Any]:
    """Observe the official-sample Python dependencies once, memoized. The first call must follow
    PyNvVideoCodec classification: importing PyTorch can rewrite LD_LIBRARY_PATH."""
    if not _DEPENDENCIES:
        _DEPENDENCIES.update({
            "numpy": _dependency(
                "numpy", lambda value: _release_number(value)[:2] >= list(NUMPY_MINIMUM)),
            "pycuda": _dependency("pycuda", lambda value: value == PYCUDA_VERSION),
            "torch": _torch_dependency(),
        })
    return _DEPENDENCIES


def _owned_member(resolved: str, distribution: Any,
                  label: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Bind one owned extension to exactly one selected-distribution member and hash it, proving
    membership only -- never that its wheel ``RECORD`` row still authenticates."""
    entries: list[str] = []
    for item in getattr(distribution, "files", None) or ():
        try:
            located = os.path.realpath(os.fspath(distribution.locate_file(item)))
        except (OSError, TypeError, ValueError):
            continue
        if located == resolved:
            entries.append(str(item))
    if len(entries) != 1:
        return None, [f"{label} must be one member of the selected {PYNVC_PACKAGE} distribution;"
                      f" matched {sorted(entries)}"]
    identity = _identity(resolved)
    if identity is None:
        return None, [f"{label} at {resolved} could not be hashed"]
    return {"path": identity["path"], "sha256": identity["sha256"], "loaded_path": resolved,
            "distribution_entry": entries[0]}, []


def _selected_primary(module: Any, distribution: Any, package_dir: str,
                      prefix: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Select the one primary codec extension exclusively through the package binding: the only
    admissible source is ``PyNvVideoCodec._PyNvVideoCodec.__file__``: the wheel ships both
    binaries but executes one, so the primary is the package's own binding."""
    origin = getattr(getattr(module, "_PyNvVideoCodec", None), "__file__", None)
    resolved = owned_extension(origin, package_dir, prefix)
    if resolved is None:
        return None, [f"{PYNVC_PACKAGE}._PyNvVideoCodec.__file__ must name a readable non-symlink"
                      f" extension directly in {package_dir} under {prefix}; saw {origin!r}"]
    stem = extension_stem(resolved)
    linked = PRIMARY_STEMS.get(stem or "")
    if linked is None:
        return None, [f"the selected primary stem {stem!r} is not an exact public stem"
                      f" {sorted(PRIMARY_STEMS)}"]
    suffix = getattr(module, "module_suffix", None)
    if suffix != linked[0]:
        return None, [f"the selected primary stem {stem} disagrees with"
                      f" {PYNVC_PACKAGE}.module_suffix {suffix!r}"]
    record, defects = _owned_member(resolved, distribution, "the selected primary extension")
    if record is None:
        return None, defects
    return {**record, "stem": stem, "linked_nvenc_api": linked[1]}, []


def _selected_auxiliary(associated: dict[str, list[str]], distribution: Any, package_dir: str,
                        prefix: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Permit zero or one auxiliary that is exactly ``PyNvVideoCodec.VersionCheck``, the helper
    the wheel's ``__init__`` imports to pick ``module_suffix``. Zero is allowed for fixtures;
    two distinct paths are ambiguous and fail."""
    candidates = [path for path, names in associated.items() if AUXILIARY_MODULE in names]
    if not candidates:
        return None, []
    if len(candidates) != 1:
        return None, [f"{AUXILIARY_MODULE} must resolve to one loaded path; saw {candidates}"]
    resolved = candidates[0]
    if (extension_stem(resolved) != AUXILIARY_STEM
            or owned_extension(resolved, package_dir, prefix) is None):
        return None, [f"{AUXILIARY_MODULE} must be a readable {AUXILIARY_STEM} extension directly"
                      f" in {package_dir} under {prefix}; saw {resolved}"]
    record, defects = _owned_member(resolved, distribution, AUXILIARY_MODULE)
    if record is None:
        return None, defects
    return {**record, "role": AUXILIARY_ROLE, "module_names": associated[resolved]}, []


def _classify_pynvc_extensions(module: Any, distribution: Any) -> tuple[
        dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """Select one primary, name at most one auxiliary, reject every other association. This keeps
    the second-or-foreign-primary check and only narrows its association rule."""
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        return None, [], [f"the imported {PYNVC_PACKAGE} package has no file location"]
    package_dir = os.path.dirname(os.path.realpath(origin))
    prefix = os.path.realpath(sys.prefix)
    associated, symlinked = associated_extensions(package_dir)
    primary, defects = _selected_primary(module, distribution, package_dir, prefix)
    auxiliary, notes = _selected_auxiliary(associated, distribution, package_dir, prefix)
    defects.extend(notes)
    if symlinked:
        defects.append(f"a loaded {PYNVC_PACKAGE} extension came through a symlink: {symlinked}")
    expected = {item["loaded_path"] for item in (primary, auxiliary) if item is not None}
    if unexpected := sorted(set(associated) - expected):
        defects.append(f"only the selected primary and one {AUXILIARY_MODULE} auxiliary may be"
                       f" loaded; also loaded {[(p, associated[p]) for p in unexpected]}")
    if missing := sorted(expected - set(associated)):
        defects.append(f"the selected {PYNVC_PACKAGE} extension is not registered in this"
                       f" interpreter's sys.modules: {missing}")
    return primary, [auxiliary] if auxiliary is not None else [], defects


def _pynvc_surface() -> dict[str, Any]:
    """Import PyNvVideoCodec and record the module and selected extension that actually loaded.
    Rule 8: these observe the executed binary; wheel-``RECORD`` attestation is not claimed."""
    module = importlib.import_module(PYNVC_PACKAGE)
    distribution = importlib.metadata.distribution(PYNVC_PACKAGE)
    primary, auxiliaries, defects = _classify_pynvc_extensions(module, distribution)
    selected = primary or {}
    origin, version = getattr(module, "__file__", None), str(distribution.version)
    suffix, linked = getattr(module, "module_suffix", None), selected.get("linked_nvenc_api")
    surface = {
        "installed": False, "imported": origin is not None, "module": PYNVC_PACKAGE,
        "module_file": os.path.realpath(origin) if origin else None,
        "module_version": str(getattr(module, "__version__", "") or "") or None,
        "distribution_version": version, "version": version,
        "linked_nvenc_api": {"status": "observed" if linked else "unknown", "value": linked,
                             "module_suffix": suffix, "provenance": "loaded module_suffix/stem",
                             "extension_file": selected.get("loaded_path")},
        "interpreter": sys.executable, "interpreter_identity": _identity(sys.executable),
        "sys_prefix": str(Path(sys.prefix).resolve()), "auxiliary_extensions": auxiliaries,
        "extension": {"path": selected.get("path"), "sha256": selected.get("sha256"),
                      "loaded_path": selected.get("loaded_path"), "module_suffix": suffix,
                      "linked_nvenc_api": linked,
                      "distribution_entry": selected.get("distribution_entry")},
        "dependencies": _dependencies(),
    }
    sealed = _sealed(surface, surface_contract_errors("pynvc", surface) + defects)
    sealed["identity"] = _pynvc_identity(sealed, distribution)
    sealed["errors"] = [{"module": PYNVC_PACKAGE, "error": item}
                        for item in sealed.get("defects") or []]
    return sealed


def _pynvc_identity(surface: dict[str, Any], distribution: Any) -> dict[str, Any]:
    """Emit the established identity: the verified member set, else exactly {status, reason}.
    ``version``/``interpreter_identity`` are additive for the capability skill; no ``record`` is
    claimed -- wheel-``RECORD`` attestation is verify_pynvc_sample.py's authority."""
    if surface["installed"] is not True:
        return {"status": "not_ready",
                "reason": "; ".join(surface.get("defects") or []) or "not observed"}
    return {
        "status": "verified", "interpreter": surface["interpreter"],
        "sys_prefix": surface["sys_prefix"], "version": surface["distribution_version"],
        "interpreter_identity": surface["interpreter_identity"],
        "dist_info_path": next(
            (os.path.dirname(os.path.realpath(os.fspath(distribution.locate_file(item))))
             for item in getattr(distribution, "files", None) or ()
             if str(item).endswith(".dist-info/RECORD")), None),
        "distribution": {"name": PYNVC_PACKAGE, "version": surface["distribution_version"]},
        "module": {"name": PYNVC_PACKAGE, "version": surface["module_version"],
                   "path": surface["module_file"]},
        "extension": surface["extension"]}


def _sealed(surface: dict[str, Any], defects: list[str]) -> dict[str, Any]:
    """Claim ``installed`` only for a surface that satisfies its own required subset."""
    surface["installed"] = not defects
    if defects:
        surface["defects"] = defects
    return surface


def _blocked(name: str, reason: str) -> dict[str, Any]:
    return {"installed": False, "defects": [f"the {name} surface is blocked: {reason}"]}


def _guarded(name: str, producer: Any) -> dict[str, Any]:
    """Produce one surface in isolation: its failure can never reach the peer surface."""
    try:
        return producer()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _blocked(name, f"{type(exc).__name__}: {exc}")


def _python_bootstrap() -> dict[str, Any]:
    """Report whether the SYSTEM interpreter can bootstrap a venv and build C code: the plan-time
    detection of a missing python3-venv or python3-dev. It reports ``missing`` rather than
    raising, and blocks nothing."""
    record: dict[str, Any] = {"executable": None, "venv_module": "missing",
                              "development_headers": "missing"}
    try:
        executable = _tool("python3")
        record["executable"] = executable
        if executable is None:
            return record
        result = _run([executable, "-I", "-c", _PYTHON_PROBE])
        observed = read_json(result["stdout"]) if result["status"] == "ok" else None
    except Exception:  # pylint: disable=broad-exception-caught
        return record
    data = observed if isinstance(observed, dict) else {}
    record["venv_module"] = "ok" if data.get("venv") is True else "missing"
    record["development_headers"] = "ok" if data.get("headers") is True else "missing"
    return record


def _readiness(runtime: str, surfaces: dict[str, Any]) -> dict[str, Any]:
    """Publish each requested surface as its own readiness layer, never one merged claim. A
    satisfied surface is ``partial``, not ``ready``: the probe proves no codec operation."""
    layers: dict[str, Any] = {}
    reasons: list[str] = []
    for name, layer in (("native", "native_sdk"), ("pynvc", "pynvc")):
        surface = surfaces.get(name)
        if not isinstance(surface, dict):
            continue
        installed = surface.get("installed") is True
        layers[layer] = {"status": "partial" if installed else "not_ready",
                         "installation": "ready" if installed else "not_ready",
                         "operation": "not_verified_by_probe"}
        reasons.extend(surface.get("defects") or [
            f"the {name} surface satisfies its required subset; operational proof still"
            " requires its own verification run"])
    states = [item["status"] for item in layers.values()]
    if runtime != "both":
        state = states[0] if states else "not_ready"
    elif states and len(set(states)) == 1:
        state = states[0]
    else:
        state = "partial"
    return {"state": state, "layers": layers, "reasons": reasons}


def _requested(surfaces: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one normalized surface, or an explicitly not-requested one in its exact shape."""
    surface = surfaces.get(name)
    if not isinstance(surface, dict):
        surface = _blocked(name, "it was not requested by this probe")
    if name != "pynvc" or "identity" in surface:
        return surface
    reason = "; ".join(surface.get("defects") or []) or "no PyNvVideoCodec import was attempted"
    return {**surface, "imported": False, "module": None, "module_file": None,
            "module_version": None, "distribution_version": None,
            "errors": [{"module": PYNVC_PACKAGE, "error": reason}],
            "linked_nvenc_api": {"status": "not_ready", "value": None, "reason": reason},
            "identity": {"status": "not_ready", "reason": reason}}


def _capabilities(gpu: int, runtime: str,
                  pynvc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Publish the frozen native-unavailable/PyNv API-query authority split."""
    if runtime == "native":
        return ({"authority": "native-adapter-unavailable", "gpu": gpu, "status": "unknown",
                 "reason": "No native capability-query adapter is bundled with this skill.",
                 "encode": {}, "decode": {}}, [])
    module = sys.modules.get(PYNVC_PACKAGE) if pynvc.get("installed") is True else None
    if module is None:
        result = {"authority": "none", "gpu": gpu, "status": "not_ready",
                  "reason": "PyNvVideoCodec was not ready; no capability query ran.",
                  "encode": {}, "decode": {}}
        return result, []
    result, warnings = query_pynvc_capabilities(
        module, gpu, _member(pynvc, "linked_nvenc_api"))
    if runtime == "both":
        result["native"] = {"authority": "native-adapter-unavailable", "status": "unknown",
                            "reason": "Native inventory is reported independently."}
        result.update({"status": "partial", "reason":
                       "PyNvVideoCodec was queried; native query authority is absent."})
    return result, warnings


def _installation(cuda: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Publish the 1.2 ``installation`` inventory from this run's observations.
    ``python.executable`` is the interpreter that produced this artifact after any validated-
    venv delegation, because that is what ``@python-from`` re-enters; never a realpath. The
    venv/header state beside it belongs to ``system_executable``."""
    bootstrap, running = _python_bootstrap(), _identity(sys.executable)
    tools = _member(native, "tools")
    return {
        "status": "observed", "mutation_performed": False, "network_probed": False,
        "python": {"executable": sys.executable, "identity": running,
                   "interpreter_identities": {"running": running},
                   "sys_prefix": str(Path(sys.prefix).resolve()), "packages": _dependencies(),
                   "version": ".".join(str(part) for part in sys.version_info[:3]),
                   "abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                   "system_executable": bootstrap["executable"],
                   "venv_module": bootstrap["venv_module"],
                   "development_headers": bootstrap["development_headers"]},
        "apt": _apt_probe(), "cuda_toolkit": cuda, "native_sdk": native,
        "native_build_prerequisites": native.get("build_prerequisites"),
        "build_tools": {name: record.get("path") for name, record in tools.items()},
        "build_tool_identities": {(str(record.get("executable")) if name == "generator"
                                   else _TOOL_ALIASES.get(name, name)): record
                                  for name, record in tools.items()},
    }


def _environment(gpu: int, runtime: str, cuda: dict[str, Any],
                 surfaces: dict[str, Any]) -> dict[str, Any]:
    """Assemble exactly the frozen closed 16-key 1.2 document and nothing else. ``libraries`` is
    empty because this probe runs no platform-loader query; the internal normalized mapping ends
    here, republished as ``pynvc`` and ``installation.native_sdk``."""
    pynvc = _requested(surfaces, "pynvc")
    linked = _member(pynvc, "linked_nvenc_api").get("value")
    capabilities, capability_warnings = _capabilities(gpu, runtime, pynvc)
    return {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "generated_at": utc_now(),
        "mode": "live", "requested_runtime": runtime, "platform": _platform_record(),
        "selected_gpu": gpu, "nvidia_smi": _nvidia_smi(gpu), "libraries": {},
        "driver_nvenc_api": {"status": "observed" if linked else "unknown", "max_supported": None,
                             "linked_nvenc_api": linked,
                             "provenance": "selected PyNvVideoCodec primary module_suffix",
                             "reason": "the probe never calls NvEncodeAPIGetMaxSupportedVersion;"
                                       " the ceiling is proven by the runtime helper"},
        "pynvc": pynvc, "installation": _installation(cuda, _requested(surfaces, "native")),
        "capabilities": capabilities,
        "warnings": [_PROBE_WARNING, *capability_warnings],
        "readiness": _readiness(runtime, surfaces),
        "command_evidence": list(_COMMAND_EVIDENCE),
    }


def _registry_selection(requested: tuple[str, ...],
                        setup_candidate: bool) -> dict[str, Any] | None:
    """Bind the fixed validated PyNv venv before any import, delegating once when needed. Returns
    None to probe under the current interpreter, or a blocked ``pynvc`` surface when the registry
    is absent or stale -- never a scan. An unregistered interpreter re-executes through the
    registered lexical one; that child does not re-delegate."""
    if "pynvc" not in requested or setup_candidate:
        return None
    try:
        registry = load_pynvc_registry()
    except RegistryNotReady as exc:
        return _blocked("pynvc", f"the validated PyNvVideoCodec venv registry is not ready"
                                 f" ({exc.reason}); next action: {exc.next_action}")
    registered = str(registry["interpreter"])
    if os.path.abspath(os.path.expanduser(sys.executable)) == os.path.abspath(registered):
        return None
    try:
        completed = run_command([registered, "-I", os.path.abspath(__file__), *sys.argv[1:]],
                                timeout=_DELEGATE_TIMEOUT)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return _blocked("pynvc", f"the registered interpreter could not be launched: {exc}")
    stdout = completed.stdout.decode("utf-8", "replace")
    sys.stderr.write(completed.stderr.decode("utf-8", "replace"))
    # Never forward a SUCCESS this probe cannot recognize. A child that answered with some
    # other document kind or major has not produced the artifact the caller asked for, and
    # relaying it at rc 0 would launder an unknown envelope into a trusted one.
    if completed.returncode == 0 and not _own_envelope(stdout):
        print(_render({"schema_version": SCHEMA_VERSION, "kind": ERROR_KIND, "status": "error",
                       "error": "the registered interpreter returned an unrecognized document"
                                f" (expected {KIND} {SCHEMA_VERSION}); refusing to forward it"}))
        raise SystemExit(3)
    sys.stdout.write(stdout)
    raise SystemExit(completed.returncode)


def _own_envelope(stdout: str) -> bool:
    """Recognize only this probe's own artifact envelope: exact kind AND exact major."""
    try:
        value = json.loads(stdout)
    except ValueError:
        return False
    return (isinstance(value, dict) and value.get("kind") == KIND
            and value.get("schema_version") == SCHEMA_VERSION)


def _render(value: Any) -> str:
    """Serialize strictly for stdout; the artifact itself is written by write_new_json."""
    return json.dumps(value, allow_nan=False, indent=1, sort_keys=True)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject a repeated key instead of letting json.loads silently keep the last one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate key in environment artifact: {key!r}")
        result[key] = value
    return result


def _reject_constant(constant: str) -> Any:
    raise ValueError(f"non-RFC JSON numeric constant is forbidden: {constant}")


def _read_artifact(path: Any) -> Any:
    """Read one artifact as size-bounded, symlink-free, unambiguous strict JSON. Oversize input
    is malformed; a repeated key remains a strict validation defect instead of being silently
    replaced by json.loads."""
    resolved = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"environment artifact exceeds the {_MAX_ARTIFACT_BYTES}-byte bound")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        raw = handle.read(_MAX_ARTIFACT_BYTES + 1)
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"environment artifact exceeds the {_MAX_ARTIFACT_BYTES}-byte bound")
    return json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)


def _reauthenticate_main(path: Path) -> int:
    """Re-validate the 1.2 shape and the identities on disk now: rc 0 valid, rc 2 invalid. A
    malformed document raises and the caller answers rc 3 on the validation-error envelope."""
    try:
        data = _read_artifact(path)
    except _DuplicateKeyError as exc:
        data, defects = {}, [str(exc)]
    else:
        if not isinstance(data, dict):
            raise ValueError("environment artifact must be a JSON object")
        defects = environment_contract_errors(data, live=True)
    print(_render({"schema_version": VALIDATION_VERSION, "kind": VALIDATION_KIND,
                   "valid": not defects, "readiness": _member(data, "readiness").get("state"),
                   "errors": defects}))
    return 0 if not defects else 2


def _mock_main(path: Path, gpu: int, runtime: str, output: Path | None) -> int:
    """Emit an admitted offline fixture as explicit mock evidence."""
    environment = _read_artifact(path)
    if not isinstance(environment, dict):
        raise ValueError("mock fixture must be a JSON object")
    environment.setdefault("schema_version", SCHEMA_VERSION)
    environment.setdefault("kind", KIND)
    environment.setdefault("generated_at", utc_now())
    environment["mode"], environment["requested_runtime"] = "mock", runtime
    environment.setdefault("selected_gpu", gpu)
    environment.setdefault("warnings", []).append(
        "Mock fixture: no live hardware or package query was performed.")
    if output is not None:
        write_new_json(output, environment)
    print(_render(environment))
    return 0


def _probe_main(gpu: int, runtime: str, output: Path, setup_candidate: bool) -> int:
    """Emit one fresh artifact; an existing --output is refused, never overwritten."""
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite an existing probe artifact: {output}")
    requested = SURFACES if runtime == "both" else (runtime,)
    selection = _registry_selection(requested, setup_candidate)
    cuda = _cuda_probe()
    surfaces: dict[str, Any] = {}
    if "native" in requested:
        surfaces["native"] = _guarded("native", lambda: _native_surface(cuda))
    if "pynvc" in requested:
        surfaces["pynvc"] = selection or _guarded("pynvc", _pynvc_surface)
    environment = _environment(gpu, runtime, cuda, surfaces)
    write_new_json(output, environment)
    print(_render(environment))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--runtime", choices=("pynvc", "native", "both"),
                        default="pynvc", help="surface selection (default: pynvc)")
    parser.add_argument("--mock-fixture", type=Path,
                        help="offline fixture; never treat it as live evidence")
    parser.add_argument("--output", type=Path,
                        help="write the nvcodec-environment 1.2 artifact here; never overwritten")
    parser.add_argument("--setup-candidate", action="store_true",
                        help=("authorized bootstrap/final-setup probe: run under the current"
                              " interpreter without validated-venv registry redirection"))
    parser.add_argument("--reauthenticate", type=Path, metavar="ENVIRONMENT",
                        help=("re-validate an existing nvcodec-environment artifact against the"
                              " identities present on disk now instead of probing"))
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.reauthenticate is not None and args.mock_fixture is not None:
        parser.error("--reauthenticate and --mock-fixture are mutually exclusive")
    if args.reauthenticate is None and args.mock_fixture is None and args.output is None:
        parser.error("a live probe requires --output")
    validating = args.reauthenticate is not None
    try:
        require_isolated()
        if validating:
            return _reauthenticate_main(args.reauthenticate)
        if args.mock_fixture is not None:
            return _mock_main(args.mock_fixture, args.gpu, args.runtime, args.output)
        return _probe_main(args.gpu, args.runtime, args.output, args.setup_candidate)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(_render({
            "schema_version": VALIDATION_VERSION if validating else SCHEMA_VERSION,
            "kind": VALIDATION_ERROR_KIND if validating else ERROR_KIND,
            "status": "error", "error": str(exc)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
