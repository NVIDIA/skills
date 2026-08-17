#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal shared core for the Jetson video setup helpers.

This private module owns isolation, strict artifacts, bounded subprocesses, public-APT binding,
and the small environment/surface schema shared by probe and planner. Installation policy, build
operations, wheel attestation, and the validated-PyNv-venv registry stay with the public helper
that owns them. The compact PyNvVideoCodec API-query adapter is shared only with the environment
probe that publishes the frozen schema-1.2 capability block.
"""

# pylint: disable=missing-function-docstring,too-many-boolean-expressions,too-many-locals
# pylint: disable=too-many-lines

from __future__ import annotations

import errno
import hashlib
import importlib.machinery
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ambient variables preserved for subprocesses; captured URL credentials are redacted.
_ENV_ALLOWLIST = (
    "HOME", "TMPDIR", "TMP", "TEMP", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
)
_ENV_OVERRIDES = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C", "LANG": "C",
                  "LANGUAGE": "C", "PYTHONNOUSERSITE": "1"}
# Import-injection variables; run_command sets PYTHONNOUSERSITE itself.
_INJECTION_VARIABLES = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "PYTHONUSERBASE",
    "PYTHONWARNINGS", "PYTHONOPTIMIZE", "LD_PRELOAD", "LD_AUDIT",
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
)
# Stock signed public Jetson origins only; internal mirrors remain ineligible.
_PUBLIC_APT_LOCATIONS = frozenset(
    ("https", "repo.download.nvidia.com", f"/jetson/{part}") for part in ("common", "som"))
_PUBLIC_SUITE_COMPONENT = re.compile(r"r\d+\.\d+/main")
_APT_BYPASS = ("trusted", "allow-insecure", "allow-weak", "allow-downgrade-to-insecure")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION_TEXT = re.compile(r"\d")
_URL_USERINFO = re.compile(rb"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")
_ENVIRONMENT_KIND = "nvcodec-environment"
# Pinned to probe_nvcodec.SCHEMA_VERSION to avoid a circular import. The 1.2 envelope is closed;
# unknown top-level keys fail validation.
_PYNVC_ENVIRONMENT_SCHEMA_VERSION = "1.2"
_ENVIRONMENT_KEYS = frozenset((
    "schema_version", "kind", "generated_at", "mode", "requested_runtime", "platform",
    "selected_gpu", "nvidia_smi", "libraries", "driver_nvenc_api", "pynvc", "installation",
    "capabilities", "warnings", "readiness", "command_evidence"))
_ENVIRONMENT_REQUIRED = ("schema_version", "kind", "platform", "selected_gpu",
                         "requested_runtime", "pynvc", "installation", "capabilities",
                         "readiness")
_ENVIRONMENT_MODES = ("live", "mock")
_READINESS_KEYS = frozenset(("state", "layers", "reasons"))
_READINESS_STATES = ("ready", "partial", "not_ready")
_RUNTIMES = ("pynvc", "native", "both")
_JETSON_MACHINES = re.compile(r"aarch64|arm64")
_SURFACES = ("native", "pynvc")
_NATIVE_PACKAGE = "nvidia-video-codec-sdk"
_NATIVE_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
_GENERATORS = ("Ninja", "Unix Makefiles")
_PY_DEPENDENCIES = ("numpy", "pycuda", "torch")
# --- Official-sample verification profiles (public) --------------------------
# Smoke uses torch-free basic encode/decode_perf; full advanced decode imports Torch.
SMOKE_PROFILE, FULL_PROFILE = "pynvc-smoke", "full-samples"
VERIFICATION_PROFILES = (SMOKE_PROFILE, FULL_PROFILE)
PROFILE_DEPENDENCIES = {SMOKE_PROFILE: ("numpy", "pycuda"), FULL_PROFILE: _PY_DEPENDENCIES}
# --- Shared extension-classification contract (public) -----------------------
# One predicate binds loaded compiled modules to the selected distribution; never scan sys.path.
PYNVC_PACKAGE = "PyNvVideoCodec"
# Public 2.1.0 primary stems are bound to module_suffix and linked NVENC API.
PRIMARY_STEMS = {"PyNvVideoCodec_121": ("_121", "12.1"), "PyNvVideoCodec_130": ("_130", "13.0")}
# The one legitimate compiled helper the wheel loads beside the primary: __init__.py imports
# .VersionCheck.DriverWrapper to pick module_suffix. Auxiliary evidence, never a primary.
AUXILIARY_MODULE, AUXILIARY_STEM = f"{PYNVC_PACKAGE}.VersionCheck", "VersionCheck"
AUXILIARY_ROLE = "driver_version_check"
EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
RECOGNIZED_STEMS = frozenset({*PRIMARY_STEMS, AUXILIARY_STEM})
_LONGEST_SUFFIXES = tuple(sorted(EXTENSION_SUFFIXES, key=len, reverse=True))
# R3: the Jetson Linux minimum has one owner; the producer decides it and consumers read it.
MINIMUM_JETSON_LINUX, MINIMUM_VERSION_TEXT = (38, 5), "38.5"
# The exact members an installed native SDK root must carry to count as complete.
NATIVE_REQUIRED_FILES = (
    "Interface/nvEncodeAPI.h", "Interface/nvcuvid.h", "Interface/cuviddec.h",
    "Samples/CMakeLists.txt", "Samples/AppEncode/AppEncCuda/CMakeLists.txt",
    "Samples/AppEncode/AppEncCuda/AppEncCuda.cpp",
    "Samples/AppDecode/AppDec/CMakeLists.txt", "Samples/AppDecode/AppDec/AppDec.cpp")


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_isolated() -> None:
    """Fail closed unless this interpreter runs isolated with no import-injection variables."""
    present = [name for name in _INJECTION_VARIABLES if os.environ.get(name)]
    _check(not present, "import injection variables must be absent: " + ", ".join(present))
    _check(sys.flags.ignore_environment and sys.flags.no_user_site,
           "setup helpers must run isolated with environment/user-site disabled (python -I)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Hashing, strict JSON, identity, and exclusive writers -------------------

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dumps(value: Any, **kwargs: Any) -> str:
    """Serialize strictly: allow_nan=False rejects non-finite floats at the source."""
    return json.dumps(value, allow_nan=False, **kwargs)


def canonical_json_sha256(value: Any) -> str:
    """Return sha256(strict JSON, sort_keys=True, separators=(",", ":")) -- the frozen form
    plan_install's plan digest and every re-plan comparison depend on."""
    return sha256_bytes(_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _symlink_free(path: Any, *, label: str) -> Path:
    """Return the absolute path, rejecting a symlink leaf or any symlinked parent component."""
    requested = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    current = Path(requested.anchor)
    for part in requested.parent.parts[1:]:
        current /= part
        _check(not current.is_symlink(), f"{label} parent contains a symlink component: {current}")
    _check(not requested.is_symlink(), f"{label} must not be a symlink: {requested}")
    return requested


def _read_regular(path: Any, *, label: str) -> tuple[bytes, int, Path]:
    """Read one regular file whole, following no symlink at any component of its path."""
    requested = _symlink_free(path, label=label)
    descriptor = os.open(requested, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        details = os.fstat(handle.fileno())
        _check(stat.S_ISREG(details.st_mode), f"{label} must be a regular file: {requested}")
        raw = handle.read()
    return raw, details.st_size, requested.parent.resolve(strict=True) / requested.name


def file_identity(path: Any, *, label: str = "file") -> dict[str, Any]:
    """Return the one canonical identity of a regular file: resolved path, size, and sha256."""
    raw, size_bytes, resolved = _read_regular(path, label=label)
    return {"path": str(resolved), "size_bytes": size_bytes, "sha256": sha256_bytes(raw)}


def _absolute(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def _version(value: Any) -> bool:
    return isinstance(value, str) and _VERSION_TEXT.match(value) is not None


def _member(value: Any, name: str) -> dict[str, Any]:
    member = value.get(name) if isinstance(value, dict) else None
    return member if isinstance(member, dict) else {}


def _identity_errors(record: Any, label: str, live: bool) -> list[str]:
    if not isinstance(record, dict) or not _absolute(record.get("path")):
        return [f"{label} must record an absolute path and SHA-256"]
    if not isinstance(record.get("sha256"), str) or not _SHA256.fullmatch(record["sha256"]):
        return [f"{label} must record an absolute path and SHA-256"]
    if not live:
        return []
    try:
        observed = file_identity(record["path"], label=label)
    except (OSError, ValueError) as exc:
        return [f"{label} is no longer readable: {exc}"]
    return [] if (
        observed["path"] == record["path"] and observed["sha256"] == record["sha256"]
    ) else [f"{label} changed after the environment probe"]


def extension_stem(path: str) -> str | None:
    """Strip exactly one running-interpreter extension suffix, longest first, from a basename.
    Never ``Path.stem`` or a bare ``.so`` rule: only EXTENSION_SUFFIXES decides the real stem."""
    name = os.path.basename(path)
    for suffix in _LONGEST_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return None


def owned_extension(path: Any, package_dir: str, prefix: str) -> str | None:
    """Resolve one loaded extension that is a readable direct child of the package under prefix."""
    if not isinstance(path, str) or not path or os.path.islink(path):
        return None
    resolved = os.path.realpath(path)
    if not os.path.isfile(resolved) or not os.access(resolved, os.R_OK):
        return None
    if os.path.dirname(resolved) != package_dir:
        return None
    return resolved if resolved.startswith(prefix + os.sep) else None


def associated_extensions(package_dir: str) -> tuple[dict[str, list[str]], list[str]]:
    """Map every loaded compiled module associated with the package to its names. Association: a
    registration below ``PyNvVideoCodec``, a recognized primary/VersionCheck stem, or a resolved
    parent that is the package directory. Only this interpreter's ``sys.modules`` is read."""
    associated: dict[str, list[str]] = {}
    symlinked: set[str] = set()
    for name, value in list(sys.modules.items()):
        origin = getattr(value, "__file__", None)
        if not isinstance(origin, str) or not origin.endswith(EXTENSION_SUFFIXES):
            continue
        resolved = os.path.realpath(origin)
        if (name == PYNVC_PACKAGE or name.startswith(f"{PYNVC_PACKAGE}.")
                or extension_stem(resolved) in RECOGNIZED_STEMS
                or os.path.dirname(resolved) == package_dir):
            associated.setdefault(resolved, []).append(name)
            if os.path.islink(origin):
                symlinked.add(origin)
    return ({path: sorted(names) for path, names in sorted(associated.items())},
            sorted(symlinked))


def root_evidence(root: str) -> dict[str, Any]:
    """Enumerate the required SDK members under one root, rejecting symlinked or escaped files."""
    base, missing, rejected = root.rstrip("/"), [], []
    for relative in NATIVE_REQUIRED_FILES:
        candidate, target = Path(root) / relative, f"{base}/{relative}"
        if not candidate.is_file():
            missing.append(target)
        elif candidate.is_symlink() or os.path.realpath(candidate) != target:
            rejected.append(target)
    return {"root": base, "status": "complete" if not missing and not rejected else "incomplete",
            "required_files": [f"{base}/{name}" for name in NATIVE_REQUIRED_FILES],
            "missing_required_files": missing, "rejected_required_files": rejected}


def surface_contract_errors(name: str, surface: Any, *, live: bool = False) -> list[str]:
    """Validate one installed surface without reading or blocking its independent peer."""
    if name not in _SURFACES or not isinstance(surface, dict):
        return [f"surfaces.{name} must be an object"]
    if name == "native":
        package, cuda = _member(surface, "package"), _member(surface, "cuda")
        tools, build = _member(surface, "tools"), _member(surface, "build_prerequisites")
        errors = [] if (
            package.get("name") == _NATIVE_PACKAGE and package.get("status") == "installed"
            and _version(package.get("version"))
        ) else [f"{_NATIVE_PACKAGE} is not installed with a package version"]
        if not _absolute(surface.get("sdk_root")):
            errors.append("no canonical native SDK root was observed")
        if cuda.get("status") != "installed" or not _version(cuda.get("version")):
            errors.append("no installed CUDA toolkit version was observed")
        for tool in _NATIVE_TOOLS:
            record = tools.get(tool)
            errors.extend(_identity_errors(record, f"surfaces.native.tools.{tool}", live))
            if (not isinstance(record, dict) or not _version(record.get("version"))
                    or (tool == "generator" and record.get("name") not in _GENERATORS)):
                errors.append(f"surfaces.native.tools.{tool} is incomplete")
        if build.get("status") != "complete" or build.get("unresolved_modules") != []:
            errors.append("AppDec pkg-config prerequisites are incomplete")
        return errors
    # Established 1.2 spellings: module_file / module_version / distribution_version. The
    # interpreter binding is read from the top level, where the probe also publishes it beside
    # the established pynvc.identity that carries the same two values.
    extension = _member(surface, "extension")
    interpreter = _member(surface, "interpreter_identity")
    errors = [] if _version(surface.get("distribution_version")) else [
        "no PyNvVideoCodec distribution version was observed"]
    if not _absolute(surface.get("interpreter")) or not _absolute(surface.get("sys_prefix")):
        errors.append("the lexical interpreter or sys_prefix is missing")
    errors.extend(_identity_errors(interpreter, "pynvc.interpreter_identity", live))
    errors.extend(_identity_errors(extension, "pynvc.extension", live))
    if extension.get("loaded_path") != extension.get("path"):
        errors.append("the loaded extension is not the extension hashed on disk")
    if (not _absolute(surface.get("module_file"))
            or surface.get("module_version") != surface.get("distribution_version")):
        errors.append("the imported module does not match the distribution")
    dependencies = surface.get("dependencies")
    if not isinstance(dependencies, dict):
        errors.append("surfaces.pynvc.dependencies must be an object")
    else:
        errors.extend(
            f"surfaces.pynvc.dependencies.{name} must carry a boolean ready"
            for name in _PY_DEPENDENCIES
            if not isinstance(dependencies.get(name), dict)
            or not isinstance(dependencies[name].get("ready"), bool))
    return errors + _auxiliary_errors(surface, live)


def _platform_errors(platform: Any) -> list[str]:
    """Validate only the platform subset the evaluator and the corpora actually consume."""
    if not isinstance(platform, dict):
        return ["platform must be an object"]
    machine = platform.get("machine")
    errors = [] if platform.get("jetson") is True else [
        "platform.jetson must be true: this contract describes a Jetson target"]
    if not isinstance(machine, str) or _JETSON_MACHINES.fullmatch(machine) is None:
        errors.append(f"platform.machine must be aarch64 or arm64, saw {machine!r}")
    compatibility = _member(_member(platform, "jetson_linux"), "compatibility")
    if not isinstance(compatibility.get("status"), str) or not compatibility["status"]:
        errors.append("platform.jetson_linux.compatibility.status is required")
    return errors


def _readiness_errors(readiness: Any) -> list[str]:
    """Require a state and at least one per-surface layer; the layers are reported, not merged."""
    if not isinstance(readiness, dict):
        return ["readiness must be an object"]
    errors = [f"unknown readiness field: {key}" for key in sorted(set(readiness) - _READINESS_KEYS)]
    if readiness.get("state") not in _READINESS_STATES:
        errors.append(f"readiness.state must be one of {list(_READINESS_STATES)}")
    layers = readiness.get("layers")
    if not isinstance(layers, dict) or not layers:
        errors.append("readiness.layers must record at least one surface layer")
    elif any(not isinstance(item, dict) or not isinstance(item.get("status"), str)
             for item in layers.values()):
        errors.append("every readiness layer must record its own status")
    if not isinstance(readiness.get("reasons", []), list):
        errors.append("readiness.reasons must be an array")
    return errors


def _installation_errors(installation: Any, live: bool) -> list[str]:
    """Require the interpreter binding that ``@python-from`` and every helper re-enter.

    The executable must be absolute in every mode; its existence and executability are checked
    only for a live re-authentication, where the recorded artifact is being replayed on disk.
    """
    if not isinstance(installation, dict):
        return ["installation must be an object"]
    python = _member(installation, "python")
    executable = python.get("executable")
    if not _absolute(executable):
        return ["installation.python.executable must be an absolute interpreter path"]
    errors = [] if not live or (
        os.path.isfile(executable) and os.access(executable, os.X_OK)
    ) else [f"installation.python.executable is not an executable file now: {executable}"]
    if live:
        errors.extend(_identity_errors(python.get("identity"), "installation.python.identity",
                                       True))
    return errors


def _auxiliary_errors(surface: Any, live: bool) -> list[str]:
    """Permit zero or one recorded auxiliary extension and re-authenticate the one present."""
    auxiliaries = surface.get("auxiliary_extensions", []) if isinstance(surface, dict) else None
    if not isinstance(auxiliaries, list) or len(auxiliaries) > 1:
        return ["pynvc.auxiliary_extensions must be an array of at most one auxiliary"]
    errors: list[str] = []
    for item in auxiliaries:
        if not isinstance(item, dict) or item.get("role") != AUXILIARY_ROLE:
            errors.append(f"a pynvc auxiliary must record role {AUXILIARY_ROLE!r}")
            continue
        errors.extend(_identity_errors(item, "pynvc.auxiliary_extensions", live))
    return errors


def environment_contract_errors(data: Any, *, live: bool = False,
                                include_surface_values: bool = True) -> list[str]:
    """Validate the closed 16-key ``nvcodec-environment`` 1.2 envelope by required subset.

    Closed only at the top level: an unknown top-level key is an error, but no nested structure
    is compared for exact-set equality, so additive evidence beneath a member never invalidates
    an otherwise conforming artifact. The two independent surfaces -- top-level ``pynvc`` and
    ``installation.native_sdk`` -- are each validated against their own required subset only
    when they claim to be installed, so one can never block its peer.
    """
    if not isinstance(data, dict):
        return ["environment must be a JSON object"]
    errors = [f"unknown top-level environment field: {key}"
              for key in sorted(set(data) - _ENVIRONMENT_KEYS)]
    errors.extend(f"missing required field: {key}"
                  for key in _ENVIRONMENT_REQUIRED if key not in data)
    errors.extend(f"{key} must be exactly {expected!r}" for key, expected in
                  (("kind", _ENVIRONMENT_KIND),
                   ("schema_version", _PYNVC_ENVIRONMENT_SCHEMA_VERSION))
                  if data.get(key) != expected)
    if data.get("mode") not in _ENVIRONMENT_MODES:
        errors.append(f"mode must be one of {list(_ENVIRONMENT_MODES)}")
    if data.get("requested_runtime") not in _RUNTIMES:
        errors.append(f"requested_runtime must be the string {list(_RUNTIMES)}, never a list")
    gpu = data.get("selected_gpu")
    if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
        errors.append("selected_gpu must be a non-negative integer")
    errors.extend(_platform_errors(data.get("platform")))
    errors.extend(_readiness_errors(data.get("readiness")))
    errors.extend(_installation_errors(data.get("installation"), live))
    errors.extend(f"{key} must be an object" for key in ("pynvc", "capabilities", "nvidia_smi",
                                                         "libraries", "driver_nvenc_api")
                  if key in data and not isinstance(data[key], dict))
    errors.extend(f"{key} must be an array" for key in ("warnings", "command_evidence")
                  if key in data and not isinstance(data[key], list))
    if not include_surface_values:
        return errors
    for name, surface in (("pynvc", data.get("pynvc")),
                          ("native", _member(data.get("installation"), "native_sdk"))):
        if isinstance(surface, dict) and surface.get("installed") is True:
            errors.extend(f"{name}: {item}"
                          for item in surface_contract_errors(name, surface, live=live))
    return errors


def require_selected_surfaces_live(data: Any, components: list[str]) -> None:
    """Reauthenticate only installed surfaces selected by one install plan."""
    records = {
        "native-sdk": ("native", _member(_member(data, "installation"), "native_sdk")),
        "pynvc": ("pynvc", _member(data, "pynvc")),
    }
    errors: list[str] = []
    for component in components:
        _check(component in records, f"unknown setup component: {component!r}")
        name, surface = records[component]
        planner_ready = (surface.get("status") == "installed" if name == "native" else
                         surface.get("imported") is True
                         and _member(surface, "identity").get("status") == "verified")
        if surface.get("installed") is True or planner_ready:
            errors.extend(f"{name}: {item}"
                          for item in surface_contract_errors(name, surface, live=True))
    _check(not errors, "selected surface failed live authentication: " + "; ".join(errors))


# --- 1.2 environment -> internal normalized surface --------------------------
# Required-subset normalization; additive keys are ignored and no 2.0 spelling is required.
_CUDA_PRESENT = {"installed", "available"}
_PACKAGE_EXTRA = {"numpy": (), "pycuda": (), "torch": (("cuda_build", "13.0"),
                  ("cuda_available", True), ("sample_readiness", "ready"))}


def profile_dependencies(profile: Any) -> tuple[str, ...]:
    """Return the Python dependencies one verification profile genuinely requires.

    An absent or unrecognized profile resolves to ``full-samples``. Evidence recorded before the
    profile split carries no profile field, and a legacy artifact must never silently relax the
    Torch requirement its own proof depended on.
    """
    return PROFILE_DEPENDENCIES.get(profile, PROFILE_DEPENDENCIES[FULL_PROFILE])


def stable_debian_version(value: Any) -> str:
    """Strip an optional epoch and reject prerelease or Debian ``really`` versions."""
    text = str(value or "").strip().split(":", 1)[-1]
    return "" if "~" in text or "really" in text.lower() else text


def cuda_release(value: Any) -> tuple[int, int] | None:
    """Return a stable CUDA major/minor pair from one Debian candidate."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", stable_debian_version(value))
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_at_least_13(value: Any) -> bool:
    """Whether a candidate is a stable CUDA 13.0-or-newer release."""
    return (release := cuda_release(value)) is not None and release >= (13, 0)


def native_sdk_release(value: Any) -> bool:
    """Whether a Debian candidate belongs to the supported native SDK 13.0 family."""
    return re.fullmatch(r"13\.0(?:\.\d+)*(?:\+[0-9A-Za-z.]+)?(?:-[0-9A-Za-z.+]+)?",
                        stable_debian_version(value)) is not None


def cuda_build_package(candidate: Any) -> str | None:
    """Map a stable CUDA 13+ candidate to its matching minimal-build meta-package."""
    release = cuda_release(candidate)
    if release is None or release < (13, 0):
        return None
    return f"cuda-minimal-build-{release[0]}-{release[1]}"


def pycuda_bootstrap_packages(candidate: Any) -> tuple[str, ...]:
    """Return the minimal exact package family needed by PyCUDA's default CURAND build."""
    build = cuda_build_package(candidate)
    if build is None:
        return ()
    suffix = build.removeprefix("cuda-minimal-build-")
    return (f"libcurand-dev-{suffix}", build)


def _nested(value: Any, path: tuple[str, ...]) -> Any:
    """Walk one nested path, yielding None -- never {} -- at the first non-object component."""
    for name in path:
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def normalized_cuda_evidence(environment: Any) -> dict[str, Any]:
    """Map installation.cuda_toolkit onto the installed/absent vocabulary consumers act on."""
    cuda = _nested(environment, ("installation", "cuda_toolkit"))
    cuda = cuda if isinstance(cuda, dict) else {}
    if cuda.get("status") not in _CUDA_PRESENT:
        return {"status": "absent", "version": None, "root": None}
    return {"status": "installed", "version": cuda.get("version"),
            "root": cuda.get("root") or _nested(cuda, ("nvcc_discovery", "root")),
            "environment_prefixes": cuda.get("environment_prefixes")}


def durable_venv_path(value: Path) -> Path | None:
    """Resolve an absolute durable venv target outside cwd and transient roots."""
    requested = value.expanduser()
    if not requested.is_absolute():
        return None
    resolved, cwd = requested.resolve(strict=False), Path.cwd().resolve()
    home = Path.home().resolve()
    # These are rejection roots, never locations used to create files.
    transient = any(resolved == root or root in resolved.parents
                    for root in map(Path, ("/tmp", "/var/tmp", "/run")))  # nosec B108
    under_worktree = resolved == cwd or (cwd not in (home, Path("/")) and cwd in resolved.parents)
    return None if under_worktree or transient else resolved


def normalized_native_surface(environment: Any) -> tuple[dict[str, Any], list[str]]:
    """Normalize the established installation.* native evidence into the internal surface."""
    native = _nested(environment, ("installation", "native_sdk"))
    native = native if isinstance(native, dict) else {}
    roots = [item for item in (native.get("complete_roots") or native.get("roots") or [])
             if _absolute(item)]
    build = _nested(environment, ("installation", "native_build_prerequisites"))
    build = build if isinstance(build, dict) else {}
    pkg_config = _nested(environment, ("installation", "build_tools", "pkg-config"))
    producer_ready = native.get("installed")
    installed = (native.get("status") == "installed" and producer_ready is True
                 and build.get("status") == "complete")
    surface = {"installed": installed, "package": native.get("package"),
               "sdk_root": roots[0] if roots else None,
               "cuda": normalized_cuda_evidence(environment),
               "build_prerequisites": build,
               "tools": {"pkg_config": {"path": pkg_config}} if pkg_config else {}}
    defects = [] if isinstance(native.get("status"), str) else [
        "installation.native_sdk.status is required"]
    if surface["installed"] and not (roots and _nested(surface, ("package", "version"))):
        defects.append("an installed native SDK requires a complete root and a package version")
    return surface, defects


def normalized_pynvc_surface(environment: Any) -> tuple[dict[str, Any], list[str]]:
    """Normalize the established pynvc + installation.python.packages evidence."""
    pynvc = environment.get("pynvc") if isinstance(environment, dict) else None
    pynvc = pynvc if isinstance(pynvc, dict) else {}
    identity = _nested(pynvc, ("identity",))
    identity = identity if isinstance(identity, dict) else {}
    packages = _nested(environment, ("installation", "python", "packages"))
    dependencies: dict[str, Any] = {}
    for name, extra in _PACKAGE_EXTRA.items():
        row = _nested(packages, (name,))
        row = row if isinstance(row, dict) else {}
        dependencies[name] = {"version": row.get("version"), "ready": bool(
            row.get("status") == "installed" and row.get("requirement_satisfied") is True
            and all(row.get(key) == value for key, value in extra))}
    surface = {"installed": pynvc.get("imported") is True and identity.get("status") == "verified",
               "version": pynvc.get("distribution_version"), "dependencies": dependencies,
               "interpreter": identity.get("interpreter"), "sys_prefix": identity.get("sys_prefix")}
    defects = [] if isinstance(pynvc.get("imported"), bool) else ["pynvc.imported is required"]
    if surface["installed"] and not (_absolute(surface["interpreter"])
                                     and _absolute(surface["sys_prefix"])):
        defects.append("a verified pynvc identity requires an absolute interpreter and sys_prefix")
    return surface, defects


def normalized_python_bootstrap(environment: Any) -> dict[str, Any]:
    """Map installation.python venv/header evidence onto the ok/missing vocabulary."""
    python = _nested(environment, ("installation", "python"))
    python = python if isinstance(python, dict) else {}
    return {
        "venv_module": "ok" if python.get("venv_module") == "ok" else "missing",
        "development_headers": "ok" if python.get("development_headers") == "ok"
        else "missing"}


def read_json(source: Any) -> Any:
    """Parse strict JSON, rejecting NaN/Infinity. A Path is read as a symlink-free regular
    file; str/bytes are parsed as the document itself."""
    def reject(constant: str) -> None:
        raise ValueError(f"non-RFC JSON numeric constant is forbidden: {constant}")
    if isinstance(source, (str, bytes, bytearray)):
        return json.loads(source, parse_constant=reject)
    return json.loads(_read_regular(source, label="json document")[0], parse_constant=reject)


def write_new_bytes(path: Any, content: bytes, *, mode: int = 0o600) -> Path:
    """Exclusively create one fresh regular file, never following or replacing an existing path."""
    requested = _symlink_free(path, label="output")
    _check(requested.parent.is_dir(), f"output parent does not exist: {requested.parent}")
    resolved = requested.parent.resolve(strict=True) / requested.name
    descriptor = os.open(
        resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return resolved


def write_new_json(path: Any, value: Any, *, mode: int = 0o600) -> Path:
    """Exclusively create one fresh strict-JSON document with a trailing newline."""
    return write_new_bytes(path, (_dumps(value, sort_keys=True, indent=1) + "\n").encode("utf-8"),
                           mode=mode)


def transaction_sibling_path(path: Path, label: str,
                             observed_stems: tuple[str, ...] = ()) -> Path:
    """Resolve one plan-sibling path and preflight every declared fresh output."""
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"APT {label} path must not be a symlink")
    if requested.parent.resolve() != Path.cwd().resolve():
        raise ValueError(f"APT {label} must be written beside the reviewed plan")
    resolved = requested.parent.resolve() / requested.name
    outputs = [resolved] + [resolved.with_name(f"{resolved.name}{'.' + stem if stem else ''}"
                                               f".{stream}.log")
                            for stem in observed_stems for stream in ("stdout", "stderr")]
    existing = next((item for item in outputs if os.path.lexists(item)), None)
    if observed_stems and existing is not None:
        raise FileExistsError(f"APT {label} output already exists: {existing}")
    return resolved


# --- Bounded setup subprocesses ----------------------------------------------

def system_executable(name: str) -> str:
    """Return the one canonical system path for ``name``.

    Raises FileNotFoundError -- never ValueError -- when zero or several distinct executables
    answer to the name; callers branch on the absent-tool case by that exception type.
    """
    matches = set()
    for directory in ("/usr/bin", "/usr/sbin", "/bin", "/sbin"):
        try:
            resolved = (Path(directory) / name).resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            matches.add(str(resolved))
    unique = sorted(matches)
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one canonical system executable for {name}, found {unique}")
    return unique[0]


def _terminate_group(process: Any) -> None:
    """SIGTERM then unconditionally SIGKILL the child's whole group: a grandchild that ignores
    SIGTERM would otherwise hold the pipes and make communicate() wait forever."""
    for number, wait in ((signal.SIGTERM, True), (signal.SIGKILL, False)):
        try:
            os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError):
            pass
        if wait:
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass


def _redact_child_output(value: bytes) -> bytes:
    return _URL_USERINFO.sub(rb"\1***@", value)


def run_command(argv: list[str], *, cwd: Any = None, timeout: float | None = None,
                extra_env: Any = None) -> "subprocess.CompletedProcess[bytes]":
    """Run one no-shell command in its own session under the sanitized environment.

    The executable is resolved against the pinned PATH and the child gets its own session so the
    whole process GROUP can be signalled. On timeout the group is terminated and reaped and
    subprocess.TimeoutExpired carries the partial byte output.
    """
    command = list(argv) if isinstance(argv, list) else []
    _check(command and all(isinstance(token, str) and "\x00" not in token for token in command)
           and command[0], "command argv must be a non-empty list of NUL-free strings")
    env = {name: value for name in _ENV_ALLOWLIST if (value := os.environ.get(name)) is not None}
    env.update(_ENV_OVERRIDES)
    for name, value in dict(extra_env or {}).items():
        _check(isinstance(name, str) and name and "=" not in name and "\x00" not in name
               and isinstance(value, str) and "\x00" not in value,
               "extra_env names and values must be NUL-free strings")
        env[name] = value
    executable = shutil.which(command[0], path=env["PATH"])
    if executable is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), command[0])
    with subprocess.Popen(  # pylint: disable=consider-using-with
        command, executable=executable, cwd=str(cwd) if cwd is not None else None, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_group(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command, timeout, _redact_child_output(stdout), _redact_child_output(stderr)
            ) from exc
        return subprocess.CompletedProcess(
            command, process.returncode, _redact_child_output(stdout), _redact_child_output(stderr)
        )


# --- Jetson Linux release evidence -------------------------------------------

def parse_jetson_release(release_text: str) -> dict[str, Any]:
    """Parse the first physical line of nv_tegra_release into structured release fields.

    Returns the literal line plus release_major/release_minor/revision/version; an unparseable
    line yields all-None rather than a guess. Minimum-version policy is the caller's.
    """
    line = release_text.splitlines()[0] if release_text else ""
    release = re.search(r"\bR(\d+)(?:\.(\d+))?", line)
    if not release or int(release.group(1)) <= 0:
        return {"release_line": line, "release_major": None, "release_minor": None,
                "revision": None, "version": None}
    revision = re.search(r"\bREVISION\s*:\s*(\d+)\.(\d+)(?![\w.])", line, re.I)
    major = int(release.group(1))
    minor = (int(release.group(2)) if release.group(2) is not None
             else int(revision.group(1)) if revision else None)
    return {"release_line": line, "release_major": major, "release_minor": minor,
            "revision": f"{revision.group(1)}.{revision.group(2)}" if revision else None,
            "version": f"{major}.{minor}" if minor is not None else None}


# --- Configured APT sources and the stock signed public origin ---------------

def parse_apt_sources(path: Any, text_value: str) -> list[dict[str, Any]]:
    """Parse active binary APT source tuples and explicit authentication-bypass options.

    Handles the one-line ``.list`` format and deb822 stanzas alike, returning one
    ``{format, uri, suite, component, trust_bypass}`` record per enabled binary tuple.
    """
    entries: list[dict[str, Any]] = []
    source = Path(path)
    if source.suffix == ".list" or source.name == "sources.list":
        line_pattern = re.compile(
            r"^\s*deb\s+(?:\[([^\]]*)\]\s+)?(\S+)\s+(\S+)\s+([^#]+?)(?:\s+#.*)?$", re.I)
        bypass_pattern = re.compile(
            rf"(?:^|\s)(?:{'|'.join(_APT_BYPASS)})\s*=\s*(?:yes|true|1)(?:\s|$)", re.I)
        for line in text_value.splitlines():
            skip = not line.strip() or line.lstrip().startswith("#")
            match = None if skip else line_pattern.match(line)
            if match:
                options, uri, suite, components = match.groups()
                entries.extend(
                    {"format": "list", "uri": uri.rstrip("/"), "suite": suite,
                     "component": component,
                     "trust_bypass": bool(bypass_pattern.search(options or ""))}
                    for component in components.split())
        return entries
    fields: dict[str, str] = {}
    key: str | None = None
    for line in [*text_value.splitlines(), ""]:
        if not line.strip():
            if (fields.get("enabled", "yes").strip().lower() not in {"no", "false", "0"}
                    and "deb" in fields.get("types", "").split()):
                bypass = any(fields.get(name, "").strip().lower() in {"yes", "true", "1"}
                             for name in _APT_BYPASS)
                entries.extend(
                    {"format": "deb822", "uri": uri.rstrip("/"), "suite": suite,
                     "component": component, "trust_bypass": bypass}
                    for uri in fields.get("uris", "").split()
                    for suite in fields.get("suites", "").split()
                    for component in fields.get("components", "").split())
            fields, key = {}, None
        elif line.lstrip().startswith("#"):
            continue
        elif line[:1].isspace() and key:
            fields[key] = fields[key] + " " + line.strip()
        else:
            name, separator, value = line.partition(":")
            key = name.strip().lower() if separator else None
            if key is not None:
                fields[key] = value.strip()
    return entries


def _sources_bind_origin(origin: dict[str, Any]) -> bool:
    """Require every configured source file behind the origin to be hash-bound and bypass-free."""
    hashes = origin.get("configured_source_sha256")
    sources = origin.get("configured_sources")
    if not isinstance(hashes, list) or not isinstance(sources, list) or not hashes or not sources:
        return False
    if not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in hashes):
        return False
    suite, separator, component = str(origin.get("suite_component", "")).partition("/")
    expected = {"uri": str(origin.get("uri", "")).rstrip("/"), "suite": suite,
                "component": component, "trust_bypass": False}
    for item in sources:
        entry = item.get("entry") if isinstance(item, dict) else None
        if (not separator or not isinstance(entry, dict)
                or not isinstance(item.get("path"), str) or not Path(item["path"]).is_absolute()
                or not isinstance(item.get("sha256"), str)
                or not _SHA256.fullmatch(item["sha256"])
                or entry.get("format") not in {"list", "deb822"}
                or {name: entry.get(name) for name in expected} != expected):
            return False
    return sorted(hashes) == sorted(item["sha256"] for item in sources)


def _authenticated_origin(origin: Any) -> bool:
    """Recognize one configured APT origin protected by the package-signature chain."""
    if not isinstance(origin, dict) or origin.get("authentication") != "apt-signature-chain":
        return False
    uri, suite_component = origin.get("uri"), origin.get("suite_component")
    if not isinstance(uri, str) or not isinstance(suite_component, str):
        return False
    try:
        parsed = urlparse(uri)
        rejected = (parsed.username, parsed.password, parsed.port) != (None, None, None)
    except ValueError:
        return False
    if rejected or parsed.params or parsed.query or parsed.fragment:
        return False
    return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            and re.fullmatch(r"[^/\s]+/[^/\s]+", suite_component) is not None
            and _sources_bind_origin(origin))


def _public_origin(origin: Any) -> bool:
    """Recognize only the signature-authenticated stock public NVIDIA Jetson origin."""
    if not _authenticated_origin(origin):
        return False
    parsed = urlparse(origin["uri"])
    path = parsed.path[:-1] if parsed.path.endswith("/") else parsed.path
    return ((parsed.scheme, parsed.hostname, path) in _PUBLIC_APT_LOCATIONS
            and _PUBLIC_SUITE_COMPONENT.fullmatch(origin["suite_component"]) is not None)


def authenticated_candidate_binding(record: Any) -> dict[str, Any] | None:
    """Return a signature-authenticated configured origin for a base package, else None."""
    if (not isinstance(record, dict) or record.get("query_status") not in {"ok", "success"}
            or not isinstance(record.get("candidate"), str) or not record["candidate"]):
        return None
    origins = record.get("candidate_origins")
    approved = [item for item in origins if _authenticated_origin(item)] if isinstance(
        origins, list) else []
    return approved[0] if approved else None


def public_candidate_binding(record: Any) -> dict[str, Any] | None:
    """Return the stock signed public origin backing an apt-cache candidate, else None.

    Fails closed for internal mirrors, unsigned or trust-bypassed sources, a suite/component
    outside exact rNN.N/main, and any origin whose configured source files are not hash-bound to
    it. This is the native-SDK release-origin decision; never widen it.
    """
    if (not isinstance(record, dict) or record.get("query_status") not in {"ok", "success"}
            or not isinstance(record.get("candidate"), str) or not record["candidate"]):
        return None
    origins = record.get("candidate_origins")
    approved = [item for item in origins if _public_origin(item)] if isinstance(
        origins, list) else []
    return approved[0] if approved else None


# --- PyNvVideoCodec capability adapter ---------------------------------------

_ENCODER_CODECS = ("h264", "hevc", "av1")
_DECODER_CODECS = {
    "mpeg1": ("MPEG1", "MPEG_1"), "mpeg2": ("MPEG2", "MPEG_2"),
    "mpeg4": ("MPEG4", "MPEG_4"), "vc1": ("VC1", "VC_1"),
    "h264": ("H264", "H_264"), "hevc": ("HEVC", "H265", "H_265"),
    "vp8": ("VP8", "VP_8"), "vp9": ("VP9", "VP_9"), "av1": ("AV1",),
    "jpeg": ("JPEG",),
}
_API_13_CAPS = ("support_yuv422_encode", "support_mvhevc_encode",
                "support_temporal_filter", "support_lookahead_level",
                "support_unidirectional_b")


def _scalar_attributes(value: Any) -> dict[str, Any]:
    """Copy only JSON-scalar fields from a mapping or a PyBind capability object."""
    names = value if isinstance(value, dict) else (
        name for name in dir(value) if not name.startswith("_"))
    result: dict[str, Any] = {}
    for name in names:
        try:
            item = value[name] if isinstance(value, dict) else getattr(value, name)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if item is None or isinstance(item, (str, int, float, bool)):
            result[str(name)] = item
    return result


def _enum_member(module: Any, containers: tuple[str, ...], names: tuple[str, ...]) -> Any:
    for container_name in containers:
        container = getattr(module, container_name, None)
        for name in names:
            if container is not None and hasattr(container, name):
                return getattr(container, name)
    return next((getattr(module, name) for name in names if hasattr(module, name)), None)


def query_pynvc_capabilities(
    module: Any, gpu: int, linked_api: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Query the established schema-1.2 PyNv encoder and baseline decoder capability fields."""
    classification = {
        role: {"evidence_source_type": "api_query_helper", "official_sample": False,
               "api_symbol": f"PyNvVideoCodec.Get{role.title()}rCaps"}
        for role in ("encode", "decode")
    }
    result: dict[str, Any] = {
        "authority": "pynvc", "gpu": gpu, "evidence_classification": classification,
        "linked_nvenc_api": linked_api, "encode": {}, "decode": {},
    }
    if gpu != 0:
        reason = ("PyNvVideoCodec capability helpers route both encoder and decoder queries to"
                  " GPU 0; nonzero-GPU capability support remains unknown.")
        result.update({"status": "unknown", "reason": reason})
        return result, [reason]
    encoder, decoder = getattr(module, "GetEncoderCaps", None), getattr(
        module, "GetDecoderCaps", None)
    for codec in _ENCODER_CODECS:
        try:
            values = _scalar_attributes(encoder(gpu, codec)) if callable(encoder) else {}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result["encode"][codec] = {"status": "unknown", "query_status": "error",
                                       "supported": None, "error": str(exc)}
            continue
        if not values:
            result["encode"][codec] = {
                "status": "unknown", "supported": None,
                "reason": ("GetEncoderCaps returned no fields" if callable(encoder)
                           else "GetEncoderCaps unavailable")}
            continue
        unavailable = ({
            key: {"status": "unknown",
                  "reason": "capability is absent from the linked NVENC 12.1 header"}
            for key in _API_13_CAPS if key not in values
        } if linked_api.get("value") == "12.1" else {})
        result["encode"][codec] = {
            "status": "capability_reported", "supported": None,
            "session_status": "opened_by_GetEncoderCaps",
            "operation_status": "not_verified_by_probe",
            "interpretation": ("GetEncoderCaps returned capability fields but no explicit codec"
                               " support result; an official encode operation is still required"),
            "values": values, "unavailable_fields": unavailable,
        }
    chroma = _enum_member(module, ("cudaVideoChromaFormat", "ChromaFormat"),
                          ("420", "YUV420"))
    for codec, aliases in _DECODER_CODECS.items():
        codec_enum = _enum_member(module, ("cudaVideoCodec", "Codec", "VideoCodec"), aliases)
        try:
            values = (_scalar_attributes(decoder(gpu, codec_enum, chroma, 8))
                      if callable(decoder) and codec_enum is not None and chroma is not None
                      else {})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result["decode"][codec] = {"status": "unknown", "query_status": "error",
                                       "supported": None, "error": str(exc)}
            continue
        if not values:
            reason = ("GetDecoderCaps unavailable" if not callable(decoder) else
                      "required codec or 4:2:0 decoder enum was not found"
                      if codec_enum is None or chroma is None else
                      "GetDecoderCaps returned no fields")
            result["decode"][codec] = {
                "status": "unknown", "supported": None, "reason": reason}
            continue
        supported_value = values.get("supported")
        supported = (False if supported_value in (0, False) else True
                     if isinstance(supported_value, (int, bool)) and int(supported_value) > 0
                     else None)
        status = "unsupported" if supported is False else (
            "capability_reported" if supported is True else "unknown")
        interpretation = (
            "bIsSupported=0; remaining zeroed OUT fields are inapplicable"
            if supported is False else "bIsSupported=1; returned limits are applicable"
            if supported is True else "bIsSupported was absent or not interpretable")
        result["decode"][codec] = {
            "status": status, "supported": supported,
            "query": {"chroma": "420", "bit_depth": 8},
            "values": values, "interpretation": interpretation,
        }
    records = [*result["encode"].values(), *result["decode"].values()]
    states = {record.get("status") for record in records}
    successful = {"capability_reported", "unsupported"}
    result["status"] = ("complete" if records and states <= successful else
                        "partial" if states & successful else "unknown")
    if result["status"] != "complete":
        result["reason"] = ("Some capability queries succeeded while others are unknown or failed"
                            if result["status"] == "partial"
                            else "No capability query completed successfully")
    return result, []
