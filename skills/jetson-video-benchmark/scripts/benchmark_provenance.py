#!/usr/bin/env python3
"""Private authentication for official native and PyNvVideoCodec samples."""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Authentication keeps exact, auditable conjunctions rather than permissive helpers.
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes,too-many-locals
# This single module owns native and PyNvVideoCodec sample authentication end to end.
# pylint: disable=too-many-lines

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import benchmark_runtime as runtime

_COMMAND_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
_DEBIAN_VERSION = re.compile(
    r"^(?:(?:[0-9]+):)?(?P<upstream>13\.0(?:\.[0-9]+)*)"
    r"(?:\+[0-9A-Za-z.]+)?(?:-[0-9A-Za-z.+]+)?$"
)
# The frozen `nvcodec-environment` 1.2 contract. Authentication pins this exact
# version and rejects anything else; additive optional keys are ignored.
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_SCHEMA_VERSION = "1.2"
LOCAL_BINDING_KIND = "nvcodec-local-runtime-binding"
LOCAL_BINDING_SCHEMA_VERSION = "1.0"
NATIVE_PACKAGE = "nvidia-video-codec-sdk"
PYNVC_REQUIRED_VERSION = "2.1.0"
# The normalized native toolchain carries exactly these five entries, each
# {path, sha256}; `generator` adds the CMake generator `name`.
_BUILD_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
_CMAKE_GENERATORS = ("Ninja", "Unix Makefiles")
_VERSION_TOKEN = re.compile(r"[0-9][0-9A-Za-z.+:~-]*")
_RELEASE = re.compile(r"(?:[0-9]+:)?([0-9]+)\.([0-9]+)")
OFFICIAL_SAMPLE_FAILURE_PATTERN = re.compile(
    r"""(?imx)^\s*(?:
        traceback\ \(most\ recent\ call\ last\)
        | error\b
        | an\ (?:unexpected\ )?error\ occurred(?:\ with\ [^:\r\n]+)?:
        | operation\ or\ configuration\ not\ supported:
        | decode\ error\ occurred\ for\ picture\b
        | create(?:encoder|decoder)\ failure:
        | (?:worker\ [^:\r\n]+:\s*)?(?:an\ unexpected\ error\ occurred|setup\ failed):
        | (?:thread|process)\ [^:\r\n]+\ error:
        | encoding\ aborted:\ setup\ failed\b
        | nvdecoder::~nvdecoder\(\):\ exception\ during\ cleanup\ \(suppressed\)
        | (?:[a-z_][a-z0-9_:~]*\s*:\s*)?[a-z_][a-z0-9_.]*
          (?:\([^\r\n]*\))?\s+returned\ error\b
        | (?:cuda(?:\ (?:driver|runtime)\ api)?|nvenc|hresult|glenum|general|ffmpeg)
          \ error\b
        | cudacheckerror\(\)\ failed\b
        | \[(?:error|fatal)\s*\](?:\[[^]\r\n]+\])?
        | failed\b
        | failure\b
        | (?:operation|command|sample|encode|decode|encoding|decoding|encoder|
            decoder|setup|test)\b[^\r\n]*\b(?:failed|failure)\b
    )"""
)
_NATIVE_ROUTES = {
    "AppEncPerf": (
        "AppEncode/AppEncPerf/AppEncPerf",
        "Samples/AppEncode/AppEncPerf/AppEncPerf.cpp",
        "Samples/AppEncode/AppEncPerf/CMakeLists.txt",
        ("libcuda.so.1", "libnvidia-encode.so.1"),
    ),
    "AppDecPerf": (
        "AppDecode/AppDecPerf/AppDecPerf",
        "Samples/AppDecode/AppDecPerf/AppDecPerf.cpp",
        "Samples/AppDecode/AppDecPerf/CMakeLists.txt",
        ("libcuda.so.1", "libnvcuvid.so.1"),
    ),
}
_PYNVC_ROUTES = {
    "samples/advanced/encode_perf.py": (
        "samples/utils/__init__.py",
        "samples/utils/Utils.py",
        "samples/utils/encode_parser.py",
        "samples/utils/frame_utils.py",
        "samples/utils/encode_parallel_utils.py",
    ),
    "samples/advanced/decode_perf.py": (
        "samples/utils/__init__.py",
        "samples/utils/decode_parser.py",
    ),
}
_NATIVE_EXTENSION_MEMBER = re.compile(
    r"PyNvVideoCodec/_?PyNvVideoCodec(?:_(?:121|130))?\.[^/]*\.so"
)
_PROBE_PROGRAM = r"""
import importlib
import importlib.metadata
import json
import pathlib
import re
import sys

dist = importlib.metadata.distribution("PyNvVideoCodec")
name = str(dist.metadata.get("Name", ""))
files = [str(item).replace("\\", "/") for item in (dist.files or [])]
records = [item for item in files if item.endswith(".dist-info/RECORD")]
if len(records) != 1:
    raise RuntimeError("expected exactly one PyNvVideoCodec RECORD")
paths = {
    item: str(pathlib.Path(dist.locate_file(pathlib.Path(item))).resolve())
    for item in files
}
# Metadata alone cannot prove which extension the interpreter actually loaded,
# so import the package and report what is live.
package = importlib.import_module("PyNvVideoCodec")
loaded = sorted({
    str(pathlib.Path(module.__file__).resolve())
    for key, module in sys.modules.items()
    if (key == "_PyNvVideoCodec" or key.endswith("._PyNvVideoCodec"))
    and getattr(module, "__file__", None)
})
dependencies = {}
for distribution, module_name in (
    ("numpy", "numpy"),
    ("pycuda", "pycuda.driver"),
    ("torch", "torch"),
):
    record = {"status": "missing", "ready": False}
    try:
        module = importlib.import_module(module_name)
        record = {
            "status": "installed",
            "ready": True,
            "version": str(importlib.metadata.version(distribution)),
        }
        if distribution == "torch":
            record["cuda_build"] = str(getattr(module.version, "cuda", ""))
            record["cuda_available"] = bool(module.cuda.is_available())
            record["ready"] = record["cuda_available"]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    dependencies[distribution] = record
print(json.dumps({
    "schema_version": "1",
    "kind": "pynvc-record-probe",
    "interpreter": str(pathlib.Path(sys.executable).resolve()),
    "sys_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "name": name,
    "normalized_name": re.sub(r"[-_.]+", "-", name).lower(),
    "version": str(dist.version),
    "imported_version": str(getattr(package, "__version__", "")),
    "module_path": str(pathlib.Path(package.__file__).resolve()),
    "loaded_extensions": loaded,
    "record_entry": records[0],
    "paths": paths,
    "dependencies": dependencies,
}, sort_keys=True))
""".strip()


class ProvenanceError(ValueError):
    """Raised when official-sample authority cannot be established."""


@dataclass(frozen=True)
class FileSeal:
    """One protected external file identity."""

    identity: Mapping[str, Any]


@dataclass(frozen=True)
class AuthenticatedSample:
    """Frozen identity and launcher for one authenticated sample."""

    surface: str
    sample: str
    launcher: tuple[str, ...]
    workspace: str
    protected_files: tuple[FileSeal, ...]
    runtime_libraries: tuple[FileSeal, ...]
    environment_workspace: str
    environment_identity: Mapping[str, Any]


@dataclass(frozen=True)
class Authentication:
    """Authenticated sample set and its report identity."""

    samples: tuple[AuthenticatedSample, ...]
    report_identity: Mapping[str, Any]
    label: str

    def token(self, sample: str) -> AuthenticatedSample:
        """Return the unique authenticated token for a sample."""
        matches = [item for item in self.samples if item.sample == sample]
        if len(matches) != 1:
            raise ProvenanceError(f"{self.label} sample is not authenticated: {sample!r}")
        return matches[0]


def _identity(path: Path, kind: str) -> dict[str, Any]:
    return runtime.snapshot_external_artifact(path, schema_version="1", kind=kind)


def _sample_record(
    *,
    surface: str,
    sample: str,
    launcher: Sequence[str],
    workspace: Path,
    protected: Sequence[Mapping[str, Any]],
    runtime_libraries: Sequence[Mapping[str, Any]],
    environment_workspace: Path,
    environment_identity: Mapping[str, Any],
) -> AuthenticatedSample:
    return AuthenticatedSample(
        surface=surface,
        sample=sample,
        launcher=tuple(launcher),
        workspace=str(workspace),
        protected_files=tuple(FileSeal(dict(item)) for item in protected),
        runtime_libraries=tuple(FileSeal(dict(item)) for item in runtime_libraries),
        environment_workspace=str(environment_workspace),
        environment_identity=dict(environment_identity),
    )


def _environment(workspace: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    value = runtime.read_verified_json(workspace, identity)
    if not isinstance(value, dict):
        raise ProvenanceError("authentication requires one environment object")
    kind = value.get("kind")
    version = value.get("schema_version")
    if kind == ENVIRONMENT_KIND:
        if version != ENVIRONMENT_SCHEMA_VERSION:
            raise ProvenanceError(
                "environment schema_version must be exactly "
                f"{ENVIRONMENT_SCHEMA_VERSION!r}; refusing unknown environment "
                f"version {version!r}"
            )
        # Preserve the frozen setup-environment contract unchanged.
        if (
            value.get("mode") != "live"
            or not isinstance(value.get("installation"), Mapping)
            or not isinstance(value.get("pynvc"), Mapping)
        ):
            raise ProvenanceError(
                f"authentication requires one live {ENVIRONMENT_KIND} "
                f"{ENVIRONMENT_SCHEMA_VERSION} with installation and pynvc facts"
            )
    elif kind == LOCAL_BINDING_KIND:
        selected_gpu = value.get("selected_gpu")
        installation = value.get("installation")
        if (
            set(value) != {
                "schema_version",
                "kind",
                "mode",
                "source",
                "requested_runtime",
                "selected_gpu",
                "installation",
                "pynvc",
            }
            or
            version != LOCAL_BINDING_SCHEMA_VERSION
            or value.get("mode") != "live"
            or value.get("source") != "benchmark-local-discovery"
            or value.get("requested_runtime") not in {"native", "pynvc", "both"}
            or isinstance(selected_gpu, bool)
            or not isinstance(selected_gpu, int)
            or selected_gpu < 0
            or "surfaces" in value
            or not isinstance(installation, Mapping)
            or not all(
                isinstance(installation.get(name), Mapping)
                for name in (
                    "native_sdk",
                    "native_build_prerequisites",
                    "cuda_toolkit",
                    "build_tool_identities",
                    "python",
                )
            )
            or not isinstance(value.get("pynvc"), Mapping)
        ):
            raise ProvenanceError("local runtime binding is malformed")
    else:
        raise ProvenanceError(
            "authentication requires a setup environment or benchmark-local "
            "runtime binding"
        )
    return value


def _version(value: Any, label: str) -> str:
    """Validate one required, non-advisory version token."""
    if not isinstance(value, str) or _VERSION_TOKEN.fullmatch(value) is None:
        raise ProvenanceError(f"environment {label} version is missing or malformed")
    return value


def _upstream(version: str) -> str:
    """Return the upstream part of a Debian version: 1:13.2.1-1 -> 13.2.1."""
    return re.sub(r"^[0-9]+:", "", version).rsplit("-", 1)[0]


def _release(version: str, label: str) -> tuple[int, int]:
    """Return the (major, minor) release pair of one version token."""
    match = _RELEASE.match(version)
    if match is None:
        raise ProvenanceError(f"environment {label} version has no major.minor release")
    return int(match.group(1)), int(match.group(2))


def _run(
    service: Any,
    commands: list[dict[str, Any]],
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stage: str,
) -> tuple[dict[str, Any], str, str]:
    result = service.run(
        list(argv),
        cwd=cwd,
        env=_COMMAND_ENV,
        timeout_seconds=timeout_seconds,
        stage=stage,
        phase="prepare",
    )
    if not isinstance(result, dict):
        raise ProvenanceError("command runner returned a non-object result")
    commands.append(result)
    stdout = runtime.read_verified_text(service.workspace, result["stdout"])
    stderr = runtime.read_verified_text(service.workspace, result["stderr"])
    if result.get("timed_out") or result.get("launch_error") or result.get("exit_code") != 0:
        raise ProvenanceError(
            f"command failed during {stage}: exit={result.get('exit_code')} "
            f"timeout={result.get('timed_out')}: {stderr[-500:]}"
        )
    return result, stdout, stderr


def _native_root(environment: Mapping[str, Any]) -> tuple[Path, str]:
    """Resolve the one canonical SDK root and its declared package version.

    The normalized native surface carries a single `sdk_root`, taken from the
    one complete root 1.2 publishes. Package ownership of this root, and of
    every file built against it, is re-authenticated live in `_package_files`
    rather than replayed from the artifact's inventory.
    """
    native = runtime.environment_surface(environment, "native") or {}
    package = native.get("package")
    root_value = native.get("sdk_root")
    if (
        native.get("installed") is not True
        or not isinstance(package, Mapping)
        or package.get("name") != NATIVE_PACKAGE
        or package.get("status") != "installed"
        or not isinstance(root_value, str)
        or not Path(root_value).is_absolute()
    ):
        raise ProvenanceError(
            f"environment does not identify one installed {NATIVE_PACKAGE} root"
        )
    raw_version = package.get("version")
    match = _DEBIAN_VERSION.fullmatch(
        raw_version.strip() if isinstance(raw_version, str) else ""
    )
    if (
        match is None
        or "~" in str(raw_version)
        or "really" in str(raw_version).lower()
    ):
        raise ProvenanceError("native SDK package is not a stable public 13.0.x release")
    root = Path(root_value).resolve(strict=True)
    if not root.is_dir():
        raise ProvenanceError("native SDK root is not an existing directory")
    return root, str(raw_version)


def _live_package_inventory(
    service: Any,
    commands: list[dict[str, Any]],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> tuple[str, set[Path], set[Path]]:
    """Return the verified live package version, directories, and files."""
    _, output, _ = _run(
        service,
        commands,
        [
            "/usr/bin/dpkg-query",
            "-W",
            "-f=${db:Status-Abbrev}\\t${Version}\\t${binary:Package}\\n",
            NATIVE_PACKAGE,
        ],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-version",
    )
    fields = output.rstrip("\n").split("\t")
    if len(fields) != 3 or fields[0] != "ii " or fields[2] != NATIVE_PACKAGE:
        raise ProvenanceError(f"live dpkg does not report installed {NATIVE_PACKAGE}")
    version = fields[1]
    match = _DEBIAN_VERSION.fullmatch(version)
    if match is None or "~" in version or "really" in version.lower():
        raise ProvenanceError("native SDK package is not a stable public 13.0.x release")
    _, listing, _ = _run(
        service,
        commands,
        ["/usr/bin/dpkg-query", "-L", NATIVE_PACKAGE],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-files",
    )
    _, verified, _ = _run(
        service,
        commands,
        ["/usr/bin/dpkg", "--verify", NATIVE_PACKAGE],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-verify",
    )
    if verified.strip():
        raise ProvenanceError("dpkg --verify reported modified native SDK files")
    entries = [Path(line) for line in listing.splitlines() if line.startswith("/")]
    owned_directories = {item.resolve(strict=True) for item in entries if item.is_dir()}
    owned_files = {item.resolve(strict=True) for item in entries if item.is_file()}
    return version, owned_directories, owned_files


def _package_files(
    service: Any,
    commands: list[dict[str, Any]],
    *,
    cwd: Path,
    timeout_seconds: float,
    version: str,
    root: Path,
) -> set[Path]:
    """Re-authenticate live package ownership and return every owned file."""
    live_version, owned_directories, owned_files = _live_package_inventory(
        service,
        commands,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    if live_version != version:
        raise ProvenanceError("live dpkg package identity differs from runtime binding")
    if root not in owned_directories:
        raise ProvenanceError(
            f"live {NATIVE_PACKAGE} listing does not own the bound SDK root: {root}"
        )
    if not any(item.is_relative_to(root) for item in owned_files):
        raise ProvenanceError(
            f"live {NATIVE_PACKAGE} listing owns no file under the SDK root"
        )
    return owned_files


def _tool_record(environment: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    native = runtime.environment_surface(environment, "native") or {}
    tools = native.get("tools")
    record = tools.get(name) if isinstance(tools, Mapping) else None
    if not isinstance(record, Mapping):
        raise ProvenanceError(f"environment did not record required build tool {name}")
    return record


def _tool(environment: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Rehash one normalized build tool against its recorded identity.

    Schema 1.2 binds a build tool by path, size and content hash and publishes
    no tool version, so the binding is the live content hash itself. Every
    identity field the artifact did publish must still match; a version is
    validated only when one is present.
    """
    record = _tool_record(environment, name)
    path = record.get("path")
    digest = record.get("sha256")
    if record.get("version") is not None:
        _version(record.get("version"), f"tools.{name}")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ProvenanceError(
            f"environment has no canonical identity for build tool {name}"
        )
    observed = _identity(Path(path), f"native-build-tool-{name}")
    if any(
        record.get(field) not in (None, observed[field])
        for field in ("path", "size_bytes", "sha256")
    ):
        raise ProvenanceError(f"build tool {name} changed after environment probe")
    return observed


def _generator(environment: Mapping[str, Any]) -> str:
    """Return the exact CMake generator named by `tools.generator.name`."""
    name = _tool_record(environment, "generator").get("name")
    if name not in _CMAKE_GENERATORS:
        raise ProvenanceError(
            f"environment generator name must be exactly one of {list(_CMAKE_GENERATORS)}"
        )
    return str(name)


def _cuda(environment: Mapping[str, Any]) -> tuple[Path, str]:
    """Return the installed CUDA root and validated version from `cuda`, not `tools`.

    Both the toolkit status and its version are required, so both are validated
    here and the version is cross-checked against the live root marker by
    `_cuda_marker`.
    """
    native = runtime.environment_surface(environment, "native") or {}
    cuda = native.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("status") != "installed":
        raise ProvenanceError("environment does not report an installed CUDA toolkit")
    version = _version(cuda.get("version"), "cuda")
    root = cuda.get("root")
    if not isinstance(root, str):
        raise ProvenanceError("environment did not record the selected CUDA root")
    return Path(root).resolve(strict=True), version


def _cuda_marker(root: Path, version: str) -> Path:
    """Validate the recorded CUDA version against the live root version marker."""
    marker = next(
        (
            candidate
            for candidate in (root / "version.json", root / "version.txt")
            if candidate.is_file()
        ),
        None,
    )
    if marker is None:
        raise ProvenanceError("selected CUDA root has no version marker")
    try:
        payload = marker.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProvenanceError(f"selected CUDA version marker is unreadable: {exc}") from exc
    if _upstream(version) not in payload:
        raise ProvenanceError(
            f"CUDA root marker does not declare environment CUDA version {version!r}"
        )
    return marker


def _root_from_suffix(path: Path, suffix: str) -> Path | None:
    """Remove one exact relative suffix from an absolute path."""
    parts = Path(suffix).parts
    if len(path.parts) < len(parts) or path.parts[-len(parts):] != parts:
        return None
    return Path(*path.parts[:-len(parts)])


def _fixed_executable(candidates: Sequence[str], label: str) -> Path:
    """Resolve one executable from a bounded, ordered path allowlist."""
    for value in candidates:
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            details = resolved.stat()
        except OSError:
            continue
        if stat.S_ISREG(details.st_mode) and details.st_mode & 0o111:
            return resolved
    raise ProvenanceError(f"required native build tool is unavailable: {label}")


def discover_local_native_surface(  # pylint: disable=too-many-locals
    *,
    workspace: Path,
    samples: Sequence[str],
    runner: Any = None,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Discover only the fixed APT SDK package and its official sample build path."""
    root_workspace = runtime.resolve_private_workspace(workspace)
    requested = tuple(samples)
    if not requested or len(set(requested)) != len(requested):
        raise ProvenanceError("native sample list must be non-empty and unique")
    unknown = sorted(set(requested) - set(_NATIVE_ROUTES))
    if unknown:
        raise ProvenanceError(f"native sample is not allowlisted: {unknown}")
    service = runner or runtime.CommandRunner(root_workspace)
    commands: list[dict[str, Any]] = []
    version, owned_directories, owned_files = _live_package_inventory(
        service,
        commands,
        cwd=root_workspace,
        timeout_seconds=timeout_seconds,
    )
    roots: set[Path] = set()
    first_suffix = _NATIVE_ROUTES[requested[0]][1]
    for owned in owned_files:
        candidate = _root_from_suffix(owned, first_suffix)
        if candidate is None or candidate not in owned_directories:
            continue
        required = {candidate / "Samples/CMakeLists.txt"}
        for sample in requested:
            _binary, source_suffix, cmake_suffix, _libraries = _NATIVE_ROUTES[sample]
            required.update({candidate / source_suffix, candidate / cmake_suffix})
        if {item.resolve(strict=True) for item in required} <= owned_files:
            roots.add(candidate.resolve(strict=True))
    if len(roots) != 1:
        raise ProvenanceError(
            "live package does not expose exactly one complete official sample root"
        )
    sdk_root = next(iter(roots))
    cmake = _fixed_executable(("/usr/bin/cmake",), "cmake")
    cxx = _fixed_executable(("/usr/bin/g++",), "g++")
    pkg_config = _fixed_executable(("/usr/bin/pkg-config",), "pkg-config")
    try:
        generator = _fixed_executable(("/usr/bin/ninja",), "ninja")
        generator_name = "Ninja"
    except ProvenanceError:
        generator = _fixed_executable(("/usr/bin/make",), "make")
        generator_name = "Unix Makefiles"
    nvcc = _fixed_executable(
        (
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-13.2/bin/nvcc",
            "/usr/local/cuda-13.1/bin/nvcc",
            "/usr/local/cuda-13.0/bin/nvcc",
        ),
        "nvcc",
    )
    _result, nvcc_stdout, nvcc_stderr = _run(
        service,
        commands,
        [str(nvcc), "--version"],
        cwd=root_workspace,
        timeout_seconds=timeout_seconds,
        stage="native-nvcc-version",
    )
    match = re.search(r"\brelease\s+([0-9]+\.[0-9]+)\b", nvcc_stdout + nvcc_stderr)
    if match is None:
        raise ProvenanceError("nvcc did not report a canonical CUDA release")
    cuda_version = match.group(1)
    cuda_root = nvcc.parent.parent.resolve(strict=True)
    _cuda_marker(cuda_root, cuda_version)
    tools = {
        "cmake": _identity(cmake, "native-build-tool-cmake"),
        "cxx": _identity(cxx, "native-build-tool-cxx"),
        "nvcc": {
            **_identity(nvcc, "native-build-tool-nvcc"),
            "version": cuda_version,
        },
        "pkg_config": _identity(pkg_config, "native-build-tool-pkg_config"),
        "generator": {
            **_identity(generator, "native-build-tool-generator"),
            "name": generator_name,
        },
    }
    published_tools = {
        "cmake": dict(tools["cmake"]),
        "g++": dict(tools["cxx"]),
        "nvcc": dict(tools["nvcc"]),
        "pkg-config": dict(tools["pkg_config"]),
        ("ninja" if generator_name == "Ninja" else "make"): dict(
            tools["generator"]
        ),
    }
    package = {
        "name": NATIVE_PACKAGE,
        "status": "installed",
        "version": version,
    }
    return {
        "surface": {
            "installed": True,
            "package": package,
            "sdk_root": str(sdk_root),
            "build_prerequisites": {
                "status": "complete",
                "unresolved_modules": [],
            },
            "cuda": {
                "status": "installed",
                "version": cuda_version,
                "root": str(cuda_root),
            },
            "tools": tools,
        },
        "evidence": {
            "native_sdk": {
                "status": "installed",
                "package": package,
                "complete_roots": [str(sdk_root)],
            },
            "native_build_prerequisites": {
                "status": "complete",
                "missing_modules": [],
                "unknown_modules": [],
            },
            "cuda_toolkit": {
                "status": "available",
                "version": cuda_version,
                "nvcc_discovery": {"root": str(cuda_root)},
            },
            "build_tool_identities": published_tools,
        },
        "commands": commands,
        "authority": "live-dpkg-package-and-package-owned-official-sources",
    }


def _compile_sources(path: Path) -> set[Path]:
    value = runtime.strict_json_loads(path.read_bytes())
    if not isinstance(value, list) or not value:
        raise ProvenanceError("CMake compile_commands.json must be non-empty")
    sources: set[Path] = set()
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ProvenanceError("compile_commands.json entry has no source")
        sources.add(Path(entry["file"]).resolve(strict=True))
    return sources


def _runtime_libraries(output: str, required: Sequence[str]) -> list[dict[str, Any]]:
    if re.search(r"(?m)=>\s+not found\s*$", output):
        raise ProvenanceError("ldd reported an unresolved runtime library")
    resolved: dict[str, Path] = {}
    for line in output.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\/\S+)\s+\(", line)
        if match:
            library, path = match.groups()
            candidate = Path(path).resolve(strict=True)
            if "stubs" in candidate.parts:
                raise ProvenanceError(f"ldd resolved a CUDA stub library: {candidate}")
            resolved[library] = candidate
    missing = [name for name in required if name not in resolved]
    if missing:
        raise ProvenanceError(f"ldd did not resolve required codec libraries: {missing}")
    return [_identity(resolved[name], f"native-runtime-library-{name}") for name in required]


def _prepare_native(  # pylint: disable=too-many-arguments,too-many-locals
    sample: str,
    *,
    root: Path,
    owned: set[Path],
    environment: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    environment_workspace: Path,
    workspace: Path,
    service: Any,
    commands: list[dict[str, Any]],
    timeout_seconds: float,
) -> tuple[AuthenticatedSample, dict[str, Any]]:
    binary_suffix, source_suffix, cmake_suffix, libraries = _NATIVE_ROUTES[sample]
    source = (root / source_suffix).resolve(strict=True)
    cmake_file = (root / cmake_suffix).resolve(strict=True)
    top_cmake = (root / "Samples/CMakeLists.txt").resolve(strict=True)
    if not {source, cmake_file, top_cmake} <= owned:
        raise ProvenanceError(f"{sample} source/CMake files are not package-owned")
    tools = {name: _tool(environment, name) for name in _BUILD_TOOLS}
    generator = _generator(environment)
    cuda_root, cuda_version = _cuda(environment)
    if not Path(tools["nvcc"]["path"]).is_relative_to(cuda_root):
        raise ProvenanceError("environment nvcc is outside selected CUDA root")
    nvcc_version = _version(_tool_record(environment, "nvcc").get("version"), "tools.nvcc")
    if _release(nvcc_version, "tools.nvcc") != _release(
        _upstream(cuda_version), "cuda"
    ):
        raise ProvenanceError("environment nvcc release differs from selected CUDA release")
    marker = _cuda_marker(cuda_root, cuda_version)
    build = runtime.create_private_workspace(workspace / f"native-build-{sample}")
    configure = [
        tools["cmake"]["path"],
        "-S",
        str(root / "Samples"),
        "-B",
        str(build),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_CXX_COMPILER={tools['cxx']['path']}",
        f"-DCMAKE_CUDA_COMPILER={tools['nvcc']['path']}",
        f"-DCMAKE_MAKE_PROGRAM={tools['generator']['path']}",
        f"-DCMAKE_PREFIX_PATH={cuda_root}",
    ]
    _run(
        service,
        commands,
        configure,
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"configure-{sample.lower()}",
    )
    _run(
        service,
        commands,
        [tools["cmake"]["path"], "--build", str(build), "--target", sample, "--parallel", "1"],
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"build-{sample.lower()}",
    )
    cache_path = build / "CMakeCache.txt"
    compile_path = build / "compile_commands.json"
    cache: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("//", "#")) and "=" in line and ":" in line:
            name_type, value = line.split("=", 1)
            cache[name_type.split(":", 1)[0]] = value
    expected = {
        "CMAKE_HOME_DIRECTORY": str(root / "Samples"),
        "CMAKE_CXX_COMPILER": tools["cxx"]["path"],
        "CMAKE_CUDA_COMPILER": tools["nvcc"]["path"],
        "CMAKE_MAKE_PROGRAM": tools["generator"]["path"],
        "CMAKE_GENERATOR": generator,
    }
    for name, value in expected.items():
        observed = cache.get(name)
        differs = (
            observed != value
            if name == "CMAKE_GENERATOR"
            else not observed or Path(observed).resolve() != Path(value).resolve()
        )
        if differs:
            raise ProvenanceError(f"CMake cache disagrees for {name}")
    compiled = _compile_sources(compile_path)
    if source not in compiled or not compiled <= owned:
        raise ProvenanceError(f"{sample} is not bound to package-owned sources")
    executable = (build / binary_suffix).resolve(strict=True)
    if not stat.S_ISREG(executable.stat().st_mode) or executable.stat().st_mode & 0o111 == 0:
        raise ProvenanceError(f"built {sample} output is not executable")
    _, linkage, _ = _run(
        service,
        commands,
        ["/usr/bin/ldd", str(executable)],
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"linkage-{sample.lower()}",
    )
    linked = _runtime_libraries(linkage, libraries)
    protected = [
        _identity(executable, "native-official-sample-executable"),
        *(_identity(path, "native-package-source") for path in sorted(compiled)),
        _identity(cache_path, "native-cmake-cache"),
        _identity(compile_path, "native-compile-commands"),
        _identity(marker, "cuda-root-version-marker"),
        *tools.values(),
    ]
    token = _sample_record(
        surface="native",
        sample=sample,
        launcher=(str(executable),),
        workspace=workspace,
        protected=protected,
        runtime_libraries=linked,
        environment_workspace=environment_workspace,
        environment_identity=environment_identity,
    )
    return token, {
        "sample": sample,
        "launcher": list(token.launcher),
        "source": str(source),
        "configure_argv": configure,
        "compiled_sources": [str(path) for path in sorted(compiled)],
        "protected_files": protected,
        "runtime_libraries": linked,
    }


def authenticate_native_samples(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    environment_workspace: Path,
    environment_identity: Mapping[str, Any],
    workspace: Path,
    samples: Sequence[str],
    runner: Any = None,
    timeout_seconds: float = 600,
    report_path: Path | None = None,
) -> Authentication:
    """Build and authenticate the benchmark's allowlisted native samples."""
    root_workspace = runtime.resolve_private_workspace(workspace)
    requested = tuple(samples)
    if not requested or len(set(requested)) != len(requested):
        raise ProvenanceError("native sample list must be non-empty and unique")
    unknown = sorted(set(requested) - set(_NATIVE_ROUTES))
    if unknown:
        raise ProvenanceError(f"native sample is not allowlisted: {unknown}")
    destination = report_path or root_workspace / "native-sample-provenance.json"
    commands: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    records: list[AuthenticatedSample] = []
    service = runner or runtime.CommandRunner(root_workspace)
    report: dict[str, Any] = {
        "schema_version": "1",
        "kind": "native-sample-provenance",
        "status": "failed",
        "environment": dict(environment_identity),
        "environment_workspace": str(
            runtime.resolve_private_workspace(environment_workspace)
        ),
        "requested_samples": list(requested),
        "samples": evidence,
        "commands": commands,
    }
    try:
        environment = _environment(environment_workspace, environment_identity)
        sdk_root, version = _native_root(environment)
        owned = _package_files(
            service,
            commands,
            cwd=root_workspace,
            timeout_seconds=timeout_seconds,
            version=version,
            root=sdk_root,
        )
        for sample in requested:
            token, item = _prepare_native(
                sample,
                root=sdk_root,
                owned=owned,
                environment=environment,
                environment_identity=environment_identity,
                environment_workspace=runtime.resolve_private_workspace(
                    environment_workspace
                ),
                workspace=root_workspace,
                service=service,
                commands=commands,
                timeout_seconds=timeout_seconds,
            )
            records.append(token)
            evidence.append(item)
        report["status"] = "authenticated"
        report["package"] = {
            "name": NATIVE_PACKAGE,
            "version": version,
            "root": str(sdk_root),
            "ownership": "re-authenticated-live",
        }
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        runtime.write_fresh_json(root_workspace, destination, report)
        raise
    identity = runtime.write_fresh_json(root_workspace, destination, report)
    return Authentication(tuple(records), identity, "native")


def _pynvc_environment(
    workspace: Path, identity: Mapping[str, Any]
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Bind the normalized Py surface identity to the live venv it names.

    The surface carries identity only: the lexical `interpreter` as invoked,
    the resolved `interpreter_identity`, `sys_prefix`, the PyNv `version`, the
    `module` and the `extension`. Every binding below is proven against the
    live filesystem rather than replayed from the artifact.
    """
    value = _environment(workspace, identity)
    pynvc = runtime.environment_surface(value, "pynvc") or {}
    interpreter_identity = pynvc.get("interpreter_identity")
    if (
        pynvc.get("installed") is not True
        or pynvc.get("version") != PYNVC_REQUIRED_VERSION
        or not isinstance(interpreter_identity, Mapping)
    ):
        raise ProvenanceError(
            f"environment lacks installed PyNvVideoCodec {PYNVC_REQUIRED_VERSION}"
        )
    launcher_value = pynvc.get("interpreter")
    resolved_value = interpreter_identity.get("path")
    prefix_value = pynvc.get("sys_prefix")
    digest = interpreter_identity.get("sha256")
    if not all(
        isinstance(item, str)
        for item in (launcher_value, resolved_value, prefix_value, digest)
    ) or not Path(str(launcher_value)).is_absolute():
        raise ProvenanceError("environment lacks selected PyNvVideoCodec venv")
    launcher = Path(str(launcher_value))
    resolved = launcher.resolve(strict=True)
    if (
        not launcher.is_file()
        or launcher.stat().st_mode & 0o111 == 0
        or resolved != Path(str(resolved_value)).resolve(strict=True)
        or launcher.parent.parent.resolve(strict=True)
        != Path(str(prefix_value)).resolve(strict=True)
    ):
        raise ProvenanceError("selected venv launcher binding is invalid")
    if _identity(resolved, "pynvc-venv-python")["sha256"] != digest:
        raise ProvenanceError("selected interpreter changed after environment probe")
    # The imported module version must equal the distribution version.
    module = pynvc.get("module")
    if (
        not isinstance(module, Mapping)
        or not isinstance(module.get("path"), str)
        or module.get("version") != PYNVC_REQUIRED_VERSION
    ):
        raise ProvenanceError(
            "environment imported PyNvVideoCodec module version differs from "
            f"distribution version {PYNVC_REQUIRED_VERSION}"
        )
    return launcher, resolved, pynvc


def _probe_pynvc(
    launcher: Path,
    *,
    workspace: Path,
    service: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    result = service.run(
        [str(launcher), "-I", "-c", _PROBE_PROGRAM],
        cwd=workspace,
        env=_COMMAND_ENV,
        timeout_seconds=timeout_seconds,
        stage="pynvc-record-probe",
        phase="probe",
    )
    stderr = runtime.read_verified_text(service.workspace, result["stderr"])
    if result.get("timed_out") or result.get("launch_error") or result.get("exit_code") != 0:
        raise ProvenanceError(f"isolated PyNvVideoCodec RECORD probe failed: {stderr[-500:]}")
    value = runtime.strict_json_loads(
        runtime.read_verified_text(service.workspace, result["stdout"])
    )
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "1"
        or value.get("kind") != "pynvc-record-probe"
        or value.get("interpreter") != str(launcher.resolve(strict=True))
        or value.get("normalized_name") != "pynvvideocodec"
        or value.get("version") != PYNVC_REQUIRED_VERSION
        or value.get("imported_version") != PYNVC_REQUIRED_VERSION
        or not isinstance(value.get("paths"), dict)
        or not isinstance(value.get("module_path"), str)
    ):
        raise ProvenanceError("isolated probe did not select PyNvVideoCodec 2.1.0")
    # Exactly one extension must actually be loaded by the interpreter.
    loaded = value.get("loaded_extensions")
    if not isinstance(loaded, list) or len(loaded) != 1 or not isinstance(loaded[0], str):
        raise ProvenanceError(
            "isolated probe did not load exactly one PyNvVideoCodec extension"
        )
    prefix = Path(value.get("sys_prefix", "")).resolve(strict=True)
    if launcher.parent.parent.resolve(strict=True) != prefix:
        raise ProvenanceError("isolated probe sys.prefix differs from selected venv")
    for member, path_value in value["paths"].items():
        if (
            not isinstance(member, str)
            or member != member.replace("\\", "/")
            or not isinstance(path_value, str)
            or not Path(path_value).resolve(strict=True).is_relative_to(prefix)
        ):
            raise ProvenanceError("wheel member resolved outside selected venv")
    value["command"] = result
    return value


def discover_local_pynvc_surface(
    *,
    interpreter: str,
    workspace: Path,
    runner: Any = None,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Probe one exact caller-selected interpreter; never search for a venv."""
    if (
        not isinstance(interpreter, str)
        or not interpreter
        or interpreter != interpreter.strip()
        or not Path(interpreter).is_absolute()
        or str(Path(interpreter)) != interpreter
    ):
        raise ProvenanceError(
            "pynvc_interpreter must be one exact canonical absolute path"
        )
    launcher = Path(interpreter)
    try:
        details = launcher.stat()
        resolved = launcher.resolve(strict=True)
    except OSError as exc:
        raise ProvenanceError(f"pynvc_interpreter is unavailable: {exc}") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o111 == 0:
        raise ProvenanceError("pynvc_interpreter must be an executable regular file")
    root = runtime.resolve_private_workspace(workspace)
    service = runner or runtime.CommandRunner(root)
    probe = _probe_pynvc(
        launcher,
        workspace=root,
        service=service,
        timeout_seconds=timeout_seconds,
    )
    extension = _loaded_extension_identity(probe)
    dependencies = probe.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ProvenanceError("isolated PyNvVideoCodec probe omitted dependencies")
    surface = {
        "installed": True,
        "version": probe["version"],
        "interpreter": interpreter,
        "interpreter_identity": _identity(resolved, "pynvc-venv-python"),
        "sys_prefix": probe["sys_prefix"],
        "extension": extension,
        "module": {
            "path": probe["module_path"],
            "version": probe["imported_version"],
        },
        "dependencies": dict(dependencies),
    }
    package_records = {
        name: {
            **dict(record),
            "requirement_satisfied": record.get("ready") is True,
        }
        for name, record in dependencies.items()
        if isinstance(record, Mapping)
    }
    return {
        "surface": surface,
        "evidence": {
            "python": {
                "interpreter_identities": {
                    "running": dict(surface["interpreter_identity"])
                },
                "packages": package_records,
            },
            "pynvc": {
                "imported": True,
                "distribution_version": probe["version"],
                "identity": {
                    "status": "verified",
                    "interpreter": interpreter,
                    "sys_prefix": probe["sys_prefix"],
                    "extension": extension,
                    "module": dict(surface["module"]),
                },
            },
        },
        "command": probe["command"],
        "authority": "exact-interpreter-wheel-record-and-loaded-extension",
    }


def _bind_probe(probe: Mapping[str, Any], pynvc: Mapping[str, Any]) -> Path:
    """Bind the live probe to the recorded Py surface and return the venv prefix.

    What this interpreter actually imports must equal what the environment
    authenticated, not merely what distribution metadata locates on disk. The
    interpreter's single loaded extension is compared against the authenticated
    `extension.path`; an environment that additively publishes `loaded_path`
    is bound to that instead, and both must name the same file.
    """
    prefix = Path(str(probe["sys_prefix"])).resolve(strict=True)
    if (
        prefix != Path(str(pynvc["sys_prefix"])).resolve(strict=True)
        or probe.get("version") != pynvc.get("version")
    ):
        raise ProvenanceError("isolated RECORD probe differs from environment")
    module = pynvc["module"]
    extension = pynvc.get("extension") if isinstance(pynvc.get("extension"), Mapping) else {}
    loaded_path = extension.get("loaded_path")
    if loaded_path is None:
        loaded_path = extension.get("path")
    if not isinstance(loaded_path, str):
        raise ProvenanceError("environment lacks the authenticated PyNvVideoCodec extension")
    if (
        probe.get("imported_version") != module.get("version")
        or Path(str(probe.get("module_path")))
        != Path(str(module.get("path"))).resolve(strict=True)
        or probe.get("loaded_extensions")
        != [str(Path(loaded_path).resolve(strict=True))]
    ):
        raise ProvenanceError(
            "isolated probe loaded a different PyNvVideoCodec module or extension "
            "than the environment recorded"
        )
    return prefix


def _record_rows(
    probe: Mapping[str, Any]
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    paths = probe.get("paths")
    member = probe.get("record_entry")
    if not isinstance(paths, dict) or not isinstance(member, str):
        raise ProvenanceError("record probe shape is invalid")
    path_value = paths.get(member)
    if not isinstance(path_value, str):
        raise ProvenanceError("record probe did not resolve RECORD")
    prefix = Path(str(probe["sys_prefix"])).resolve(strict=True)
    record_path = Path(path_value).resolve(strict=True)
    if not record_path.is_relative_to(prefix):
        raise ProvenanceError("wheel RECORD is outside selected venv")
    try:
        rows = list(
            csv.reader(
                io.StringIO(record_path.read_bytes().decode("utf-8", errors="strict"))
            )
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ProvenanceError(f"wheel RECORD is invalid: {exc}") from exc
    parsed: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] != row[0].replace("\\", "/"):
            raise ProvenanceError("wheel RECORD row is not canonical")
        if row[0] in parsed:
            raise ProvenanceError(f"wheel RECORD duplicate member: {row[0]!r}")
        parsed[row[0]] = (row[1], row[2])
    return parsed, _identity(record_path, "pynvc-wheel-record")


def _loaded_extension_identity(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the one loaded native extension is owned by the selected wheel."""
    loaded = probe.get("loaded_extensions")
    paths = probe.get("paths")
    if (
        not isinstance(loaded, list)
        or len(loaded) != 1
        or not isinstance(loaded[0], str)
        or not isinstance(paths, Mapping)
    ):
        raise ProvenanceError("probe did not expose one loaded wheel extension")
    prefix = Path(str(probe["sys_prefix"])).resolve(strict=True)
    extension_path = Path(loaded[0]).resolve(strict=True)
    if not extension_path.is_relative_to(prefix):
        raise ProvenanceError("loaded PyNvVideoCodec extension is outside selected venv")
    members = [
        member
        for member, value in paths.items()
        if isinstance(member, str)
        and isinstance(value, str)
        and Path(value).resolve(strict=True) == extension_path
    ]
    rows, _record_identity = _record_rows(probe)
    if len(members) != 1 or members[0] not in rows:
        raise ProvenanceError("loaded extension is not one unique wheel RECORD member")
    member = members[0]
    encoded_hash, size_text = rows[member]
    if not encoded_hash.startswith("sha256=") or not size_text.isdecimal():
        raise ProvenanceError("loaded extension has no wheel RECORD SHA-256/size")
    payload = extension_path.read_bytes()
    actual = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode()
    )
    matches_record = (
        encoded_hash.removeprefix("sha256=") == actual
        and int(size_text) == len(payload)
    )
    known_stale = (
        probe.get("version") == PYNVC_REQUIRED_VERSION
        and _NATIVE_EXTENSION_MEMBER.fullmatch(member) is not None
    )
    if not matches_record and not known_stale:
        raise ProvenanceError("loaded extension differs from wheel RECORD")
    identity = _identity(extension_path, "pynvc-extension")
    return {
        **identity,
        "loaded_path": identity["path"],
        "record_member": member,
        "record_consistency": (
            "matches_record" if matches_record else "known_upstream_stale_record"
        ),
    }


def _record_members(
    probe: Mapping[str, Any], members: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    rows, _identity_value = _record_rows(probe)
    paths = probe["paths"]
    prefix = Path(str(probe["sys_prefix"])).resolve(strict=True)
    result: list[dict[str, Any]] = []
    for member in members:
        if member not in rows or member not in paths:
            raise ProvenanceError(f"wheel RECORD member is missing: {member!r}")
        encoded_hash, size_text = rows[member]
        if not encoded_hash.startswith("sha256=") or not size_text.isdecimal():
            raise ProvenanceError(f"wheel RECORD member has no SHA-256/size: {member}")
        path = Path(paths[member]).resolve(strict=True)
        if not path.is_relative_to(prefix):
            raise ProvenanceError("wheel member resolved outside selected venv")
        payload = path.read_bytes()
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode()
        )
        if encoded_hash.removeprefix("sha256=") != actual or int(size_text) != len(payload):
            raise ProvenanceError(f"installed wheel member differs from RECORD: {member}")
        result.append(
            {
                "member": member,
                "record_hash": encoded_hash,
                "identity": _identity(path, "pynvc-wheel-member"),
            }
        )
    return tuple(result)


def _extension_seal(
    pynvc: Mapping[str, Any], prefix: Path
) -> list[dict[str, Any]]:
    """Seal the extension the interpreter actually loaded.

    Hashing a file found through distribution metadata would prove only that a
    matching file exists on disk, so the live bytes must hash to the recorded
    digest inside the selected venv prefix, and `_bind_probe` has already
    required the interpreter's one loaded extension to be this exact file. An
    environment that additively publishes `extension.loaded_path` must name the
    same file; 1.2 publishes none and nothing is inferred from its absence.
    """
    extension = pynvc.get("extension") if isinstance(pynvc.get("extension"), Mapping) else {}
    path_value = extension.get("path")
    digest = extension.get("sha256")
    loaded_path = extension.get("loaded_path")
    if loaded_path is None:
        loaded_path = path_value
    if (
        not isinstance(path_value, str)
        or not isinstance(digest, str)
        or not isinstance(loaded_path, str)
    ):
        raise ProvenanceError("environment lacks a PyNvVideoCodec extension identity")
    if loaded_path != path_value:
        raise ProvenanceError(
            "environment loaded extension differs from the authenticated extension"
        )
    identity = _identity(Path(path_value), "pynvc-extension")
    if identity["path"] != path_value or identity["sha256"] != digest:
        raise ProvenanceError("installed extension differs from the environment identity")
    if not Path(identity["path"]).is_relative_to(prefix):
        raise ProvenanceError("PyNvVideoCodec extension resolved outside selected venv")
    return [identity]


def authenticate_pynvc_samples(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    environment_workspace: Path,
    environment_identity: Mapping[str, Any],
    workspace: Path,
    samples: Sequence[str],
    runner: Any = None,
    timeout_seconds: float = 600,
    report_path: Path | None = None,
) -> Authentication:
    """Authenticate the benchmark's wheel-owned Python performance samples."""
    root = runtime.resolve_private_workspace(workspace)
    env_root = runtime.resolve_private_workspace(environment_workspace)
    requested = tuple(samples)
    if not requested or len(set(requested)) != len(requested):
        raise ProvenanceError("PyNvVideoCodec sample list must be non-empty and unique")
    unknown = sorted(set(requested) - set(_PYNVC_ROUTES))
    if unknown:
        raise ProvenanceError(f"PyNvVideoCodec sample is not allowlisted: {unknown}")
    destination = report_path or root / "pynvc-sample-provenance.json"
    report: dict[str, Any] = {
        "schema_version": "1",
        "kind": "pynvc-sample-provenance",
        "status": "failed",
        "environment": dict(environment_identity),
        "requested_samples": list(requested),
        "samples": [],
    }
    try:
        launcher, interpreter, pynvc = _pynvc_environment(
            env_root, environment_identity
        )
        service = runner or runtime.CommandRunner(root)
        probe = _probe_pynvc(
            launcher,
            workspace=root,
            service=service,
            timeout_seconds=timeout_seconds,
        )
        prefix = _bind_probe(probe, pynvc)
        record_member = probe.get("record_entry")
        if not isinstance(record_member, str) or not record_member.endswith(
            ".dist-info/RECORD"
        ):
            raise ProvenanceError("wheel RECORD member is invalid")
        metadata = record_member.removesuffix("RECORD") + "METADATA"
        wheel = record_member.removesuffix("RECORD") + "WHEEL"
        # The package module is proven live rather than replayed:
        # `_record_members` requires it to be an exact wheel RECORD member
        # resolving inside the selected venv prefix.
        common = _record_members(
            probe, ("PyNvVideoCodec/__init__.py", metadata, wheel)
        )
        _rows, record_identity = _record_rows(probe)
        extension = _extension_seal(pynvc, prefix)
        interpreter_identity = _identity(interpreter, "pynvc-venv-python")
        records: list[AuthenticatedSample] = []
        for sample in requested:
            members = (sample, *_PYNVC_ROUTES[sample])
            owned = _record_members(probe, members)
            protected = [
                interpreter_identity,
                record_identity,
                *(item["identity"] for item in common),
                *(item["identity"] for item in owned),
                *extension,
            ]
            token = _sample_record(
                surface="pynvc",
                sample=sample,
                launcher=(str(launcher), "-I", probe["paths"][sample]),
                workspace=root,
                protected=protected,
                runtime_libraries=(),
                environment_workspace=env_root,
                environment_identity=environment_identity,
            )
            records.append(token)
            report["samples"].append(
                {
                    "sample": sample,
                    "launcher": list(token.launcher),
                    "record_members": list(members),
                    "protected_files": protected,
                }
            )
        report["status"] = "authenticated"
        report["distribution"] = {
            "reported_name": probe["name"],
            "normalized_name": probe["normalized_name"],
            "version": probe["version"],
        }
        report["record_probe_command"] = probe["command"]
        report["record"] = record_identity
        report["record_member"] = record_member
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        runtime.write_fresh_json(root, destination, report)
        raise
    identity = runtime.write_fresh_json(root, destination, report)
    return Authentication(tuple(records), identity, "PyNvVideoCodec")


def run_authenticated(  # pylint: disable=too-many-arguments
    token: AuthenticatedSample,
    arguments: Sequence[str],
    *,
    workspace: Path,
    runner: Any,
    cwd: Path,
    timeout_seconds: float,
    stage: str,
    phase: str,
) -> dict[str, Any]:
    """Immediately rehash every seal, then launch the exact sample argv."""
    if not isinstance(token, AuthenticatedSample):
        raise ProvenanceError("official sample has no authenticated record")
    root = runtime.resolve_private_workspace(workspace)
    if root != Path(token.workspace).resolve(strict=True):
        raise ProvenanceError("launch workspace differs from authenticated workspace")
    command_cwd = Path(cwd).resolve(strict=True)
    if not command_cwd.is_relative_to(root):
        raise ProvenanceError("official sample cwd must be inside private workspace")
    suffix = tuple(arguments)
    if any(not isinstance(item, str) or "\0" in item for item in suffix):
        raise ProvenanceError("official sample arguments must be NUL-free strings")
    runtime.verify_artifact(
        Path(token.environment_workspace), token.environment_identity
    )
    for item in (*token.protected_files, *token.runtime_libraries):
        runtime.verify_external_artifact(item.identity)
    executable = Path(token.launcher[0])
    if executable.stat().st_mode & 0o111 == 0:
        raise ProvenanceError("authenticated launcher is no longer executable")
    expected_kind = {
        "native": "native-official-sample-executable",
        "pynvc": "pynvc-venv-python",
    }.get(token.surface)
    seals = [
        seal
        for seal in token.protected_files
        if seal.identity.get("kind") == expected_kind
    ]
    if expected_kind is None or len(seals) != 1:
        raise ProvenanceError("authenticated launcher seal is absent or duplicated")
    if executable.resolve(strict=True) != Path(seals[0].identity["path"]):
        raise ProvenanceError("authenticated launcher resolves outside sealed target")
    argv = [*token.launcher, *suffix]
    result = runner.run(
        argv,
        cwd=command_cwd,
        env=_COMMAND_ENV,
        timeout_seconds=timeout_seconds,
        stage=stage,
        phase=phase,
    )
    expected = {
        "argv": argv,
        "cwd": str(command_cwd),
        "stage": stage,
        "phase": phase,
        "timeout_seconds": float(timeout_seconds),
    }
    if not isinstance(result, dict) or any(
        result.get(key) != value for key, value in expected.items()
    ):
        raise ProvenanceError("command evidence does not match authenticated launch")
    return result
