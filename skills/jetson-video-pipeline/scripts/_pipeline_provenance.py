#!/usr/bin/env python3
"""Private official-sample authentication for pipeline operations."""
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import _pipeline_runtime as artifact_io
# The two authenticators intentionally validate complete provenance
# conjunctions in one owner.
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-lines
command_runner = artifact_io
_COMMAND_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
_DEBIAN_VERSION = re.compile(
    r"^(?:(?P<epoch>[0-9]+):)?(?P<upstream>13\.0(?:\.[0-9]+)*)"
    r"(?:\+[0-9A-Za-z.]+)?(?:-[0-9A-Za-z.+]+)?$"
)
# The normalized native toolchain names exactly these five build tools; the
# generator record adds its CMake generator name.
_BUILD_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
_CMAKE_GENERATORS = frozenset({"Ninja", "Unix Makefiles"})
_PKG_CONFIG_MODULES = ("libavcodec", "libavformat", "libavutil", "libswresample")
_SYSTEM_TOOLS = {
    "cmake": Path("/usr/bin/cmake"),
    "g++": Path("/usr/bin/g++"),
    "pkg-config": Path("/usr/bin/pkg-config"),
    "ninja": Path("/usr/bin/ninja"),
    "make": Path("/usr/bin/make"),
}
_DIRECT_OUTPUT_LIMIT = 1024 * 1024
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
NATIVE_ROUTES = {
    "AppEncCuda": (
        "AppEncode/AppEncCuda/AppEncCuda",
        "Samples/AppEncode/AppEncCuda/AppEncCuda.cpp",
        "Samples/AppEncode/AppEncCuda/CMakeLists.txt",
        ("libcuda.so.1", "libnvidia-encode.so.1"),
    ),
    "AppDec": (
        "AppDecode/AppDec/AppDec",
        "Samples/AppDecode/AppDec/AppDec.cpp",
        "Samples/AppDecode/AppDec/CMakeLists.txt",
        ("libcuda.so.1", "libnvcuvid.so.1"),
    ),
    "AppTrans": (
        "AppTranscode/AppTrans/AppTrans",
        "Samples/AppTranscode/AppTrans/AppTrans.cpp",
        "Samples/AppTranscode/AppTrans/CMakeLists.txt",
        ("libcuda.so.1", "libnvcuvid.so.1", "libnvidia-encode.so.1"),
    ),
}
PYNVC_ROUTES = {
    "samples/basic/encode.py": (
        "samples/utils/__init__.py",
        "samples/utils/Utils.py",
        "samples/utils/encode_parser.py",
        "samples/utils/frame_utils.py",
    ),
    "samples/advanced/decode.py": (
        "samples/utils/__init__.py",
        "samples/utils/Utils.py",
        "samples/utils/decode_parser.py",
    ),
    "samples/basic/create_video_segments.py": (
        "samples/utils/__init__.py",
        "samples/utils/transcode_parser.py",
        "samples/basic/segments.txt",
        "samples/basic/transcode_config.json",
    ),
}


class ProvenanceError(ValueError):
    """Official sample authority cannot be established."""


PynvcProvenanceError = ProvenanceError


@dataclass(frozen=True)
class FileSeal:
    """One protected external file identity."""
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class AuthenticatedSample:
    """Frozen authority for one exact sample launcher."""
    surface: str
    sample: str
    launcher: tuple[str, ...]
    workspace: str
    protected_files: tuple[FileSeal, ...]
    runtime_libraries: tuple[FileSeal, ...]
    environment_workspace: str
    environment_identity: Mapping[str, Any]


def _select(
    samples: Sequence[AuthenticatedSample], sample: str, surface: str
) -> AuthenticatedSample:
    matches = [item for item in samples if item.sample == sample]
    if len(matches) != 1:
        raise ProvenanceError(f"{surface} sample is not authenticated: {sample!r}")
    return matches[0]


@dataclass(frozen=True)
class NativeAuthentication:
    """Native sample records and provenance report."""
    samples: tuple[AuthenticatedSample, ...]
    report_identity: Mapping[str, Any]

    def token(self, sample: str) -> AuthenticatedSample:
        """Return the unique authenticated native sample."""
        return _select(self.samples, sample, "native")


@dataclass(frozen=True)
class PynvcAuthentication:
    """PyNvVideoCodec sample records and provenance report."""
    samples: tuple[AuthenticatedSample, ...]
    report_identity: Mapping[str, Any]

    def token(self, sample: str) -> AuthenticatedSample:
        """Return the unique authenticated Python sample."""
        return _select(self.samples, sample, "PyNvVideoCodec")


def _identity(path: Path, kind: str) -> dict[str, Any]:
    return artifact_io.snapshot_external_artifact(
        path, schema_version="1", kind=kind
    )


def _sample_record(
    *,
    surface: str,
    sample: str,
    launcher: tuple[str, ...],
    workspace: Path,
    protected: Sequence[Mapping[str, Any]],
    runtime: Sequence[Mapping[str, Any]],
    environment_workspace: Path,
    environment_identity: Mapping[str, Any],
) -> AuthenticatedSample:
    return AuthenticatedSample(
        surface=surface,
        sample=sample,
        launcher=launcher,
        workspace=str(workspace),
        protected_files=tuple(FileSeal(dict(item)) for item in protected),
        runtime_libraries=tuple(FileSeal(dict(item)) for item in runtime),
        environment_workspace=str(environment_workspace),
        environment_identity=dict(environment_identity),
    )


def _direct_read_only(argv: Sequence[str], *, timeout_seconds: float) -> str:
    """Run one fixed read-only probe without creating an evidence workspace."""
    try:
        completed = subprocess.run(
            list(argv),
            cwd="/",
            env=_COMMAND_ENV,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(f"read-only local probe failed: {argv[0]}: {exc}") from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > _DIRECT_OUTPUT_LIMIT
        or len(completed.stderr) > _DIRECT_OUTPUT_LIMIT
    ):
        detail = completed.stderr[-500:].decode("utf-8", errors="replace")
        raise ProvenanceError(
            f"read-only local probe failed: {list(argv)!r}: "
            f"exit={completed.returncode}: {detail}"
        )
    return completed.stdout.decode("utf-8", errors="strict")


def _plain_identity(path: Path) -> dict[str, Any]:
    identity = _identity(path, "local-runtime-file")
    return {
        field: identity[field] for field in ("path", "size_bytes", "sha256")
    }


def _system_tool(name: str) -> Path:
    candidate = _SYSTEM_TOOLS.get(name)
    if candidate is None:
        raise ProvenanceError(f"native build tool is not allowlisted: {name}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(
            f"required native build tool is missing: {name}"
        ) from exc
    if (
        not resolved.is_relative_to(Path("/usr/bin"))
        or not resolved.is_file()
        or resolved.stat().st_mode & 0o111 == 0
    ):
        raise ProvenanceError(f"required native build tool is not executable: {name}")
    return resolved


def _local_cuda() -> tuple[dict[str, Any], Path]:
    candidates = [Path("/usr/local/cuda")]
    candidates.extend(
        sorted(Path("/usr/local").glob("cuda-*"), reverse=True)
    )
    roots: list[Path] = []
    for candidate in candidates:
        try:
            root = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if root not in roots and (root / "bin" / "nvcc").is_file():
            roots.append(root)
    if not roots:
        raise ProvenanceError("no installed CUDA toolkit with nvcc was found")
    root = roots[0]
    output = _direct_read_only(
        [str(root / "bin" / "nvcc"), "--version"], timeout_seconds=15
    )
    match = re.search(r"\brelease\s+(13\.[0-9]+(?:\.[0-9]+)*)", output)
    if match is None:
        raise ProvenanceError("selected nvcc does not report a CUDA 13.x release")
    return {
        "status": "available",
        "version": match.group(1),
        "root": str(root),
        "nvcc_discovery": {"root": str(root), "nvcc": str(root / "bin" / "nvcc")},
    }, root


def _local_build_tools(cuda_root: Path) -> dict[str, Any]:
    tools = {
        "cmake": _plain_identity(_system_tool("cmake")),
        "g++": _plain_identity(_system_tool("g++")),
        "nvcc": _plain_identity((cuda_root / "bin" / "nvcc").resolve(strict=True)),
        "pkg-config": _plain_identity(_system_tool("pkg-config")),
    }
    for executable, generator in (("ninja", "Ninja"), ("make", "Unix Makefiles")):
        try:
            candidate = _system_tool(executable)
        except ProvenanceError:
            continue
        tools[executable] = {
            **_plain_identity(candidate),
            "name": generator,
            "executable": executable,
        }
        break
    else:
        raise ProvenanceError("neither Ninja nor Make is installed")
    return tools


def _local_build_prerequisites(pkg_config: str) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    unresolved: list[str] = []
    for module in _PKG_CONFIG_MODULES:
        try:
            version = _direct_read_only(
                [pkg_config, "--modversion", module], timeout_seconds=15
            ).strip()
        except ProvenanceError:
            version = ""
        ready = bool(version)
        modules[module] = {
            "status": "available" if ready else "missing",
            "version": version or None,
        }
        if not ready:
            unresolved.append(module)
    return {
        "status": "complete" if not unresolved else "incomplete",
        "unresolved_modules": unresolved,
        "consumer": "Samples/AppDecode/AppDec",
        "required_modules": list(_PKG_CONFIG_MODULES),
        "modules": modules,
    }


def _local_native_evidence(
    *, timeout_seconds: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the installed native package enough to plan its own routes."""
    package_output = _direct_read_only(
        [
            "/usr/bin/dpkg-query",
            "-W",
            "-f=${db:Status-Abbrev}\\t${Version}\\t${binary:Package}\\n",
            "nvidia-video-codec-sdk",
        ],
        timeout_seconds=timeout_seconds,
    )
    fields = package_output.rstrip("\n").split("\t")
    if len(fields) != 3 or fields[0] != "ii " or fields[2] != "nvidia-video-codec-sdk":
        raise ProvenanceError("native Video Codec SDK package is not installed by dpkg")
    _public_version(fields[1])
    verified = _direct_read_only(
        ["/usr/bin/dpkg", "--verify", "nvidia-video-codec-sdk"],
        timeout_seconds=timeout_seconds,
    )
    if verified.strip():
        raise ProvenanceError("dpkg --verify reported modified native SDK files")
    listing = _direct_read_only(
        ["/usr/bin/dpkg-query", "-L", "nvidia-video-codec-sdk"],
        timeout_seconds=timeout_seconds,
    )
    roots = sorted(
        {
            str(Path(line).parent.parent.resolve(strict=True))
            for line in listing.splitlines()
            if line.startswith("/") and line.endswith("/Samples/CMakeLists.txt")
        }
    )
    if len(roots) != 1:
        raise ProvenanceError(
            "native package does not expose exactly one canonical SDK sample root"
        )
    cuda, cuda_root = _local_cuda()
    tools = _local_build_tools(cuda_root)
    prerequisites = _local_build_prerequisites(tools["pkg-config"]["path"])
    native = {
        "status": "installed",
        "package": {
            "name": "nvidia-video-codec-sdk",
            "status": "installed",
            "version": fields[1],
        },
        "complete_roots": roots,
    }
    return native, cuda, tools, prerequisites


_LOCAL_PYNVC_PROBE = r"""
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import pathlib
import re
import sys

distribution = importlib.metadata.distribution("PyNvVideoCodec")
module = __import__("PyNvVideoCodec")
dist_version = str(distribution.version)
module_version = str(getattr(module, "__version__", "") or "")
if dist_version != "2.1.0" or module_version != dist_version:
    raise RuntimeError("distribution and imported module must both be PyNvVideoCodec 2.1.0")
prefix = pathlib.Path(sys.prefix).resolve(strict=True)
launcher = pathlib.Path(sys.executable)
if launcher.parent.parent.resolve(strict=True) != prefix:
    raise RuntimeError("selected interpreter does not belong to its reported venv prefix")
files = [str(item).replace("\\", "/") for item in (distribution.files or [])]
paths = {
    item: pathlib.Path(distribution.locate_file(pathlib.Path(item))).resolve(strict=True)
    for item in files
}
records = [item for item in files if item.endswith(".dist-info/RECORD")]
if len(records) != 1:
    raise RuntimeError("expected exactly one PyNvVideoCodec RECORD")
record_path = paths[records[0]]
rows = {}
for row in csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))):
    if len(row) != 3 or not row[0] or row[0] in rows:
        raise RuntimeError("wheel RECORD is not canonical")
    rows[row[0]] = (row[1], row[2])

native_extension_member = re.compile(
    r"PyNvVideoCodec/_?PyNvVideoCodec(?:_(?:121|130))?\.[^/]*\.so"
)

def owned_identity(value, label, allow_native_stale=False):
    lexical = pathlib.Path(value)
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(prefix):
        raise RuntimeError(f"{label} is outside selected venv")
    matches = [name for name, path in paths.items() if path == resolved]
    if len(matches) != 1 or matches[0] not in rows:
        raise RuntimeError(f"{label} is not one unique wheel member")
    encoded, size_text = rows[matches[0]]
    payload = resolved.read_bytes()
    actual = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    matches_record = (
        encoded.startswith("sha256=")
        and encoded[7:] == actual
        and size_text == str(len(payload))
    )
    known_stale = (
        allow_native_stale
        and dist_version == "2.1.0"
        and native_extension_member.fullmatch(matches[0]) is not None
        and encoded.startswith("sha256=")
        and size_text.isdecimal()
    )
    if not matches_record and not known_stale:
        raise RuntimeError(f"{label} differs from wheel RECORD")
    return {
        "path": str(resolved), "loaded_path": str(resolved),
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "record_member": matches[0],
        "record_consistency": (
            "matches_record" if matches_record else "known_upstream_stale_record"
        ),
    }

module_path = getattr(module, "__file__", None)
extension_path = getattr(getattr(module, "_PyNvVideoCodec", None), "__file__", None)
if not module_path or not extension_path:
    raise RuntimeError("imported module or executed extension path is unavailable")
module_identity = owned_identity(module_path, "imported module")
module_identity["version"] = module_version
extension_identity = owned_identity(
    extension_path, "executed extension", allow_native_stale=True
)
suffix = str(getattr(module, "module_suffix", "") or "")
if suffix not in {"_121", "_130"}:
    raise RuntimeError("executed extension API suffix is not recognized")
extension_identity["module_suffix"] = suffix
packages = {}
for name in ("numpy", "pycuda", "torch"):
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    ready = version is not None
    packages[name] = {
        "status": "installed" if ready else "missing",
        "version": version,
        "requirement_satisfied": ready,
    }
if packages["torch"]["status"] == "installed":
    try:
        import torch
        cuda_build = getattr(torch.version, "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_build = None
        cuda_available = False
    torch_ready = (
        packages["torch"]["version"] == "2.9.1+cu130"
        and cuda_build == "13.0"
        and cuda_available
    )
    packages["torch"].update({
        "cuda_build": cuda_build,
        "cuda_available": cuda_available,
        "requirement_satisfied": torch_ready,
    })
target = launcher.resolve(strict=True)
target_payload = target.read_bytes()
interpreter_identity = {
    "path": str(target), "size_bytes": len(target_payload),
    "sha256": hashlib.sha256(target_payload).hexdigest(),
}
print(json.dumps({
    "pynvc": {
        "imported": True,
        "distribution_version": dist_version,
        "identity": {
            "status": "verified",
            "interpreter": str(launcher),
            "sys_prefix": str(prefix),
            "module": module_identity,
            "extension": extension_identity,
        },
    },
    "python": {
        "executable": str(launcher),
        "sys_prefix": str(prefix),
        "interpreter_identities": {"running": interpreter_identity},
        "packages": packages,
    },
}, sort_keys=True))
""".strip()


def _local_pynvc_evidence(
    interpreter: str, *, timeout_seconds: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(interpreter, str)
        or not interpreter
        or interpreter != interpreter.strip()
        or not Path(interpreter).is_absolute()
        or str(Path(interpreter)) != interpreter
    ):
        raise ProvenanceError(
            "pynvc_interpreter must be one canonical absolute path"
        )
    launcher = Path(interpreter)
    try:
        target = launcher.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError("pynvc_interpreter is unavailable") from exc
    if not launcher.is_file() or target.stat().st_mode & 0o111 == 0:
        raise ProvenanceError("pynvc_interpreter is not executable")
    output = _direct_read_only(
        [str(launcher), "-I", "-c", _LOCAL_PYNVC_PROBE],
        timeout_seconds=timeout_seconds,
    )
    try:
        value = json.loads(output)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("local PyNvVideoCodec probe returned invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"pynvc", "python"}
        or not isinstance(value.get("pynvc"), dict)
        or not isinstance(value.get("python"), dict)
    ):
        raise ProvenanceError("local PyNvVideoCodec probe returned an invalid shape")
    return dict(value["pynvc"]), dict(value["python"])


def build_local_runtime_binding(
    required_surfaces: Sequence[str],
    *,
    pynvc_interpreter: str | None,
    gpu: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build an in-process-only read-only authority for requested surfaces.

    The returned document is private evidence staged by the consumer. It is
    never accepted from a request and never substitutes for a supplied setup
    artifact. Each requested surface is probed independently.
    """
    selected = tuple(required_surfaces)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(surface not in artifact_io.SURFACE_ORDER for surface in selected)
    ):
        raise ProvenanceError("local binding surfaces are invalid")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ProvenanceError("local binding gpu must be a non-negative integer")
    errors: dict[str, str] = {}
    native: dict[str, Any] = {
        "status": "missing",
        "package": {
            "name": "nvidia-video-codec-sdk",
            "status": "missing",
            "version": None,
        },
        "complete_roots": [],
    }
    cuda: dict[str, Any] = {
        "status": "absent",
        "version": None,
        "nvcc_discovery": {"root": None, "nvcc": None},
    }
    tools: dict[str, Any] = {}
    prerequisites: dict[str, Any] = {
        "status": "incomplete",
        "unresolved_modules": list(_PKG_CONFIG_MODULES),
    }
    pynvc: dict[str, Any] = {
        "imported": False,
        "distribution_version": None,
        "identity": {"status": "not_ready"},
    }
    python: dict[str, Any] = {
        "interpreter_identities": {},
        "packages": {},
    }
    if "native" in selected:
        try:
            native, cuda, tools, prerequisites = _local_native_evidence(
                timeout_seconds=timeout_seconds
            )
        except (OSError, ValueError) as exc:
            errors["native"] = str(exc)
    if "pynvc" in selected:
        if pynvc_interpreter is None:
            errors["pynvc"] = (
                "PyNvVideoCodec was not evaluated because no exact absolute "
                "pynvc_interpreter was supplied"
            )
        else:
            try:
                pynvc, python = _local_pynvc_evidence(
                    pynvc_interpreter, timeout_seconds=timeout_seconds
                )
            except (OSError, ValueError) as exc:
                errors["pynvc"] = str(exc)
    runtime = (
        "both" if len(selected) == 2 else selected[0]
    )
    binding = {
        "schema_version": artifact_io.LOCAL_RUNTIME_BINDING_SCHEMA_VERSION,
        "kind": artifact_io.LOCAL_RUNTIME_BINDING_KIND,
        "mode": "live",
        "source": "pipeline-local-discovery",
        "requested_runtime": runtime,
        "selected_gpu": gpu,
        "installation": {
            "native_sdk": native,
            "cuda_toolkit": cuda,
            "native_build_prerequisites": prerequisites,
            "build_tool_identities": tools,
            "python": python,
        },
        "pynvc": pynvc,
    }
    artifact_io.validate_local_runtime_binding(
        binding,
        label="local runtime",
        error_type=ProvenanceError,
        required_surfaces=selected,
    )
    return binding, errors


def _environment_value(
    workspace: Path, identity: Mapping[str, Any], label: str
) -> dict[str, Any]:
    required = ("pynvc",) if label.lower().startswith("py") else ("native",)
    return artifact_io.read_live_environment(
        workspace,
        identity,
        label=label,
        error_type=ProvenanceError,
        required_surfaces=required,
    )


def _run_prepare(
    service: Any,
    commands: list[dict[str, Any]],
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stage: str,
) -> tuple[dict[str, Any], str, str]:
    return artifact_io.run_prepare_command(
        service,
        commands,
        argv,
        command_environment=_COMMAND_ENV,
        error_type=ProvenanceError,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage=stage,
    )


def _public_version(value: Any) -> str:
    match = _DEBIAN_VERSION.fullmatch(
        value.strip() if isinstance(value, str) else ""
    )
    if match is None or "~" in str(value) or "really" in str(value).lower():
        raise ProvenanceError(
            "native SDK package is not a stable public 13.0.x release"
        )
    return match.group("upstream")


def _native_surface(environment: Mapping[str, Any]) -> tuple[str, str]:
    """Read the required native subset of one schema-1.2 environment.

    The normalized surface carries one canonical ``sdk_root`` string, taken
    from the single complete root 1.2 publishes. Ownership of that root is
    re-authenticated live in ``_authenticated_sdk_root`` against ``dpkg``
    rather than replayed from the artifact's inventory.
    """
    native = artifact_io.environment_surface(environment, "native")
    package = native.get("package") if native is not None else None
    sdk_root = native.get("sdk_root") if native is not None else None
    if (
        native is None
        or not isinstance(package, Mapping)
        or package.get("name") != "nvidia-video-codec-sdk"
        or package.get("status") != "installed"
        or not isinstance(sdk_root, str)
        or not Path(sdk_root).is_absolute()
    ):
        raise ProvenanceError(
            "environment does not publish one installed native SDK package and root"
        )
    _public_version(package.get("version"))
    return sdk_root, str(package["version"])


def _tool_identity(
    tools: Mapping[str, Any], name: str
) -> dict[str, Any]:
    record = tools.get(name)
    if not isinstance(record, Mapping):
        raise ProvenanceError(f"environment did not publish required build tool {name}")
    path = record.get("path")
    sha256 = record.get("sha256")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise ProvenanceError(
            f"environment has no canonical path and SHA-256 for build tool {name}"
        )
    # The returned value stays an exact artifact identity: it is sealed and
    # re-verified before every launch, so it carries no extra keys.
    observed = _identity(Path(path), f"native-build-tool-{name}")
    # Schema 1.2 binds a build tool by path, size and content hash; every field
    # it published must still match, and it publishes no tool version.
    if any(
        record.get(field) not in (None, observed[field])
        for field in ("path", "size_bytes", "sha256")
    ):
        raise ProvenanceError(f"build tool {name} changed after the environment probe")
    return observed


def _native_toolchain(
    environment: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str, Path, Path]:
    """Re-hash the published build toolchain and resolve the CUDA root live."""
    native = artifact_io.environment_surface(environment, "native") or {}
    records = native.get("tools")
    if not isinstance(records, Mapping):
        raise ProvenanceError("environment publishes no native build tools")
    generator_record = records.get("generator")
    generator = (
        generator_record.get("name") if isinstance(generator_record, Mapping) else None
    )
    if generator not in _CMAKE_GENERATORS:
        raise ProvenanceError(
            f"environment generator must be one of {sorted(_CMAKE_GENERATORS)}"
        )
    tools = {name: _tool_identity(records, name) for name in _BUILD_TOOLS}
    cuda = native.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("status") != "installed":
        raise ProvenanceError("environment does not report an installed CUDA toolkit")
    cuda_version = cuda.get("version")
    if (
        not isinstance(cuda_version, str)
        or not cuda_version
        or not cuda_version.startswith("13.")
    ):
        raise ProvenanceError("environment does not report a required CUDA 13.x version")
    cuda_root_value = cuda.get("root")
    if not isinstance(cuda_root_value, str):
        raise ProvenanceError("environment did not record the selected CUDA root")
    cuda_root = Path(cuda_root_value).resolve(strict=True)
    if not Path(tools["nvcc"]["path"]).is_relative_to(cuda_root):
        raise ProvenanceError("environment nvcc is outside the selected CUDA root")
    marker = next(
        (
            path
            for path in (cuda_root / "version.json", cuda_root / "version.txt")
            if path.is_file()
        ),
        None,
    )
    if marker is None:
        raise ProvenanceError("selected CUDA root has no version marker")
    return tools, str(generator), cuda_root, marker


def _package_authority(
    service: Any,
    commands: list[dict[str, Any]],
    *,
    cwd: Path,
    timeout_seconds: float,
    expected_version: str,
) -> tuple[set[Path], set[Path]]:
    """Re-authenticate live package ownership and payload integrity."""
    _, output, _ = _run_prepare(
        service,
        commands,
        [
            "/usr/bin/dpkg-query",
            "-W",
            "-f=${db:Status-Abbrev}\\t${Version}\\t${binary:Package}\\n",
            "nvidia-video-codec-sdk",
        ],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-version",
    )
    if output.rstrip("\n").split("\t") != [
        "ii ",
        expected_version,
        "nvidia-video-codec-sdk",
    ]:
        raise ProvenanceError("live dpkg package identity differs from the environment")
    _, listing, _ = _run_prepare(
        service,
        commands,
        ["/usr/bin/dpkg-query", "-L", "nvidia-video-codec-sdk"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-files",
    )
    owned: set[Path] = set()
    directories: set[Path] = set()
    for line in listing.splitlines():
        if not line.startswith("/"):
            continue
        entry = Path(line)
        if entry.is_dir():
            directories.add(entry.resolve(strict=True))
        elif entry.is_file():
            owned.add(entry.resolve(strict=True))
    _, verified, _ = _run_prepare(
        service,
        commands,
        ["/usr/bin/dpkg", "--verify", "nvidia-video-codec-sdk"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage="native-package-verify",
    )
    if verified.strip():
        raise ProvenanceError("dpkg --verify reported modified native SDK files")
    return owned, directories


def _authenticated_sdk_root(declared: str, directories: set[Path]) -> Path:
    """Prove the published SDK root is a live package-owned directory."""
    try:
        root = Path(declared).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(
            f"published native SDK root is unavailable: {declared}"
        ) from exc
    if not root.is_dir() or root not in directories:
        raise ProvenanceError(
            "published native SDK root is not a live package-owned directory"
        )
    return root


def _cache_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        name_type, value = line.split("=", 1)
        name, _field_type = name_type.split(":", 1)
        values[name] = value
    return values


def _compile_sources(path: Path) -> set[Path]:
    value = artifact_io.strict_json_loads(path.read_bytes())
    if not isinstance(value, list) or not value:
        raise ProvenanceError("CMake compile_commands.json must be a non-empty array")
    if any(
        not isinstance(entry, dict) or not isinstance(entry.get("file"), str)
        for entry in value
    ):
        raise ProvenanceError("compile_commands.json entry has no source file")
    return {Path(entry["file"]).resolve(strict=True) for entry in value}


def _runtime_libraries(
    output: str, required: Sequence[str]
) -> list[dict[str, Any]]:
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
        raise ProvenanceError(
            f"ldd did not resolve required codec libraries: {missing}"
        )
    return [
        _identity(resolved[name], f"native-runtime-library-{name}")
        for name in required
    ]


def _prepare_native(  # pylint: disable=too-many-arguments,too-many-locals
    sample: str,
    *,
    sdk_root: Path,
    owned: set[Path],
    environment: dict[str, Any],
    environment_identity: Mapping[str, Any],
    environment_workspace: Path,
    workspace: Path,
    service: Any,
    commands: list[dict[str, Any]],
    timeout_seconds: float,
) -> tuple[AuthenticatedSample, dict[str, Any]]:
    binary_suffix, source_suffix, cmake_suffix, libraries = NATIVE_ROUTES[sample]
    source = (sdk_root / source_suffix).resolve(strict=True)
    cmake_file = (sdk_root / cmake_suffix).resolve(strict=True)
    top_cmake = (sdk_root / "Samples/CMakeLists.txt").resolve(strict=True)
    if not {source, cmake_file, top_cmake} <= owned:
        raise ProvenanceError(f"{sample} source/CMake files are not package-owned")
    tools, generator, cuda_root, marker = _native_toolchain(environment)
    build = artifact_io.create_private_workspace(
        workspace / f"native-build-{sample}"
    )
    configure = [
        tools["cmake"]["path"],
        "-S",
        str(sdk_root / "Samples"),
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
    _run_prepare(
        service,
        commands,
        configure,
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"configure-{sample.lower()}",
    )
    _run_prepare(
        service,
        commands,
        [
            tools["cmake"]["path"],
            "--build",
            str(build),
            "--target",
            sample,
            "--parallel",
            "1",
        ],
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"build-{sample.lower()}",
    )
    cache_path = build / "CMakeCache.txt"
    compile_path = build / "compile_commands.json"
    expected_cache = {
        "CMAKE_HOME_DIRECTORY": str(sdk_root / "Samples"),
        "CMAKE_CXX_COMPILER": tools["cxx"]["path"],
        "CMAKE_CUDA_COMPILER": tools["nvcc"]["path"],
        "CMAKE_MAKE_PROGRAM": tools["generator"]["path"],
        "CMAKE_GENERATOR": generator,
    }
    cache = _cache_values(cache_path)
    for name, expected in expected_cache.items():
        observed = cache.get(name)
        differs = (
            observed != expected
            if name == "CMAKE_GENERATOR"
            else not observed or Path(observed).resolve() != Path(expected).resolve()
        )
        if differs:
            raise ProvenanceError(f"CMake cache disagrees for {name}")
    compiled = _compile_sources(compile_path)
    if source not in compiled or not compiled <= owned:
        raise ProvenanceError(
            f"{sample} compile target is not bound to package-owned sources"
        )
    executable = (build / binary_suffix).resolve(strict=True)
    if not stat.S_ISREG(executable.stat().st_mode) or executable.stat().st_mode & 0o111 == 0:
        raise ProvenanceError(f"built {sample} output is not executable")
    _, linkage, _ = _run_prepare(
        service,
        commands,
        ["/usr/bin/ldd", str(executable)],
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stage=f"linkage-{sample.lower()}",
    )
    runtime = _runtime_libraries(linkage, libraries)
    protected = [
        _identity(executable, "native-official-sample-executable"),
        *(_identity(path, "native-package-source") for path in sorted(compiled)),
        _identity(cache_path, "native-cmake-cache"),
        _identity(compile_path, "native-compile-commands"),
        _identity(marker, "cuda-root-version-marker"),
        *tools.values(),
    ]
    record = _sample_record(
        surface="native",
        sample=sample,
        launcher=(str(executable),),
        workspace=workspace,
        protected=protected,
        runtime=runtime,
        environment_workspace=environment_workspace,
        environment_identity=environment_identity,
    )
    return record, {
        "sample": sample,
        "launcher": list(record.launcher),
        "source": str(source),
        "configure_argv": configure,
        "compiled_sources": [str(path) for path in sorted(compiled)],
        "protected_files": protected,
        "runtime_libraries": runtime,
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
) -> NativeAuthentication:
    """Build and authenticate exact pipeline-owned native sample routes."""
    root = artifact_io.resolve_private_workspace(workspace)
    env_root = artifact_io.resolve_private_workspace(environment_workspace)
    requested = tuple(samples)
    if not requested or len(set(requested)) != len(requested):
        raise ProvenanceError("native sample list must be non-empty and unique")
    unknown = sorted(set(requested) - set(NATIVE_ROUTES))
    if unknown:
        raise ProvenanceError(f"native sample is not allowlisted: {unknown}")
    destination = report_path or root / "native-sample-provenance.json"
    commands: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    records: list[AuthenticatedSample] = []
    service = runner or command_runner.CommandRunner(root)
    report: dict[str, Any] = {
        "schema_version": "1",
        "kind": "native-sample-provenance",
        "status": "failed",
        "environment": dict(environment_identity),
        "environment_workspace": str(env_root),
        "requested_samples": list(requested),
        "samples": evidence,
        "commands": commands,
    }
    try:
        environment = _environment_value(env_root, environment_identity, "native")
        declared_root, package_version = _native_surface(environment)
        owned, directories = _package_authority(
            service,
            commands,
            cwd=root,
            timeout_seconds=timeout_seconds,
            expected_version=package_version,
        )
        sdk_root = _authenticated_sdk_root(declared_root, directories)
        for sample in requested:
            record, item = _prepare_native(
                sample,
                sdk_root=sdk_root,
                owned=owned,
                environment=environment,
                environment_identity=environment_identity,
                environment_workspace=env_root,
                workspace=root,
                service=service,
                commands=commands,
                timeout_seconds=timeout_seconds,
            )
            records.append(record)
            evidence.append(item)
        report["status"] = "authenticated"
        report["package"] = {
            "name": "nvidia-video-codec-sdk",
            "version": package_version,
            "root": str(sdk_root),
            "root_authentication": {
                "declared_root": declared_root,
                "method": "live-dpkg-ownership",
                "package_owned_directory": True,
                "payload_verified": True,
            },
        }
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        artifact_io.write_fresh_json(root, destination, report)
        raise
    return NativeAuthentication(
        tuple(records), artifact_io.write_fresh_json(root, destination, report)
    )


_PROBE_PROGRAM = r"""
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
    raise RuntimeError("expected exactly one PyNvVideoCodec RECORD entry")
root = pathlib.Path(dist.locate_file(pathlib.Path("."))).resolve()
paths = {
    item: str(pathlib.Path(dist.locate_file(pathlib.Path(item))).resolve())
    for item in files
}
print(json.dumps({
    "schema_version": "1", "kind": "pynvc-record-probe",
    "interpreter": str(pathlib.Path(sys.executable).resolve()),
    "sys_prefix": str(pathlib.Path(sys.prefix).resolve()), "name": name,
    "normalized_name": re.sub(r"[-_.]+", "-", name).lower(),
    "version": str(dist.version), "record_entry": records[0],
    "distribution_root": str(root), "paths": paths,
}, sort_keys=True))
""".strip()


def _probe_record(
    *,
    python: Path,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    root = artifact_io.resolve_private_workspace(workspace)
    launcher = Path(python)
    if not launcher.is_absolute():
        raise ProvenanceError("selected virtual-environment Python must be absolute")
    target = launcher.resolve(strict=True)
    if not target.is_file() or target.stat().st_mode & 0o111 == 0:
        raise ProvenanceError("selected virtual-environment Python is not executable")
    result = runner.run(
        [str(launcher), "-I", "-c", _PROBE_PROGRAM],
        cwd=root,
        env=_COMMAND_ENV,
        timeout_seconds=timeout_seconds,
        stage="pynvc-record-probe",
        phase="probe",
    )
    runner_root = artifact_io.resolve_private_workspace(Path(runner.workspace))
    stderr = artifact_io.read_verified_text(runner_root, result["stderr"])
    if (
        result.get("timed_out")
        or result.get("launch_error")
        or result.get("exit_code") != 0
    ):
        raise ProvenanceError(
            f"isolated PyNvVideoCodec RECORD probe failed: {stderr[-500:]}"
        )
    value = artifact_io.strict_json_loads(
        artifact_io.read_verified_text(runner_root, result["stdout"])
    )
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "1"
        or value.get("kind") != "pynvc-record-probe"
        or value.get("interpreter") != str(target)
        or re.sub(r"[-_.]+", "-", str(value.get("name", ""))).lower()
        != "pynvvideocodec"
        or value.get("normalized_name") != "pynvvideocodec"
        or value.get("version") != "2.1.0"
        or not isinstance(value.get("paths"), dict)
    ):
        raise ProvenanceError(
            "isolated probe did not select exact PyNvVideoCodec 2.1.0"
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


def _within_prefix(probe: Mapping[str, Any], path_value: Any, label: str) -> Path:
    prefix_value = probe.get("sys_prefix")
    if (
        not isinstance(prefix_value, str)
        or not Path(prefix_value).is_absolute()
        or not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
    ):
        raise ProvenanceError(f"{label} path is invalid")
    prefix = Path(prefix_value).resolve(strict=True)
    candidate = Path(path_value).resolve(strict=True)
    if not candidate.is_relative_to(prefix):
        raise ProvenanceError(f"{label} resolved outside selected venv")
    return candidate


def _record_rows(
    probe: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    paths = probe.get("paths")
    member = probe.get("record_entry")
    if not isinstance(paths, dict) or not isinstance(member, str):
        raise ProvenanceError("record probe shape is invalid")
    record_path = _within_prefix(probe, paths.get(member), "wheel RECORD")
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
        if (
            len(row) != 3
            or not row[0]
            or row[0] != row[0].replace("\\", "/")
            or row[0] in parsed
        ):
            raise ProvenanceError("wheel RECORD row is not unique and canonical")
        parsed[row[0]] = (row[1], row[2])
    return parsed, _identity(record_path, "pynvc-wheel-record")


def _record_members(
    probe: Mapping[str, Any], members: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    rows, _record_identity = _record_rows(probe)
    paths = probe["paths"]
    result = []
    for member in members:
        if member not in rows or member not in paths:
            raise ProvenanceError(f"wheel RECORD member is missing: {member!r}")
        encoded_hash, size_text = rows[member]
        if not encoded_hash.startswith("sha256=") or not size_text.isdecimal():
            raise ProvenanceError(f"wheel RECORD member has no SHA-256/size: {member}")
        path = _within_prefix(probe, paths[member], "wheel member")
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


def _pynvc_environment(
    workspace: Path, identity: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    """Resolve the schema-1.2 Py surface identity into a launchable venv.

    The normalized surface carries identity only: the exact lexical
    ``interpreter``, the resolved ``interpreter_identity``, ``sys_prefix``, the
    PyNv ``version`` and the ``extension``. Everything the artifact asserts is
    re-authenticated live from the isolated RECORD probe.
    """
    value = _environment_value(workspace, identity, "PyNVC")
    surface = artifact_io.environment_surface(value, "pynvc")
    if surface is None or surface.get("version") != "2.1.0":
        raise ProvenanceError(
            "environment lacks one installed PyNvVideoCodec 2.1.0 surface"
        )
    launcher_value = surface.get("interpreter")
    prefix = surface.get("sys_prefix")
    resolved = surface.get("interpreter_identity")
    interpreter = resolved.get("path") if isinstance(resolved, Mapping) else None
    if (
        not isinstance(launcher_value, str)
        or not Path(launcher_value).is_absolute()
        or not isinstance(prefix, str)
        or not isinstance(interpreter, str)
    ):
        raise ProvenanceError("environment lacks selected PyNvVideoCodec venv")
    launcher = Path(launcher_value)
    target = launcher.resolve(strict=True)
    if (
        not launcher.is_file()
        or launcher.stat().st_mode & 0o111 == 0
        or target != Path(interpreter).resolve(strict=True)
        or launcher.parent.parent.resolve(strict=True) != Path(prefix).resolve(strict=True)
    ):
        raise ProvenanceError("selected venv launcher or prefix is invalid")
    return value, launcher, target, surface


def _metadata_members(probe: Mapping[str, Any]) -> tuple[str, str, str]:
    record = probe.get("record_entry")
    if not isinstance(record, str) or not record.endswith(".dist-info/RECORD"):
        raise ProvenanceError("wheel RECORD member is invalid")
    root = record.removesuffix("RECORD")
    metadata, wheel = root + "METADATA", root + "WHEEL"
    paths = probe.get("paths")
    if not isinstance(paths, dict) or metadata not in paths or wheel not in paths:
        raise ProvenanceError("wheel METADATA/WHEEL members are missing")
    return metadata, wheel, record


def _module_member(surface: Mapping[str, Any], probe: Mapping[str, Any]) -> str:
    """Bind the imported module to the exact wheel RECORD member.

    The imported ``module.version`` must equal the distribution version, so an
    interpreter that imported some other build fails closed here rather than
    being inferred as correct from metadata alone.
    """
    module = surface.get("module")
    paths = probe.get("paths")
    if (
        not isinstance(module, Mapping)
        or not isinstance(paths, Mapping)
        or not isinstance(module.get("path"), str)
    ):
        raise ProvenanceError("environment module identity is invalid")
    if module.get("version") != surface.get("version"):
        raise ProvenanceError(
            "imported module version differs from the distribution version"
        )
    if module["version"] != probe.get("version"):
        raise ProvenanceError(
            "imported module version differs from the live distribution probe"
        )
    matches = [name for name, path in paths.items() if path == module["path"]]
    if matches != ["PyNvVideoCodec/__init__.py"]:
        raise ProvenanceError("package module is not an exact wheel RECORD member")
    return matches[0]


def _extension_identity(
    surface: Mapping[str, Any], probe: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Bind the authenticated extension to live bytes inside the venv prefix.

    Hashing a ``.so`` found through distribution metadata proves only that a
    matching file exists on disk, so the authenticated path is resolved inside
    the probe's own prefix and re-hashed here. ``loaded_path`` is additive: 1.2
    publishes none, and an environment that does carry one must name the very
    same file rather than a second build.
    """
    extension = surface.get("extension")
    if not isinstance(extension, Mapping):
        raise ProvenanceError(
            "environment publishes no PyNvVideoCodec extension identity"
        )
    path_value = extension.get("path")
    sha256 = extension.get("sha256")
    loaded_path = extension.get("loaded_path")
    if loaded_path is None:
        loaded_path = path_value
    if (
        not isinstance(path_value, str)
        or not isinstance(sha256, str)
        or not isinstance(loaded_path, str)
        or not path_value.endswith(".so")
    ):
        raise ProvenanceError(
            "environment publishes no PyNvVideoCodec extension identity"
        )
    if loaded_path != path_value:
        raise ProvenanceError(
            "the interpreter loaded a different extension than the authenticated one"
        )
    candidate = _within_prefix(probe, path_value, "PyNvVideoCodec extension")
    identity = _identity(candidate, "pynvc-extension")
    if identity["sha256"] != sha256:
        raise ProvenanceError(
            "installed PyNvVideoCodec extension differs from the environment"
        )
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
) -> PynvcAuthentication:
    """Authenticate exact pipeline-owned PyNvVideoCodec wheel samples."""
    root = artifact_io.resolve_private_workspace(workspace)
    env_root = artifact_io.resolve_private_workspace(environment_workspace)
    requested = tuple(samples)
    if not requested or len(set(requested)) != len(requested):
        raise ProvenanceError("PyNvVideoCodec sample list must be non-empty and unique")
    unknown = sorted(set(requested) - set(PYNVC_ROUTES))
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
        _value, launcher, target, surface = _pynvc_environment(
            env_root, environment_identity
        )
        service = runner or command_runner.CommandRunner(root)
        probe = _probe_record(
            python=launcher,
            workspace=root,
            runner=service,
            timeout_seconds=timeout_seconds,
        )
        if (
            Path(str(probe.get("sys_prefix"))).resolve(strict=True)
            != Path(str(surface.get("sys_prefix"))).resolve(strict=True)
        ):
            raise ProvenanceError(
                "isolated RECORD probe differs from environment identity"
            )
        metadata, wheel, record = _metadata_members(probe)
        module = _module_member(surface, probe)
        common = _record_members(probe, (module, metadata, wheel))
        _rows, record_identity = _record_rows(probe)
        extension_protected = _extension_identity(surface, probe)
        interpreter_identity = _identity(target, "pynvc-venv-python")
        published = surface.get("interpreter_identity")
        if (
            not isinstance(published, Mapping)
            or published.get("sha256") != interpreter_identity["sha256"]
        ):
            raise ProvenanceError(
                "selected venv interpreter differs from the environment identity"
            )
        records: list[AuthenticatedSample] = []
        for sample in requested:
            members = (sample, *PYNVC_ROUTES[sample])
            owned = _record_members(probe, members)
            protected = [
                interpreter_identity,
                record_identity,
                *(item["identity"] for item in common),
                *(item["identity"] for item in owned),
                *extension_protected,
            ]
            record_value = _sample_record(
                surface="pynvc",
                sample=sample,
                launcher=(str(launcher), "-I", probe["paths"][sample]),
                workspace=root,
                protected=protected,
                runtime=(),
                environment_workspace=env_root,
                environment_identity=environment_identity,
            )
            records.append(record_value)
            report["samples"].append(
                {
                    "sample": sample,
                    "launcher": list(record_value.launcher),
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
        report["record_member"] = record
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        artifact_io.write_fresh_json(root, destination, report)
        raise
    return PynvcAuthentication(
        tuple(records), artifact_io.write_fresh_json(root, destination, report)
    )


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
    """Rehash every seal immediately before launching one exact sample."""
    return artifact_io.run_authenticated_sample(
        token,
        arguments,
        token_type=AuthenticatedSample,
        error_type=ProvenanceError,
        command_environment=_COMMAND_ENV,
        workspace=workspace,
        runner=runner,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stage=stage,
        phase=phase,
    )
