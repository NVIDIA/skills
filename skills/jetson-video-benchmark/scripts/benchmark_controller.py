#!/usr/bin/env python3
"""Run compact, authenticated Video Codec SDK performance measurements."""

# This single domain owner covers throughput benchmark and camera-capacity routes.
# pylint: disable=too-many-lines

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import re
import stat
import sys
from decimal import Decimal, DecimalException
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

if not sys.flags.ignore_environment or not sys.flags.no_user_site:
    raise SystemExit("invoke this producer with isolated Python: python3 -I")

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

import benchmark_provenance as provenance  # noqa: E402  # pylint: disable=wrong-import-position
import benchmark_runtime as runtime  # noqa: E402  # pylint: disable=wrong-import-position

pynvc_sample_provenance = provenance
sample_provenance = provenance
artifact_io = runtime
command_runner = runtime
surface_router = runtime

# Controller functions bind one request to several explicit evidence fields.
# pylint: disable=too-many-arguments,too-many-locals

SURFACES = ("native", "pynvc")
# The frozen `nvcodec-environment` 1.2 contract. Consumers pin this exact
# version and reject anything else rather than reinterpreting a differently
# shaped artifact; additive optional keys are ignored.
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_SCHEMA_VERSION = "1.2"
# 1.2 requests exactly one runtime, as a string rather than a list.
ENVIRONMENT_RUNTIMES = ("pynvc", "native", "both")
LOCAL_BINDING_KIND = "nvcodec-local-runtime-binding"
LOCAL_BINDING_SCHEMA_VERSION = "1.0"
NATIVE_PACKAGE = "nvidia-video-codec-sdk"
PYNVC_REQUIRED_VERSION = "2.1.0"
# The normalized native toolchain carries exactly these five entries.
REQUIRED_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
ACCEPTED_MEDIA_INPUTS = ("local_path", "http_url", "https_url")
ENCODE_CODECS = frozenset({"h264", "hevc", "av1"})
DECODE_CODECS = frozenset({*ENCODE_CODECS, "vp9"})
IDENTITY_KEYS = frozenset(
    {"schema_version", "kind", "path", "size_bytes", "sha256"}
)
PIXEL_FORMATS = {
    "nv12": (3, 2),
    "yuv420p": (3, 2),
    "p010le": (3, 1),
    "yuv444p": (3, 1),
    "yuv444p16le": (6, 1),
    "nv16": (2, 1),
    "p210le": (4, 1),
    "bgra": (4, 1),
    "rgba": (4, 1),
}
CANONICAL_PIXEL_FORMATS = {
    "NV12": "nv12",
    "YUV420": "yuv420p",
    "P010": "p010le",
    "NV16": "nv16",
    "P210": "p210le",
    "YUV444": "yuv444p",
    "YUV444_16BIT": "yuv444p16le",
    "ARGB": "bgra",
    "ABGR": "rgba",
}
NATIVE_FORMATS = {
    "NV12": "nv12",
    "YUV420": "iyuv",
    "P010": "p010",
    "NV16": "nv16",
    "P210": "p210",
    "YUV444": "yuv444",
    "YUV444_16BIT": "yuv444p16",
    "ARGB": "bgra",
    "ABGR": "abgr",
}
# The released PyNvVideoCodec 2.1 encode_perf.py helper contract caps every
# performance encode at this many frames per worker (encode_parser.py
# PERF_MAX_FRAMES); the sample silently limits longer inputs.
PYNVC_PERF_MAX_FRAMES_PER_WORKER = 1000
_PYNVC_CAP_AUTHORITY = (
    "authenticated PyNvVideoCodec 2.1 encode_perf.py helper contract"
)
_PYNVC_DECODE_MP_OMISSION = (
    "authenticated PyNvVideoCodec 2.1 decode_perf.py does not report stream "
    "dimensions; megapixels per second is omitted"
)
_FAILURE_PATTERN = sample_provenance.OFFICIAL_SAMPLE_FAILURE_PATTERN
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_HELP_OPTION = re.compile(r"(?<![A-Za-z0-9_])(-[A-Za-z][A-Za-z0-9_-]*)")
_COMMAND_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
_NATIVE_CODECS = {
    "AVC/H.264": "h264",
    "H.265/HEVC": "hevc",
    "AV1": "av1",
    "VP9": "vp9",
}
_NATIVE_LAYOUTS = {
    ("YUV 420", 8): "NV12",
    ("YUV 420", 10): "P010",
    ("YUV 422", 8): "NV16",
    ("YUV 422", 10): "P210",
    ("YUV 444", 8): "YUV444",
    ("YUV 444", 10): "YUV444_16BIT",
}
_RECIPE_CLI = (
    Path(__file__).absolute().parents[2]
    / "jetson-video-recipe"
    / "scripts"
    / "recipes"
    / "recipe_model.py"
)
_SYSTEM_PYTHON = Path("/usr/bin/python3")


class BenchmarkError(ValueError):
    """Raised when a benchmark request or its evidence is invalid."""


class DependencyRequired(BenchmarkError):
    """Raised when an independently installable public skill CLI is absent."""

    def __init__(self, dependency: Mapping[str, Any]) -> None:
        self.dependency = dict(dependency)
        super().__init__(str(self.dependency["reason"]))


def _installed_sibling(name: str) -> bool:
    """Return whether the lexical installed catalog contains one sibling."""
    candidate = Path(__file__).absolute().parents[2] / name / "SKILL.md"
    try:
        details = candidate.lstat()
        resolved_root = candidate.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        return False
    return (
        stat.S_ISREG(details.st_mode)
        and not stat.S_ISLNK(details.st_mode)
        and resolved_root.is_dir()
        and resolved == resolved_root / "SKILL.md"
    )


def _setup_dependency(*, needed_for: str, reason: str) -> dict[str, Any]:
    """Return one actionable setup handoff without making setup a prerequisite."""
    skill = "jetson-video-setup"
    installed = _installed_sibling(skill)
    action = (
        f"use {skill} for {needed_for} and retry this stage"
        if installed
        else f"install {skill}, use it for {needed_for}, and retry this stage"
    )
    return {
        "skill": skill,
        "installed": installed,
        "needed_for": needed_for,
        "reason": reason,
        "next_action": action,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{label} must be an object")
    return value


def _workspace(path: Path) -> Path:
    """Resolve an existing private workspace or create it after request preflight."""
    requested = Path(path).expanduser()
    if requested.exists():
        root = artifact_io.resolve_private_workspace(requested)
        if any(root.iterdir()):
            raise BenchmarkError("benchmark workspace must be fresh and empty")
        return root
    return artifact_io.create_private_workspace(requested)


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{label} must be a finite number > {minimum}")
    result = float(value)
    if not math.isfinite(result) or result <= minimum:
        raise BenchmarkError(f"{label} must be a finite number > {minimum}")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BenchmarkError(f"{label} must be a non-empty canonical string")
    return value


def _artifact(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    record = _mapping(value, label)
    identity = _mapping(record.get("identity"), f"{label}.identity")
    identity_copy = dict(identity)
    locator = identity_copy.get("path")
    if not isinstance(locator, str):
        raise BenchmarkError(f"{label}.identity.path must be a string")
    if Path(locator).is_absolute():
        path = artifact_io.verify_external_artifact(identity_copy)
    else:
        workspace_value = record.get("workspace")
        if not isinstance(workspace_value, str):
            raise BenchmarkError(
                f"{label}.workspace is required for a workspace-relative identity"
            )
        path = artifact_io.verify_artifact(Path(workspace_value), identity_copy)
    return path, identity_copy


def _external_identity(
    value: Any,
    label: str,
    *,
    schema_version: str,
    kinds: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Verify one exact portable identity and its current external bytes."""
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise BenchmarkError(f"{label} must be an exact portable artifact identity")
    identity = dict(value)
    if identity.get("schema_version") != schema_version:
        raise BenchmarkError(f"{label} must use schema {schema_version}")
    if identity.get("kind") not in kinds:
        raise BenchmarkError(f"{label} kind must be one of {list(kinds)}")
    try:
        path = artifact_io.verify_external_artifact(identity)
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"{label} identity is not current: {exc}") from exc
    return path, identity


def _environment(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _path, identity = _external_identity(
        request.get("environment"),
        "environment",
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        kinds=(ENVIRONMENT_KIND,),
    )
    try:
        value = artifact_io.read_verified_external_json(identity)
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"environment identity is not current: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("environment must contain one JSON object")
    version = value.get("schema_version")
    if version != ENVIRONMENT_SCHEMA_VERSION:
        raise BenchmarkError(
            f"environment schema_version must be exactly {ENVIRONMENT_SCHEMA_VERSION!r}; "
            f"refusing unknown environment version {version!r}"
        )
    # Required subset only: unknown additive keys are ignored by design.
    if (
        value.get("kind") != ENVIRONMENT_KIND
        or value.get("mode") != "live"
        or not isinstance(value.get("installation"), Mapping)
        or not isinstance(value.get("pynvc"), Mapping)
    ):
        raise BenchmarkError(
            f"environment must contain live {ENVIRONMENT_KIND} "
            f"{ENVIRONMENT_SCHEMA_VERSION} installation and pynvc facts"
        )
    if value.get("requested_runtime") not in ENVIRONMENT_RUNTIMES:
        raise BenchmarkError(
            "environment requested_runtime must be exactly one of "
            f"{list(ENVIRONMENT_RUNTIMES)}"
        )
    selected_gpu = value.get("selected_gpu")
    if (
        isinstance(selected_gpu, bool)
        or not isinstance(selected_gpu, int)
        or selected_gpu < 0
    ):
        raise BenchmarkError("environment selected_gpu must be a non-negative integer")
    return identity, value


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_absolute(value: Any) -> bool:
    return _is_text(value) and Path(str(value)).is_absolute()


def _is_version(value: Any) -> bool:
    """Accept one non-advisory version token."""
    return _is_text(value) and str(value)[0].isdigit()


def _numpy_requirement_satisfied(value: Any) -> bool:
    """Implement the released NumPy >=1.24 runtime floor without extra imports."""
    if not isinstance(value, str):
        return False
    match = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?"
        r"(?:\.post[0-9]+)?(?:\+[0-9A-Za-z.-]+)?",
        value,
    )
    return match is not None and tuple(int(part) for part in match.groups()[:2]) >= (1, 24)


def _is_hash_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_absolute(value.get("path"))
        and _is_text(value.get("sha256"))
    )


def _is_tool(record: Any, name: str) -> bool:
    """One normalized build tool: {path, sha256} (+ generator name).

    1.2 binds a build tool by path and content hash and publishes no tool
    version, so a version is honoured when present and never required.
    """
    return (
        _is_hash_identity(record)
        and (record.get("version") is None or _is_version(record.get("version")))
        and (name != "generator" or _is_text(record.get("name")))
    )


def _native_reasons(native: Mapping[str, Any]) -> list[str]:
    """Structurally validate the required native subset at planning time.

    Additive keys are ignored and no exact-set equality is used. A malformed
    native surface is blocked here rather than at execute-time authentication,
    so it can never consume a run slot or re-couple the two surfaces by timing.
    """
    reasons: list[str] = []
    if native.get("installed") is not True:
        reasons.append("live environment does not report the native SDK installed")
    package = native.get("package")
    if (
        not isinstance(package, Mapping)
        or package.get("name") != NATIVE_PACKAGE
        or package.get("status") != "installed"
        or not _is_version(package.get("version"))
    ):
        reasons.append(
            f"live environment does not report an installed {NATIVE_PACKAGE} package"
        )
    if not _is_absolute(native.get("sdk_root")):
        reasons.append("live environment carries no canonical native sdk_root")
    cuda = native.get("cuda")
    if (
        not isinstance(cuda, Mapping)
        or cuda.get("status") != "installed"
        or not _is_version(cuda.get("version"))
    ):
        reasons.append(
            "live environment does not report an installed CUDA toolkit version"
        )
    tools = native.get("tools")
    incomplete = sorted(
        name
        for name in REQUIRED_TOOLS
        if not _is_tool(tools.get(name) if isinstance(tools, Mapping) else None, name)
    )
    if incomplete:
        reasons.append(
            "live environment build tool identities are incomplete: "
            + ", ".join(incomplete)
        )
    return reasons


def _pynvc_encode_dependency_gaps(pynvc: Mapping[str, Any]) -> list[str]:
    """Return missing or incompatible dependencies for the official encode sample."""
    dependencies = pynvc.get("dependencies")
    dependencies = dependencies if isinstance(dependencies, Mapping) else {}
    incomplete = []
    for name in ("numpy", "pycuda", "torch"):
        record = dependencies.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("status") != "installed"
            or record.get("ready") is not True
        ):
            incomplete.append(name)
    torch = dependencies.get("torch")
    numpy = dependencies.get("numpy")
    if (
        isinstance(numpy, Mapping)
        and not _numpy_requirement_satisfied(numpy.get("version"))
        and "numpy" not in incomplete
    ):
        incomplete.append("numpy")
    pycuda = dependencies.get("pycuda")
    if (
        isinstance(pycuda, Mapping)
        and pycuda.get("version") != "2026.1"
        and "pycuda" not in incomplete
    ):
        incomplete.append("pycuda")
    if (
        isinstance(torch, Mapping)
        and (
            torch.get("version") != "2.9.1+cu130"
            or torch.get("cuda_build") != "13.0"
            or torch.get("cuda_available") is not True
        )
        and "torch" not in incomplete
    ):
        incomplete.append("torch")
    return sorted(incomplete)


def _pynvc_reasons(pynvc: Mapping[str, Any], operation: str) -> list[str]:
    """Structurally validate the required Py subset at planning time.

    Schema 1.2 authenticates the extension by wheel-member identity and does
    not publish which file the interpreter actually loaded, so
    `extension.loaded_path` is additive: when a document carries it, it must
    equal the authenticated `extension.path`, and it is never inferred when
    absent. The imported `module.version` must equal the distribution version;
    both comparisons fail closed.
    """
    reasons: list[str] = []
    version = pynvc.get("version")
    if (
        pynvc.get("installed") is not True
        or version != PYNVC_REQUIRED_VERSION
        or not _is_hash_identity(pynvc.get("interpreter_identity"))
        or not _is_absolute(pynvc.get("interpreter"))
        or not _is_absolute(pynvc.get("sys_prefix"))
    ):
        reasons.append(
            "live environment does not report installed PyNvVideoCodec "
            f"{PYNVC_REQUIRED_VERSION} with an interpreter identity"
        )
    extension = pynvc.get("extension")
    loaded_path = extension.get("loaded_path") if isinstance(extension, Mapping) else None
    if not _is_hash_identity(extension) or (
        loaded_path is not None
        and (not _is_absolute(loaded_path) or loaded_path != extension.get("path"))
    ):
        reasons.append(
            "live environment does not carry one authenticated PyNvVideoCodec "
            "extension identity"
        )
    module = pynvc.get("module")
    imported_version = module.get("version") if isinstance(module, Mapping) else None
    if (
        not isinstance(module, Mapping)
        or not _is_absolute(module.get("path"))
        or imported_version != version
    ):
        reasons.append(
            "live environment imported PyNvVideoCodec module version "
            f"{imported_version!r} differs from distribution version {version!r}"
        )
    if operation == "encode":
        incomplete = _pynvc_encode_dependency_gaps(pynvc)
        if incomplete:
            reasons.append(
                "Py encode_perf dependencies are incomplete: "
                + ", ".join(sorted(incomplete))
                + ". A jetson-video-setup pynvc-smoke environment deliberately carries no"
                " Torch. Remedy: provision a NEW full-samples PyNvVideoCodec venv through"
                " jetson-video-setup (plan_install.py --profile full-samples with a new"
                " --venv). A smoke venv is never installed into or upgraded in place."
            )
    return reasons


def _stage_environment(
    workspace: Path, environment: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Stage live facts only because official provenance consumes workspace identities."""
    root = artifact_io.create_private_workspace(workspace / "environment-facts")
    identity = artifact_io.write_fresh_json(
        root, root / "nvcodec-environment.json", dict(environment)
    )
    return root, identity


def _stage_local_binding(
    workspace: Path, binding: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Stage controller-owned local authority without claiming setup provenance."""
    root = artifact_io.create_private_workspace(workspace / "local-runtime-binding")
    identity = artifact_io.write_fresh_json(
        root, root / "runtime-binding.json", dict(binding)
    )
    return root, identity


def _runtime_authority(
    request: Mapping[str, Any],
    *,
    operation: str,
    selected_gpu: int,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Use supplied setup evidence or discover only explicitly selected local surfaces."""
    if "environment" in request:
        identity, value = _environment(request)
        environment_gpu = int(value["selected_gpu"])
        if environment_gpu != selected_gpu:
            raise BenchmarkError(
                f"request/recipe gpu {selected_gpu} differs from environment "
                f"selected_gpu {environment_gpu}"
            )
        return {
            "source": "setup_environment",
            "selected_gpu": environment_gpu,
            "value": value,
            "external_identity": identity,
            "diagnostics": {},
        }
    requested = request.get("surface", "auto")
    if not isinstance(requested, str) or requested not in (*SURFACES, "auto", "both"):
        raise BenchmarkError("surface must be native, pynvc, auto, or both")
    targets = [requested] if requested in SURFACES else list(SURFACES)
    surfaces: dict[str, Any] = {}
    diagnostics: dict[str, str] = {}
    evaluations: dict[str, dict[str, Any]] = {
        surface: {
            "status": (
                "pending" if surface in targets else "not_inspected"
            )
        }
        for surface in SURFACES
    }
    installation: dict[str, Any] = {
        "native_sdk": {},
        "native_build_prerequisites": {},
        "cuda_toolkit": {},
        "build_tool_identities": {},
        "python": {},
    }
    pynvc_evidence: dict[str, Any] = {}
    sample = "AppEncPerf" if operation == "encode" else "AppDecPerf"
    for surface in targets:
        try:
            if surface == "native":
                discovery = provenance.discover_local_native_surface(
                    workspace=workspace,
                    samples=(sample,),
                    runner=runner,
                    timeout_seconds=timeout_seconds,
                )
            else:
                interpreter = request.get("pynvc_interpreter")
                if interpreter is None:
                    reason = (
                        "pynvc_interpreter is required for local PyNvVideoCodec "
                        "benchmarking; provide one exact absolute interpreter path"
                    )
                    diagnostics[surface] = reason
                    evaluations[surface] = {
                        "status": "not_evaluated",
                        "reason": reason,
                        "next_action": "provide pynvc_interpreter and retry",
                    }
                    continue
                discovery = provenance.discover_local_pynvc_surface(
                    interpreter=interpreter,
                    workspace=workspace,
                    runner=runner,
                    timeout_seconds=timeout_seconds,
                )
            surfaces[surface] = dict(discovery["surface"])
            evidence = _mapping(discovery.get("evidence"), "local discovery evidence")
            if surface == "native":
                installation.update(dict(evidence))
            else:
                installation["python"] = dict(
                    _mapping(evidence.get("python"), "local Python evidence")
                )
                pynvc_evidence = dict(
                    _mapping(evidence.get("pynvc"), "local PyNvVideoCodec evidence")
                )
            evaluations[surface] = {"status": "eligible"}
        except (OSError, ValueError) as exc:
            diagnostics[surface] = f"{type(exc).__name__}: {exc}"
            evaluations[surface] = {
                "status": "failed",
                "reason": diagnostics[surface],
            }
    return {
        "source": "local_discovery",
        "selected_gpu": selected_gpu,
        "value": {
            "schema_version": LOCAL_BINDING_SCHEMA_VERSION,
            "kind": LOCAL_BINDING_KIND,
            "mode": "live",
            "source": "benchmark-local-discovery",
            "requested_runtime": (
                requested if requested in SURFACES else "both"
            ),
            "selected_gpu": selected_gpu,
            "installation": installation,
            "pynvc": pynvc_evidence,
        },
        "normalized_surfaces": surfaces,
        "diagnostics": diagnostics,
        "evaluations": evaluations,
    }


def _runtime_result_fields(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve legacy environment output and label local binding separately."""
    if authority.get("source") == "setup_environment":
        return {"environment": dict(authority["external_identity"])}
    _mapping(authority.get("value"), "local runtime binding")
    normalized = _mapping(
        authority.get("normalized_surfaces", {}), "normalized local surfaces"
    )
    result: dict[str, Any] = {
        "runtime_authority": {
            "source": "local_discovery",
            "selected_gpu": authority["selected_gpu"],
            "surfaces": sorted(normalized),
            "diagnostics": dict(authority.get("diagnostics", {})),
            "evaluations": dict(authority.get("evaluations", {})),
        }
    }
    identity = authority.get("staged_identity")
    if isinstance(identity, Mapping):
        result["runtime_authority"]["binding"] = dict(identity)
        result["runtime_authority"]["binding_workspace"] = str(
            authority["staged_workspace"]
        )
    return result


def _surface_plan(
    request: Mapping[str, Any],
    environment: Mapping[str, Any],
    operation: str,
    diagnostics: Mapping[str, str] | None = None,
    normalized_surfaces: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested = request.get("surface", "auto")
    if not isinstance(requested, str):
        raise BenchmarkError("surface must be a string")
    if requested not in (*SURFACES, "auto", "both"):
        raise BenchmarkError("surface must be native, pynvc, auto, or both")
    inspected = {requested} if requested in SURFACES else set(SURFACES)
    failures = diagnostics or {}
    eligibility: dict[str, dict[str, Any]] = {}
    for surface in SURFACES:
        if surface not in inspected:
            reasons = [f"{surface} was not inspected for explicit {requested} routing"]
        elif surface in failures:
            reasons = [failures[surface]]
        elif surface == "native":
            value = (
                normalized_surfaces.get(surface, {})
                if normalized_surfaces is not None
                else (runtime.environment_surface(environment, surface) or {})
            )
            reasons = _native_reasons(value if isinstance(value, Mapping) else {})
        else:
            value = (
                normalized_surfaces.get(surface, {})
                if normalized_surfaces is not None
                else (runtime.environment_surface(environment, surface) or {})
            )
            reasons = _pynvc_reasons(
                value if isinstance(value, Mapping) else {}, operation
            )
        eligibility[surface] = {"eligible": not reasons, "reasons": reasons}
    try:
        return surface_router.build_surface_plan(requested, eligibility)
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc


def _runtime_setup_dependencies(
    request: Mapping[str, Any],
    authority: Mapping[str, Any],
    environment: Mapping[str, Any],
    operation: str,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Describe only inspected, ineligible SDK surfaces as setup handoffs."""
    requested = request.get("surface", "auto")
    inspected = {requested} if requested in SURFACES else set(SURFACES)
    eligible = set(plan.get("eligible_surfaces", []))
    diagnostics = authority.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    evaluations = authority.get("evaluations", {})
    evaluations = evaluations if isinstance(evaluations, Mapping) else {}
    normalized = authority.get("normalized_surfaces", {})
    normalized = normalized if isinstance(normalized, Mapping) else {}
    dependencies: list[dict[str, Any]] = []
    for surface in SURFACES:
        if surface not in inspected or surface in eligible:
            continue
        evaluation = evaluations.get(surface)
        if (
            isinstance(evaluation, Mapping)
            and evaluation.get("status") == "not_evaluated"
        ):
            # An omitted exact Py interpreter is a selector request, not evidence
            # that installation or repair is required.
            continue
        if surface in diagnostics:
            reasons = [str(diagnostics[surface])]
            value: Mapping[str, Any] = {}
        else:
            candidate = (
                normalized.get(surface, {})
                if authority.get("source") == "local_discovery"
                else (runtime.environment_surface(environment, surface) or {})
            )
            value = candidate if isinstance(candidate, Mapping) else {}
            reasons = (
                _native_reasons(value)
                if surface == "native"
                else _pynvc_reasons(value, operation)
            )
        if not reasons:
            continue
        needed_for = f"{surface} SDK installation or repair"
        dependency = _setup_dependency(
            needed_for=needed_for,
            reason="; ".join(reasons),
        )
        dependency["surface"] = surface
        if (
            surface == "pynvc"
            and operation == "encode"
            and value.get("installed") is True
        ):
            gaps = _pynvc_encode_dependency_gaps(value)
            if gaps:
                dependency["required_profile"] = "full-samples"
                dependency["missing_dependencies"] = gaps
                dependency["needed_for"] = (
                    "PyNvVideoCodec full-samples environment provisioning"
                )
                dependency["next_action"] = (
                    "use jetson-video-setup to provision a new full-samples "
                    "PyNvVideoCodec environment, then retry this stage"
                    if dependency["installed"]
                    else "install jetson-video-setup, provision a new full-samples "
                    "PyNvVideoCodec environment, then retry this stage"
                )
        dependencies.append(dependency)
    return dependencies


def _dependency_result_fields(
    dependencies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose one convenient primary dependency plus the complete independent list."""
    if not dependencies:
        return {}
    values = [dict(value) for value in dependencies]
    return {"dependency": values[0], "dependencies": values}


def _mode(request: Mapping[str, Any]) -> str:
    value = request.get("mode", "execute")
    if value not in {"dry_run", "execute"}:
        raise BenchmarkError("mode must be exactly 'dry_run' or 'execute'")
    return str(value)


def _operation(request: Mapping[str, Any], route: str) -> str:
    default = route if route in {"encode", "decode"} else "encode"
    value = request.get("operation", default)
    if value not in {"encode", "decode"}:
        raise BenchmarkError("operation must be exactly 'encode' or 'decode'")
    return str(value)


def _process_model(request: Mapping[str, Any]) -> str:
    value = request.get("process_model", "thread")
    if value not in {"thread", "process"}:
        raise BenchmarkError("process_model must be exactly 'thread' or 'process'")
    return str(value)


def _input(request: Mapping[str, Any], operation: str) -> dict[str, Any]:
    field = "input" if operation == "encode" else "encoded_artifact"
    record = _mapping(request.get(field), field)
    path, identity = _artifact(record, field)
    metadata = _mapping(record.get("metadata"), f"{field}.metadata")
    width = _integer(metadata.get("width"), f"{field}.metadata.width")
    height = _integer(metadata.get("height"), f"{field}.metadata.height")
    frames = _integer(metadata.get("frames"), f"{field}.metadata.frames")
    fps = _number(metadata.get("fps"), f"{field}.metadata.fps")
    codec = _string(metadata.get("codec"), f"{field}.metadata.codec").lower()
    allowed_codecs = ENCODE_CODECS if operation == "encode" else DECODE_CODECS
    if codec not in allowed_codecs:
        raise BenchmarkError(f"unsupported codec for {operation}: {codec}")
    fmt = _string(metadata.get("format"), f"{field}.metadata.format").upper()
    if fmt not in NATIVE_FORMATS:
        raise BenchmarkError(f"unsupported raw input format: {fmt}")
    if "source_url" not in record:
        raise BenchmarkError(f"{field}.source_url is required")
    raw_source_url = record.get("source_url")
    source_url = (
        None
        if raw_source_url is None
        else _string(raw_source_url, f"{field}.source_url")
    )
    if source_url is not None:
        try:
            parsed = urlparse(source_url)
            port = parsed.port
            valid_source_url = bool(
                parsed.scheme in {"http", "https"}
                and parsed.netloc
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and "\\" not in source_url
                and not any(
                    character.isspace()
                    or ord(character) < 0x20
                    or ord(character) == 0x7F
                    for character in source_url
                )
                and (port is None or 1 <= port <= 65535)
            )
        except (UnicodeError, ValueError):
            valid_source_url = False
        if not valid_source_url:
            raise BenchmarkError(
                f"{field}.source_url must be null or one exact HTTP(S) URL"
            )
    license_value = _string(record.get("license"), f"{field}.license")
    attribution = _string(record.get("attribution"), f"{field}.attribution")
    if operation == "encode":
        factor = PIXEL_FORMATS.get(CANONICAL_PIXEL_FORMATS[fmt])
        if factor is None:
            raise BenchmarkError(f"unsupported raw input format: {fmt}")
        numerator, denominator = factor
        pixels = width * height
        if pixels * numerator % denominator:
            raise BenchmarkError("raw input geometry does not form whole frames")
        expected_size = pixels * numerator // denominator * frames
        if identity.get("size_bytes") != expected_size:
            raise BenchmarkError(
                f"raw input has {identity.get('size_bytes')} bytes; expected {expected_size}"
            )
    return {
        "path": str(path),
        "identity": identity,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "codec": codec,
        "format": fmt,
        "source_url": source_url,
        "license": license_value,
        "attribution": attribution,
    }


def _recipe(
    request: Mapping[str, Any],
    *,
    workspace: Path,
    timeout_seconds: float,
    validation_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _RECIPE_CLI.exists():
        skill = "jetson-video-recipe"
        needed_for = "encode recipe replay validation"
        raise DependencyRequired(
            {
                "skill": skill,
                "installed": False,
                "public_cli": str(_RECIPE_CLI),
                "needed_for": needed_for,
                "reason": (
                    f"I can run {needed_for}, but it requires the {skill} skill, "
                    f"which is not installed. Install {skill} and retry this stage."
                ),
                "next_action": (
                    f"install {skill} beside this skill and retry the unchanged "
                    "benchmark request"
                ),
            }
        )
    _recipe_path, identity = _external_identity(
        request.get("recipe"),
        "recipe",
        schema_version="2.0",
        kinds=("nvcodec-recipe",),
    )
    try:
        value = artifact_io.read_verified_external_json(identity)
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"recipe identity is not current: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("recipe identity must contain one schema-2 recipe object")
    try:
        recipe_root = _RECIPE_CLI.parents[2].resolve(strict=True)
        recipe_details = _RECIPE_CLI.lstat()
        recipe_cli = _RECIPE_CLI.resolve(strict=True)
        if stat.S_ISLNK(recipe_details.st_mode) or not stat.S_ISREG(recipe_details.st_mode):
            raise BenchmarkError("recipe public CLI must be a regular file")
        if recipe_cli != recipe_root / "scripts" / "recipes" / "recipe_model.py":
            raise BenchmarkError("recipe public CLI path is not canonical")
        python = _SYSTEM_PYTHON.resolve(strict=True)
        validation_workspace = artifact_io.create_private_workspace(
            workspace / f"recipe-validation-{validation_index:04d}"
        )
        validation_runner = command_runner.CommandRunner(validation_workspace)
        validation_identity = artifact_io.write_fresh_json(
            validation_workspace,
            validation_workspace / "recipe.json",
            value,
        )
        validation_recipe = artifact_io.verify_artifact(
            validation_workspace, validation_identity
        )
        result = validation_runner.run(
            [
                str(python),
                "-I",
                str(recipe_cli),
                "validate",
                "--recipe",
                str(validation_recipe),
            ],
            cwd=validation_workspace,
            env=_COMMAND_ENV,
            timeout_seconds=timeout_seconds,
            stage="validate-recipe",
            phase="verify",
        )
        stdout = artifact_io.read_verified_text(
            validation_workspace, result["stdout"]
        )
        stderr = artifact_io.read_verified_text(
            validation_workspace, result["stderr"]
        )
        if (
            result.get("timed_out")
            or result.get("launch_error")
            or result.get("exit_code") != 0
        ):
            detail = stderr.strip() or stdout.strip()
            raise BenchmarkError(
                f"recipe validator failed with exit {result.get('exit_code')}: "
                f"{detail[-500:]}"
            )
        outcome = artifact_io.strict_json_loads(stdout)
        if outcome != {"valid": True, "errors": []}:
            raise BenchmarkError("recipe validator returned an invalid success result")
        artifact_io.verify_artifact(validation_workspace, validation_identity)
    except (OSError, ValueError, TypeError) as exc:
        raise BenchmarkError(f"invalid benchmark recipe: {exc}") from exc
    try:
        artifact_io.verify_external_artifact(identity)
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"recipe identity is not current: {exc}") from exc
    return identity, value


def _bind_recipe_workload(
    data: Mapping[str, Any],
    recipe: Mapping[str, Any],
    request: Mapping[str, Any],
    environment_gpu: int,
) -> None:
    """Reject any request that would silently rebind a validated recipe."""
    intent = _mapping(recipe.get("encoder_intent"), "recipe.encoder_intent")
    for field in ("codec", "width", "height", "format", "fps"):
        if data[field] != intent.get(field):
            raise BenchmarkError(
                f"recipe {field} {intent.get(field)!r} differs from "
                f"input metadata {data[field]!r}"
            )
    recipe_frames = intent.get("frame_count")
    if recipe_frames is not None and recipe_frames != data["frames"]:
        raise BenchmarkError(
            f"recipe frame_count {recipe_frames!r} differs from "
            f"input metadata frames {data['frames']!r}"
        )
    recipe_gpu = _integer(intent.get("gpu"), "recipe.encoder_intent.gpu", minimum=0)
    requested_gpu = _integer(
        request.get("gpu", recipe_gpu),
        "gpu",
        minimum=0,
    )
    if requested_gpu != recipe_gpu:
        raise BenchmarkError(
            f"request gpu {requested_gpu} differs from recipe gpu {recipe_gpu}"
        )
    if recipe_gpu != environment_gpu:
        raise BenchmarkError(
            f"recipe gpu {recipe_gpu} differs from environment selected_gpu "
            f"{environment_gpu}"
        )


def _authentication(
    operation: str,
    selected: Sequence[str],
    *,
    environment_workspace: Path,
    environment_identity: Mapping[str, Any],
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    """Authenticate every selected leaf before any benchmark leaf launches."""
    authenticated: dict[str, dict[str, Any]] = {}
    if "native" in selected:
        sample = "AppEncPerf" if operation == "encode" else "AppDecPerf"
        try:
            value = sample_provenance.authenticate_native_samples(
                environment_workspace=environment_workspace,
                environment_identity=environment_identity,
                workspace=workspace,
                samples=(sample,),
                runner=runner,
                timeout_seconds=timeout_seconds,
                report_path=workspace / f"native-{operation}-provenance.json",
            )
            authenticated["native"] = {"status": "authenticated", "value": value}
        except (OSError, ValueError) as exc:
            authenticated["native"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    if "pynvc" in selected:
        sample = (
            "samples/advanced/encode_perf.py"
            if operation == "encode"
            else "samples/advanced/decode_perf.py"
        )
        try:
            value = pynvc_sample_provenance.authenticate_pynvc_samples(
                environment_workspace=environment_workspace,
                environment_identity=environment_identity,
                workspace=workspace,
                samples=(sample,),
                runner=runner,
                timeout_seconds=timeout_seconds,
                report_path=workspace / f"pynvc-{operation}-provenance.json",
            )
            authenticated["pynvc"] = {"status": "authenticated", "value": value}
        except (OSError, ValueError) as exc:
            authenticated["pynvc"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return authenticated


def _token(authentication: Any, operation: str, surface: str) -> Any:
    if surface == "native":
        return authentication.token("AppEncPerf" if operation == "encode" else "AppDecPerf")
    return authentication.token(
        "samples/advanced/encode_perf.py"
        if operation == "encode"
        else "samples/advanced/decode_perf.py"
    )


def _token_evidence(token: Any) -> dict[str, Any]:
    return {
        "surface": token.surface,
        "sample": token.sample,
        "launcher": list(token.launcher),
        "protected_files": [dict(item.identity) for item in token.protected_files],
        "runtime_libraries": [dict(item.identity) for item in token.runtime_libraries],
    }


def _run_sample(
    token: Any,
    arguments: Sequence[str],
    *,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
    stage: str,
    phase: str,
) -> dict[str, Any]:
    if "-loop" in arguments:
        raise BenchmarkError("benchmark repetition must never use -loop")
    try:
        return sample_provenance.run_authenticated(
            token,
            tuple(arguments),
            workspace=workspace,
            runner=runner,
            cwd=workspace,
            timeout_seconds=timeout_seconds,
            stage=stage,
            phase=phase,
        )
    except (ValueError, OSError) as exc:
        raise BenchmarkError(str(exc)) from exc


def _successful(
    result: Mapping[str, Any],
    label: str,
    *,
    complete_text: str | None = None,
) -> None:
    if (
        result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("launch_error") is not None
    ):
        raise BenchmarkError(f"{label} command did not complete successfully")
    combined = (
        complete_text
        if complete_text is not None
        else f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
    )
    if _FAILURE_PATTERN.search(combined):
        raise BenchmarkError(f"{label} command reported a failure marker")


def _command_text(workspace: Path, result: Mapping[str, Any]) -> str:
    """Read complete captured logs when command identities are available."""
    stdout = result.get("stdout")
    stderr = result.get("stderr")
    if stdout is None and stderr is None:
        return f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
    if not isinstance(stdout, Mapping) or not isinstance(stderr, Mapping):
        raise BenchmarkError("command result has incomplete log identities")
    try:
        return (
            artifact_io.read_verified_text(workspace, stdout)
            + "\n"
            + artifact_io.read_verified_text(workspace, stderr)
        )
    except ValueError as exc:
        raise BenchmarkError(f"command log identity is invalid: {exc}") from exc


def _native_help(
    token: Any,
    *,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
) -> tuple[set[str], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    advertised: set[str] = set()
    for option, name in (("-h", "h"), ("-A", "all")):
        result = _run_sample(
            token,
            (option,),
            workspace=workspace,
            runner=runner,
            timeout_seconds=timeout_seconds,
            stage=f"benchmark-help-{name}",
            phase="probe",
        )
        _successful(result, f"native help {option}")
        evidence.append(result)
        advertised.update(_HELP_OPTION.findall(_command_text(workspace, result)))
    return advertised, evidence


def _option_names(arguments: Sequence[str]) -> set[str]:
    return {
        token for token in arguments if token.startswith("-") and token != "-"
    }


def _projection_plan(
    operation: str,
    selected_surfaces: Sequence[str],
    recipe: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Separate exact recipe surfaces from independently blocked projections."""
    if operation == "decode":
        return list(selected_surfaces), {}
    if recipe is None:
        raise BenchmarkError("encode benchmark recipe is absent")
    projections = _mapping(recipe.get("projections"), "recipe.projections")
    losses = _mapping(recipe.get("projection_losses"), "recipe.projection_losses")
    selected: list[str] = []
    blocked: dict[str, dict[str, Any]] = {}
    for surface in selected_surfaces:
        projection = _mapping(projections.get(surface), f"recipe.projections.{surface}")
        if projection.get("status") == "exact":
            selected.append(surface)
            continue
        raw_losses = losses.get(surface)
        projection_losses = (
            [dict(_mapping(item, f"recipe.projection_losses.{surface}")) for item in raw_losses]
            if isinstance(raw_losses, list)
            else []
        )
        blocked[surface] = {
            "status": "blocked",
            "surface": surface,
            "reason": f"recipe has no exact {surface} projection",
            "projection_losses": projection_losses,
        }
    return selected, blocked


def _frame_plan(
    operation: str,
    data: Mapping[str, Any],
    selected_surfaces: Sequence[str],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the per-worker frame plan honoring the PyNv 2.1 perf encode cap."""
    source = int(data["frames"])
    capped = (
        operation == "encode"
        and "pynvc" in selected_surfaces
        and source > PYNVC_PERF_MAX_FRAMES_PER_WORKER
    )
    if capped and request.get("require_source_frames") is True:
        raise BenchmarkError(
            "request requires every source frame, but the "
            f"{_PYNVC_CAP_AUTHORITY} caps performance encode at "
            f"{PYNVC_PERF_MAX_FRAMES_PER_WORKER} frames per worker; use a "
            "native-only surface or an input at or below the limit"
        )
    return {
        "source_frames_per_worker": source,
        "effective_frames_per_worker": (
            PYNVC_PERF_MAX_FRAMES_PER_WORKER if capped else source
        ),
        "cap_applied": capped,
        "cap_limit_frames_per_worker": PYNVC_PERF_MAX_FRAMES_PER_WORKER,
        "cap_authority": _PYNVC_CAP_AUTHORITY,
    }


def _effective_data(
    data: Mapping[str, Any], frame_plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    effective = int(frame_plan["effective_frames_per_worker"])
    if effective == int(data["frames"]):
        return data
    return {**data, "frames": effective}


def _frame_accounting(
    frame_plan: Mapping[str, Any], workers: int
) -> dict[str, Any]:
    return {
        **frame_plan,
        "workers": workers,
        "expected_aggregate_frames": (
            int(frame_plan["effective_frames_per_worker"]) * workers
        ),
    }


def _native_arguments(
    operation: str,
    data: Mapping[str, Any],
    recipe: Mapping[str, Any] | None,
    request: Mapping[str, Any],
) -> list[str]:
    workers = _integer(request.get("workers", 1), "workers")
    if operation == "decode":
        gpu = _integer(request.get("gpu", 0), "gpu", minimum=0)
        arguments = ["-i", data["path"], "-gpu", str(gpu), "-thread", str(workers)]
        if request.get("single_context") is True:
            arguments.append("-single")
        if request.get("host_memory") is True:
            arguments.append("-host")
        return arguments
    assert recipe is not None
    intent = _mapping(recipe.get("encoder_intent"), "recipe.encoder_intent")
    gpu = _integer(intent.get("gpu"), "recipe.encoder_intent.gpu", minimum=0)
    projection = _mapping(recipe["projections"].get("native"), "recipe.projections.native")
    if projection.get("status") != "exact":
        raise BenchmarkError("recipe has no exact native projection")
    raw_options = projection.get("cli_options")
    if not isinstance(raw_options, list) or any(not isinstance(item, str) for item in raw_options):
        raise BenchmarkError("native recipe cli_options must be a list of strings")
    skip_with_value = {"-s", "-if", "-gpu", "-codec"}
    controls: list[str] = []
    index = 0
    while index < len(raw_options):
        option = raw_options[index]
        if not option.startswith("-"):
            raise BenchmarkError("native recipe projection contains a value without an option")
        has_value = index + 1 < len(raw_options) and not raw_options[index + 1].startswith("-")
        value = raw_options[index + 1] if has_value else None
        if option in skip_with_value and value is None:
            raise BenchmarkError(f"native recipe projection option {option} lacks its value")
        if option not in skip_with_value:
            controls.append(option)
            if value is not None:
                controls.append(value)
        index += 2 if has_value else 1
    fmt = NATIVE_FORMATS.get(str(data["format"]))
    if fmt is None:
        raise BenchmarkError(f"native AppEncPerf does not map format {data['format']}")
    return [
        "-i", data["path"], "-s", f"{data['width']}x{data['height']}",
        "-if", fmt, "-gpu", str(gpu), "-frame", str(data["frames"]),
        "-thread", str(workers), "-codec", str(data["codec"]), *controls,
    ]


def _pynvc_arguments(
    operation: str,
    data: Mapping[str, Any],
    recipe: Mapping[str, Any] | None,
    request: Mapping[str, Any],
    *,
    workspace: Path,
    variant: str,
) -> list[str]:
    workers = _integer(request.get("workers", 1), "workers")
    process_model = _process_model(request)
    if operation == "decode":
        gpu = _integer(request.get("gpu", 0), "gpu", minimum=0)
        return [
            "-i", data["path"], "-n", str(workers), "-m", str(process_model),
            "-g", str(gpu), "-f", str(data["frames"]),
        ]
    assert recipe is not None
    intent = _mapping(recipe.get("encoder_intent"), "recipe.encoder_intent")
    gpu = _integer(intent.get("gpu"), "recipe.encoder_intent.gpu", minimum=0)
    projection = _mapping(recipe["projections"].get("pynvc"), "recipe.projections.pynvc")
    if projection.get("status") != "exact":
        raise BenchmarkError("recipe has no exact pynvc projection")
    config = _mapping(projection.get("config"), "recipe.projections.pynvc.config")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", variant)[:48] or "benchmark"
    config_path = workspace / f"pynvc-{safe_name}-config.json"
    artifact_io.write_fresh_bytes(
        workspace,
        config_path,
        artifact_io.canonical_json_bytes(dict(config)),
        schema_version="1",
        kind="pynvc-encoder-config",
    )
    return [
        "-m", str(process_model), "-i", data["path"],
        "-s", f"{data['width']}x{data['height']}", "-if", str(data["format"]),
        "-n", str(workers), "-f", str(data["frames"]), "-g", str(gpu),
        "-c", str(data["codec"]), "-json", str(config_path),
    ]


def _pynvc_config_identity(
    arguments: Sequence[str], workspace: Path
) -> dict[str, Any]:
    """Bind the one generated encoder config to the exact -json operand."""
    if arguments.count("-json") != 1:
        raise BenchmarkError("PyNvVideoCodec encode argv must contain one -json option")
    index = arguments.index("-json")
    if index + 1 == len(arguments):
        raise BenchmarkError("PyNvVideoCodec encode -json option lacks its value")
    return artifact_io.snapshot_artifact(
        workspace,
        Path(arguments[index + 1]),
        schema_version="1",
        kind="pynvc-encoder-config",
    )


def _generated_config_result(
    workspace: Path, identity: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return config evidence only after its generated bytes remain current."""
    if identity is None:
        return {}
    artifact_io.verify_artifact(workspace, identity)
    return {"generated_config": dict(identity)}


def _values(pattern: str, text: str, label: str) -> list[float]:
    raw = re.findall(pattern, text, flags=re.MULTILINE)
    if not raw:
        raise BenchmarkError(f"{label} marker is absent")
    values = [float(item) for item in raw]
    if any(not math.isfinite(item) or item <= 0 for item in values):
        raise BenchmarkError(f"{label} marker is non-finite or non-positive")
    if any(not math.isclose(item, values[-1], rel_tol=1e-9, abs_tol=1e-9) for item in values[:-1]):
        raise BenchmarkError(f"{label} markers contradict one another")
    return values


def _quantized_value(
    pattern: str,
    text: str,
    label: str,
) -> tuple[Decimal, Decimal]:
    """Return a numeric marker and half of its displayed rounding interval."""
    raw = re.findall(pattern, text, flags=re.MULTILINE)
    if not raw:
        raise BenchmarkError(f"{label} marker is absent")
    try:
        values = [Decimal(item) for item in raw]
    except (DecimalException, ValueError) as exc:
        raise BenchmarkError(f"{label} marker is invalid") from exc
    if any(not item.is_finite() or item <= 0 for item in values):
        raise BenchmarkError(f"{label} marker is non-finite or non-positive")
    try:
        float_values = [float(item) for item in values]
    except (OverflowError, ValueError) as exc:
        raise BenchmarkError(f"{label} marker exceeds the supported numeric range") from exc
    if any(not math.isfinite(item) or item <= 0 for item in float_values):
        raise BenchmarkError(f"{label} marker exceeds the supported numeric range")
    if any(item != values[-1] for item in values[:-1]):
        raise BenchmarkError(f"{label} markers contradict one another")
    try:
        half_widths = [
            Decimal(1).scaleb(item.as_tuple().exponent) / 2
            for item in values
        ]
    except DecimalException as exc:
        raise BenchmarkError(f"{label} display precision is invalid") from exc
    if any(
        not item.is_finite() or item <= 0
        for item in half_widths
    ):
        raise BenchmarkError(f"{label} display precision is invalid")
    if any(item != half_widths[-1] for item in half_widths[:-1]):
        raise BenchmarkError(f"{label} display precisions contradict one another")
    return values[-1], half_widths[-1]


def _quantized_rate_matches(
    frames: int,
    elapsed: Decimal,
    elapsed_half_width: Decimal,
    fps: Decimal,
    fps_half_width: Decimal,
) -> bool:
    """Return whether rounded elapsed/FPS intervals can describe one rate."""
    elapsed_value = Fraction(elapsed)
    elapsed_width = Fraction(elapsed_half_width)
    fps_value = Fraction(fps)
    fps_width = Fraction(fps_half_width)
    elapsed_low = elapsed_value - elapsed_width
    elapsed_high = elapsed_value + elapsed_width
    reported_low = max(fps_value - fps_width, Fraction(0))
    reported_high = fps_value + fps_width
    # Compare the positive intervals without division. Fraction preserves every
    # displayed digit, independent of the process-global Decimal precision.
    return (
        reported_low * elapsed_low
        <= frames
        <= reported_high * elapsed_high
    )


def _frames(pattern: str, text: str, label: str) -> int:
    raw = re.findall(pattern, text, flags=re.MULTILINE)
    if not raw:
        raise BenchmarkError(f"{label} marker is absent")
    values = [int(item) for item in raw]
    if any(item <= 0 for item in values) or any(item != values[-1] for item in values[:-1]):
        raise BenchmarkError(f"{label} markers are invalid or contradictory")
    return values[-1]


def _native_decode_metadata(
    text: str,
    expected: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    """Bind AppDecPerf's repeated official input description to the artifact."""
    patterns = {
        "codec": r"^\s*Codec\s*:\s*([^\r\n]+?)\s*$",
        "frame_rate": r"^\s*Frame rate\s*:\s*([^\r\n]+?)\s*$",
        "coded_size": r"^\s*Coded size\s*:\s*\[(\d+),\s*(\d+)\]\s*$",
        "display_area": (
            r"^\s*Display area\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*$"
        ),
        "chroma": r"^\s*Chroma\s*:\s*([^\r\n]+?)\s*$",
        "bit_depth": r"^\s*Bit depth\s*:\s*(\d+)\s*$",
    }
    matches = {
        field: re.findall(pattern, text, flags=re.MULTILINE)
        for field, pattern in patterns.items()
    }
    for field, values in matches.items():
        if len(values) != workers:
            raise BenchmarkError(
                f"AppDecPerf expected {workers} {field} marker(s); found {len(values)}"
            )
        if any(value != values[-1] for value in values[:-1]):
            raise BenchmarkError(f"AppDecPerf {field} markers contradict one another")
    codec = _NATIVE_CODECS.get(str(matches["codec"][-1]).strip())
    coded_width, coded_height = (int(item) for item in matches["coded_size"][-1])
    left, top, right, bottom = (int(item) for item in matches["display_area"][-1])
    chroma = str(matches["chroma"][-1]).strip()
    bit_depth = int(matches["bit_depth"][-1])
    pixel_format = _NATIVE_LAYOUTS.get((chroma, bit_depth))
    rate = re.fullmatch(
        rf"({_NUMBER})\s*/\s*({_NUMBER})\s*=\s*({_NUMBER})\s+fps",
        str(matches["frame_rate"][-1]).strip(),
    )
    if rate is None:
        raise BenchmarkError("AppDecPerf frame-rate marker is malformed")
    numerator, denominator, reported_fps = (float(item) for item in rate.groups())
    if (
        denominator == 0
        or not all(math.isfinite(item) for item in (numerator, denominator, reported_fps))
        or not math.isclose(
            numerator / denominator, reported_fps, rel_tol=1e-4, abs_tol=1e-3
        )
    ):
        raise BenchmarkError("AppDecPerf frame-rate marker is contradictory")
    observed = {
        "codec": codec,
        "coded_width": coded_width,
        "coded_height": coded_height,
        "display_area": [left, top, right, bottom],
        "width": right - left,
        "height": bottom - top,
        "format": pixel_format,
        "bit_depth": bit_depth,
        "chroma": chroma,
        "fps": reported_fps,
        "instances": workers,
    }
    for field in ("codec", "width", "height", "format"):
        if observed[field] != expected[field]:
            raise BenchmarkError(
                f"AppDecPerf reported {field} {observed[field]}; expected {expected[field]}"
            )
    if not math.isclose(
        reported_fps, float(expected["fps"]), rel_tol=1e-4, abs_tol=1e-3
    ):
        raise BenchmarkError(
            f"AppDecPerf reported fps {reported_fps}; expected {expected['fps']}"
        )
    if not (0 <= left < right <= coded_width and 0 <= top < bottom <= coded_height):
        raise BenchmarkError("AppDecPerf reported an invalid display area")
    return observed


def _parse_performance(
    surface: str,
    operation: str,
    result: Mapping[str, Any],
    *,
    workspace: Path,
    expected_frames: int,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Four official samples have four source-defined terminal marker grammars.
    # pylint: disable=too-many-branches
    text = _command_text(workspace, result)
    _successful(
        result,
        f"{surface} {operation} benchmark",
        complete_text=text,
    )
    elapsed_decimal = None
    elapsed_half_width = None
    fps_decimal = None
    fps_half_width = None
    if surface == "native" and operation == "encode":
        matches = re.findall(
            rf"nTotal=(\d+),\s*time=({_NUMBER}) seconds,\s*FPS=({_NUMBER})", text
        )
        if not matches:
            raise BenchmarkError("AppEncPerf result marker is absent")
        parsed = [
            (int(frames), float(elapsed), float(fps))
            for frames, elapsed, fps in matches
        ]
        if any(item != parsed[-1] for item in parsed[:-1]):
            raise BenchmarkError("AppEncPerf result markers contradict one another")
        frames, elapsed, fps = parsed[-1]
    elif surface == "native":
        frames = _frames(
            r"Total Frames Decoded=(\d+)\s+FPS\s*=", text, "AppDecPerf frames"
        )
        fps = _values(
            rf"Total Frames Decoded=\d+\s+FPS\s*=\s*({_NUMBER})",
            text,
            "AppDecPerf FPS",
        )[-1]
        elapsed = None
    elif operation == "encode":
        frames = _frames(r"^Total frames processed:\s*(\d+)\s*$", text, "Py encode frames")
        elapsed_decimal, elapsed_half_width = _quantized_value(
            rf"^Duration:\s*({_NUMBER}) seconds\s*$",
            text,
            "Py encode duration",
        )
        fps_decimal, fps_half_width = _quantized_value(
            rf"^Total FPS:\s*({_NUMBER})\s*$",
            text,
            "Py encode FPS",
        )
        elapsed = float(elapsed_decimal)
        fps = float(fps_decimal)
    else:
        frames = _frames(r"^Total frames decoded:\s*(\d+)\s*$", text, "Py decode frames")
        fps = _values(rf"^Total FPS:\s*({_NUMBER})\s*$", text, "Py decode FPS")[-1]
        elapsed = _values(rf"^Total wall time:\s*({_NUMBER})s\s*$", text, "Py decode wall time")[-1]
    if frames != expected_frames:
        raise BenchmarkError(f"sample reported {frames} frames; expected {expected_frames}")
    if not math.isfinite(fps) or fps <= 0:
        raise BenchmarkError("sample FPS is non-finite or non-positive")
    if elapsed is not None and not (surface == "pynvc" and operation == "decode"):
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise BenchmarkError("sample elapsed time is non-finite or non-positive")
        derived = frames / elapsed
        if (
            elapsed_decimal is not None
            and elapsed_half_width is not None
            and fps_decimal is not None
            and fps_half_width is not None
        ):
            consistent = _quantized_rate_matches(
                frames,
                elapsed_decimal,
                elapsed_half_width,
                fps_decimal,
                fps_half_width,
            )
        else:
            consistent = math.isclose(
                fps,
                derived,
                rel_tol=0.05,
                abs_tol=0.1,
            )
        if not consistent:
            raise BenchmarkError(
                f"sample FPS {fps} contradicts frames/elapsed derived FPS {derived}"
            )
    parsed_result = {"frames": frames, "elapsed_seconds": elapsed, "fps": fps}
    if surface == "native" and operation == "decode":
        if expected_metadata is None:
            raise BenchmarkError("native decode metadata expectation is absent")
        parsed_result["input_media"] = _native_decode_metadata(
            text,
            expected_metadata,
            _integer(expected_frames // int(expected_metadata["frames"]), "workers"),
        )
    return parsed_result


def _summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fps = [float(run["fps"]) for run in runs]
    raw_megapixels = [run.get("megapixels_per_second") for run in runs]
    if not fps or any(not math.isfinite(item) or item <= 0 for item in fps):
        raise BenchmarkError("benchmark statistics contain a non-finite or non-positive metric")
    megapixels: list[float] | None
    omission_reason = None
    if raw_megapixels and all(item is None for item in raw_megapixels):
        reasons = {
            _mapping(run.get("metric_omissions"), "metric_omissions").get(
                "megapixels_per_second"
            )
            for run in runs
        }
        if len(reasons) != 1:
            raise BenchmarkError("benchmark repetitions contain contradictory metric omissions")
        omission_reason = _string(
            reasons.pop(),
            "metric_omissions.megapixels_per_second",
        )
        megapixels = None
    elif any(item is None for item in raw_megapixels):
        raise BenchmarkError("benchmark repetitions mix present and omitted MP/s metrics")
    else:
        megapixels = [float(item) for item in raw_megapixels]
        if any(not math.isfinite(item) or item <= 0 for item in megapixels):
            raise BenchmarkError(
                "benchmark statistics contain a non-finite or non-positive metric"
            )
    fps_scale = max(fps)
    summary = {
        "repetitions": len(runs),
        "fps": {
            "mean": (
                fps_scale
                * (math.fsum(item / fps_scale for item in fps) / len(fps))
            ),
            "minimum": min(fps),
            "maximum": fps_scale,
        },
    }
    if megapixels is None:
        summary["megapixels_per_second"] = None
        summary["metric_omissions"] = {
            "megapixels_per_second": omission_reason,
        }
    else:
        megapixels_scale = max(megapixels)
        summary["megapixels_per_second"] = {
            "mean": (
                megapixels_scale
                * (
                    math.fsum(item / megapixels_scale for item in megapixels)
                    / len(megapixels)
                )
            ),
            "minimum": min(megapixels),
            "maximum": megapixels_scale,
        }
    summary_values = list(summary["fps"].values())
    if megapixels is not None:
        summary_values += list(summary["megapixels_per_second"].values())
    if any(not math.isfinite(item) or item <= 0 for item in summary_values):
        raise BenchmarkError("benchmark statistics overflowed the supported numeric range")
    return summary


def _megapixels_per_second(fps: float, width: int, height: int) -> float:
    value = fps * width * height / 1_000_000
    if not math.isfinite(value) or value <= 0:
        raise BenchmarkError("megapixels per second exceeds the supported numeric range")
    return value


def _run_surface(
    surface: str,
    operation: str,
    request: Mapping[str, Any],
    *,
    variant: str,
    data: Mapping[str, Any],
    recipe: Mapping[str, Any] | None,
    authentication: Any,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
    frame_plan: Mapping[str, Any],
    native_help: tuple[set[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    token = _token(authentication, operation, surface)
    effective_data = _effective_data(data, frame_plan)
    if surface == "native":
        arguments = _native_arguments(operation, effective_data, recipe, request)
        if operation == "encode":
            if native_help is None:
                raise BenchmarkError("native benchmark help evidence is absent")
            advertised, help_evidence = native_help
            emitted = _option_names(arguments)
            if "-loop" in emitted:
                raise BenchmarkError("AppEncPerf argv must never contain -loop")
            missing = sorted(emitted - advertised)
            if missing:
                raise BenchmarkError(
                    f"installed native sample help does not advertise {missing}"
                )
        else:
            advertised = set()
            help_evidence = []
    else:
        help_evidence = []
        advertised = set()
        arguments = _pynvc_arguments(
            operation, effective_data, recipe, request, workspace=workspace, variant=variant
        )
    generated_config = (
        _pynvc_config_identity(arguments, workspace)
        if surface == "pynvc" and operation == "encode"
        else None
    )
    workers = _integer(request.get("workers", 1), "workers")
    expected_frames = int(effective_data["frames"]) * workers
    repetitions = _integer(
        request.get("measured_repetitions", 3),
        "measured_repetitions",
        minimum=3,
    )
    warmup_result = _run_sample(
        token,
        arguments,
        workspace=workspace,
        runner=runner,
        timeout_seconds=timeout_seconds,
        stage=f"benchmark-{surface}-{variant}",
        phase="warmup",
    )
    _parse_performance(
        surface,
        operation,
        warmup_result,
        workspace=workspace,
        expected_frames=expected_frames,
        expected_metadata=effective_data,
    )
    measured: list[dict[str, Any]] = []
    for index in range(repetitions):
        command = _run_sample(
            token,
            arguments,
            workspace=workspace,
            runner=runner,
            timeout_seconds=timeout_seconds,
            stage=f"benchmark-{surface}-{variant}-{index + 1}",
            phase="measure",
        )
        metrics = _parse_performance(
            surface,
            operation,
            command,
            workspace=workspace,
            expected_frames=expected_frames,
            expected_metadata=effective_data,
        )
        if surface == "pynvc" and operation == "decode":
            megapixels_per_second = None
            metric_omissions = {
                "megapixels_per_second": _PYNVC_DECODE_MP_OMISSION,
            }
        else:
            metric_width = int(data["width"])
            metric_height = int(data["height"])
            if surface == "native" and operation == "decode":
                observed = _mapping(metrics.get("input_media"), "input_media")
                metric_width = int(observed["width"])
                metric_height = int(observed["height"])
            megapixels_per_second = _megapixels_per_second(
                float(metrics["fps"]),
                metric_width,
                metric_height,
            )
            metric_omissions = None
        metrics.update(
            {
                "repetition": index + 1,
                "megapixels_per_second": megapixels_per_second,
                "command": command,
            }
        )
        if metric_omissions is not None:
            metrics["metric_omissions"] = metric_omissions
        measured.append(metrics)
    result = {
        "status": "completed",
        "surface": surface,
        "sample": token.sample,
        "sample_authority": _token_evidence(token),
        "provenance_report": dict(authentication.report_identity),
        "input": {
            key: data[key]
            for key in (
                "identity", "width", "height", "frames", "fps", "codec", "format",
                "source_url", "license", "attribution",
            )
        },
        "help_commands": help_evidence,
        "advertised_options": sorted(advertised),
        "benchmark_arguments": list(arguments),
        "frame_accounting": _frame_accounting(frame_plan, workers),
        "warmup": warmup_result,
        "measured_runs": measured,
        "summary": _summarize(measured),
    }
    result.update(_generated_config_result(workspace, generated_config))
    return result


def _variants(request: Mapping[str, Any], route: str) -> list[tuple[str, dict[str, Any]]]:
    if route == "camera_capacity":
        counts = request.get("worker_counts")
        if (
            not isinstance(counts, list)
            or len(counts) < 2
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0
                for item in counts
            )
            or counts[0] != 1
            or any(right <= left for left, right in zip(counts, counts[1:]))
        ):
            raise BenchmarkError(
                "camera_capacity worker_counts must be a strictly increasing "
                "integer series beginning at 1"
            )
        return [
            (f"workers-{count}", {**request, "workers": count}) for count in counts
        ]
    if route != "compare":
        return [(str(request.get("name", route)), dict(request))]
    raw = request.get("variants")
    if not isinstance(raw, list) or len(raw) < 2:
        raise BenchmarkError("compare requires at least two variants")
    result: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(raw):
        variant = dict(_mapping(item, f"variants[{index}]"))
        name = _string(variant.pop("name", None), f"variants[{index}].name")
        merged = dict(request)
        merged.pop("variants", None)
        merged.update(variant)
        result.append((name, merged))
    if len({name for name, _item in result}) != len(result):
        raise BenchmarkError("compare variant names must be unique")
    return result


def _media_input_preflight(
    request: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    """Identify absent user media before workspace, environment, or sample work."""
    operation = _operation(request, route)
    mode = _mode(request)
    field = "input" if operation == "encode" else "encoded_artifact"
    variants = _variants(request, route)
    missing = [name for name, variant in variants if variant.get(field) is None]
    if not missing:
        return None
    return {
        "schema_version": "2.0",
        "kind": "nvcodec-benchmark-result",
        "route": route,
        "mode": mode,
        "operation": operation,
        "status": "input_required",
        "gate": "media_input",
        "accepted_inputs": list(ACCEPTED_MEDIA_INPUTS),
        "next_action": "provide_media_path_or_url",
        "synthetic_input_allowed": False,
        "required_field": field,
        "missing_variants": missing,
    }


def _pynvc_interpreter_preflight(
    request: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    """Require one exact selector only when the local request selects Py."""
    if request.get("environment") is not None:
        return None
    surface = request.get("surface", "auto")
    interpreter = request.get("pynvc_interpreter")
    missing_for_selected = surface in {"pynvc", "both"} and interpreter is None
    selector_is_valid = False
    if isinstance(interpreter, str):
        candidate = Path(interpreter)
        if (
            interpreter
            and interpreter == interpreter.strip()
            and candidate.is_absolute()
            and str(candidate) == interpreter
        ):
            try:
                details = candidate.stat()
            except OSError:
                pass
            else:
                selector_is_valid = bool(
                    stat.S_ISREG(details.st_mode) and details.st_mode & 0o111
                )
    malformed_for_considered = (
        surface in {"pynvc", "both", "auto"}
        and interpreter is not None
        and not selector_is_valid
    )
    if not missing_for_selected and not malformed_for_considered:
        return None
    return {
        "schema_version": "2.0",
        "kind": "nvcodec-benchmark-result",
        "route": route,
        "mode": _mode(request),
        "operation": _operation(request, route),
        "status": "input_required",
        "gate": "surface_selector",
        "requested_surface": surface,
        "surface": "pynvc",
        "required_field": "pynvc_interpreter",
        "reason": (
            "provide one canonical absolute path to an existing executable regular-file "
            "Python interpreter for the PyNvVideoCodec environment; no environment "
            "scan is performed"
        ),
        "next_action": "provide_pynvc_interpreter_and_retry",
        "next_actions": [
            "provide_pynvc_interpreter_and_retry",
            "set_surface_native_if_only_native_is_required",
        ],
        "variants": [],
        "no_mutation_performed": True,
    }


def _validate_compare(
    prepared: Sequence[
        tuple[
            str,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any] | None,
            Mapping[str, Any] | None,
        ]
    ],
) -> None:
    if len(prepared) != 2 or {name.lower() for name, *_rest in prepared} != {"p4", "p5"}:
        raise BenchmarkError("compare requires exactly P4 and P5 variants")
    canonical: dict[str, bytes] = {}
    for name, variant, data, _recipe_identity, recipe in prepared:
        if recipe is None:
            raise BenchmarkError("P4/P5 comparison requires encode recipes")
        # _recipe() has already replayed the recipe from its original intent and
        # current catalog. The effective encoder intent is therefore sufficient
        # here; projection/default provenance cannot conceal a semantic change.
        intent = dict(_mapping(recipe.get("encoder_intent"), "recipe.encoder_intent"))
        preset = str(intent.pop("preset", "")).lower()
        if preset != name.lower():
            raise BenchmarkError(f"{name} variant recipe preset is {preset!r}")
        workload = {
            key: data[key]
            for key in (
                "path", "identity", "width", "height", "frames", "fps", "codec",
                "format", "source_url", "license", "attribution",
            )
        }
        facts = dict(variant)
        facts["input"] = {
            "request": variant.get("input"),
            "effective": workload,
        }
        facts["recipe"] = intent
        facts.update(
            {
                "mode": _mode(variant),
                "operation": _operation(variant, "compare"),
                "surface": variant.get("surface", "auto"),
                "workers": _integer(variant.get("workers", 1), "workers"),
                "gpu": _integer(
                    variant.get("gpu", intent.get("gpu")),
                    "gpu",
                    minimum=0,
                ),
                "process_model": _process_model(variant),
                "single_context": variant.get("single_context") is True,
                "host_memory": variant.get("host_memory") is True,
                "measured_repetitions": _integer(
                    variant.get("measured_repetitions", 3),
                    "measured_repetitions",
                    minimum=3,
                ),
                "require_source_frames": (
                    variant.get("require_source_frames") is True
                ),
            }
        )
        canonical[name.lower()] = artifact_io.canonical_json_bytes(facts)
    if canonical["p4"] != canonical["p5"]:
        raise BenchmarkError(
            "P4/P5 comparison must keep workload and every non-preset control identical"
        )


def _planned_operation(
    surface: str,
    operation: str,
    request: Mapping[str, Any],
    *,
    variant: str,
    data: Mapping[str, Any],
    recipe: Mapping[str, Any] | None,
    workspace: Path,
) -> dict[str, Any]:
    """Describe one candidate invocation without authentication or execution."""
    sample = (
        ("AppEncPerf" if operation == "encode" else "AppDecPerf")
        if surface == "native"
        else (
            "samples/advanced/encode_perf.py"
            if operation == "encode"
            else "samples/advanced/decode_perf.py"
        )
    )
    generated_config = None
    if surface == "native":
        arguments = _native_arguments(operation, data, recipe, request)
    else:
        arguments = _pynvc_arguments(
            operation,
            data,
            recipe,
            request,
            workspace=workspace,
            variant=variant,
        )
        if operation == "encode":
            generated_config = _pynvc_config_identity(arguments, workspace)
    result = {
        "status": "planned",
        "surface": surface,
        "sample": sample,
        "candidate_arguments": arguments,
        "sample_authentication": "deferred_until_execute",
        "operation_verification": "not_attempted",
    }
    if surface == "native" and operation == "encode":
        result["option_validation"] = (
            "pending_installed_AppEncPerf_-h_and_-A_advertisement"
        )
    if generated_config is not None:
        result["generated_config"] = generated_config
    return result


def _benchmark_route(
    request: Mapping[str, Any],
    *,
    route: str,
    workspace: Path,
    runner: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    # One route-level transaction preserves independent surface outcomes.
    # pylint: disable=too-many-branches,too-many-statements
    operation = _operation(request, route)
    mode = _mode(request)
    variants = _variants(request, route)
    requested_surface = request.get("surface", "auto")
    prepared: list[
        tuple[
            str,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, Any] | None,
        ]
    ] = []
    selected_gpus: list[int] = []
    for validation_index, (name, variant) in enumerate(variants, start=1):
        data = _input(variant, operation)
        if operation == "encode":
            recipe_identity, recipe = _recipe(
                variant,
                workspace=workspace,
                timeout_seconds=timeout_seconds,
                validation_index=validation_index,
            )
            intent = _mapping(recipe.get("encoder_intent"), "recipe.encoder_intent")
            recipe_gpu = _integer(
                intent.get("gpu"), "recipe.encoder_intent.gpu", minimum=0
            )
            # First bind request and recipe without requiring setup-owned evidence.
            _bind_recipe_workload(data, recipe, variant, recipe_gpu)
            selected_gpus.append(recipe_gpu)
        else:
            recipe_identity, recipe = None, None
            selected_gpus.append(
                _integer(variant.get("gpu", 0), "gpu", minimum=0)
            )
        prepared.append((name, variant, data, recipe_identity, recipe))
    if len(set(selected_gpus)) != 1:
        raise BenchmarkError("all benchmark variants must select the same GPU")
    selected_gpu = selected_gpus[0]
    authority = _runtime_authority(
        request,
        operation=operation,
        selected_gpu=selected_gpu,
        workspace=workspace,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    environment = _mapping(authority["value"], "runtime authority")
    plan = _surface_plan(
        request,
        environment,
        operation,
        _mapping(authority.get("diagnostics", {}), "runtime diagnostics"),
        (
            None
            if authority["source"] == "setup_environment"
            else _mapping(
                authority.get("normalized_surfaces", {}),
                "normalized local surfaces",
            )
        ),
    )
    if authority["source"] == "local_discovery" and authority.get(
        "normalized_surfaces"
    ):
        binding_workspace, binding_identity = _stage_local_binding(
            workspace, environment
        )
        authority["staged_workspace"] = binding_workspace
        authority["staged_identity"] = binding_identity
    authority_fields = _runtime_result_fields(authority)
    setup_dependencies = _runtime_setup_dependencies(
        request, authority, environment, operation, plan
    )
    dependency_fields = _dependency_result_fields(setup_dependencies)
    if not plan["selected_surfaces"] and (requested_surface != "both" or mode == "execute"):
        return {
            "schema_version": "2.0", "kind": "nvcodec-benchmark-result",
            "route": route, "status": plan["classification"], "mode": mode,
            **authority_fields,
            **dependency_fields,
            "surface_plan": plan, "variants": [],
        }
    variant_surfaces: dict[str, list[str]] = {}
    projection_blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for name, _variant, _data, _recipe_identity, recipe_value in prepared:
        active, blocked = _projection_plan(
            operation,
            plan["selected_surfaces"],
            recipe_value,
        )
        variant_surfaces[name] = active
        projection_blocks[name] = blocked
    frame_plans = {
        name: _frame_plan(operation, data, variant_surfaces[name], variant)
        for name, variant, data, _recipe_identity, _recipe_value in prepared
    }
    if route == "compare":
        _validate_compare(prepared)
    if route == "camera_capacity":
        camera_fps = _number(request.get("camera_fps"), "camera_fps")
        safety_margin = _number(request.get("safety_margin"), "safety_margin")
        if safety_margin > 1:
            raise BenchmarkError("safety_margin must be greater than zero and at most 1")
    reported_surfaces = (
        SURFACES if requested_surface == "both" else tuple(plan["selected_surfaces"])
    )
    if mode == "dry_run":
        planned_variants: list[dict[str, Any]] = []
        branch_states: list[str] = []
        for name, variant, data, recipe_identity, recipe_value in prepared:
            planned_workers = _integer(variant.get("workers", 1), "workers")
            accounting = _frame_accounting(frame_plans[name], planned_workers)
            planned_operations: list[dict[str, Any]] = []
            for surface in reported_surfaces:
                if surface not in plan["selected_surfaces"]:
                    planned = {
                        "status": "blocked",
                        "surface": surface,
                        "reason": (
                            "surface is not eligible in the current live environment"
                        ),
                    }
                elif surface in projection_blocks[name]:
                    planned = dict(projection_blocks[name][surface])
                else:
                    try:
                        planned = _planned_operation(
                            surface,
                            operation,
                            variant,
                            variant=name,
                            data=_effective_data(data, frame_plans[name]),
                            recipe=recipe_value,
                            workspace=workspace,
                        )
                    except (OSError, ValueError) as exc:
                        planned = {
                            "status": "blocked",
                            "surface": surface,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                planned["frame_accounting"] = dict(accounting)
                planned_operations.append(planned)
                branch_states.append(str(planned["status"]))
            planned_variant = {
                "name": name,
                "operation": operation,
                "surfaces": list(reported_surfaces),
                "frame_accounting": accounting,
                "planned_operations": planned_operations,
            }
            if recipe_identity is not None:
                planned_variant["recipe"] = dict(recipe_identity)
            planned_variants.append(planned_variant)
        if (
            branch_states
            and all(state == "planned" for state in branch_states)
            and plan["classification"] == "ready"
        ):
            dry_run_status = "planned"
        elif "planned" in branch_states:
            dry_run_status = "partial"
        else:
            dry_run_status = "blocked"
        return {
            "schema_version": "2.0", "kind": "nvcodec-benchmark-result",
            "route": route, "status": dry_run_status, "mode": mode,
            **authority_fields, **dependency_fields, "surface_plan": plan,
            "variants": planned_variants,
        }
    # Recheck supplied setup evidence; local authority was discovered and staged here.
    if authority["source"] == "setup_environment":
        environment_source = authority["external_identity"]
        artifact_io.verify_external_artifact(environment_source)
        environment_workspace, environment_identity = _stage_environment(
            workspace, environment
        )
    else:
        environment_workspace = authority["staged_workspace"]
        environment_identity = authority["staged_identity"]
    for _name, _variant, _data, recipe_identity, _recipe_value in prepared:
        if recipe_identity is not None:
            artifact_io.verify_external_artifact(recipe_identity)
    authentication_surfaces = [
        surface
        for surface in SURFACES
        if any(
            surface in variant_surfaces[name]
            for name, _variant, _data, _recipe_identity, _recipe_value in prepared
        )
    ]
    authentication = _authentication(
        operation,
        authentication_surfaces,
        environment_workspace=environment_workspace,
        environment_identity=environment_identity,
        workspace=workspace,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    native_help: tuple[set[str], list[dict[str, Any]]] | None = None
    native_authority = authentication.get("native")
    if (
        operation == "encode"
        and native_authority
        and native_authority["status"] == "authenticated"
    ):
        try:
            native_help = _native_help(
                _token(native_authority["value"], operation, "native"),
                workspace=workspace,
                runner=runner,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, ValueError) as exc:
            native_authority.clear()
            native_authority.update({
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    results: list[dict[str, Any]] = []
    for name, variant, data, recipe_identity, recipe in prepared:
        planned_workers = _integer(variant.get("workers", 1), "workers")
        accounting = _frame_accounting(frame_plans[name], planned_workers)
        branches: dict[str, Any] = {}
        for surface in reported_surfaces:
            if surface not in plan["selected_surfaces"]:
                branches[surface] = {
                    "status": "blocked",
                    "surface": surface,
                    "reason": "surface is not eligible in the current live environment",
                }
                continue
            if surface in projection_blocks[name]:
                branches[surface] = dict(projection_blocks[name][surface])
                continue
            authority = authentication[surface]
            if authority["status"] != "authenticated":
                branches[surface] = {
                    "status": "failed",
                    "surface": surface,
                    "reason": authority["reason"],
                }
                continue
            try:
                branches[surface] = _run_surface(
                    surface,
                    operation,
                    variant,
                    variant=name,
                    data=data,
                    recipe=recipe,
                    authentication=authority["value"],
                    workspace=workspace,
                    runner=runner,
                    timeout_seconds=timeout_seconds,
                    frame_plan=frame_plans[name],
                    native_help=native_help,
                )
            except (OSError, ValueError) as exc:
                branches[surface] = {
                    "status": "failed",
                    "surface": surface,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        for branch in branches.values():
            branch["frame_accounting"] = dict(accounting)
        result_variant = {"name": name, "operation": operation, "surfaces": branches}
        if recipe_identity is not None:
            result_variant["recipe"] = dict(recipe_identity)
        results.append(result_variant)
    branch_states = [
        branch["status"]
        for item in results
        for branch in item["surfaces"].values()
    ]
    completed = branch_states.count("completed")
    if completed and completed == len(branch_states) and plan["classification"] == "ready":
        status = "completed"
    elif completed:
        status = "partial"
    elif branch_states and all(state == "blocked" for state in branch_states):
        status = "blocked"
    else:
        status = "failed"
    output: dict[str, Any] = {
        "schema_version": "2.0", "kind": "nvcodec-benchmark-result",
        "route": route, "status": status, "mode": mode,
        **authority_fields, **dependency_fields, "surface_plan": plan,
        "variants": results,
    }
    if route == "compare":
        output["comparison"] = {
            surface: {
                item["name"]: item["surfaces"][surface]["summary"]
                for item in results
                if item["surfaces"].get(surface, {}).get("status") == "completed"
            }
            for surface in SURFACES
            if any(surface in item["surfaces"] for item in results)
        }
    if route == "camera_capacity":
        capacity: dict[str, Any] = {}
        for surface in SURFACES:
            points = []
            for item in results:
                workers = int(item["name"].removeprefix("workers-"))
                branch = item["surfaces"].get(surface, {"status": "blocked"})
                mean_fps = (
                    branch["summary"]["fps"]["mean"]
                    if branch["status"] == "completed"
                    else None
                )
                points.append(
                    {
                        "workers": workers,
                        "status": branch["status"],
                        "mean_fps": mean_fps,
                        "margin_adjusted_fps": (
                            mean_fps * safety_margin if mean_fps is not None else None
                        ),
                        "meets_requested_load": (
                            mean_fps is not None
                            and mean_fps * safety_margin >= workers * camera_fps
                        ),
                    }
                )
            if any(point["status"] != "blocked" for point in points):
                passing = [
                    point["workers"] for point in points if point["meets_requested_load"]
                ]
                capacity[surface] = {
                    "requested_fps_per_camera": camera_fps,
                    "safety_margin": safety_margin,
                    "tested_points": points,
                    "safe_tested_camera_count": max(passing, default=0),
                    "scope": (
                        "codec-only measured estimate; camera capture and end-to-end "
                        "latency remain unverified"
                    ),
                }
        output["camera_capacity"] = capacity
    return output


ROUTE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "encode": _benchmark_route,
    "decode": _benchmark_route,
    "compare": _benchmark_route,
    "camera_capacity": _benchmark_route,
}


def run_benchmark_request(
    request: Mapping[str, Any],
    *,
    workspace: Path,
    runner: Any = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Execute one benchmark-domain request in a private fresh workspace."""
    value = _mapping(request, "request")
    if value.get("schema_version") != "1.0" or value.get("kind") != "nvcodec-benchmark-request":
        raise BenchmarkError("request must be a schema-1.0 nvcodec-benchmark-request")
    route = value.get("route")
    if route not in ROUTE_HANDLERS:
        raise BenchmarkError(f"route must be exactly one of {sorted(ROUTE_HANDLERS)}")
    timeout = _number(timeout_seconds, "timeout_seconds")
    input_required = _media_input_preflight(value, str(route))
    if input_required is not None:
        return input_required
    selector_required = _pynvc_interpreter_preflight(value, str(route))
    if selector_required is not None:
        return selector_required
    root = _workspace(Path(workspace))
    service = runner or command_runner.CommandRunner(root)
    handler = ROUTE_HANDLERS[str(route)]
    try:
        return handler(
            value,
            route=str(route),
            workspace=root,
            runner=service,
            timeout_seconds=timeout,
        )
    except DependencyRequired as exc:
        return {
            "schema_version": "2.0",
            "kind": "nvcodec-benchmark-result",
            "route": route,
            "mode": _mode(value),
            "operation": _operation(value, str(route)),
            "status": "dependency_required",
            "gate": "skill_dependency",
            "dependency": exc.dependency,
            "variants": [],
        }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for one request JSON and one fresh private workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = artifact_io.strict_json_loads(args.request.read_bytes())
        result = run_benchmark_request(request, workspace=args.workspace)
        if result.get("status") == "input_required":
            print(
                artifact_io.canonical_json_bytes(result).decode("utf-8"),
                end="",
            )
        else:
            workspace = artifact_io.resolve_private_workspace(args.workspace)
            artifact_io.write_fresh_json(workspace, args.output, result)
    except (OSError, ValueError) as exc:
        print(f"benchmark request failed: {exc}", file=sys.stderr)
        return 2
    return 0 if result.get("status") in {"completed", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
