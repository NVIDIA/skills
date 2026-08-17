#!/usr/bin/env python3
"""Run compact, authenticated Video SDK pipeline routes and verify handoffs."""

# The route handlers deliberately keep their complete producer/consumer facts
# local so boundary evidence stays reviewable.
# pylint: disable=too-many-arguments,too-many-locals,too-many-lines

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

if not sys.flags.ignore_environment or not sys.flags.no_user_site:
    raise SystemExit("invoke this producer with isolated Python: python3 -I")

# Isolated mode excludes the script directory. Add only this skill's fixed
# private module directory; sibling skills remain subprocess boundaries.
_SCRIPT_DIR = Path(__file__).absolute().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# pylint: disable=wrong-import-position,import-error
import _pipeline_provenance as sample_provenance  # noqa: E402
import _pipeline_runtime as artifact_io  # noqa: E402
# pylint: enable=wrong-import-position,import-error

command_runner = artifact_io
pynvc_sample_provenance = sample_provenance

SCHEMA_VERSION = "1.0"
REQUEST_KIND = "nvcodec-pipeline-request"
RESULT_KIND = "nvcodec-pipeline-result"
IDENTITY_KEYS = {"schema_version", "kind", "path", "size_bytes", "sha256"}
BOUNDARY_KEYS = {
    "producer", "consumer", "artifact_path", "artifact_size_bytes",
    "artifact_sha256", "status", "reason",
}
ROUTE_TABLE = {
    "encode_decode": "_run_encode_decode",
    "native_transcode": "_run_native_transcode",
    "pynvc_segments": "_run_pynvc_segments",
    "container_triage": "_run_container_triage",
    "av1_verify": "_run_av1_verify",
    "acceptance": "_run_acceptance",
}
COMMAND_KEYS = {
    "schema_version", "kind", "command_id", "stage", "phase", "argv", "cwd",
    "environment_keys", "started_at", "ended_at", "duration_seconds",
    "approval_wait_seconds", "timeout_seconds", "exit_code", "timed_out",
    "launch_error", "stdout", "stderr", "stdout_tail", "stderr_tail",
}
NATIVE_DECODE = re.compile(r"^Total frame decoded: ([0-9]+)$", re.MULTILINE)
NATIVE_TRANSCODE_LEGACY = re.compile(r"\(#totFrames=([0-9]+)\)")
NATIVE_TRANSCODE_CURRENT = re.compile(
    r"^Total frame transcoded: ([0-9]+)$", re.MULTILINE
)
NATIVE_SAVED = re.compile(r"^Saved in file (.+) of (8|10) bit depth$", re.MULTILINE)
NATIVE_ENCODE = re.compile(r"^Total frames encoded: ([0-9]+)$", re.MULTILINE)
PYNVC_ALL_DECODE = re.compile(
    r"^Successfully decoded all ([0-9]+) frames to (.+)$", re.MULTILINE
)
PYNVC_LIMITED_DECODE = re.compile(
    r"^Successfully decoded requested ([0-9]+) frames to (.+)$", re.MULTILINE
)
SEGMENT_CREATED = re.compile(r"^\u2713 Created: (.+)$", re.MULTILINE)
SEGMENT_SUMMARY = re.compile(
    r"^Summary: ([0-9]+) segments created successfully$", re.MULTILINE
)
EXPLICIT_FAILURE = sample_provenance.OFFICIAL_SAMPLE_FAILURE_PATTERN
FAILURE_DETAIL_LIMIT = 200
ACCEPTANCE_STATIC_FILES = (
    "report.md",
    "capability-report.md",
    "review-findings.md",
    "timing.json",
    "task-results.jsonl",
    "commands.jsonl",
    "evidence/summary.json",
    "evidence/representative-content.json",
)
ACCEPTANCE_TASKS = (
    "readiness",
    "capabilities",
    "content",
    "p4_recipe",
    "p5_recipe",
    "p4_encode_decode",
    "p5_encode_decode",
    "p4_benchmark",
    "p5_benchmark",
    "handoffs",
)
ACCEPTANCE_LEDGER_REFERENCES = ("commands", "task_results", "timing")
REPRESENTATIVE_CONTENT_FIELDS = (
    "source_url", "license", "attribution", "path", "size_bytes", "sha256",
)
MEDIA_INPUT_ROUTES = {
    "encode_decode",
    "native_transcode",
    "pynvc_segments",
    "container_triage",
    "av1_verify",
    "acceptance",
}
DIRECT_INPUT_ROUTES = {"native_transcode", "pynvc_segments", "av1_verify"}


def _acceptance_files(request: Mapping[str, Any]) -> tuple[str, ...]:
    filename = request.get("validation_result_filename")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".json")
        or filename in {"summary.json", "representative-content.json"}
    ):
        raise PipelineError(
            "validation_result_filename must be one fresh non-reserved JSON filename"
        )
    return (*ACCEPTANCE_STATIC_FILES, f"evidence/{filename}")


class PipelineError(ValueError):
    """A pipeline request or operation violates the customer contract."""


class SurfaceInputRequired(PipelineError):
    """One exact selector is needed to authenticate a manually installed SDK."""

    def __init__(self, field: str, surface: str, reason: str):
        super().__init__(reason)
        self.field = field
        self.surface = surface
        self.reason = reason


class SurfaceSetupRequired(PipelineError):
    """The selected SDK surface failed independent local authentication."""

    def __init__(self, surface: str, reason: str):
        super().__init__(reason)
        self.surface = surface
        self.reason = reason


class BoundaryFailure(PipelineError):
    """One named producer-to-consumer boundary failed."""

    def __init__(self, producer: str, consumer: str, path: str, reason: str):
        super().__init__(reason)
        self.producer = producer
        self.consumer = consumer
        self.path = path
        self.reason = reason


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


_providers_cache: tuple[Callable[..., Any], ...] | None = None  # pylint: disable=invalid-name


def _encode_controller() -> Any:
    """Load the encode owner only for a route that consumes encode contracts."""
    try:
        return importlib.import_module("encode_controller")
    except ModuleNotFoundError as exc:
        if exc.name == "encode_controller":
            raise PipelineError(
                "pipeline encode controller is missing; reinstall jetson-video-pipeline"
            ) from exc
        raise


def _run_encode_request(*args: Any, **kwargs: Any) -> Any:
    """Delegate without importing the encode owner on unrelated routes."""
    return _encode_controller().run_encode_request(*args, **kwargs)


def _providers() -> tuple[Callable[..., Any], ...]:
    global _providers_cache  # pylint: disable=global-statement
    if _providers_cache is None:
        _providers_cache = (
            _run_encode_request,
            sample_provenance.authenticate_native_samples,
            pynvc_sample_provenance.authenticate_pynvc_samples,
            sample_provenance.run_authenticated,
        )
    return _providers_cache


def _workspace(path: Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists():
        root = artifact_io.resolve_private_workspace(requested)
        if any(root.iterdir()):
            raise PipelineError("pipeline workspace must be fresh and empty")
        return root
    return artifact_io.create_private_workspace(requested)


def _timeout(value: Any, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError("timeout_seconds must be a positive finite number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0 or seconds > maximum:
        raise PipelineError(f"timeout_seconds must be in (0, {maximum}]")
    return seconds


def _request(value: Any, timeout_ceiling: float) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineError("request must be a JSON object")
    request = dict(value)
    if request.get("schema_version") != SCHEMA_VERSION or request.get("kind") != REQUEST_KIND:
        raise PipelineError(
            f"request identity must be schema {SCHEMA_VERSION} kind {REQUEST_KIND}"
        )
    if request.get("route") not in ROUTE_TABLE:
        raise PipelineError(f"route must be one of {sorted(ROUTE_TABLE)}")
    request["mode"] = request.get("mode", "execute")
    if request["mode"] not in {"dry_run", "execute"}:
        raise PipelineError("mode must be dry_run or execute")
    request["timeout_seconds"] = _timeout(
        request.get("timeout_seconds", timeout_ceiling), timeout_ceiling
    )
    return request


def _exact_http_url(value: Any, *, label: str) -> str:
    """Return one unchanged, credential-free HTTP(S) URL."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(
            character.isspace()
            or ord(character) < 0x20
            or ord(character) == 0x7F
            for character in value
        )
    ):
        raise PipelineError(f"{label} must be one exact HTTP(S) URL")
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
    except (UnicodeError, ValueError) as exc:
        raise PipelineError(f"{label} must be one exact HTTP(S) URL") from exc
    if not (
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and hostname
        and username is None
        and password is None
        and (port is None or 1 <= port <= 65535)
    ):
        raise PipelineError(f"{label} must be one exact HTTP(S) URL")
    return value


def _input_required(reason: str) -> dict[str, Any]:
    """Return the common no-work result for absent meaningful media."""
    return {
        "status": "input_required",
        "gate": "media_input",
        "reason": reason,
        "accepted_inputs": ["local_path", "http_url", "https_url"],
        "next_action": "provide_media_path_or_url",
        "synthetic_input_allowed": False,
        "retrieval_attempted": False,
        "operations": {},
        "planned_operations": [],
        "no_mutation_performed": True,
    }


def _media_preflight(  # pylint: disable=too-many-return-statements,too-many-branches
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve missing user media before any route creates a workspace or starts work."""
    route = request["route"]
    if route not in MEDIA_INPUT_ROUTES:
        return None
    if route == "container_triage":
        supplied_input = request.get("input")
        supplied_url = request.get("source_url")
        if supplied_input is not None and supplied_url is not None:
            raise PipelineError(
                "container triage requires exactly one user source: input or source_url"
            )
        if supplied_input is None and supplied_url is None:
            return _input_required("container triage requires user-provided media")
        gate = request.get("target_eligibility")
        if isinstance(gate, Mapping) and gate.get("eligible") is False:
            _eligible, reasons = _target_gate(request)
            return {
                "status": "blocked",
                "gate": "target_eligibility",
                "reasons": reasons,
                "retrieval_attempted": False,
                "operations": {},
                "planned_operations": [],
                "no_mutation_performed": True,
            }
        if supplied_input is None:
            _exact_http_url(
                supplied_url, label="container source_url"
            )
        return None
    if route in DIRECT_INPUT_ROUTES:
        if request.get("input") is None:
            return _input_required(f"{route} requires user-provided media")
        return None
    if route == "encode_decode":
        supplied = request.get("encode_request_ref")
        if supplied is None:
            return _input_required(
                "encode/decode requires a user-media-bound encode request"
            )
        reference = _external(
            supplied, label="encode_request_ref",
            schema="1.0", kinds=("nvcodec-encode-request",),
        )
        encode_request = artifact_io.read_verified_external_json(reference)
        if not isinstance(encode_request, Mapping):
            raise PipelineError("encode_request_ref must contain a JSON object")
        if (
            encode_request.get("schema_version") != "1.0"
            or encode_request.get("kind") != "nvcodec-encode-request"
        ):
            raise PipelineError("encode request has an invalid schema or kind")
        if encode_request.get("input") is None:
            result = _input_required("encode/decode requires user-provided media")
            result["encode_request"] = reference
            return result
        return None
    raw_references = request.get("references")
    if (
        not isinstance(raw_references, Mapping)
        or not raw_references
        or raw_references.get("content") is None
    ):
        return _input_required(
            "acceptance references require user-provided content evidence"
        )
    return None


def _planned_workspace(path: Path) -> Path:
    """Resolve a proposed workspace locator without creating it."""
    requested = Path(path).expanduser()
    if requested.exists():
        root = artifact_io.resolve_private_workspace(requested)
        if any(root.iterdir()):
            raise PipelineError("pipeline workspace must be fresh and empty")
        return root
    try:
        parent = requested.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PipelineError("pipeline workspace parent is unavailable") from exc
    return parent / requested.name


def _external(
    value: Any, *, label: str, schema: str | None = None,
    kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise PipelineError(f"{label} must be an exact portable artifact identity")
    identity = dict(value)
    if schema is not None and identity.get("schema_version") != schema:
        raise PipelineError(f"{label} must use schema {schema}")
    if kinds is not None and identity.get("kind") not in kinds:
        raise PipelineError(f"{label} kind must be one of {list(kinds)}")
    try:
        artifact_io.verify_external_artifact(identity)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"{label} is not current: {exc}") from exc
    return identity


def _bind_environment_gpu(
    environment: Mapping[str, Any], gpu: Any, *, label: str
) -> int:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise PipelineError(f"{label} gpu must be a non-negative integer")
    if environment.get("selected_gpu") != gpu:
        raise PipelineError(f"{label} gpu must equal environment selected_gpu")
    return gpu


def _environment(
    request: Mapping[str, Any],
    *,
    required_surfaces: Sequence[str] = artifact_io.SURFACE_ORDER,
    selected_gpu: int | None = None,
    local_errors: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    supplied = request.get("environment")
    if supplied is not None:
        identity = _external(
            supplied, label="environment",
            schema=artifact_io.ENVIRONMENT_SCHEMA_VERSION,
            kinds=(artifact_io.ENVIRONMENT_KIND,),
        )
        value = artifact_io.validate_environment_envelope(
            artifact_io.read_verified_external_json(identity),
            label="pipeline",
            error_type=PipelineError,
            required_surfaces=required_surfaces,
        )
    else:
        gpu = request.get("gpu", 0) if selected_gpu is None else selected_gpu
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise PipelineError("selected gpu must be a non-negative integer")
        if "pynvc" in required_surfaces:
            interpreter = request.get("pynvc_interpreter")
            if (
                not isinstance(interpreter, str)
                or not interpreter
                or interpreter != interpreter.strip()
                or not Path(interpreter).is_absolute()
                or str(Path(interpreter)) != interpreter
            ):
                raise SurfaceInputRequired(
                    "pynvc_interpreter",
                    "pynvc",
                    "provide one canonical absolute Python interpreter for the existing "
                    "PyNvVideoCodec environment; no environment scan is performed",
                )
        value, errors = sample_provenance.build_local_runtime_binding(
            required_surfaces,
            pynvc_interpreter=request.get("pynvc_interpreter"),
            gpu=gpu,
            timeout_seconds=float(request.get("timeout_seconds", 600)),
        )
        if local_errors is not None:
            local_errors.update(errors)
        if len(required_surfaces) == 1 and required_surfaces[0] in errors:
            raise SurfaceSetupRequired(required_surfaces[0], errors[required_surfaces[0]])
        identity = None
    # Direct pipeline routes select exactly one SDK.  Convert structurally
    # valid but unready supplied evidence into the same setup handoff as local
    # authentication.  Multi-surface encode planning deliberately handles its
    # two surfaces independently in encode_controller._surface_plan.
    if len(required_surfaces) == 1:
        surface = required_surfaces[0]
        defects = (
            artifact_io.native_surface_defects(value)
            if surface == "native"
            else artifact_io.pynvc_surface_defects(value)
        )
        if defects:
            raise SurfaceSetupRequired(surface, "; ".join(defects))
    if selected_gpu is not None:
        _bind_environment_gpu(value, selected_gpu, label="request")
    elif "gpu" in request:
        _bind_environment_gpu(value, request["gpu"], label="request")
    return identity, value


def _pynvc_full_samples_preflight(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Block Torch-dependent Py routes before workspace creation or launch."""
    route = request["route"]
    if route == "container_triage":
        if request.get("surface") != "pynvc":
            return None
    elif route != "pynvc_segments":
        return None
    identity, environment = _environment(request, required_surfaces=("pynvc",))
    reasons = artifact_io.pynvc_full_sample_defects(environment)
    if not reasons:
        return None
    installed = _installed_sibling("jetson-video-setup")
    action = (
        "use jetson-video-setup to provision a new full-samples "
        "PyNvVideoCodec environment, then retry"
        if installed
        else "install jetson-video-setup, provision a new full-samples "
        "PyNvVideoCodec environment, then retry"
    )
    return {
        "status": "dependency_required",
        "gate": "pynvc_full_samples",
        "surface": "pynvc",
        "required_profile": "full-samples",
        "reasons": reasons,
        "dependency": {
            "skill": "jetson-video-setup",
            "installed": installed,
            "needed_for": "PyNvVideoCodec full-samples environment provisioning",
            "next_action": action,
        },
        "retrieval_attempted": False,
        "operations": {},
        "planned_operations": [],
        "planned_output_paths": [],
        "validated_identities": (
            {"environment": identity} if identity is not None else {}
        ),
        "runtime_authority": (
            "supplied_setup_environment"
            if identity is not None
            else "consumer_local_binding"
        ),
        "no_mutation_performed": True,
    }


def _recipe(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _external(
        request.get("recipe"), label="recipe", schema="2.0",
        kinds=("nvcodec-recipe",),
    )
    value = artifact_io.read_verified_external_json(identity)
    try:
        artifact_io.validate_recipe_identity(identity)
    except artifact_io.SkillDependencyError:
        # Preserve the public composition handoff.  SkillDependencyError is a
        # ValueError subclass, so the generic wrapper below would otherwise
        # erase its structured dependency_required result.
        raise
    except ValueError as exc:
        raise PipelineError(f"recipe validation failed: {exc}") from exc
    return identity, value


def _stage_environment(branch: Path, environment: Mapping[str, Any]) -> dict[str, Any]:
    return artifact_io.write_fresh_json(
        branch, branch / "environment.json", dict(environment)
    )


def _promote(root: Path, child: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    path = artifact_io.verify_artifact(child, identity)
    return artifact_io.snapshot_artifact(
        root, path, schema_version=identity["schema_version"], kind=identity["kind"]
    )


def _command_text(runner: Any, result: Mapping[str, Any]) -> tuple[str, str]:
    runner_workspace = artifact_io.resolve_private_workspace(Path(runner.workspace))
    return (
        artifact_io.read_verified_text(
            runner_workspace, result["stdout"], max_bytes=16 * 1024 * 1024
        ),
        artifact_io.read_verified_text(
            runner_workspace, result["stderr"], max_bytes=16 * 1024 * 1024
        ),
    )


def _first_failure_line(stdout: str, stderr: str) -> str | None:
    """Return one bounded, printable line that matched the failure vocabulary."""
    for output in (stdout, stderr):
        for line in output.splitlines():
            if EXPLICIT_FAILURE.search(line):
                printable = "".join(
                    character if character.isprintable() else " "
                    for character in line
                )
                return re.sub(r"\s+", " ", printable).strip()[:FAILURE_DETAIL_LIMIT]
    return None


def _command(
    result: Any, *, runner: Any, argv: Sequence[str], cwd: Path,
    timeout: float, stage: str, phase: str,
) -> tuple[dict[str, Any], str, str]:
    expected = {
        "schema_version": "1", "kind": "command-result", "argv": list(argv),
        "cwd": str(cwd), "timeout_seconds": float(timeout), "stage": stage,
        "phase": phase,
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != COMMAND_KEYS
        or any(result.get(key) != value for key, value in expected.items())
    ):
        raise PipelineError("command result does not bind the exact launch")
    if (
        result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("launch_error") is not None
    ):
        raise PipelineError(
            f"command failed: exit={result.get('exit_code')} "
            f"timeout={result.get('timed_out')} launch={result.get('launch_error')}"
        )
    for key in ("duration_seconds", "approval_wait_seconds"):
        number = result.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise PipelineError(f"command {key} is invalid")
    if result["approval_wait_seconds"] != 0:
        raise PipelineError("non-interactive command cannot report approval wait")
    if not isinstance(result["environment_keys"], list) or not all(
        isinstance(item, str) for item in result["environment_keys"]
    ):
        raise PipelineError("command environment-key evidence is invalid")
    stdout, stderr = _command_text(runner, result)
    failure_line = _first_failure_line(stdout, stderr)
    if failure_line is not None:
        raise PipelineError(
            "command output contains an explicit failure marker: "
            f"{failure_line}"
        )
    return dict(result), stdout, stderr


def _launch(
    token: Any, suffix: Sequence[str], *, branch: Path, runner: Any,
    timeout: float, stage: str, phase: str,
) -> tuple[dict[str, Any], str, str]:
    launch = _providers()[3]
    result = launch(
        token, list(suffix), workspace=branch, runner=runner, cwd=branch,
        timeout_seconds=timeout, stage=stage, phase=phase,
    )
    return _command(
        result, runner=runner, argv=[*token.launcher, *suffix], cwd=branch,
        timeout=timeout, stage=stage, phase=phase,
    )


def _authenticate(
    surface: str, samples: Sequence[str], *, root: Path, branch: Path,
    environment: Mapping[str, Any], runner: Any, timeout: float,
) -> tuple[Any, dict[str, Any]]:
    environment_identity = _stage_environment(branch, environment)
    provider = _providers()[1] if surface == "native" else _providers()[2]
    authentication = provider(
        environment_workspace=branch, environment_identity=environment_identity,
        workspace=branch, samples=tuple(samples), runner=runner,
        timeout_seconds=timeout, report_path=branch / "sample-provenance.json",
    )
    report = _promote(root, branch, authentication.report_identity)
    return authentication, report


def _native_decode_count(text: str, expected_path: Path | None = None) -> int:
    matches = NATIVE_DECODE.findall(text)
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise PipelineError("AppDec must report one positive 'Total frame decoded' marker")
    if expected_path is not None:
        saved = re.findall(r"^Saved in file (.+) in .+ format$", text, re.MULTILINE)
        if saved != [str(expected_path)]:
            raise PipelineError("AppDec saved-output marker does not name the requested output")
    return int(matches[0])


def _pynvc_decode_count(text: str, expected_path: Path) -> int:
    matches = PYNVC_ALL_DECODE.findall(text) + PYNVC_LIMITED_DECODE.findall(text)
    if len(matches) != 1:
        raise PipelineError("Py decode must report exactly one successful frame marker")
    count, path = matches[0]
    if int(count) <= 0 or path != str(expected_path):
        raise PipelineError("Py decode marker has a nonpositive count or wrong output path")
    return int(count)


def _decode(
    surface: str, authentication: Any, encoded: Path, output: Path, *,
    root: Path, branch: Path, runner: Any, timeout: float, gpu: int,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise PipelineError(f"decode output is not fresh: {output}")
    if surface == "native":
        sample = "AppDec"
        suffix = ["-i", str(encoded), "-o", str(output), "-gpu", str(gpu)]
    else:
        sample = "samples/advanced/decode.py"
        suffix = ["-i", str(encoded), "-o", str(output), "-g", str(gpu), "-d", "0"]
        if expected_frames is not None:
            suffix.extend(["-f", str(expected_frames)])
    command, stdout, _stderr = _launch(
        authentication.token(sample), suffix, branch=branch, runner=runner,
        timeout=timeout, stage=f"{surface}-decode", phase="verify",
    )
    frames = (
        _native_decode_count(stdout, output)
        if surface == "native"
        else _pynvc_decode_count(stdout, output)
    )
    if expected_frames is not None and frames != expected_frames:
        raise PipelineError(
            f"decode produced {frames} frames but producer reported {expected_frames}"
        )
    identity = artifact_io.snapshot_artifact(
        root, output, schema_version="1", kind="nvcodec-decoded-frames"
    )
    if identity["size_bytes"] <= 0:
        raise PipelineError("decode produced no frame bytes")
    return {"command": command, "frames": frames, "output": identity, "independent": True}


def _boundary(
    producer: str, consumer: str, identity: Mapping[str, Any], *,
    status: str, reason: str = "",
) -> dict[str, Any]:
    return {
        "producer": producer, "consumer": consumer,
        "artifact_path": identity.get("path", ""),
        "artifact_size_bytes": identity.get("size_bytes"),
        "artifact_sha256": identity.get("sha256"),
        "status": status, "reason": reason,
    }


def _run_encode_decode(  # pylint: disable=too-many-branches,too-many-statements
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    encode_request_ref = _external(
        request.get("encode_request_ref"), label="encode_request_ref",
        schema="1.0", kinds=("nvcodec-encode-request",),
    )
    encode_request = artifact_io.read_verified_external_json(encode_request_ref)
    if not isinstance(encode_request, Mapping):
        raise PipelineError("encode_request_ref must contain a JSON object")
    child = artifact_io.create_private_workspace(root / "encode-decode")
    encoded = _providers()[0](
        encode_request, workspace=child, runner=runner, timeout_seconds=timeout
    )
    if not isinstance(encoded, Mapping) or encoded.get("kind") != "nvcodec-encode-result":
        raise PipelineError("encode controller returned an invalid result")
    artifact_io.verify_external_artifact(encode_request_ref)
    encoded_result_ref = artifact_io.write_fresh_json(
        root, child / "encode-controller-result.json", dict(encoded)
    )
    operations: dict[str, Any] = {}
    boundaries: list[dict[str, Any]] = []
    verified = 0
    for surface in encoded.get("selected_surfaces", []):
        operation = encoded.get("operations", {}).get(surface)
        if not isinstance(operation, Mapping) or operation.get("status") != "operation_verified":
            reason = (
                "; ".join(operation.get("reasons", []))
                if isinstance(operation, Mapping) else "missing operation"
            )
            boundaries.append({
                "producer": f"{surface}-encode", "consumer": f"{surface}-decode",
                "artifact_path": "", "artifact_size_bytes": None,
                "artifact_sha256": None, "status": "failed", "reason": reason,
            })
            operations[surface] = (
                dict(operation) if isinstance(operation, Mapping)
                else {"status": "operation_failed"}
            )
            continue
        encode_data = operation.get("encode", {})
        decode_data = operation.get("decode", {})
        encoded_identity = encode_data.get("output")
        decoded_identity = decode_data.get("output")
        try:
            if (
                not isinstance(encoded_identity, Mapping)
                or not isinstance(decoded_identity, Mapping)
            ):
                raise PipelineError("operation omits encode or decode output identity")
            promoted_encoded = _promote(root, child, encoded_identity)
            promoted_decoded = _promote(root, child, decoded_identity)
            if (
                decode_data.get("independent") is not True
                or decode_data.get("frames", 0) <= 0
                or decode_data.get("consumed_bitstream_sha256") != promoted_encoded["sha256"]
            ):
                raise PipelineError("independent decode did not consume the exact encoded SHA-256")
            artifact_io.verify_artifact(root, promoted_encoded)
            boundaries.append(
                _boundary(
                    f"{surface}-encode", f"{surface}-decode", promoted_encoded,
                    status="verified",
                )
            )
            operations[surface] = {
                "status": "operation_verified", "encoded": promoted_encoded,
                "decoded": promoted_decoded, "encode_frames": encode_data.get("frames"),
                "decode_frames": decode_data.get("frames"),
            }
            verified += 1
        except (OSError, ValueError) as exc:
            boundaries.append(
                _boundary(
                    f"{surface}-encode", f"{surface}-decode",
                    encoded_identity if isinstance(encoded_identity, Mapping) else {},
                    status="failed", reason=str(exc),
                )
            )
            operations[surface] = {"status": "operation_failed", "reasons": [str(exc)]}
    encoded_operations = encoded.get("operations", {})
    if isinstance(encoded_operations, Mapping):
        for surface in ("native", "pynvc"):
            operation = encoded_operations.get(surface)
            if surface not in operations and isinstance(operation, Mapping):
                operations[surface] = dict(operation)
    selected = encoded.get("selected_surfaces", [])
    if encoded.get("status") in {
        "selection_required",
        "input_required",
        "dependency_required",
    }:
        status = "selection_required"
        if encoded.get("status") != "selection_required":
            status = str(encoded["status"])
    elif not selected:
        status = "blocked"
    elif verified == len(selected) and encoded.get("classification") == "ready":
        status = "complete"
    elif verified:
        status = "partial"
    else:
        status = "failed"
    result = {
        "status": status, "surface_status": encoded.get("status"),
        "selected_surfaces": list(selected), "operations": operations,
        "boundaries": boundaries, "encode_request": encode_request_ref,
        "encode_result": encoded_result_ref, "encode_workspace": str(child),
    }
    if isinstance(encoded.get("surface_plan"), Mapping):
        result["surface_plan"] = dict(encoded["surface_plan"])
    if isinstance(encoded.get("capability_validation"), Mapping):
        result["surface_evaluations"] = dict(encoded["capability_validation"])
    for key in ("gate", "surface", "required_field", "reason", "next_action", "dependency"):
        if key in encoded:
            result[key] = encoded[key]
    return result


def _run_native_transcode(
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    _recipe_identity, recipe = _recipe(request)
    content = _external(request.get("input"), label="input video")
    input_path = artifact_io.verify_external_artifact(content)
    intent = recipe["encoder_intent"]
    _environment_identity, environment = _environment(
        request,
        required_surfaces=("native",),
        selected_gpu=intent.get("gpu"),
    )
    _bind_environment_gpu(environment, intent.get("gpu"), label="recipe")
    if intent.get("codec") != "hevc" or recipe["projections"]["native"]["status"] != "exact":
        raise PipelineError("native_transcode requires an exact HEVC native recipe")
    branch = artifact_io.create_private_workspace(root / "native-transcode")
    authentication, provenance = _authenticate(
        "native", ("AppTrans", "AppDec"), root=root, branch=branch,
        environment=environment, runner=runner, timeout=timeout,
    )
    output = branch / "transcoded.hevc"
    decoded = branch / "transcoded-decoded.yuv"
    options = _transcode_options(recipe)
    suffix = [
        "-i", str(input_path), "-o", str(output), "-gpu", str(intent["gpu"]),
        *options,
    ]
    return _execute_native_transcode(
        authentication=authentication, suffix=suffix, output=output,
        decoded=decoded, root=root, branch=branch, runner=runner,
        timeout=timeout, intent=intent, content=content, provenance=provenance,
    )


def _transcode_options(recipe: Mapping[str, Any]) -> list[str]:
    """Project schema-2 encoder controls onto AppTrans without raw-input flags."""
    projection = recipe["projections"]["native"]["cli_options"]
    options: list[str] = []
    index = 0
    while index < len(projection):
        option = projection[index]
        if option == "-temporalaq":
            options.append(option)
            index += 1
            continue
        if index + 1 >= len(projection):
            raise PipelineError("native recipe projection has an incomplete option")
        value = projection[index + 1]
        if option not in {"-s", "-if", "-gpu"}:
            options.extend([option, value])
        index += 2
    return options


def _execute_native_transcode(
    *, authentication: Any, suffix: Sequence[str], output: Path, decoded: Path,
    root: Path, branch: Path, runner: Any, timeout: float,
    intent: Mapping[str, Any], content: Mapping[str, Any], provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Launch and verify the already-prepared AppTrans-to-AppDec boundary."""
    try:
        command, stdout, _stderr = _launch(
            authentication.token("AppTrans"), suffix, branch=branch, runner=runner,
            timeout=timeout, stage="native-transcode", phase="execute",
        )
    except (OSError, ValueError) as exc:
        raise BoundaryFailure(
            "AppTrans", "AppDec", str(output), str(exc)
        ) from exc
    frame_matches = (
        NATIVE_TRANSCODE_LEGACY.findall(stdout)
        + NATIVE_TRANSCODE_CURRENT.findall(stdout)
    )
    saved_matches = NATIVE_SAVED.findall(stdout)
    if len(frame_matches) != 1 or int(frame_matches[0]) <= 0:
        raise BoundaryFailure(
            "AppTrans", "AppDec", str(output),
            "AppTrans must report exactly one positive frame marker",
        )
    bit_depth = "10" if intent["format"] in {"P010", "P210", "YUV444_16BIT"} else "8"
    if saved_matches != [(str(output), bit_depth)]:
        raise BoundaryFailure(
            "AppTrans", "AppDec", str(output),
            "AppTrans saved-output marker is missing or mismatched",
        )
    frames = int(frame_matches[0])
    encoded = artifact_io.snapshot_artifact(
        root, output, schema_version="1", kind="nvcodec-hevc-bitstream"
    )
    if encoded["size_bytes"] <= 0:
        raise BoundaryFailure("AppTrans", "AppDec", str(output), "AppTrans output is empty")
    artifact_io.verify_external_artifact(content)
    artifact_io.verify_artifact(root, encoded)
    try:
        decode = _decode(
            "native", authentication, output, decoded, root=root, branch=branch,
            runner=runner, timeout=timeout, gpu=int(intent["gpu"]),
            expected_frames=frames,
        )
    except (OSError, ValueError) as exc:
        raise BoundaryFailure(
            "AppTrans", "AppDec", str(output), str(exc)
        ) from exc
    artifact_io.verify_artifact(root, encoded)
    return {
        "status": "complete", "surface": "native", "provenance": provenance,
        "producer": {"sample": "AppTrans", "command": command, "frames": frames, "output": encoded},
        "consumer": {"sample": "AppDec", **decode, "consumed_sha256": encoded["sha256"]},
        "boundaries": [_boundary("AppTrans", "AppDec", encoded, status="verified")],
    }


def _protected_data(token: Any, filename: str) -> Path:
    matches: list[Path] = []
    for seal in token.protected_files:
        identity = seal.identity
        path = artifact_io.verify_external_artifact(identity)
        if path.name == filename:
            matches.append(path)
    if len(matches) != 1:
        raise PipelineError(f"authenticated sample must bind exactly one {filename}")
    return matches[0]


def _segment_rows(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"cannot read authenticated segments.txt: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise PipelineError(f"segments.txt line {number} must have two values")
        try:
            start, end = (float(item) for item in fields)
        except ValueError as exc:
            raise PipelineError(f"segments.txt line {number} is not numeric") from exc
        if not all(math.isfinite(item) for item in (start, end)) or start < 0 or start >= end:
            raise PipelineError(f"segments.txt line {number} has an invalid range")
        rows.append((start, end))
    if not rows:
        raise PipelineError("authenticated segments.txt is empty")
    return rows


def _record_unmaterialized_segment_failure(
    outputs: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    *,
    expected_path: str,
    reason: str,
) -> None:
    outputs.append(
        {
            "expected_segment_path": expected_path,
            "status": "operation_failed",
            "reason": reason,
        }
    )
    boundaries.append(
        _boundary(
            "create_video_segments.py",
            "advanced/decode.py",
            {"path": expected_path},
            status="failed",
            reason=reason,
        )
    )


def _run_pynvc_segments(
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    content = _external(request.get("input"), label="input video")
    input_path = artifact_io.verify_external_artifact(content)
    gpu = request.get("gpu", 0)
    if type(gpu) is not int or gpu < 0:  # pylint: disable=unidiomatic-typecheck
        raise PipelineError("gpu must be a non-negative integer")
    _environment_identity, environment = _environment(
        request, required_surfaces=("pynvc",), selected_gpu=gpu
    )
    _bind_environment_gpu(environment, gpu, label="request")
    branch = artifact_io.create_private_workspace(root / "pynvc-segments")
    samples = ("samples/basic/create_video_segments.py", "samples/advanced/decode.py")
    authentication, provenance = _authenticate(
        "pynvc", samples, root=root, branch=branch, environment=environment,
        runner=runner, timeout=timeout,
    )
    segment_token = authentication.token(samples[0])
    schedule = _protected_data(segment_token, "segments.txt")
    configuration = _protected_data(segment_token, "transcode_config.json")
    rows = _segment_rows(schedule)
    template = branch / "segment.mp4"
    expected = [
        branch / f"segment_{start:g}_{end:g}.mp4" for start, end in rows
    ]
    if any(path.exists() for path in expected):
        raise PipelineError("segment outputs must be fresh")
    suffix = [
        "-i", str(input_path), "-s", str(schedule), "-c", str(configuration),
        "-o", str(template), "-g", str(gpu),
    ]
    command, stdout, _stderr = _launch(
        segment_token, suffix, branch=branch, runner=runner, timeout=timeout,
        stage="pynvc-segment", phase="execute",
    )
    summary = SEGMENT_SUMMARY.findall(stdout)
    created = SEGMENT_CREATED.findall(stdout)
    if summary != [str(len(rows))] or created != [str(path) for path in expected]:
        raise BoundaryFailure(
            "create_video_segments.py", "advanced/decode.py", str(template),
            "segment markers do not exactly match authenticated segments.txt rows",
        )
    outputs: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    verified = 0
    for index, segment in enumerate(expected, 1):
        try:
            identity = artifact_io.snapshot_artifact(
                root, segment, schema_version="1", kind="nvcodec-video-segment"
            )
        except (OSError, ValueError) as exc:
            expected_path = segment.relative_to(root).as_posix()
            _record_unmaterialized_segment_failure(
                outputs,
                boundaries,
                expected_path=expected_path,
                reason=str(exc),
            )
            continue
        if identity["size_bytes"] <= 0:
            boundaries.append(
                _boundary(
                    "create_video_segments.py", "advanced/decode.py", identity,
                    status="failed", reason="segment is empty",
                )
            )
            outputs.append({"segment": identity, "status": "operation_failed"})
            continue
        try:
            artifact_io.verify_artifact(root, identity)
            decode = _decode(
                "pynvc", authentication, segment, branch / f"segment-{index}.yuv",
                root=root, branch=branch, runner=runner, timeout=timeout, gpu=gpu,
            )
            artifact_io.verify_artifact(root, identity)
            decode["consumed_sha256"] = identity["sha256"]
            outputs.append({"segment": identity, "status": "operation_verified", "decode": decode})
            boundaries.append(
                _boundary(
                    "create_video_segments.py", "advanced/decode.py", identity,
                    status="verified",
                )
            )
            verified += 1
        except (OSError, ValueError) as exc:
            outputs.append({"segment": identity, "status": "operation_failed", "reason": str(exc)})
            boundaries.append(
                _boundary(
                    "create_video_segments.py", "advanced/decode.py", identity,
                    status="failed", reason=str(exc),
                )
            )
    artifact_io.verify_external_artifact(content)
    return {
        "status": (
            "complete" if verified == len(expected)
            else ("partial" if verified else "failed")
        ),
        "surface": "pynvc", "provenance": provenance,
        "schedule": artifact_io.snapshot_external_artifact(
            schedule, schema_version="1", kind="pynvc-segment-schedule"
        ),
        "producer": {"sample": samples[0], "command": command, "segments": len(expected)},
        "segments": outputs, "boundaries": boundaries,
    }


def _target_gate(request: Mapping[str, Any]) -> tuple[bool, list[str]]:
    value = request.get("target_eligibility")
    if not isinstance(value, Mapping) or set(value) != {"eligible", "reasons"}:
        raise PipelineError("target_eligibility must contain exactly eligible and reasons")
    eligible = value.get("eligible")
    reasons = value.get("reasons")
    if not isinstance(eligible, bool) or not isinstance(reasons, list) or not all(
        isinstance(reason, str) and reason for reason in reasons
    ):
        raise PipelineError("target_eligibility fields are invalid")
    if eligible and reasons:
        raise PipelineError("eligible target must not contain failure reasons")
    if not eligible and not reasons:
        reasons = ["target is not an eligible released Jetson route"]
    return eligible, list(reasons)


def _retrieve(
    request: Mapping[str, Any], *, root: Path, branch: Path, runner: Any,
    timeout: float,
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    supplied = request.get("input")
    if supplied is not None:
        identity = _external(supplied, label="container input")
        return identity, artifact_io.verify_external_artifact(identity), None
    url = _exact_http_url(
        request.get("source_url"), label="container source_url"
    )
    name = request.get("content_name", "content.mp4")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise PipelineError("content_name must be one plain filename")
    destination = branch / name
    curl = Path("/usr/bin/curl")
    argv = [
        str(curl), "--fail", "--silent", "--show-error",
        "--output", str(destination), "--write-out", "%{http_code}", url,
    ]
    result = runner.run(
        argv, cwd=branch, env={
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }, timeout_seconds=timeout, stage="content-retrieval", phase="prepare",
    )
    command, stdout, _stderr = _command(
        result, runner=runner, argv=argv, cwd=branch, timeout=timeout,
        stage="content-retrieval", phase="prepare",
    )
    if re.fullmatch(r"2[0-9]{2}", stdout.strip()) is None:
        raise PipelineError(
            "content retrieval must complete without an HTTP redirect"
        )
    identity = artifact_io.snapshot_artifact(
        root, destination, schema_version="1", kind="nvcodec-video-content"
    )
    if identity["size_bytes"] <= 0:
        raise PipelineError("retrieved content is empty")
    return identity, destination, command


def _run_container_triage(
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    eligible, reasons = _target_gate(request)
    if not eligible:
        return {
            "status": "blocked", "gate": "target_eligibility", "reasons": reasons,
            "retrieval_attempted": False, "operations": {}, "boundaries": [],
        }
    surface = request.get("surface")
    if not isinstance(surface, str) or surface not in {"native", "pynvc"}:
        raise PipelineError("container_triage surface must be native or pynvc")
    gpu = request.get("gpu", 0)
    if type(gpu) is not int or gpu < 0:  # pylint: disable=unidiomatic-typecheck
        raise PipelineError("gpu must be a non-negative integer")
    _environment_identity, environment = _environment(
        request, required_surfaces=(surface,), selected_gpu=gpu
    )
    _bind_environment_gpu(environment, gpu, label="request")
    branch = artifact_io.create_private_workspace(root / "container-triage")
    sample = "AppDec" if surface == "native" else "samples/advanced/decode.py"
    authentication, provenance = _authenticate(
        surface, (sample,), root=root, branch=branch, environment=environment,
        runner=runner, timeout=timeout,
    )
    content, path, retrieval = _retrieve(
        request, root=root, branch=branch, runner=runner, timeout=timeout
    )
    output = branch / "decoded.yuv"
    try:
        decode = _decode(
            surface, authentication, path, output, root=root, branch=branch,
            runner=runner, timeout=timeout, gpu=gpu,
        )
        if Path(content["path"]).is_absolute():
            artifact_io.verify_external_artifact(content)
        else:
            artifact_io.verify_artifact(root, content)
        decode["consumed_sha256"] = content["sha256"]
        raw_video = artifact_io.snapshot_external_artifact(
            output, schema_version="1", kind="nvcodec-raw-video"
        )
        artifact_io.verify_external_artifact(raw_video)
        decode["raw_video"] = raw_video
        boundary = _boundary(sample, sample, content, status="verified")
        boundary["producer"] = f"{sample} intrinsic libavformat demux"
        boundary["consumer"] = f"{sample} NVDEC decode"
        return {
            "status": "complete", "gate": "decode",
            "retrieval_attempted": retrieval is not None,
            "retrieval": retrieval, "content": content, "provenance": provenance,
            "capability_query": request.get("decoder_capability"),
            "demux": {
                "sample": sample, "implementation": "intrinsic-libavformat",
                "status": "operation_verified",
            },
            "decode": decode, "boundaries": [boundary],
        }
    except (OSError, ValueError) as exc:
        boundary = _boundary(sample, sample, content, status="failed", reason=str(exc))
        boundary["producer"] = f"{sample} intrinsic libavformat demux"
        boundary["consumer"] = f"{sample} NVDEC decode"
        return {
            "status": "failed", "gate": "container_or_decode",
            "retrieval_attempted": retrieval is not None,
            "retrieval": retrieval, "content": content, "provenance": provenance,
            "capability_query": request.get("decoder_capability"),
            "reason": str(exc), "boundaries": [boundary],
        }


def _ivf(
    path: Path, *, width: int, height: int, expected_frames: int
) -> dict[str, Any]:
    try:
        structure = _encode_controller()._structure(  # pylint: disable=protected-access
            path, "av1", width, height, expected_frames
        )
    except (OSError, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    return {
        **structure,
        "walked_frames": structure["walked_frame_count"],
        "exact_eof": True,
    }


def _run_av1_verify(
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    _recipe_identity, recipe = _recipe(request)
    raw = _external(
        request.get("input"), label="raw input", schema="1", kinds=("nvcodec-raw-video",)
    )
    raw_path = artifact_io.verify_external_artifact(raw)
    intent = recipe["encoder_intent"]
    _environment_identity, environment = _environment(
        request,
        required_surfaces=("native",),
        selected_gpu=intent.get("gpu"),
    )
    _bind_environment_gpu(environment, intent.get("gpu"), label="recipe")
    frame_count = intent.get("frame_count")
    if (
        intent.get("codec") != "av1"
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise PipelineError("av1_verify requires an AV1 recipe with positive frame_count")
    if recipe["projections"]["native"]["status"] != "exact":
        raise PipelineError("av1_verify requires an exact native projection")
    branch = artifact_io.create_private_workspace(root / "av1-verify")
    authentication, provenance = _authenticate(
        "native", ("AppEncCuda", "AppDec"), root=root, branch=branch,
        environment=environment, runner=runner, timeout=timeout,
    )
    operations: dict[str, Any] = {}
    boundaries: list[dict[str, Any]] = []
    verified = 0
    for mode, flag in (("host", "0"), ("vidmem", "1")):
        output = branch / f"av1-{mode}.av1"
        decoded = branch / f"av1-{mode}.yuv"
        suffix = [
            "-i", str(raw_path), "-o", str(output),
            *recipe["projections"]["native"]["cli_options"],
            "-outputInVidMem", flag,
        ]
        try:
            command, stdout, _stderr = _launch(
                authentication.token("AppEncCuda"), suffix, branch=branch,
                runner=runner, timeout=timeout, stage=f"av1-{mode}-encode",
                phase="execute",
            )
            markers = NATIVE_ENCODE.findall(stdout)
            if markers != [str(frame_count)]:
                raise PipelineError("AppEncCuda did not report the exact recipe frame count")
            structure = _ivf(
                output,
                width=int(intent["width"]),
                height=int(intent["height"]),
                expected_frames=frame_count,
            )
            if structure["walked_frames"] != frame_count:
                raise PipelineError("walked IVF frame count differs from AppEncCuda")
            encoded = artifact_io.snapshot_artifact(
                root, output, schema_version="1", kind="nvcodec-av1-bitstream"
            )
            artifact_io.verify_artifact(root, encoded)
            decode = _decode(
                "native", authentication, output, decoded, root=root, branch=branch,
                runner=runner, timeout=timeout, gpu=int(intent["gpu"]),
                expected_frames=frame_count,
            )
            artifact_io.verify_artifact(root, encoded)
            decode["consumed_sha256"] = encoded["sha256"]
            operations[mode] = {
                "status": "operation_verified", "encode": {
                    "command": command, "frames": frame_count, "output": encoded,
                    "structure": structure,
                }, "decode": decode,
            }
            boundaries.append(_boundary("AppEncCuda", "AppDec", encoded, status="verified"))
            verified += 1
        except (OSError, ValueError) as exc:
            operations[mode] = {"status": "operation_failed", "reason": str(exc)}
            boundaries.append({
                "producer": "AppEncCuda", "consumer": "AppDec",
                "artifact_path": str(output), "artifact_size_bytes": None,
                "artifact_sha256": None, "status": "failed", "reason": str(exc),
            })
    artifact_io.verify_external_artifact(raw)
    return {
        "status": "complete" if verified == 2 else ("partial" if verified else "failed"),
        "surface": "native", "provenance": provenance,
        "operations": operations, "boundaries": boundaries,
    }


def _jsonl_values(
    identity: Mapping[str, Any], label: str,
) -> tuple[list[dict[str, Any]], bytes]:
    path = artifact_io.verify_external_artifact(identity)
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise PipelineError(f"cannot read {label}: {exc}") from exc
    values = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = artifact_io.strict_json_loads(line)
        if not isinstance(value, dict):
            raise PipelineError(f"{label} line {number} is not a JSON object")
        values.append(artifact_io.canonical_json_bytes(value).rstrip(b"\n"))
    if not values:
        raise PipelineError(f"{label} contains no JSON objects")
    payload = b"\n".join(values) + b"\n"
    return [artifact_io.strict_json_loads(item) for item in values], payload


def _jsonl(identity: Mapping[str, Any], label: str) -> bytes:
    return _jsonl_values(identity, label)[1]


def _stage_object(refs: Mapping[str, Mapping[str, Any]], name: str) -> dict[str, Any]:
    value = artifact_io.read_verified_external_json(refs[name])
    if not isinstance(value, dict):
        raise PipelineError(f"acceptance stage {name} must contain one JSON object")
    return value


def _complete_stage(value: Mapping[str, Any], name: str) -> None:
    if value.get("status") not in {"complete", "completed", "operation_verified", "ready"}:
        raise PipelineError(f"acceptance stage {name} is not complete")


def _content_stage(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = value.get("representative_content", value)
    if not isinstance(record, Mapping):
        raise PipelineError("acceptance content evidence is malformed")
    if record is value:
        if value.get("kind") != "representative-content" or value.get("verified") is not True:
            raise PipelineError("acceptance content evidence is not verified")
    else:
        _complete_stage(value, "content")
    required = set(REPRESENTATIVE_CONTENT_FIELDS)
    if not required.issubset(record):
        raise PipelineError(
            "acceptance content evidence omits source URL field, license, attribution, "
            "path, size, or SHA-256"
        )
    source_url = record["source_url"]
    if source_url is not None:
        _exact_http_url(
            source_url, label="acceptance content source_url"
        )
    if any(
        not isinstance(record[field], str)
        or not record[field]
        or record[field] != record[field].strip()
        for field in ("license", "attribution")
    ):
        raise PipelineError("acceptance content license and attribution must be nonempty")
    if (
        isinstance(record["size_bytes"], bool)
        or not isinstance(record["size_bytes"], int)
        or record["size_bytes"] <= 0
    ):
        raise PipelineError("acceptance content size_bytes must be positive")
    if (
        not isinstance(record["path"], str)
        or not isinstance(record["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise PipelineError("acceptance content path and SHA-256 are invalid")
    observed = artifact_io.snapshot_external_artifact(
        Path(record["path"]), schema_version="1", kind="nvcodec-video-content"
    )
    if (observed["path"], observed["size_bytes"], observed["sha256"]) != (
        record["path"], record["size_bytes"], record["sha256"],
    ):
        raise PipelineError("acceptance content path, size, or SHA-256 is stale")
    expected_size = record.get("expected_size_bytes", record["size_bytes"])
    expected_sha = record.get("expected_sha256", record["sha256"])
    valid_expected_size = (
        not isinstance(expected_size, bool)
        and isinstance(expected_size, int)
        and expected_size == record["size_bytes"]
    )
    valid_expected_sha = (
        isinstance(expected_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None
        and expected_sha == record["sha256"]
    )
    if not valid_expected_size or not valid_expected_sha:
        raise PipelineError("acceptance content expected identity differs from observed identity")
    metadata = {field: record[field] for field in REPRESENTATIVE_CONTENT_FIELDS}
    source = {
        "kind": "representative-content",
        **metadata,
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha,
        "verified": True,
    }
    return metadata, source


def _recipe_stage(
    value: Mapping[str, Any],
    preset: str,
    identity: Mapping[str, Any],
) -> None:
    try:
        artifact_io.validate_recipe_identity(identity)
    except artifact_io.SkillDependencyError:
        raise
    except ValueError as exc:
        raise PipelineError(f"acceptance {preset} recipe is invalid: {exc}") from exc
    observed = value.get("encoder_intent", {}).get("preset")
    if not isinstance(observed, str) or observed.lower() != preset:
        raise PipelineError(f"acceptance {preset} recipe does not select {preset.upper()}")


def _workspace_identity(
    value: Any,
    workspace: Path,
    *,
    label: str,
    kinds: Sequence[str],
    schema: str = "1",
    owner: Path | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != IDENTITY_KEYS
        or value.get("schema_version") != schema
        or value.get("kind") not in kinds
    ):
        raise PipelineError(f"{label} must be one exact artifact identity")
    try:
        artifact_path = artifact_io.verify_artifact(workspace, value)
        if owner is not None:
            artifact_path.relative_to(artifact_io.resolve_private_workspace(owner))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"{label} is not current: {exc}") from exc
    return dict(value)


def _stage_workspace(
    value: Mapping[str, Any],
    identity: Mapping[str, Any],
    preset: str,
) -> tuple[Path, Path]:
    child = value.get("encode_workspace")
    if not isinstance(child, str):
        raise PipelineError(f"acceptance {preset} encode workspace is missing")
    try:
        child_path = artifact_io.resolve_private_workspace(Path(child))
        root = artifact_io.resolve_private_workspace(child_path.parent)
        result_path = artifact_io.verify_external_artifact(identity)
        result_path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"acceptance {preset} encode workspace/result ownership is invalid: {exc}"
        ) from exc
    if child_path != root / "encode-decode":
        raise PipelineError(f"acceptance {preset} encode workspace is inconsistent")
    return root, child_path


def _exact_boundary(
    value: Any,
    *,
    producer: str,
    consumer: str,
    identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected = _boundary(producer, consumer, identity, status="verified")
    if (
        not isinstance(value, Mapping)
        or set(value) != BOUNDARY_KEYS
        or dict(value) != expected
    ):
        raise PipelineError(f"{label} does not match its exact encoded artifact")
    return expected


def _encode_decode_stage(
    value: Mapping[str, Any],
    preset: str,
    *,
    stage_identity: Mapping[str, Any],
    recipe_identity: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        value.get("schema_version") != "1.0"
        or value.get("kind") != "nvcodec-pipeline-result"
        or value.get("route") != "encode_decode"
    ):
        raise PipelineError(f"acceptance {preset} encode/decode envelope is invalid")
    _complete_stage(value, f"{preset}_encode_decode")
    root, encode_workspace = _stage_workspace(value, stage_identity, preset)
    encode_request = _external(
        value.get("encode_request"),
        label=f"acceptance {preset} encode request",
        schema="1.0",
        kinds=("nvcodec-encode-request",),
    )
    request_value = artifact_io.read_verified_external_json(encode_request)
    if not isinstance(request_value, Mapping):
        raise PipelineError(f"acceptance {preset} encode request is invalid")
    if request_value.get("recipe") != dict(recipe_identity):
        raise PipelineError(
            f"acceptance {preset} encode request did not consume its accepted recipe"
        )
    input_identity = _external(
        request_value.get("input"),
        label=f"acceptance {preset} encode input",
        schema="1",
        kinds=("nvcodec-raw-video",),
    )
    try:
        _encode_controller()._validate_raw(  # pylint: disable=protected-access
            recipe, input_identity
        )
    except ValueError as exc:
        raise PipelineError(
            f"acceptance {preset} encode input differs from its recipe: {exc}"
        ) from exc
    _workspace_identity(
        value.get("encode_result"),
        root,
        label=f"acceptance {preset} nested encode result",
        kinds=("nvcodec-encode-result",),
        schema="1.0",
        owner=encode_workspace,
    )
    operations = value.get("operations")
    if not isinstance(operations, Mapping) or not operations:
        raise PipelineError(f"acceptance {preset} encode/decode operations are missing")
    selected = value.get("selected_surfaces")
    if not isinstance(selected, list) or set(selected) != set(operations):
        raise PipelineError(
            f"acceptance {preset} selected surfaces and operations differ"
        )
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != len(selected):
        raise PipelineError(f"acceptance {preset} boundaries are incomplete")
    by_producer = {
        item.get("producer"): item
        for item in boundaries
        if isinstance(item, Mapping)
    }
    if len(by_producer) != len(boundaries):
        raise PipelineError(f"acceptance {preset} boundaries have duplicate producers")
    verified_boundaries = []
    expected_frames = recipe["encoder_intent"].get("frame_count")
    for surface in selected:
        operation = operations[surface]
        if (
            surface not in {"native", "pynvc"}
            or not isinstance(operation, Mapping)
            or set(operation) != {
                "status", "encoded", "decoded", "encode_frames", "decode_frames",
            }
            or operation.get("status") != "operation_verified"
        ):
            raise PipelineError(
                f"acceptance {preset} {surface} encode/decode is not operation_verified"
            )
        encoded = _workspace_identity(
            operation.get("encoded"),
            root,
            label=f"acceptance {preset} {surface} encoded artifact",
            kinds=(f"nvcodec-{recipe['encoder_intent']['codec']}-bitstream",),
            owner=encode_workspace,
        )
        decoded = _workspace_identity(
            operation.get("decoded"),
            root,
            label=f"acceptance {preset} {surface} decoded artifact",
            kinds=("nvcodec-decoded-frames",),
            owner=encode_workspace,
        )
        if (
            encoded["size_bytes"] <= 0
            or decoded["size_bytes"] <= 0
            or operation.get("encode_frames") != expected_frames
            or operation.get("decode_frames") != expected_frames
        ):
            raise PipelineError(
                f"acceptance {preset} encode/decode outputs or frame counts are invalid"
            )
        verified_boundaries.append(_exact_boundary(
            by_producer.get(f"{surface}-encode"),
            producer=f"{surface}-encode",
            consumer=f"{surface}-decode",
            identity=encoded,
            label=f"acceptance {preset} {surface} boundary",
        ))
    return {"input": input_identity, "boundaries": verified_boundaries}


def _pynvc_benchmark_config(
    branch: Mapping[str, Any],
    arguments: Sequence[str],
    recipe: Mapping[str, Any],
    preset: str,
) -> None:
    """Bind the exact -json operand to the accepted recipe projection."""
    option_index = arguments.index("-json")
    if option_index + 1 >= len(arguments):
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark -json option lacks its value"
        )
    config_operand = Path(arguments[option_index + 1])
    if not config_operand.is_absolute():
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark config path is not absolute"
        )
    try:
        config_workspace = artifact_io.resolve_private_workspace(config_operand.parent)
        config_identity = _workspace_identity(
            branch.get("generated_config"),
            config_workspace,
            label=f"acceptance {preset} pynvc generated config",
            kinds=("pynvc-encoder-config",),
        )
        config_path = artifact_io.verify_artifact(config_workspace, config_identity)
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark config is invalid: {exc}"
        ) from exc
    if (
        config_identity["path"] != config_operand.name
        or config_path != config_operand
    ):
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark config differs from -json operand"
        )
    projection = recipe.get("projections", {}).get("pynvc", {})
    if (
        not isinstance(projection, Mapping)
        or projection.get("status") != "exact"
    ):
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark has no exact accepted projection"
        )
    expected_config = projection.get("config")
    try:
        config_matches = (
            isinstance(expected_config, Mapping)
            and artifact_io.read_verified_bytes(config_workspace, config_identity)
            == artifact_io.canonical_json_bytes(dict(expected_config))
        )
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark config is stale: {exc}"
        ) from exc
    if not config_matches:
        raise PipelineError(
            f"acceptance {preset} pynvc benchmark config differs from accepted recipe"
        )


def _benchmark_recipe_option(
    surface: str,
    branch: Mapping[str, Any],
    arguments: Sequence[str],
    recipe: Mapping[str, Any],
    preset: str,
) -> None:
    option = "-preset" if surface == "native" else "-json"
    if arguments.count(option) != 1:
        raise PipelineError(
            f"acceptance {preset} {surface} benchmark must contain "
            f"one {option} recipe option"
        )
    if surface == "pynvc":
        _pynvc_benchmark_config(branch, arguments, recipe, preset)
        return
    option_index = arguments.index(option)
    if (
        option_index + 1 >= len(arguments)
        or arguments[option_index + 1].lower() != preset
    ):
        raise PipelineError(
            f"acceptance {preset} native benchmark preset differs from recipe"
        )


def _benchmark_stage(  # pylint: disable=too-many-branches
    value: Mapping[str, Any],
    preset: str,
    *,
    recipe_identity: Mapping[str, Any],
    recipe: Mapping[str, Any],
    input_identity: Mapping[str, Any],
) -> None:
    if (
        value.get("kind") != "nvcodec-benchmark-result"
        or value.get("status") != "completed"
    ):
        raise PipelineError(f"acceptance {preset} benchmark is not completed")
    variants = value.get("variants")
    matching = [
        item for item in variants if isinstance(item, Mapping)
        and str(item.get("name", "")).lower() == preset
    ] if isinstance(variants, list) else []
    if len(matching) != 1:
        raise PipelineError(
            f"acceptance benchmark must contain exactly one {preset.upper()} variant"
        )
    if matching[0].get("recipe") != dict(recipe_identity):
        raise PipelineError(
            f"acceptance {preset} benchmark did not consume its accepted recipe"
        )
    _external(
        matching[0]["recipe"],
        label=f"acceptance {preset} benchmark recipe",
        schema="2.0",
        kinds=("nvcodec-recipe",),
    )
    surfaces = matching[0].get("surfaces")
    if not isinstance(surfaces, Mapping) or not surfaces:
        raise PipelineError(f"acceptance {preset} benchmark surfaces are missing")
    if any(surface not in {"native", "pynvc"} for surface in surfaces):
        raise PipelineError(f"acceptance {preset} benchmark contains an unknown surface")
    for surface, branch in surfaces.items():
        if not isinstance(branch, Mapping) or branch.get("status") != "completed":
            raise PipelineError(f"acceptance {preset} {surface} benchmark is incomplete")
        input_record = branch.get("input")
        if (
            not isinstance(input_record, Mapping)
            or input_record.get("identity") != dict(input_identity)
        ):
            raise PipelineError(
                f"acceptance {preset} {surface} benchmark did not consume "
                "the encode/decode input"
            )
        _external(
            input_record["identity"],
            label=f"acceptance {preset} {surface} benchmark input",
            schema="1",
            kinds=("nvcodec-raw-video",),
        )
        intent = recipe["encoder_intent"]
        for key in ("width", "height", "frame_count", "fps", "codec", "format"):
            observed_key = "frames" if key == "frame_count" else key
            if input_record.get(observed_key) != intent.get(key):
                raise PipelineError(
                    f"acceptance {preset} {surface} benchmark input differs "
                    "from the recipe"
                )
        authority = branch.get("sample_authority")
        launcher = authority.get("launcher") if isinstance(authority, Mapping) else None
        arguments = branch.get("benchmark_arguments")
        if (
            not isinstance(launcher, list)
            or not launcher
            or not isinstance(arguments, list)
            or any(not isinstance(item, str) or "\0" in item for item in [*launcher, *arguments])
        ):
            raise PipelineError(
                f"acceptance {preset} {surface} benchmark exact argv is missing"
            )
        _benchmark_recipe_option(surface, branch, arguments, recipe, preset)
        expected_argv = [*launcher, *arguments]
        measured = branch.get("measured_runs")
        if not isinstance(measured, list) or len(measured) < 3:
            raise PipelineError(
                f"acceptance {preset} {surface} benchmark lacks warmup, measurements, or summary"
            )
        _benchmark_command(
            branch.get("warmup"), "warmup", expected_argv,
            f"acceptance {preset} {surface} warmup",
        )
        fps_values: list[float] = []
        megapixel_values: list[float] = []
        for index, run in enumerate(measured, 1):
            if not isinstance(run, Mapping) or run.get("repetition") != index:
                raise PipelineError(
                    f"acceptance {preset} {surface} benchmark repetitions are not sequential"
                )
            fps_values.append(_positive_finite(run.get("fps"), "FPS", preset, surface))
            megapixel_values.append(
                _positive_finite(
                    run.get("megapixels_per_second"), "megapixels/second", preset, surface
                )
            )
            _benchmark_command(
                run.get("command"), "measure", expected_argv,
                f"acceptance {preset} {surface} measurement {index}",
            )
        _benchmark_summary(
            branch.get("summary"), fps_values, megapixel_values, preset, surface
        )


def _positive_finite(value: Any, label: str, preset: str, surface: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(
            f"acceptance {preset} {surface} benchmark {label} must be positive and finite"
        )
    observed = float(value)
    if not math.isfinite(observed) or observed <= 0:
        raise PipelineError(
            f"acceptance {preset} {surface} benchmark {label} must be positive and finite"
        )
    return observed


def _benchmark_command(
    value: Any, phase: str, expected_argv: Sequence[str], label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != COMMAND_KEYS:
        raise PipelineError(f"{label} must retain one exact command result")
    argv = value.get("argv")
    exit_code = value.get("exit_code")
    successful_exit = (
        isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code == 0
    )
    successful = (
        value.get("schema_version") == "1"
        and value.get("kind") == "command-result"
        and value.get("phase") == phase
        and isinstance(argv, list)
        and argv == list(expected_argv)
        and bool(argv)
        and all(isinstance(item, str) and "\0" not in item for item in argv)
        and successful_exit
        and value.get("timed_out") is False
        and value.get("launch_error") is None
    )
    if not successful:
        raise PipelineError(
            f"{label} command phase, argv, exit code, or timeout evidence is invalid"
        )


def _benchmark_summary(
    value: Any,
    fps_values: Sequence[float],
    megapixel_values: Sequence[float],
    preset: str,
    surface: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("repetitions") != len(fps_values):
        raise PipelineError(f"acceptance {preset} {surface} benchmark summary is inconsistent")
    for key, observed_values in (
        ("fps", fps_values), ("megapixels_per_second", megapixel_values),
    ):
        metrics = value.get(key)
        if not isinstance(metrics, Mapping) or set(metrics) != {"mean", "minimum", "maximum"}:
            raise PipelineError(
                f"acceptance {preset} {surface} benchmark {key} summary is incomplete"
            )
        expected = {
            "mean": sum(observed_values) / len(observed_values),
            "minimum": min(observed_values),
            "maximum": max(observed_values),
        }
        for name, expected_value in expected.items():
            actual = _positive_finite(metrics.get(name), f"summary {key}.{name}", preset, surface)
            if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise PipelineError(
                    f"acceptance {preset} {surface} benchmark {key} summary is inconsistent"
                )


def _verified_boundaries(
    value: Mapping[str, Any],
    name: str,
    expected: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise PipelineError(f"acceptance {name} boundaries are missing")
    if any(
        not isinstance(boundary, Mapping)
        or set(boundary) != BOUNDARY_KEYS
        or boundary.get("status") != "verified"
        for boundary in boundaries
    ):
        raise PipelineError(f"acceptance {name} contains an unverified boundary")
    if expected is not None:
        def canonical(item: Mapping[str, Any]) -> str:
            return json.dumps(
                dict(item), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )

        if Counter(map(canonical, boundaries)) != Counter(map(canonical, expected)):
            raise PipelineError(
                "acceptance handoffs differ from the freshly verified "
                "encode/decode boundaries"
            )


def _task_results(
    identity: Mapping[str, Any], refs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bytes]:
    rows, payload = _jsonl_values(identity, "task_results")
    by_task: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        if set(row) != {"task_id", "status", "evidence"}:
            raise PipelineError(
                f"task_results line {number} fields must be task_id, status, evidence"
            )
        task_id = row["task_id"]
        evidence = row["evidence"]
        if task_id not in ACCEPTANCE_TASKS or task_id in by_task:
            raise PipelineError(f"task_results line {number} has duplicate or unknown task_id")
        if row["status"] != "complete":
            raise PipelineError(f"task_results task {task_id} is not complete")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(name, str) or name not in refs for name in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            raise PipelineError(f"task_results task {task_id} has invalid evidence names")
        by_task[task_id] = row
    missing = sorted(set(ACCEPTANCE_TASKS) - set(by_task))
    if missing:
        raise PipelineError(f"task_results is missing required tasks {missing}")
    return rows, payload


def _validate_acceptance_stages(
    refs: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], bytes,
    dict[str, Any], dict[str, Any],
]:
    values = {name: _stage_object(refs, name) for name in ACCEPTANCE_TASKS}
    _complete_stage(values["readiness"], "readiness")
    _complete_stage(values["capabilities"], "capabilities")
    content_metadata, content_source = _content_stage(values["content"])
    recipes = {
        preset: values[f"{preset}_recipe"] for preset in ("p4", "p5")
    }
    for preset in ("p4", "p5"):
        _recipe_stage(recipes[preset], preset, refs[f"{preset}_recipe"])
    p4_intent = dict(recipes["p4"]["encoder_intent"])
    p5_intent = dict(recipes["p5"]["encoder_intent"])
    p4_intent.pop("preset", None)
    p5_intent.pop("preset", None)
    if p4_intent != p5_intent:
        raise PipelineError("acceptance P4/P5 recipes must differ only by preset")
    encode_results = {
        preset: _encode_decode_stage(
            values[f"{preset}_encode_decode"],
            preset,
            stage_identity=refs[f"{preset}_encode_decode"],
            recipe_identity=refs[f"{preset}_recipe"],
            recipe=recipes[preset],
        )
        for preset in ("p4", "p5")
    }
    if encode_results["p4"]["input"] != encode_results["p5"]["input"]:
        raise PipelineError("acceptance P4/P5 inputs differ")
    if any(
        content_source[key] != encode_results["p4"]["input"][key]
        for key in ("path", "size_bytes", "sha256")
    ):
        raise PipelineError(
            "acceptance content is not the exact raw artifact consumed by encode"
        )
    expected_boundaries = []
    for preset in ("p4", "p5"):
        expected_boundaries.extend(encode_results[preset]["boundaries"])
        _benchmark_stage(
            values[f"{preset}_benchmark"],
            preset,
            recipe_identity=refs[f"{preset}_recipe"],
            recipe=recipes[preset],
            input_identity=encode_results[preset]["input"],
        )
    _complete_stage(values["handoffs"], "handoffs")
    _verified_boundaries(
        values["handoffs"], "handoffs", expected=expected_boundaries
    )
    rows, task_payload = _task_results(refs["task_results"], refs)
    return values, rows, task_payload, content_metadata, content_source


def _reference_records(refs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name, "path": identity["path"],
            "size_bytes": identity["size_bytes"], "sha256": identity["sha256"],
            "schema_version": identity["schema_version"], "kind": identity["kind"],
        }
        for name, identity in sorted(refs.items())
    ]


def _markdown(title: str, records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [f"# {title}", "", "| Evidence | Size | SHA-256 |", "|---|---:|---|"]
    lines.extend(
        f"| `{item['name']}` | {item['size_bytes']} | `{item['sha256']}` |"
        for item in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _run_acceptance(
    request: Mapping[str, Any], *, root: Path, runner: Any, timeout: float,
) -> dict[str, Any]:
    del runner, timeout
    raw_refs = request.get("references")
    if not isinstance(raw_refs, Mapping) or not raw_refs:
        raise PipelineError("acceptance requires a non-empty references object")
    refs = {
        name: _external(identity, label=f"references.{name}")
        for name, identity in raw_refs.items()
        if isinstance(name, str) and name
    }
    if len(refs) != len(raw_refs):
        raise PipelineError("acceptance reference names must be non-empty strings")
    required = set(ACCEPTANCE_TASKS) | set(ACCEPTANCE_LEDGER_REFERENCES)
    missing = sorted(required - set(refs))
    if missing:
        raise PipelineError(f"acceptance references are missing {missing}")
    files = _acceptance_files(request)
    validation_file = files[-1]
    (
        _stage_values, task_rows, task_payload, content_metadata, content_source,
    ) = _validate_acceptance_stages(refs)
    evidence = artifact_io.create_private_workspace(root / "evidence")
    del evidence
    records = _reference_records(refs)
    setup = [item for item in records if item["name"] in {"readiness", "capabilities", "content"}]
    handoffs = [item for item in records if "handoff" in item["name"]]
    payloads: dict[str, bytes] = {
        "report.md": _markdown("Jetson Video SDK acceptance", records),
        "capability-report.md": _markdown("Capability evidence", setup),
        "review-findings.md": _markdown("Review findings", handoffs or records),
        "task-results.jsonl": task_payload,
        "commands.jsonl": _jsonl(refs["commands"], "commands"),
    }
    timing_value = artifact_io.read_verified_external_json(refs["timing"])
    if not isinstance(timing_value, dict):
        raise PipelineError("timing reference must contain a JSON object")
    envelopes = {
        "timing.json": {
            "schema_version": "1", "kind": "nvcodec-acceptance-timing",
            "source": refs["timing"], "value": timing_value,
        },
        "evidence/summary.json": {
            "schema_version": "1", "kind": "nvcodec-acceptance-evidence",
            "representative_content": content_metadata,
        },
        "evidence/representative-content.json": content_source,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for relative in files[:-1]:
        path = root / relative
        if relative == "evidence/representative-content.json":
            outputs[relative] = artifact_io.write_fresh_bytes(
                root, path, artifact_io.canonical_json_bytes(envelopes[relative]),
                schema_version="1", kind="representative-content",
            )
        elif relative in envelopes:
            outputs[relative] = artifact_io.write_fresh_json(
                root, path, envelopes[relative]
            )
        else:
            kind = (
                "nvcodec-acceptance-jsonl" if relative.endswith(".jsonl")
                else "nvcodec-acceptance-markdown"
            )
            outputs[relative] = artifact_io.write_fresh_bytes(
                root, path, payloads[relative], schema_version="1", kind=kind
            )
    for identity in refs.values():
        artifact_io.verify_external_artifact(identity)
    if (
        (root / "evidence" / "manifest-sha256.txt").exists()
        or (root / "inventory.json").exists()
    ):
        raise PipelineError(
            "acceptance assembler must not create the sealing manifest or inventory"
        )
    return {
        "status": "complete", "references": records, "outputs": outputs,
        "validated_tasks": [row["task_id"] for row in task_rows],
        "validation_result_path": validation_file,
        "assembler_owned_files": list(files),
        "seal_pending": True, "manifest_created": False, "inventory_created": False,
        "boundaries": [],
    }


def _planned_operation(
    sample: str, arguments: Sequence[str], output: Path | None = None,
) -> dict[str, Any]:
    """Describe one exact authenticated-sample suffix without launching it."""
    value: dict[str, Any] = {
        "sample": sample, "arguments": list(arguments),
        "launcher": "resolved_by_sample_authentication",
    }
    if output is not None:
        value["output_path"] = str(output)
    return value


def _dry_run(  # pylint: disable=too-many-branches,too-many-statements,protected-access
    request: Mapping[str, Any], root: Path,
) -> dict[str, Any]:
    """Validate one route and return a read-only launch plan."""
    route = request["route"]
    planned: list[dict[str, Any]] = []
    outputs: list[str] = []
    identities: dict[str, Any] = {}
    selected_surface_plan: dict[str, Any] | None = None
    surface_evaluations: dict[str, Any] | None = None
    if route == "encode_decode":
        reference = _external(
            request.get("encode_request_ref"), label="encode_request_ref",
            schema="1.0", kinds=("nvcodec-encode-request",),
        )
        encode_request = artifact_io.read_verified_external_json(reference)
        if not isinstance(encode_request, Mapping):
            raise PipelineError("encode_request_ref must contain a JSON object")
        if (
            encode_request.get("schema_version") != "1.0"
            or encode_request.get("kind") != "nvcodec-encode-request"
        ):
            raise PipelineError("encode request has an invalid schema or kind")
        surface = encode_request.get("surface")
        if (
            not isinstance(surface, str)
            or surface not in {"native", "pynvc", "auto", "both"}
        ):
            raise PipelineError("encode request has an invalid surface")
        buffer_mode = encode_request.get("buffer_mode", "gpu")
        if (
            not isinstance(buffer_mode, str)
            or buffer_mode not in {"cpu", "gpu"}
        ):
            raise PipelineError("encode request has an invalid buffer mode")
        recipe_identity = _external(
            encode_request.get("recipe"), label="encode recipe", schema="2.0",
            kinds=("nvcodec-recipe",),
        )
        capability_identity = (
            _external(
                encode_request.get("capability_report"),
                label="encode capability report",
                schema=artifact_io.CAPABILITY_REPORT_SCHEMA_VERSION,
                kinds=(artifact_io.CAPABILITY_REPORT_KIND,),
            )
            if encode_request.get("capability_report") is not None
            else None
        )
        recipe = artifact_io.read_verified_external_json(recipe_identity)
        artifact_io.validate_recipe_identity(recipe_identity)
        intent = recipe.get("encoder_intent", {})
        recipe_gpu = intent.get("gpu") if isinstance(intent, Mapping) else None
        local_auto_without_pynvc = (
            surface == "auto"
            and encode_request.get("environment") is None
            and encode_request.get("pynvc_interpreter") is None
        )
        if surface in {"native", "pynvc"}:
            required_surfaces = (surface,)
        elif surface == "auto" and local_auto_without_pynvc:
            # An unavailable optional Py selector must not block a valid native
            # auto route. The unexamined peer is disclosed below.
            required_surfaces = ("native",)
        else:
            required_surfaces = artifact_io.SURFACE_ORDER
        local_errors: dict[str, str] = {}
        environment_identity, environment_value = _environment(
            {
                "environment": encode_request.get("environment"),
                "pynvc_interpreter": encode_request.get("pynvc_interpreter"),
                "timeout_seconds": request["timeout_seconds"],
            },
            required_surfaces=required_surfaces,
            selected_gpu=recipe_gpu,
            local_errors=local_errors,
        )
        input_identity = _external(
            encode_request.get("input"), label="encode input", schema="1",
            kinds=("nvcodec-raw-video",),
        )
        _bind_environment_gpu(
            environment_value,
            recipe_gpu,
            label="recipe",
        )
        if local_auto_without_pynvc:
            local_errors["pynvc"] = (
                "PyNvVideoCodec was not evaluated because no exact absolute "
                "pynvc_interpreter was supplied"
            )
        try:
            surface_plan, capability_details = _encode_controller()._surface_plan(
                surface,
                recipe,
                environment_value,
                buffer_mode,
                recipe_identity=recipe_identity,
                environment_identity=environment_identity,
                capability_identity=capability_identity,
                local_errors=local_errors,
            )
        except ValueError as exc:
            raise PipelineError(f"encode surface planning failed: {exc}") from exc
        if not surface_plan["selected_surfaces"]:
            return {
                "status": surface_plan["classification"],
                "surface_plan": surface_plan,
                "surface_evaluations": capability_details,
                "planned_operations": [], "planned_output_paths": [],
                "validated_identities": {
                    "encode_request": reference,
                    **(
                        {"environment": environment_identity}
                        if environment_identity is not None
                        else {}
                    ),
                },
                "no_mutation_performed": True,
            }
        selected_surface_plan = surface_plan
        surface_evaluations = capability_details
        codec = recipe["encoder_intent"]["codec"]
        raw_path = Path(input_identity["path"])
        surfaces = surface_plan["selected_surfaces"]
        for surface in surfaces:
            branch = root / "encode-decode" / surface
            encoded = branch / f"encoded.{codec}"
            decoded = branch / "decoded.yuv"
            if surface == "native":
                projection = recipe["projections"]["native"]["cli_options"]
                encode_args = ["-i", str(raw_path), "-o", str(encoded), *projection]
                decode_args = [
                    "-i", str(encoded), "-o", str(decoded), "-gpu",
                    str(recipe["encoder_intent"]["gpu"]),
                ]
                samples = ("AppEncCuda", "AppDec")
            else:
                intent = recipe["encoder_intent"]
                configuration = branch / "encode-config.json"
                encode_args = [
                    "-i", str(raw_path), "-o", str(encoded), "-s",
                    f"{intent['width']}x{intent['height']}", "-m",
                    buffer_mode, "-if", intent["format"],
                    "-f", str(intent["frame_count"]), "-g", str(intent["gpu"]),
                    "-c", codec, "-json", str(configuration),
                ]
                decode_args = [
                    "-i", str(encoded), "-o", str(decoded), "-g", str(intent["gpu"]),
                    "-d", "0", "-f", str(intent["frame_count"]),
                ]
                samples = ("samples/basic/encode.py", "samples/advanced/decode.py")
            planned.extend((
                _planned_operation(samples[0], encode_args, encoded),
                _planned_operation(samples[1], decode_args, decoded),
            ))
            outputs.extend((str(encoded), str(decoded)))
        identities["encode_request"] = reference
        if environment_identity is not None:
            identities["environment"] = environment_identity
        if capability_identity is not None:
            identities["capability_report"] = capability_identity
    elif route == "native_transcode":
        identities["recipe"], recipe = _recipe(request)
        identities["input"] = _external(request.get("input"), label="input video")
        if (
            recipe["encoder_intent"].get("codec") != "hevc"
            or recipe["projections"]["native"]["status"] != "exact"
        ):
            raise PipelineError(
                "native_transcode requires an exact HEVC native recipe"
            )
        output = root / "native-transcode" / "transcoded.hevc"
        decoded = root / "native-transcode" / "transcoded-decoded.yuv"
        intent = recipe["encoder_intent"]
        environment_identity, _environment_value = _environment(
            request,
            required_surfaces=("native",),
            selected_gpu=intent.get("gpu"),
        )
        if environment_identity is not None:
            identities["environment"] = environment_identity
        _bind_environment_gpu(_environment_value, intent.get("gpu"), label="recipe")
        transcode_args = [
            "-i", identities["input"]["path"], "-o", str(output), "-gpu",
            str(intent["gpu"]), *_transcode_options(recipe),
        ]
        planned.extend((
            _planned_operation("AppTrans", transcode_args, output),
            _planned_operation(
                "AppDec", [
                    "-i", str(output), "-o", str(decoded), "-gpu",
                    str(intent["gpu"]),
                ], decoded,
            ),
        ))
        outputs.extend((str(output), str(decoded)))
    elif route == "pynvc_segments":
        identities["input"] = _external(request.get("input"), label="input video")
        template = root / "pynvc-segments" / "segment.mp4"
        gpu = request.get("gpu", 0)
        if type(gpu) is not int or gpu < 0:  # pylint: disable=unidiomatic-typecheck
            raise PipelineError("gpu must be a non-negative integer")
        environment_identity, _environment_value = _environment(
            request, required_surfaces=("pynvc",), selected_gpu=gpu
        )
        if environment_identity is not None:
            identities["environment"] = environment_identity
        _bind_environment_gpu(_environment_value, gpu, label="request")
        schedule = "${AUTHENTICATED_SAMPLE_DIR}/segments.txt"
        configuration = "${AUTHENTICATED_SAMPLE_DIR}/transcode_config.json"
        planned.extend((
            _planned_operation(
                "samples/basic/create_video_segments.py",
                [
                    "-i", identities["input"]["path"], "-s", schedule,
                    "-c", configuration, "-o", str(template), "-g", str(gpu),
                ], template,
            ),
            _planned_operation(
                "samples/advanced/decode.py",
                ["for each output derived from authenticated segments.txt"],
            ),
        ))
        outputs.append(str(template))
    elif route == "container_triage":
        eligible, reasons = _target_gate(request)
        if not eligible:
            return {
                "status": "blocked", "gate": "target_eligibility",
                "reasons": reasons, "planned_operations": [],
                "planned_output_paths": [], "validated_identities": {},
                "no_mutation_performed": True,
            }
        surface = request.get("surface")
        if not isinstance(surface, str) or surface not in {"native", "pynvc"}:
            raise PipelineError("container_triage surface must be native or pynvc")
        gpu_value = request.get("gpu", 0)
        if (
            type(gpu_value) is not int  # pylint: disable=unidiomatic-typecheck
            or gpu_value < 0
        ):
            raise PipelineError("gpu must be a non-negative integer")
        environment_identity, _environment_value = _environment(
            request, required_surfaces=(surface,), selected_gpu=gpu_value
        )
        if environment_identity is not None:
            identities["environment"] = environment_identity
        _bind_environment_gpu(_environment_value, gpu_value, label="request")
        if request.get("input") is not None:
            identities["input"] = _external(
                request.get("input"), label="container input"
            )
            source = identities["input"]["path"]
        else:
            _exact_http_url(
                request.get("source_url"), label="container source_url"
            )
            source = str(root / "container-triage" / request.get("content_name", "content.mp4"))
        output = root / "container-triage" / "decoded.yuv"
        sample = "AppDec" if surface == "native" else "samples/advanced/decode.py"
        gpu = str(gpu_value)
        arguments = ["-i", source, "-o", str(output)]
        arguments.extend(["-gpu", gpu] if surface == "native" else ["-g", gpu, "-d", "0"])
        planned.append(_planned_operation(sample, arguments, output))
        outputs.append(str(output))
    elif route == "av1_verify":
        identities["recipe"], recipe = _recipe(request)
        identities["input"] = _external(
            request.get("input"), label="raw input", schema="1",
            kinds=("nvcodec-raw-video",),
        )
        intent = recipe["encoder_intent"]
        environment_identity, _environment_value = _environment(
            request,
            required_surfaces=("native",),
            selected_gpu=intent.get("gpu"),
        )
        if environment_identity is not None:
            identities["environment"] = environment_identity
        _bind_environment_gpu(_environment_value, intent.get("gpu"), label="recipe")
        frame_count = intent.get("frame_count")
        if (
            intent.get("codec") != "av1"
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count <= 0
        ):
            raise PipelineError(
                "av1_verify requires an AV1 recipe with positive frame_count"
            )
        if recipe["projections"]["native"]["status"] != "exact":
            raise PipelineError("av1_verify requires an exact native projection")
        for mode, flag in (("host", "0"), ("vidmem", "1")):
            output = root / "av1-verify" / f"av1-{mode}.av1"
            decoded = root / "av1-verify" / f"av1-{mode}.yuv"
            planned.extend((
                _planned_operation(
                    "AppEncCuda", [
                        "-i", identities["input"]["path"], "-o", str(output),
                        *recipe["projections"]["native"]["cli_options"],
                        "-outputInVidMem", flag,
                    ], output,
                ),
                _planned_operation(
                    "AppDec", [
                        "-i", str(output), "-o", str(decoded), "-gpu",
                        str(recipe["encoder_intent"]["gpu"]),
                    ], decoded,
                ),
            ))
            outputs.extend((str(output), str(decoded)))
    else:
        raw_refs = request.get("references")
        if not isinstance(raw_refs, Mapping) or not raw_refs:
            raise PipelineError("acceptance requires a non-empty references object")
        identities = {
            name: _external(identity, label=f"references.{name}")
            for name, identity in raw_refs.items()
            if isinstance(name, str) and name
        }
        required = set(ACCEPTANCE_TASKS) | set(ACCEPTANCE_LEDGER_REFERENCES)
        if len(identities) != len(raw_refs) or required - set(identities):
            raise PipelineError("acceptance references are incomplete")
        _validate_acceptance_stages(identities)
        _jsonl(identities["commands"], "commands")
        timing_value = artifact_io.read_verified_external_json(identities["timing"])
        if not isinstance(timing_value, dict):
            raise PipelineError("timing reference must contain a JSON object")
        outputs.extend(
            str(root / relative) for relative in _acceptance_files(request)
        )
    result = {
        "status": "planned", "planned_operations": planned,
        "planned_output_paths": outputs, "validated_identities": identities,
        "no_mutation_performed": True,
    }
    if selected_surface_plan is not None:
        result["surface_plan"] = selected_surface_plan
        result["surface_evaluations"] = surface_evaluations
        if selected_surface_plan["classification"] != "ready":
            result["status"] = "partial"
    return result


def run_pipeline_request(
    request: Mapping[str, Any], *, workspace: Path, runner: Any = None,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Execute one route without fallback and return honest boundary status."""
    ceiling = _timeout(timeout_seconds, float(timeout_seconds))
    value = _request(request, ceiling)
    base = {
        "schema_version": SCHEMA_VERSION, "kind": RESULT_KIND,
        "route": value["route"], "mode": value["mode"],
        "status": "failed", "boundaries": [],
    }
    workspace_created = False
    try:
        preflight = _media_preflight(value)
        if preflight is not None:
            base.update(preflight)
            return base
        preflight = _pynvc_full_samples_preflight(value)
        if preflight is not None:
            base.update(preflight)
            return base
        if value["mode"] == "dry_run":
            root = _planned_workspace(Path(workspace))
            base.update(_dry_run(value, root))
            return base
        root = _workspace(Path(workspace))
        workspace_created = True
        service = runner or command_runner.CommandRunner(root)
        handler = globals()[ROUTE_TABLE[value["route"]]]
        try:
            result = handler(
                value, root=root, runner=service, timeout=value["timeout_seconds"]
            )
            base.update(result)
        except BoundaryFailure as exc:
            base.update({
                "status": "failed", "reason": exc.reason,
                "boundaries": [{
                    "producer": exc.producer, "consumer": exc.consumer,
                    "artifact_path": exc.path, "artifact_size_bytes": None,
                    "artifact_sha256": None, "status": "failed", "reason": exc.reason,
                }],
            })
    except artifact_io.SkillDependencyError as exc:
        base.update({
            "status": "dependency_required",
            "gate": "skill_dependency",
            "dependency": {
                "skill": exc.skill,
                "needed_for": exc.needed_for,
                "reason": str(exc),
                "next_action": f"install_{exc.skill}_and_retry_stage",
            },
            "no_mutation_performed": not workspace_created,
        })
    except SurfaceInputRequired as exc:
        base.update({
            "status": "input_required",
            "gate": "surface_selector",
            "surface": exc.surface,
            "required_field": exc.field,
            "reason": exc.reason,
            "next_action": f"provide_{exc.field}",
            "no_mutation_performed": not workspace_created,
        })
    except SurfaceSetupRequired as exc:
        installed = _installed_sibling("jetson-video-setup")
        action = (
            f"use jetson-video-setup for the {exc.surface} surface and retry"
            if installed
            else "install jetson-video-setup, configure the selected surface, and retry"
        )
        base.update({
            "status": "dependency_required",
            "gate": "sdk_setup",
            "surface": exc.surface,
            "reason": exc.reason,
            "dependency": {
                "skill": "jetson-video-setup",
                "installed": installed,
                "needed_for": f"{exc.surface} SDK installation or repair",
                "next_action": action,
            },
            "no_mutation_performed": not workspace_created,
        })
    return base


def main(argv: list[str] | None = None) -> int:
    """Run one JSON request and write one fresh JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = artifact_io.preflight_fresh_output(args.workspace, args.output)
        request = artifact_io.strict_json_loads(args.request.read_bytes())
        if not isinstance(request, Mapping):
            raise PipelineError("request must be a JSON object")
        acceptance_output: Path | None = None
        if request.get("route") == "acceptance":
            planned_root = args.workspace.expanduser().resolve(strict=False)
            validation_file = _acceptance_files(request)[-1]
            expected = planned_root / validation_file
            requested = output
            if requested != expected:
                raise PipelineError(
                    "acceptance --output must be exactly "
                    f"evidence/{Path(validation_file).name} within --workspace"
                )
            acceptance_output = requested
        result = run_pipeline_request(request, workspace=args.workspace)
        if result.get("no_mutation_performed") is True:
            root = _workspace(args.workspace)
            if acceptance_output is not None:
                artifact_io.create_private_workspace(root / "evidence")
        else:
            root = artifact_io.resolve_private_workspace(args.workspace)
        artifact_io.write_fresh_json(root, output, result)
        if request.get("route") == "acceptance" and result["status"] == "complete":
            observed = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }
            expected = set(_acceptance_files(request))
            if observed != expected:
                raise PipelineError(
                    "acceptance CLI did not produce exactly the nine pre-seal files"
                )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] in {"complete", "planned"} else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
