#!/usr/bin/env python3
"""Execute one authenticated recipe through official encode and decode samples."""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

if not sys.flags.ignore_environment or not sys.flags.no_user_site:
    raise SystemExit("invoke this producer with isolated Python: python3 -I")

# Isolated mode excludes the script directory. Add only this skill's fixed
# private module directory; no sibling skill is imported as Python code.
_SCRIPT_DIR = Path(__file__).absolute().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# pylint: disable=wrong-import-position,import-error
import _pipeline_provenance as sample_provenance  # noqa: E402
import _pipeline_runtime as artifact_io  # noqa: E402
# pylint: enable=wrong-import-position,import-error

# This controller intentionally keeps request planning and both operation
# branches together so their artifact and marker contracts cannot diverge.
# pylint: disable=too-many-lines

command_runner = artifact_io
surface_router = artifact_io

SCHEMA_VERSION = "1.0"
REQUEST_KIND = "nvcodec-encode-request"
RESULT_KIND = "nvcodec-encode-result"
IDENTITY_KEYS = {"schema_version", "kind", "path", "size_bytes", "sha256"}
COMMAND_KEYS = {
    "schema_version",
    "kind",
    "command_id",
    "stage",
    "phase",
    "argv",
    "cwd",
    "environment_keys",
    "started_at",
    "ended_at",
    "duration_seconds",
    "approval_wait_seconds",
    "timeout_seconds",
    "exit_code",
    "timed_out",
    "launch_error",
    "stdout",
    "stderr",
    "stdout_tail",
    "stderr_tail",
}
REQUEST_REQUIRED = {
    "schema_version",
    "kind",
    "surface",
    "recipe",
    "input",
    "timeout_seconds",
}
# capability_report stays optional and purely additive: it refines a live
# classification when a request supplies one, and no surface fails for its
# absence, so an independently constructed request needs no new field.
REQUEST_OPTIONAL = {
    "buffer_mode",
    "capability_report",
    "environment",
    "pynvc_interpreter",
}
NON_MEDIA_REQUEST_REQUIRED = REQUEST_REQUIRED - {"input"}
ACCEPTED_MEDIA_INPUTS = ("local_path", "http_url", "https_url")
SURFACES = ("native", "pynvc")
SAMPLES = {
    "native": ("AppEncCuda", "AppDec"),
    "pynvc": ("samples/basic/encode.py", "samples/advanced/decode.py"),
}
EXTENSIONS = {"h264": "h264", "hevc": "hevc", "av1": "av1"}
FAILURE_PATTERN = sample_provenance.OFFICIAL_SAMPLE_FAILURE_PATTERN
_provenance_providers: tuple[Any, Any, Any] | None = None  # pylint: disable=invalid-name


class EncodeError(ValueError):
    """The request or official operation failed its evidence contract."""


def _providers() -> tuple[Any, Any, Any]:
    global _provenance_providers  # pylint: disable=global-statement
    if _provenance_providers is None:
        _provenance_providers = (
            sample_provenance.authenticate_native_samples,
            sample_provenance.authenticate_pynvc_samples,
            sample_provenance.run_authenticated,
        )
    return _provenance_providers


def _artifact_identity(value: Any, label: str) -> dict[str, Any]:
    """Accept one exact, currently valid portable artifact identity."""
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise EncodeError(f"{label} must be an exact portable artifact identity")
    copied = dict(value)
    try:
        artifact_io.verify_external_artifact(copied)
    except (OSError, ValueError) as exc:
        raise EncodeError(f"{label} identity is not current: {exc}") from exc
    return copied


def _identity(value: Any, *, schema: str, kind: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise EncodeError(f"{label} must be an exact portable artifact identity")
    if value.get("schema_version") != schema or value.get("kind") != kind:
        raise EncodeError(f"{label} identity must be schema {schema} kind {kind}")
    return _artifact_identity(value, label)


def _positive_timeout(value: Any, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncodeError("request timeout_seconds must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > maximum:
        raise EncodeError(f"request timeout_seconds must be in (0, {float(maximum)}]")
    return timeout


def _workspace(path: Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists():
        root = artifact_io.resolve_private_workspace(requested)
        if any(root.iterdir()):
            raise EncodeError("encode workspace must be fresh and empty")
        return root
    return artifact_io.create_private_workspace(requested)


def _request_envelope(value: Any, timeout_ceiling: float) -> dict[str, Any]:
    """Validate non-media request fields without opening artifact identities."""
    if not isinstance(value, Mapping):
        raise EncodeError("request must be a JSON object")
    if (
        set(value) - (REQUEST_REQUIRED | REQUEST_OPTIONAL)
        or not NON_MEDIA_REQUEST_REQUIRED.issubset(value)
    ):
        raise EncodeError(
            f"request fields must be required {sorted(REQUEST_REQUIRED)} plus optional "
            f"{sorted(REQUEST_OPTIONAL)}"
        )
    request = dict(value)
    if request["schema_version"] != SCHEMA_VERSION or request["kind"] != REQUEST_KIND:
        raise EncodeError(f"request identity must be schema {SCHEMA_VERSION} kind {REQUEST_KIND}")
    if (
        not isinstance(request["surface"], str)
        or request["surface"] not in {"native", "pynvc", "auto", "both"}
    ):
        raise EncodeError("surface must be native, pynvc, auto, or both")
    request["buffer_mode"] = request.get("buffer_mode", "gpu")
    if (
        not isinstance(request["buffer_mode"], str)
        or request["buffer_mode"] not in {"cpu", "gpu"}
    ):
        raise EncodeError("buffer_mode must be cpu or gpu")
    request["timeout_seconds"] = _positive_timeout(request["timeout_seconds"], timeout_ceiling)
    return request


def _request(value: Any, timeout_ceiling: float) -> dict[str, Any]:
    request = _request_envelope(value, timeout_ceiling)
    request["recipe"] = _identity(
        request["recipe"], schema="2.0", kind="nvcodec-recipe", label="recipe"
    )
    if request.get("environment") is not None:
        request["environment"] = _identity(
            request["environment"],
            schema=artifact_io.ENVIRONMENT_SCHEMA_VERSION,
            kind=artifact_io.ENVIRONMENT_KIND,
            label="environment",
        )
    interpreter = request.get("pynvc_interpreter")
    if interpreter is not None and (
        not isinstance(interpreter, str)
        or not interpreter
        or interpreter != interpreter.strip()
        or not Path(interpreter).is_absolute()
    ):
        raise EncodeError("pynvc_interpreter must be one exact absolute path")
    if request.get("capability_report") is not None:
        request["capability_report"] = _identity(
            request["capability_report"],
            schema=artifact_io.CAPABILITY_REPORT_SCHEMA_VERSION,
            kind=artifact_io.CAPABILITY_REPORT_KIND,
            label="capability report",
        )
    if request.get("input") is not None:
        request["input"] = _identity(
            request["input"], schema="1", kind="nvcodec-raw-video", label="input"
        )
    return request


def _input_required_result() -> dict[str, Any]:
    """Return the non-mutating terminal state for an absent user media input."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": "input_required",
        "gate": "media_input",
        "accepted_inputs": list(ACCEPTED_MEDIA_INPUTS),
        "next_action": "provide_media_path_or_url",
        "synthetic_input_allowed": False,
        "selected_surfaces": [],
        "operations": {},
        "output_artifacts": {},
    }


def _interpreter_required_result() -> dict[str, Any]:
    """Ask for the one explicit Py interpreter without scanning environments."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": "input_required",
        "gate": "surface_selector",
        "surface": "pynvc",
        "required_field": "pynvc_interpreter",
        "reason": (
            "provide the exact absolute Python interpreter for the existing "
            "PyNvVideoCodec environment; no environment scan is performed"
        ),
        "next_action": "provide_pynvc_interpreter",
        "selected_surfaces": [],
        "operations": {},
        "output_artifacts": {},
    }


def _setup_dependency(surface: str, reason: str) -> dict[str, Any]:
    """Describe the exact setup handoff when local authentication fails."""
    candidate = Path(__file__).absolute().parents[2] / "jetson-video-setup" / "SKILL.md"
    try:
        details = candidate.lstat()
        resolved_root = candidate.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        installed = False
    else:
        installed = bool(
            stat.S_ISREG(details.st_mode)
            and not stat.S_ISLNK(details.st_mode)
            and resolved_root.is_dir()
            and resolved == resolved_root / "SKILL.md"
        )
    action = (
        f"use jetson-video-setup for the {surface} surface and retry"
        if installed
        else "install jetson-video-setup, configure the selected surface, and retry"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": "dependency_required",
        "gate": "sdk_setup",
        "surface": surface,
        "reason": reason,
        "dependency": {
            "skill": "jetson-video-setup",
            "installed": installed,
            "needed_for": f"{surface} SDK installation or repair",
            "next_action": action,
        },
        "selected_surfaces": [],
        "operations": {},
        "output_artifacts": {},
    }


def _skill_dependency(error: artifact_io.SkillDependencyError) -> dict[str, Any]:
    """Preserve a missing sibling as a customer-actionable result envelope."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": "dependency_required",
        "gate": "skill_dependency",
        "dependency": {
            "skill": error.skill,
            "needed_for": error.needed_for,
            "reason": str(error),
            "next_action": f"install_{error.skill}_and_retry_stage",
        },
        "selected_surfaces": [],
        "operations": {},
        "output_artifacts": {},
    }


def _frame_bytes(intent: Mapping[str, Any]) -> int:
    width = intent["width"]
    height = intent["height"]
    fmt = intent["format"]
    if fmt in {"NV12", "YUV420", "P010"} and (width % 2 or height % 2):
        raise EncodeError(f"{fmt} execution requires even width and height")
    if fmt in {"NV16", "P210"} and width % 2:
        raise EncodeError(f"{fmt} execution requires even width")
    pixels = width * height
    sizes = {
        "NV12": pixels * 3 // 2,
        "YUV420": pixels * 3 // 2,
        "P010": pixels * 3,
        "NV16": pixels * 2,
        "P210": pixels * 4,
        "YUV444": pixels * 3,
        "YUV444_16BIT": pixels * 6,
        "ARGB": pixels * 4,
        "ABGR": pixels * 4,
    }
    try:
        return sizes[fmt]
    except KeyError as exc:
        raise EncodeError(f"no exact raw frame-size contract for {fmt}") from exc


def _validate_raw(recipe: Mapping[str, Any], identity: Mapping[str, Any]) -> int:
    intent = recipe["encoder_intent"]
    count = intent.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EncodeError("executable recipe requires an exact positive frame_count")
    expected = _frame_bytes(intent) * count
    if identity["size_bytes"] != expected:
        raise EncodeError(
            f"raw input size must equal exactly {count} complete {intent['format']} "
            f"frames ({expected} bytes)"
        )
    return count


def _native_readiness(environment: Mapping[str, Any]) -> list[str]:
    """Block a malformed or unready native surface during planning."""
    reasons = artifact_io.native_surface_defects(environment)
    surface = artifact_io.environment_surface(environment, "native") or {}
    cuda = surface.get("cuda")
    if isinstance(cuda, Mapping) and not (
        str(cuda.get("version", "")).startswith("13.")
        and isinstance(cuda.get("root"), str)
        and cuda["root"]
    ):
        reasons.append("live environment does not report an installed CUDA 13.x toolkit root")
    return reasons


def _pynvc_readiness(environment: Mapping[str, Any]) -> list[str]:
    """Block a malformed or unready Py surface during planning."""
    return artifact_io.pynvc_full_sample_defects(environment)


def _surface_plan(  # pylint: disable=too-many-arguments,too-many-locals
    requested_surface: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
    buffer_mode: str,
    *,
    recipe_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any] | None,
    capability_identity: Mapping[str, Any] | None = None,
    local_errors: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decide eligibility per surface at planning time, never later.

    Each surface is judged on its own required subset and its own live
    classification. One surface's defect never raises out of this function, so
    a malformed surface is blocked here while its peer stays independently
    actionable.
    """
    candidates: dict[str, Any] = {}
    details: dict[str, dict[str, Any]] = {}
    evaluated_surfaces = (
        (requested_surface,)
        if requested_surface in SURFACES
        else SURFACES
    )
    for surface in SURFACES:
        if surface not in evaluated_surfaces:
            reason = f"{surface} was not evaluated because it was not selected"
            candidates[surface] = {"eligible": False, "reasons": [reason]}
            details[surface] = {
                "projection": "not_evaluated",
                "live": {"classification": "not_evaluated", "reasons": [reason]},
                "eligible": False,
                "reasons": [reason],
            }
            continue
        projection = recipe["projections"][surface]
        reasons = []
        try:
            _decoded_layout(recipe["encoder_intent"])
        except EncodeError as exc:
            reasons.append(str(exc))
        if projection["status"] != "exact":
            reasons.append("recipe projection has blocking losses")
        reasons.extend(
            _native_readiness(environment)
            if surface == "native"
            else _pynvc_readiness(environment)
        )
        if local_errors and surface in local_errors:
            reasons.append(local_errors[surface])
        blocked = bool(reasons)
        live = _live_classification(
            surface,
            blocked=blocked,
            recipe_identity=recipe_identity,
            environment_identity=environment_identity,
            capability_identity=capability_identity,
            buffer_mode=buffer_mode,
        )
        if not blocked and surface == "pynvc" and live["classification"] != "compatible":
            reasons.extend(live["reasons"])
        candidates[surface] = {"eligible": not reasons, "reasons": reasons}
        details[surface] = {
            "projection": projection["status"],
            "live": live,
            "eligible": not reasons,
            "reasons": list(reasons),
        }
        local_reason = local_errors.get(surface) if local_errors else None
        local_reason_text = str(local_reason) if isinstance(local_reason, str) else ""
        if (
            surface == "pynvc"
            and "not evaluated" in local_reason_text
            and "pynvc_interpreter" in local_reason_text
        ):
            details[surface]["evaluation_status"] = "not_evaluated"
            details[surface]["next_action"] = "provide_pynvc_interpreter_and_retry"
    return surface_router.build_surface_plan(requested_surface, candidates), details


def _live_classification(  # pylint: disable=too-many-arguments
    surface: str,
    *,
    blocked: bool,
    recipe_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any] | None,
    capability_identity: Mapping[str, Any] | None,
    buffer_mode: str,
) -> dict[str, Any]:
    """Classify one surface without letting its failure reach its peer."""
    if blocked:
        return {
            "classification": "not_evaluated",
            "reasons": ["surface was blocked during planning validation"],
        }
    try:
        return artifact_io.check_live_recipe(
            recipe_identity,
            environment_identity,
            surface,
            buffer_mode=buffer_mode,
            capability_identity=capability_identity,
        )
    except (OSError, ValueError) as exc:
        return {
            "classification": "unknown",
            "reasons": [f"live recipe classification failed: {type(exc).__name__}: {exc}"],
        }


def _stage_environment(root: Path, branch: Path, environment: dict[str, Any]) -> dict[str, Any]:
    del root
    return artifact_io.write_fresh_json(branch, branch / "environment.json", environment)


def _configuration(
    branch: Path, surface: str, recipe: dict[str, Any]
) -> tuple[dict[str, Any], Path | None]:
    if surface == "pynvc":
        value = recipe["projections"]["pynvc"]["config"]
        path = branch / "encode-config.json"
        identity = artifact_io.write_fresh_bytes(
            branch,
            path,
            artifact_io.canonical_json_bytes(value),
            schema_version="1",
            kind="pynvc-encode-config",
        )
        return identity, path
    value = {
        "schema_version": "1",
        "kind": "native-encode-projection",
        "cli_options": recipe["projections"]["native"]["cli_options"],
    }
    path = branch / "encode-projection.json"
    return artifact_io.write_fresh_json(branch, path, value), path


def _root_identity(root: Path, child: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    path = artifact_io.verify_artifact(child, identity)
    return artifact_io.snapshot_artifact(
        root, path, schema_version=identity["schema_version"], kind=identity["kind"]
    )


def _command_texts(runner: Any, result: Mapping[str, Any]) -> tuple[str, str]:
    logs = artifact_io.resolve_private_workspace(Path(runner.workspace))
    return (
        artifact_io.read_verified_text(logs, result["stdout"], max_bytes=16 * 1024 * 1024),
        artifact_io.read_verified_text(logs, result["stderr"], max_bytes=16 * 1024 * 1024),
    )


def _command_evidence(  # pylint: disable=too-many-arguments
    result: Any,
    *,
    token: Any,
    suffix: list[str],
    branch: Path,
    runner: Any,
    timeout: float,
    stage: str,
    phase: str,
) -> tuple[dict[str, Any], str, str]:
    expected = {
        "schema_version": "1",
        "kind": "command-result",
        "argv": [*token.launcher, *suffix],
        "cwd": str(branch),
        "stage": stage,
        "phase": phase,
        "timeout_seconds": float(timeout),
    }
    if (
        not isinstance(result, dict)
        or set(result) != COMMAND_KEYS
        or any(result.get(key) != value for key, value in expected.items())
    ):
        raise EncodeError("authenticated command result does not bind the exact launch")
    if (
        result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("launch_error") is not None
    ):
        raise EncodeError("authenticated command did not exit successfully")
    for key in ("duration_seconds", "approval_wait_seconds"):
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise EncodeError(f"command result {key} is invalid")
    if result["approval_wait_seconds"] != 0:
        raise EncodeError("non-interactive command cannot report approval wait")
    if not isinstance(result["started_at"], str) or not isinstance(result["ended_at"], str):
        raise EncodeError("command UTC timestamps are invalid")
    if not isinstance(result["command_id"], str) or not result["command_id"]:
        raise EncodeError("command ID is invalid")
    if not isinstance(result["environment_keys"], list) or not all(
        isinstance(item, str) for item in result["environment_keys"]
    ):
        raise EncodeError("command environment evidence is invalid")
    if not isinstance(result["stdout_tail"], str) or not isinstance(result["stderr_tail"], str):
        raise EncodeError("command log tails are invalid")
    if (
        result["stdout"].get("kind") != "command-stdout-log"
        or result["stderr"].get("kind") != "command-stderr-log"
    ):
        raise EncodeError("command log identities have invalid kinds")
    stdout, stderr = _command_texts(runner, result)
    if FAILURE_PATTERN.search(stdout) or FAILURE_PATTERN.search(stderr):
        raise EncodeError("authenticated command output contains an explicit failure marker")
    return dict(result), stdout, stderr


def _exact_marker(text: str, pattern: str, expected: int, label: str) -> None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if matches != [str(expected)]:
        raise EncodeError(f"{label} must report exactly {expected} frames once")


def _structure(  # pylint: disable=too-many-branches,too-many-locals
    path: Path, codec: str, width: int, height: int, frame_count: int
) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(1024 * 1024)
    if not prefix:
        raise EncodeError("encoded output is empty")
    if codec == "av1":
        if len(prefix) < 32 or prefix[:4] != b"DKIF":
            raise EncodeError("AV1 output is not an IVF AV01 stream")
        version = int.from_bytes(prefix[4:6], "little")
        header_size = int.from_bytes(prefix[6:8], "little")
        if version != 0 or header_size != 32 or prefix[8:12] != b"AV01":
            raise EncodeError("AV1 output has an invalid IVF version, header, or fourcc")
        if (
            int.from_bytes(prefix[12:14], "little") != width
            or int.from_bytes(prefix[14:16], "little") != height
        ):
            raise EncodeError("AV1 IVF dimensions differ from the recipe")
        header_count = int.from_bytes(prefix[24:28], "little")
        if header_count not in {frame_count, 0xFFFF}:
            raise EncodeError(
                "AV1 IVF header frame count is neither exact nor the "
                "AppEncCuda 0xFFFF sentinel"
            )
        size = path.stat().st_size
        walked = 0
        with path.open("rb") as stream:
            stream.seek(32)
            position = 32
            while position < size:
                frame_header = stream.read(12)
                if len(frame_header) != 12:
                    raise EncodeError("AV1 IVF has a truncated frame header")
                payload_size = int.from_bytes(frame_header[:4], "little")
                if payload_size <= 0 or position + 12 + payload_size > size:
                    raise EncodeError("AV1 IVF has a zero or truncated frame payload")
                stream.seek(payload_size, os.SEEK_CUR)
                position += 12 + payload_size
                walked += 1
            if position != size or stream.read(1):
                raise EncodeError("AV1 IVF has trailing bytes")
        if walked != frame_count:
            raise EncodeError(f"AV1 IVF contains {walked} frames, expected {frame_count}")
        return {
            "format": "ivf-av01",
            "header_frame_count": header_count,
            "walked_frame_count": walked,
        }
    starts = [match.end() for match in re.finditer(b"\x00\x00\x01", prefix)]
    if not starts:
        raise EncodeError(f"{codec} output has no Annex-B start code")
    nal_types = []
    for offset in starts:
        if offset < len(prefix):
            nal_types.append(
                prefix[offset] & 0x1F if codec == "h264" else (prefix[offset] >> 1) & 0x3F
            )
    required = 7 if codec == "h264" else 33
    if required not in nal_types:
        raise EncodeError(f"{codec} output has no authenticated sequence-parameter NAL")
    return {"format": f"annex-b-{codec}", "sequence_parameter_nal": required}


def _suffixes(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    surface: str,
    recipe: dict[str, Any],
    raw: Path,
    encoded: Path,
    decoded: Path,
    config: Path | None,
    frame_count: int,
    buffer_mode: str,
) -> tuple[list[str], list[str]]:
    intent = recipe["encoder_intent"]
    if surface == "native":
        encode = [
            "-i",
            str(raw),
            "-o",
            str(encoded),
            *recipe["projections"]["native"]["cli_options"],
        ]
        decode = ["-i", str(encoded), "-o", str(decoded), "-gpu", str(intent["gpu"])]
    else:
        if config is None:
            raise EncodeError("Py encode config was not staged")
        encode = [
            "-i",
            str(raw),
            "-o",
            str(encoded),
            "-s",
            f"{intent['width']}x{intent['height']}",
            "-m",
            buffer_mode,
            "-if",
            intent["format"],
            "-f",
            str(frame_count),
            "-g",
            str(intent["gpu"]),
            "-c",
            intent["codec"],
            "-json",
            str(config),
        ]
        decode = [
            "-i",
            str(encoded),
            "-o",
            str(decoded),
            "-g",
            str(intent["gpu"]),
            "-d",
            "0",
            "-f",
            str(frame_count),
        ]
    return encode, decode


def _verify_encode_markers(
    surface: str,
    encode_stdout: str,
    encoded: Path,
    count: int,
    buffer_mode: str,
) -> None:
    if surface == "native":
        _exact_marker(encode_stdout, r"^Total frames encoded: ([0-9]+)$", count, "native encode")
        return
    _exact_marker(
        encode_stdout,
        rf"^Completed encoding ([0-9]+) frames using {buffer_mode.upper()} buffers$",
        count,
        "Py encode",
    )
    output_lines = re.findall(r"^Output file: (.+)$", encode_stdout, flags=re.MULTILINE)
    if output_lines != [str(encoded), str(encoded)]:
        raise EncodeError("Py encode must print the same requested output path exactly twice")
    if len(re.findall(r"^ENCODING COMPLETE$", encode_stdout, flags=re.MULTILINE)) != 1:
        raise EncodeError("Py encode completion marker is missing or duplicated")


def _verify_markers(
    surface: str,
    decode_stdout: str,
    decoded: Path,
    count: int,
) -> str:
    if surface == "native":
        _exact_marker(decode_stdout, r"^Total frame decoded: ([0-9]+)$", count, "native decode")
        saved = re.findall(
            rf"^Saved in file {re.escape(str(decoded))} in "
            r"(NV12|P016|YUV444|YUV444P16|NV16|P216) format$",
            decode_stdout,
            flags=re.MULTILINE,
        )
        if len(saved) != 1:
            raise EncodeError("native decode must report one exact saved output layout")
        return saved[0]
    success = re.findall(
        r"^Successfully decoded requested ([0-9]+) frames to (.+)$",
        decode_stdout,
        flags=re.MULTILINE,
    )
    if success != [(str(count), str(decoded))]:
        raise EncodeError("Py decode did not report the exact requested frame count and output")
    return "derived"


def _decoded_layout(intent: Mapping[str, Any]) -> tuple[str, int]:
    pixels = intent["width"] * intent["height"]
    layouts = {
        "NV12": ("NV12", pixels * 3 // 2),
        "YUV420": ("NV12", pixels * 3 // 2),
        "P010": ("P016", pixels * 3),
        "NV16": ("NV16", pixels * 2),
        "P210": ("P216", pixels * 4),
        "YUV444": ("YUV444", pixels * 3),
        "YUV444_16BIT": ("YUV444P16", pixels * 6),
    }
    try:
        return layouts[intent["format"]]
    except KeyError as exc:
        raise EncodeError(
            f"exact decoder output layout is not proven for {intent['format']}"
        ) from exc


def _execute_branch(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    surface: str,
    root: Path,
    branch: Path,
    authentication: Any,
    recipe: dict[str, Any],
    request: dict[str, Any],
    runner: Any,
    config_identity: dict[str, Any],
    config_path: Path | None,
    raw_path: Path,
    frame_count: int,
    launch: Any,
) -> dict[str, Any]:
    codec = recipe["encoder_intent"]["codec"]
    encoded = branch / f"encoded.{EXTENSIONS[codec]}"
    decoded = branch / "decoded.yuv"
    if encoded.exists() or decoded.exists():
        raise EncodeError("operation outputs must be fresh")
    encode_suffix, decode_suffix = _suffixes(
        surface,
        recipe,
        raw_path,
        encoded,
        decoded,
        config_path,
        frame_count,
        request["buffer_mode"],
    )
    encode_token = authentication.token(SAMPLES[surface][0])
    decode_token = authentication.token(SAMPLES[surface][1])
    timeout = request["timeout_seconds"]
    encode_result = launch(
        encode_token,
        encode_suffix,
        workspace=branch,
        runner=runner,
        cwd=branch,
        timeout_seconds=timeout,
        stage=f"{surface}-encode",
        phase="execute",
    )
    encode_command, encode_stdout, _ = _command_evidence(
        encode_result,
        token=encode_token,
        suffix=encode_suffix,
        branch=branch,
        runner=runner,
        timeout=timeout,
        stage=f"{surface}-encode",
        phase="execute",
    )
    _verify_encode_markers(
        surface,
        encode_stdout,
        encoded,
        frame_count,
        request["buffer_mode"],
    )
    encoded_identity = artifact_io.snapshot_artifact(
        root, encoded, schema_version="1", kind=f"nvcodec-{codec}-bitstream"
    )
    if encoded_identity["size_bytes"] <= 0:
        raise EncodeError("encoded output is empty")
    structure = _structure(
        encoded,
        codec,
        recipe["encoder_intent"]["width"],
        recipe["encoder_intent"]["height"],
        frame_count,
    )
    decode_result = launch(
        decode_token,
        decode_suffix,
        workspace=branch,
        runner=runner,
        cwd=branch,
        timeout_seconds=timeout,
        stage=f"{surface}-decode",
        phase="verify",
    )
    decode_command, decode_stdout, _ = _command_evidence(
        decode_result,
        token=decode_token,
        suffix=decode_suffix,
        branch=branch,
        runner=runner,
        timeout=timeout,
        stage=f"{surface}-decode",
        phase="verify",
    )
    decoded_identity = artifact_io.snapshot_artifact(
        root, decoded, schema_version="1", kind="nvcodec-decoded-frames"
    )
    if decoded_identity["size_bytes"] <= 0:
        raise EncodeError("independent decode produced no frame bytes")
    observed_layout = _verify_markers(
        surface,
        decode_stdout,
        decoded,
        frame_count,
    )
    expected_layout, bytes_per_frame = _decoded_layout(recipe["encoder_intent"])
    if surface == "native" and observed_layout != expected_layout:
        raise EncodeError(f"native decoder reported {observed_layout}, expected {expected_layout}")
    expected_decoded_size = bytes_per_frame * frame_count
    if decoded_identity["size_bytes"] != expected_decoded_size:
        raise EncodeError(
            "independent decode byte count differs from the exact evidenced layout: "
            f"expected {expected_decoded_size}, got {decoded_identity['size_bytes']}"
        )
    artifact_io.verify_external_artifact(request["recipe"])
    if request.get("environment") is not None:
        artifact_io.verify_external_artifact(request["environment"])
    artifact_io.verify_external_artifact(request["input"])
    artifact_io.verify_artifact(branch, config_identity)
    artifact_io.verify_artifact(root, encoded_identity)
    artifact_io.verify_artifact(root, decoded_identity)
    return {
        "surface": surface,
        "status": "operation_verified",
        "reasons": [],
        "provenance": {
            "report": _root_identity(root, branch, authentication.report_identity),
            "samples": list(SAMPLES[surface]),
        },
        "configuration": _root_identity(root, branch, config_identity),
        "encode": {
            "command": encode_command,
            "frames": frame_count,
            "output": encoded_identity,
            "structure": structure,
        },
        "decode": {
            "command": decode_command,
            "frames": frame_count,
            "output": decoded_identity,
            "independent": True,
            "layout": expected_layout,
            "expected_size_bytes": expected_decoded_size,
            "consumed_bitstream_sha256": encoded_identity["sha256"],
        },
    }


def run_encode_request(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    request: Mapping[str, Any],
    *,
    workspace: Path,
    runner: Any = None,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Validate, preauthenticate, encode, independently decode, and report."""
    ceiling = _positive_timeout(timeout_seconds, float("inf"))
    envelope = _request_envelope(request, ceiling)
    if envelope.get("input") is None:
        return _input_required_result()
    value = _request(request, ceiling)
    if (
        value["surface"] in {"pynvc", "both"}
        and value.get("environment") is None
        and value.get("pynvc_interpreter") is None
    ):
        return _interpreter_required_result()
    try:
        artifact_io.validate_recipe_identity(value["recipe"])
    except artifact_io.SkillDependencyError as exc:
        return _skill_dependency(exc)
    recipe = artifact_io.read_verified_external_json(value["recipe"])
    intent = recipe.get("encoder_intent")
    recipe_gpu = intent.get("gpu") if isinstance(intent, Mapping) else None
    if isinstance(recipe_gpu, bool) or not isinstance(recipe_gpu, int) or recipe_gpu < 0:
        raise EncodeError("recipe encoder_intent.gpu must be a non-negative integer")
    required_surfaces = (
        (value["surface"],) if value["surface"] in SURFACES else SURFACES
    )
    local_errors: dict[str, str] = {}
    if value.get("environment") is not None:
        environment_identity: Mapping[str, Any] | None = value["environment"]
        environment = artifact_io.validate_environment_envelope(
            artifact_io.read_verified_external_json(value["environment"]),
            label="encode",
            error_type=EncodeError,
            required_surfaces=required_surfaces,
        )
    else:
        if value.get("capability_report") is not None:
            raise EncodeError(
                "capability_report requires the exact supplied environment to which it is bound"
            )
        environment_identity = None
        environment, local_errors = sample_provenance.build_local_runtime_binding(
            required_surfaces,
            pynvc_interpreter=value.get("pynvc_interpreter"),
            gpu=recipe_gpu,
            timeout_seconds=value["timeout_seconds"],
        )
    if recipe_gpu != environment.get("selected_gpu"):
        raise EncodeError(
            "recipe encoder_intent.gpu must equal environment selected_gpu"
        )
    root = _workspace(Path(workspace))
    frame_count = _validate_raw(recipe, value["input"])
    raw_path = artifact_io.verify_external_artifact(value["input"])
    plan, capability_details = _surface_plan(
        value["surface"],
        recipe,
        environment,
        value["buffer_mode"],
        recipe_identity=value["recipe"],
        environment_identity=environment_identity,
        capability_identity=value.get("capability_report"),
        local_errors=local_errors,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": "blocked",
        "classification": plan["classification"],
        "selected_surfaces": plan["selected_surfaces"],
        "surface_plan": plan,
        "capability_validation": capability_details,
        "request": {
            "surface": value["surface"],
            "buffer_mode": value["buffer_mode"],
            "timeout_seconds": value["timeout_seconds"],
            "recipe": value["recipe"],
            "capability_report": value.get("capability_report"),
            "environment": value.get("environment"),
            "runtime_authority": (
                "supplied_setup_environment"
                if value.get("environment") is not None
                else "consumer_local_binding"
            ),
            "pynvc_interpreter": value.get("pynvc_interpreter"),
            "input": value["input"],
        },
        "operations": {},
        "output_artifacts": {},
        "reasons": list(plan["reasons"]),
    }
    selected = plan["selected_surfaces"]
    if value["surface"] in SURFACES and not selected:
        selected_surface = value["surface"]
        readiness = (
            _native_readiness(environment)
            if selected_surface == "native"
            else _pynvc_readiness(environment)
        )
        setup_reasons = [
            *readiness,
            *(
                [local_errors[selected_surface]]
                if selected_surface in local_errors
                else []
            ),
        ]
        if setup_reasons:
            return _setup_dependency(
                selected_surface, "; ".join(dict.fromkeys(setup_reasons))
            )
    if value["surface"] == "both":
        for surface in SURFACES:
            if surface not in selected:
                result["operations"][surface] = {
                    "surface": surface,
                    "status": "blocked",
                    "reasons": list(capability_details[surface]["reasons"]),
                }
    if not selected:
        result["status"] = (
            "selection_required" if plan["classification"] == "selection_required" else "blocked"
        )
        return result
    service = runner or command_runner.CommandRunner(root)
    native_auth, pynvc_auth, launch = _providers()
    prepared: dict[str, tuple[Path, dict[str, Any], Path | None, dict[str, Any]]] = {}
    authentications: dict[str, Any] = {}
    for surface in selected:
        branch = artifact_io.create_private_workspace(root / surface)
        environment_identity = _stage_environment(root, branch, environment)
        config_identity, config_path = _configuration(branch, surface, recipe)
        prepared[surface] = (branch, config_identity, config_path, environment_identity)
    # Attempt every selected authentication before the first codec launch.
    for surface in selected:
        try:
            branch, _config_identity, _config_path, environment_identity = prepared[surface]
            authenticate = native_auth if surface == "native" else pynvc_auth
            authentications[surface] = authenticate(
                environment_workspace=branch,
                environment_identity=environment_identity,
                workspace=branch,
                samples=SAMPLES[surface],
                runner=service,
                timeout_seconds=value["timeout_seconds"],
                report_path=branch / "sample-provenance.json",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            reason = f"preflight authentication failed: {type(exc).__name__}: {exc}"
            result["reasons"].append(f"{surface}: {reason}")
            result["operations"][surface] = {
                "surface": surface,
                "status": "blocked",
                "reasons": [reason],
            }
    verified = 0
    for surface in (item for item in selected if item in authentications):
        branch, config_identity, config_path, _environment_identity = prepared[surface]
        try:
            operation = _execute_branch(
                surface=surface,
                root=root,
                branch=branch,
                authentication=authentications[surface],
                recipe=recipe,
                request=value,
                runner=service,
                config_identity=config_identity,
                config_path=config_path,
                raw_path=raw_path,
                frame_count=frame_count,
                launch=launch,
            )
            verified += 1
            result["operations"][surface] = operation
            result["output_artifacts"][surface] = operation["encode"]["output"]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            reason = f"{type(exc).__name__}: {exc}"
            result["operations"][surface] = {
                "surface": surface,
                "status": "operation_failed",
                "reasons": [reason],
            }
            result["reasons"].append(f"{surface}: {reason}")
    if verified == len(selected) and plan["classification"] == "ready":
        result["status"] = "operation_verified"
    elif verified:
        result["status"] = "partial"
    elif any(item.get("status") == "blocked" for item in result["operations"].values()):
        result["status"] = "blocked"
    else:
        result["status"] = "operation_failed"
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the thin JSON-only encode CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = artifact_io.preflight_fresh_output(args.workspace, args.output)
        request = artifact_io.strict_json_loads(args.request.read_bytes())
        result = run_encode_request(request, workspace=args.workspace)
        root = (
            _workspace(args.workspace)
            if result.get("status") in {"input_required", "dependency_required"}
            else artifact_io.resolve_private_workspace(args.workspace)
        )
        artifact_io.write_fresh_json(root, output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "operation_verified" else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
