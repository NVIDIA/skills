#!/usr/bin/env python3
"""Private artifact, command, routing, and recipe runtime for this skill."""

# This module is the skill's single private runtime owner: artifact identity,
# command execution, surface routing, environment validation and the recipe
# bridge each keep their complete contract local and reviewable in one file.
# pylint: disable=too-many-lines

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

IDENTITY_KEYS = frozenset(
    {"schema_version", "kind", "path", "size_bytes", "sha256"}
)
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_TAIL_BYTES = 4096
_STAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
COMMAND_PHASES = (
    "probe",
    "plan",
    "prepare",
    "install",
    "warmup",
    "measure",
    "verify",
    "execute",
)
SURFACE_ORDER = ("native", "pynvc")
REQUESTED_SURFACES = ("native", "pynvc", "auto", "both")
ENVIRONMENT_KIND = "nvcodec-environment"
# The frozen `nvcodec-environment` 1.2 contract. Consumers pin this exact
# version and validate its required subset only; additive keys are tolerated.
ENVIRONMENT_SCHEMA_VERSION = "1.2"
ENVIRONMENT_RUNTIMES = ("pynvc", "native", "both")
LOCAL_RUNTIME_BINDING_KIND = "nvcodec-local-runtime-binding"
LOCAL_RUNTIME_BINDING_SCHEMA_VERSION = "1.0"
CAPABILITY_REPORT_KIND = "nvcodec-encoder-capability-report"
CAPABILITY_REPORT_SCHEMA_VERSION = "1.0"
NATIVE_PACKAGE_NAME = "nvidia-video-codec-sdk"
REQUIRED_BUILD_TOOLS = ("cmake", "cxx", "nvcc", "pkg_config", "generator")
# 1.2 publishes `installation.build_tool_identities` under the executable's own
# name; these are the internal names the toolchain is consumed under.
_PUBLISHED_BUILD_TOOLS = (
    ("cmake", "cmake"),
    ("cxx", "g++"),
    ("nvcc", "nvcc"),
    ("pkg_config", "pkg-config"),
)
# CMake generator preference, highest first, with its exact generator name.
_PUBLISHED_GENERATORS = (("ninja", "Ninja"), ("make", "Unix Makefiles"))
_ELIGIBILITY_KEYS = frozenset({"eligible", "reasons"})
_RECIPE_OUTPUT_LIMIT = 1024 * 1024


class ArtifactError(ValueError):
    """An artifact violates the portable content contract."""


class CommandError(ValueError):
    """A command violates the deterministic execution contract."""


class RecipeBridgeError(ValueError):
    """The installed recipe skill rejected a recipe operation."""


class SkillDependencyError(RecipeBridgeError):
    """A route cannot continue because one required sibling skill is absent."""

    def __init__(self, skill: str, needed_for: str):
        self.skill = skill
        self.needed_for = needed_for
        super().__init__(
            f"I can run {needed_for}, but it requires the {skill} skill, "
            f"which is not installed. Install {skill} and retry this stage."
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object name: {key!r}")
        value[key] = item
    return value


def strict_json_loads(payload: bytes | str) -> Any:
    """Decode strict UTF-8 JSON without duplicate names or non-finites."""
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ArtifactError(f"JSON is not UTF-8: {exc}") from exc
    elif isinstance(payload, str):
        text = payload
    else:
        raise ArtifactError("JSON payload must be bytes or text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"invalid strict JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode deterministic strict JSON as UTF-8 with one trailing newline."""
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            f"value cannot be represented as strict JSON: {exc}"
        ) from exc
    return (rendered + "\n").encode("utf-8")


def _contract_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ArtifactError(f"{label} must be a non-empty canonical string")
    return value


def _contract_path(value: Any) -> str:
    locator = _contract_string(value, "path")
    if any(ord(character) < 32 or ord(character) == 127 for character in locator):
        raise ArtifactError("path must not contain control characters")
    return locator


def _bound(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{label} must be a non-negative integer")
    if value > maximum:
        raise ArtifactError(f"{label} exceeds the {maximum}-byte bound")
    return value


def _existing_regular(path: Path) -> Path:
    requested = Path(_contract_path(str(path)))
    try:
        candidate = requested.expanduser().resolve(strict=True)
        details = candidate.stat()
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"artifact is not an accessible file: {path}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise ArtifactError(f"artifact is not a regular file: {candidate}")
    return candidate


def create_private_workspace(path: Path) -> Path:
    """Create one fresh canonical managed workspace with exact mode 0700."""
    requested = Path(_contract_path(str(path))).expanduser()
    try:
        parent = requested.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(
            f"workspace parent is unavailable: {requested.parent}"
        ) from exc
    candidate = parent / requested.name
    try:
        os.mkdir(candidate, mode=0o700)
        os.chmod(candidate, 0o700)
    except OSError as exc:
        raise ArtifactError(
            f"cannot create fresh workspace {candidate}: {exc}"
        ) from exc
    return resolve_private_workspace(candidate)


def resolve_private_workspace(path: Path) -> Path:
    """Resolve an existing managed workspace and require exact mode 0700."""
    requested = Path(_contract_path(str(path)))
    try:
        candidate = requested.expanduser().resolve(strict=True)
        details = candidate.stat()
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"workspace is unavailable: {path}") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ArtifactError(f"workspace is not a directory: {candidate}")
    if stat.S_IMODE(details.st_mode) != 0o700:
        raise ArtifactError(f"workspace mode must be exactly 0700: {candidate}")
    return candidate


def preflight_fresh_output(workspace: Path, path: Path) -> Path:
    """Validate a fresh result destination without creating the workspace."""
    requested_root = Path(_contract_path(str(workspace))).expanduser()
    if requested_root.exists():
        root = resolve_private_workspace(requested_root)
    else:
        if requested_root.is_symlink():
            raise ArtifactError(f"workspace must not be a symlink: {requested_root}")
        try:
            parent = requested_root.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactError(
                f"workspace parent is unavailable: {requested_root.parent}"
            ) from exc
        root = parent / requested_root.name
    requested = Path(_contract_path(str(path))).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        candidate = candidate.resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactError(f"artifact is outside managed workspace: {candidate}") from exc
    if candidate == root or candidate.exists() or candidate.is_symlink():
        raise ArtifactError(f"artifact output must be a fresh file: {candidate}")
    return candidate


def _workspace_path(workspace: Path, path: Path, *, must_exist: bool) -> Path:
    root = resolve_private_workspace(workspace)
    requested = Path(_contract_path(str(path)))
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(
            f"workspace artifact is unavailable: {candidate}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(
            f"artifact is outside managed workspace: {resolved}"
        ) from exc
    if must_exist:
        return _existing_regular(resolved)
    try:
        parent = resolved.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(
            f"artifact output parent is unavailable: {resolved.parent}"
        ) from exc
    if not parent.is_dir():
        raise ArtifactError(f"artifact output parent is not a directory: {parent}")
    return parent / resolved.name


def _hash(path: Path, maximum: int = MAX_ARTIFACT_BYTES) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise ArtifactError(
                        f"artifact exceeds the {maximum}-byte bound: {path}"
                    )
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot read artifact {path}: {exc}") from exc
    return total, digest.hexdigest()


def _identity(
    path_locator: str,
    size_bytes: int,
    sha256: str,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": _contract_string(schema_version, "schema_version"),
        "kind": _contract_string(kind, "kind"),
        "path": _contract_path(path_locator),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _validate_identity(identity: Mapping[str, Any]) -> tuple[str, int, str]:
    if not isinstance(identity, Mapping) or set(identity) != IDENTITY_KEYS:
        raise ArtifactError(
            f"artifact identity fields must be exactly {sorted(IDENTITY_KEYS)}"
        )
    _contract_string(identity["schema_version"], "schema_version")
    _contract_string(identity["kind"], "kind")
    locator = _contract_path(identity["path"])
    size_bytes = _bound(identity["size_bytes"], "size_bytes", MAX_ARTIFACT_BYTES)
    sha256 = identity["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ArtifactError("sha256 must be 64 lowercase hexadecimal characters")
    return locator, size_bytes, sha256


def snapshot_external_artifact(
    path: Path,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    """Snapshot an external file with a canonical absolute path locator."""
    candidate = _existing_regular(Path(path))
    size_bytes, sha256 = _hash(candidate)
    return _identity(
        str(candidate),
        size_bytes,
        sha256,
        schema_version=schema_version,
        kind=kind,
    )


def verify_external_artifact(identity: Mapping[str, Any]) -> Path:
    """Verify one external identity and return its canonical current path."""
    locator, expected_size, expected_sha = _validate_identity(identity)
    if not Path(locator).is_absolute():
        raise ArtifactError("external artifact path must be absolute")
    candidate = _existing_regular(Path(locator))
    if str(candidate) != locator:
        raise ArtifactError("external artifact path must be canonical")
    if _hash(candidate) != (expected_size, expected_sha):
        raise ArtifactError(f"external artifact changed: {candidate}")
    return candidate


def snapshot_artifact(
    workspace: Path,
    path: Path,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    """Snapshot a workspace file with a portable relative path locator."""
    root = resolve_private_workspace(workspace)
    candidate = _workspace_path(root, path, must_exist=True)
    size_bytes, sha256 = _hash(candidate)
    return _identity(
        candidate.relative_to(root).as_posix(),
        size_bytes,
        sha256,
        schema_version=schema_version,
        kind=kind,
    )


def _workspace_identity_path(
    workspace: Path, identity: Mapping[str, Any]
) -> Path:
    locator, _size_bytes, _sha256 = _validate_identity(identity)
    pure = PurePosixPath(locator)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != locator
    ):
        raise ArtifactError(
            "workspace artifact path must be a canonical relative locator"
        )
    return _workspace_path(workspace, Path(*pure.parts), must_exist=True)


def verify_artifact(workspace: Path, identity: Mapping[str, Any]) -> Path:
    """Verify one workspace identity and return its canonical current path."""
    candidate = _workspace_identity_path(workspace, identity)
    _locator, expected_size, expected_sha = _validate_identity(identity)
    if _hash(candidate) != (expected_size, expected_sha):
        raise ArtifactError(f"workspace artifact changed: {candidate}")
    return candidate


def write_fresh_bytes(
    workspace: Path,
    path: Path,
    payload: bytes,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    """Write one new 0600 workspace artifact and return its identity."""
    if not isinstance(payload, bytes):
        raise ArtifactError("artifact payload must be bytes")
    _bound(len(payload), "artifact payload", MAX_ARTIFACT_BYTES)
    root = resolve_private_workspace(workspace)
    candidate = _workspace_path(root, path, must_exist=False)
    try:
        descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ArtifactError(f"cannot create fresh artifact {candidate}: {exc}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass
        raise ArtifactError(f"cannot write fresh artifact {candidate}: {exc}") from exc
    return snapshot_artifact(
        root,
        candidate,
        schema_version=schema_version,
        kind=kind,
    )


def write_fresh_json(workspace: Path, path: Path, value: Any) -> dict[str, Any]:
    """Write strict JSON, taking schema version and kind from its envelope."""
    if not isinstance(value, dict):
        raise ArtifactError("JSON artifact must be an object")
    return write_fresh_bytes(
        workspace,
        path,
        canonical_json_bytes(value),
        schema_version=_contract_string(value.get("schema_version"), "schema_version"),
        kind=_contract_string(value.get("kind"), "kind"),
    )


def _read_bounded_exact_bytes(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    max_bytes: int,
) -> bytes:
    """Read no more than one byte past an artifact's expected bounded size.

    Requesting exactly ``size_bytes + 1`` means an artifact that grew after its
    identity was recorded is never fully materialised: the extra byte only
    proves that it grew. Length and SHA-256 are both re-verified after the read,
    so the returned bytes are exactly the ones the identity names.
    """
    bound = _bound(max_bytes, "max_bytes", MAX_ARTIFACT_BYTES)
    size_bytes = _bound(expected_size, "size_bytes", bound)
    try:
        with path.open("rb") as stream:
            payload = stream.read(size_bytes + 1)
    except OSError as exc:
        raise ArtifactError(f"cannot read artifact {path}: {exc}") from exc
    if len(payload) > bound:
        raise ArtifactError(f"artifact exceeds the {bound}-byte read bound: {path}")
    if len(payload) != size_bytes:
        raise ArtifactError(f"artifact changed while it was read: {path}")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ArtifactError(f"artifact changed while it was read: {path}")
    return payload


def _verified_payload(
    workspace: Path,
    identity: Mapping[str, Any],
    maximum: int,
) -> bytes:
    bound = _bound(maximum, "max_bytes", MAX_ARTIFACT_BYTES)
    _locator, size_bytes, sha256 = _validate_identity(identity)
    _bound(size_bytes, "size_bytes", bound)
    path = verify_artifact(workspace, identity)
    return _read_bounded_exact_bytes(
        path,
        expected_size=size_bytes,
        expected_sha256=sha256,
        max_bytes=bound,
    )


def read_verified_bytes(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> bytes:
    """Verify and bounded-read one workspace byte artifact."""
    return _verified_payload(workspace, identity, max_bytes)


def read_verified_text(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> str:
    """Verify and bounded-read one UTF-8 workspace text artifact."""
    try:
        return _verified_payload(workspace, identity, max_bytes).decode(
            "utf-8", errors="strict"
        )
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"verified artifact is not UTF-8: {exc}") from exc


def read_verified_json(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Any:
    """Verify, bounded-read, and strictly parse one workspace JSON artifact."""
    return strict_json_loads(_verified_payload(workspace, identity, max_bytes))


def read_verified_external_json(
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Any:
    """Verify, bounded-read, and strictly parse one external JSON artifact."""
    bound = _bound(max_bytes, "max_bytes", MAX_ARTIFACT_BYTES)
    _locator, size_bytes, sha256 = _validate_identity(identity)
    _bound(size_bytes, "size_bytes", bound)
    path = verify_external_artifact(identity)
    return strict_json_loads(
        _read_bounded_exact_bytes(
            path,
            expected_size=size_bytes,
            expected_sha256=sha256,
            max_bytes=bound,
        )
    )


def _command_argv(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CommandError("argv must be a non-empty sequence of strings")
    copied = list(value)
    if not copied or not copied[0]:
        raise CommandError("argv[0] must be a non-empty executable name")
    for index, token in enumerate(copied):
        if not isinstance(token, str) or "\x00" in token:
            raise CommandError(f"argv[{index}] must be a NUL-free string")
    return copied


def _command_cwd(value: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CommandError(f"cwd is unavailable: {value}") from exc
    if not candidate.is_dir():
        raise CommandError(f"cwd is not a directory: {candidate}")
    return candidate


def _command_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CommandError("env must be an object of string names and values")
    copied: dict[str, str] = {}
    for key, item in value.items():
        invalid_key = (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
        )
        invalid_value = not isinstance(item, str) or "\x00" in item
        if invalid_key or invalid_value:
            raise CommandError("environment names and values must be valid strings")
        copied[key] = item
    return dict(sorted(copied.items()))


def _positive_seconds(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise CommandError(f"{label} must be a positive finite number")
    return float(value)


class CommandRunner:
    """Run no-shell commands in process groups with complete log evidence."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        termination_grace_seconds: float = 0.25,
    ) -> None:
        self.workspace = resolve_private_workspace(Path(workspace))
        self.log_directory = create_private_workspace(
            self.workspace / "command-logs"
        )
        self.termination_grace_seconds = _positive_seconds(
            termination_grace_seconds, "termination_grace_seconds"
        )
        self._sequence = 0

    def _paths(self, stage: str, phase: str) -> tuple[str, Path, Path]:
        self._sequence += 1
        stem = f"{self._sequence:06d}-{stage}-{phase}"
        return (
            stem,
            self.log_directory / f"{stem}.stdout.log",
            self.log_directory / f"{stem}.stderr.log",
        )

    @staticmethod
    def _tail(path: Path) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - _TAIL_BYTES), os.SEEK_SET)
                payload = stream.read(_TAIL_BYTES)
        except OSError as exc:
            raise CommandError(f"cannot read command log tail {path}: {exc}") from exc
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _terminate_group(process: subprocess.Popen[Any], grace: float) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    # pylint: disable=too-many-arguments,too-many-locals
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str],
        env: Mapping[str, str],
        phase: str,
        stage: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute one exact argv and return timing, exit, and log identities."""
        command = _command_argv(argv)
        command_cwd = _command_cwd(cwd)
        command_env = _command_environment(env)
        if phase not in COMMAND_PHASES:
            raise CommandError(f"phase must be exactly one of {list(COMMAND_PHASES)}")
        if not isinstance(stage, str) or _STAGE_PATTERN.fullmatch(stage) is None:
            raise CommandError("stage must be 1-64 safe ASCII characters")
        timeout = _positive_seconds(timeout_seconds, "timeout_seconds")
        command_id, stdout_path, stderr_path = self._paths(stage, phase)
        started_at = datetime.now(timezone.utc).isoformat()
        started_ns = time.monotonic_ns()
        process: subprocess.Popen[Any] | None = None
        timed_out = False
        launch_error: str | None = None
        exit_code: int | None = None
        write_fresh_bytes(
            self.log_directory,
            stdout_path,
            b"",
            schema_version="1",
            kind="command-stdout-log",
        )
        write_fresh_bytes(
            self.log_directory,
            stderr_path,
            b"",
            schema_version="1",
            kind="command-stderr-log",
        )
        with stdout_path.open("wb") as stdout_stream, stderr_path.open(
            "wb"
        ) as stderr_stream:
            try:
                process = subprocess.Popen(  # pylint: disable=consider-using-with
                    command,
                    cwd=str(command_cwd),
                    env=command_env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    shell=False,
                    start_new_session=True,
                )
                try:
                    exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate_group(process, self.termination_grace_seconds)
                    exit_code = process.returncode
            except OSError as exc:
                launch_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            self._terminate_group(process, self.termination_grace_seconds)
            exit_code = process.returncode
        ended_ns = time.monotonic_ns()
        stdout_identity = snapshot_artifact(
            self.workspace,
            stdout_path,
            schema_version="1",
            kind="command-stdout-log",
        )
        stderr_identity = snapshot_artifact(
            self.workspace,
            stderr_path,
            schema_version="1",
            kind="command-stderr-log",
        )
        return {
            "schema_version": "1",
            "kind": "command-result",
            "command_id": command_id,
            "stage": stage,
            "phase": phase,
            "argv": command,
            "cwd": str(command_cwd),
            "environment_keys": sorted(command_env),
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (ended_ns - started_ns) / 1_000_000_000,
            "approval_wait_seconds": 0.0,
            "timeout_seconds": timeout,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "launch_error": launch_error,
            "stdout": stdout_identity,
            "stderr": stderr_identity,
            "stdout_tail": self._tail(stdout_path),
            "stderr_tail": self._tail(stderr_path),
        }


def run_authenticated_sample(  # pylint: disable=too-many-arguments,too-many-locals
    token: Any,
    arguments: Sequence[str],
    *,
    token_type: type,
    error_type: type[ValueError],
    command_environment: Mapping[str, str],
    workspace: Path,
    runner: Any,
    cwd: Path,
    timeout_seconds: float,
    stage: str,
    phase: str,
) -> dict[str, Any]:
    """Rehash every authority seal immediately before an exact launch."""
    if not isinstance(token, token_type):
        raise error_type("official sample has no authenticated record")
    root = resolve_private_workspace(workspace)
    if root != Path(token.workspace).resolve(strict=True):
        raise error_type("launch workspace differs from authenticated workspace")
    command_cwd = Path(cwd).resolve(strict=True)
    try:
        command_cwd.relative_to(root)
    except ValueError as exc:
        raise error_type("official sample cwd must be inside workspace") from exc
    suffix = tuple(arguments)
    if any(not isinstance(item, str) or "\0" in item for item in suffix):
        raise error_type("official sample arguments must be NUL-free strings")
    verify_artifact(Path(token.environment_workspace), token.environment_identity)
    for item in (*token.protected_files, *token.runtime_libraries):
        verify_external_artifact(item.identity)
    executable = Path(token.launcher[0])
    if executable.stat().st_mode & 0o111 == 0:
        raise error_type("authenticated launcher is no longer executable")
    expected_kind = {
        "native": "native-official-sample-executable",
        "pynvc": "pynvc-venv-python",
    }.get(token.surface)
    launcher_seals = [
        seal
        for seal in token.protected_files
        if seal.identity.get("kind") == expected_kind
    ]
    if expected_kind is None or len(launcher_seals) != 1:
        raise error_type("authenticated launcher seal is absent or duplicated")
    if executable.resolve(strict=True) != Path(launcher_seals[0].identity["path"]):
        raise error_type("authenticated launcher resolves outside sealed target")
    argv = [*token.launcher, *suffix]
    result = runner.run(
        argv,
        cwd=command_cwd,
        env=command_environment,
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
        raise error_type("command evidence does not match authenticated launch")
    return result


def validate_environment_envelope(
    value: Any,
    *,
    label: str,
    error_type: type[ValueError],
    required_surfaces: Sequence[str] = SURFACE_ORDER,
) -> dict[str, Any]:
    """Accept one live environment envelope by required subset, ignoring extras.

    Additive optional keys are permitted by the schema, so this validates the
    required subset only and never compares the document's key set for
    equality. Any other version is an unknown version and is rejected.
    """
    selected = tuple(required_surfaces)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(surface not in SURFACE_ORDER for surface in selected)
    ):
        raise error_type(
            f"{label} required surfaces must be a non-empty subset of "
            f"{list(SURFACE_ORDER)}"
        )
    if not isinstance(value, dict):
        raise error_type(f"{label} environment must be a JSON object")
    if value.get("kind") != ENVIRONMENT_KIND:
        raise error_type(f"{label} artifact is not one {ENVIRONMENT_KIND} document")
    if value.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise error_type(
            f"{label} requires {ENVIRONMENT_KIND} schema "
            f"{ENVIRONMENT_SCHEMA_VERSION}, not {value.get('schema_version')!r}"
        )
    if value.get("mode") != "live":
        raise error_type(f"{label} requires one live environment artifact")
    if value.get("requested_runtime") not in ENVIRONMENT_RUNTIMES:
        raise error_type(
            f"{label} environment requested_runtime must be exactly one of "
            f"{list(ENVIRONMENT_RUNTIMES)}"
        )
    if not isinstance(value.get("installation"), Mapping):
        raise error_type(f"{label} environment has no installation facts")
    if "pynvc" in selected and not isinstance(value.get("pynvc"), Mapping):
        raise error_type(f"{label} environment has no selected pynvc facts")
    selected_gpu = value.get("selected_gpu")
    if (
        isinstance(selected_gpu, bool)
        or not isinstance(selected_gpu, int)
        or selected_gpu < 0
    ):
        raise error_type(f"{label} environment selected_gpu is malformed")
    return value


def validate_local_runtime_binding(
    value: Any,
    *,
    label: str,
    error_type: type[ValueError],
    required_surfaces: Sequence[str],
) -> dict[str, Any]:
    """Validate one consumer-owned read-only binding for selected surfaces."""
    selected = tuple(required_surfaces)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(surface not in SURFACE_ORDER for surface in selected)
    ):
        raise error_type(
            f"{label} required surfaces must be a non-empty subset of "
            f"{list(SURFACE_ORDER)}"
        )
    if not isinstance(value, dict):
        raise error_type(f"{label} local runtime binding must be a JSON object")
    expected = {
        "schema_version",
        "kind",
        "mode",
        "source",
        "requested_runtime",
        "selected_gpu",
        "installation",
        "pynvc",
    }
    if set(value) != expected:
        raise error_type(
            f"{label} local runtime binding fields must be exactly {sorted(expected)}"
        )
    if (
        value.get("schema_version") != LOCAL_RUNTIME_BINDING_SCHEMA_VERSION
        or value.get("kind") != LOCAL_RUNTIME_BINDING_KIND
        or value.get("mode") != "live"
    ):
        raise error_type(
            f"{label} requires live {LOCAL_RUNTIME_BINDING_KIND} "
            f"{LOCAL_RUNTIME_BINDING_SCHEMA_VERSION}"
        )
    if value.get("source") != "pipeline-local-discovery":
        raise error_type(f"{label} local binding source is invalid")
    selected_gpu = value.get("selected_gpu")
    if (
        isinstance(selected_gpu, bool)
        or not isinstance(selected_gpu, int)
        or selected_gpu < 0
    ):
        raise error_type(f"{label} local binding selected_gpu is malformed")
    if value.get("requested_runtime") not in ENVIRONMENT_RUNTIMES:
        raise error_type(
            f"{label} local binding requested_runtime must be exactly one of "
            f"{list(ENVIRONMENT_RUNTIMES)}"
        )
    if not isinstance(value.get("installation"), Mapping):
        raise error_type(f"{label} local binding has no installation facts")
    if "pynvc" in selected and not isinstance(value.get("pynvc"), Mapping):
        raise error_type(f"{label} local binding has no selected pynvc facts")
    return value


def _member(value: Any, key: str) -> Mapping[str, Any]:
    """Return one nested mapping member, or an empty mapping when it is absent."""
    member = value.get(key) if isinstance(value, Mapping) else None
    return member if isinstance(member, Mapping) else {}


def _normalized_prerequisites(record: Any) -> dict[str, Any] | None:
    """Recover the internal unresolved-module list from the 1.2 AppDec record.

    Schema 1.2 publishes ``missing_modules``/``unknown_modules`` and sets
    ``status`` to ``complete`` exactly when neither is populated and pkg-config
    itself was available, so the internal list is recoverable without loss. An
    incomplete record that itemizes nothing leaves every required module
    unproven.
    """
    if not isinstance(record, Mapping):
        return None
    if isinstance(record.get("unresolved_modules"), list):
        return dict(record)
    unresolved = sorted(
        {
            str(name)
            for key in ("missing_modules", "unknown_modules")
            for name in (record.get(key) or [])
        }
    )
    if not unresolved and record.get("status") != "complete":
        unresolved = [str(name) for name in (record.get("required_modules") or ["pkg-config"])]
    return {**record, "unresolved_modules": unresolved}


def _normalized_native(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the 1.2 ``installation`` tree into the internal native view."""
    installation = _member(environment, "installation")
    native = _member(installation, "native_sdk")
    cuda = _member(installation, "cuda_toolkit")
    identities = _member(installation, "build_tool_identities")
    roots = native.get("complete_roots")
    tools = {
        internal: dict(_member(identities, published))
        for internal, published in _PUBLISHED_BUILD_TOOLS
        if isinstance(identities.get(published), Mapping)
    }
    for published, generator in _PUBLISHED_GENERATORS:
        record = identities.get(published)
        if isinstance(record, Mapping) and _absolute(record.get("path")):
            tools["generator"] = {**record, "name": generator}
            break
    return {
        "installed": native.get("status") == "installed",
        "package": native.get("package"),
        # 1.2 records a root inventory; exactly one complete root is the
        # canonical SDK root, and an ambiguous inventory selects none.
        "sdk_root": roots[0] if isinstance(roots, list) and len(roots) == 1 else None,
        "build_prerequisites": _normalized_prerequisites(
            installation.get("native_build_prerequisites")
        ),
        "cuda": {
            "status": "installed" if cuda.get("status") == "available" else cuda.get("status"),
            "version": cuda.get("version"),
            "root": _member(cuda, "nvcc_discovery").get("root"),
        },
        "tools": tools,
    }


def _normalized_pynvc(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the 1.2 ``pynvc``/``installation.python`` trees into one view."""
    pynvc = _member(environment, "pynvc")
    identity = _member(pynvc, "identity")
    python = _member(_member(environment, "installation"), "python")
    packages = _member(python, "packages")
    return {
        "installed": (
            pynvc.get("imported") is True and identity.get("status") == "verified"
        ),
        "version": pynvc.get("distribution_version"),
        "interpreter": identity.get("interpreter"),
        # 1.2 requires the verified wheel interpreter to be the probe
        # interpreter, so its running identity is this surface's identity.
        "interpreter_identity": _member(python, "interpreter_identities").get("running"),
        "sys_prefix": identity.get("sys_prefix"),
        "extension": identity.get("extension"),
        "module": identity.get("module"),
        "dependencies": {
            name: {**record, "ready": record.get("requirement_satisfied") is True}
            for name, record in packages.items()
            if isinstance(record, Mapping)
        },
    }


def environment_surface(
    environment: Mapping[str, Any], name: str
) -> dict[str, Any] | None:
    """Return one installed independent surface, or None when it is absent.

    Schema 1.2 publishes ``installation`` and ``pynvc``, never a ``surfaces``
    key, so the per-surface view stays an internal normalized representation
    derived here. A document that additively carries an already normalized
    ``surfaces`` mapping is used as supplied. Surfaces stay structurally
    independent: either may be absent or report ``installed: false`` with no
    effect on the other.
    """
    if not isinstance(environment, Mapping):
        return None
    recognized_kind = environment.get("kind") in {
        ENVIRONMENT_KIND,
        LOCAL_RUNTIME_BINDING_KIND,
    }
    if recognized_kind and name == "native":
        surface = _normalized_native(environment)
    elif recognized_kind and name == "pynvc":
        surface = _normalized_pynvc(environment)
    else:
        surface = None
    if not isinstance(surface, Mapping) or surface.get("installed") is not True:
        return None
    return dict(surface)


def read_live_environment(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
    error_type: type[ValueError],
    required_surfaces: Sequence[str] = SURFACE_ORDER,
) -> dict[str, Any]:
    """Read one supplied setup envelope or consumer-owned local binding."""
    value = read_verified_json(workspace, identity)
    if isinstance(value, Mapping) and value.get("kind") == LOCAL_RUNTIME_BINDING_KIND:
        return validate_local_runtime_binding(
            value,
            label=f"{label} authentication",
            error_type=error_type,
            required_surfaces=required_surfaces,
        )
    return validate_environment_envelope(
        value,
        label=f"{label} authentication",
        error_type=error_type,
        required_surfaces=required_surfaces,
    )


def _text(value: Any) -> bool:
    """Return whether one value is a non-empty canonical string."""
    return isinstance(value, str) and bool(value) and value == value.strip()


def _absolute(value: Any) -> bool:
    return _text(value) and PurePosixPath(str(value)).is_absolute()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _extension_defects(surface: Mapping[str, Any]) -> list[str]:
    """Require the authenticated extension identity, and honour a loaded path.

    Schema 1.2 authenticates the extension by wheel-member identity and does
    not publish which file the interpreter actually loaded, so ``loaded_path``
    is additive: when a document does carry it, it must name the very same
    file, and it is never inferred when absent. The live isolated RECORD probe
    remains the authority for what was loaded.
    """
    extension = surface.get("extension")
    if not isinstance(extension, Mapping):
        return ["pynvc extension identity is missing"]
    if not (_absolute(extension.get("path")) and _sha256(extension.get("sha256"))):
        return ["pynvc extension path or sha256 is missing or malformed"]
    loaded = extension.get("loaded_path")
    if loaded is None:
        return []
    if not _absolute(loaded):
        return ["pynvc extension loaded_path is malformed"]
    if loaded != extension["path"]:
        return [
            "the interpreter loaded a different extension than the authenticated one: "
            f"{loaded!r} is not {extension['path']!r}"
        ]
    return []


def _module_defects(surface: Mapping[str, Any]) -> list[str]:
    """Require the imported module to agree with the distribution version."""
    module = surface.get("module")
    if not isinstance(module, Mapping):
        return ["pynvc imported module identity is missing"]
    if not (_text(module.get("version")) and _absolute(module.get("path"))):
        return ["pynvc module version or path is missing or malformed"]
    if module["version"] != surface.get("version"):
        return [
            "imported module version differs from the distribution version: "
            f"{module['version']!r} is not {surface.get('version')!r}"
        ]
    return []


def native_surface_defects(environment: Mapping[str, Any]) -> list[str]:
    """Structurally validate the native surface's required subset at planning.

    Additive keys are ignored and no key set is compared for equality. A
    malformed native surface is blocked here rather than at execute-time
    authentication, so the Py surface stays independently actionable.
    """
    surface = environment_surface(environment, "native")
    if surface is None:
        return ["native surface is absent or does not report installed"]
    defects: list[str] = []
    package = surface.get("package")
    if (
        not isinstance(package, Mapping)
        or package.get("name") != NATIVE_PACKAGE_NAME
        or package.get("status") != "installed"
        or not _text(package.get("version"))
    ):
        defects.append("native package name, status, or version is missing or malformed")
    if not _absolute(surface.get("sdk_root")):
        defects.append("native sdk_root is missing or is not an absolute path")
    prerequisites = surface.get("build_prerequisites")
    if (
        not isinstance(prerequisites, Mapping)
        or prerequisites.get("status") != "complete"
        or prerequisites.get("unresolved_modules") != []
    ):
        defects.append("native build prerequisites are incomplete or malformed")
    cuda = surface.get("cuda")
    if (
        not isinstance(cuda, Mapping)
        or cuda.get("status") != "installed"
        or not _text(cuda.get("version"))
    ):
        defects.append("native CUDA status or version is missing or malformed")
    tools = surface.get("tools")
    if not isinstance(tools, Mapping):
        defects.append("native build tools are missing")
    else:
        # 1.2 identifies each build tool by path and content hash and publishes
        # no tool version, so a version is honoured when present and never
        # required. The content hash is what the launch is bound to.
        incomplete = [
            name
            for name in REQUIRED_BUILD_TOOLS
            if not isinstance(tools.get(name), Mapping)
            or not _absolute(tools[name].get("path"))
            or not _sha256(tools[name].get("sha256"))
            or (tools[name].get("version") is not None
                and not _text(tools[name].get("version")))
            or (
                name == "generator"
                and not _text(tools[name].get("name"))
            )
        ]
        if incomplete:
            defects.append(
                "native build tools are missing a required path or SHA-256: "
                f"{incomplete}"
            )
    return defects


def pynvc_surface_defects(environment: Mapping[str, Any]) -> list[str]:
    """Structurally validate the Py surface's required subset at planning."""
    surface = environment_surface(environment, "pynvc")
    if surface is None:
        return ["pynvc surface is absent or does not report installed"]
    defects: list[str] = []
    if not _text(surface.get("version")):
        defects.append("pynvc version is missing or malformed")
    if not _absolute(surface.get("interpreter")):
        defects.append("pynvc lexical interpreter is missing or is not an absolute path")
    identity = surface.get("interpreter_identity")
    if (
        not isinstance(identity, Mapping)
        or not _absolute(identity.get("path"))
        or not _sha256(identity.get("sha256"))
    ):
        defects.append("pynvc interpreter_identity path or sha256 is missing or malformed")
    if not _absolute(surface.get("sys_prefix")):
        defects.append("pynvc sys_prefix is missing or is not an absolute path")
    defects.extend(_extension_defects(surface))
    defects.extend(_module_defects(surface))
    return defects


def pynvc_full_sample_defects(environment: Mapping[str, Any]) -> list[str]:
    """Require the verified dependency closure used by full PyNv sample routes."""
    defects = pynvc_surface_defects(environment)
    surface = environment_surface(environment, "pynvc")
    if surface is None:
        return defects
    dependencies = surface.get("dependencies")
    dependencies = dependencies if isinstance(dependencies, Mapping) else {}
    incomplete = [
        name
        for name in ("numpy", "pycuda")
        if not isinstance(dependencies.get(name), Mapping)
        or dependencies[name].get("status") != "installed"
        or dependencies[name].get("ready") is not True
    ]
    torch = dependencies.get("torch")
    if not (
        isinstance(torch, Mapping)
        and torch.get("status") == "installed"
        and torch.get("version") == "2.9.1+cu130"
        and torch.get("cuda_build") == "13.0"
        and torch.get("cuda_available") is True
        and torch.get("ready") is True
    ):
        incomplete.append("torch")
    if incomplete:
        defects.append(
            "authenticated Py full-samples routes require ready NumPy and PyCUDA plus exact "
            "CUDA-enabled Torch 2.9.1+cu130; incomplete dependencies: "
            + ", ".join(sorted(set(incomplete)))
            + ". A jetson-video-setup pynvc-smoke environment deliberately omits Torch. "
            "Provision a NEW full-samples PyNvVideoCodec venv with plan_install.py "
            "--profile full-samples --venv NEW_PATH; never upgrade the smoke venv in place."
        )
    return defects


def run_prepare_command(  # pylint: disable=too-many-arguments
    service: Any,
    commands: list[dict[str, Any]],
    argv: Sequence[str],
    *,
    command_environment: Mapping[str, str],
    error_type: type[ValueError],
    cwd: Path,
    timeout_seconds: float,
    stage: str,
) -> tuple[dict[str, Any], str, str]:
    """Run one authenticated preparation command and consume its evidence."""
    result = service.run(
        list(argv),
        cwd=cwd,
        env=command_environment,
        timeout_seconds=timeout_seconds,
        stage=stage,
        phase="prepare",
    )
    if not isinstance(result, dict):
        raise error_type("command runner returned a non-object result")
    commands.append(result)
    stdout = read_verified_text(service.workspace, result["stdout"])
    stderr = read_verified_text(service.workspace, result["stderr"])
    if (
        result.get("timed_out")
        or result.get("launch_error")
        or result.get("exit_code") != 0
    ):
        raise error_type(
            f"command failed during {stage}: exit={result.get('exit_code')} "
            f"timeout={result.get('timed_out')}: {stderr[-500:]}"
        )
    return result, stdout, stderr


def _candidate(value: Any, surface: str) -> tuple[bool, list[str]]:
    if isinstance(value, bool):
        return value, [] if value else [f"{surface} is not eligible"]
    if not isinstance(value, Mapping) or set(value) != _ELIGIBILITY_KEYS:
        raise ValueError(
            f"eligibility.{surface} must contain exactly eligible and reasons"
        )
    eligible = value["eligible"]
    reasons = value["reasons"]
    if not isinstance(eligible, bool) or not isinstance(reasons, list):
        raise ValueError(f"eligibility.{surface} has invalid types")
    if any(
        not isinstance(reason, str)
        or not reason
        or reason != reason.strip()
        for reason in reasons
    ):
        raise ValueError(f"eligibility.{surface}.reasons are invalid")
    if len(set(reasons)) != len(reasons):
        raise ValueError(f"eligibility.{surface}.reasons contain duplicates")
    if eligible and reasons:
        raise ValueError(f"eligible surface {surface} must not contain reasons")
    return eligible, reasons or ([] if eligible else [f"{surface} is not eligible"])


def build_surface_plan(
    requested_surface: str,
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact native/Py surface-selection truth table."""
    if requested_surface not in REQUESTED_SURFACES:
        raise ValueError(
            f"requested_surface must be exactly one of {list(REQUESTED_SURFACES)}"
        )
    if not isinstance(eligibility, Mapping) or set(eligibility) != set(SURFACE_ORDER):
        raise ValueError(f"eligibility fields must be exactly {list(SURFACE_ORDER)}")
    eligible: list[str] = []
    surface_reasons: dict[str, list[str]] = {}
    for surface in SURFACE_ORDER:
        ready, reasons = _candidate(eligibility[surface], surface)
        if ready:
            eligible.append(surface)
        surface_reasons[surface] = reasons
    selected: list[str] = []
    reasons: list[str] = []
    if requested_surface in SURFACE_ORDER:
        if requested_surface in eligible:
            classification = "ready"
            selected = [requested_surface]
        else:
            classification = "blocked"
            reasons = list(surface_reasons[requested_surface])
    elif requested_surface == "auto":
        if len(eligible) == 1:
            classification = "ready"
            selected = list(eligible)
        elif len(eligible) == 2:
            classification = "selection_required"
            reasons = [
                "auto requires an explicit surface because native and pynvc are eligible"
            ]
        else:
            classification = "blocked"
            reasons = [
                reason
                for surface in SURFACE_ORDER
                for reason in surface_reasons[surface]
            ]
    else:
        selected = list(eligible)
        classification = "ready" if len(eligible) == 2 else "blocked"
        reasons = [
            reason
            for surface in SURFACE_ORDER
            if surface not in eligible
            for reason in surface_reasons[surface]
        ]
    return {
        "requested_surface": requested_surface,
        "classification": classification,
        "selected_surfaces": selected,
        "eligible_surfaces": eligible,
        "reasons": reasons,
    }


def _recipe_cli() -> Path:
    """Resolve the fixed sibling recipe public CLI without importing it."""
    skill_root = (
        Path(__file__).absolute().parents[2]
        / "jetson-video-recipe"
    )
    candidate = (
        skill_root
        / "scripts"
        / "recipes"
        / "recipe_model.py"
    )
    try:
        resolved_root = skill_root.resolve(strict=True)
        details = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SkillDependencyError(
            "jetson-video-recipe", "this recipe-bearing pipeline stage"
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise RecipeBridgeError("the recipe public CLI must be a regular file")
    if resolved != resolved_root / "scripts" / "recipes" / "recipe_model.py":
        raise RecipeBridgeError("the recipe public CLI path is not canonical")
    return resolved


def _recipe_command(
    arguments: Sequence[str],
    timeout_seconds: float = 60,
    *,
    accepted_returncodes: Sequence[int] = (0,),
) -> Any:
    """Execute and strictly consume one size-checked recipe CLI JSON response."""
    argv = [sys.executable, "-I", str(_recipe_cli()), *arguments]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(Path(__file__).resolve().parent),
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecipeBridgeError(f"recipe public CLI launch failed: {exc}") from exc
    if (
        len(completed.stdout) > _RECIPE_OUTPUT_LIMIT
        or len(completed.stderr) > _RECIPE_OUTPUT_LIMIT
    ):
        raise RecipeBridgeError("recipe public CLI output exceeded its bound")
    try:
        value = strict_json_loads(completed.stdout)
    except ValueError as exc:
        raise RecipeBridgeError(
            "recipe public CLI did not return one strict JSON response"
        ) from exc
    if completed.returncode not in accepted_returncodes:
        detail = value.get("error") if isinstance(value, dict) else None
        raise RecipeBridgeError(
            f"recipe public CLI rejected the request: {detail or completed.returncode}"
        )
    return value


def validate_recipe_identity(identity: Mapping[str, Any]) -> None:
    """Replay one current recipe through its owning public CLI."""
    recipe = verify_external_artifact(identity)
    value = _recipe_command(("validate", "--recipe", str(recipe)))
    if value != {"valid": True, "errors": []}:
        raise RecipeBridgeError("recipe public CLI returned an invalid validation envelope")


def _bind_capability_report(
    capability_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
) -> Path:
    """Require the report to be the one produced from this exact environment.

    The report records the environment artifact's ``{path, size_bytes,
    sha256}``. Recipe fails closed on a mismatch; the pairing is also refused
    here so a cached report is never sent alongside a fresh environment.
    """
    capability = verify_external_artifact(capability_identity)
    report = read_verified_external_json(capability_identity)
    if (
        not isinstance(report, dict)
        or report.get("kind") != CAPABILITY_REPORT_KIND
        or report.get("schema_version") != CAPABILITY_REPORT_SCHEMA_VERSION
    ):
        raise RecipeBridgeError(
            f"capability report must be schema {CAPABILITY_REPORT_SCHEMA_VERSION} "
            f"kind {CAPABILITY_REPORT_KIND}"
        )
    bound = report.get("environment")
    if not isinstance(bound, Mapping):
        raise RecipeBridgeError("capability report records no bound environment")
    if any(
        bound.get(field) != environment_identity.get(field)
        for field in ("path", "size_bytes", "sha256")
    ):
        raise RecipeBridgeError(
            "capability report was produced from a different environment artifact"
        )
    return capability


def check_live_recipe(
    recipe_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any] | None,
    surface: str,
    *,
    buffer_mode: str,
    capability_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one projection through the recipe skill's public CLI.

    The recipe is required. A supplied setup environment is an optional but
    authoritative fast path; when absent, the recipe owner performs only its
    own media-free projection check. A capability report can be accepted only
    with the exact environment to which it is bound.
    """
    recipe = verify_external_artifact(recipe_identity)
    arguments = [
        "check-live",
        "--recipe",
        str(recipe),
        "--surface",
        surface,
        "--buffer-mode",
        buffer_mode,
    ]
    if environment_identity is not None:
        environment = verify_external_artifact(environment_identity)
        arguments.extend(["--environment", str(environment)])
    if capability_identity is not None:
        if environment_identity is None:
            raise RecipeBridgeError(
                "a capability report requires its exact supplied environment identity"
            )
        capability = _bind_capability_report(capability_identity, environment_identity)
        arguments.extend(["--capability-report", str(capability)])
    value = _recipe_command(tuple(arguments), accepted_returncodes=(0, 2))
    if (
        not isinstance(value, dict)
        or value.get("classification") not in {"compatible", "unknown", "unsupported"}
        or not isinstance(value.get("reasons"), list)
    ):
        raise RecipeBridgeError("recipe public CLI returned an invalid live envelope")
    return value
