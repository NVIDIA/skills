#!/usr/bin/env python3
"""Private artifact, command-evidence, and surface-routing runtime."""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The routing truth table and exact command shape are intentionally explicit.
# pylint: disable=too-many-boolean-expressions,too-many-branches

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
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
_PHASES = {
    "probe",
    "plan",
    "prepare",
    "install",
    "warmup",
    "measure",
    "verify",
    "execute",
}
_SURFACES = ("native", "pynvc")
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


class ArtifactError(ValueError):
    """Raised when an artifact fails its portable content contract."""


class CommandError(ValueError):
    """Raised when a command violates the execution-evidence contract."""


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
    """Decode strict UTF-8 JSON without duplicates or non-finite values."""
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
    """Encode deterministic strict JSON with one trailing newline."""
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
    try:
        candidate = Path(_contract_path(str(path))).expanduser().resolve(strict=True)
        details = candidate.stat()
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"artifact is not an accessible file: {path}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise ArtifactError(f"artifact is not a regular file: {candidate}")
    return candidate


def create_private_workspace(path: Path) -> Path:
    """Create one fresh canonical workspace with exact mode 0700."""
    requested = Path(_contract_path(str(path))).expanduser()
    try:
        parent = requested.parent.resolve(strict=True)
        candidate = parent / requested.name
        os.mkdir(candidate, mode=0o700)
        os.chmod(candidate, 0o700)
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"cannot create fresh workspace {requested}: {exc}") from exc
    return resolve_private_workspace(candidate)


def resolve_private_workspace(path: Path) -> Path:
    """Resolve an existing workspace and require exact mode 0700."""
    try:
        candidate = Path(_contract_path(str(path))).expanduser().resolve(strict=True)
        details = candidate.stat()
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"workspace is unavailable: {path}") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ArtifactError(f"workspace is not a directory: {candidate}")
    if stat.S_IMODE(details.st_mode) != 0o700:
        raise ArtifactError(f"workspace mode must be exactly 0700: {candidate}")
    return candidate


def _workspace_path(workspace: Path, path: Path, *, must_exist: bool) -> Path:
    root = resolve_private_workspace(workspace)
    requested = Path(_contract_path(str(path)))
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactError(f"artifact is outside or unavailable: {candidate}") from exc
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
            while chunk := stream.read(_HASH_CHUNK_BYTES):
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
    locator: str,
    size_bytes: int,
    sha256: str,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": _contract_string(schema_version, "schema_version"),
        "kind": _contract_string(kind, "kind"),
        "path": _contract_path(locator),
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
    digest = identity["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactError("sha256 must be 64 lowercase hexadecimal characters")
    return locator, size_bytes, digest


def snapshot_external_artifact(
    path: Path, *, schema_version: str, kind: str
) -> dict[str, Any]:
    """Snapshot an external file with a canonical absolute locator."""
    candidate = _existing_regular(path)
    size_bytes, digest = _hash(candidate)
    return _identity(
        str(candidate),
        size_bytes,
        digest,
        schema_version=schema_version,
        kind=kind,
    )


def verify_external_artifact(identity: Mapping[str, Any]) -> Path:
    """Verify an external identity and return its current canonical path."""
    locator, expected_size, expected_digest = _validate_identity(identity)
    if not Path(locator).is_absolute():
        raise ArtifactError("external artifact path must be absolute")
    candidate = _existing_regular(Path(locator))
    if str(candidate) != locator:
        raise ArtifactError("external artifact path must be canonical")
    if _hash(candidate) != (expected_size, expected_digest):
        raise ArtifactError(f"external artifact changed: {candidate}")
    return candidate


def snapshot_artifact(
    workspace: Path,
    path: Path,
    *,
    schema_version: str,
    kind: str,
) -> dict[str, Any]:
    """Snapshot a workspace file with a portable relative locator."""
    root = resolve_private_workspace(workspace)
    candidate = _workspace_path(root, path, must_exist=True)
    size_bytes, digest = _hash(candidate)
    return _identity(
        candidate.relative_to(root).as_posix(),
        size_bytes,
        digest,
        schema_version=schema_version,
        kind=kind,
    )


def _workspace_identity_path(
    workspace: Path, identity: Mapping[str, Any]
) -> Path:
    locator, _size, _digest = _validate_identity(identity)
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
    """Verify one workspace identity and return its current path."""
    candidate = _workspace_identity_path(workspace, identity)
    _locator, expected_size, expected_digest = _validate_identity(identity)
    if _hash(candidate) != (expected_size, expected_digest):
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
        root, candidate, schema_version=schema_version, kind=kind
    )


def write_fresh_json(
    workspace: Path, path: Path, value: Any
) -> dict[str, Any]:
    """Write a strict JSON object using its envelope for identity metadata."""
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
        raise ArtifactError(f"cannot read verified artifact {path}: {exc}") from exc
    if len(payload) > bound:
        raise ArtifactError(f"artifact exceeds the {bound}-byte read bound: {path}")
    if len(payload) != size_bytes:
        raise ArtifactError(f"artifact changed while it was read: {path}")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ArtifactError(f"artifact changed while it was read: {path}")
    return payload


def _verified_payload(
    workspace: Path, identity: Mapping[str, Any], maximum: int
) -> bytes:
    bound = _bound(maximum, "max_bytes", MAX_ARTIFACT_BYTES)
    _locator, size_bytes, digest = _validate_identity(identity)
    _bound(size_bytes, "size_bytes", bound)
    path = verify_artifact(workspace, identity)
    return _read_bounded_exact_bytes(
        path,
        expected_size=size_bytes,
        expected_sha256=digest,
        max_bytes=bound,
    )


def read_verified_text(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> str:
    """Verify and bounded-read a UTF-8 workspace artifact."""
    try:
        return _verified_payload(workspace, identity, max_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"verified artifact is not UTF-8: {exc}") from exc


def read_verified_json(
    workspace: Path,
    identity: Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Any:
    """Verify, bounded-read, and strictly parse workspace JSON."""
    return strict_json_loads(_verified_payload(workspace, identity, max_bytes))


def read_verified_external_json(
    identity: Mapping[str, Any], *, max_bytes: int = MAX_JSON_BYTES
) -> Any:
    """Verify, bounded-read, and strictly parse external JSON."""
    bound = _bound(max_bytes, "max_bytes", MAX_ARTIFACT_BYTES)
    _locator, size_bytes, digest = _validate_identity(identity)
    _bound(size_bytes, "size_bytes", bound)
    path = verify_external_artifact(identity)
    return strict_json_loads(
        _read_bounded_exact_bytes(
            path,
            expected_size=size_bytes,
            expected_sha256=digest,
            max_bytes=bound,
        )
    )


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
        unresolved = [
            str(name) for name in (record.get("required_modules") or ["pkg-config"])
        ]
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
        if isinstance(record, Mapping) and isinstance(record.get("path"), str):
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
            "status": (
                "installed" if cuda.get("status") == "available" else cuda.get("status")
            ),
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
    extension = dict(_member(identity, "extension"))
    extension_record = _member(extension, "record")
    extension["sha256"] = extension.get("sha256") or extension_record.get("sha256")
    extension["size_bytes"] = (
        extension.get("size_bytes") or extension_record.get("size_bytes")
    )
    return {
        "installed": (
            pynvc.get("imported") is True and identity.get("status") == "verified"
        ),
        "version": pynvc.get("distribution_version"),
        "interpreter": python.get("executable") or identity.get("interpreter"),
        # 1.2 requires the verified wheel interpreter to be the probe
        # interpreter, so its running identity is this surface's identity.
        "interpreter_identity": _member(python, "interpreter_identities").get("running"),
        "sys_prefix": identity.get("sys_prefix"),
        "extension": extension,
        "module": identity.get("module"),
        "dependencies": {
            name: {**record, "ready": record.get("requirement_satisfied") is True}
            for name, record in packages.items()
            if isinstance(record, Mapping)
        },
    }


def environment_surface(
    environment: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    """Return one independent surface view of a 1.2 environment, or None.

    Schema 1.2 publishes ``installation`` and ``pynvc``, never a ``surfaces``
    key, so every serialized document is normalized only from those raw facts.
    A supplied top-level ``surfaces`` member is never authoritative. Surfaces
    are structurally independent: either may be absent without affecting the
    other.
    """
    if not isinstance(environment, Mapping):
        return None
    if name == "native":
        return _normalized_native(environment)
    if name == "pynvc":
        return _normalized_pynvc(environment)
    return None


def _positive_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"{label} must be a positive finite number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise CommandError(f"{label} must be a positive finite number")
    return seconds


def _command_argv(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CommandError("argv must be a non-empty sequence of strings")
    copied = list(value)
    if not copied or not copied[0]:
        raise CommandError("argv[0] must be a non-empty executable name")
    if any(not isinstance(token, str) or "\x00" in token for token in copied):
        raise CommandError("argv tokens must be NUL-free strings")
    return copied


def _command_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CommandError("env must be an object of string names and values")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(item, str)
            or "\x00" in item
        ):
            raise CommandError("env names and values must be canonical NUL-free strings")
        copied[key] = item
    return dict(sorted(copied.items()))


class CommandRunner:
    """Run exact argv without a shell and retain bounded complete evidence."""

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
                stream.seek(max(0, stream.tell() - _TAIL_BYTES), os.SEEK_SET)
                payload = stream.read(_TAIL_BYTES)
        except OSError as exc:
            raise CommandError(f"cannot read command log tail {path}: {exc}") from exc
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _terminate(process: subprocess.Popen[Any], grace: float) -> None:
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
        """Execute one validated argv and return timing, exit, and log identities."""
        command = _command_argv(argv)
        try:
            command_cwd = Path(cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CommandError(f"cwd is unavailable: {cwd}") from exc
        if not command_cwd.is_dir():
            raise CommandError(f"cwd is not a directory: {command_cwd}")
        command_env = _command_environment(env)
        if phase not in _PHASES:
            raise CommandError(f"phase must be exactly one of {sorted(_PHASES)}")
        if not isinstance(stage, str) or _STAGE_PATTERN.fullmatch(stage) is None:
            raise CommandError("stage must be 1-64 canonical ASCII characters")
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
                    self._terminate(process, self.termination_grace_seconds)
                    exit_code = process.returncode
            except OSError as exc:
                launch_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            self._terminate(process, self.termination_grace_seconds)
            exit_code = process.returncode
        ended_at = datetime.now(timezone.utc).isoformat()
        duration = (time.monotonic_ns() - started_ns) / 1_000_000_000
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
            "ended_at": ended_at,
            "duration_seconds": duration,
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


def _eligibility_candidate(value: Any, surface: str) -> tuple[bool, list[str]]:
    if isinstance(value, bool):
        return value, [] if value else [f"{surface} is not eligible"]
    if not isinstance(value, Mapping) or set(value) != {"eligible", "reasons"}:
        raise ValueError(
            f"eligibility.{surface} must contain exactly eligible and reasons"
        )
    eligible = value["eligible"]
    reasons = value["reasons"]
    if not isinstance(eligible, bool) or not isinstance(reasons, list):
        raise ValueError(f"eligibility.{surface} has invalid fields")
    if any(
        not isinstance(reason, str) or not reason or reason != reason.strip()
        for reason in reasons
    ):
        raise ValueError(f"eligibility.{surface}.reasons are not canonical")
    if len(set(reasons)) != len(reasons):
        raise ValueError(f"eligibility.{surface}.reasons contain duplicates")
    if eligible and reasons:
        raise ValueError(f"eligible surface {surface} must not have reasons")
    return eligible, reasons or ([] if eligible else [f"{surface} is not eligible"])


def build_surface_plan(
    requested_surface: str, eligibility: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact native/Py/auto/both routing truth table."""
    if requested_surface not in (*_SURFACES, "auto", "both"):
        raise ValueError("requested_surface must be native, pynvc, auto, or both")
    if not isinstance(eligibility, Mapping) or set(eligibility) != set(_SURFACES):
        raise ValueError(f"eligibility fields must be exactly {list(_SURFACES)}")
    candidates = {
        surface: _eligibility_candidate(eligibility[surface], surface)
        for surface in _SURFACES
    }
    eligible = [surface for surface in _SURFACES if candidates[surface][0]]
    selected: list[str] = []
    reasons: list[str] = []
    if requested_surface in _SURFACES:
        if requested_surface in eligible:
            classification, selected = "ready", [requested_surface]
        else:
            classification = "blocked"
            reasons.extend(candidates[requested_surface][1])
    elif requested_surface == "auto":
        if len(eligible) == 1:
            classification, selected = "ready", list(eligible)
        elif len(eligible) == 2:
            classification = "selection_required"
            reasons.append(
                "auto requires an explicit surface because native and pynvc are eligible"
            )
        else:
            classification = "blocked"
            for surface in _SURFACES:
                reasons.extend(candidates[surface][1])
    else:
        selected = list(eligible)
        classification = "ready" if len(eligible) == 2 else "blocked"
        for surface in _SURFACES:
            if surface not in eligible:
                reasons.extend(candidates[surface][1])
    return {
        "requested_surface": requested_surface,
        "classification": classification,
        "selected_surfaces": selected,
        "eligible_surfaces": eligible,
        "reasons": reasons,
    }
