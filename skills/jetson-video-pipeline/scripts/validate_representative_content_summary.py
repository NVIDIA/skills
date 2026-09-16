#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate compact-review representative-content metadata against fresh evidence."""

# pylint: disable=missing-function-docstring,too-many-branches,too-many-locals,too-many-statements
# pylint: disable=unidiomatic-typecheck

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if not sys.flags.ignore_environment or not sys.flags.no_user_site:
    raise SystemExit("invoke this producer with isolated Python: python3 -I")

SCHEMA_VERSION = "1.0"
KIND = "nvcodec-representative-content-summary-validation"
MAX_JSON_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPRESENTATIVE_CONTENT_FIELDS = (
    "source_url",
    "license",
    "attribution",
    "path",
    "size_bytes",
    "sha256",
)
SOURCE_ARTIFACT_FIELDS = (
    "kind",
    *REPRESENTATIVE_CONTENT_FIELDS,
    "expected_size_bytes",
    "expected_sha256",
    "verified",
)


class ReviewValidationError(ValueError):
    """An input cannot be safely read as current review evidence."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-RFC JSON numeric constant is forbidden: {value}")


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Render strict finite JSON with caller-selected deterministic options."""
    _require_finite_json(value)
    return json.dumps(value, allow_nan=False, **kwargs)


def exclusive_write_bytes(
    path: Path, content: bytes, *, mode: int = 0o600
) -> Path:
    """Create one fresh regular file without following path symlinks."""
    requested = path.expanduser().absolute()
    if not requested.parent.is_dir():
        raise FileNotFoundError(
            f"evidence parent directory does not exist: {requested.parent}"
        )
    current = Path(requested.anchor)
    for part in requested.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"evidence parent contains a symlink component: {current}"
            )
    resolved = requested.parent.resolve(strict=True) / requested.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        resolved.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return resolved


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        _require_finite_json(value)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ReviewValidationError(f"{label} cannot be parsed strictly: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewValidationError(f"{label} must contain one JSON object")
    return value


def _require_finite_json(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number is forbidden at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{location}[{index}]")


def _valid_canonical_absolute_path_text(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    try:
        value.encode("utf-8")
        candidate = Path(value)
    except (UnicodeError, ValueError):
        return False
    return candidate.is_absolute() and str(candidate) == value


def _canonical_existing_path(path: Path, label: str) -> Path:
    path_text = str(path)
    if not _valid_canonical_absolute_path_text(path_text):
        raise ReviewValidationError(
            f"{label} path must already be one canonical absolute path"
        )
    requested = Path(path_text)
    current = Path(requested.anchor)
    try:
        for part in requested.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ReviewValidationError(
                    f"{label} path contains a symlink component: {current}"
                )
        resolved = requested.resolve(strict=True)
    except ReviewValidationError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise ReviewValidationError(f"{label} does not resolve: {exc}") from exc
    if requested != resolved:
        raise ReviewValidationError(
            f"{label} path must already be one canonical absolute path: {requested}"
        )
    return resolved


def _open_anchored_regular(path: Path, label: str) -> tuple[int, int]:
    """Open a file through a held, no-symlink parent-directory chain."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_descriptor = -1
    descriptor = -1
    try:
        parent_descriptor = os.open(path.anchor, directory_flags)
        for component in path.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = child
        if not path.name:
            raise ReviewValidationError(f"{label} must name one regular file")
        descriptor = os.open(path.name, file_flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ReviewValidationError(
                f"{label} must be one stable, non-symlink regular file"
            )
        return descriptor, parent_descriptor
    except ReviewValidationError:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise ReviewValidationError(f"{label} cannot be opened safely: {exc}") from exc


def _read_authenticated_file(
    path: Path,
    label: str,
    *,
    capture: bool,
    expected_size: int | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    resolved = _canonical_existing_path(path, label)
    descriptor, parent_descriptor = _open_anchored_regular(resolved, label)

    captured = bytearray() if capture else None
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ReviewValidationError(f"{label} must be a regular file")
        if expected_size is not None and initial.st_size != expected_size:
            raise ReviewValidationError(
                f"{label} current size differs before hashing: "
                f"expected {expected_size}, observed {initial.st_size}"
            )
        if capture and initial.st_size > MAX_JSON_BYTES:
            raise ReviewValidationError(
                f"{label} exceeds the {MAX_JSON_BYTES}-byte strict JSON limit"
            )
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
            if captured is not None:
                if observed_size > MAX_JSON_BYTES:
                    raise ReviewValidationError(
                        f"{label} exceeds the {MAX_JSON_BYTES}-byte strict JSON limit"
                    )
                captured.extend(chunk)
        terminal = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(initial, field) != getattr(terminal, field)
            for field in identity_fields
        ):
            raise ReviewValidationError(f"{label} changed while it was being read")
        try:
            named = os.stat(
                resolved.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _canonical_existing_path(resolved, label)
            visible = resolved.lstat()
        except ReviewValidationError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise ReviewValidationError(
                f"{label} disappeared after it was read: {exc}"
            ) from exc
        for observed in (named, visible):
            if (
                not stat.S_ISREG(observed.st_mode)
                or (observed.st_dev, observed.st_ino)
                != (terminal.st_dev, terminal.st_ino)
            ):
                raise ReviewValidationError(
                    f"{label} path changed while it was being read"
                )
        if observed_size != terminal.st_size:
            raise ReviewValidationError(f"{label} size changed while it was being read")
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)

    identity = {
        "path": str(resolved),
        "size_bytes": observed_size,
        "sha256": digest.hexdigest(),
        "device": terminal.st_dev,
        "inode": terminal.st_ino,
        "mode": terminal.st_mode,
        "mtime_ns": terminal.st_mtime_ns,
        "ctime_ns": terminal.st_ctime_ns,
    }
    return identity, bytes(captured) if captured is not None else None


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _valid_source_url(value: Any) -> bool:
    if value is None:
        return True
    if not _nonempty_text(value):
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    if "\\" in value or any(
        character.isspace()
        or ord(character) < 0x20
        or ord(character) == 0x7F
        for character in value
    ):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
    except (UnicodeError, ValueError):
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and bool(hostname)
        and username is None
        and password is None
        and (port is None or 1 <= port <= 65535)
    )


def _valid_positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _metadata_errors(record: Any, label: str) -> list[str]:
    errors: list[str] = []
    expected = set(REPRESENTATIVE_CONTENT_FIELDS)
    if not isinstance(record, dict):
        return [f"{label} must be an exact representative-content object"]
    if set(record) != expected:
        missing = sorted(expected - set(record))
        extra = sorted(set(record) - expected)
        errors.append(f"{label} fields differ: missing={missing}, extra={extra}")
    if not _valid_source_url(record.get("source_url")):
        errors.append(
            f"{label}.source_url must be null for user-local input or one safe HTTP(S) URL"
        )
    for field in ("license", "attribution"):
        if not _nonempty_text(record.get(field)):
            errors.append(f"{label}.{field} must be one nonempty trimmed string")
    path_value = record.get("path")
    if not _valid_canonical_absolute_path_text(path_value):
        errors.append(f"{label}.path must be one nonempty canonical absolute path")
    if not _valid_positive_integer(record.get("size_bytes")):
        errors.append(f"{label}.size_bytes must be one exact positive integer")
    if not _valid_sha256(record.get("sha256")):
        errors.append(f"{label}.sha256 must be one lowercase 64-hex digest")
    return errors


def _source_artifact_errors(source: Any) -> list[str]:
    errors: list[str] = []
    expected = set(SOURCE_ARTIFACT_FIELDS)
    if not isinstance(source, dict):
        return ["representative-content source artifact must be one JSON object"]
    if set(source) != expected:
        missing = sorted(expected - set(source))
        extra = sorted(set(source) - expected)
        errors.append(
            "representative-content source artifact fields differ: "
            f"missing={missing}, extra={extra}"
        )
    if source.get("kind") != "representative-content":
        errors.append("representative-content source artifact kind is invalid")
    if source.get("verified") is not True:
        errors.append("representative-content source artifact verified must be true")
    metadata = {
        field: source.get(field) for field in REPRESENTATIVE_CONTENT_FIELDS
    }
    errors.extend(_metadata_errors(metadata, "representative-content source metadata"))
    expected_size = source.get("expected_size_bytes")
    if not _valid_positive_integer(expected_size):
        errors.append("source expected_size_bytes must be one exact positive integer")
    elif type(source.get("size_bytes")) is int and expected_size != source.get("size_bytes"):
        errors.append("source expected_size_bytes differs from size_bytes")
    expected_sha = source.get("expected_sha256")
    if not _valid_sha256(expected_sha):
        errors.append("source expected_sha256 must be one lowercase 64-hex digest")
    elif isinstance(source.get("sha256"), str) and expected_sha != source.get("sha256"):
        errors.append("source expected_sha256 differs from sha256")
    return errors


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def validate_representative_content_summary(
    summary_path: Path,
    source_artifact_path: Path,
) -> tuple[dict[str, Any], int]:
    """Freshly authenticate both JSON inputs and the representative content file."""
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "invalid",
        "valid": False,
        "validation_mode": "strict_fresh_read_only",
        "inputs_mutated": False,
    }
    errors: list[str] = []
    summary: dict[str, Any] | None = None
    source: dict[str, Any] | None = None

    try:
        summary_identity, summary_raw = _read_authenticated_file(
            summary_path, "summary artifact", capture=True
        )
        result["summary_artifact"] = summary_identity
        assert summary_raw is not None
        summary = _strict_json_object(summary_raw, "summary artifact")
    except (OSError, ReviewValidationError) as exc:
        errors.append(str(exc))

    try:
        source_identity, source_raw = _read_authenticated_file(
            source_artifact_path, "representative-content source artifact", capture=True
        )
        result["source_artifact"] = source_identity
        assert source_raw is not None
        source = _strict_json_object(source_raw, "representative-content source artifact")
    except (OSError, ReviewValidationError) as exc:
        errors.append(str(exc))

    summary_metadata: Any = None
    if summary is not None:
        summary_metadata = summary.get("representative_content")
        if "representative_content" not in summary:
            errors.append("summary artifact is missing representative_content")
        errors.extend(_metadata_errors(summary_metadata, "summary representative_content"))

    source_metadata: dict[str, Any] | None = None
    if source is not None:
        errors.extend(_source_artifact_errors(source))
        source_metadata = {
            field: source.get(field) for field in REPRESENTATIVE_CONTENT_FIELDS
        }

    if isinstance(summary_metadata, dict) and source_metadata is not None:
        for field in REPRESENTATIVE_CONTENT_FIELDS:
            if not _exact_json_equal(summary_metadata.get(field), source_metadata.get(field)):
                errors.append(
                    f"summary representative_content.{field} differs from source artifact"
                )

    content_identity: dict[str, Any] | None = None
    if source_metadata is not None:
        content_path = source_metadata.get("path")
        content_size = source_metadata.get("size_bytes")
        content_sha = source_metadata.get("sha256")
        if (
            _valid_canonical_absolute_path_text(content_path)
            and _valid_positive_integer(content_size)
            and _valid_sha256(content_sha)
        ):
            try:
                content_identity, _unused = _read_authenticated_file(
                    Path(content_path),
                    "representative content",
                    capture=False,
                    expected_size=content_size,
                )
                result["content_artifact"] = content_identity
            except (OSError, ReviewValidationError) as exc:
                errors.append(str(exc))
            else:
                if content_identity["size_bytes"] != content_size:
                    errors.append(
                        "representative content current size differs from source artifact"
                    )
                if content_identity["sha256"] != content_sha:
                    errors.append(
                        "representative content current SHA-256 differs from source artifact"
                    )

    if not errors and isinstance(summary_metadata, dict):
        result.update(
            {
                "status": "verified",
                "valid": True,
                "representative_content": summary_metadata,
                "errors": [],
                "exit_code": 0,
            }
        )
        return result, 0

    result.update({"errors": errors, "exit_code": 2})
    return result, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit 0: verified; exit 2: invalid or unauthenticated input; "
            "exit 3: validator or fresh-output failure. Inputs are direct JSON files; "
            "stdin and command-stream substitution are not accepted."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Canonical absolute path to the compact review evidence/summary.json",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        required=True,
        help="Canonical absolute path to the original representative-content.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional fresh JSON result path; an existing path is never overwritten",
    )
    return parser


def _error_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "error",
        "valid": False,
        "validation_mode": "strict_fresh_read_only",
        "inputs_mutated": False,
        "errors": [reason],
        "exit_code": 3,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result, code = validate_representative_content_summary(
            args.summary, args.source_artifact
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result = _error_result(f"unexpected validator failure: {exc}")
        code = 3

    rendered = strict_json_dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        try:
            exclusive_write_bytes(args.output, rendered.encode("utf-8"))
        except (OSError, ValueError) as exc:
            failed = _error_result(f"fresh validation output could not be created: {exc}")
            print(strict_json_dumps(failed, sort_keys=True, separators=(",", ":")))
            return 3
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
