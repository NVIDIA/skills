#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an AppEncCuda AV1 IVF artifact without claiming codec operation success."""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# pylint: disable=wrong-import-position
from _capability_contract import (  # noqa: E402
    exclusive_write_bytes,
    strict_json_dumps,
)

SCHEMA_VERSION = "1.0"
IVF_FILE_HEADER = struct.Struct("<4sHH4sHHIIII")
IVF_FRAME_HEADER = struct.Struct("<IQ")
APPENC_CUDA_FRAME_COUNT_SENTINEL = 0xFFFF


class IvfValidationError(ValueError):
    """The artifact does not satisfy the exact AppEncCuda AV1 IVF contract."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _validated_expected(width: int, height: int, frames: int) -> dict[str, int]:
    for label, value, maximum in (
        ("width", width, 0xFFFF),
        ("height", height, 0xFFFF),
        ("expected frame count", frames, 0xFFFFFFFF),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > maximum
        ):
            raise ValueError(f"{label} must be an integer in 1..{maximum}")
    return {"width": width, "height": height, "frame_count": frames}


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    content = handle.read(size)
    if len(content) != size:
        raise IvfValidationError(
            f"{label} is truncated: expected {size} bytes, found {len(content)}"
        )
    return content


def _discard_exact(handle: BinaryIO, size: int, label: str) -> None:
    remaining = size
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise IvfValidationError(
                f"{label} is truncated: expected {size} bytes, found {size - remaining}"
            )
        remaining -= len(chunk)


def _file_identity_and_handle(path: Path) -> tuple[dict[str, Any], BinaryIO, os.stat_result]:
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise IvfValidationError(f"input must not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise IvfValidationError(f"input must be a regular file: {resolved}")
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    try:
        digest = hashlib.sha256()
        observed = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            observed += len(chunk)
            digest.update(chunk)
        if observed != initial.st_size:
            raise IvfValidationError("input size changed while hashing")
        handle.seek(0)
    except Exception:
        handle.close()
        raise
    identity = {
        "path": str(resolved),
        "device": initial.st_dev,
        "inode": initial.st_ino,
        "size_bytes": initial.st_size,
        "sha256": digest.hexdigest(),
    }
    return identity, handle, initial


def _parse_ivf(
    handle: BinaryIO, *, width: int, height: int, expected_frames: int
) -> dict[str, Any]:
    raw_header = _read_exact(handle, IVF_FILE_HEADER.size, "IVF file header")
    (
        signature,
        version,
        header_size,
        fourcc,
        observed_width,
        observed_height,
        rate,
        scale,
        header_frame_count,
        unused,
    ) = IVF_FILE_HEADER.unpack(raw_header)
    if signature != b"DKIF":
        raise IvfValidationError("IVF signature is not DKIF")
    if version != 0:
        raise IvfValidationError(f"IVF version {version} is not 0")
    if header_size != IVF_FILE_HEADER.size:
        raise IvfValidationError(
            f"IVF header size {header_size} is not {IVF_FILE_HEADER.size}"
        )
    if fourcc != b"AV01":
        raise IvfValidationError("IVF fourcc is not AV01")
    if (observed_width, observed_height) != (width, height):
        raise IvfValidationError(
            "IVF geometry "
            f"{observed_width}x{observed_height} does not match {width}x{height}"
        )
    allowed_header_counts = {expected_frames, APPENC_CUDA_FRAME_COUNT_SENTINEL}
    if header_frame_count not in allowed_header_counts:
        raise IvfValidationError(
            "IVF header frame count is neither the expected count nor the "
            "AppEncCuda 0xFFFF sentinel"
        )

    payload_total = 0
    minimum_frame_size: int | None = None
    maximum_frame_size: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    for index in range(expected_frames):
        raw_frame_header = _read_exact(
            handle, IVF_FRAME_HEADER.size, f"IVF frame {index} header"
        )
        frame_size, timestamp = IVF_FRAME_HEADER.unpack(raw_frame_header)
        if frame_size <= 0:
            raise IvfValidationError(f"IVF frame {index} has an empty payload")
        _discard_exact(handle, frame_size, f"IVF frame {index} payload")
        payload_total += frame_size
        minimum_frame_size = (
            frame_size
            if minimum_frame_size is None
            else min(minimum_frame_size, frame_size)
        )
        maximum_frame_size = (
            frame_size
            if maximum_frame_size is None
            else max(maximum_frame_size, frame_size)
        )
        if first_timestamp is None:
            first_timestamp = timestamp
        last_timestamp = timestamp

    if handle.read(1):
        raise IvfValidationError(
            "IVF contains trailing bytes or more frame records than expected"
        )
    return {
        "signature": signature.decode("ascii"),
        "version": version,
        "header_size_bytes": header_size,
        "fourcc": fourcc.decode("ascii"),
        "width": observed_width,
        "height": observed_height,
        "rate": rate,
        "scale": scale,
        "header_frame_count": header_frame_count,
        "header_frame_count_kind": (
            "appenc_cuda_sentinel"
            if header_frame_count == APPENC_CUDA_FRAME_COUNT_SENTINEL
            else "exact"
        ),
        "unused": unused,
        "record_count": expected_frames,
        "payload_size_bytes": payload_total,
        "minimum_frame_size_bytes": minimum_frame_size,
        "maximum_frame_size_bytes": maximum_frame_size,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def validate_appenc_av1_ivf(
    path: Path, *, width: int, height: int, expected_frames: int
) -> tuple[dict[str, Any], int]:
    """Return compact structural evidence and a fail-closed process status."""
    expected = _validated_expected(width, height, expected_frames)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvcodec-appenc-av1-ivf-structure-validation",
        "status": "invalid",
        "scope": "container_structure_only",
        "operation_verified": False,
        "authenticated_decode_required": True,
        "expected": expected,
    }
    handle: BinaryIO | None = None
    try:
        identity, handle, initial = _file_identity_and_handle(path)
        result["input"] = identity
        parsed = _parse_ivf(
            handle,
            width=width,
            height=height,
            expected_frames=expected_frames,
        )
        terminal = os.fstat(handle.fileno())
        initial_identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        )
        terminal_identity = (
            terminal.st_dev,
            terminal.st_ino,
            terminal.st_size,
            terminal.st_mtime_ns,
            terminal.st_ctime_ns,
        )
        if terminal_identity != initial_identity:
            raise IvfValidationError("input identity changed while validating")
    except (OSError, IvfValidationError) as exc:
        result["reason"] = str(exc)
        return result, 2
    finally:
        if handle is not None:
            handle.close()
    result.update({"status": "structure_verified", "parsed": parsed})
    return result, 0


class _ExitThreeParser(argparse.ArgumentParser):
    """Convert malformed argv into the producer's strict JSON error contract."""

    def error(self, message: str) -> Any:
        raise ValueError(f"{self.prog}: {message}")


def main() -> int:
    parser = _ExitThreeParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Fresh AppEncCuda AV1 output")
    parser.add_argument("--width", type=_positive_int, required=True)
    parser.add_argument("--height", type=_positive_int, required=True)
    parser.add_argument("--expected-frames", type=_positive_int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional fresh JSON evidence path; existing paths are never overwritten",
    )
    try:
        args = parser.parse_args()
    except ValueError as exc:
        print(strict_json_dumps({
            "schema_version": SCHEMA_VERSION,
            "kind": "nvcodec-appenc-av1-ivf-structure-validation-error",
            "status": "error",
            "scope": "container_structure_only",
            "operation_verified": False,
            "authenticated_decode_required": True,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 3
    try:
        result, code = validate_appenc_av1_ivf(
            args.input,
            width=args.width,
            height=args.height,
            expected_frames=args.expected_frames,
        )
    except ValueError as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "nvcodec-appenc-av1-ivf-structure-validation",
            "status": "invalid",
            "scope": "container_structure_only",
            "operation_verified": False,
            "authenticated_decode_required": True,
            "reason": str(exc),
        }
        code = 2
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "nvcodec-appenc-av1-ivf-structure-validation-error",
            "status": "error",
            "scope": "container_structure_only",
            "operation_verified": False,
            "authenticated_decode_required": True,
            "error": str(exc),
        }
        code = 3
    rendered = strict_json_dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        try:
            exclusive_write_bytes(args.output, rendered.encode("utf-8"))
        except (OSError, ValueError) as exc:
            failed = dict(result)
            failed.update(
                {
                    "validation_status": result["status"],
                    "status": "evidence_write_failed",
                    "evidence_output": str(args.output.expanduser().absolute()),
                    "reason": str(exc),
                }
            )
            print(
                strict_json_dumps(failed, sort_keys=True, separators=(",", ":"))
            )
            return 3
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
