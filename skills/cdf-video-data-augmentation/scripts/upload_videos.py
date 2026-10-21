#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Validate and upload local Video Data Augmentation inputs to S3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DatasetError(ValueError):
    """Raised when local input does not follow the VDA layout."""


@dataclass(frozen=True)
class VideoEntry:
    local_path: Path
    relative_path: PurePosixPath


def _check_video_signature(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(16)
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov"} and header[4:8] != b"ftyp":
        raise DatasetError(f"File does not have an ISO BMFF video signature: {path}")
    if suffix == ".avi" and not (header.startswith(b"RIFF") and header[8:12] == b"AVI "):
        raise DatasetError(f"File does not have an AVI signature: {path}")
    if suffix in {".mkv", ".webm"} and not header.startswith(b"\x1aE\xdf\xa3"):
        raise DatasetError(f"File does not have an EBML video signature: {path}")


def collect_videos(root: Path, *, verify_content: bool = True) -> list[VideoEntry]:
    """Return validated videos from one file or a flat directory."""
    root = root.expanduser()
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(path for path in root.iterdir() if not path.name.startswith("."))
    else:
        raise DatasetError(f"Input path is not a file or directory: {root}")

    entries: list[VideoEntry] = []
    for path in candidates:
        if path.is_symlink():
            raise DatasetError(f"Symlinks are not allowed: {path}")
        if path.is_dir():
            raise DatasetError(f"Nested directories are not supported: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise DatasetError(f"Unsupported file in VDA input: {path}")
        if path.stat().st_size == 0:
            raise DatasetError(f"Video is empty: {path}")
        if verify_content:
            _check_video_signature(path)
        entries.append(VideoEntry(path, PurePosixPath(path.name)))

    if not entries:
        raise DatasetError(f"Input directory has no supported videos: {root}")
    return entries


def parse_s3_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URL, got: {value}")
    return parsed.netloc, parsed.path.strip("/")


def validate_destination_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("--destination-path must be a non-empty relative path without '..'")
    return path


def destination_url(base: str, destination_path: str) -> str:
    bucket, prefix = parse_s3_url(base)
    destination = validate_destination_path(destination_path)
    key = "/".join(part for part in (prefix, destination.as_posix()) if part)
    return f"s3://{bucket}/{key}/"


def upload_dataset(entries: list[VideoEntry], remote_root: str) -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "boto3 is required for upload; install it or use --validate-only"
        ) from exc

    bucket, prefix = parse_s3_url(remote_root)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 2},
            s3={"addressing_style": "path"},
        ),
    )
    for entry in entries:
        key = "/".join(
            part for part in (prefix.rstrip("/"), entry.relative_path.as_posix()) if part
        )
        LOGGER.info("Uploading %s", entry.relative_path)
        client.upload_file(str(entry.local_path), bucket, key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and upload one video or a flat video directory for VDA."
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--destination-path")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-content-check", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        entries = collect_videos(Path(args.path), verify_content=not args.skip_content_check)
        LOGGER.info("Validated %d video(s).", len(entries))
        if args.validate_only:
            return 0
        if not args.destination_path:
            raise ValueError("--destination-path is required unless --validate-only is used")
        base = os.environ.get("UPLOAD_DESTINATION", "").strip()
        if not base:
            raise ValueError("UPLOAD_DESTINATION is required for upload")
        remote_root = destination_url(base, args.destination_path)
        upload_dataset(entries, remote_root)
        print(f"Input path: {remote_root}")
        return 0
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
