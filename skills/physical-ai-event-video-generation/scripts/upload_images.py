#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Validate and upload a flat directory of Event Video Generation seed images to S3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
SIGNATURE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DatasetError(ValueError):
    """Raised when a local dataset does not follow the expected layout."""


@dataclass(frozen=True)
class ImageEntry:
    local_path: Path
    relative_path: PurePosixPath


def _check_image_signature(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(16)
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    if path.suffix.lower() in {".jpg", ".jpeg"} and not is_jpeg:
        raise DatasetError(
            f"File has a JPEG extension but not a JPEG signature: {path}"
        )
    if path.suffix.lower() == ".png" and not is_png:
        raise DatasetError(f"File has a PNG extension but not a PNG signature: {path}")


def collect_images(root: Path, *, verify_content: bool = True) -> list[ImageEntry]:
    """Return validated images directly under ``root``.

    Unlike the person-crop workflows, Event Video Generation input has no subdirectory
    convention: every accepted file directly under the dataset root is one independent seed
    image, or ``root`` may itself be a single image file.
    """
    root = root.expanduser()

    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise DatasetError(f"Unsupported file: {root}")
        if root.stat().st_size == 0:
            raise DatasetError(f"Image is empty: {root}")
        if verify_content and root.suffix.lower() in SIGNATURE_EXTENSIONS:
            _check_image_signature(root)
        return [ImageEntry(root, PurePosixPath(root.name))]

    if not root.is_dir():
        raise DatasetError(f"Dataset path is not a file or directory: {root}")

    entries: list[ImageEntry] = []
    for path in sorted(p for p in root.iterdir() if not p.name.startswith(".")):
        if path.is_symlink():
            raise DatasetError(f"Symlinks are not allowed in the dataset: {path}")
        if path.is_dir():
            raise DatasetError(
                f"Event Video Generation input has no subdirectory convention; "
                f"found a directory at the dataset root: {path}"
            )
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise DatasetError(f"Unsupported file in dataset: {path}")
        if path.stat().st_size == 0:
            raise DatasetError(f"Image is empty: {path}")
        if verify_content and path.suffix.lower() in SIGNATURE_EXTENSIONS:
            _check_image_signature(path)
        entries.append(ImageEntry(path, PurePosixPath(path.name)))

    if not entries:
        raise DatasetError(f"Dataset directory has no supported images: {root}")
    return entries


def parse_s3_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URL, got: {value}")
    return parsed.netloc, parsed.path.strip("/")


def validate_destination_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            "--destination-path must be a non-empty relative path without '..'"
        )
    return path


def destination_url(base: str, destination_path: str) -> str:
    bucket, prefix = parse_s3_url(base)
    destination = validate_destination_path(destination_path)
    parts = [part for part in (prefix, destination.as_posix()) if part]
    key = "/".join(parts)
    return f"s3://{bucket}/{key}/"


def upload_dataset(entries: list[ImageEntry], remote_root: str) -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - depends on sandbox packaging
        raise RuntimeError(
            "boto3 is required for upload; install it or use --validate-only"
        ) from exc

    bucket, prefix = parse_s3_url(remote_root)
    # Leave unset when absent so boto3 resolves the region from its own configuration chain.
    region = (
        os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None
    )
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

    def strip_expect_header(request, **_kwargs):
        # Some sandboxed network proxies mishandle boto3's 100-continue flow.
        for header in ("Expect", "expect"):
            if header in request.headers:
                del request.headers[header]

    client.meta.events.register("request-created.s3.PutObject", strip_expect_header)
    client.meta.events.register("request-created.s3.UploadPart", strip_expect_header)

    for entry in entries:
        key = "/".join(
            part
            for part in (prefix.rstrip("/"), entry.relative_path.as_posix())
            if part
        )
        LOGGER.info("Uploading %s", entry.relative_path)
        client.upload_file(str(entry.local_path), bucket, key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and upload one seed image or a flat directory of seed images "
            "for Event Video Generation."
        )
    )
    parser.add_argument(
        "--path", required=True, help="Local seed image file or flat directory root."
    )
    parser.add_argument(
        "--destination-path",
        help="Relative path appended to UPLOAD_DESTINATION. Required unless --validate-only.",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate without uploading."
    )
    parser.add_argument(
        "--skip-content-check",
        action="store_true",
        help="Check layout/extensions but skip JPEG/PNG signature validation.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    args = build_parser().parse_args()
    try:
        entries = collect_images(
            Path(args.path), verify_content=not args.skip_content_check
        )
        LOGGER.info("Validated %d image(s).", len(entries))
        if args.validate_only:
            return 0
        if not args.destination_path:
            raise ValueError(
                "--destination-path is required unless --validate-only is used"
            )
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
