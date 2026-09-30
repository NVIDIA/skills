#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for PAIDF Cosmos Predict scripts."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GENERATION_SETTINGS = SKILL_DIR / "assets" / "default_generation_settings.json"
REQUIRED_INPUT_FIELDS = ("id", "media_path")
WRITE_PROBE_PREFIX = ".paidf_write_probe"


def absolute_path(path: str | Path) -> Path:
    """Return an expanded absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def absolute_media_path(path: str) -> str:
    """Return an absolute media path for deterministic hashing and lookup."""
    return str(absolute_path(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and require each non-empty line to be an object."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as stable JSONL, creating the parent directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def permissions_hint(path: Path) -> str:
    """Return a user-actionable permissions repair command for a path."""
    quoted = shlex.quote(str(path))
    return (
        "Fix ownership or permissions before retrying, for example: "
        f"sudo chown -R \"$(id -u):$(id -g)\" {quoted} && chmod -R u+rwX {quoted}"
    )


def ensure_container_writable_directory(path: Path, label: str) -> None:
    """Create a directory and verify that arbitrary container users can write it."""
    path = path.expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"{label} directory cannot be created: {path}. {permissions_hint(path)}"
        ) from exc

    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")

    try:
        path.chmod(
            path.stat().st_mode
            | stat.S_IRWXU
            | stat.S_IRWXG
            | stat.S_IRWXO
        )
    except OSError as exc:
        raise PermissionError(
            f"{label} directory permissions cannot be updated for container writes: "
            f"{path}. {permissions_hint(path)}"
        ) from exc

    probe = path / f"{WRITE_PROBE_PREFIX}_{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(
            f"{label} directory is not writable: {path}. {permissions_hint(path)}"
        ) from exc


def ensure_writable_file_target(path: Path, label: str) -> None:
    """Verify that a file path can be created or overwritten."""
    ensure_container_writable_directory(path.expanduser().parent, f"{label} parent")
    if path.exists() and not path.is_file():
        raise IsADirectoryError(f"{label} path is not a file: {path}")
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(
            f"{label} file is not writable: {path}. {permissions_hint(path)}"
        )


def ensure_readable_file(path: Path, label: str) -> None:
    """Verify that a required input file exists and is readable."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} file is not readable: {path}")


def ensure_readable_directory(path: Path, label: str) -> None:
    """Verify that a required input directory exists and is searchable."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")
    if not os.access(path, os.R_OK | os.X_OK):
        raise PermissionError(
            f"{label} directory is not readable: {path}. {permissions_hint(path)}"
        )


def models_probe_url(base_url: str) -> str:
    """Return the /models URL used to preflight an OpenAI-compatible base URL."""
    return f"{base_url.rstrip('/')}/models"


def verify_captioning_base_url(base_url: str, timeout_seconds: float = 5.0) -> None:
    """Fail unless the VLM captioning base URL passes the /models probe."""
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "VLM captioning endpoint base URL must include scheme and host: "
            f"{base_url!r}"
        )

    models_url = models_probe_url(base_url)
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds) as response:  # nosec B310 - https endpoint from skill config
            if 200 <= response.status < 500:
                return
            detail = f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if 200 <= exc.code < 500:
            return
        detail = f"HTTP {exc.code}: {exc.reason}"
    except (OSError, urllib.error.URLError) as exc:
        detail = str(exc)

    raise RuntimeError(
        f"VLM captioning base URL did not pass the /models preflight probe at {models_url}: {detail}. "
        "Start or fix the captioning service outside this skill."
    )


def normalize_media_dir(value: str) -> Path:
    """Normalize the required host media directory path."""
    return absolute_path(value)


def path_is_under(path: str, root: Path) -> bool:
    """Return whether a normalized path is equal to or below a root."""
    root_text = str(root).rstrip("/")
    return path == root_text or path.startswith(f"{root_text}/")


def validate_media_dir(media_path: str, media_dir: Path) -> None:
    """Ensure an input media path lives under the required media directory."""
    if path_is_under(media_path, media_dir):
        return
    raise ValueError(f"media_path {media_path!r} is not under --media-dir: {media_dir}")


def input_records(input_jsonl: Path, media_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read input JSONL rows and convert required id/media_path fields."""
    records: list[dict[str, Any]] = []
    for index, record in enumerate(read_jsonl(input_jsonl), start=1):
        for field in REQUIRED_INPUT_FIELDS:
            if field not in record:
                raise ValueError(
                    f"{input_jsonl}: record {index} is missing required field {field!r}"
                )
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(
                    f"{input_jsonl}: record {index} field {field!r} must be a non-empty string"
                )
        media_path = absolute_media_path(record["media_path"])
        if media_dir is not None:
            validate_media_dir(media_path, media_dir)
        records.append({"id": record["id"], "media_path": media_path})
    return records


def unique_media_paths(input_jsonl: Path, media_dir: Path) -> list[dict[str, Any]]:
    """Deduplicate absolute media paths while preserving associated input ids."""
    unique: dict[str, dict[str, Any]] = {}
    for record in input_records(input_jsonl, media_dir):
        item = unique.setdefault(
            record["media_path"],
            {
                "media_path": record["media_path"],
                "row_count": 0,
                "ids": [],
            },
        )
        item["row_count"] += 1
        if record["id"] not in item["ids"]:
            item["ids"].append(record["id"])
    return [unique[key] for key in sorted(unique)]


def deterministic_id(media_path: str) -> str:
    """Return the stable 16-character ID used for generated output paths."""
    return hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:16]


def build_path_map(input_jsonl: Path, output_dir: Path, media_dir: Path) -> list[dict[str, Any]]:
    """Build deterministic PAIDF output paths for each unique media path."""
    output_dir = absolute_path(output_dir)
    mappings: list[dict[str, Any]] = []
    for record in unique_media_paths(input_jsonl, media_dir):
        sample_id = deterministic_id(record["media_path"])
        generated_path = str(output_dir / "generated" / "videos" / f"{sample_id}.mp4")
        caption_path = str(output_dir / "captions" / f"{sample_id}.txt")
        metadata_path = str(output_dir / "generated" / "metadata" / f"{sample_id}.json")
        mappings.append(
            {
                **record,
                "sample_id": sample_id,
                "absolute_media_path": record["media_path"],
                "host_media_path": record["media_path"],
                "container_media_path": record["media_path"],
                "host_generated_video_path": generated_path,
                "container_generated_video_path": generated_path,
                "host_caption_path": caption_path,
                "container_caption_path": caption_path,
                "host_metadata_path": metadata_path,
                "container_metadata_path": metadata_path,
            }
        )
    return mappings


def read_generation_settings(path: Path) -> dict[str, Any]:
    """Read and validate the generation settings JSON object."""
    ensure_readable_file(path, "generation settings")
    with path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return settings


def settings_section(settings: dict[str, Any], section_path: str) -> dict[str, Any]:
    """Return a nested settings object addressed by a dotted path."""
    value: Any = settings
    for part in section_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Generation settings missing {section_path!r}")
        value = value[part]
    if not isinstance(value, dict):
        raise ValueError(f"Generation settings field {section_path!r} must be an object")
    return value


def yaml_scalar(value: Any) -> str:
    """Serialize a primitive value into the small YAML subset PAIDF needs."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def dump_yaml(value: Any, indent: int = 0) -> list[str]:
    """Serialize dict/list/scalar data into deterministic block-style YAML."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines or [f"{prefix}[]"]
    return [f"{prefix}{yaml_scalar(value)}"]
