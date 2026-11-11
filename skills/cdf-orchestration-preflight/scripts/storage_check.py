#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check that every storage URL in a workflow payload resolves through the deployed MSC config.

DAG tasks resolve storage URLs with the longest matching ``path_mapping`` prefix from the
``multistorageclient-configuration-secret`` Secret. A payload bucket without a mapping fails at
runtime, often after GPU services have started; this check catches it before a run is triggered.
It is payload-agnostic: every string in the payload that is a storage URL is checked, whatever
its field name. Credential values from the Secret are never printed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SECRET_NAME = "multistorageclient-configuration-secret"
SECRET_KEY = "multistorageclient_configuration_secret"
DEFAULT_NAMESPACE = "sdg-workflow"
MSC_SCHEME = "msc://"
STORAGE_SCHEMES = frozenset({"s3", "gs", "msc", "az", "ais"})
# DAG tasks build this POSIX profile for the mounted I/O-cache PVC; it is not in the Secret.
BUILTIN_PROFILES = frozenset({"ngc-io-cache"})
S3_HOST = re.compile(r"(^|\.)s3([.-][a-z0-9-]+)*\.amazonaws\.com$")


class StorageCheckError(RuntimeError):
    """A safe-to-display storage check error."""


def parse_config(text: str) -> dict[str, Any]:
    # DAG tasks read this value with ast.literal_eval, so accept both JSON and Python literals.
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            config = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise StorageCheckError("storage configuration is not valid JSON") from None
    if not isinstance(config, dict):
        raise StorageCheckError("storage configuration must be a JSON object")
    return config


def read_deployed_config(namespace: str) -> dict[str, Any]:
    command = [
        "kubectl",
        "get",
        "secret",
        SECRET_NAME,
        "-n",
        namespace,
        "-o",
        f"jsonpath={{.data.{SECRET_KEY}}}",
    ]
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except FileNotFoundError:
        raise StorageCheckError("kubectl is required to read the deployed configuration") from None
    except subprocess.TimeoutExpired:
        raise StorageCheckError(f"kubectl timed out reading Secret {SECRET_NAME}") from None
    if result.returncode != 0:
        raise StorageCheckError(
            f"could not read Secret {SECRET_NAME} in namespace {namespace}: "
            f"{result.stderr.strip()[:500]}"
        )
    encoded = result.stdout.strip()
    if not encoded:
        raise StorageCheckError(f"Secret {SECRET_NAME} has no {SECRET_KEY} key")
    try:
        text = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise StorageCheckError(f"Secret {SECRET_NAME} does not hold UTF-8 data") from None
    return parse_config(text)


def mapping_prefixes(config: dict[str, Any]) -> dict[str, str]:
    mapping = config.get("path_mapping") or {}
    if not isinstance(mapping, dict):
        raise StorageCheckError("storage configuration path_mapping must be an object")
    return {
        source: target
        for source, target in mapping.items()
        if isinstance(source, str) and source.endswith("/") and isinstance(target, str)
    }


def _walk(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value.strip()


def is_storage_url(value: str, prefixes: dict[str, str]) -> bool:
    parsed = urlparse(value)
    if parsed.scheme in STORAGE_SCHEMES:
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    # Service endpoints are plain HTTP(S); only S3 hosts and explicitly mapped prefixes are storage.
    return bool(S3_HOST.search(parsed.hostname or "")) or any(
        value.startswith(prefix) for prefix in prefixes
    )


def storage_urls(payload: Any, config: dict[str, Any]) -> dict[str, str]:
    """Return every storage URL in ``payload``, keyed by its field path."""
    prefixes = mapping_prefixes(config)
    return {path: value for path, value in _walk(payload, "") if is_storage_url(value, prefixes)}


def _profile_of(msc_url: str) -> str:
    return msc_url[len(MSC_SCHEME) :].lstrip("/").split("/", 1)[0]


def resolve_profile(url: str, config: dict[str, Any]) -> str | None:
    """Return the MSC profile a DAG task would use for ``url``, or None if nothing maps it."""
    if url.startswith(MSC_SCHEME):
        return _profile_of(url)
    # Directory inputs may omit the trailing slash; DAG tasks normalize them before listing.
    candidates = (url, url.rstrip("/") + "/")
    prefixes = mapping_prefixes(config)
    best = max(
        (prefix for prefix in prefixes if any(c.startswith(prefix) for c in candidates)),
        key=len,
        default=None,
    )
    if best is None or not prefixes[best].startswith(MSC_SCHEME):
        return None
    return _profile_of(prefixes[best])


def check_paths(paths: dict[str, str], config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (resolved lines, problem lines) for each payload storage URL."""
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise StorageCheckError("storage configuration defines no profiles")
    resolved: list[str] = []
    problems: list[str] = []
    for field, url in paths.items():
        profile = resolve_profile(url, config)
        if profile is None:
            problems.append(f"{field}: {url} matches no path_mapping prefix")
        elif profile not in profiles and profile not in BUILTIN_PROFILES:
            problems.append(f"{field}: {url} maps to profile '{profile}', which is not defined")
        else:
            resolved.append(f"{field}: {url} -> profile '{profile}'")
    return resolved, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that every storage URL in a CDF Orchestration workflow payload is mapped by "
            "the deployed multistorageclient configuration."
        )
    )
    parser.add_argument("--payload", required=True, help="Payload JSON file to check.")
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Controller namespace holding {SECRET_NAME} (default: {DEFAULT_NAMESPACE}).",
    )
    parser.add_argument(
        "--config",
        help="Read the MSC configuration from this local JSON file instead of the cluster.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
        if args.config:
            config = parse_config(Path(args.config).read_text(encoding="utf-8"))
        else:
            config = read_deployed_config(args.namespace)
        paths = storage_urls(payload, config)
        if not paths:
            print("Storage mapping: ready (payload contains no storage URLs)")
            return 0
        resolved, problems = check_paths(paths, config)
    except (OSError, json.JSONDecodeError, StorageCheckError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for line in resolved:
        print(f"OK: {line}")
    if not problems:
        print("Storage mapping: ready")
        return 0
    for line in problems:
        print(f"UNMAPPED: {line}", file=sys.stderr)
    print("Configured path_mapping prefixes:", file=sys.stderr)
    for prefix in sorted(mapping_prefixes(config)) or ["(none)"]:
        print(f"  {prefix}", file=sys.stderr)
    print("Storage mapping: NOT READY", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
