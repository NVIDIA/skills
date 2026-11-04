#!/usr/bin/env -S uv run --quiet --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Reject resolved Compose files that are unsafe to deploy."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

SENTINELS = (
    "/path/to/deploy/docker",
    "<HOST_IP>",
)
UNRESOLVED_INTERPOLATION = re.compile(r"(?<!\$)\$\{[^}]+\}")
FILE_TARGET_SUFFIXES = {
    ".conf",
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}
GENERATED_BIND_NAMES = {".wdm-env"}
DEFAULT_CONTAINER_TAG = "develop-latest"
# Repositories whose tag containers.env derives from VSS_CONTAINER_TAG; matched
# by last path segment so a registry override stays covered.
MANAGED_IMAGE_NAMES = frozenset(
    {
        "sdr-mw-l",
        "vss-agent",
        "vss-agent-ui",
        "vss-alert-ms",
        "vss-behavior-analytics",
        "vss-configurator",
        "vss-rt-config-adaptor",
        "vss-rt-cv",
        "vss-rt-cv-mv3dt-bev-fusion",
        "vss-rt-cv-mv3dt-config-init",
        "vss-rt-embed",
        "vss-rt-vlm",
        "vss-video-analytics-api",
        "vss-video-summarization",
        "vss-vios-ingress",
        "vss-vios-nvstreamer",
        "vss-vios-sensor",
        "vss-vios-streamprocessing",
    }
)
NGC_TRIGGER = re.compile(r"nvcr\.io/|(?<![\w.-])ngc:")
NGC_MODEL_REF = re.compile(r"(?<![\w.-])ngc:")
NGC_SECRET_KEYS = ("NGC_API_KEY", "NGC_CLI_API_KEY")


def walk_strings(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from walk_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_strings(child, f"{location}[{index}]")


def bind_mounts(
    document: dict[str, Any],
) -> Iterator[tuple[str, str, str, bool]]:
    for service_name, service in (document.get("services") or {}).items():
        if not isinstance(service, dict):
            continue
        for volume in service.get("volumes") or []:
            if isinstance(volume, dict) and volume.get("type") == "bind":
                source = volume.get("source")
                target = volume.get("target")
                if isinstance(source, str) and isinstance(target, str):
                    yield service_name, source, target, volume.get("read_only") is True


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def iter_env(service: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    env = service.get("environment")
    if isinstance(env, dict):
        yield from env.items()
    elif isinstance(env, list):
        for item in env:
            if isinstance(item, str):
                key, separator, value = item.partition("=")
                yield key, (value if separator else None)


def secret_errors(document: dict[str, Any], extra_required: set[str]) -> list[str]:
    errors: list[str] = []
    services = document.get("services") or {}

    ngc_required = any(NGC_TRIGGER.search(value) for _, value in walk_strings(document))

    monitored = set(extra_required)
    if ngc_required:
        monitored.update(NGC_SECRET_KEYS)
    if not monitored:
        return errors

    seen = {key: False for key in monitored}
    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue
        for key, value in iter_env(service):
            if key not in monitored:
                continue
            seen[key] = True
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(
                    f"service {service_name!r} environment {key!r} resolved empty; a "
                    "mode-required credential baked as '' cannot be supplied at deploy "
                    "time — set it and regenerate resolved.yml"
                )

    ngc_model_ref = any(
        NGC_MODEL_REF.search(value) for _, value in walk_strings(document)
    )
    if ngc_model_ref and not (seen["NGC_API_KEY"] or seen["NGC_CLI_API_KEY"]):
        errors.append(
            "resolved model references an ngc: model path but no "
            "NGC_API_KEY/NGC_CLI_API_KEY is wired into any service environment"
        )
    for key in extra_required:
        if not seen[key]:
            errors.append(
                f"required credential {key!r} is absent from every service environment"
            )

    return errors


def image_tag(image: str) -> str | None:
    repository, separator, tag = image.rpartition(":")
    if not separator or not repository or "/" in tag:
        return None
    return tag


def container_tag_errors(
    document: dict[str, Any], expected_tag: str | None
) -> list[str]:
    """Report managed images left on the `containers.env` fallback tag.

    Only the fallback is an error: an explicit `VSS_*_TAG` pin and an SBSA
    suffix are both meant to supersede the common tag. Images outside
    `MANAGED_IMAGE_NAMES` are not governed by `VSS_CONTAINER_TAG`, so their
    tags are never checked.
    """
    if not expected_tag or expected_tag == DEFAULT_CONTAINER_TAG:
        return []

    errors: list[str] = []
    for service_name, service in sorted((document.get("services") or {}).items()):
        if not isinstance(service, dict):
            continue
        image = service.get("image")
        if not isinstance(image, str):
            continue
        tag = image_tag(image)
        if tag is None:
            continue
        name = image.rpartition(":")[0].rsplit("/", 1)[-1]
        if name not in MANAGED_IMAGE_NAMES:
            continue
        if tag == DEFAULT_CONTAINER_TAG or tag.startswith(f"{DEFAULT_CONTAINER_TAG}-"):
            errors.append(
                f"service {service_name!r} image {image} kept the "
                f"{DEFAULT_CONTAINER_TAG!r} default instead of the requested tag "
                f"{expected_tag!r}; export VSS_CONTAINER_TAG before "
                "docker compose config and regenerate"
            )

    return errors


def alerts_ui_errors(document: dict[str, Any]) -> list[str]:
    """Reject CV verification controls on an always-on RT-VLM deployment."""
    services = document.get("services") or {}
    bridge = services.get("alert-bridge")
    ui = services.get("vss-ui")
    if not isinstance(bridge, dict) or not isinstance(ui, dict):
        return []
    bridge_env = dict(iter_env(bridge))
    if str(bridge_env.get("ALERT_AGENT_ALWAYS_ON", "")).lower() != "true":
        return []
    key = "NEXT_PUBLIC_ALERTS_TAB_MANAGE_ALERTS_SUB_TAB_ENABLE_CV_ALERTS_VERIFICATION"
    value = dict(iter_env(ui)).get(key)
    if value is False or value == "false":
        return []
    return [
        (
            f"service 'vss-ui' must set {key}=false for always-on RT-VLM alerts; "
            "set it in override.env and regenerate resolved.yml"
        )
    ]


def validate_document(
    document: dict[str, Any],
    repo_root: Path,
    extra_required: set[str] | None = None,
    expected_container_tag: str | None = None,
) -> list[str]:
    errors: list[str] = []
    services = document.get("services")

    if not isinstance(services, dict) or not services:
        errors.append("resolved Compose model has no services")

    for location, value in walk_strings(document):
        for sentinel in SENTINELS:
            if sentinel in value:
                errors.append(f"{location} contains placeholder {sentinel!r}")
        if UNRESOLVED_INTERPOLATION.search(value):
            errors.append(f"{location} contains unresolved Compose interpolation")

    for service_name, source_text, target_text, read_only in bind_mounts(document):
        source = Path(source_text)
        if not source.is_absolute() or not is_within(source, repo_root):
            continue
        checked_in_source = read_only or bool(
            {"developer-profiles", "industry-profiles"} & set(source.parts)
        )
        if not checked_in_source:
            continue
        if not source.exists():
            template_source = source.with_name(f"{source.name}.tmpl")
            if source.name in GENERATED_BIND_NAMES or template_source.is_file():
                continue
            errors.append(
                f"service {service_name!r} bind source does not exist: {source}"
            )
            continue
        if source.is_dir() and Path(target_text).suffix in FILE_TARGET_SUFFIXES:
            errors.append(
                f"service {service_name!r} mounts directory {source} "
                f"onto file target {target_text}"
            )

    errors.extend(secret_errors(document, extra_required or set()))
    errors.extend(container_tag_errors(document, expected_container_tag))
    errors.extend(alerts_ui_errors(document))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("resolved_yml", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--required-secret",
        action="append",
        default=[],
        metavar="KEY",
        help="env key that must resolve to a non-empty literal (repeatable)",
    )
    parser.add_argument(
        "--expect-container-tag",
        metavar="TAG",
        help=(
            "common VSS_CONTAINER_TAG the build selected; fail when any managed image "
            f"stayed on the {DEFAULT_CONTAINER_TAG!r} default"
        ),
    )
    args = parser.parse_args()
    if args.expect_container_tag is not None:
        args.expect_container_tag = args.expect_container_tag.strip()
        if not args.expect_container_tag:
            parser.error("--expect-container-tag requires a non-empty tag")
    return args


def main() -> None:
    args = parse_args()
    try:
        with args.resolved_yml.open() as resolved_file:
            document = yaml.safe_load(resolved_file) or {}
    except FileNotFoundError:
        print(f"ERROR: {args.resolved_yml} not found", file=sys.stderr)
        raise SystemExit(1)
    except yaml.YAMLError as error:
        print(
            f"ERROR: failed to parse {args.resolved_yml}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    errors = validate_document(
        document,
        args.repo_root,
        set(args.required_secret),
        args.expect_container_tag,
    )
    if errors:
        print(
            f"ERROR: {args.resolved_yml} failed pre-deployment validation:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)

    print(
        f"Validated {args.resolved_yml}: no stale placeholders, invalid "
        "checked-in bind sources, or empty mode-required credentials"
    )
    if args.expect_container_tag:
        print(
            f"Container image tag {args.expect_container_tag}: no image stayed "
            f"on the {DEFAULT_CONTAINER_TAG} default"
        )


if __name__ == "__main__":
    main()
