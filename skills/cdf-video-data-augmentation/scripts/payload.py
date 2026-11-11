#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Render and validate standalone Video Data Augmentation DAG payloads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODELS = {
    "language": "Qwen/Qwen3.8-27B-FP8",
    "transfer": "nvidia/Cosmos3-Nano",
}
DEFAULT_VARIABLE_DISTRIBUTION = {
    "variables": {
        "weather": {"cloudy": 0.5, "sunny": 0.5},
        "time_of_day": {"morning": 0.5, "afternoon": 0.5},
    }
}
SERVICE_KEYS = ("vlm_service", "llm_service", "cosmos_transfer_service")
DEFAULT_CHECKERS = (
    "metropolis.hallucination",
    "metropolis.attribute_verification",
    "metropolis.physical_plausibility",
)
SERVICE_SECTIONS = ("cosmos", "captioning", "visual_qa", "reasoning")
OUTPUT_SECTIONS = SERVICE_SECTIONS + (
    "training_export",
    "super_resolution",
    "detection_and_tracking",
)


class PayloadError(ValueError):
    """Raised when a payload violates the bundled operational contract."""


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PayloadError(f"{path} must be a non-empty string")
    return value.strip()


def _reject_placeholder(value: str, path: str) -> str:
    if "<" in value or ">" in value:
        raise PayloadError(f"{path} contains a placeholder: {value}")
    return value


def _storage_path(value: Any, path: str, *, writable: bool = False) -> str:
    value = _reject_placeholder(_nonempty_string(value, path), path)
    parsed = urlparse(value)
    allowed = {"s3"} if writable else {"s3", "http", "https"}
    if parsed.scheme not in allowed or not parsed.netloc:
        schemes = "s3://" if writable else "s3://, http://, or https://"
        raise PayloadError(f"{path} must be a {schemes} URL")
    return value.rstrip("/") if writable else value


def _service_url(value: Any, path: str, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise PayloadError(f"{path} is required in external service mode")
        return None
    value = _reject_placeholder(_nonempty_string(value, path), path)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PayloadError(f"{path} must be an http:// or https:// URL")
    return value.rstrip("/")


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise PayloadError(f"{path} must be >= {minimum}")
    return value


def _weighted_distribution(value: Any, path: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise PayloadError(f"{path} must be a non-empty object")
    result: dict[str, float] = {}
    for choice, weight in value.items():
        choice = _nonempty_string(choice, f"{path} choice")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise PayloadError(f"{path}.{choice} weight must be numeric")
        number = float(weight)
        if not math.isfinite(number) or number < 0:
            raise PayloadError(f"{path}.{choice} weight must be finite and non-negative")
        result[choice] = number
    if sum(result.values()) <= 0:
        raise PayloadError(f"{path} weights must have a positive total")
    return result


def validate_variable_distribution(value: Any) -> dict[str, Any]:
    """Validate the direct weighted-variable shape used by the bundled VDA config."""
    if value is None:
        return deepcopy(DEFAULT_VARIABLE_DISTRIBUTION)
    if not isinstance(value, dict):
        raise PayloadError("cosmos.variable_distribution must be an object")
    variables = value.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise PayloadError("cosmos.variable_distribution.variables must be a non-empty object")
    normalized = {
        _nonempty_string(name, "variable name"): _weighted_distribution(
            distribution,
            f"cosmos.variable_distribution.variables.{name}",
        )
        for name, distribution in variables.items()
    }
    result = deepcopy(value)
    result["variables"] = normalized
    return result


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PayloadError(f"{name} must be an object")
    return deepcopy(value)


def _align_nested(
    section: dict[str, Any],
    name: str,
    *,
    output_directory: str,
    external_services: bool,
) -> None:
    if section.get("output_directory", output_directory) != output_directory:
        raise PayloadError(f"{name}.output_directory must match top-level output_directory")
    section["output_directory"] = output_directory
    if name in SERVICE_SECTIONS:
        nested_mode = section.get("external_services", external_services)
        if nested_mode is not external_services:
            raise PayloadError(f"{name}.external_services must match top-level external_services")
        section["external_services"] = external_services


def normalize_payload(raw: Any, *, platform: str = "k8s") -> dict[str, Any]:
    """Validate and normalize the operational fields consumed by the VDA DAG."""
    if not isinstance(raw, dict):
        raise PayloadError("payload must be a JSON object")
    if platform not in {"k8s", "osmo"}:
        raise PayloadError("platform must be 'k8s' or 'osmo'")

    payload = deepcopy(raw)
    external = payload.get("external_services", False)
    if not isinstance(external, bool):
        raise PayloadError("external_services must be a boolean")
    if platform == "osmo" and not external:
        raise PayloadError("OSMO requires external_services=true")

    input_path = _storage_path(payload.get("input_path"), "input_path")
    output = _storage_path(payload.get("output_directory"), "output_directory", writable=True)
    max_videos = payload.get("max_videos", 1)
    if max_videos is not None:
        max_videos = _integer(max_videos, "max_videos")
        if max_videos <= 0:
            max_videos = None

    lifecycle_raw = payload.get("service_lifecycle") or {}
    if not isinstance(lifecycle_raw, dict):
        raise PayloadError("service_lifecycle must be an object")
    lifecycle: dict[str, Any] = {}
    expected_enabled = not external
    for service in SERVICE_KEYS:
        entry = lifecycle_raw.get(service) or {}
        if not isinstance(entry, dict):
            raise PayloadError(f"service_lifecycle.{service} must be an object")
        enabled = entry.get("enabled", expected_enabled)
        if enabled is not expected_enabled:
            raise PayloadError(
                f"service_lifecycle.{service}.enabled must be {expected_enabled} "
                f"when external_services is {external}"
            )
        lifecycle[service] = {
            **entry,
            "enabled": expected_enabled,
            "replicas": _integer(
                entry.get("replicas", 1),
                f"service_lifecycle.{service}.replicas",
                minimum=1,
            ),
        }

    sections = {name: _section(payload, name) for name in OUTPUT_SECTIONS}
    for name, section in sections.items():
        _align_nested(
            section,
            name,
            output_directory=output,
            external_services=external,
        )

    cosmos = sections["cosmos"]
    cosmos["num_augmentation"] = _integer(
        cosmos.get("num_augmentation", 1), "cosmos.num_augmentation", minimum=1
    )
    cosmos["variable_distribution"] = validate_variable_distribution(
        cosmos.get("variable_distribution")
    )
    for field in ("vlm_model", "llm_model", "cosmos_transfer_model"):
        if cosmos.get(field) is not None:
            cosmos[field] = _nonempty_string(cosmos[field], f"cosmos.{field}")
    cosmos["vlm_service_url"] = _service_url(
        cosmos.get("vlm_service_url"), "cosmos.vlm_service_url", required=external
    )
    cosmos["llm_service_url"] = _service_url(
        cosmos.get("llm_service_url"), "cosmos.llm_service_url", required=external
    )
    cosmos["cosmos_transfer_service_url"] = _service_url(
        cosmos.get("cosmos_transfer_service_url"),
        "cosmos.cosmos_transfer_service_url",
        required=external,
    )
    if cosmos.get("base_config_path") is not None:
        cosmos["base_config_path"] = _storage_path(
            cosmos["base_config_path"], "cosmos.base_config_path"
        )

    overrides = cosmos.get("checker_overrides") or {}
    if not isinstance(overrides, dict):
        raise PayloadError("cosmos.checker_overrides must be an object")
    custom_base = cosmos.get("base_config_path") is not None
    for checker, enabled in overrides.items():
        _nonempty_string(checker, "cosmos.checker_overrides key")
        if not isinstance(enabled, bool):
            raise PayloadError(f"cosmos.checker_overrides.{checker} must be a boolean")
        if not custom_base and checker not in DEFAULT_CHECKERS:
            raise PayloadError(f"unsupported bundled evaluator check: {checker}")
    cosmos["checker_overrides"] = overrides
    evaluator_url = _service_url(
        cosmos.get("cosmos_evaluator_service_url"),
        "cosmos.cosmos_evaluator_service_url",
        required=False,
    )
    cosmos["cosmos_evaluator_service_url"] = evaluator_url
    any_default_check_enabled = any(overrides.get(name, True) for name in DEFAULT_CHECKERS)
    if platform == "osmo" and any_default_check_enabled and not evaluator_url:
        raise PayloadError(
            "OSMO requires cosmos.cosmos_evaluator_service_url while any bundled "
            "evaluator check is enabled"
        )

    captioning = sections["captioning"]
    summary_enabled = captioning.get("enable_llm_summary", True)
    if not isinstance(summary_enabled, bool):
        raise PayloadError("captioning.enable_llm_summary must be a boolean")
    captioning["enable_llm_summary"] = summary_enabled
    captioning["vlm_service_url"] = _service_url(
        captioning.get("vlm_service_url"),
        "captioning.vlm_service_url",
        required=external,
    )
    captioning["llm_service_url"] = _service_url(
        captioning.get("llm_service_url"),
        "captioning.llm_service_url",
        required=external and summary_enabled,
    )

    visual_qa = sections["visual_qa"]
    visual_qa["vlm_service_url"] = _service_url(
        visual_qa.get("vlm_service_url"),
        "visual_qa.vlm_service_url",
        required=external,
    )
    visual_qa["llm_service_url"] = _service_url(
        visual_qa.get("llm_service_url"),
        "visual_qa.llm_service_url",
        required=external,
    )

    reasoning = sections["reasoning"]
    reasoning["llm_service_url"] = _service_url(
        reasoning.get("llm_service_url"),
        "reasoning.llm_service_url",
        required=external,
    )

    payload.update(sections)
    payload.update(
        input_path=input_path,
        max_videos=max_videos,
        output_directory=output,
        external_services=external,
        service_lifecycle=lifecycle,
    )
    return payload


def render_payload(args: argparse.Namespace) -> dict[str, Any]:
    external = args.service_mode == "external"
    distribution = deepcopy(DEFAULT_VARIABLE_DISTRIBUTION)
    if args.variable_distribution:
        distribution = json.loads(Path(args.variable_distribution).read_text(encoding="utf-8"))
    disabled = set(args.disable_check or [])
    language_model = args.language_model
    raw = {
        "input_path": args.input_path,
        "max_videos": args.max_videos,
        "output_directory": args.output_directory,
        "external_services": external,
        "cosmos": {
            "vlm_service_url": args.vlm_url,
            "llm_service_url": args.llm_url,
            "cosmos_transfer_service_url": args.cosmos_transfer_url,
            "cosmos_evaluator_service_url": args.cosmos_evaluator_url,
            "vlm_model": language_model,
            "llm_model": language_model,
            "cosmos_transfer_model": DEFAULT_MODELS["transfer"],
            "num_augmentation": args.num_augmentation,
            "variable_distribution": distribution,
            "checker_overrides": {name: name not in disabled for name in DEFAULT_CHECKERS},
        },
        "captioning": {
            "vlm_service_url": args.vlm_url,
            "llm_service_url": args.llm_url,
            "vlm_model": language_model,
            "llm_model": language_model,
            "enable_llm_summary": not args.disable_caption_summary,
        },
        "visual_qa": {
            "vlm_service_url": args.vlm_url,
            "llm_service_url": args.llm_url,
            "vlm_model": language_model,
            "llm_model": language_model,
            "generation_mode": "window-vlm-llm",
            "input_source": "auto",
            "include_reasoning": True,
            "media_mode": "auto",
        },
        "reasoning": {
            "llm_service_url": args.llm_url,
            "llm_model": language_model,
            "reasoning_mode": "keep",
        },
        "training_export": {"formats": ["tao-vl-reason-v1.0"]},
        "super_resolution": {},
        "detection_and_tracking": {},
    }
    return normalize_payload(raw, platform=args.platform)


def _max_videos(value: str) -> int | None:
    if value.lower() in {"all", "none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer or 'all'") from exc
    return None if parsed <= 0 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or validate a Video Data Augmentation DAG payload."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Render a validated payload.")
    render.add_argument("--input-path", required=True)
    render.add_argument("--output-directory", required=True)
    render.add_argument("--platform", choices=("k8s", "osmo"), default="k8s")
    render.add_argument("--service-mode", choices=("external", "internal"), required=True)
    render.add_argument("--max-videos", type=_max_videos, default=1)
    render.add_argument("--num-augmentation", type=int, default=1)
    render.add_argument("--variable-distribution")
    render.add_argument("--vlm-url")
    render.add_argument("--llm-url")
    render.add_argument("--cosmos-transfer-url")
    render.add_argument("--cosmos-evaluator-url")
    render.add_argument("--language-model", default=DEFAULT_MODELS["language"])
    render.add_argument("--disable-check", action="append", choices=DEFAULT_CHECKERS)
    render.add_argument("--disable-caption-summary", action="store_true")
    render.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate", help="Validate a payload.")
    validate.add_argument("--payload", required=True)
    validate.add_argument("--platform", choices=("k8s", "osmo"), default="k8s")
    validate.add_argument("--output")
    return parser


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        print(f"Validated payload: {output}")
    else:
        print(rendered, end="")


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "render":
            payload = render_payload(args)
            _write_json(payload, args.output)
        else:
            raw = json.loads(Path(args.payload).read_text(encoding="utf-8"))
            _write_json(normalize_payload(raw, platform=args.platform), args.output)
        return 0
    except (OSError, json.JSONDecodeError, PayloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
