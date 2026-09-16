#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Render and validate standalone payloads for the SDG ``image_attribute_augmentation_dag``."""

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
    "vlm": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "llm": "Qwen/Qwen2.5-14B-Instruct",
    "image_edit": "Qwen/Qwen-Image-Edit-2511",
}
DEFAULT_VARIABLE_DISTRIBUTION = {
    "variables": {
        "top_outer_color": {"black": 1.0},
        "top_outer_type": {"hoodie": 1.0},
        "bottom_type": {"jeans": 1.0},
        "bottom_color": {"blue": 1.0},
        "shoe_type": {"sneakers": 1.0},
        "shoe_color": {"white": 1.0},
    }
}
SERVICE_KEYS = ("vlm_service", "llm_service", "image_edit_service")
ATTRIBUTE_SEARCH_KEY = "event_and_person_attribute_search"
DEFAULT_ATTRIBUTE_SEARCH_MODE = "image_pas"


class PayloadError(ValueError):
    """Raised when a payload violates the bundled contract."""


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PayloadError(f"{path} must be a non-empty string")
    return value


def _optional_model(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path)


def _service_url(value: Any, path: str, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise PayloadError(f"{path} is required in external service mode")
        return None
    value = _nonempty_string(value, path)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PayloadError(f"{path} must be an http:// or https:// URL")
    return value


def _storage_path(value: Any, path: str) -> str:
    value = _nonempty_string(value, path)
    parsed = urlparse(value)
    if parsed.scheme not in {"s3", "http", "https"} or not parsed.netloc:
        raise PayloadError(f"{path} must be an s3://, http://, or https:// storage URL")
    return value


def _optional_storage_path(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _storage_path(value, path)


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
        _nonempty_string(choice, f"{path} choice")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise PayloadError(f"{path}.{choice} weight must be numeric")
        number = float(weight)
        if not math.isfinite(number) or number < 0:
            raise PayloadError(
                f"{path}.{choice} weight must be finite and non-negative"
            )
        result[choice] = number
    if sum(result.values()) <= 0:
        raise PayloadError(f"{path} weights must have a positive total")
    return result


def _variable_config(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise PayloadError(f"{path} must be a non-empty object")
    if all(
        isinstance(weight, (int, float)) and not isinstance(weight, bool)
        for weight in value.values()
    ):
        return _weighted_distribution(value, path)
    result: dict[str, Any] = {}
    for lookup_value, distribution in value.items():
        _nonempty_string(lookup_value, f"{path} lookup value")
        result[lookup_value] = _weighted_distribution(
            distribution, f"{path}.{lookup_value}"
        )
    return result


def validate_variable_distribution(value: Any) -> dict[str, Any]:
    if value is None or value == {}:
        return deepcopy(DEFAULT_VARIABLE_DISTRIBUTION)
    if not isinstance(value, dict):
        raise PayloadError("cosmos.variable_distribution must be an object")
    variables = value.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise PayloadError(
            "cosmos.variable_distribution.variables must be a non-empty object"
        )

    normalized_variables = {
        name: _variable_config(config, f"cosmos.variable_distribution.variables.{name}")
        for name, config in variables.items()
    }
    possible_values: dict[str, set[str]] = {}
    lookup_distributions = [
        config
        for config in normalized_variables.values()
        if config and all(isinstance(item, dict) for item in config.values())
    ]
    for name, config in normalized_variables.items():
        if config and all(isinstance(item, dict) for item in config.values()):
            continue
        resolved: set[str] = set()
        for base_value in config:
            choices = {base_value}
            for lookup in lookup_distributions:
                if base_value in lookup:
                    choices = set(lookup[base_value])
                    break
            resolved.update(choices)
        possible_values[name] = resolved

    conditional = value.get("conditional_variables") or {}
    if not isinstance(conditional, dict):
        raise PayloadError(
            "cosmos.variable_distribution.conditional_variables must be an object"
        )
    normalized_conditional: dict[str, Any] = {}
    for name, config in conditional.items():
        path = f"cosmos.variable_distribution.conditional_variables.{name}"
        if not isinstance(config, dict):
            raise PayloadError(f"{path} must be an object")
        parent = _nonempty_string(config.get("depends_on"), f"{path}.depends_on")
        if parent not in possible_values:
            raise PayloadError(f"{path} depends on unknown sampled variable '{parent}'")
        distributions = config.get("distributions")
        if not isinstance(distributions, dict) or not distributions:
            raise PayloadError(f"{path}.distributions must be a non-empty object")
        missing = possible_values[parent] - set(distributions)
        if missing:
            raise PayloadError(
                f"{path}.distributions is missing {parent} values: {', '.join(sorted(missing))}"
            )
        normalized_conditional[name] = {
            "depends_on": parent,
            "distributions": {
                parent_value: _weighted_distribution(
                    distribution, f"{path}.distributions.{parent_value}"
                )
                for parent_value, distribution in distributions.items()
            },
        }

    result: dict[str, Any] = {"variables": normalized_variables}
    if normalized_conditional:
        result["conditional_variables"] = normalized_conditional
    return result


def normalize_payload(raw: Any) -> dict[str, Any]:
    """Validate and return the fields consumed by ``ImageAttributeAugmentationDagPayloadConfig``."""
    if not isinstance(raw, dict):
        raise PayloadError("payload must be a JSON object")
    external = raw.get("external_services", True)
    if not isinstance(external, bool):
        raise PayloadError("external_services must be a boolean")
    reporting = raw.get("enable_performance_reporting", False)
    if not isinstance(reporting, bool):
        raise PayloadError("enable_performance_reporting must be a boolean")
    output = _storage_path(raw.get("output_directory"), "output_directory")
    max_imgs = raw.get("max_imgs", 1)
    if max_imgs is not None:
        max_imgs = _integer(max_imgs, "max_imgs")

    lifecycle_raw = raw.get("service_lifecycle") or {}
    if not isinstance(lifecycle_raw, dict):
        raise PayloadError("service_lifecycle must be an object")
    lifecycle: dict[str, Any] = {}
    expected_enabled = not external
    for service in SERVICE_KEYS:
        entry = lifecycle_raw.get(service) or {}
        if not isinstance(entry, dict):
            raise PayloadError(f"service_lifecycle.{service} must be an object")
        replicas = _integer(
            entry.get("replicas", 1), f"service_lifecycle.{service}.replicas", minimum=1
        )
        lifecycle[service] = {"enabled": expected_enabled, "replicas": replicas}

    cosmos_raw = raw.get("cosmos") or {}
    search_raw = raw.get(ATTRIBUTE_SEARCH_KEY) or {}
    if not isinstance(cosmos_raw, dict) or not isinstance(search_raw, dict):
        raise PayloadError(f"cosmos and {ATTRIBUTE_SEARCH_KEY} must be objects")
    for name, nested in (("cosmos", cosmos_raw), (ATTRIBUTE_SEARCH_KEY, search_raw)):
        if nested.get("output_directory", output) != output:
            raise PayloadError(
                f"{name}.output_directory must match top-level output_directory"
            )
        if nested.get("external_services", external) is not external:
            raise PayloadError(
                f"{name}.external_services must match top-level external_services"
            )

    cosmos = {
        "vlm_service_url": _service_url(
            cosmos_raw.get("vlm_service_url"),
            "cosmos.vlm_service_url",
            required=external,
        ),
        "llm_service_url": _service_url(
            cosmos_raw.get("llm_service_url"),
            "cosmos.llm_service_url",
            required=external,
        ),
        "image_edit_service_url": _service_url(
            cosmos_raw.get("image_edit_service_url"),
            "cosmos.image_edit_service_url",
            required=external,
        ),
        "vlm_model": _optional_model(cosmos_raw.get("vlm_model"), "cosmos.vlm_model"),
        "llm_model": _optional_model(cosmos_raw.get("llm_model"), "cosmos.llm_model"),
        "image_edit_model": _optional_model(
            cosmos_raw.get("image_edit_model"), "cosmos.image_edit_model"
        ),
        "num_augmentation": _integer(
            cosmos_raw.get("num_augmentation", 1), "cosmos.num_augmentation", minimum=1
        ),
        "external_services": external,
        "output_directory": output,
        "base_config_path": _optional_storage_path(
            cosmos_raw.get("base_config_path"), "cosmos.base_config_path"
        ),
        "variable_distribution": validate_variable_distribution(
            cosmos_raw.get("variable_distribution")
        ),
    }
    attribute_search = {
        "mode": _nonempty_string(
            search_raw.get("mode", DEFAULT_ATTRIBUTE_SEARCH_MODE),
            f"{ATTRIBUTE_SEARCH_KEY}.mode",
        ),
        "vlm_service_url": _service_url(
            search_raw.get("vlm_service_url"),
            f"{ATTRIBUTE_SEARCH_KEY}.vlm_service_url",
            required=False,
        ),
        "llm_service_url": _service_url(
            search_raw.get("llm_service_url"),
            f"{ATTRIBUTE_SEARCH_KEY}.llm_service_url",
            required=external,
        ),
        "vlm_model": _optional_model(
            search_raw.get("vlm_model"), f"{ATTRIBUTE_SEARCH_KEY}.vlm_model"
        ),
        "llm_model": _optional_model(
            search_raw.get("llm_model"), f"{ATTRIBUTE_SEARCH_KEY}.llm_model"
        ),
        "external_services": external,
        "output_directory": output,
    }

    return {
        "input_path": _storage_path(raw.get("input_path"), "input_path"),
        "max_imgs": max_imgs,
        "output_directory": output,
        "external_services": external,
        "enable_performance_reporting": reporting,
        "service_lifecycle": lifecycle,
        "cosmos": cosmos,
        ATTRIBUTE_SEARCH_KEY: attribute_search,
    }


def render_payload(args: argparse.Namespace) -> dict[str, Any]:
    external = args.service_mode == "external"
    distribution = None
    if args.variable_distribution:
        distribution = json.loads(
            Path(args.variable_distribution).read_text(encoding="utf-8")
        )
    raw = {
        "input_path": args.input_path,
        "max_imgs": args.max_imgs,
        "output_directory": args.output_directory,
        "external_services": external,
        "enable_performance_reporting": args.enable_performance_reporting,
        "cosmos": {
            "vlm_service_url": args.vlm_url,
            "llm_service_url": args.llm_url,
            "image_edit_service_url": args.image_edit_url,
            "vlm_model": args.vlm_model,
            "llm_model": args.llm_model,
            "image_edit_model": args.image_edit_model,
            "num_augmentation": args.num_augmentation,
            "base_config_path": args.base_config_path,
            "variable_distribution": distribution,
        },
        ATTRIBUTE_SEARCH_KEY: {
            "vlm_service_url": args.vlm_url,
            "llm_service_url": args.llm_url,
            "vlm_model": args.vlm_model,
            "llm_model": args.llm_model,
        },
    }
    return normalize_payload(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or validate an SDG Image Attribute Augmentation DAG payload."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Render a validated payload.")
    render.add_argument("--input-path", required=True)
    render.add_argument("--output-directory", required=True)
    render.add_argument(
        "--service-mode", choices=("external", "internal"), required=True
    )
    render.add_argument("--max-imgs", type=int, default=1)
    render.add_argument("--num-augmentation", type=int, default=1)
    render.add_argument(
        "--variable-distribution", help="JSON file with variables/conditionals."
    )
    render.add_argument(
        "--base-config-path",
        help="Optional s3://, http://, or https:// override for the DAG's base cosmos_config.yaml.",
    )
    render.add_argument("--vlm-url")
    render.add_argument("--llm-url")
    render.add_argument("--image-edit-url")
    render.add_argument("--vlm-model", default=DEFAULT_MODELS["vlm"])
    render.add_argument("--llm-model", default=DEFAULT_MODELS["llm"])
    render.add_argument("--image-edit-model", default=DEFAULT_MODELS["image_edit"])
    render.add_argument(
        "--enable-performance-reporting",
        action="store_true",
        help="Write a performance YAML/HTML dashboard under the run's reports/ directory.",
    )
    render.add_argument("--output", required=True, help="Destination JSON file.")
    validate = subparsers.add_parser(
        "validate", help="Validate and normalize a payload."
    )
    validate.add_argument("--payload", required=True)
    validate.add_argument("--output", help="Optional path for normalized JSON.")
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
            _write_json(normalize_payload(raw), args.output)
        return 0
    except (OSError, json.JSONDecodeError, PayloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
