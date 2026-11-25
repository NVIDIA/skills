# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataclass field helpers with self-documenting, validated metadata.

This is the same field-metadata contract used by the PAS notebook runtime and
mirrors ``nvidia_tao_core.config.utils.types``: descriptions, defaults, bounds,
and supported options live beside each field instead of in a separate parser.
"""

from __future__ import annotations

import copy
from dataclasses import field
from typing import Any


def _base_metadata(
    value_type: str,
    value: Any,
    default_value: Any,
    meta_args: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "display_name": "",
        "value_type": value_type,
        "description": "",
        "default_value": default_value,
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
    }
    metadata.update(meta_args)
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return metadata


def STR_FIELD(value: Any, **meta_args: Any):
    """Return a documented string field with optional valid choices."""
    return field(
        default=value,
        metadata=_base_metadata("string", value, "", meta_args),
    )


def INT_FIELD(value: Any, **meta_args: Any):
    """Return a documented integer field with optional numeric bounds."""
    return field(
        default=value,
        metadata=_base_metadata("int", value, "", meta_args),
    )


def FLOAT_FIELD(value: Any, **meta_args: Any):
    """Return a documented float field with optional numeric bounds."""
    return field(
        default=value,
        metadata=_base_metadata("float", value, "", meta_args),
    )


def BOOL_FIELD(value: Any, **meta_args: Any):
    """Return a documented Boolean field."""
    return field(
        default=value,
        metadata=_base_metadata("bool", value, "", meta_args),
    )


def DATACLASS_FIELD(default_instance: Any, **meta_args: Any):
    """Return a documented nested-dataclass field with an isolated default."""
    return field(
        default_factory=lambda: copy.deepcopy(default_instance),
        metadata=_base_metadata("collection", "", "", meta_args),
    )
