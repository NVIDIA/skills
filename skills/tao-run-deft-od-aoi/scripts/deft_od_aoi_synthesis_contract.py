#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bounded-synthesis policy validation and yield derivation."""

from __future__ import annotations

import math
from typing import Any


MASK_BRANCHES_PER_NEIGHBOR = 2
SELECTION_MODES = {"all_eligible", "generated_per_type_plan"}


def resolve_candidate_topn(
    synthesis: dict[str, Any], legacy_candidate_overfetch: Any = None,
) -> int:
    """Resolve the synthesis-only clean-neighbor search depth."""
    value = synthesis.get("candidate_topn", legacy_candidate_overfetch)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("synthesis.candidate_topn must be a positive integer")
    neighbors = synthesis.get("max_neighbors_per_fn")
    if (isinstance(neighbors, int) and not isinstance(neighbors, bool)
            and value < neighbors):
        raise ValueError(
            "synthesis.candidate_topn must be at least max_neighbors_per_fn"
        )
    return value


def validate_synthesis_contract(synthesis: dict[str, Any]) -> tuple[str, int]:
    selection = synthesis.get("fn_selection") or {}
    if not isinstance(selection, dict):
        raise ValueError("synthesis.fn_selection must be a mapping")
    if "images_per_fn" in selection:
        raise ValueError(
            "synthesis.fn_selection.images_per_fn is derived from max_neighbors_per_fn"
        )
    mode = str(selection.get("mode") or "generated_per_type_plan")
    if mode not in SELECTION_MODES:
        raise ValueError(f"unsupported synthesis.fn_selection.mode: {mode}")
    neighbors = synthesis.get("max_neighbors_per_fn")
    if (isinstance(neighbors, bool) or not isinstance(neighbors, int)
            or neighbors < 1):
        raise ValueError("synthesis.max_neighbors_per_fn must be a positive integer")
    if "candidate_topn" in synthesis:
        resolve_candidate_topn(synthesis)
    if "cumulative_fraction_of_total_defects" in synthesis:
        try:
            fraction = float(synthesis["cumulative_fraction_of_total_defects"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "synthesis.cumulative_fraction_of_total_defects must be in [0, 1)"
            ) from error
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError(
                "synthesis.cumulative_fraction_of_total_defects must be in [0, 1)"
            )
    elif mode == "generated_per_type_plan":
        raise ValueError(
            "generated_per_type_plan requires "
            "synthesis.cumulative_fraction_of_total_defects"
        )
    return mode, MASK_BRANCHES_PER_NEIGHBOR * neighbors
