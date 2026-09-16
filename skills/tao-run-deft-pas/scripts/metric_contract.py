# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and comparison helpers for PAS DEFT metric contracts.

A People Attribute Search (PAS) metric contract is a dict with the keys
``metric_name``, ``query_type``, ``op`` and ``target``.  ``target`` may be
``None``: that is a valid contract meaning "no gate; run to max_iterations",
and a null-target contract never passes.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Iterable


OPERATORS = {"<", "<=", ">", ">="}
MINIMIZING_OPERATORS = {"<", "<="}
_OPERATOR_ALIASES = {
    "lt": "<",
    "le": "<=",
    "lte": "<=",
    "gt": ">",
    "ge": ">=",
    "gte": ">=",
}

# Gate-able metrics from nvidia_pas_metrics_aggregate.csv. All are
# higher-is-better except Zero@5 (fraction of queries with zero hits in the
# top five results), which is lower-is-better.
METRIC_NAMES = ("mAP", "Rank-1", "Rank-5", "Separability", "Match@5", "Zero@5")
LOWER_IS_BETTER_METRICS = {"Zero@5"}
# The PAS aggregate evaluator emits one row for each of these three
# gate-able splits. Broader caption categories remain valid gap-analysis input
# filters, but they are not KPI rows in nvidia_pas_metrics_aggregate.csv.
QUERY_TYPES = ("easy", "medium", "hard")


def normalize_operator(value: str) -> str:
    operator = _OPERATOR_ALIASES.get(str(value).strip().lower(), str(value).strip())
    if operator not in OPERATORS:
        raise ValueError(
            f"metric operator must be one of {sorted(OPERATORS)}, got {value!r}"
        )
    return operator


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def compare(value: float, operator: str, target: float) -> bool:
    operator = normalize_operator(operator)
    if operator == "<":
        return value < target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    return value >= target


def normalize_metric_name(value: Any) -> str:
    """Return the canonical spelling of a known PAS metric name."""
    text = str(value).strip()
    canonical = {name.lower(): name for name in METRIC_NAMES}.get(text.lower())
    if canonical is None:
        raise ValueError(
            f"metric_contract.metric_name must be one of {list(METRIC_NAMES)}, "
            f"got {value!r}"
        )
    return canonical


def normalize_query_type(value: Any) -> str:
    """Return the canonical spelling of a known PAS query type."""
    text = str(value).strip()
    canonical = {name.lower(): name for name in QUERY_TYPES}.get(text.lower())
    if canonical is None:
        raise ValueError(
            f"metric_contract.query_type must be one of {list(QUERY_TYPES)}, "
            f"got {value!r}"
        )
    return canonical


def metric_direction(metric_name: str) -> str:
    """Return ``"lower"`` for lower-is-better metrics, else ``"higher"``."""
    return "lower" if normalize_metric_name(metric_name) in LOWER_IS_BETTER_METRICS else "higher"


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an PAS metric contract.

    Unknown metric names, query types, and operators raise ``ValueError``.
    An operator whose gate direction disagrees with the metric's natural
    direction (e.g. ``Rank-1 <= x`` or ``Zero@5 >= x``) only warns: it is
    unusual but a customer may legitimately request it.  A ``None`` target
    is valid and means "no gate; run to max_iterations".
    """
    if not isinstance(contract, dict):
        raise ValueError("metric_contract must be an object")
    metric_name = normalize_metric_name(contract.get("metric_name"))
    query_type = normalize_query_type(contract.get("query_type"))
    operator = normalize_operator(str(contract.get("op", "")))
    raw_target = contract.get("target")
    target = (
        None
        if raw_target is None
        else finite_number(raw_target, field="metric_contract.target")
    )

    minimizing = operator in MINIMIZING_OPERATORS
    if metric_direction(metric_name) == "lower" and not minimizing:
        warnings.warn(
            f"{metric_name} is lower-is-better but op {operator!r} gates upward; "
            "double-check the metric contract",
            stacklevel=2,
        )
    elif metric_direction(metric_name) == "higher" and minimizing:
        warnings.warn(
            f"{metric_name} is higher-is-better but op {operator!r} gates downward; "
            "double-check the metric contract",
            stacklevel=2,
        )

    normalized = dict(contract)
    normalized.update(
        {
            "metric_name": metric_name,
            "query_type": query_type,
            "op": operator,
            "target": target,
        }
    )
    return normalized


def contract_from_state(state: dict[str, Any]) -> dict[str, Any]:
    contract = state.get("metric_contract")
    if contract is None:
        raise ValueError("state.metric_contract is required")
    return validate_contract(contract)


def render_target(contract: dict[str, Any]) -> str:
    contract = validate_contract(contract)
    if contract["target"] is None:
        return (
            f"{contract['metric_name']} ({contract['query_type']}): "
            "no gate; run to max_iterations"
        )
    return (
        f"{contract['metric_name']} ({contract['query_type']}) "
        f"{contract['op']} {contract['target']:g}"
    )


def result_from_iteration(
    info: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any] | None:
    raw = info.get("metric_result")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("metric_result must be an object")
    result = dict(raw)
    result_metric = str(result.get("metric_name", "")).strip()
    if result_metric.lower() != contract["metric_name"].lower():
        raise ValueError(
            f"metric_result.metric_name={result.get('metric_name')!r} does not "
            f"match contract metric_name {contract['metric_name']!r}"
        )
    result["metric_name"] = contract["metric_name"]
    result_query_type = str(result.get("query_type", "")).strip()
    if result_query_type.lower() != contract["query_type"].lower():
        raise ValueError(
            f"metric_result.query_type={result.get('query_type')!r} does not "
            f"match contract query_type {contract['query_type']!r}"
        )
    result["query_type"] = contract["query_type"]
    result["value"] = finite_number(result.get("value"), field="metric_result.value")
    return result


def result_passes(
    contract: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Return (passed, failed-metric-names) for a result against a contract.

    A null-target contract is ungated: it never passes (the loop runs to
    max_iterations) and reports no failures.
    """
    value = finite_number(result.get("value"), field="metric_result.value")
    if contract.get("target") is None:
        return False, []
    target = finite_number(contract.get("target"), field="metric_contract.target")
    failures: list[str] = []
    if not compare(value, contract["op"], target):
        failures.append(contract["metric_name"])
    return not failures, failures


def pick_best(
    candidates: Iterable[tuple[str, dict[str, Any], dict[str, Any]]],
    contract: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Pick the best (label, info, result) candidate for the contract metric.

    Ordering follows the approved comparison operator.  The contract is the
    source of truth even when its direction is unusual for the metric.
    """
    materialized = list(candidates)
    if not materialized:
        raise ValueError("no metric-bearing candidates")
    reverse = normalize_operator(contract["op"]) not in MINIMIZING_OPERATORS
    return sorted(
        materialized,
        key=lambda item: float(item[2]["value"]),
        reverse=reverse,
    )[0]
