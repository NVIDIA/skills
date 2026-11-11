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
import re
import warnings
from typing import Any, Iterable


OPERATORS = {"<", "<=", ">", ">="}
MINIMIZING_OPERATORS = {"<", "<="}
METRIC_EVIDENCE_VERSION = "1"
RELATIVE_METRIC_SCHEMA_VERSION = "1"
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


def _relative_change(
    value: float,
    reference: float,
    *,
    minimizing: bool,
) -> tuple[float, str]:
    """Return the signed delta and operator-directed comparison outcome."""
    delta = value - reference
    if math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12):
        return delta, "unchanged"
    improved = delta < 0.0 if minimizing else delta > 0.0
    return delta, "improved" if improved else "regressed"


def relative_evidence_required(state: dict[str, Any]) -> bool:
    """Return whether a run requires persisted relative metric evidence.

    Absence identifies legacy runs. Any present but unsupported version fails
    closed instead of silently falling back to legacy warnings. A future bump
    must update both version constants, evolve ``relative_metric_summary``, and
    teach the audit how to validate or migrate the preceding schema.
    """
    version = state.get("metric_evidence_version")
    if version is None:
        return False
    if version != METRIC_EVIDENCE_VERSION:
        raise ValueError(
            "unsupported metric_evidence_version "
            f"{version!r}; expected {METRIC_EVIDENCE_VERSION!r}"
        )
    return True


def relative_metric_summary(
    state: dict[str, Any],
    iter_label: str,
    *,
    current_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical per-round metric and relative-change evidence.

    ``current_result`` is accepted for the narrow pre-commit window in which
    the evaluator result has been parsed but is not yet present in state. All
    reference values come from already committed state, and comparisons follow
    the approved operator so lower-is-better contracts are described correctly.
    """
    if iter_label == "baseline":
        number = 0
    else:
        match = re.fullmatch(r"iter([1-9][0-9]*)", iter_label)
        if match is None:
            raise ValueError("iter_label must be baseline or iterN (N >= 1)")
        number = int(match.group(1))

    contract = contract_from_state(state)
    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        raise ValueError("state.iterations must be an object")

    if current_result is None:
        current_info = iterations.get(iter_label)
        if not isinstance(current_info, dict):
            raise ValueError(f"state.iterations.{iter_label} must be an object")
        current = result_from_iteration(current_info, contract)
    else:
        if current_result.get("iter_label") != iter_label:
            raise ValueError(
                f"metric result iter_label={current_result.get('iter_label')!r} "
                f"does not match {iter_label!r}"
            )
        if current_result.get("op") is not None and normalize_operator(
            str(current_result["op"])
        ) != contract["op"]:
            raise ValueError("metric result op does not match the metric contract")
        raw_target = current_result.get("target")
        target = contract["target"]
        if (raw_target is None) != (target is None) or (
            raw_target is not None
            and not math.isclose(
                finite_number(raw_target, field="metric_result.target"),
                target,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("metric result target does not match the metric contract")
        current = result_from_iteration(
            {"metric_result": current_result}, contract
        )
    if current is None:  # pragma: no cover - guarded by the branches above
        raise ValueError(f"metric result for {iter_label} is required")

    value = float(current["value"])
    minimizing = contract["op"] in MINIMIZING_OPERATORS
    summary: dict[str, Any] = {
        "schema_version": RELATIVE_METRIC_SCHEMA_VERSION,
        "iter_label": iter_label,
        "metric_name": contract["metric_name"],
        "query_type": contract["query_type"],
        "op": contract["op"],
        "target": contract["target"],
        "value": value,
    }

    if iter_label == "baseline":
        summary.update(
            {
                "baseline_value": None,
                "delta_from_baseline": None,
                "comparison_to_baseline": None,
                "previous_label": None,
                "previous_value": None,
                "delta_from_previous": None,
                "comparison_to_previous": None,
            }
        )
        return summary

    baseline_info = iterations.get("baseline")
    if not isinstance(baseline_info, dict):
        raise ValueError("state.iterations.baseline must be an object")
    baseline = result_from_iteration(baseline_info, contract)
    if baseline is None:
        raise ValueError("committed baseline metric result is required")
    baseline_value = float(baseline["value"])
    delta, outcome = _relative_change(
        value, baseline_value, minimizing=minimizing
    )
    summary.update(
        {
            "baseline_value": baseline_value,
            "delta_from_baseline": delta,
            "comparison_to_baseline": outcome,
        }
    )

    previous_label = "baseline" if number == 1 else f"iter{number - 1}"
    previous_info = iterations.get(previous_label)
    if not isinstance(previous_info, dict):
        raise ValueError(f"state.iterations.{previous_label} must be an object")
    previous = result_from_iteration(previous_info, contract)
    if previous is None:
        raise ValueError(f"committed {previous_label} metric result is required")
    previous_value = float(previous["value"])
    delta, outcome = _relative_change(
        value, previous_value, minimizing=minimizing
    )
    summary.update(
        {
            "previous_label": previous_label,
            "previous_value": previous_value,
            "delta_from_previous": delta,
            "comparison_to_previous": outcome,
        }
    )
    return summary


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
