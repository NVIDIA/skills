# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and comparison helpers for DEFT customer KPI contracts."""

from __future__ import annotations

import math
import pathlib
import re
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


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("metric_contract must be an object")
    name = contract.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(
            "metric_contract.name must match [a-z][a-z0-9_]*"
        )
    display_name = contract.get("display_name", name)
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("metric_contract.display_name must be non-empty")
    operator = normalize_operator(str(contract.get("operator", "")))
    target = finite_number(contract.get("target"), field="metric_contract.target")
    unit = contract.get("unit", "")
    if not isinstance(unit, str):
        raise ValueError("metric_contract.unit must be a string")

    evaluator = contract.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("metric_contract.evaluator must be an object")
    evaluator_type = evaluator.get("type")
    if evaluator_type != "artifact":
        raise ValueError("metric_contract.evaluator.type must be artifact")
    producer = evaluator.get("producer")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("artifact metric evaluator requires producer")
    path_template = evaluator.get("path_template")
    if not isinstance(path_template, str) or not path_template:
        raise ValueError("artifact metric evaluator requires path_template")
    if "{iter_label}" not in path_template:
        raise ValueError(
            "artifact metric evaluator path_template must contain {iter_label}"
        )
    expanded = path_template.replace("{iter_label}", "baseline")
    if "{" in expanded or "}" in expanded:
        raise ValueError(
            "artifact metric evaluator path_template has an unsupported placeholder"
        )
    if not pathlib.Path(expanded).is_absolute():
        raise ValueError(
            "artifact metric evaluator path_template must resolve to an absolute path"
        )

    constraints = contract.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("metric_contract.constraints must be a list")
    normalized_constraints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise ValueError(f"metric constraint {index} must be an object")
        constraint_name = constraint.get("name")
        if not isinstance(constraint_name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", constraint_name
        ):
            raise ValueError(f"metric constraint {index} has invalid name")
        if constraint_name in seen:
            raise ValueError(f"duplicate metric constraint {constraint_name!r}")
        seen.add(constraint_name)
        constraint_display_name = constraint.get("display_name", constraint_name)
        if (
            not isinstance(constraint_display_name, str)
            or not constraint_display_name.strip()
        ):
            raise ValueError(
                f"metric constraint {constraint_name}.display_name must be non-empty"
            )
        constraint_unit = constraint.get("unit", "")
        if not isinstance(constraint_unit, str):
            raise ValueError(
                f"metric constraint {constraint_name}.unit must be a string"
            )
        normalized_constraints.append(
            {
                "name": constraint_name,
                "display_name": constraint_display_name.strip(),
                "operator": normalize_operator(str(constraint.get("operator", ""))),
                "target": finite_number(
                    constraint.get("target"),
                    field=f"metric constraint {constraint_name}.target",
                ),
                "unit": constraint_unit,
            }
        )

    normalized = dict(contract)
    normalized.update(
        {
            "name": name,
            "display_name": display_name.strip(),
            "operator": operator,
            "target": target,
            "unit": unit,
            "evaluator": dict(evaluator),
            "constraints": normalized_constraints,
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
    unit = contract["unit"]
    suffix = unit if unit == "%" else (f" {unit}" if unit else "")
    return (
        f"{contract['display_name']} {contract['operator']} "
        f"{contract['target']:g}{suffix}"
    )


def result_from_iteration(
    info: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any] | None:
    raw = info.get("metric_result")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("metric_result must be an object")
        result = dict(raw)
        if result.get("name") != contract["name"]:
            raise ValueError(
                f"metric_result.name={result.get('name')!r} does not match "
                f"contract name {contract['name']!r}"
            )
        result["value"] = finite_number(
            result.get("value"), field="metric_result.value"
        )
        if "unit" not in result or not isinstance(result["unit"], str):
            raise ValueError("metric_result.unit must be present and be a string")
        if result["unit"] != contract["unit"]:
            raise ValueError(
                f"metric_result.unit={result['unit']!r} does not match "
                f"contract unit {contract['unit']!r}"
            )
        if not isinstance(result.get("constraints", {}), dict):
            raise ValueError("metric_result.constraints must be an object")
        return result
    return None


def result_passes(
    contract: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    value = finite_number(result.get("value"), field="metric_result.value")
    if not compare(value, contract["operator"], contract["target"]):
        failures.append(contract["name"])
    failures.extend(constraint_failures(contract, result))
    return not failures, failures


def constraint_failures(
    contract: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    """Return missing or failed secondary constraint names."""
    failures: list[str] = []
    constraint_values = result.get("constraints", {})
    for constraint in contract.get("constraints", []):
        name = constraint["name"]
        if name not in constraint_values:
            failures.append(f"{name}:missing")
            continue
        constraint_value = finite_number(
            constraint_values[name], field=f"metric_result.constraints.{name}"
        )
        if not compare(
            constraint_value, constraint["operator"], constraint["target"]
        ):
            failures.append(name)
    return failures


def pick_best(
    candidates: Iterable[tuple[str, dict[str, Any], dict[str, Any]]],
    contract: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    materialized = list(candidates)
    if not materialized:
        raise ValueError("no metric-bearing candidates")
    feasible = [
        candidate
        for candidate in materialized
        if not constraint_failures(contract, candidate[2])
    ]
    selection_pool = feasible or materialized
    reverse = contract["operator"] not in MINIMIZING_OPERATORS
    return sorted(
        selection_pool,
        key=lambda item: float(item[2]["value"]),
        reverse=reverse,
    )[0]
