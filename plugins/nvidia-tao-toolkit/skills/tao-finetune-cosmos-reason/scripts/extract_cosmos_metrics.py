#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and extract authoritative Cosmos metrics from structured outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


SUCCESS = {"SUCCESS", "COMPLETE", "COMPLETED"}
FAILURE = {"FAILURE", "FAILED", "ERROR", "CANCELLED", "CANCELED"}
VLM_COMPONENTS = ("vision_encoder", "vision_projector", "language_model", "lm_head")
COMPONENT_FIELDS = (
    "total_parameters",
    "trainable_parameters",
    "frozen_parameters",
    "trainable_parameter_tensors",
    "parameter_tensors_with_grad",
    "grad_norm",
)


class MetricError(ValueError):
    pass


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        complete = json.loads(text)
    except json.JSONDecodeError:
        complete = None
    if isinstance(complete, dict):
        values = complete.get("records", [complete])
        if isinstance(values, list) and all(isinstance(item, dict) for item in values):
            return values
    if isinstance(complete, list) and all(isinstance(item, dict) for item in complete):
        if not complete:
            raise MetricError("structured status file contains no records")
        return complete
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetricError(f"invalid JSON status record at line {line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            records.extend(value)
        else:
            raise MetricError(f"status record at line {line_number} is not an object")
    if not records:
        raise MetricError("structured status file contains no records")
    return records


def metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("kpi", "metrics", "data"):
        value = record.get(key)
        if isinstance(value, Mapping):
            result.update(value)
    for key, value in record.items():
        if "/" in str(key):
            result[key] = value
    return result


def _weighted_event(record: Mapping[str, Any], prefix: str) -> dict[str, Any] | None:
    values = metrics(record)
    average_keys = (f"{prefix}/avg_loss",) if prefix == "val" else ("train/avg_loss",)
    average = next((finite(values.get(key)) for key in average_keys if finite(values.get(key)) is not None), None)
    numerator = finite(values.get(f"{prefix}/loss_numerator"))
    denominator = finite(values.get(f"{prefix}/valid_label_count"))
    if average is None and numerator is not None and denominator and denominator > 0:
        average = numerator / denominator
    if average is None:
        return None
    if numerator is None or denominator is None or denominator <= 0:
        raise MetricError(f"{prefix} average loss is missing its global numerator or valid-label denominator")
    calculated = numerator / denominator
    if not math.isclose(average, calculated, rel_tol=1e-6, abs_tol=1e-8):
        raise MetricError(f"{prefix} average loss {average} does not match numerator/denominator {calculated}")
    return {
        "average": average, "numerator": numerator, "denominator": denominator,
        "epoch": metrics(record).get("epoch", record.get("epoch")),
        "step": metrics(record).get("step", record.get("step", record.get("iteration"))),
    }


def _visual_gradient_event(record: Mapping[str, Any]) -> dict[str, Any] | None:
    values = metrics(record)
    status = values.get("model/components/visual_gradient_contract")
    if status is None:
        return None
    components: dict[str, dict[str, Any]] = {}
    for component in VLM_COMPONENTS:
        prefix = f"model/components/{component}"
        component_values = {
            field: values[f"{prefix}/{field}"]
            for field in COMPONENT_FIELDS
            if f"{prefix}/{field}" in values
        }
        if component_values:
            components[component] = component_values
    return {
        "status": status,
        "components": components,
        "epoch": values.get("epoch", record.get("epoch")),
        "step": values.get("step", record.get("step", record.get("iteration"))),
    }


def load_evaluation(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"average_validation_accuracy": None, "numerator": None, "denominator": None, "per_task": {}, "excluded_tasks": [], "aggregation": None, "coverage": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    overall = value.get("overall", value) if isinstance(value, dict) else {}
    accuracy = finite(overall.get("accuracy") if isinstance(overall, Mapping) else None)
    correct = finite(overall.get("correct") if isinstance(overall, Mapping) else None)
    total = finite(overall.get("total") if isinstance(overall, Mapping) else None)
    if accuracy is None or correct is None or total is None or total <= 0:
        raise MetricError("evaluation output must contain finite overall accuracy, correct numerator, and positive total denominator")
    if not math.isclose(accuracy, correct / total, rel_tol=1e-6, abs_tol=1e-8):
        raise MetricError("validation accuracy does not match correct/total")
    return {
        "average_validation_accuracy": accuracy, "numerator": correct, "denominator": total,
        "per_task": value.get("per_task", value.get("categories", {})),
        "excluded_tasks": value.get("excluded_tasks", []), "aggregation": value.get("aggregation"),
        "coverage": value.get("coverage"), "evaluator_version": value.get("evaluator_version"),
    }


def summarize_records(records: list[dict[str, Any]], evaluation: Mapping[str, Any] | None = None, backend: str = "auto", require_complete: bool = True) -> dict[str, Any]:
    train_events: list[dict[str, Any]] = []
    validation_events: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    failures: list[str] = []
    states: list[str] = []
    visual_gradient_events: list[dict[str, Any]] = []
    inferred = backend
    for record in records:
        message = str(record.get("message", ""))
        status = str(record.get("status", "")).upper()
        values = metrics(record)
        phase = str(record.get("phase", values.get("phase", "")))
        if status:
            states.append(status)
        if status in FAILURE:
            failures.append(message or status)
        if inferred == "auto":
            haystack = f"{message} {record.get('component', '')}".casefold()
            if "framework" in haystack:
                inferred = "cosmos-framework"
            elif "cosmos-rl" in haystack:
                inferred = "cosmos-rl"
        checkpoint_path = record.get("checkpoint_path", values.get("checkpoint_path", values.get("checkpoint/path")))
        if phase in {"checkpoint_saved", "checkpoint_complete", "checkpoint_submitted"} or checkpoint_path:
            checkpoints.append({"path": checkpoint_path, "phase": phase, "epoch": values.get("epoch", record.get("epoch"))})
        # Step/heartbeat train loss is intentionally never treated as complete-run average.
        if (
            "train/loss_numerator" in values
            and "train/valid_label_count" in values
        ):
            train_events.append(_weighted_event(record, "train"))
        # Only a validation-complete event is authoritative.
        if phase == "validation_complete" and ("val/avg_loss" in values or "val/loss_numerator" in values):
            validation_events.append(_weighted_event(record, "val"))
        visual_gradient_event = _visual_gradient_event(record)
        if visual_gradient_event is not None:
            visual_gradient_events.append(visual_gradient_event)
    train_events = [item for item in train_events if item]
    validation_events = [item for item in validation_events if item]
    terminal = next((state for state in reversed(states) if state in SUCCESS | FAILURE), states[-1] if states else "UNKNOWN")
    evaluation_summary = dict(evaluation or load_evaluation(None))
    missing = []
    if not train_events:
        missing.append("globally reduced token-weighted average training loss")
    if not validation_events:
        missing.append("globally reduced token-weighted final validation loss")
    if evaluation_summary.get("average_validation_accuracy") is None:
        missing.append("average validation accuracy")
    if terminal not in SUCCESS | FAILURE:
        missing.append("terminal TAO status")
    if require_complete and missing:
        raise MetricError("completed metric report is incomplete: " + ", ".join(missing))
    return {
        "schema_version": 2, "backend": inferred, "terminal_status": terminal,
        "average_training_loss": train_events[-1] if train_events else None,
        "average_validation_loss": validation_events[-1] if validation_events else None,
        "validation_history": validation_events,
        "first_update_visual_gradient_contract": (
            visual_gradient_events[0] if visual_gradient_events else None
        ),
        "evaluation": evaluation_summary, "checkpoints": checkpoints, "failures": failures,
        "missing": missing, "source": "structured_status_and_repository_evaluator",
    }


def diagnostic_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    signatures = {
        "out_of_memory": "out of memory", "nccl": "nccl", "decoder": "nvcuvid",
        "python_permission": "permission denied", "traceback": "traceback",
    }
    return {
        "source": "log_diagnostic_only", "authoritative_metrics": False,
        "failure_signatures": [name for name, pattern in signatures.items() if pattern in text.casefold()],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--backend", choices=("auto", "cosmos-framework", "cosmos-rl"), default="auto")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluation = load_evaluation(args.evaluation_json)
        summary = summarize_records(records_from_jsonl(args.status_file), evaluation, args.backend, not args.allow_incomplete)
        if args.log_file:
            summary["diagnostics"] = diagnostic_log(args.log_file)
    except (OSError, json.JSONDecodeError, MetricError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "text":
        print(f"status: {summary['terminal_status']}")
        print(f"average training loss: {summary['average_training_loss']}")
        print(f"average validation loss: {summary['average_validation_loss']}")
        print(f"average validation accuracy: {summary['evaluation']['average_validation_accuracy']}")
        print(
            "first-update visual-gradient contract: "
            f"{summary['first_update_visual_gradient_contract']}"
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["terminal_status"] in FAILURE else 0


if __name__ == "__main__":
    raise SystemExit(main())
