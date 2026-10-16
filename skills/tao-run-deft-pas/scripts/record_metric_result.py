# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate evaluator JSON and atomically commit an PAS DEFT evaluate result.

The metric result JSON (written by parse_pas_metrics.py) is validated against
the contract stored in ``${RESULTS_DIR}/deft_state.json`` under
``metric_contract.{metric_name,query_type,op,target}``; ``passed`` is
recomputed from the contract before the result is committed to state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

from metric_contract import (
    contract_from_state,
    normalize_operator,
    relative_metric_summary,
    result_from_iteration,
    result_passes,
)
from pas_deft.pas_artifacts import PAS_METRICS_AGGREGATE_FILENAME
from parse_pas_metrics import build_result

EXPECTED_WORKFLOW = "tao-run-deft-pas"
EXPECTED_SCHEMA_VERSION = "1"


def _required_file(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(value.expanduser()))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    if path.resolve() != path:
        raise ValueError(f"{name} must not traverse a symlink: {path}")
    return path


def _required_dir(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(value.expanduser()))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing directory: {path}")
    if path.resolve() != path:
        raise ValueError(f"{name} must not traverse a symlink: {path}")
    return path


def _write_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _validate_against_contract(
    raw_result: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    schema_version = raw_result.get("schema_version")
    if schema_version is not None and str(schema_version) != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"metric result schema_version must be {EXPECTED_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    workflow = raw_result.get("workflow")
    if workflow is not None and workflow != EXPECTED_WORKFLOW:
        raise ValueError(
            f"metric result workflow must be {EXPECTED_WORKFLOW!r}, got {workflow!r}"
        )
    if raw_result.get("op") is not None:
        result_op = normalize_operator(str(raw_result["op"]))
        if result_op != contract["op"]:
            raise ValueError(
                f"metric result op {result_op!r} does not match contract op "
                f"{contract['op']!r}"
            )
    result_target = raw_result.get("target")
    contract_target = contract["target"]
    if (result_target is None) != (contract_target is None):
        raise ValueError(
            f"metric result target {result_target!r} does not match contract "
            f"target {contract_target!r}"
        )
    if result_target is not None and not math.isclose(
        float(result_target), contract_target, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"metric result target {result_target!r} does not match contract "
            f"target {contract_target!r}"
        )
    # Checks metric_name/query_type against the contract and value finiteness.
    result = result_from_iteration({"metric_result": raw_result}, contract)
    if result is None:  # pragma: no cover - guarded by the provisional object
        raise ValueError("metric result JSON is empty")
    return result


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("iter label must be baseline or iterN (N >= 1)")
    results_dir = _required_dir(args.results_dir, name="results dir")
    state_path = _required_file(results_dir / "deft_state.json", name="deft state")
    result_path = _required_file(args.metric_result, name="metric result JSON")
    phase_dir = (
        results_dir / "zs"
        if args.iter_label == "baseline"
        else results_dir / f"iter_{args.iter_label[4:]}"
    )
    expected_result_path = pathlib.Path(
        os.path.abspath(phase_dir / "evaluate" / "metric_result.json")
    )
    if result_path != expected_result_path:
        raise ValueError(
            f"metric result for {args.iter_label} must be {expected_result_path}, "
            f"got {result_path}"
        )
    metrics_path = _required_file(args.metrics_csv, name="metrics CSV")
    expected_metrics_path = pathlib.Path(
        os.path.abspath(
            phase_dir / "evaluate" / PAS_METRICS_AGGREGATE_FILENAME
        )
    )
    if metrics_path != expected_metrics_path:
        raise ValueError(
            f"metrics CSV for {args.iter_label} must be {expected_metrics_path}, "
            f"got {metrics_path}"
        )

    state = json.loads(state_path.read_text())
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    contract = contract_from_state(state)
    raw_result = json.loads(result_path.read_text())
    if not isinstance(raw_result, dict):
        raise ValueError("metric result JSON root must be an object")
    result = _validate_against_contract(raw_result, contract)
    if raw_result.get("iter_label") != args.iter_label:
        raise ValueError(
            f"metric result iter_label={raw_result.get('iter_label')!r} does not "
            f"match {args.iter_label!r}"
        )
    try:
        recorded_source = pathlib.Path(
            os.path.abspath(
                pathlib.Path(str(raw_result.get("source_csv", ""))).expanduser()
            )
        )
    except (OSError, ValueError) as exc:
        raise ValueError("metric result source_csv is invalid") from exc
    if recorded_source != metrics_path:
        raise ValueError(
            f"metric result source_csv must be {metrics_path}, got {recorded_source}"
        )
    recomputed = build_result(
        argparse.Namespace(
            metrics_csv=metrics_path,
            metric_name=contract["metric_name"],
            query_type=contract["query_type"],
            op=contract["op"],
            target=contract["target"],
            iter_label=args.iter_label,
        )
    )
    if not math.isclose(
        float(result["value"]), float(recomputed["value"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(
            f"metric result value {result['value']} disagrees with source CSV "
            f"value {recomputed['value']}"
        )
    result.update(recomputed)
    result["evidence_path"] = str(result_path)
    # The contract in deft_state.json is authoritative: recompute passed
    # rather than trusting the evaluator JSON (a null target never passes).
    passed, _ = result_passes(contract, result)
    result["passed"] = passed
    result["relative_change"] = relative_metric_summary(
        state,
        args.iter_label,
        current_result=result,
    )

    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        raise ValueError("state.iterations must be an object")
    phase = iterations.setdefault(args.iter_label, {"status": "in_progress"})
    if not isinstance(phase, dict):
        raise ValueError(f"state.iterations.{args.iter_label} must be an object")
    phase.update(
        {
            "status": "complete",
            "stage_completed": "evaluate",
            "metric_result": result,
        }
    )

    _write_atomic(state_path, state)
    return result


if __name__ == "__main__":
    print("record_metric_result: internal module; use commit_stage.py", file=sys.stderr)
    raise SystemExit(2)
