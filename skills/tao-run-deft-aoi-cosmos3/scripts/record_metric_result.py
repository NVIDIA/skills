# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and record a frozen Benchmark metric result in Cosmos3 DEFT state."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

from metric_contract import contract_from_state, result_from_iteration, result_passes


def _required_file(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = value.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    return path.resolve()


def _required_checkpoint(
    value: pathlib.Path | None, *, name: str
) -> pathlib.Path:
    if value is None:
        raise ValueError(f"{name} is required")
    path = value.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if not path.exists():
        raise ValueError(f"{name} must exist: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"{name} must not be empty: {path}")
    if path.is_dir() and not any(path.iterdir()):
        raise ValueError(f"{name} directory must not be empty: {path}")
    return path.resolve()


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


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("iter label must be baseline or iterN (N >= 1)")
    state_path = _required_file(args.state_path, name="state path")
    result_path = _required_file(args.result_json, name="metric result JSON")
    benchmark_results = _required_file(
        args.benchmark_results, name="Benchmark results JSON"
    )
    training_spec = (
        _required_file(args.training_spec, name="training spec")
        if args.training_spec
        else None
    )

    state = json.loads(state_path.read_text())
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    if args.iter_label == "baseline":
        evaluated_model = state.get("config", {}).get("base_model")
        if not isinstance(evaluated_model, str) or not evaluated_model:
            raise ValueError("state.config.base_model is required for baseline")
        checkpoint = None
    else:
        checkpoint = _required_checkpoint(
            args.best_ckpt, name="best checkpoint"
        )
        evaluated_model = str(checkpoint)
    contract = contract_from_state(state)
    evaluator = contract["evaluator"]
    if evaluator["type"] == "artifact":
        expected_result = pathlib.Path(
            evaluator["path_template"].replace("{iter_label}", args.iter_label)
        ).expanduser().resolve()
        if result_path != expected_result:
            raise ValueError(
                "metric result JSON does not match the configured artifact path: "
                f"expected {expected_result}, got {result_path}"
            )
    raw_result = json.loads(result_path.read_text())
    if not isinstance(raw_result, dict):
        raise ValueError("metric result JSON root must be an object")
    raw_result["evidence_path"] = str(result_path)
    provisional = {"metric_result": raw_result}
    result = result_from_iteration(provisional, contract)
    if result is None:  # pragma: no cover - guarded by the provisional object
        raise ValueError("metric result JSON is empty")
    passed, _ = result_passes(contract, result)
    result["passed"] = passed

    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        raise ValueError("state.iterations must be an object")
    phase = iterations.setdefault(args.iter_label, {"status": "in_progress"})
    if not isinstance(phase, dict):
        raise ValueError(f"state.iterations.{args.iter_label} must be an object")
    phase.update(
        {
            "status": "complete",
            "stage_completed": "benchmark_metrics",
            "evaluated_model": evaluated_model,
            "benchmark_results_json": str(benchmark_results),
            "metric_result": result,
        }
    )
    if checkpoint is not None:
        phase["best_ckpt_path"] = str(checkpoint)
    if training_spec is not None:
        phase["training_spec"] = str(training_spec)

    _write_atomic(state_path, state)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", required=True, type=pathlib.Path)
    parser.add_argument(
        "--iter-label", required=True, help='"baseline" or "iter1", "iter2", ...'
    )
    parser.add_argument("--result-json", required=True, type=pathlib.Path)
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument(
        "--benchmark-results", required=True, type=pathlib.Path
    )
    parser.add_argument("--training-spec", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = commit(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"record_metric_result: {exc}", file=sys.stderr)
        return 2
    print(
        f"metric={result['name']} value={result['value']:.6g} "
        f"passed={str(result['passed']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
