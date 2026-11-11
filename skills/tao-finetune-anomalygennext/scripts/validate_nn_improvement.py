#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish an AnomalyGenNext adapter only when Average.nn_score improves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT = re.compile(r"iter_(\d{9})\.pt")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"empty KPI file: {path}") from exc
        if not header or header[0] != "kpi" or "Average" not in header:
            raise ValueError(f"invalid KPI header: {path}")
        column = header.index("Average")
        for row in rows:
            if row and row[0] == "nn_score" and len(row) > column:
                value = float(row[column])
                if math.isfinite(value):
                    return value
    raise ValueError(f"missing finite Average.nn_score: {path}")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    errors, scores = [], {}
    try:
        recipe = yaml.safe_load(args.recipe.read_text())
        dataset = str(recipe["dataset_name"])
        pairs = recipe["anomaly_types"]
        types = [f"{pair[0]}+{pair[1]}" for pair in pairs]
        if not types or len(types) != len(set(types)):
            raise ValueError("recipe anomaly_types are empty or duplicated")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"invalid recipe: {exc}")
        dataset, types = None, []
    valid = args.run_dir / "valid"
    for path in sorted(valid.glob("*/valid_kpi.csv")) if valid.is_dir() else []:
        try:
            scores[int(path.parent.name)] = _score(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if 0 not in scores:
        errors.append("missing iteration-zero Average.nn_score baseline")
    models = args.run_dir / "checkpoints" / "model"
    eligible = {step: score for step, score in scores.items()
                if step > 0 and (models / f"iter_{step:09d}.pt").is_file()}
    if not eligible:
        errors.append("no scored post-baseline checkpoint")
    best_step = max(eligible, key=eligible.get) if eligible else None
    best_score = eligible.get(best_step) if best_step is not None else None
    pointer = args.run_dir / "checkpoints" / "best_checkpoint.txt"
    selected = pointer.read_text().strip() if pointer.is_file() else ""
    match = CHECKPOINT.fullmatch(selected)
    if not match or best_step is None or int(match.group(1)) != best_step:
        errors.append("best checkpoint pointer does not select maximum eligible NN")
    source = models / selected if selected else None
    if source is None or not source.is_file():
        errors.append("selected model checkpoint is missing")
    baseline = scores.get(0)
    improvement = best_score - baseline if baseline is not None and best_score is not None else None
    if improvement is not None and not improvement > args.min_improvement:
        errors.append(f"NN score did not improve: {improvement} <= {args.min_improvement}")
    outputs = (args.checkpoint_output, args.recipe_output)
    if any(path.exists() for path in outputs):
        errors.append("refusing to overwrite published checkpoint or recipe")
    if not errors:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, args.checkpoint_output)
        shutil.copy2(args.recipe, args.recipe_output)
    state = "COMPLETE" if not errors else "ERROR"
    summary = {"status": state, "metric": "Average.nn_score", "direction": "max",
               "baseline_iteration": 0, "baseline_score": baseline, "best_iteration": best_step,
               "best_score": best_score, "improvement": improvement,
               "required_min_improvement_exclusive": args.min_improvement,
               "scores": {str(key): scores[key] for key in sorted(scores)}, "errors": errors}
    _write(args.output, summary)
    handoff = {"schema_version": 1, "status": state, "dataset_name": dataset,
               "checkpoint": str(args.checkpoint_output.resolve()) if not errors else None,
               "checkpoint_sha256": _sha(args.checkpoint_output) if not errors else None,
               "recipe": str(args.recipe_output.resolve()) if not errors else None,
               "recipe_sha256": _sha(args.recipe_output) if not errors else None,
               "anomaly_types": types, "metric": "Average.nn_score",
               "baseline_score": baseline, "best_score": best_score,
               "improvement": improvement, "errors": errors}
    _write(args.handoff_output, handoff)
    _write(args.status_output, {"status": state, "job_id": args.job_id,
                                "summary": str(args.output.resolve()),
                                "training_handoff": str(args.handoff_output.resolve()),
                                "message": "NN score improved" if not errors else "; ".join(errors)})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    args.output = args.results_dir / "nn_improvement_summary.json"
    args.handoff_output = args.results_dir / "training_handoff.json"
    args.checkpoint_output = args.results_dir / "best_model_checkpoint.pt"
    args.recipe_output = args.results_dir / "canonical_recipe.yaml"
    args.status_output = args.results_dir / "status.json"
    result = validate(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
