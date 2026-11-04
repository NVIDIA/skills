# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract one PAS metric from nvidia_pas_metrics_aggregate.csv into JSON.

TAO ``clip evaluate`` writes a CSV with one row per QueryType:

    Dataset,QueryType,EasyAttribute,num_queries,gallery_size,avg_gt_per_query,\
mAP,Rank-1,Rank-5,Separability,Match@5,Zero@5,First Pos

This script selects the row for ``--query-type``, reads the ``--metric-name``
column, optionally gates it against ``--target`` with ``--op``, and writes a
metric_result.json consumed by record_metric_result.py.  Exit status is 0
whenever the JSON is written successfully — even when the gate is not met —
and nonzero only on structural errors (missing file/column/row, bad value).
Pure stdlib on purpose so it runs under any provisioned interpreter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

WORKFLOW = "tao-run-deft-pas"
SCHEMA_VERSION = "1"
OPERATORS = (">=", ">", "<=", "<")
# Identity columns of the aggregate CSV; every other column is numeric and a
# valid --metric-name.
IDENTITY_COLUMNS = ("Dataset", "QueryType", "EasyAttribute")


def _compare(value: float, operator: str, target: float) -> bool:
    if operator == "<":
        return value < target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    if operator == ">=":
        return value >= target
    raise ValueError(f"metric operator must be one of {list(OPERATORS)}, got {operator!r}")


def _load_rows(csv_path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"metrics CSV has no header row: {csv_path}")
    if "QueryType" not in fieldnames:
        raise ValueError(
            f"metrics CSV is missing the QueryType column: {csv_path} "
            f"(columns: {fieldnames})"
        )
    if not rows:
        raise ValueError(f"metrics CSV has no data rows: {csv_path}")
    return fieldnames, rows


def _select_row(rows: list[dict[str, str]], query_type: str) -> dict[str, str]:
    wanted = query_type.strip().lower()
    matches = [
        row
        for row in rows
        if str(row.get("QueryType", "")).strip().lower() == wanted
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        datasets = sorted(
            {str(row.get("Dataset", "")).strip() or "<empty>" for row in matches}
        )
        raise ValueError(
            f"query type {query_type!r} is ambiguous in metrics CSV: "
            f"{len(matches)} matching rows (Dataset values: {datasets})"
        )
    available = sorted(
        {str(row.get("QueryType", "")).strip() for row in rows} - {""}
    )
    raise ValueError(
        f"query type {query_type!r} not found in metrics CSV; "
        f"available query types: {available}"
    )


def _resolve_metric_column(fieldnames: list[str], metric_name: str) -> str:
    numeric_columns = [name for name in fieldnames if name not in IDENTITY_COLUMNS]
    lookup = {name.lower(): name for name in numeric_columns}
    resolved = lookup.get(metric_name.strip().lower())
    if resolved is None:
        raise ValueError(
            f"metric {metric_name!r} is not a numeric column of the metrics CSV; "
            f"available metric columns: {numeric_columns}"
        )
    return resolved


def _write_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = args.metrics_csv.expanduser()
    if not csv_path.is_file():
        raise ValueError(f"metrics CSV must be an existing file: {csv_path}")
    csv_path = csv_path.resolve()

    fieldnames, rows = _load_rows(csv_path)
    metric_column = _resolve_metric_column(fieldnames, args.metric_name)
    row = _select_row(rows, args.query_type)

    raw_value = str(row.get(metric_column) or "").strip()
    try:
        value = float(raw_value)
    except ValueError:
        raise ValueError(
            f"metric column {metric_column!r} for query type "
            f"{row.get('QueryType')!r} is not numeric: {raw_value!r}"
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"metric column {metric_column!r} for query type "
            f"{row.get('QueryType')!r} must be finite, got {raw_value!r}"
        )
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN (N >= 1)")

    # A null target is a valid "no gate" contract: never passed, exit 0.
    passed = False if args.target is None else _compare(value, args.op, args.target)

    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "iter_label": args.iter_label,
        "metric_name": metric_column,
        "query_type": str(row.get("QueryType", "")).strip(),
        "value": value,
        "op": args.op,
        "target": args.target,
        "passed": passed,
        "source_csv": str(csv_path),
        "row": dict(row),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        required=True,
        type=pathlib.Path,
        help="Path to nvidia_pas_metrics_aggregate.csv",
    )
    parser.add_argument(
        "--metric-name",
        default="Rank-1",
        help="Numeric column header to extract (default: Rank-1)",
    )
    parser.add_argument(
        "--query-type",
        default="medium",
        help="QueryType row to select, case-insensitive (default: medium)",
    )
    parser.add_argument("--op", default=">=", choices=OPERATORS)
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Gate target; omit for an ungated run (passed is always false)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Path to write metric_result.json",
    )
    parser.add_argument(
        "--iter-label", required=True, help='"baseline" or "iter1", "iter2", ...'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_result(args)
        output_path = args.output.expanduser().resolve()
        _write_atomic(output_path, result)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"parse_pas_metrics: {exc}", file=sys.stderr)
        return 2
    target = "none" if result["target"] is None else f"{result['target']:g}"
    print(
        f"metric={result['metric_name']} query_type={result['query_type']} "
        f"value={result['value']:.6g} op={result['op']} target={target} "
        f"passed={str(result['passed']).lower()} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
