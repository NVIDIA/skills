#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the aggregate mAP from ``kpi_calc.csv`` and write it beside the CSV.

``analytics kpi_analyze`` prints the aggregate mAP to stdout and writes it nowhere.
``kpi_calc.csv`` carries one row per class with that class's AP, but no aggregate
row, so the only record of the number the loop compares phases on is a stream that
has to stay captured for the whole stage -- tens of minutes on a large KPI set. A
driver whose shell calls time out, a lost pipe, or a container reaped before
``docker logs`` runs all lose it, and ``commit_stage.py --map-value`` then has
nothing to record.

The aggregate is the unweighted mean of the per-class APs, so it can be recomputed
from the CSV exactly. This reads the CSV, writes ``kpi_summary.json`` next to it,
and prints the value for ``--map-value``.

Rows are identified two ways, because both CSV shapes are in circulation.
tao-data-services #31 added a ``class_name`` column; images built before it write
one unlabeled row per class. With the column, ``Summary`` rows are excluded by name
and every AP is reported against its class. Without it, ``--expect-classes`` is the
only guard -- a row that is not a target class would silently shift the mean, so a
count that disagrees is an error rather than a wrong number.

Inputs:  --kpi-csv, --expect-classes, --out
Output:  the mAP on stdout; kpi_summary.json beside the CSV

Exits 1 on an unreadable CSV, a row count that disagrees with --expect-classes, or
an AP outside [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kpi-csv", required=True,
                        help="kpi_calc.csv written by analytics kpi_analyze.")
    parser.add_argument("--expect-classes", type=int, default=None,
                        help="Number of target classes. On an image predating "
                             "tao-data-services#31 the CSV has no class column, and this "
                             "is the only thing that catches a row which is not one of "
                             "them. Harmless to pass either way.")
    parser.add_argument("--out", default=None,
                        help="Where to write the summary. Default: kpi_summary.json "
                             "beside the CSV.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        csv_path = Path(args.kpi_csv).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"--kpi-csv is not a file: {csv_path}")

        with csv_path.open(encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if any(v.strip() for v in r.values())]
        if not rows:
            raise ValueError(f"{csv_path}: no data rows")
        if "AP" not in rows[0]:
            raise ValueError(
                f"{csv_path}: no AP column; found {sorted(rows[0])}. This does not look "
                "like a kpi_calc.csv")

        # `Summary` is an aggregate row, not a class. kpi.is_internal true appends one;
        # averaging it back in double-counts. Only nameable once class_name exists.
        labelled = "class_name" in rows[0]
        if labelled:
            rows = [r for r in rows if (r.get("class_name") or "").strip().lower() != "summary"]
            if not rows:
                raise ValueError(f"{csv_path}: every row is a Summary row")

        aps: list[float] = []
        for index, row in enumerate(rows):
            raw = (row.get("AP") or "").strip()
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: row {index} has a non-numeric AP {raw!r}") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{csv_path}: row {index} has AP {value}, outside [0, 1]")
            aps.append(value)

        classes = ([str(r.get("class_name", "")).strip() for r in rows] if labelled
                   else [None] * len(rows))
        if args.expect_classes is not None and len(aps) != args.expect_classes:
            raise ValueError(
                f"{csv_path} holds {len(aps)} rows but the run targets "
                f"{args.expect_classes} class(es). The aggregate is the mean of the "
                f"per-class APs, so an extra row -- the Summary row "
                f"`kpi.is_internal: true` appends, for instance -- moves it. Score with "
                f"is_internal false, or pass the row count this CSV should have")

        map_value = sum(aps) / len(aps)
        summary = {
            "kpi_csv": str(csv_path),
            "map_value": map_value,
            "class_count": len(aps),
            "per_class_ap": aps,
            "class_names": classes if labelled else None,
            "per_class": (dict(zip(classes, aps)) if labelled else None),
            "class_names_source": ("kpi_calc.csv" if labelled else
                                   "absent - image predates tao-data-services#31; "
                                   "read them from kpi_analyze.log in row order"),
            "sequence_names": [r.get("Sequence Name") for r in rows],
        }
        out = Path(args.out).expanduser().resolve() if args.out else csv_path.parent / "kpi_summary.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(f"classes:  {len(aps)}")
        if labelled:
            for name, value in zip(classes, aps):
                print(f"  {name}: {value:.4f}")
        else:
            print(f"per-class AP: {[round(a, 4) for a in aps]}")
            print("  (no class_name column: this image predates tao-data-services#31, so "
                  "the order is the one kpi_analyze.log prints)")
        print(f"wrote {out}")
        print(map_value)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"summarize_kpi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
