#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow a KPI class mapping to the run's target classes.

Reads the mapping supplied for the run, keeps only the entries naming a target
class, and writes them in target-class order. Alias lists are copied verbatim.

Inputs:  --mapping (list of single-key dicts, or a nested mapping), --target-classes
Output:  the same shape, filtered; --report-json records kept and dropped classes

Errors:
    a target class absent from the mapping — it would be trained but never scored
    a class whose value is a str rather than a list — ``construct_category_map``
        iterates the value, so a string matches nothing and every metric is zero
        while the run still exits 0

Classes present in the mapping but not in the targets are dropped: ``kpi_analyze``
scores every class it is given, and one the model cannot predict contributes a
constant AP of 0 that compresses the iteration-over-iteration movement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_mapping(path: Path) -> list[dict]:
    """Load as a list of single-key dicts, converting a nested mapping if given."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict):
        data = [{k: v} for k, v in data.items()]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of single-key mappings, got {type(data).__name__}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapping", required=True,
                        help="The KPI class mapping supplied for the run.")
    parser.add_argument("--target-classes", action="append", required=True,
                        help="Target classes; repeatable and/or comma-separated. Normally "
                             "state.config.target_classes.")
    parser.add_argument("--out", required=True, help="Where to write the narrowed mapping.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()

        targets: list[str] = []
        for raw in args.target_classes:
            for item in raw.split(","):
                item = item.strip()
                if item and item not in targets:
                    targets.append(item)
        if not targets:
            raise ValueError("--target-classes resolved to nothing")

        src = Path(args.mapping).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"--mapping does not exist: {src}")
        mapping = load_mapping(src)

        available = {}
        for entry in mapping:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"{src}: every entry must be a single-key mapping, got {entry!r}")
            (name, aliases), = entry.items()
            available[str(name)] = aliases

        missing = [t for t in targets if t not in available]
        if missing:
            raise ValueError(
                f"target class(es) {missing} are absent from {src.name} (it has "
                f"{sorted(available)}). They would be trained and never scored, so the mAP "
                "would describe a different class set than the run."
            )

        kept = [{t: available[t]} for t in targets]
        dropped = [n for n in available if n not in targets]

        stringly = [t for t in targets if isinstance(available[t], str)]
        if stringly:
            raise ValueError(
                f"class(es) {stringly} map to a string rather than a list of aliases. "
                "construct_category_map iterates the value, so a string is read character by "
                "character and matches nothing — every metric comes out zero with no error."
            )

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(kept, fh, sort_keys=False, default_flow_style=False)

        report = {
            "source_mapping": str(src),
            "target_classes": targets,
            "kept": {t: available[t] for t in targets},
            "dropped": dropped,
        }
        if args.report_json:
            rp = Path(args.report_json).expanduser().resolve()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"kpi mapping -> {out}")
        for t in targets:
            print(f"  {t:16s} {len(available[t])} alias(es)")
        if dropped:
            print(f"  dropped (not trained, so not scored): {', '.join(dropped)}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
