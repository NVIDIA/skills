#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that every place a class list appears in a DEFT OD run agrees.

The run states its classes in five places and nothing compares them:

  * ``deft_state.json`` ``config.target_classes`` — what the run trains
  * the KPI class mapping — what gets scored
  * ``dataset.infer_data_sources.captions`` — Grounding DINO's label map
  * the staged ODVG ``labelmap.json`` — what training's labels mean
  * ``classes.yaml`` — how the pool was folded

A disagreement does not raise anywhere. Grounding DINO assigns a detection to a
class by the *position* of the caption it matched, so a reordered or short caption
list relabels every prediction: the run exits 0, box counts barely move, and the
per-class AP is measuring different classes than it names.

Checks, for whichever inputs are given:

  captions vs KPI classes    every scored class must be predictable
  captions vs labelmap       same order; the labelmap's index is the label id
  captions vs max_labels     must be equal
  target classes             must be a subset of captions
  classes.yaml               must declare exactly the target classes

Exits 1 on any mismatch, naming both sides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load(path: str):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    text = p.read_text(encoding="utf-8")
    return yaml.safe_load(text) if p.suffix.lower() in {".yaml", ".yml"} else json.loads(text)


def _mapping_classes(data) -> list[str]:
    """Target class names from a KPI mapping or a classes.yaml.

    Accepts the same shapes the rest of the skill does: a list of single-key
    mappings, a bare mapping, or a mapping under a ``classes:`` root key. Without
    the last of those, a documented classes.yaml yields the single class name
    ``classes`` and every comparison against it is a false mismatch --
    prepare_class_mappings_for_mining_data_prep.py unwraps it, so this must too.
    """
    if isinstance(data, dict) and set(data) == {"classes"}:
        data = data["classes"]
    if isinstance(data, list):
        return [k for entry in data if isinstance(entry, dict) for k in entry]
    if isinstance(data, dict):
        return list(data)
    raise ValueError("class mapping must be a list of single-key mappings or a mapping")


def _labelmap_order(data) -> list[str]:
    """Class names in label-id order from an ODVG labelmap ({"0": "car", ...})."""
    if not isinstance(data, dict):
        raise ValueError("labelmap.json must be a mapping of label id to class name")
    return [name for _, name in sorted(data.items(), key=lambda kv: int(kv[0]))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", default=None,
                        help='JSON list, e.g. \'["bicycle","car","person"]\'.')
    parser.add_argument("--inference-spec", default=None,
                        help="Read captions and max_labels from this spec instead.")
    parser.add_argument("--state", default=None, help="deft_state.json.")
    parser.add_argument("--kpi-mapping", default=None, help="KPI class mapping yaml.")
    parser.add_argument("--labelmap", default=None, help="Staged ODVG labelmap.json.")
    parser.add_argument("--classes", default=None, help="Pool classes.yaml.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        problems: list[str] = []
        seen: dict[str, list[str]] = {}

        captions: list[str] | None = None
        max_labels = None
        if args.captions:
            captions = json.loads(args.captions)
        elif args.inference_spec:
            spec = _load(args.inference_spec)
            sources = (spec.get("dataset") or {}).get("infer_data_sources") or {}
            captions = sources.get("captions")
            max_labels = (spec.get("dataset") or {}).get("max_labels")
        if captions is not None:
            if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
                raise ValueError("captions must be a list of strings")
            seen["captions"] = captions
            if len(set(captions)) != len(captions):
                problems.append(f"captions contains duplicates: {captions}")

        if args.state:
            config = (_load(args.state) or {}).get("config") or {}
            targets = config.get("target_classes")
            if targets:
                seen["state.config.target_classes"] = list(targets)

        if args.kpi_mapping:
            seen["kpi mapping"] = _mapping_classes(_load(args.kpi_mapping))
        if args.labelmap:
            seen["odvg labelmap"] = _labelmap_order(_load(args.labelmap))
        if args.classes:
            seen["classes.yaml"] = _mapping_classes(_load(args.classes))

        if len(seen) < 2:
            raise ValueError("give at least two class sources to compare")

        # Every scored class must be predictable. A ground-truth class missing from
        # captions is a permanent false negative and its objects land on whichever
        # caption the model matches instead.
        if captions is not None and "kpi mapping" in seen:
            missing = [c for c in seen["kpi mapping"] if c not in captions]
            if missing:
                problems.append(
                    f"KPI scores {missing} but they are absent from captions {captions}; "
                    f"they can never be predicted and their objects become false positives "
                    f"on another class"
                )

        # Order is the label map, not a presentation detail.
        if captions is not None and "odvg labelmap" in seen:
            lm = seen["odvg labelmap"]
            shared = [c for c in captions if c in lm]
            if [c for c in lm if c in captions] != shared:
                problems.append(
                    f"captions order {captions} disagrees with the ODVG labelmap order {lm}; "
                    f"the labelmap index is the label id, so every prediction is relabeled"
                )

        if captions is not None and max_labels is not None and max_labels != len(captions):
            problems.append(
                f"dataset.max_labels is {max_labels} but there are {len(captions)} captions")

        if captions is not None and "state.config.target_classes" in seen:
            extra = [c for c in seen["state.config.target_classes"] if c not in captions]
            if extra:
                problems.append(
                    f"target classes {extra} are absent from captions {captions}; "
                    f"the loop cannot improve a class the model cannot predict")

        if "classes.yaml" in seen and "state.config.target_classes" in seen:
            a, b = set(seen["classes.yaml"]), set(seen["state.config.target_classes"])
            if a != b:
                problems.append(
                    f"classes.yaml declares {sorted(a)} but the run targets {sorted(b)}; "
                    f"the pool was folded to a different class set than the loop trains")

        report = {"sources": seen, "problems": problems}
        if args.report_json:
            Path(args.report_json).expanduser().resolve().write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")

        for name, values in seen.items():
            print(f"{name}: {values}")
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(f"OK: {len(seen)} class source(s) agree")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"verify_class_contract: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
