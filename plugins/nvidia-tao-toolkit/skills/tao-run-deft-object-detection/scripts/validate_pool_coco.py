#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the source pool's COCO is actually usable before the loop trusts it.

The prep chain hands the fold to TAO — Co-DETR's ``inference.category_mapping``
and then ``annotations convert``'s ``kitti.mapping``. Both fail *quietly* when the
class names do not line up: an unmatched name is a log warning, an unmapped
detection is silently dropped, and the conversion still prints
``Execution status: PASS`` and exits 0 while writing a COCO with no annotations.
Nothing downstream notices until training produces a model that cannot detect a
class, several GPU-hours later.

So verify the artifact rather than instrument the transformation:

1. **Every target class carries annotations.** A target with zero boxes means the
   pool holds no examples of it. If that class also appears in the KPI ground
   truth, gap analysis marks those images weak on every iteration while mining is
   structurally unable to find anything — the loop runs its full course looking
   healthy and cannot improve it. Hard error.
2. **Case mismatches are named.** Both TAO consumers match exactly, so ``Car``
   against a mapping that says ``car`` silently drops every box. This is the most
   likely way a fold goes wrong, and the least visible.
3. **Unmapped source classes are reported with counts.** A large ``truck=1523``
   against an identity mapping is the signal that a fold is missing.
4. **Counts reconcile.** Pool images vs COCO images vs annotations, with the known
   causes of a legitimate gap called out so a benign shortfall is not chased.

``--labels-dir`` is optional: without it checks 1 and 4 still run against the COCO
alone. Give it the labels Co-DETR wrote and checks 2 and 3 become available too,
since they need the pre-fold class names.

Stdlib only apart from PyYAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
KITTI_NUMERIC_FIELDS = 14


def split_kitti_line(line: str) -> str | None:
    """Return the class name from a KITTI line, or None if it is malformed.

    The name is not always one token — COCO has ``traffic light`` and ``stop
    sign`` — so find where the numeric tail starts instead of taking ``parts[0]``.
    """
    parts = line.split()
    for idx, token in enumerate(parts):
        try:
            float(token)
        except ValueError:
            continue
        if idx == 0 or len(parts) - idx < KITTI_NUMERIC_FIELDS:
            return None
        return " ".join(parts[:idx])
    return None


def load_targets(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    classes = data.get("classes", data) if isinstance(data, dict) else data
    if isinstance(classes, list):
        return [str(c) for c in classes]
    if isinstance(classes, dict):
        return [str(t) for t in classes]
    raise ValueError(f"{path}: 'classes' must be a list or a mapping.")


def label_histogram(labels_dir: Path) -> tuple[Counter, int]:
    """Class histogram of a KITTI label directory, plus the count of empty files."""
    histogram: Counter = Counter()
    empty = 0
    for label_file in sorted(labels_dir.glob("*.txt")):
        seen = False
        with label_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                name = split_kitti_line(line)
                if name is not None:
                    histogram[name] += 1
                    seen = True
        if not seen:
            empty += 1
    return histogram, empty


def validate(args: argparse.Namespace) -> tuple[dict, list[str], list[str]]:
    coco_path = Path(args.coco).expanduser().resolve()
    if not coco_path.is_file():
        raise FileNotFoundError(f"--coco does not exist: {coco_path}")
    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    targets = load_targets(Path(args.classes).expanduser().resolve())
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    per_class: Counter = Counter()
    for ann in coco.get("annotations", []):
        per_class[categories.get(ann.get("category_id"), "<unknown>")] += 1

    errors: list[str] = []
    warnings: list[str] = []

    # 1. every target class carries annotations
    missing_cat = [t for t in targets if t not in categories.values()]
    empty_cat = [t for t in targets if t in categories.values() and per_class[t] == 0]
    for target in missing_cat:
        errors.append(f"target class '{target}' is absent from the COCO categories entirely")
    for target in empty_cat:
        errors.append(f"target class '{target}' has a category but zero annotations")

    n_images = len(coco.get("images", []))
    n_anns = len(coco.get("annotations", []))
    if n_anns == 0:
        errors.append(
            "the COCO has zero annotations. The usual cause is a mapping whose values are "
            "strings rather than lists (`- car: car`): both TAO consumers iterate the value, "
            "so a string yields the class names 'c','a','r' and nothing matches"
        )

    report: dict = {
        "coco": str(coco_path),
        "target_classes": targets,
        # Grounding DINO's dataset.max_labels caps how many class phrases go into each
        # training caption: the classes present in the image plus randomly sampled ones
        # that are not, which is what teaches the model what a class is NOT. It must
        # equal the class count — any larger value behaves identically (there are no
        # further negatives to sample) and any smaller value truncates them. Settled
        # here, once, so the train spec never has to guess it or carry a stale constant.
        "max_labels": len(targets),
        "coco_images": n_images,
        "coco_annotations": n_anns,
        "annotations_by_class": dict(per_class),
        "prep_inputs": dict(
            item.split("=", 1) for item in args.record if "=" in item
        ),
        "categories": list(categories.values()),
    }

    # 4. reconcile against the pool
    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser().resolve()
        pool = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        report["pool_images"] = len(pool)
        shortfall = len(pool) - n_images
        report["images_not_in_coco"] = shortfall
        if shortfall:
            warnings.append(
                f"{shortfall} of {len(pool)} pool image(s) are absent from the COCO. "
                "Expected when an image has no surviving box: the converter skips images "
                "with no valid annotation (kitti.no_skip defaults to False). Those images "
                "still appear in the embedding parquet, so mining can select one and "
                "staging will report it as missing an annotation — size --min-success-rate "
                "on stage_mined_odvg.py accordingly"
            )

    # 2 & 3 need the pre-fold class names
    if args.labels_dir:
        labels_dir = Path(args.labels_dir).expanduser().resolve()
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"--labels-dir does not exist: {labels_dir}")
        histogram, empty_files = label_histogram(labels_dir)
        report["label_classes"] = dict(histogram)
        report["label_boxes"] = int(sum(histogram.values()))
        report["empty_label_files"] = empty_files

        folded = set(categories.values())
        unmapped = {name: n for name, n in histogram.items() if name not in folded}

        # 2. case-only mismatches — the silent killer, since both consumers match exactly
        lowered = {t.lower(): t for t in folded}
        case_only = {name: lowered[name.lower()]
                     for name in unmapped if name.lower() in lowered}
        if case_only:
            for got, want in sorted(case_only.items()):
                errors.append(
                    f"class '{got}' differs from target '{want}' only by case, and every one "
                    f"of its {unmapped[got]} box(es) was dropped. TAO matches class names "
                    f"exactly — Co-DETR checks `orig_name not in name_to_id` and the "
                    f"converter does `labels2cat.get(row_p[0])`. Add '{got}' verbatim to the "
                    f"sources for '{want}'"
                )
        report["case_only_mismatches"] = case_only

        # 3. genuinely unmapped classes — the missing-fold signal
        dropped = {n: c for n, c in unmapped.items() if n not in case_only}
        report["dropped_by_class"] = dropped
        report["boxes_dropped"] = int(sum(dropped.values()))
        if dropped:
            top = sorted(dropped.items(), key=lambda kv: -kv[1])[:8]
            warnings.append(
                "source class(es) present in the labels but in no group, so dropped: "
                + ", ".join(f"{n}={c}" for n, c in top)
                + ". Intended if you meant to discard them; a missing fold otherwise"
            )

        kept = sum(c for n, c in histogram.items() if n in folded)
        if kept and n_anns < kept:
            warnings.append(
                f"{kept} folded box(es) in the labels but {n_anns} COCO annotation(s). "
                "The converter calls drop_duplicates() per image, so byte-identical KITTI "
                "lines collapse into one, and it discards boxes left degenerate after "
                "clipping. A small gap is those; a large one is not"
            )

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", required=True, help="The COCO json `annotations convert` wrote.")
    parser.add_argument("--classes", required=True, help="The same classes.yaml used to build the fold.")
    parser.add_argument("--labels-dir", default=None,
                        help="KITTI labels Co-DETR wrote, pre-fold. Enables the case-mismatch "
                             "and dropped-class checks.")
    parser.add_argument("--images-dir", default=None,
                        help="Pool image directory. Enables the image-count reconciliation.")
    parser.add_argument("--record", action="append", default=[], metavar="KEY=VALUE",
                        help="Provenance to store under prep_inputs in the report: the "
                             "checkpoint, container image and settings the pool was built "
                             "with. Existence alone cannot show a pool is current, so a "
                             "later run compares these before reusing it. Repeatable.")
    parser.add_argument("--report-json", default=None, help="Where to write the full report.")
    parser.add_argument("--allow-empty-classes", action="store_true",
                        help="Downgrade the empty-target-class error to a warning. Reasonable "
                             "only when a class is listed defensively and its absence from this "
                             "pool is expected.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report, errors, warnings = validate(args)

        print(f"coco images={report['coco_images']} annotations={report['coco_annotations']}")
        print(f"max_labels={report['max_labels']} (one per target class; "
              f"dataset.max_labels for the train spec)")
        for name in report["target_classes"]:
            print(f"  {name:24s} {report['annotations_by_class'].get(name, 0):9d}")
        if "pool_images" in report:
            print(f"pool images={report['pool_images']} "
                  f"not_in_coco={report['images_not_in_coco']}")
        if "label_boxes" in report:
            print(f"labels: {report['label_boxes']} box(es), "
                  f"{report['empty_label_files']} empty file(s), "
                  f"{report['boxes_dropped']} dropped")

        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        empty_class_errors = [e for e in errors if "target class" in e]
        other_errors = [e for e in errors if e not in empty_class_errors]
        if args.allow_empty_classes and empty_class_errors:
            for err in empty_class_errors:
                print(f"WARNING: {err}", file=sys.stderr)
            empty_class_errors = []

        for err in empty_class_errors + other_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        if empty_class_errors or other_errors:
            return 1

        print("OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
