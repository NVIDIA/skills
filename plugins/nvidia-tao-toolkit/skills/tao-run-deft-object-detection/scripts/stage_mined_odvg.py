#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage mined images into a trainable ODVG source for Grounding DINO.

Reads the miner's ``final_unique_files.parquet``, copies each mined image into
an output image directory, looks up that image's ODVG annotation from the
source-pool annotation tree, renumbers ``image_id`` sequentially, remaps each
instance label through the labelmap, and writes ``tmm_odvg.jsonl`` plus
``labelmap.json``.

Differences from the reference KFP implementation, both deliberate:

* ``tmm_odvg.jsonl`` is **truncated** before writing. The reference opened it in
  append mode, so re-running an iteration silently duplicated every entry.
* Both ``filepath`` and ``source_filepath`` are accepted as the image column,
  because the two miner generations disagree on the name.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_COLUMNS = ("filepath", "source_filepath")


def resolve_image_column(df: pd.DataFrame) -> str:
    for column in IMAGE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(
        f"mined parquet has none of {IMAGE_COLUMNS}; columns are {list(df.columns)}"
    )


def load_annotations(annotations_base_dir: Path) -> tuple[dict[str, dict], dict | None, set[str]]:
    """Index every ODVG record under ``annotations_base_dir`` by ``file_name``."""
    if not annotations_base_dir.is_dir():
        raise NotADirectoryError(f"annotations-base-dir does not exist: {annotations_base_dir}")

    labelmap: dict | None = None
    for labelmap_file in sorted(annotations_base_dir.rglob("*labelmap.json")):
        try:
            with labelmap_file.open("r", encoding="utf-8") as fh:
                labelmap = json.load(fh)
            print(f"Found labelmap: {labelmap_file}")
            break
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read {labelmap_file}: {exc}", file=sys.stderr)

    by_file_name: dict[str, dict] = {}
    # Records are keyed by basename, so two source directories holding the same
    # file name resolve to whichever was read last. Those names are unusable:
    # staging would attach one image's boxes to a different image.
    ambiguous: set[str] = set()
    categories: set[str] = set()
    jsonl_files = sorted(annotations_base_dir.rglob("*.jsonl"))
    for jsonl_file in jsonl_files:
        try:
            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    file_name = record.get("file_name")
                    if not file_name:
                        continue
                    if file_name in by_file_name and by_file_name[file_name] != record:
                        ambiguous.add(file_name)
                    by_file_name[file_name] = record
                    for instance in record.get("detection", {}).get("instances", []):
                        if "category" in instance:
                            categories.add(instance["category"])
        except OSError as exc:
            print(f"WARNING: could not read {jsonl_file}: {exc}", file=sys.stderr)

    for name in sorted(ambiguous):
        by_file_name.pop(name, None)
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} basename(s) carry conflicting annotations and were "
              f"dropped: {sorted(ambiguous)[:5]}", file=sys.stderr)
    print(f"Indexed {len(by_file_name)} annotations from {len(jsonl_files)} JSONL file(s)")
    return by_file_name, labelmap, categories


def build_labelmap(labelmap: dict | None, categories: set[str]) -> tuple[dict, dict[str, int]]:
    if labelmap is None:
        if not categories:
            raise ValueError(
                "no labelmap.json found and no categories observed — cannot build a label map"
            )
        labelmap = {str(idx): category for idx, category in enumerate(sorted(categories))}
        print(f"No labelmap found; synthesized one from {len(categories)} categories")
    category_to_label = {category: int(key) for key, category in labelmap.items()}
    return labelmap, category_to_label


def stage(args: argparse.Namespace) -> dict[str, Any]:
    mined_parquet = Path(args.mined_parquet).expanduser().resolve()
    images_dir = Path(args.output_images_dir).expanduser().resolve()
    annotations_dir = Path(args.output_annotations_dir).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    by_file_name, raw_labelmap, categories = load_annotations(
        Path(args.annotations_base_dir).expanduser().resolve()
    )
    labelmap, category_to_label = build_labelmap(raw_labelmap, categories)

    df = pd.read_parquet(mined_parquet)
    image_column = resolve_image_column(df)
    df = df.drop_duplicates(subset=[image_column])
    print(f"Staging {len(df)} unique mined image(s) from {mined_parquet}")

    # Destination names are basenames, so two distinct sources sharing one would
    # overwrite each other and the survivor would be annotated with the other's
    # boxes. Refuse rather than silently stage corrupt training data; renaming is
    # not an option because the annotation index is keyed by basename.
    basenames: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for raw_path in df[image_column]:
        src = str(raw_path)
        name = Path(src).name
        if name in basenames and basenames[name] != src:
            collisions.append((basenames[name], src))
        basenames.setdefault(name, src)
    if collisions:
        detail = "; ".join(f"{Path(a).name}: {a} vs {b}" for a, b in collisions[:5])
        raise ValueError(
            f"{len(collisions)} mined image(s) share a basename with a different source "
            f"path. Staging flattens to the basename, so one would overwrite the other and "
            f"inherit its annotations. {detail}"
        )

    odvg_path = annotations_dir / "tmm_odvg.jsonl"
    copied_images = 0
    written_annotations = 0
    relabeled = 0
    missing_images: list[str] = []
    missing_annotations: list[str] = []

    # Truncating write: re-running an iteration must not duplicate entries.
    with odvg_path.open("w", encoding="utf-8") as odvg_fh:
        for raw_path in df[image_column]:
            src = Path(str(raw_path))
            if not src.is_file():
                missing_images.append(str(src))
                continue

            shutil.copy2(src, images_dir / src.name)
            copied_images += 1

            record = by_file_name.get(src.name)
            if record is None:
                missing_annotations.append(src.name)
                continue

            record = json.loads(json.dumps(record))  # deep copy
            record["image_id"] = written_annotations
            for instance in record.get("detection", {}).get("instances", []):
                category = instance.get("category")
                if category in category_to_label:
                    new_label = category_to_label[category]
                    if instance.get("label") != new_label:
                        relabeled += 1
                    instance["label"] = new_label
            odvg_fh.write(json.dumps(record) + "\n")
            written_annotations += 1

    with (annotations_dir / "labelmap.json").open("w", encoding="utf-8") as fh:
        json.dump(labelmap, fh, indent=2)

    report = {
        "mined_parquet": str(mined_parquet),
        "image_column": image_column,
        "mined_unique": int(len(df)),
        "images_copied": copied_images,
        "annotations_written": written_annotations,
        "labels_remapped": relabeled,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "labelmap_size": len(labelmap),
        "odvg_path": str(odvg_path),
        "images_dir": str(images_dir),
    }
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-parquet", required=True, help="Miner final_unique_files.parquet.")
    parser.add_argument("--annotations-base-dir", required=True,
                        help="Source-pool ODVG tree (searched recursively for *.jsonl and *labelmap.json).")
    parser.add_argument("--output-images-dir", required=True)
    parser.add_argument("--output-annotations-dir", required=True)
    parser.add_argument("--report-json", default=None, help="Where to write the staging report JSON.")
    parser.add_argument("--min-success-rate", type=float, default=0.9,
                        help="Fail when annotations_written/mined_unique falls below this (0.0-1.0).")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report = stage(args)
        total = report["mined_unique"]
        written = report["annotations_written"]
        rate = (written / total) if total else 0.0
        print(
            f"staged images={report['images_copied']} annotations={written}/{total} "
            f"({rate:.1%}) remapped={report['labels_remapped']} "
            f"missing_images={len(report['missing_images'])} "
            f"missing_annotations={len(report['missing_annotations'])}"
        )
        if written == 0:
            print("ERROR: no annotations were staged — training would have no new data.", file=sys.stderr)
            return 1
        if rate < args.min_success_rate:
            print(
                f"ERROR: staging success rate {rate:.1%} is below --min-success-rate "
                f"{args.min_success_rate:.1%}.",
                file=sys.stderr,
            )
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
