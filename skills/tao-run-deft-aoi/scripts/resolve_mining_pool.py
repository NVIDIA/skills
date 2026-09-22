#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Resolve every mining-pool image against the staged images root."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys


def _candidate_paths(
    row: dict[str, str],
    images_root: pathlib.Path,
    csv_dir: pathlib.Path,
    light: str,
    image_ext: str,
) -> list[pathlib.Path]:
    raw = row.get("filepath") or row.get("input_path") or row.get("image_path") or ""
    value = pathlib.Path(raw)
    candidates = [value] if value.is_absolute() else [csv_dir / value]
    relative = pathlib.Path(*value.parts[1:]) if value.parts[:1] == ("images",) else value
    candidates.append(images_root / relative)
    object_name = row.get("object_name", "")
    filename = f"{object_name}_{light}{image_ext}" if object_name else ""
    if filename:
        if value.is_absolute():
            candidates.append(value / filename)
        else:
            candidates.extend((csv_dir / value / filename, images_root / relative / filename))
    return [path.expanduser() for path in candidates]


def _validate_golden(
    row: dict[str, str],
    images_root: pathlib.Path,
    csv_dir: pathlib.Path,
    light: str,
    image_ext: str,
) -> int:
    raw = row.get("golden_path", "")
    if not raw:
        return 0
    value = pathlib.Path(raw)
    filename = f"{row.get('object_name', '')}_{light}{image_ext}"
    candidates = [value] if value.is_absolute() else [csv_dir / value, images_root / value]
    if row.get("object_name"):
        if value.is_absolute():
            candidates.append(value / filename)
        else:
            candidates.extend((csv_dir / value / filename, images_root / value / filename))
    return len({path.resolve() for path in candidates if path.is_file()})


def resolve(
    csv_path: pathlib.Path,
    images_root: pathlib.Path,
    output: pathlib.Path,
    *,
    light: str = "SolderLight",
    image_ext: str = ".jpg",
) -> dict[str, int]:
    images_root = images_root.expanduser().resolve()
    csv_path = csv_path.expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("mining pool CSV has no header")
        rows = list(reader)
        fieldnames = [*reader.fieldnames]
    if not rows:
        raise ValueError("mining pool CSV has zero data rows")
    if "filepath" not in fieldnames:
        fieldnames.append("filepath")
    failures: list[str] = []
    for index, row in enumerate(rows, start=2):
        existing = {
            path.resolve()
            for path in _candidate_paths(
                row, images_root, csv_path.parent, light, image_ext
            )
            if path.is_file()
        }
        if len(existing) != 1:
            failures.append(f"row {index}: input resolved {len(existing)} files")
            continue
        golden_count = _validate_golden(
            row, images_root, csv_path.parent, light, image_ext
        )
        if row.get("golden_path") and golden_count != 1:
            failures.append(f"row {index}: golden resolved {golden_count} files")
            continue
        resolved = existing.pop()
        try:
            row["filepath"] = str(resolved.relative_to(images_root))
        except ValueError:
            row["filepath"] = str(resolved)
    if failures:
        raise ValueError("; ".join(failures[:20]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "missing": 0, "ambiguous": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument("--images-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-csv", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--light", default="SolderLight")
    parser.add_argument("--image-ext", default=".jpg")
    args = parser.parse_args(argv)
    try:
        image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
        summary = resolve(
            args.csv,
            args.images_root,
            args.output_csv,
            light=args.light,
            image_ext=image_ext,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"resolve_mining_pool: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
