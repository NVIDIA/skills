#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a staged ODVG source is self-consistent before training.

Cross-checks the staged image directory against ``tmm_odvg.jsonl``:

* every ODVG record must reference an image present on disk (otherwise training
  fails partway through an epoch, after a GPU is already allocated)
* images with no ODVG record are dead weight; ``--prune`` removes them

Stdlib only — no pandas required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--odvg", required=True, help="tmm_odvg.jsonl")
    parser.add_argument("--key-field", default="file_name",
                        help="ODVG field holding the image file name.")
    parser.add_argument("--prune", action="store_true",
                        help="Delete images that have no ODVG record.")
    args = parser.parse_args()

    try:
        image_dir = Path(args.image_dir).expanduser().resolve()
        odvg_path = Path(args.odvg).expanduser().resolve()
        if not image_dir.is_dir():
            raise NotADirectoryError(f"--image-dir does not exist: {image_dir}")
        if not odvg_path.is_file():
            raise FileNotFoundError(f"--odvg does not exist: {odvg_path}")

        referenced: list[str] = []
        malformed = 0
        with odvg_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    print(f"WARNING: malformed JSON at {odvg_path}:{line_no}", file=sys.stderr)
                    continue
                key = record.get(args.key_field)
                if not key:
                    malformed += 1
                    print(
                        f"WARNING: record at {odvg_path}:{line_no} has no '{args.key_field}'",
                        file=sys.stderr,
                    )
                    continue
                referenced.append(Path(str(key)).name)

        on_disk = {p.name for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        referenced_set = set(referenced)

        missing_images = sorted(referenced_set - on_disk)
        orphan_images = sorted(on_disk - referenced_set)
        duplicates = len(referenced) - len(referenced_set)

        print(
            f"odvg_records={len(referenced)} unique={len(referenced_set)} "
            f"images_on_disk={len(on_disk)} missing={len(missing_images)} "
            f"orphans={len(orphan_images)} duplicate_records={duplicates} malformed={malformed}"
        )

        if args.prune and orphan_images:
            for name in orphan_images:
                (image_dir / name).unlink()
            print(f"Pruned {len(orphan_images)} orphan image(s)")
            orphan_images = []

        failures = []
        if not referenced:
            failures.append("ODVG file contains no usable records")
        if missing_images:
            failures.append(
                f"{len(missing_images)} ODVG record(s) reference missing images "
                f"(first: {missing_images[:5]})"
            )
        if duplicates:
            failures.append(f"{duplicates} duplicate ODVG record(s) — re-ran staging in append mode?")
        if malformed:
            # Counted but not fatal, this let one good record carry a file of
            # unparsable ones into training as though staging had succeeded.
            failures.append(
                f"{malformed} malformed ODVG record(s): unparsable JSON, or no file_name. "
                f"Their images stage without labels."
            )

        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1

        if orphan_images:
            print(
                f"NOTE: {len(orphan_images)} image(s) have no ODVG record and will be "
                "ignored during training. Re-run with --prune to remove them."
            )
        print("OK: staged ODVG source is consistent")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
