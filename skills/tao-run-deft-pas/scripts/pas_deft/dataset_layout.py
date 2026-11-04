# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only PAS dataset layout discovery.

This mirrors the PAS reference notebook's layout report so users can see the
actual crop, pairs/query, and caption locations before those paths are used by
the DEFT configuration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from pas_deft.pairs_io import iter_json_records


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _peek_json_records(path: Path, sample_limit: int = 3) -> dict[str, Any] | None:
    """Return row and query-type counts for a readable JSON record file."""
    sample: list[dict[str, Any]] = []
    query_types: dict[str, int] = {}
    count = 0
    try:
        for row in iter_json_records(path):
            if not isinstance(row, dict):
                continue
            count += 1
            if len(sample) < sample_limit:
                sample.append(row)
            query_type = str(row.get("query_type") or "").strip()
            if query_type:
                query_types[query_type] = query_types.get(query_type, 0) + 1
    except (OSError, ValueError):
        return None
    return {"count": count, "sample": sample, "query_types": query_types}


def report_dataset_layout(
    dataset_root: str | os.PathLike[str], max_depth: int = 6, top_n: int = 15
) -> dict[str, list[Any]]:
    """Scan and report crop, pairs/query, and caption locations.

    The operation is read-only. Directories containing images are classified
    as crops, JSON files whose sampled records contain ``caption`` or
    ``image_path`` are classified as pairs/query files, and directories
    containing ``.txt`` files are classified as attribute metadata.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {root}")

    image_dirs: list[tuple[str, int]] = []
    text_dirs: list[tuple[str, int]] = []
    query_files: list[tuple[str, dict[str, Any]]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative = current.relative_to(root)
        depth = len(relative.parts)
        if depth >= max_depth:
            dirnames[:] = []

        extension_counts: dict[str, int] = {}
        for name in filenames:
            extension = Path(name).suffix.lower()
            extension_counts[extension] = extension_counts.get(extension, 0) + 1

        image_count = sum(extension_counts.get(ext, 0) for ext in IMAGE_EXTENSIONS)
        text_count = extension_counts.get(".txt", 0)
        if image_count:
            image_dirs.append((str(current), image_count))
        if text_count:
            text_dirs.append((str(current), text_count))

        for name in filenames:
            if not name.lower().endswith(".json"):
                continue
            path = current / name
            info = _peek_json_records(path)
            if info and any(
                "caption" in row or "image_path" in row for row in info["sample"]
            ):
                query_files.append((str(path), info))

    print(f"=== Dataset layout report: {root} ===\n")
    print(f"Crops (image directories): {len(image_dirs)}")
    for path, count in sorted(image_dirs, key=lambda item: -item[1])[:top_n]:
        print(f"  {Path(path).relative_to(root)}: {count} images")

    print(f"\nQueries (pairs/query JSON files): {len(query_files)}")
    for path, info in query_files:
        query_types = ", ".join(
            f"{name}={count}" for name, count in sorted(info["query_types"].items())
        )
        print(
            f"  {Path(path).relative_to(root)}: {info['count']} rows "
            f"({query_types or 'no query_type field'})"
        )

    print(f"\nAttribute metadata (caption/.txt directories): {len(text_dirs)}")
    for path, count in sorted(text_dirs, key=lambda item: -item[1])[:top_n]:
        print(f"  {Path(path).relative_to(root)}: {count} .txt files")

    return {
        "image_dirs": image_dirs,
        "query_files": query_files,
        "text_dirs": text_dirs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the dataset report from the skill workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args(argv)
    report_dataset_layout(args.dataset_root, args.max_depth, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
