#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record mined source filepaths so later DEFT iterations can filter them out."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_common import absolute_path, atomic_write_parquet


def mined_filepaths(mined_neighbors_parquet: Path) -> set[str]:
    """Read source filepaths from one mined-neighbor parquet."""
    if not mined_neighbors_parquet.is_file():
        print(f"mined-neighbor parquet not found, skipping: {mined_neighbors_parquet}")
        return set()
    neighbors = pd.read_parquet(mined_neighbors_parquet)
    column = "source_filepath" if "source_filepath" in neighbors.columns else "filepath"
    if column not in neighbors.columns:
        raise ValueError(f"{mined_neighbors_parquet}: missing required column 'source_filepath' or 'filepath'")
    return set(neighbors[column].astype(str).tolist())


def record_mined_paths(mined_neighbors_parquets: list[Path], mined_log_parquet: Path) -> int:
    """Union this iteration's mined source paths into the cumulative mined-path log."""
    paths: set[str] = set()
    for parquet in mined_neighbors_parquets:
        paths.update(mined_filepaths(parquet))
    if mined_log_parquet.is_file():
        existing = pd.read_parquet(mined_log_parquet)
        if "filepath" not in existing.columns:
            raise ValueError(f"{mined_log_parquet}: missing required column 'filepath'")
        paths.update(existing["filepath"].astype(str).tolist())
    output = pd.DataFrame({"filepath": sorted(paths)})
    atomic_write_parquet(output, mined_log_parquet)
    return len(output)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-log-parquet", required=True, type=Path)
    parser.add_argument("--mined-neighbors-parquet", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    mined_log_parquet = absolute_path(args.mined_log_parquet)
    mined_neighbors_parquets = [absolute_path(path) for path in args.mined_neighbors_parquet]
    count = record_mined_paths(mined_neighbors_parquets, mined_log_parquet)
    print(f"Wrote {count} unique mined source paths -> {mined_log_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
