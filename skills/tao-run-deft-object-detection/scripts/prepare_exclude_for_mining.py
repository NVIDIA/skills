#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulate the mined-image exclude set across iterations.

Merges this iteration's ``final_unique_files.parquet`` with the previous
cumulative parquet and de-duplicates, producing the ``exclude_path`` the next
iteration's miner uses to avoid re-mining images it already has.

``--parquet-b`` is optional so iteration 1 works: with no previous cumulative,
the output is just this iteration's mined set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

IMAGE_COLUMNS = ("filepath", "source_filepath")


def resolve_image_column(df: pd.DataFrame, origin: str) -> str:
    for column in IMAGE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"{origin} has none of {IMAGE_COLUMNS}; columns are {list(df.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-a", required=True, help="This iteration's mined parquet.")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration number. Required to accept a missing --parquet-b, and "
                             "only iteration 1 may: every later iteration has a previous "
                             "cumulative exclude set, and continuing without it re-mines "
                             "images the model has already trained on.")
    parser.add_argument("--parquet-b", default=None,
                        help="Previous cumulative parquet. Omit on iteration 1.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        frames = []

        path_a = Path(args.parquet_a).expanduser().resolve()
        if not path_a.is_file():
            raise FileNotFoundError(f"--parquet-a does not exist: {path_a}")
        df_a = pd.read_parquet(path_a)
        col_a = resolve_image_column(df_a, str(path_a))
        frames.append(df_a[[col_a]].rename(columns={col_a: "filepath"}))

        if args.parquet_b:
            path_b = Path(args.parquet_b).expanduser()
            if path_b.is_file():
                df_b = pd.read_parquet(path_b)
                col_b = resolve_image_column(df_b, str(path_b))
                frames.append(df_b[[col_b]].rename(columns={col_b: "filepath"}))
            elif args.iteration == 1:
                print(f"NOTE: no previous cumulative at iteration 1: {path_b}")
            else:
                # Silently proceeding here drops every earlier exclusion, so the next
                # mine re-selects images already trained on and the iterations stop
                # being disjoint — with nothing in the output to show it happened.
                raise FileNotFoundError(
                    f"--parquet-b does not exist: {path_b}. Only iteration 1 may run without "
                    f"a previous cumulative exclude set; pass --iteration 1 if that is the "
                    f"case, otherwise fix the path. Continuing would re-mine images earlier "
                    f"iterations already used."
                )

        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["filepath"])
        merged = merged.sort_values("filepath").reset_index(drop=True)

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output, index=False)

        total_in = sum(len(f) for f in frames)
        print(f"Wrote cumulative exclude set: {output}")
        print(f"rows_in={total_in} unique_out={len(merged)} duplicates_removed={total_in - len(merged)}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
