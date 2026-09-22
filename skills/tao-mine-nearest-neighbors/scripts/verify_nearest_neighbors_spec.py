#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an existing TAO Data Services TMM nearest-neighbor spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prepare_nearest_neighbors_spec import load_yaml, validate_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to nearest-neighbors YAML.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        output_parquet = Path(str(config["output_parquet"]))
        print(f"OK: nearest-neighbors spec is valid: {spec}")
        print(f"Output parquet: {output_parquet}")
        print(f"Expected summary: {output_parquet.with_name('mining_summary.txt')}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
