#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the single results.json from a completed Cosmos Reason evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_common import absolute_path, find_results_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Print the one completed evaluation results path."""
    args = parse_args()
    print(find_results_json(absolute_path(args.evaluate_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
