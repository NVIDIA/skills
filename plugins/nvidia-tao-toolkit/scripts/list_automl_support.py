#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print packaged TAO AutoML model support without scanning model folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# "python3 -I <script>" leaves the folder of the script off the import path, so
# add it for the sibling modules. A run as a script writes no __pycache__ here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.dont_write_bytecode = True

from list_tao_models import build_automl_support, format_automl_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        # Default to the bank this script ships in; $TAO_SKILL_BANK_PATH overrides it.
        default=Path(
            os.environ.get("TAO_SKILL_BANK_PATH") or Path(__file__).resolve().parents[1]
        ),
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--action",
        default="",
        help="Filter support to one action such as train, evaluate, inference, distill, prune, or quantize.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the support summary helper."""
    args = parse_args()
    support = build_automl_support(args.skill_bank, action_filter=args.action)
    if args.format == "json":
        print(json.dumps(support, indent=2, sort_keys=True))
    else:
        print(format_automl_text(support))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
