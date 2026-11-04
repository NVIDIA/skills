#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the saved iteration from an RSL-RL checkpoint using PyTorch's safe loader.

Usage: checkpoint.py /absolute/path/to/model_N.pt
Arguments: one checkpoint file path.
Output: the nonnegative saved iteration on stdout; diagnostics on stderr.
Exit codes: 0 for success, 2 for an unreadable or invalid checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        import torch

        data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        step = data["iter"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint iter must be a nonnegative integer")
        print(step)
        return 0
    except Exception as error:
        print(f"cannot read checkpoint iteration: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
