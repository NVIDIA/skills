#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render final Cosmos3 evidence and atomically close a DEFT run."""

from __future__ import annotations

import argparse
import pathlib
import sys

import commit_stage
from render_report import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument(
        "--stop-reason", required=True, choices=("metric_met", "max_iterations")
    )
    parser.add_argument("--duration-sec", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        report = render(args.results_dir.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001 - normalize deterministic finalization errors
        print(f"finalize_run: {exc}", file=sys.stderr)
        return 2
    return commit_stage.main(
        [
            "--results-dir", str(args.results_dir),
            "--iter-label", args.iter_label,
            "--stage", "loop_stop",
            "--summary", "deterministic finalization",
            "--duration-sec", str(args.duration_sec),
            "--stop-reason", args.stop_reason,
            "--final-report", str(report),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
