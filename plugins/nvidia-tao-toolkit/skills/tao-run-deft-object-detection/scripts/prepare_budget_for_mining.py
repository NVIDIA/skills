#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute ``desired_unique_count`` for the miner.

The reference pipeline fixes the mining budget across every iteration: it takes
the weak-image count from the **first** iteration and multiplies it by a
constant. Later iterations reuse that same number even as their own weak-image
count shrinks, so the amount of data added per iteration stays roughly constant
instead of decaying as the model improves.

Pass ``--weak-parquet`` pointing at iteration 1's weak-images parquet on every
iteration to reproduce that behavior; pass the current iteration's parquet
instead to let the budget track the shrinking gap set.

``--pool-size`` and ``--remaining-iterations`` turn the budget into a feasibility
check. Note the limit: ``--weak-parquet`` is required and the only weak parquet comes
from iteration 1's ``gap_analysis``, so the earliest this can run is after the baseline
has been scored. It bounds the *remaining* iterations, not the first one. A run whose
pool cannot support its ``max_iterations`` still pays for a full baseline before
hearing so.

The pool is finite and iterations exclude what earlier ones already mined,
so a run needs roughly ``budget x remaining_iterations`` unmined images left to
finish. When it does not have them, the shortfall is knowable here — before the
train, inference and KPI stages of this iteration run — rather than at the next
iteration's ``mine``, an hour or more later. The miner raises on an exhausted pool
rather than returning an empty result, so the run stops either way; the only
question is how much GPU time is spent first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-parquet", required=True,
                        help="Weak-images parquet whose row count seeds the budget.")
    parser.add_argument("--multiplier", type=int, required=True,
                        help="Budget multiplier applied to the weak-image count.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Floor for the resulting budget.")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Optional ceiling, e.g. the source-pool size.")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Source-pool image count. With --remaining-iterations, checks "
                             "the run can actually supply the iterations still configured.")
    parser.add_argument("--already-mined", type=int, default=0,
                        help="Rows in the cumulative exclude parquet: pool images earlier "
                             "iterations already took and this one cannot.")
    parser.add_argument("--remaining-iterations", type=int, default=None,
                        help="Iterations still to run, including this one.")
    parser.add_argument("--fail-on-shortfall", action="store_true",
                        help="Exit 2 instead of warning when the pool cannot supply the "
                             "remaining iterations.")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    try:
        if args.multiplier < 1:
            raise ValueError("--multiplier must be at least 1.")

        weak_parquet = Path(args.weak_parquet).expanduser().resolve()
        if not weak_parquet.is_file():
            raise FileNotFoundError(f"--weak-parquet does not exist: {weak_parquet}")

        weak_count = int(len(pd.read_parquet(weak_parquet)))
        if weak_count == 0:
            print("ERROR: weak-images parquet has zero rows — nothing to mine for.", file=sys.stderr)
            return 1

        budget = weak_count * args.multiplier
        clamped_reason = None
        if budget < args.min_count:
            clamped_reason = f"raised to --min-count {args.min_count}"
            budget = args.min_count
        if args.max_count is not None and budget > args.max_count:
            clamped_reason = f"clamped to --max-count {args.max_count}"
            budget = args.max_count

        # Feasibility. Reported whenever the pool size is known, because "the run
        # cannot reach the iteration count it was configured for" is a fact about
        # the run, not an error in this stage.
        shortfall = None
        if args.pool_size is not None and args.remaining_iterations is not None:
            available = max(0, args.pool_size - args.already_mined)
            required = budget * args.remaining_iterations
            shortfall = {
                "pool_size": args.pool_size,
                "already_mined": args.already_mined,
                "available": available,
                "required": required,
                "remaining_iterations": args.remaining_iterations,
                "feasible_iterations": available // budget if budget else 0,
                "sufficient": available >= required,
            }

        report = {
            "weak_parquet": str(weak_parquet),
            "weak_count": weak_count,
            "multiplier": args.multiplier,
            "desired_unique_count": budget,
            "clamped": clamped_reason,
            "pool_feasibility": shortfall,
        }
        if args.report_json:
            report_path = Path(args.report_json).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

        detail = f" ({clamped_reason})" if clamped_reason else ""
        print(f"weak_count={weak_count} x multiplier={args.multiplier} -> {budget}{detail}",
              file=sys.stderr)

        if shortfall and not shortfall["sufficient"]:
            feasible = shortfall["feasible_iterations"]
            print(
                f"POOL SHORTFALL: {shortfall['available']} unmined pool images remain "
                f"({args.pool_size} pool - {args.already_mined} already mined), but "
                f"{shortfall['remaining_iterations']} more iterations at a budget of {budget} "
                f"need {shortfall['required']}.\n"
                f"  This pool can supply {feasible} more full iteration(s). The run will stop "
                f"early when the pool is spent;\n"
                f"  the miner raises on an exhausted pool rather than returning nothing. "
                f"Either lower max_iterations to\n"
                f"  {feasible}, enlarge the pool, or accept the early stop and record it with "
                f"`commit_stage.py --stage loop_stop --pool-remaining 0`.",
                file=sys.stderr,
            )
            if args.fail_on_shortfall:
                return 2

        # stdout carries only the number, so callers can capture it directly.
        print(budget)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
