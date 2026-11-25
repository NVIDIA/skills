#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nb_stats: the numbers behind every verdict: score ranges, paired differences, speed verdicts.

A score is a share of test items answered right. This module gives its 95% range (Wilson for
right-or-wrong items, a normal range for partial credit), the paired range of the difference
between two endpoints answering the same items, and the three-way accuracy verdict. It also
decides `better`, `worse`, `within_noise` or `same` for one speed metric across repeats.
Run it to print the defaults: python3 scripts/nb_stats.py --defaults.
Exit codes: 0 ok, 2 bad input. Never prompts. Stdlib only.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_common import NOISE_FLOOR_PCT, ContractParser, emit

Z95 = 1.959964
DEFAULT_MARGIN_POINTS = 1.0
# fewer paired items than this give no pass or fail: the normal range needs about 30 to hold
MIN_VERDICT_ITEMS = 30


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson range for a share of right answers. It stays sensible near 0 and 1."""
    if n <= 0:
        raise ValueError("no items")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def normal_range(values: list[float], z: float = Z95) -> tuple[float, float, float]:
    """The mean and its normal 95% range. One item has no spread, so the range is the mean."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def score_summary(values: list[float]) -> dict[str, float]:
    """One score in percent with its 95% range. Rewards are 0 to 1, one value per item."""
    if not values:
        raise ValueError("no items")
    if all(v in (0.0, 1.0) for v in values):
        low, high = wilson(sum(values), len(values))
        mean = sum(values) / len(values)
    else:
        mean, low, high = normal_range(values)
    return {
        "items": float(len(values)),
        "score": round(100 * mean, 3),
        "low": round(100 * low, 3),
        "high": round(100 * high, 3),
    }


def paired_difference(base: list[float], cand: list[float]) -> dict[str, float]:
    """The candidate minus the baseline, in points, with its 95% range, on the same items."""
    if len(base) != len(cand) or not base:
        raise ValueError("paired runs need the same items")
    diffs = [c - b for b, c in zip(base, cand, strict=True)]
    mean, low, high = normal_range(diffs)
    if low == high:
        # Every item moved by the same amount, so the spread is zero and proves nothing. The rule
        # of three bounds the share of items that could still differ at 95%: 3 / items.
        low, high = max(-1.0, mean - 3 / len(diffs)), min(1.0, mean + 3 / len(diffs))
    return {
        "items": float(len(diffs)),
        "difference": round(100 * mean, 3),
        "low": round(100 * low, 3),
        "high": round(100 * high, 3),
    }


def accuracy_verdict(
    low: float, high: float, margin: float = DEFAULT_MARGIN_POINTS, *, items: float
) -> str:
    """`pass` only when the whole range is above minus the margin. A straddle is never a pass.

    Fewer than MIN_VERDICT_ITEMS paired items is always `inconclusive`.
    """
    if items < MIN_VERDICT_ITEMS:
        return "inconclusive"
    if low > -margin:
        return "pass"
    if high < -margin:
        return "fail"
    return "inconclusive"


def speed_verdict(base: list[float], cand: list[float], lower_is_better: bool) -> dict[str, Any]:
    """Compare one speed metric over the repeats of a baseline and a candidate.

    `same` when the means differ by under the noise floor, `within_noise` when the repeat ranges
    overlap, otherwise `better` or `worse` by the direction that is good for this metric.
    """
    if not base or not cand:
        raise ValueError("both sides need at least one repeat")
    base_mean, cand_mean = statistics.fmean(base), statistics.fmean(cand)
    change = 0.0 if base_mean == 0 else 100 * (cand_mean - base_mean) / base_mean
    if abs(change) < NOISE_FLOOR_PCT:
        verdict = "same"
    elif max(min(base), min(cand)) <= min(max(base), max(cand)):
        verdict = "within_noise"
    else:
        improved = (cand_mean < base_mean) == lower_is_better
        verdict = "better" if improved else "worse"
    return {
        "baseline": round(base_mean, 3),
        "candidate": round(cand_mean, 3),
        "change_pct": round(change, 2),
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Statistics for nim-benchmark. Run it to print the defaults.",
        epilog="Example: python3 scripts/nb_stats.py --defaults",
    )
    parser.add_argument(
        "--defaults", action="store_true", help="print the margin, item minimum and noise floor"
    )
    parser.parse_args(argv)
    emit(
        True,
        "defaults",
        "done",
        {
            "margin_points": DEFAULT_MARGIN_POINTS,
            "min_verdict_items": MIN_VERDICT_ITEMS,
            "noise_floor_pct": NOISE_FLOOR_PCT,
        },
    )


if __name__ == "__main__":
    main()
