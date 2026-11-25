# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nb_stats.py: score ranges, paired differences and verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_stats  # noqa: E402


def test_wilson_matches_a_known_value() -> None:
    low, high = nb_stats.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_stays_inside_zero_and_one_at_the_edges() -> None:
    low, high = nb_stats.wilson(100, 100)
    assert 0.96 < low < 1.0 and high == 1.0
    assert nb_stats.wilson(0, 50)[0] == 0.0


def test_score_summary_for_right_or_wrong_items_uses_wilson() -> None:
    summary = nb_stats.score_summary([1.0] * 90 + [0.0] * 10)
    assert summary["score"] == 90.0
    assert summary["low"] == pytest.approx(82.6, abs=0.2)


def test_score_summary_for_partial_credit_uses_a_normal_range() -> None:
    summary = nb_stats.score_summary([0.5, 0.75, 1.0, 0.25])
    assert summary["score"] == 62.5
    assert summary["low"] < 62.5 < summary["high"]


def test_gsm8k_range_is_about_plus_minus_1_6_points() -> None:
    # 1,319 items at 90% accuracy: the figure the design doc quotes
    right = round(0.9 * 1319)
    summary = nb_stats.score_summary([1.0] * right + [0.0] * (1319 - right))
    assert (summary["high"] - summary["low"]) / 2 == pytest.approx(1.6, abs=0.1)


def test_paired_range_is_tighter_than_the_unpaired_one() -> None:
    # 1,319 items; the endpoints disagree on 5% of them, all in the candidate's favor or against
    base = [1.0] * 1319
    cand = [1.0] * 1319
    for i in range(33):
        cand[i] = 0.0  # the candidate loses 33 items
    for i in range(33, 66):
        base[i] = 0.0  # and wins 33 others: the net difference is zero
    paired = nb_stats.paired_difference(base, cand)
    assert paired["difference"] == 0.0
    assert (paired["high"] - paired["low"]) / 2 == pytest.approx(1.2, abs=0.15)


def test_paired_difference_needs_the_same_items() -> None:
    with pytest.raises(ValueError, match="same items"):
        nb_stats.paired_difference([1.0], [1.0, 0.0])


def test_accuracy_verdict_has_three_outcomes() -> None:
    assert nb_stats.accuracy_verdict(-0.4, 0.8, items=1000) == "pass"
    assert nb_stats.accuracy_verdict(-3.0, -1.5, items=1000) == "fail"
    assert nb_stats.accuracy_verdict(-2.0, 0.5, items=1000) == "inconclusive"


def test_a_range_that_straddles_the_margin_is_never_a_pass() -> None:
    assert nb_stats.accuracy_verdict(-1.0, 2.0, items=1000) == "inconclusive"
    assert nb_stats.accuracy_verdict(-0.99, 2.0, items=1000) == "pass"


def test_speed_verdict_same_within_noise_better_and_worse() -> None:
    same = nb_stats.speed_verdict([100, 101, 99], [100.5, 100, 100], True)
    assert same["verdict"] == "same"
    overlap = nb_stats.speed_verdict([100, 110, 120], [105, 112, 125], True)
    assert overlap["verdict"] == "within_noise"
    better = nb_stats.speed_verdict([100, 101, 102], [80, 81, 82], True)
    assert better["verdict"] == "better" and better["change_pct"] == -19.8
    worse = nb_stats.speed_verdict([100, 101, 102], [130, 131, 132], True)
    assert worse["verdict"] == "worse"


def test_speed_verdict_knows_that_more_throughput_is_better() -> None:
    assert nb_stats.speed_verdict([500, 505, 510], [600, 605, 610], False)["verdict"] == "better"
    assert nb_stats.speed_verdict([500, 505, 510], [400, 405, 410], False)["verdict"] == "worse"


def test_speed_verdict_needs_repeats_on_both_sides() -> None:
    with pytest.raises(ValueError, match="at least one repeat"):
        nb_stats.speed_verdict([], [1.0], True)


def test_script_prints_the_defaults() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_stats.py"), "--defaults"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert json.loads(proc.stdout)["facts"] == {
        "margin_points": 1.0,
        "min_verdict_items": 30,
        "noise_floor_pct": 2.0,
    }


def test_too_few_items_give_no_pass_or_fail() -> None:
    few = nb_stats.MIN_VERDICT_ITEMS - 1
    assert nb_stats.accuracy_verdict(-0.4, 0.8, items=few) == "inconclusive"
    assert nb_stats.accuracy_verdict(-30.0, -20.0, items=few) == "inconclusive"
    assert nb_stats.accuracy_verdict(-0.4, 0.8, items=nb_stats.MIN_VERDICT_ITEMS) == "pass"


def test_all_equal_differences_do_not_give_a_zero_width_range() -> None:
    one = nb_stats.paired_difference([1.0], [1.0])
    assert one["low"] == -100.0 and one["high"] == 100.0  # one item proves nothing
    few = nb_stats.paired_difference([1.0] * 30, [1.0] * 30)
    assert few["low"] == -10.0 and few["high"] == 10.0  # the rule of three: 3 / 30 items
    assert nb_stats.accuracy_verdict(few["low"], few["high"], items=30) == "inconclusive"
    many = nb_stats.paired_difference([1.0] * 1000, [1.0] * 1000)
    assert many["low"] == -0.3 and many["high"] == 0.3
    assert nb_stats.accuracy_verdict(many["low"], many["high"], items=1000) == "pass"
