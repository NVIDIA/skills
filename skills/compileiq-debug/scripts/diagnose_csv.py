#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heuristic diagnostics for a CompileIQ dump_results CSV.

CLI:
    python diagnose_csv.py results.csv
    python diagnose_csv.py --self-test

Prints a generation-summary table and flags one or more of:
- INCREASING_INVALID_RATE  (mutation arm spreading bad configs)
- STALLED_CONVERGENCE      (no improvement in best score over recent gens)
- HIGH_VARIANCE            (validation-time noise; check GPU clocks)
- NO_SCORE_COLUMN          (generation present but no score_1 or score column)
- HEALTHY                  (none of the above)

Exit code = number of flagged pathologies (0 = healthy).
"""
from __future__ import annotations

import argparse
import io
import sys
from typing import NamedTuple

import pandas as pd


INVALID_SENTINEL = 1e10


class Diagnosis(NamedTuple):
    flags: list[str]
    summary: pd.DataFrame
    notes: list[str]


def diagnose(df: pd.DataFrame) -> Diagnosis:
    notes: list[str] = []

    if "generation" not in df.columns:
        return Diagnosis(["NO_GENERATION_COLUMN"], pd.DataFrame(),
                         ["CSV missing 'generation' column; not a CompileIQ dump_results CSV?"])

    if "score_1" not in df.columns and "score" in df.columns:
        df = df.rename(columns={"score": "score_1"})
    elif "score_1" not in df.columns:
        return Diagnosis(["NO_SCORE_COLUMN"], pd.DataFrame(),
                         ["CSV missing 'score_1' or 'score' column."])

    df = df.copy()
    df["score_numeric"] = pd.to_numeric(df["score_1"], errors="coerce")

    summary = df.groupby("generation").agg(
        n=("score_numeric", "size"),
        invalid=("score_numeric",
                 lambda s: int(s.isna().sum() + (s > INVALID_SENTINEL).sum())),
        best=("score_numeric", "min"),
        mean=("score_numeric", "mean"),
        std=("score_numeric", "std"),
    ).reset_index()
    summary["invalid_pct"] = (summary["invalid"] / summary["n"] * 100).round(1)
    summary["cv_pct"] = (summary["std"] / summary["mean"] * 100).round(2)

    flags: list[str] = []

    if len(summary) >= 3:
        early = summary["invalid_pct"].head(max(1, len(summary) // 3)).mean()
        late = summary["invalid_pct"].tail(max(1, len(summary) // 3)).mean()
        if late > early + 10:
            flags.append("INCREASING_INVALID_RATE")
            notes.append(
                f"invalid_pct rose from ~{early:.1f}% (early gens) "
                f"to ~{late:.1f}% (late gens)"
            )

    if len(summary) >= 4:
        recent_best = summary["best"].tail(max(2, len(summary) // 4))
        if recent_best.notna().sum() >= 2:
            first = recent_best.iloc[0]
            last = recent_best.iloc[-1]
            improvement = (first - last) / abs(first) if first else 0
            if abs(improvement) < 0.01:
                flags.append("STALLED_CONVERGENCE")
                notes.append(
                    f"best score barely moved in last {len(recent_best)} gens "
                    f"({recent_best.iloc[0]:.4g} -> {recent_best.iloc[-1]:.4g})"
                )

    if summary["cv_pct"].dropna().median() > 15:
        flags.append("HIGH_VARIANCE")
        notes.append(
            f"median per-generation CV% is {summary['cv_pct'].median():.1f}% "
            "(>15% = noisy)"
        )

    if not flags:
        flags = ["HEALTHY"]
        notes.append("no pathology heuristics triggered")

    return Diagnosis(flags, summary, notes)


def render(diag: Diagnosis) -> str:
    buf = io.StringIO()
    buf.write("Flags: " + ", ".join(diag.flags) + "\n")
    buf.write("\nGeneration summary:\n")
    if not diag.summary.empty:
        buf.write(diag.summary.to_string(index=False))
        buf.write("\n")
    for n in diag.notes:
        buf.write(f"\nNote: {n}\n")
    return buf.getvalue()


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Heuristic diagnostics for a CompileIQ dump_results CSV."
    )
    ap.add_argument("csv", nargs="?", help="Path to results.csv")
    ap.add_argument("--self-test", action="store_true", help="Run synthetic checks and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.csv:
        ap.error("csv argument required (unless --self-test)")

    df = pd.read_csv(args.csv)
    diag = diagnose(df)
    print(render(diag))
    flagged = [f for f in diag.flags if f != "HEALTHY"]
    return len(flagged)


def _self_test() -> int:
    import numpy as np
    rng = np.random.default_rng(0)

    # Synthetic 1: clean convergence — should be HEALTHY
    rows = []
    best = 1.0
    for gen in range(5):
        for _ in range(10):
            rows.append({"generation": gen, "score_1": best + rng.uniform(0, 0.1)})
        best *= 0.85
    df_healthy = pd.DataFrame(rows)
    d = diagnose(df_healthy)
    assert "HEALTHY" in d.flags, f"expected HEALTHY, got {d.flags}"
    print(f"healthy: {d.flags}  OK")

    # Synthetic 2: rising invalid rate — should flag INCREASING_INVALID_RATE
    rows = []
    for gen in range(6):
        for i in range(10):
            score = float("nan") if i < gen * 2 else 1.0 + rng.uniform(0, 0.1)
            rows.append({"generation": gen, "score_1": score})
    df_invalid = pd.DataFrame(rows)
    d = diagnose(df_invalid)
    assert "INCREASING_INVALID_RATE" in d.flags, f"expected INCREASING_INVALID_RATE, got {d.flags}"
    print(f"rising_invalid: {d.flags}  OK")

    # Synthetic 3: stalled best score — should flag STALLED_CONVERGENCE
    rows = []
    for gen in range(8):
        for _ in range(10):
            rows.append({"generation": gen, "score_1": 1.0 + rng.uniform(0, 0.01)})
    df_stalled = pd.DataFrame(rows)
    d = diagnose(df_stalled)
    assert "STALLED_CONVERGENCE" in d.flags, f"expected STALLED_CONVERGENCE, got {d.flags}"
    print(f"stalled: {d.flags}  OK")

    # Synthetic 4: high variance — should flag HIGH_VARIANCE
    rows = []
    for gen in range(4):
        for _ in range(20):
            rows.append({"generation": gen, "score_1": rng.normal(loc=1.0, scale=0.3)})
    df_noisy = pd.DataFrame(rows)
    d = diagnose(df_noisy)
    assert "HIGH_VARIANCE" in d.flags, f"expected HIGH_VARIANCE, got {d.flags}"
    print(f"noisy: {d.flags}  OK")

    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
