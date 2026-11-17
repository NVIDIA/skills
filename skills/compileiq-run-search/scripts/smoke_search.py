#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compileiq-run-search smoke test.

Runs a 2-generation search on a trivial objective and asserts the result has
the expected shape. Useful as a quick "Search() can drive my Python" check
before plugging in a real kernel.

Exit code 0 on success, non-zero on failure.

Usage:
    python smoke_search.py
"""
from __future__ import annotations

import sys

from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, ProblemType
from compileiq.worker import MultiProcessWorker
import compileiq.search_spaces.base as ss


def objective(config: dict) -> float:
    """x**2 + y; minimized by small x and small y."""
    return float(config["x"]) ** 2 + float(config["y"])


def main() -> int:
    search_space = {
        "x": ss.range(start=1.0, end=10.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
    }
    search_config = SearchConfiguration(
        problem_type=ProblemType.MIN,
        generations=2,
        pool_size=8,
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=search_config,
        worker_type=MultiProcessWorker,
    )
    results = tuner.start(num_workers=2)
    best = results.get_best_result()

    assert isinstance(best, dict), f"expected dict, got {type(best)}"
    assert "score_1" in best, f"missing score_1 in {best}"
    assert "params" in best, f"missing params in {best}"
    assert isinstance(best["score_1"], (int, float)), best["score_1"]
    assert best["score_1"] >= 0.0, best["score_1"]

    print(f"PASS: best score_1={best['score_1']:.4f}  params={best['params']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
