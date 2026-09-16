#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for truthful Cosmos3 metrics and split validation."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import analyze_gaps  # noqa: E402
import validate_split_contract  # noqa: E402


def _record(target: str, label: str) -> dict:
    return {
        "images": [target, "images/golden.png"],
        "conversations": [
            {"from": "human", "value": "Inspect this component."},
            {"from": "gpt", "value": label},
        ],
    }


def _write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _role_paths(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        role: _write_json(
            root / f"{role}.json",
            [_record(f"images/{role}.png", "OK")],
        )
        for role in ("proxy", "benchmark", "mining")
    }


class MetricTruthfulnessTests(unittest.TestCase):
    def test_unknown_predictions_count_against_accuracy(self) -> None:
        summary, *_ = analyze_gaps.analyze(
            [
                {"gt": "OK", "response": "OK"},
                {"gt": "NG", "response": "NG"},
                {"gt": "NG", "response": "Unable to classify"},
            ],
            kpi_metric="accuracy",
            evaluation_role="benchmark",
        )

        self.assertEqual(summary["parseable_samples"], 2)
        self.assertEqual(summary["unknown_samples"], 1)
        self.assertAlmostEqual(summary["metrics"]["accuracy"], 2 / 3)

    def test_unknown_count_is_reported_in_artifact_and_cli_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = _write_json(
                root / "results.json",
                [
                    {"gt": "OK", "response": "OK"},
                    {"gt": "NG", "response": "not sure"},
                ],
            )
            output_dir = root / "metrics"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = analyze_gaps.main(
                    [
                        "--results-json",
                        str(results),
                        "--output-dir",
                        str(output_dir),
                        "--evaluation-role",
                        "benchmark",
                        "--kpi-metric",
                        "accuracy",
                    ]
                )

            self.assertEqual(exit_code, 0)
            metric_result = json.loads(
                (output_dir / "metric_result.json").read_text()
            )
            self.assertEqual(metric_result["samples"], 2)
            self.assertEqual(metric_result["parseable_samples"], 1)
            self.assertEqual(metric_result["unknown_samples"], 1)
            self.assertEqual(metric_result["constraints"]["unknown_predictions"], 1)
            self.assertEqual(metric_result["value"], 0.5)
            self.assertIn("unknown_samples=1", stdout.getvalue())


class SyntheticSplitTruthfulnessTests(unittest.TestCase):
    def test_ok_labelled_synthetic_record_is_rejected_with_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            role_paths = _role_paths(root)
            synthetic = _write_json(
                root / "synthetic.json",
                [
                    _record("images/synthetic-ng.png", "NG"),
                    _record("images/synthetic-ok.png", "OK"),
                ],
            )
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = validate_split_contract.main(
                    [
                        "--workspace",
                        str(root),
                        "--proxy",
                        str(role_paths["proxy"]),
                        "--benchmark",
                        str(role_paths["benchmark"]),
                        "--mining",
                        str(role_paths["mining"]),
                        "--synthetic",
                        str(synthetic),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertRegex(
                stderr.getvalue(),
                r"synthetic\.json\[1\]: synthetic label must be exactly NG, got 'OK'",
            )

    def test_ng_only_synthetic_records_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            synthetic = _write_json(
                root / "synthetic.json",
                [
                    _record("images/synthetic-1.png", "NG"),
                    _record("images/synthetic-2.png", "NG"),
                ],
            )

            summary = validate_split_contract.validate(
                {**_role_paths(root), "synthetic": synthetic},
                media_root=root,
            )

            self.assertEqual(summary["records"]["synthetic"], 2)
            self.assertEqual(summary["roles"]["synthetic"], "anomalygen_sdg")


if __name__ == "__main__":
    unittest.main()
