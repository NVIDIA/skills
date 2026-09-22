#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in ("commit_stage", "record_metric_result", "render_report"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import commit_stage  # noqa: E402


FULL_RCCA_REPORT = """# Cosmos3 Proxy RCCA Report: baseline

## 1. Verdict
The Proxy KPI is not reachable without another iteration.

## 2. False-Accept Breakdown
Bridge: 2 (100%).

## 3. False-Reject Breakdown
None.

## 4. Top-K Worst Samples
sample-1: bridge was missed with high confidence.

## 5. Per-Defect Analysis
Bridge is the only observed under-detection gap.

## 6. Recommended Actions
Mine bridge examples and allocate two bridge SDG samples.
"""


class Cosmos3RccaReportContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = pathlib.Path(self.temporary.name) / "results"
        self.proxy_dir = self.results / "baseline/proxy_rcca"
        self.proxy_dir.mkdir(parents=True)
        state = {
            "version": 5,
            "status": "in_progress",
            "results_dir": str(self.results),
            "max_iterations": 1,
            "current_iteration": 0,
            "iterations": {"baseline": {"status": "in_progress"}},
            "events": [],
        }
        self.state_path = self.results / "deft_state.json"
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        (self.proxy_dir / "gaps_summary.json").write_text(
            '{"accuracy": 0.5}\n', encoding="utf-8"
        )
        (self.proxy_dir / "false_accepts.json").write_text(
            '[{"id": "sample-1", "defect_type": "bridge"}]\n',
            encoding="utf-8",
        )
        (self.proxy_dir / "false_rejects.json").write_text(
            "[]\n", encoding="utf-8"
        )

    def _commit(self, *extra: str) -> int:
        with mock.patch.object(
            commit_stage,
            "render_html_report",
            return_value=self.results / "DEFT_Loop_Report.html",
        ):
            return commit_stage.main(
                [
                    "--results-dir",
                    str(self.results),
                    "--iter-label",
                    "baseline",
                    "--stage",
                    "proxy_rcca",
                    "--proxy-gaps-summary",
                    str(self.proxy_dir / "gaps_summary.json"),
                    "--false-accepts",
                    str(self.proxy_dir / "false_accepts.json"),
                    "--false-rejects",
                    str(self.proxy_dir / "false_rejects.json"),
                    "--duration-sec",
                    "1",
                    "--summary",
                    "Proxy RCCA complete",
                    *extra,
                ]
            )

    def test_proxy_rcca_commit_requires_report(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = self._commit()

        self.assertEqual(rc, 2)
        self.assertIn("--rcca-report is required", stderr.getvalue())
        state = json.loads(self.state_path.read_text())
        self.assertNotIn("rcca_report", state["iterations"]["baseline"])

    def test_proxy_rcca_commit_rejects_missing_sections(self) -> None:
        report = self.proxy_dir / "RCCA_Report.md"
        report.write_text(
            "# Incomplete\n\n## 1. Verdict\nNeeds another iteration.\n",
            encoding="utf-8",
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = self._commit("--rcca-report", str(report))

        self.assertEqual(rc, 2)
        message = stderr.getvalue()
        self.assertIn("missing required section heading(s)", message)
        self.assertIn("False-Accept Breakdown", message)
        self.assertIn("references/RCCA_REPORT_TEMPLATE.md", message)

    def test_proxy_rcca_commit_records_valid_report(self) -> None:
        manifest = json.loads(
            commit_stage.RCCA_ARTIFACT_MANIFEST.read_text(encoding="utf-8")
        )
        classes = manifest["artifact_classes"]
        produced = classes["container_or_script_produced_required"]
        self.assertEqual(
            [entry["path"] for entry in produced],
            [
                "gaps_summary.json",
                "false_accepts.json",
                "false_rejects.json",
            ],
        )
        self.assertEqual(
            [entry["state_field"] for entry in produced],
            [
                "proxy_gaps_summary",
                "false_accepts_json",
                "false_rejects_json",
            ],
        )
        report_entry = classes["agent_produced_required"][0]
        self.assertEqual(report_entry["path"], "RCCA_Report.md")
        self.assertEqual(report_entry["state_field"], "rcca_report")
        self.assertEqual(
            report_entry["validation"]["required_headings"],
            [
                "Verdict",
                "False-Accept Breakdown",
                "False-Reject Breakdown",
                "Top-K Worst Samples",
                "Per-Defect Analysis",
                "Recommended Actions",
            ],
        )
        report = self.proxy_dir / "RCCA_Report.md"
        report.write_text(FULL_RCCA_REPORT, encoding="utf-8")

        rc = self._commit("--rcca-report", str(report))

        self.assertEqual(rc, 0)
        state = json.loads(self.state_path.read_text())
        phase = state["iterations"]["baseline"]
        self.assertEqual(phase["rcca_report"], str(report.resolve()))
        self.assertEqual(phase["stage_completed"], "proxy_rcca")
        self.assertEqual(phase["status"], "complete")


if __name__ == "__main__":
    unittest.main()
