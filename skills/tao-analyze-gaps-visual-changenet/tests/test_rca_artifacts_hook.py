# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = SKILL_ROOT / "hooks" / "rca-artifacts-check.sh"


class RcaArtifactsHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.report_dir = pathlib.Path(self.temporary.name)

        pq.write_table(
            pa.table(
                {
                    "filepath": ["sample.jpg"],
                    "label": ["PASS"],
                    "siamese_score": [0.25],
                    "weakness": [-0.25],
                }
            ),
            self.report_dir / "kpi_gaps.parquet",
        )
        (self.report_dir / "threshold.txt").write_text("0.5\n", encoding="utf-8")
        (self.report_dir / "weak_samples_breakdown.txt").write_text(
            "PASS: 1 (100.0%) — 0 misclassified, 1 marginal\n",
            encoding="utf-8",
        )
        (self.report_dir / "RCA_Report.md").write_text(
            "# VCN Gap Analysis Report\n",
            encoding="utf-8",
        )
        images = self.report_dir / "rca_images"
        images.mkdir()
        for index in range(10):
            (images / f"sample_{index:02d}.jpg").write_bytes(b"fixture image")

        self.assertEqual(
            {entry.name for entry in self.report_dir.iterdir()},
            {
                "kpi_gaps.parquet",
                "threshold.txt",
                "weak_samples_breakdown.txt",
                "RCA_Report.md",
                "rca_images",
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_hook(self) -> str:
        environment = os.environ.copy()
        environment["RCA_HOOKS"] = "1"
        payload = {
            "tool_name": "Write",
            "tool_input": {"file_path": str(self.report_dir / "RCA_Report.md")},
        }
        result = subprocess.run(
            ["bash", str(HOOK)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_complete_container_and_agent_contract_has_no_warnings(self) -> None:
        self.assertEqual(self.run_hook(), "")

    def test_missing_artifact_warning_does_not_require_metrics_json(self) -> None:
        (self.report_dir / "threshold.txt").unlink()

        output = self.run_hook()

        self.assertIn("MISSING ARTIFACT: threshold.txt", output)
        self.assertNotIn("metrics.json", output)


if __name__ == "__main__":
    unittest.main()
