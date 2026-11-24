# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import commit_stage
import prepare_card_rca


class CardRcaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.rd = self.workspace / "results/run_fixture"
        images = self.workspace / "kpi/images"
        images.mkdir(parents=True)
        rows = [("PASS", "p1", 0.9), ("PASS", "p2", 0.1), ("missing", "m1", 0.2), ("missing", "m2", 0.8)]
        for _, name, _ in rows:
            (images / f"{name}_SolderLight.jpg").write_bytes(b"fixture-image")
        inference = self.rd / "baseline/inference/best_val/inference/inference.csv"
        inference.parent.mkdir(parents=True)
        pd.DataFrame({"input_path": ["x"] * 4, "object_name": [r[1] for r in rows],
                      "label": [r[0] for r in rows], "siamese_score": [r[2] for r in rows]}
                     ).to_csv(inference, index=False)
        (self.rd / "deft_state.json").write_text(json.dumps({"iterations": {"baseline": {
            "status": "complete", "stage_completed": "evaluate", "inference_csv": str(inference)}}}))
        self.out = self.rd / "baseline/rca_results/1700000000"
        self.out.mkdir(parents=True)
        pd.DataFrame({"filepath": [str(images / f"{r[1]}_SolderLight.jpg") for r in rows],
                      "object_name": [r[1] for r in rows], "label": [r[0] for r in rows],
                      "siamese_score": [r[2] for r in rows],
                      "weakness": [0.4, -0.4, 0.3, -0.3]}).to_parquet(self.out / "kpi_gaps.parquet")
        (self.out / "threshold.txt").write_text("0.5\n")
        (self.out / "weak_samples_breakdown.txt").write_text("PASS: 2 total, 1 misclassified\n")

    def test_report_and_images_satisfy_the_rca_commit(self):
        summary = prepare_card_rca.prepare(self.rd, self.workspace, "baseline", self.out)

        self.assertFalse(summary["unreachable"])
        self.assertEqual(summary["target_defects"], ["missing"])
        self.assertEqual(len(list((self.out / "rca_images").iterdir())), 4)
        report = (self.out / "RCA_Report.md").read_text()
        self.assertIn("TP=1", report)
        self.assertIn("FN=1", report)
        self.assertIn(prepare_card_rca.PENDING, report)
        rc = commit_stage.main([
            "--results-dir", str(self.rd), "--iter-label", "baseline", "--stage", "rca",
            "--duration-sec", "1", "--summary", "fixture",
            "--rca-gaps", str(self.out / "kpi_gaps.parquet"),
            "--rca-report", str(self.out / "RCA_Report.md"), "--rca-threshold", "0.5",
            "--rca-target-defect", "missing",
        ])
        self.assertEqual(rc, 0)
        phase = json.loads((self.rd / "deft_state.json").read_text())["iterations"]["baseline"]
        self.assertEqual(phase["stage_completed"], "rca")
        self.assertEqual(phase["rca_target_defects"], ["missing"])

    def test_card_commit_command_passes_report_threshold_and_each_target(self):
        prepare_card_rca.prepare(self.rd, self.workspace, "baseline", self.out)
        (self.out / "rca_target_defects.txt").write_text("missing\nshift\n")
        skill = Path(__file__).resolve().parents[1]
        commands = re.findall(r"\x60{3}bash\n(.*?)\x60{3}", (skill / "cards/30-post-evaluate.md").read_text(), re.S)
        scripts = self.workspace / "fake-skill/scripts"
        scripts.mkdir(parents=True)
        (scripts / "commit_stage.py").write_text(
            f"import sys, json\nsys.path.insert(0, {str(skill / 'scripts')!r})\n"
            "import commit_stage\n"
            "print(json.dumps(vars(commit_stage._parser().parse_args()), default=str))\n"
        )
        result = subprocess.run(
            ["bash", "-c", commands[-1]], capture_output=True, text=True, timeout=20,
            env={"PATH": os.environ["PATH"], "DPY": sys.executable, "SKILL_ROOT": str(scripts.parent),
                 "RD": str(self.rd), "ITER": "baseline", "STAGE_T0": "0"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args["stage"], "rca")
        self.assertEqual(args["rca_report"], str(self.out / "RCA_Report.md"))
        self.assertEqual(args["rca_gaps"], str(self.out / "kpi_gaps.parquet"))
        self.assertEqual(args["rca_threshold"], 0.5)
        self.assertEqual(args["rca_target_defect"], ["missing", "shift"])

    def test_unreachable_kpi_writes_only_the_abridged_report(self):
        for name in ("kpi_gaps.parquet", "threshold.txt", "weak_samples_breakdown.txt"):
            (self.out / name).unlink()
        (self.out / "unreachable_kpi.txt").write_text("max recall 0.8 < 1.0\n")

        summary = prepare_card_rca.prepare(self.rd, self.workspace, "baseline", self.out)

        self.assertTrue(summary["unreachable"])
        self.assertIn("## KPI Unreachable", (self.out / "RCA_Report.md").read_text())
        self.assertFalse((self.out / "rca_images").exists())

    def test_rca_dir_outside_the_iteration_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_card_rca.prepare(self.rd, self.workspace, "iter1", self.out)


if __name__ == "__main__":
    unittest.main()
