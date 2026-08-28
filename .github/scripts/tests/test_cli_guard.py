#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The null-rate guard must actually block a regeneration, not just report.

Builds a throwaway repo with one skill, generates benchmarks.json from it,
then removes a field from the source report and regenerates. That second run
is the silent-degradation scenario and must fail loudly.

The probe field is `environment`. It used to be `pass_threshold_pct`, but that
field is now in MIGRATING_FIELDS — v3 cards stopped emitting it, so it drifts
to null by design and no longer blocks. Probing with it would have made these
tests pass for the wrong reason.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aggregate_benchmarks as agg  # noqa: E402

REPORT = """# Skill Benchmark: foo

> **Overall verdict: PASS**

## Evaluation Metadata

- Skill: `foo`
- Evaluation date: 2026-06-01
{environment_line}{threshold_line}- Tasks: 4 evaluation tasks
- Attempts per task: 1

## Results

| Dimension | claude-code |
|-----------|-------------|
| Accuracy  | 90% (+10%)  |
"""

ENVIRONMENT_LINE = "- Environment: `local`\n"
THRESHOLD_LINE = "- Pass threshold: 50%\n"


class TestGuardBlocksRegeneration(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "skills" / "foo").mkdir(parents=True)
        (self.root / "components.d").mkdir()
        self.report = self.root / "skills" / "foo" / "BENCHMARK.md"
        # Baseline: both fields present, benchmarks.json records them.
        self._write()
        (self.root / "benchmarks.json").write_text(agg.generate(self.root))
        self.addCleanup(shutil.rmtree, self.root)

    def _write(self, *, environment=True, threshold=True):
        self.report.write_text(
            REPORT.format(
                environment_line=ENVIRONMENT_LINE if environment else "",
                threshold_line=THRESHOLD_LINE if threshold else "",
            )
        )

    def _run(self, *extra):
        argv = sys.argv
        sys.argv = ["aggregate_benchmarks.py", "--repo-root", str(self.root), *extra]
        try:
            return agg.main()
        finally:
            sys.argv = argv

    def _written(self):
        return json.loads((self.root / "benchmarks.json").read_text())["skills"][0]

    def test_regeneration_succeeds_when_nothing_empties(self):
        self.assertEqual(self._run(), 0)

    def test_regeneration_fails_when_a_field_empties(self):
        self._write(environment=False)
        self.assertEqual(self._run(), 1)

    def test_failed_run_leaves_benchmarks_json_untouched(self):
        before = (self.root / "benchmarks.json").read_text()
        self._write(environment=False)
        self._run()
        self.assertEqual((self.root / "benchmarks.json").read_text(), before)

    def test_escape_hatch_allows_a_deliberate_format_change(self):
        """An upstream format change must be landable without editing code."""
        self._write(environment=False)
        self.assertEqual(self._run("--allow-null-regressions"), 0)
        self.assertIsNone(self._written()["environment"])


class TestMigratingFieldExemption(unittest.TestCase):
    """A field mid-retirement must not block, and must not hide anything else.

    v3 cards dropped the "- Pass threshold: N%" line, so every skill that
    re-runs CI adds one null. Without the exemption the guard blocked every
    sync that migrated any skill — it fired on cuopt-server-api-python on
    2026-08-28 and stalled metadata regeneration for hours.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "skills" / "foo").mkdir(parents=True)
        (self.root / "components.d").mkdir()
        self.report = self.root / "skills" / "foo" / "BENCHMARK.md"
        self.report.write_text(
            REPORT.format(environment_line=ENVIRONMENT_LINE, threshold_line=THRESHOLD_LINE)
        )
        (self.root / "benchmarks.json").write_text(agg.generate(self.root))
        self.addCleanup(shutil.rmtree, self.root)

    def _write(self, *, environment=True, threshold=True):
        self.report.write_text(
            REPORT.format(
                environment_line=ENVIRONMENT_LINE if environment else "",
                threshold_line=THRESHOLD_LINE if threshold else "",
            )
        )

    def _run(self, *extra):
        argv = sys.argv
        sys.argv = ["aggregate_benchmarks.py", "--repo-root", str(self.root), *extra]
        try:
            return agg.main()
        finally:
            sys.argv = argv

    def test_pass_threshold_pct_is_exempt(self):
        self.assertIn("pass_threshold_pct", agg.MIGRATING_FIELDS)

    def test_losing_only_a_migrating_field_does_not_block(self):
        self._write(threshold=False)
        self.assertEqual(self._run(), 0)
        written = json.loads((self.root / "benchmarks.json").read_text())["skills"][0]
        self.assertIsNone(written["pass_threshold_pct"])
        # The rest of the row must survive the write.
        self.assertEqual(written["environment"], "local")

    def test_exemption_does_not_mask_a_real_regression(self):
        """Both fields empty at once: the non-exempt one must still block."""
        self._write(environment=False, threshold=False)
        self.assertEqual(self._run(), 1)

    def test_migrating_drift_is_still_reported(self):
        """Exempt does not mean silent — the drift must stay visible."""
        self._write(threshold=False)
        regressions = agg.null_rate_regressions(
            json.loads((self.root / "benchmarks.json").read_text()),
            json.loads(agg.generate(self.root)),
        )
        self.assertIn("pass_threshold_pct", regressions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
