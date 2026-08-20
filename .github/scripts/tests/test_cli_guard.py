#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The null-rate guard must actually block a regeneration, not just report.

Builds a throwaway repo with one skill, generates benchmarks.json from it,
then removes a field from the source report and regenerates. That second run
is the silent-degradation scenario and must fail loudly.
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
{profile_line}- Environment: `local`
- Tasks: 4 evaluation tasks
- Attempts per task: 1

## Results

| Dimension | claude-code |
|-----------|-------------|
| Accuracy  | 90% (+10%)  |
"""

PROFILE_LINE = "- NVSkills-Eval profile: `external`\n"


class TestGuardBlocksRegeneration(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "skills" / "foo").mkdir(parents=True)
        (self.root / "components.d").mkdir()
        self.report = self.root / "skills" / "foo" / "BENCHMARK.md"
        # Baseline: the field is present, benchmarks.json records it.
        self.report.write_text(REPORT.format(profile_line=PROFILE_LINE))
        (self.root / "benchmarks.json").write_text(agg.generate(self.root))
        self.addCleanup(shutil.rmtree, self.root)

    def _run(self, *extra):
        argv = sys.argv
        sys.argv = ["aggregate_benchmarks.py", "--repo-root", str(self.root), *extra]
        try:
            return agg.main()
        finally:
            sys.argv = argv

    def _drop_the_field(self):
        self.report.write_text(REPORT.format(profile_line=""))

    def test_regeneration_succeeds_when_nothing_empties(self):
        self.assertEqual(self._run(), 0)

    def test_regeneration_fails_when_a_field_empties(self):
        self._drop_the_field()
        self.assertEqual(self._run(), 1)

    def test_failed_run_leaves_benchmarks_json_untouched(self):
        before = (self.root / "benchmarks.json").read_text()
        self._drop_the_field()
        self._run()
        self.assertEqual((self.root / "benchmarks.json").read_text(), before)

    def test_escape_hatch_allows_a_deliberate_format_change(self):
        """An upstream format change must be landable without editing code."""
        self._drop_the_field()
        self.assertEqual(self._run("--allow-null-regressions"), 0)
        written = json.loads((self.root / "benchmarks.json").read_text())
        self.assertIsNone(written["skills"][0]["profile"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
