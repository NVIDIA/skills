#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aggregate_benchmarks.py report parsing.

Fixtures are verbatim BENCHMARK.md files from the catalog, one per report
layout in circulation:

  v2 — SkillEvaluator 0.9.x  (skills/cuopt-developer, evaluated 2026-06)
  v3 — SkillEvaluator 1.3.x  (skills/nemotron-speech, evaluated 2026-08)

v3 dropped two fields the parser reads (`NVSkills-Eval profile` and
`Pass threshold`) and added three it does not (`Evaluator version`,
`Dataset digest`, `Validation status`). These tests pin both halves of that.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aggregate_benchmarks as agg  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
V2 = FIXTURES / "v2_cuopt_developer.md"
V3 = FIXTURES / "v3_nemotron_speech.md"


class TestV3ProvenanceFields(unittest.TestCase):
    """v3 carries per-run provenance the parser currently discards."""

    def test_captures_evaluator_version(self):
        entry = agg.parse_benchmark(V3)
        self.assertEqual(entry["evaluator_version"], "1.3.2")

    def test_captures_dataset_digest(self):
        entry = agg.parse_benchmark(V3)
        self.assertEqual(
            entry["dataset_digest"],
            "sha256:7da18a129d0ad5efdc392d764333668f68f9acf996152684bc2464427afdb20f",
        )

    def test_captures_validation_status(self):
        entry = agg.parse_benchmark(V3)
        self.assertEqual(entry["validation_status"], "passed")

    def test_v2_reports_lack_the_new_fields_and_stay_none(self):
        """Older reports must not error, just leave the new fields empty."""
        entry = agg.parse_benchmark(V2)
        self.assertIsNone(entry["evaluator_version"])
        self.assertIsNone(entry["dataset_digest"])
        self.assertIsNone(entry["validation_status"])


class TestNoFabricatedProvenance(unittest.TestCase):
    """The v3 glossary mentions '50%' as static template prose.

    It is byte-identical across skills with 4, 5 and 18 tasks, so it is not a
    per-skill measurement. Parsing it would stamp 50.0 on every skill
    regardless of what it was actually evaluated against — a fabricated
    provenance claim in the file whose job is recording provenance. None is
    the honest value.
    """

    def test_pass_threshold_is_none_for_v3_not_scraped_from_glossary(self):
        entry = agg.parse_benchmark(V3)
        self.assertIsNone(entry["pass_threshold_pct"])

    def test_profile_is_none_for_v3(self):
        entry = agg.parse_benchmark(V3)
        self.assertIsNone(entry["profile"])


class TestExistingBehaviourStillWorks(unittest.TestCase):
    """Regression guard: the v2 path and shared fields must not change."""

    def test_v2_still_parses_legacy_fields(self):
        entry = agg.parse_benchmark(V2)
        self.assertEqual(entry["profile"], "external")
        self.assertEqual(entry["pass_threshold_pct"], 50.0)

    def test_v3_shared_fields_parse(self):
        entry = agg.parse_benchmark(V3)
        self.assertEqual(entry["skill"], "nemotron-speech")
        self.assertEqual(entry["evaluation_date"], "2026-08-20")
        self.assertEqual(entry["environment"], "local")
        self.assertEqual(entry["tasks"], 18)
        self.assertEqual(entry["attempts_per_task"], 1)
        self.assertEqual(entry["verdict"], "PASS")


class TestNullRateRegressionGuard(unittest.TestCase):
    """A field emptying across a regeneration must fail loudly.

    This is the generic guard for the whole class of silent degradation:
    the regeneration succeeds, the schema stays valid, --check passes, and a
    column quietly goes null. Both the profile/pass_threshold drift and the
    2026-08-03 disappearance of cuopt-multi-objective-exploration are this
    shape.
    """

    def test_flags_a_field_that_lost_values(self):
        old = {"skills": [{"skill": "a", "profile": "external"},
                          {"skill": "b", "profile": "external"}]}
        new = {"skills": [{"skill": "a", "profile": None},
                          {"skill": "b", "profile": None}]}
        regressions = agg.null_rate_regressions(old, new)
        self.assertIn("profile", regressions)
        self.assertEqual(regressions["profile"], (0, 2))

    def test_silent_on_unchanged_null_rates(self):
        old = {"skills": [{"skill": "a", "profile": None}]}
        new = {"skills": [{"skill": "a", "profile": None}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})

    def test_silent_when_a_field_gains_values(self):
        old = {"skills": [{"skill": "a", "profile": None}]}
        new = {"skills": [{"skill": "a", "profile": "external"}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})

    def test_ignores_skills_absent_from_the_old_file(self):
        """A newly added skill with empty fields is not a regression."""
        old = {"skills": [{"skill": "a", "profile": "external"}]}
        new = {"skills": [{"skill": "a", "profile": "external"},
                          {"skill": "b", "profile": None}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
