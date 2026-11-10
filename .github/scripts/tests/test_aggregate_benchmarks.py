#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aggregate_benchmarks.py report parsing.

Fixtures are verbatim BENCHMARK.md files from the catalog, one per report
layout in circulation:

  v2 — SkillEvaluator 0.9.x  (skills/cuopt-developer, evaluated 2026-06)
  v3 — SkillEvaluator 1.3.x  (skills/nemotron-speech, evaluated 2026-08)

v3 dropped two fields the parser read (`NVSkills-Eval profile` and
`Pass threshold`) and added three it does not (`Evaluator version`,
`Dataset digest`, `Validation status`).

`profile` is now retired: NVSkills-Eval is an internal name and does not
belong in a published report, so the parser no longer looks for it.
`pass_threshold_pct` is kept and left None on v3 — it is a real provenance
value that v3 turned into template prose. These tests pin both decisions.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aggregate_benchmarks as agg  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
V2 = FIXTURES / "v2_cuopt_developer.md"
V3 = FIXTURES / "v3_nemotron_speech.md"
SELF_ASSESSED = FIXTURES / "self_assessed_daqiri.md"


class TestSelfAssessedReports(unittest.TestCase):
    """A reported PASS needs the publication and execution checks as well."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "skills" / "daqiri-onboarding" / "BENCHMARK.md"
        self.path.parent.mkdir(parents=True)
        self.report = SELF_ASSESSED.read_text(encoding="utf-8")

    def parse(self, report):
        self.path.write_text(report, encoding="utf-8")
        return agg.parse_benchmark(self.path)

    def test_eligible_report_retains_scores_and_zero_uplift(self):
        entry = self.parse(self.report)
        self.assertEqual(entry["verdict"], "PASS")
        self.assertEqual(entry["skill"], "daqiri-onboarding")
        self.assertEqual(len(entry["results"]), 4)
        self.assertEqual(entry["results"][0]["agents"]["codex"],
                         {"score_pct": 100.0, "uplift_pct": 0.0})
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.report)

    def test_aggregation_uses_the_same_verdict(self):
        self.parse(self.report)
        data = json.loads(agg.generate(self.root))
        self.assertEqual(data["skills"][0]["verdict"], "PASS")
        self.assertTrue(data["skills"][0]["has_results"])
        self.assertEqual(data["skills"][0]["average_uplift_pct"], 0.0)
        self.assertEqual(data["result_row_count"], 8)

    def test_non_passing_producer_verdicts_are_preserved(self):
        for verdict in ("FAIL", "NEUTRAL"):
            with self.subTest(verdict=verdict):
                entry = self.parse(self.report.replace(
                    "Reported evaluation verdict: PASS", f"Reported evaluation verdict: {verdict}"))
                self.assertEqual(entry["verdict"], verdict)

    def test_missing_invalid_or_conflicting_summary_fields_do_not_pass(self):
        replacements = [
            ("Publication eligible", "Publication blocked"),
            ("Publication eligible", "Publication not assessed"),
            ("Publication eligible", "Publication unknown"),
            ("> **Publication eligible**", ""),
            ("> **Publication eligible**", "> **Publication eligible**\n> **Publication blocked**"),
            ("Evaluation source: self-assessed", "Evaluation source: unknown"),
            ("Quality assessment: SELF-ASSESSED", "Quality assessment: FAILED"),
            ("Reported execution: completed", "Reported execution: failed"),
            ("Reported execution: completed", "Reported execution: incomplete"),
            ("Reported execution: completed", "Reported execution: cancelled"),
            ("Reported evaluation verdict: PASS", "Reported evaluation verdict: NOT RECORDED"),
            ("Reported evaluation verdict: PASS", "Reported evaluation verdict: PASS only when ..."),
            ("## Publication Recommendation", "## Example"),
            ("## Evaluation Metadata", "## Example Metadata"),
        ]
        for field in ("Evaluation source: self-assessed", "Quality assessment: SELF-ASSESSED",
                      "Reported execution: completed", "Reported evaluation verdict: PASS"):
            replacements.extend([
                (f"- {field}\n", ""),
                (f"- {field}", f"- {field}\n- {field}"),
                (f"- {field}", f"- {field}\n- {field.split(':')[0]}: unknown"),
            ])
        for old, new in replacements:
            with self.subTest(old=old, new=new):
                self.assertNotEqual(self.parse(self.report.replace(old, new))["verdict"], "PASS")

    def test_execution_errors_block_an_otherwise_eligible_report(self):
        report = self.report.replace("## Evaluation Metadata",
                                     "**Reported execution errors:**\n\n- timed out\n\n## Evaluation Metadata")
        self.assertNotEqual(self.parse(report)["verdict"], "PASS")

    def test_example_and_methodology_text_cannot_supply_a_missing_summary(self):
        intro, body = self.report.split("## Evaluation Metadata", 1)
        for wrapper in ("```markdown\n{}\n```", "~~~markdown\n{}\n~~~", "<!--\n{}\n-->"):
            with self.subTest(wrapper=wrapper):
                report = wrapper.format(intro) + "\n## Evaluation Metadata" + body
                self.assertNotEqual(self.parse(report)["verdict"], "PASS")
        report = "# Skill Benchmark: daqiri-onboarding\n\n## Evaluation Metadata" + body
        report += "\n## Scoring Methodology\n" + intro
        self.assertNotEqual(self.parse(report)["verdict"], "PASS")

    def test_explicit_non_passing_overall_verdict_is_not_overridden(self):
        for verdict in ("FAIL", "INCOMPLETE", "NEUTRAL"):
            for section in ("Publication Recommendation", "Evaluation Metadata", "Scoring Methodology"):
                with self.subTest(verdict=verdict, section=section):
                    report = self.report.replace(f"## {section}",
                                                 f"## {section}\n\n- Overall verdict: {verdict}")
                    self.assertEqual(self.parse(report)["verdict"], verdict)

    def test_mixed_verdict_layouts_cannot_hide_a_later_failure(self):
        report = self.report.replace("## Evaluation Metadata",
                                    "## Evaluation Metadata\n\n- Overall verdict: PASS\n- Overall verdict: FAIL")
        self.assertNotEqual(self.parse(report)["verdict"], "PASS")

    def test_conflicting_overall_pass_cannot_hide_blocked_publication(self):
        report = self.report.replace("Publication eligible", "Publication blocked")
        report = report.replace("## Publication Recommendation",
                                "> **Overall verdict: PASS**\n\n## Publication Recommendation")
        self.assertNotEqual(self.parse(report)["verdict"], "PASS")


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

    def test_profile_is_not_emitted_at_all(self):
        """Retired field: absent from every entry, both layouts."""
        self.assertNotIn("profile", agg.parse_benchmark(V3))
        self.assertNotIn("profile", agg.parse_benchmark(V2))


class TestExistingBehaviourStillWorks(unittest.TestCase):
    """Regression guard: the v2 path and shared fields must not change."""

    def test_v2_still_parses_legacy_fields(self):
        entry = agg.parse_benchmark(V2)
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
    column quietly goes null. Both the pass_threshold drift and the
    2026-08-03 disappearance of cuopt-multi-objective-exploration are this
    shape.
    """

    def test_flags_a_field_that_lost_values(self):
        old = {"skills": [{"skill": "a", "environment": "k8s-sandbox"},
                          {"skill": "b", "environment": "k8s-sandbox"}]}
        new = {"skills": [{"skill": "a", "environment": None},
                          {"skill": "b", "environment": None}]}
        regressions = agg.null_rate_regressions(old, new)
        self.assertIn("environment", regressions)
        self.assertEqual(regressions["environment"], (0, 2))

    def test_silent_on_unchanged_null_rates(self):
        old = {"skills": [{"skill": "a", "environment": None}]}
        new = {"skills": [{"skill": "a", "environment": None}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})

    def test_silent_when_a_field_gains_values(self):
        old = {"skills": [{"skill": "a", "environment": None}]}
        new = {"skills": [{"skill": "a", "environment": "k8s-sandbox"}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})

    def test_ignores_skills_absent_from_the_old_file(self):
        """A newly added skill with empty fields is not a regression."""
        old = {"skills": [{"skill": "a", "environment": "k8s-sandbox"}]}
        new = {"skills": [{"skill": "a", "environment": "k8s-sandbox"},
                          {"skill": "b", "environment": None}]}
        self.assertEqual(agg.null_rate_regressions(old, new), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
