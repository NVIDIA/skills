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
        # foo must be registered, or it is an orphan the sync would prune and
        # the guard deliberately stops tracking. An unregistered fixture would
        # make these tests pass or fail for a reason unrelated to the parser.
        (self.root / "components.d" / "demo.yml").write_text(
            "name: Demo\nrepo: NVIDIA/demo\nskills:\n"
            "  - path: skills/foo/\n    catalog_dir: foo\n"
        )
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
        # foo must be registered, or it is an orphan the sync would prune and
        # the guard deliberately stops tracking. An unregistered fixture would
        # make these tests pass or fail for a reason unrelated to the parser.
        (self.root / "components.d" / "demo.yml").write_text(
            "name: Demo\nrepo: NVIDIA/demo\nskills:\n"
            "  - path: skills/foo/\n    catalog_dir: foo\n"
        )
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


class TestComponentIsADerivedJoin(unittest.TestCase):
    """`component` is not measurement data and must not gate a PR.

    Every other field in benchmarks.json is parsed out of a skill's
    BENCHMARK.md. `component` is not: it is joined in from components.d at
    generation time. That makes it the one field that changes the instant a
    PR edits components.d, while benchmarks.json cannot be regenerated until
    the sync has run and renamed the directories on disk.

    On 2026-09-01 that cost a release. #519 renamed catalog_dirs, which is a
    components.d-only change by design — the sync writes the renamed dirs and
    prune-orphans deletes the old ones. But:

      * --check regenerated, saw component values move, and failed;
      * the remedy it printed (regenerate) hit the null guard, because the
        deregistered dirs still on disk now matched no component and went
        null: "component: 0 -> 2 nulls";
      * so the check demanded a regeneration the generator refused to do.

    The only escape was hand-deleting the catalog dirs, which then required a
    signing run that cannot fire on a fork PR. Two gates, the second caused
    by the first.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "components.d").mkdir()
        for name in ("foo", "bar"):
            (self.root / "skills" / name).mkdir(parents=True)
            (self.root / "skills" / name / "BENCHMARK.md").write_text(
                REPORT.format(
                    environment_line=ENVIRONMENT_LINE,
                    threshold_line=THRESHOLD_LINE,
                ).replace("`foo`", f"`{name}`")
            )
        self._register(foo="Alpha", bar="Alpha")
        (self.root / "benchmarks.json").write_text(agg.generate(self.root))
        self.addCleanup(shutil.rmtree, self.root)

    def _register(self, **dirs):
        """Rewrite components.d so exactly these catalog_dirs are registered."""
        by_component = {}
        for catalog_dir, component in dirs.items():
            by_component.setdefault(component, []).append(catalog_dir)
        for f in (self.root / "components.d").glob("*.yml"):
            f.unlink()
        for component, entries in by_component.items():
            body = f"name: {component}\nrepo: NVIDIA/demo\nskills:\n"
            for d in entries:
                body += f"  - path: skills/{d}/\n    catalog_dir: {d}\n"
            (self.root / "components.d" / f"{component.lower()}.yml").write_text(body)

    def _run(self, *extra):
        argv = sys.argv
        sys.argv = ["aggregate_benchmarks.py", "--repo-root", str(self.root), *extra]
        try:
            return agg.main()
        finally:
            sys.argv = argv

    # --- --check ---------------------------------------------------------

    def test_check_passes_when_a_skill_moves_component(self):
        """The #519 case: a registered skill reassigned to another product."""
        self._register(foo="Alpha", bar="Beta")

        self.assertEqual(self._run("--check"), 0)

    def test_check_passes_when_a_dir_is_deregistered_pending_prune(self):
        """Deregistered dirs are deleted by prune-orphans after the sync."""
        self._register(foo="Alpha")

        self.assertEqual(self._run("--check"), 0)

    def test_check_still_fails_on_a_real_measurement_change(self):
        """Ignoring component must not blind the check to actual drift."""
        (self.root / "skills" / "foo" / "BENCHMARK.md").write_text(
            REPORT.format(environment_line="", threshold_line=THRESHOLD_LINE)
        )

        self.assertEqual(self._run("--check"), 1)

    def test_check_still_fails_when_a_skill_appears(self):
        (self.root / "skills" / "baz").mkdir(parents=True)
        (self.root / "skills" / "baz" / "BENCHMARK.md").write_text(
            REPORT.format(
                environment_line=ENVIRONMENT_LINE, threshold_line=THRESHOLD_LINE
            ).replace("`foo`", "`baz`")
        )

        self.assertEqual(self._run("--check"), 1)

    # --- the null guard --------------------------------------------------

    def test_deregistered_dir_going_null_does_not_block_regeneration(self):
        """The exact refusal that trapped #519."""
        self._register(foo="Alpha")

        self.assertEqual(self._run(), 0)

    def test_a_registered_skill_losing_a_field_still_blocks(self):
        """The orphan exemption must not disarm the guard generally."""
        (self.root / "skills" / "foo" / "BENCHMARK.md").write_text(
            REPORT.format(environment_line="", threshold_line=THRESHOLD_LINE)
        )

        self.assertEqual(self._run(), 1)

    def test_an_exception_without_a_component_is_not_an_orphan(self):
        """catalog-exceptions membership decides survival, not the component.

        Reading registration off load_component_map() would drop an exception
        that omits `component:` and silently stop guarding it.
        """
        self._register(foo="Alpha")
        (self.root / "catalog-exceptions.yml").write_text(
            "exceptions:\n  - dir: bar\n    reason: manually curated\n"
        )

        self.assertIn("bar", agg.registered_catalog_dirs(self.root))
