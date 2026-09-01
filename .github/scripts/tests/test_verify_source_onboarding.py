#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for verify_source_onboarding.py.

The properties that matter, and what breaks if they fail:

  drift detection     A skill whose content moved after signing is dropped by
                      the sync with no notification. This is the whole point
                      of the check; if it stops catching a stale BENCHMARK.md
                      the gate is decorative. Modelled on the real 2026-08-31
                      paidf-orchestration failure, where all four skills
                      carried a signature covering an earlier BENCHMARK.md.

  card name variants  Three spellings are in use across the catalog today
                      (skill-card.md 343, SKILLCARD.yaml 47, card.yaml 18).
                      Recognising only one would fail most valid onboardings.

  new vs pre-existing Roughly 30 catalog skills already carry signature drift
                      (tracked in #216/#357). Blocking on paths a PR did not
                      add would hold authors hostage to drift they cannot fix,
                      so only added paths are allowed to fail the build.

  empty benchmarks    A report can say PASS while carrying no measurements at
                      all. Accepting the verdict alone would wave through a
                      skill with a template report and no evidence behind it.
"""

import base64
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_source_onboarding as vso  # noqa: E402


BENCHMARK_PASS = """# Evaluation Report

## Evaluation Summary

- Skill: `demo-skill`
- Evaluation date: 2026-08-31
- Dataset: 6 evaluation tasks
- Attempts per task: 1
- Pass threshold: 50%
- Overall verdict: PASS

## Results

| Dimension | Claude Code | Codex |
|---|---|---|
| 1. Security | 100% (+10%) | 100% (+10%) |
| 2. Correctness | 90% (+40%) | 95% (+45%) |
"""

# Same header, but the Results section carries no measurement rows.
BENCHMARK_PASS_NO_RESULTS = BENCHMARK_PASS.split("## Results")[0] + "## Results\n\nNot yet run.\n"

BENCHMARK_FAIL = BENCHMARK_PASS.replace("Overall verdict: PASS", "Overall verdict: FAIL")


def sign(skill_dir: Path) -> None:
    """Write a skill.oms.sig covering every file currently in skill_dir.

    Mirrors the real bundle shape: a Sigstore envelope wrapping a DSSE
    payload whose predicate.resources lists name + sha256 per file.
    """
    resources = []
    for f in sorted(skill_dir.rglob("*")):
        if not f.is_file() or f.name == vso.SIG_NAME:
            continue
        resources.append({
            "name": str(f.relative_to(skill_dir)),
            "digest": hashlib.sha256(f.read_bytes()).hexdigest(),
        })
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": skill_dir.name, "digest": {"sha256": "0" * 64}}],
        "predicateType": "https://model_signing/signature/v1.0",
        "predicate": {"resources": resources},
    }
    payload = base64.b64encode(json.dumps(statement).encode()).decode()
    (skill_dir / vso.SIG_NAME).write_text(
        json.dumps({"dsseEnvelope": {"payload": payload, "payloadType": "x"}})
    )


def make_skill(root: Path, name: str = "demo-skill", *, card: str = "skill-card.md",
               benchmark: str = BENCHMARK_PASS, omit: tuple = ()) -> Path:
    """A complete, correctly-signed skill directory."""
    d = root / name
    (d / "evals").mkdir(parents=True)
    files = {
        "SKILL.md": "---\nname: demo-skill\n---\n\nDo the thing.\n",
        card: "# Skill Card\n",
        "BENCHMARK.md": benchmark,
        "evals/evals.json": '{"tasks": []}\n',
    }
    for rel, body in files.items():
        if rel in omit:
            continue
        (d / rel).write_text(body)
    sign(d)
    return d


class CheckSkillTests(unittest.TestCase):
    """The per-skill verification, run against a directory on disk."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_correctly_signed_skill_passes(self):
        self.assertEqual(vso.check_skill(make_skill(self.root)), [])

    def test_file_modified_after_signing_is_reported_by_name(self):
        """The 2026-08-31 defect: BENCHMARK.md regenerated after the signing run."""
        d = make_skill(self.root)
        with (d / "BENCHMARK.md").open("a") as fh:
            fh.write("\n<!-- regenerated after signing -->\n")

        problems = vso.check_skill(d)

        self.assertEqual(len(problems), 1, problems)
        self.assertIn("BENCHMARK.md", problems[0])
        self.assertIn("MISMATCH", problems[0])
        # Nothing else drifted; a report naming extra files would send the
        # author re-signing things that were never wrong.
        self.assertNotIn("SKILL.md", problems[0])

    def test_signed_file_deleted_after_signing_is_reported(self):
        d = make_skill(self.root)
        (d / "evals" / "evals.json").unlink()

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("evals/evals.json", problems)

    def test_unparseable_signature_is_a_failure_not_a_pass(self):
        """A corrupt bundle must not read as 'no problems found'."""
        d = make_skill(self.root)
        (d / vso.SIG_NAME).write_text("not json")

        problems = "\n".join(vso.check_skill(d))

        self.assertIn(vso.SIG_NAME, problems)

    def test_missing_signature_is_reported(self):
        d = make_skill(self.root)
        (d / vso.SIG_NAME).unlink()

        problems = "\n".join(vso.check_skill(d))

        self.assertIn(vso.SIG_NAME, problems)


class RequiredFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_each_skill_card_spelling_is_accepted(self):
        for card in ("skill-card.md", "SKILLCARD.yaml", "card.yaml"):
            with self.subTest(card=card):
                d = make_skill(self.root, name=f"skill-{card}", card=card)
                self.assertEqual(vso.check_skill(d), [])

    def test_missing_skill_card_is_reported(self):
        d = make_skill(self.root, omit=("skill-card.md",))

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("skill card", problems)

    def test_missing_evals_is_reported(self):
        d = make_skill(self.root, omit=("evals/evals.json",))

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("evals/evals.json", problems)

    def test_missing_skill_md_is_reported(self):
        d = make_skill(self.root, omit=("SKILL.md",))

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("SKILL.md", problems)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_failing_verdict_is_reported(self):
        d = make_skill(self.root, benchmark=BENCHMARK_FAIL)

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("FAIL", problems)

    def test_pass_verdict_without_measurements_is_reported(self):
        """PASS on a report carrying no result rows is not evidence."""
        d = make_skill(self.root, benchmark=BENCHMARK_PASS_NO_RESULTS)

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("no results", problems.lower())

    def test_missing_benchmark_is_reported(self):
        d = make_skill(self.root, omit=("BENCHMARK.md",))

        problems = "\n".join(vso.check_skill(d))

        self.assertIn("BENCHMARK.md", problems)


class DeclaredPathTests(unittest.TestCase):
    """Parsing components.d/<slug>.yml without a YAML dependency."""

    COMPONENT = """name: Physical AI Orchestration
repo: NVIDIA/paidf-orchestration
description: Build, run and monitor pipelines.
skills:
  - path: skills/write-dag/
    catalog_dir: paidf-orchestration-write-dag
  - path: skills/orchestration-setup/
    catalog_dir: paidf-orchestration-setup
links:
  discussions: false
"""

    def test_reads_repo_and_paths(self):
        c = vso.parse_component(self.COMPONENT)

        self.assertEqual(c["repo"], "NVIDIA/paidf-orchestration")
        self.assertEqual(c["paths"], ["skills/write-dag", "skills/orchestration-setup"])

    def test_ref_defaults_to_main(self):
        self.assertEqual(vso.parse_component(self.COMPONENT)["ref"], "main")

    def test_explicit_ref_is_honoured(self):
        """AIQ tracks develop; defaulting to main would verify the wrong tree."""
        c = vso.parse_component("repo: NVIDIA-AI-Blueprints/aiq\nref: develop\n"
                                "skills:\n  - path: skills/aiq-research/\n")

        self.assertEqual(c["ref"], "develop")


class BlockingScopeTests(unittest.TestCase):
    """Only paths the PR adds may fail the build."""

    BASE = ("repo: NVIDIA/demo\nskills:\n"
            "  - path: skills/existing/\n")
    HEAD = ("repo: NVIDIA/demo\nskills:\n"
            "  - path: skills/existing/\n"
            "  - path: skills/added/\n")

    def test_added_path_is_blocking(self):
        self.assertEqual(vso.added_paths(self.BASE, self.HEAD), {"skills/added"})

    def test_unchanged_path_is_not_blocking(self):
        self.assertNotIn("skills/existing", vso.added_paths(self.BASE, self.HEAD))

    def test_brand_new_component_blocks_on_every_path(self):
        """A new file has no base version, so all of its skills are new."""
        self.assertEqual(
            vso.added_paths(None, self.HEAD),
            {"skills/existing", "skills/added"},
        )

    def test_removing_a_path_adds_nothing(self):
        self.assertEqual(vso.added_paths(self.HEAD, self.BASE), set())

    def test_pre_existing_drift_does_not_fail_the_run(self):
        """Drift on a path the PR did not touch is reported, never blocking."""
        findings = [
            vso.Finding("skills/existing", "BENCHMARK.md: MISMATCH", blocking=False),
        ]
        self.assertEqual(vso.exit_code(findings), 0)

    def test_drift_on_an_added_path_fails_the_run(self):
        findings = [
            vso.Finding("skills/added", "BENCHMARK.md: MISMATCH", blocking=True),
        ]
        self.assertEqual(vso.exit_code(findings), 1)


if __name__ == "__main__":
    unittest.main()
