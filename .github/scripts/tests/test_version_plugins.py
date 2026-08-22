#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for version-plugins.py decision logic.

Focuses on the `decide`/`build_plan` layer, which is pure and needs no git
or worktree access. The module file carries a hyphen, so it is loaded via
importlib rather than a plain `import`.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("version_plugins", _SCRIPTS / "version-plugins.py")
vp = importlib.util.module_from_spec(_spec)
sys.modules["version_plugins"] = vp
_spec.loader.exec_module(vp)


def analysis(**kw):
    defaults = dict(
        name="foo",
        yaml_path=Path("plugins.d/foo.yml"),
        base_version=vp.SemVer.parse("1.0.0"),
        head_version=vp.SemVer.parse("1.0.0"),
        builder_changed_version=False,
        change_kind=vp.ChangeKind.NONE,
        is_new=False,
        structural_reasons=[],
    )
    defaults.update(kw)
    return vp.PluginAnalysis(**defaults)


class TestNewPluginAcceptsInitialVersion(unittest.TestCase):
    """A plugin yaml newly introduced at head has no base version to
    increase from; its initial version is the builder's stamp and must be
    accepted, not reported as a non-monotonic edit."""

    def test_new_plugin_is_accepted(self):
        a = analysis(
            is_new=True,
            builder_changed_version=True,
            change_kind=vp.ChangeKind.STRUCTURAL,
            structural_reasons=["plugin yaml is newly introduced"],
        )
        verdict, payload = vp.decide(a)
        self.assertEqual(verdict, "accept")
        self.assertIsNone(payload)

    def test_new_plugin_lands_in_no_ops_not_findings(self):
        plan = vp.build_plan(
            [analysis(
                is_new=True,
                builder_changed_version=True,
                change_kind=vp.ChangeKind.STRUCTURAL,
                structural_reasons=["plugin yaml is newly introduced"],
            )]
        )
        self.assertEqual(plan.findings, [])
        self.assertEqual(len(plan.no_ops), 1)


class TestExistingPluginDecisions(unittest.TestCase):
    """Pre-existing behavior must not change."""

    def test_builder_set_version_accepts_when_monotonic(self):
        a = analysis(
            base_version=vp.SemVer.parse("1.0.0"),
            head_version=vp.SemVer.parse("1.1.0"),
            builder_changed_version=True,
            change_kind=vp.ChangeKind.CONTENT,
        )
        verdict, _ = vp.decide(a)
        self.assertEqual(verdict, "accept")

    def test_builder_set_version_fails_when_not_increasing(self):
        a = analysis(
            base_version=vp.SemVer.parse("1.0.0"),
            head_version=vp.SemVer.parse("1.0.0"),
            builder_changed_version=True,
            change_kind=vp.ChangeKind.CONTENT,
        )
        verdict, payload = vp.decide(a)
        self.assertEqual(verdict, "fail")
        self.assertIn("version did not increase", payload)

    def test_major_skip_is_rejected(self):
        a = analysis(
            base_version=vp.SemVer.parse("1.0.0"),
            head_version=vp.SemVer.parse("3.0.0"),
            builder_changed_version=True,
            change_kind=vp.ChangeKind.CONTENT,
        )
        verdict, payload = vp.decide(a)
        self.assertEqual(verdict, "fail")
        self.assertIn("major version jumped by more than 1", payload)

    def test_no_change_is_noop(self):
        verdict, payload = vp.decide(analysis())
        self.assertEqual(verdict, "noop")
        self.assertIsNone(payload)

    def test_unbumped_structural_change_auto_bumps_minor(self):
        a = analysis(change_kind=vp.ChangeKind.STRUCTURAL, structural_reasons=["skills added: x"])
        verdict, payload = vp.decide(a)
        self.assertEqual(verdict, "bump")
        self.assertEqual(payload, "1.1.0")

    def test_unbumped_content_change_auto_bumps_patch(self):
        a = analysis(change_kind=vp.ChangeKind.CONTENT)
        verdict, payload = vp.decide(a)
        self.assertEqual(verdict, "bump")
        self.assertEqual(payload, "1.0.1")


class TestSemVerHelpers(unittest.TestCase):
    def test_parse_rejects_prerelease_tags(self):
        with self.assertRaises(ValueError):
            vp.SemVer.parse("1.0.0-rc1")

    def test_bumped_part(self):
        base = vp.SemVer.parse("1.2.3")
        self.assertEqual(base.bumped_part(vp.SemVer.parse("1.2.4")), "patch")
        self.assertEqual(base.bumped_part(vp.SemVer.parse("1.3.0")), "minor")
        self.assertEqual(base.bumped_part(vp.SemVer.parse("2.0.0")), "major")
        self.assertIsNone(base.bumped_part(vp.SemVer.parse("1.2.3")))
        self.assertIsNone(base.bumped_part(vp.SemVer.parse("1.2.2")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
