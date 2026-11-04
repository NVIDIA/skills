# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency conflicts must block launch even when locked installation provenance passes."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_checks


def distribution(name, version, *requires):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=list(requires))


class DependencyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "uv.lock").write_text('[[package]]\nname="torchvision"\nversion="0.29.0"\n')

    def audit(self, packages, overrides=()):
        (self.root / "pyproject.toml").write_text(
            '[tool.uv]\noverride-dependencies=' + json.dumps(list(overrides)) + '\n')
        with patch.object(runtime_checks.sys, "prefix", str(self.root / ".venv")), \
             patch.object(runtime_checks.importlib.metadata, "distributions", return_value=packages):
            return runtime_checks.dependencies(self.root)

    def test_consistent_requirements(self):
        result = self.audit([distribution("trainer", "1", "torchvision>=0.25"),
                             distribution("torchvision", "0.29.0")])
        self.assertEqual(result, {"checked_distributions": 2, "project_override_exceptions": []})

    def test_missing_dependency_is_not_excused_by_override(self):
        with self.assertRaisesRegex(RuntimeError, "ipywidgets is missing"):
            self.audit([distribution("open3d", "0.19.0", "ipywidgets>=8.0")], ["ipywidgets>=8.0"])

    def test_undeclared_conflict_blocks(self):
        with self.assertRaisesRegex(RuntimeError, "requires torchvision==0.26.0"):
            self.audit([distribution("isaacsim-core", "6.1.0.0", "torchvision==0.26.0"),
                        distribution("torchvision", "0.29.0")])

    def test_explicit_satisfied_override_is_recorded(self):
        result = self.audit([distribution("isaacsim-core", "6.1.0.0", "torchvision==0.26.0"),
                             distribution("torchvision", "0.29.0")], ["torchvision>=0.25"])
        exception, = result["project_override_exceptions"]
        self.assertEqual(exception["requirement"], "torchvision==0.26.0")
        self.assertEqual(exception["override"], "torchvision>=0.25")

    def test_unsatisfied_override_still_blocks(self):
        with self.assertRaisesRegex(RuntimeError, "requires torchvision==0.26.0"):
            self.audit([distribution("isaacsim-core", "6.1.0.0", "torchvision==0.26.0"),
                        distribution("torchvision", "0.29.0")], ["torchvision>=0.30"])

    def test_satisfied_requirement_does_not_excuse_lock_drift(self):
        with self.assertRaisesRegex(RuntimeError, "differs from uv.lock"):
            self.audit([distribution("trainer", "1", "torchvision>=0.25"),
                        distribution("torchvision", "0.30.0")])

    def test_inactive_extra_is_not_required(self):
        result = self.audit([distribution("trainer", "1", "optional-package; extra == 'docs'")])
        self.assertEqual(result["project_override_exceptions"], [])

    def test_override_for_other_platform_does_not_excuse_conflict(self):
        with self.assertRaisesRegex(RuntimeError, "requires torchvision==0.26.0"):
            self.audit([distribution("isaacsim-core", "6.1.0.0", "torchvision==0.26.0"),
                        distribution("torchvision", "0.29.0")], ['torchvision>=0.25; sys_platform == "never"'])

    def test_conflicting_onnx_providers_block(self):
        with self.assertRaisesRegex(RuntimeError, "both provide the onnxruntime import"):
            self.audit([distribution("onnxruntime", "1.26.0"), distribution("onnxruntime-gpu", "1.26.0")])

    def test_gpu_supplement_satisfies_sim_without_installing_locked_cpu(self):
        with (self.root / "uv.lock").open("a") as stream:
            stream.write('[[package]]\nname="onnxruntime"\nversion="1.26.0"\n')
        result = self.audit([
            distribution("isaacsim-robot", "6.1.0.0", "onnxruntime-gpu==1.26.0"),
            distribution("onnxruntime-gpu", "1.26.0"),
            distribution("open3d", "0.19.0", "ipywidgets>=8.0.4"),
            distribution("ipywidgets", "8.1.9"),
        ])
        self.assertEqual(result["project_override_exceptions"], [])

    def test_gpu_supplement_does_not_waive_real_cpu_requirement(self):
        with self.assertRaisesRegex(RuntimeError, "onnxruntime is missing"):
            self.audit([distribution("other-consumer", "0.7.0", "onnxruntime"),
                        distribution("onnxruntime-gpu", "1.26.0")])

    def test_reviewed_leapp_gpu_provider_is_reported(self):
        with (self.root / "uv.lock").open("a") as stream:
            stream.write('[[package]]\nname="leapp"\nversion="0.6.1"\n')
        result = self.audit([distribution("leapp", "0.6.1", "onnxruntime>=1.20.0"),
                             distribution("onnxruntime-gpu", "1.26.0")])
        substitution, = result["provider_substitutions"]
        self.assertEqual(substitution["provider"], "onnxruntime-gpu")
        self.assertEqual(substitution["version"], "0.6.1")
        for version, requirement in [("1.25.0", "onnxruntime>=1.20.0"),
                                     ("1.26.0", "onnxruntime>=1.27.0")]:
            with self.subTest(version=version, requirement=requirement):
                with self.assertRaisesRegex(RuntimeError, "onnxruntime is missing"):
                    self.audit([distribution("leapp", "0.6.1", requirement),
                                distribution("onnxruntime-gpu", version)])

    def test_unreviewed_leapp_provider_does_not_pass(self):
        for version in ("0.6.1", "0.7.0", "0.7.1"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(RuntimeError, "onnxruntime is missing"):
                    self.audit([distribution("leapp", version, "onnxruntime>=1.20.0"),
                                distribution("onnxruntime-gpu", "1.26.0")])

    def test_other_locked_leapp_version_is_not_implicitly_supported(self):
        with (self.root / "uv.lock").open("a") as stream:
            stream.write('[[package]]\nname="leapp"\nversion="0.7.0"\n')
        with self.assertRaisesRegex(RuntimeError, "onnxruntime is missing"):
            self.audit([distribution("leapp", "0.7.0", "onnxruntime>=1.20.0"),
                        distribution("onnxruntime-gpu", "1.26.0")])

    def test_supported_leapp_must_match_lock(self):
        with (self.root / "uv.lock").open("a") as stream:
            stream.write('[[package]]\nname="leapp"\nversion="0.7.0"\n')
        with self.assertRaisesRegex(RuntimeError, "differs from uv.lock"):
            self.audit([distribution("leapp", "0.6.1", "onnxruntime>=1.20.0"),
                        distribution("onnxruntime-gpu", "1.26.0")])


if __name__ == "__main__":
    unittest.main()
