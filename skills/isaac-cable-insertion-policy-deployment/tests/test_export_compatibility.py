# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the export-only reset adapter without launching a simulator."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ExportCompatibilityTests(unittest.TestCase):
    def test_export_gate_uses_the_training_audits_supported_leapp_version(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location("test_training_support", scripts / "training_support.py")
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        checks = support.load_training_module("runtime_checks")
        modules = {"leapp": SimpleNamespace(__file__="/test/leapp/__init__.py"),
                   "onnxruntime": SimpleNamespace(__version__="1.26.0",
                       get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])}
        with patch.object(support, "load_training_module", return_value=checks), \
             patch.object(checks, "installation", return_value={}), \
             patch.object(checks, "dependencies", return_value={}), \
             patch.dict(sys.modules, modules):
            with patch("importlib.metadata.version", return_value="0.6.1"):
                result = support.check_export_environment(Path("/test/training"))
                self.assertEqual(result["leapp_version"], "0.6.1")
                self.assertIn("CUDAExecutionProvider", result["onnxruntime_providers"])
            with patch("importlib.metadata.version", return_value="0.7.0"):
                with self.assertRaisesRegex(RuntimeError, "requires locked LEAPP 0.6.1"):
                    support.check_export_environment(Path("/test/training"))

    def test_startup_unchanged_reset_int32_and_restoration_on_failure(self):
        calls = []

        class EventManager:
            _env = SimpleNamespace(num_envs=2, device="cuda:0")

            def apply(self, mode, env_ids=None, *args, **kwargs):
                calls.append((mode, env_ids))

        original = EventManager.apply
        factory = SimpleNamespace(get_delta_dof_pos=object())
        torch = SimpleNamespace(arange=Mock(return_value=[0, 1]), int32=object())

        def upstream_export():
            manager = EventManager()
            manager.apply("startup")
            manager.apply("reset", slice(None))
            manager.apply("interval", [1])
            self.assertIs(factory._get_delta_dof_pos, factory.get_delta_dof_pos)
            raise RuntimeError("intentional export failure")

        upstream = SimpleNamespace(export_rsl_rl_agent=upstream_export)
        modules = {"torch": torch, "isaaclab.managers": SimpleNamespace(EventManager=EventManager),
                   "isaaclab_tasks.contrib.automate": SimpleNamespace(factory_control=factory),
                   "isaaclab_training.cli": SimpleNamespace(export_policy=upstream)}
        path = Path(__file__).resolve().parents[1] / "scripts/export_policy.py"
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("test_export_wrapper", path)
            wrapper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                wrapper.compatible_export()
        self.assertEqual(calls, [("startup", None), ("reset", [0, 1]), ("interval", [1])])
        torch.arange.assert_called_once_with(2, device="cuda:0", dtype=torch.int32)
        self.assertIs(EventManager.apply, original)
        self.assertFalse(hasattr(factory, "_get_delta_dof_pos"))


if __name__ == "__main__":
    unittest.main()
