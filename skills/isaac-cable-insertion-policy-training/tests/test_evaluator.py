# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check evaluator startup against the standalone release's public runner API."""

from pathlib import Path
import json
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class RunnerReached(Exception):
    """Stop before loading a checkpoint or running simulation steps."""


class EvaluatorTests(unittest.TestCase):
    def test_campaign_source_drift_is_rejected_before_importing_simulation(self):
        from storage import SCHEMA_VERSION, training_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/isaaclab_training/__init__.py"
            source.parent.mkdir(parents=True)
            source.write_text("# original\n")
            (root / "pyproject.toml").write_text("# project\n")
            (root / "uv.lock").write_text("# lock\n")
            (root / "session.json").write_text(json.dumps({
                "schema_version": SCHEMA_VERSION, "plan": {"training_root": str(root)},
                "state": {"attempts": [{"run_id": "baseline", "training_sha256": training_fingerprint(root)}]},
            }))
            source.write_text("# changed\n")
            argv = [str(SCRIPTS / "evaluate.py"), "--checkpoint", str(root / "model.pt"),
                    "--output", str(root / "result.json"), "--seed", "7",
                    "--session", str(root), "--run-id", "baseline"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"gymnasium": None, "torch": None}):
                with self.assertRaisesRegex(RuntimeError, "changed since this run"):
                    runpy.run_path(str(SCRIPTS / "evaluate.py"), run_name="__main__")

    def run_startup(self, runner_class):
        agent = SimpleNamespace(class_name=runner_class, seed=0, num_steps_per_env=128,
                                clip_actions=1.0, device="cuda:0",
                                to_dict=lambda: {"class_name": runner_class, "seed": 35001})
        env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), seed=0,
                                 events=SimpleNamespace(reset_plug_curriculum=SimpleNamespace(params={})),
                                 observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)))
        base = SimpleNamespace(device="cuda:0", _reset_idx=lambda ids: None)
        raw = SimpleNamespace(unwrapped=base)
        wrapped = object()
        self.launch = MagicMock()
        self.convert = MagicMock(return_value=agent)
        self.runner = MagicMock(side_effect=RunnerReached)
        modules = {
            "gymnasium": SimpleNamespace(make=MagicMock(return_value=raw)),
            "torch": MagicMock(),
            "isaaclab.app": SimpleNamespace(launch_simulation=self.launch),
            # Deliberately expose only names that exist in the installed release.
            "isaaclab_rl.rsl_rl": SimpleNamespace(
                RslRlVecEnvWrapper=MagicMock(return_value=wrapped),
                handle_deprecated_rsl_rl_cfg=self.convert),
            "isaaclab_tasks.utils": SimpleNamespace(resolve_task_config=lambda *a, **k: (env_cfg, agent)),
            "rsl_rl.runners": SimpleNamespace(OnPolicyRunner=self.runner),
            "isaaclab_training": SimpleNamespace(),
            "isaaclab_training.tasks": SimpleNamespace(),
        }
        argv = [str(SCRIPTS / "evaluate.py"), "--checkpoint", "/unused/model.pt",
                "--output", "/unused/result.json", "--seed", "35001"]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
             patch("importlib.metadata.version", return_value="5.4.1") as version:
            expected = RunnerReached if runner_class == "OnPolicyRunner" else ValueError
            with self.assertRaises(expected) as raised:
                runpy.run_path(str(SCRIPTS / "evaluate.py"), run_name="__main__")
        version.assert_called_once_with("rsl-rl-lib")
        self.convert.assert_called_once_with(agent, "5.4.1")
        return wrapped, raised.exception

    def test_public_api_reaches_runner_with_resolved_configuration(self):
        wrapped, _ = self.run_startup("OnPolicyRunner")
        self.runner.assert_called_once_with(wrapped, {"class_name": "OnPolicyRunner", "seed": 35001},
                                            log_dir=None, device="cuda:0")
        self.launch.assert_called_once()

    def test_unsupported_runner_is_rejected_before_simulation(self):
        _, error = self.run_startup("DistillationRunner")
        self.assertIn("Unsupported evaluation runner: DistillationRunner", str(error))
        self.launch.assert_not_called()
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
