# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline rejection tests for repository staging and simulation identity."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "shared"))
import prepare_cli_repository as apt
from simulation_gate import SIM_PLUGIN, SIM_STAGE, require_local_domain, require_simulation


class RepositoryGate(unittest.TestCase):
    def test_changed_key_is_rejected_before_gpg_or_source_write(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(apt.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                apt.stage_key(b"untrusted replacement", Path(tmp), "/usr/bin/gpg")
            run.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_verified_bundle_is_processed_without_sudo_in_private_home(self):
        payload = b"test public bundle"
        with tempfile.TemporaryDirectory() as tmp, patch.object(apt.subprocess, "run") as run:
            with patch.object(apt, "KEY_SHA256", hashlib.sha256(payload).hexdigest()):
                apt.stage_key(payload, Path(tmp), "/usr/bin/gpg")
            command = run.call_args.args[0]
            self.assertEqual(command[0], "/usr/bin/gpg")
            self.assertNotIn("sudo", command)
            self.assertEqual((Path(tmp) / "gnupg").stat().st_mode & 0o777, 0o700)
            self.assertEqual((Path(tmp) / "repository.list").read_text(), apt.SOURCE)
            self.assertIn("signed-by=", apt.SOURCE)


class SimulationGate(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "ROS_DOMAIN_ID": "42", "ROS_LOCALHOST_ONLY": "1",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        }
        self.identity = {
            "stage": SIM_STAGE, "playing": True, "domain": 42,
            "localhost_only": "1", "discovery_range": "LOCALHOST",
        }
        self.components = [("IsaacSimSystem", SIM_PLUGIN)]
        self.providers = [("isaacsim_launcher", "/")]

    def verify(self, **kwargs):
        values = dict(identity=self.identity, domain=42, components=self.components,
                      providers=self.providers, use_sim_time=True)
        values.update(kwargs)
        require_simulation(**values)

    def test_matching_simulation(self):
        self.assertEqual(require_local_domain(self.environment), 42)
        self.verify()

    def test_nonlocal_or_implicit_domain_rejected(self):
        for key, value in (("ROS_DOMAIN_ID", "0"), ("ROS_DOMAIN_ID", ""),
                           ("ROS_LOCALHOST_ONLY", "0"), ("ROS_STATIC_PEERS", "peer"),
                           ("RMW_IMPLEMENTATION", "other")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                require_local_domain({**self.environment, key: value})

    def test_wrong_stage_domain_or_stopped_timeline(self):
        for key, value in (("stage", "other.usd"), ("domain", 43), ("playing", False),
                           ("localhost_only", None), ("discovery_range", "SUBNET")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(identity={**self.identity, key: value})

    def test_physical_mixed_missing_or_duplicate_provider_rejected(self):
        for components in ([], [("G1", "physical_driver")],
                           self.components + [("G1", "physical_driver")]):
            with self.subTest(components=components), self.assertRaises(ValueError):
                self.verify(components=components)
        for providers in ([], [("hardware_launcher", "/")], self.providers * 2):
            with self.subTest(providers=providers), self.assertRaises(ValueError):
                self.verify(providers=providers)
        with self.assertRaises(ValueError):
            self.verify(use_sim_time=False)


class ControlBoundary(unittest.TestCase):
    def test_identity_failure_sends_no_control_command(self):
        import importlib.util
        modules = {name: MagicMock() for name in (
            'rclpy', 'controller_manager_msgs', 'controller_manager_msgs.srv',
            'geometry_msgs', 'geometry_msgs.msg', 'rcl_interfaces',
            'rcl_interfaces.msg', 'rcl_interfaces.srv', 'sensor_msgs',
            'sensor_msgs.msg', 'std_srvs', 'std_srvs.srv',
        )}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                'guard_under_test', ROOT / 'scripts/guarded_gantry_release.py')
            guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(guard)
        node = guard.rclpy.create_node.return_value
        node.create_subscription.side_effect = lambda _type, _topic, cb, _qos: cb(MagicMock())
        environment = {
            'ROS_DOMAIN_ID': '42', 'ROS_LOCALHOST_ONLY': '1',
            'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST',
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        }
        with patch.dict(guard.os.environ, environment, clear=True), patch.object(
                sys, 'argv', ['guard', '--allow-simulation-gantry-release']), patch.object(
                guard, 'verify_simulation', side_effect=ValueError('physical target')):
            self.assertEqual(guard.main(), 1)
        node.create_client.return_value.call_async.assert_not_called()
        node.create_publisher.return_value.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
