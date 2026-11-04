# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-boundary tests without starting Kit or ROS."""
import importlib.util
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from sim_record import summarize_policy_sync


class PolicySyncTests(unittest.TestCase):
    def setUp(self):
        # The controller extends a USD ScriptNode whose compute function already
        # exists. Supply that entrypoint before loading the complete local module;
        # Kit/ROS helpers are only needed when their controller functions run.
        spec = importlib.util.spec_from_file_location(
            "cable_test_controller", SCRIPTS / "osc_scene_controller.py")
        controller = importlib.util.module_from_spec(spec)
        controller.compute = Mock()
        spec.loader.exec_module(controller)
        self.apply = Mock()
        command = patch.object(controller, "_cartesian_joint_command", self.apply)
        command.start()
        self.addCleanup(command.stop)
        self.sync = controller._synchronize_policy
        self.state = NS(control_mode="policy", latest_cartesian_record=None,
                        rclpy=NS(spin_once=Mock()), node=object())
        self.db = NS(per_instance_state=self.state,
                     inputs=NS(simulationTime=1.0, deltaSimulationTime=1 / 240))
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def record(self, sec, nanosec=0):
        return {"msg": NS(header=NS(stamp=NS(sec=sec, nanosec=nanosec)))}

    def test_waits_through_stale_messages_then_applies_matching_action(self):
        messages = iter([self.record(0, 966666667), self.record(1)])
        self.state.rclpy.spin_once.side_effect = lambda *a, **k: setattr(
            self.state, "latest_cartesian_record", next(messages))
        with patch("time.monotonic", side_effect=itertools.count(0, 0.01)):
            self.sync(self.db)
        self.assertEqual(self.state.rclpy.spin_once.call_count, 2)
        self.apply.assert_called_once_with(self.db, 1 / 240)

    def test_joint_mode_and_intermediate_physics_steps_do_not_request_inference(self):
        self.state.control_mode = "joint"
        self.sync(self.db)
        self.state.control_mode = "policy"
        self.db.inputs.simulationTime += 1 / 240
        self.sync(self.db)
        self.apply.assert_not_called()
        self.state.rclpy.spin_once.assert_not_called()

    def test_boundary_cannot_apply_twice(self):
        self.state.latest_cartesian_record = self.record(1)
        self.sync(self.db)
        self.sync(self.db)
        self.apply.assert_called_once()

    def test_missing_response_times_out_and_marks_recording_invalid(self):
        with patch("time.monotonic", side_effect=[0, 6]), self.assertRaises(TimeoutError):
            self.sync(self.db)
        self.assertIn("timed out", os.environ["CABLE_POLICY_SYNC_FAILED"])
        self.apply.assert_not_called()

    def test_future_command_is_rejected(self):
        self.state.latest_cartesian_record = self.record(2)
        with self.assertRaises(RuntimeError):
            self.sync(self.db)
        self.assertIn("Future", os.environ["CABLE_POLICY_SYNC_FAILED"])
        self.apply.assert_not_called()

    def test_complete_matching_evidence_is_summarized(self):
        rows = [{"sim_time_s": i / 30, "command_stamp_s": i / 30, "wait_wall_s": 0.02}
                for i in range(3)]
        self.assertEqual(summarize_policy_sync(rows, 3),
                         {"steps": 3, "policy_hz": 30, "maximum_wait_wall_s": 0.02})

    def test_incomplete_stale_skipped_and_nonfinite_evidence_is_rejected(self):
        for stamps, commands, waits in [([], [], []), ([0], [0.1], [0.01]),
                                       ([0, 2 / 30], [0, 2 / 30], [0.01, 0.01]),
                                       ([0], [0], [float("nan")])]:
            with self.subTest(stamps=stamps, commands=commands, waits=waits):
                rows = [dict(sim_time_s=s, command_stamp_s=c, wait_wall_s=w)
                        for s, c, w in zip(stamps, commands, waits)]
                with self.assertRaises(ValueError):
                    summarize_policy_sync(rows, 1)
