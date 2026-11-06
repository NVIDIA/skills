# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline regression checks for paused startup and release readiness."""

import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_runtime
import preflight


class RuntimeReadiness(unittest.TestCase):
    def check(self, states, nodes=None):
        clocks = iter((1, 2))
        nodes = nodes or ["/planner_server", "/controller_server", "/bt_navigator"]

        def run(command, timeout, *, stdout_only=False):
            if command[1:3] == ["topic", "list"]:
                return 0, "/clock\n/cmd_vel\n/tf\n/tf_static\n/joint_states\n/imu"
            if command[1:3] == ["topic", "echo"]:
                return 0, f"sec: {next(clocks)}\nnanosec: 0"
            if command[1:3] == ["lifecycle", "nodes"]:
                return 0, "\n".join(nodes)
            if command[1:3] == ["lifecycle", "get"]:
                return states.get(command[-1], (0, "active [3]"))
            return 0, "Translation: [0, 0, 0]"

        with patch.object(sys, "argv", ["check_runtime"]), patch.object(
                check_runtime.shutil, "which", return_value="/usr/bin/ros2"), patch.object(
                check_runtime, "run", side_effect=run), patch.object(
                check_runtime.time, "sleep"), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return check_runtime.main()

    def test_active_id_three_passes(self):
        self.assertEqual(self.check({}), 0)

    def test_inactive_wrong_id_and_failed_query_are_rejected(self):
        for state in [(0, "inactive [2]"), (0, "active [2]"), (0, "not active [3]"),
                      (1, "active [3]"), (124, "active [3]"), (0, ""),
                      (0, "active [3]\ninactive [2]"), (0, "active [3]\nactive [3]")]:
            with self.subTest(state=state):
                self.assertEqual(self.check({"/planner_server": state}), 1)

    def test_successful_state_query_uses_stdout_despite_ros_warnings(self):
        result = types.SimpleNamespace(returncode=0, stdout="active [3]\n",
                                       stderr="[WARN] ROS_LOCALHOST_ONLY is deprecated\n")
        with patch.object(check_runtime.subprocess, "run", return_value=result):
            self.assertEqual(check_runtime.run(["ros2", "lifecycle", "get"], 1,
                                              stdout_only=True), (0, "active [3]"))

    def test_failed_state_query_keeps_stderr(self):
        result = types.SimpleNamespace(returncode=1, stdout="active [3]\n",
                                       stderr="Node query failed\n")
        with patch.object(check_runtime.subprocess, "run", return_value=result):
            code, output = check_runtime.run(["ros2", "lifecycle", "get"], 1,
                                             stdout_only=True)
        self.assertEqual(code, 1)
        self.assertIn("Node query failed", output)

    def test_duplicate_or_similarly_named_nodes_are_rejected(self):
        for nodes in [["/planner_server", "/other/planner_server", "/controller_server",
                       "/bt_navigator"], ["/not_planner_server", "/controller_server",
                                            "/bt_navigator"]]:
            with self.subTest(nodes=nodes):
                self.assertEqual(self.check({}, nodes), 1)


class PausedPreparation(unittest.TestCase):
    def setUp(self):
        self.omni = types.ModuleType("omni")
        self.omni.__path__ = []
        self.omni.usd = MagicMock()
        self.omni.timeline = MagicMock()
        self.omni.kit = types.ModuleType("omni.kit")
        self.omni.kit.__path__ = []
        self.omni.kit.app = MagicMock()
        self.omni.ext = types.ModuleType("omni.ext")
        self.omni.ext.IExt = object
        self.modules = {
            "omni": self.omni, "omni.usd": self.omni.usd,
            "omni.timeline": self.omni.timeline, "omni.kit": self.omni.kit,
            "omni.kit.app": self.omni.kit.app, "omni.ext": self.omni.ext,
        }
        with patch.dict(sys.modules, self.modules):
            self.library = runpy.run_path(str(ROOT / "scripts/prepare_simulation.py"),
                                         init_globals={"_PREPARATION_IMPORT_ONLY": True})
        self.context = self.omni.usd.get_context.return_value
        self.context.get_stage_loading_status.return_value = (0, 0, 0)
        self.stage = self.context.get_stage.return_value
        self.stage.GetRootLayer.return_value.identifier = self.library["SIM_STAGE"]
        self.attributes = {}
        prims = []
        for i in range(5):
            prim = MagicMock()
            prim.GetPath.return_value = f"/World/ROSContext{i}"
            for name, value in [("inputs:useDomainIDEnvVar", False), ("inputs:domain_id", 0)]:
                attr = MagicMock()
                attr.Get.return_value = value
                attr.Set.side_effect = lambda value, attr=attr: setattr(attr.Get, "return_value", value)
                self.attributes[(i, name)] = attr
            prim.GetAttribute.side_effect = lambda name, i=i: self.attributes[(i, name)]
            prims.append(prim)
        self.stage.Traverse.return_value = prims
        self.timeline = self.omni.timeline.get_timeline_interface.return_value
        self.timeline.is_playing.return_value = False
        self.timeline.get_end_time.return_value = 20.0
        self.timeline.is_looping.return_value = True
        self.timeline.set_end_time.side_effect = lambda value: setattr(
            self.timeline.get_end_time, "return_value", value)
        self.timeline.set_looping.side_effect = lambda value: setattr(
            self.timeline.is_looping, "return_value", value)
        self.environment = {
            "ROS_DOMAIN_ID": "73", "ROS_LOCALHOST_ONLY": "1",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        }

    def prepare(self):
        return self.library["prepare"](self.context, self.timeline, self.environment)

    def assert_no_mutation(self):
        self.timeline.set_end_time.assert_not_called()
        self.timeline.set_looping.assert_not_called()
        self.timeline.play.assert_not_called()
        for attr in self.attributes.values():
            attr.Set.assert_not_called()

    def test_domains_range_and_loop_are_read_back_without_play_or_save(self):
        result = self.prepare()
        self.assertEqual(result["domain"], 73)
        self.assertEqual(len(result["ros_contexts"]), 5)
        self.assertEqual(result["end_time"], 86400.0)
        self.assertIs(result["looping"], False)
        self.assertIs(result["playing"], False)
        self.timeline.play.assert_not_called()
        self.stage.Save.assert_not_called()

    def test_playing_loading_wrong_or_missing_stage_cannot_be_prepared(self):
        self.timeline.is_playing.return_value = True
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.timeline.is_playing.return_value = False
        self.context.get_stage_loading_status.return_value = (0, 0, 1)
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.context.get_stage_loading_status.return_value = (0, 0, 0)
        self.stage.GetRootLayer.return_value.identifier = "/other/g1_ros_navigation.usd"
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.context.get_stage.return_value = None
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assert_no_mutation()

    def test_invalid_discovery_and_missing_contexts_cannot_be_prepared(self):
        valid = self.environment.copy()
        for key, value in [("ROS_DOMAIN_ID", "0"), ("ROS_DOMAIN_ID", "233"),
                           ("ROS_LOCALHOST_ONLY", "0"), ("ROS_STATIC_PEERS", "peer")]:
            self.environment = {**valid, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
        self.environment = valid
        self.stage.Traverse.return_value = []
        with self.assertRaisesRegex(RuntimeError, "no authored"):
            self.prepare()
        self.assert_no_mutation()

    def test_failed_domain_and_timeline_readback_block_startup(self):
        attr = self.attributes[(0, "inputs:domain_id")]
        attr.Set.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "domain readback"):
            self.prepare()
        self.timeline.play.assert_not_called()
        attr.Get.return_value = 73
        self.timeline.set_looping.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "range/loop readback"):
            self.prepare()
        self.timeline.play.assert_not_called()

    def test_timeline_changes_are_committed_before_readback(self):
        pending = {}
        self.timeline.set_end_time.side_effect = lambda value: pending.update(end_time=value)
        self.timeline.set_looping.side_effect = lambda value: pending.update(looping=value)

        def commit():
            self.timeline.get_end_time.return_value = pending["end_time"]
            self.timeline.is_looping.return_value = pending["looping"]

        self.timeline.commit.side_effect = commit
        result = self.prepare()
        self.assertEqual(result["end_time"], 86400.0)
        self.assertIs(result["looping"], False)
        self.timeline.commit.assert_called_once_with()
        self.timeline.play.assert_not_called()

    def test_file_executes_in_remote_namespace_without_main_name(self):
        with patch.dict(sys.modules, self.modules), patch.dict(
                "os.environ", self.environment, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT / "scripts/prepare_simulation.py"), run_name="remote.scene")
        self.assertIn('"playing": false', output.getvalue())
        self.timeline.play.assert_not_called()

    def test_extension_prepares_without_starting_physics(self):
        path = ROOT / ("extensions/humanoid_cloud_control.autoplay/"
                       "humanoid_cloud_control/autoplay/extension.py")
        with patch.dict(sys.modules, self.modules), patch.dict(
                "os.environ", self.environment, clear=True):
            spec = importlib.util.spec_from_file_location("autoplay_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            asyncio.run(module.Extension()._prepare_when_ready())
        self.assertEqual(self.timeline.get_end_time(), 86400.0)
        self.timeline.play.assert_not_called()


class HostPreflight(unittest.TestCase):
    def check(self, *, os_id="ubuntu", os_version="24.04", sim_version="6.0.1",
              encoding=False, encoded=True):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for marker in ["isaac-sim.sh", "isaac-sim.streaming.sh", "Dockerfile",
                           "exts/isaacsim.code_editor.python_server/config/extension.toml"]:
                path = directory / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (directory / "VERSION").write_text(sim_version)
            (directory / "exts/isaacsim.ros2.core/jazzy/lib").mkdir(parents=True)
            (directory / "src/isaac_ros_deploy/isaac_deploy/"
             "isaac_ros_deploy_isaac_sim_extension").mkdir(parents=True)
            commands = []

            def probe(command):
                commands.append(command)
                if command[0] == "/usr/bin/nvidia-smi":
                    return True, "L40, 595.58.03, 8.9, 46068"
                if command[0] == "/usr/bin/ffmpeg":
                    return encoded, "" if encoded else "encoder unavailable"
                return True, "1.0"

            argv = ["preflight", "--no-discover", "--isaac-path", tmp,
                    "--isaac-ros-ws", tmp, "--web-viewer-context", tmp,
                    "--map", str(ROOT / "assets/carter_warehouse_navigation.yaml")]
            if encoding:
                argv.append("--check-encoding")
            with patch.object(sys, "argv", argv), patch.object(
                    preflight.platform, "machine", return_value="x86_64"), patch.object(
                    preflight.platform, "freedesktop_os_release",
                    return_value={"ID": os_id, "VERSION_ID": os_version}), patch.object(
                    preflight.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), \
                    patch.object(preflight, "command_output", side_effect=probe), patch.object(
                    preflight.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
                    patch.object(preflight, "nvenc_library_available", return_value=True), \
                    patch.object(preflight.importlib.util, "find_spec", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()) as output, \
                    contextlib.redirect_stderr(io.StringIO()):
                code = preflight.main()
            return code, output.getvalue(), commands

    def test_supported_host_distinguishes_library_from_media(self):
        code, output, commands = self.check()
        self.assertEqual(code, 0, output)
        self.assertIn("encode capability untested", output)
        self.assertIn("browser media and runtime readiness remain separate gates", output)
        self.assertFalse(any(command[0].endswith("ffmpeg") for command in commands))

    def test_wrong_os_or_sim_release_fails(self):
        for options in [{"os_id": "debian"}, {"os_version": "22.04"},
                        {"sim_version": "6.1.0"}, {"sim_version": "unknown"}]:
            with self.subTest(options=options):
                code, output, _ = self.check(**options)
                self.assertEqual(code, 1, output)

    def test_requested_encoding_requires_a_successful_probe(self):
        for encoded in [True, False]:
            with self.subTest(encoded=encoded):
                code, output, commands = self.check(encoding=True, encoded=encoded)
                self.assertEqual(code, 0 if encoded else 1, output)
                self.assertTrue(any("h264_nvenc" in command for command in commands))


class GPURequirements(unittest.TestCase):
    def test_supported_driver_capability_and_memory(self):
        self.assertEqual(preflight.gpu_requirements("L40, 595.58.03, 8.9, 46068"), [])
        self.assertEqual(preflight.gpu_requirements("GPU, 600.1, 9.0, 8192"), [])

    def test_old_driver_architecture_small_memory_and_missing_data_fail(self):
        for value in ["GPU, 535.161, 8.9, 46068", "GPU, 595.58, 7.5, 16384",
                      "GPU, 595.58, 8.9, 4096", "", "GPU, N/A, 8.9, 46068",
                      "GPU, 595.58, 8.9, NaN", "GPU, 595.58"]:
            with self.subTest(value=value):
                self.assertTrue(preflight.gpu_requirements(value))


if __name__ == "__main__":
    unittest.main()
