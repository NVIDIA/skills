# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for buffered trainers, video preflight, and truthful timeouts."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_checks  # noqa: E402
import supervisor  # noqa: E402


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        logs = self.root / "logs"
        logs.mkdir()
        patcher = patch.object(supervisor, "LOGS", logs)
        patcher.start()
        self.addCleanup(patcher.stop)
        supervisor.record_eula_acceptance("explicit_user_reply")
        spec = {"id": "baseline", "purpose": "smoke", "max_iterations": 4,
                "num_steps_per_env": 64, "video": False}
        raw = {"schema_version": 1, "training_root": str(self.root), "runs": [spec],
               "evaluation": {"enabled": False}, "timeouts": {"startup_seconds": 1, "stall_seconds": 60}}
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(raw))
        self.session = supervisor.create_session(plan_file)
        self.plan, self.state = supervisor.session_state(self.session)
        self.console = self.session / "train.log"
        self.console.touch()
        self.attempt = {"run_id": "baseline", "run_name": "unique_attempt", "spec": self.plan["runs"][0],
                        "status": "running", "console": "train.log", "started_epoch": time.time() - 5,
                        "last_iteration": None, "last_progress_epoch": time.time() - 5,
                        "pid": 123, "process_birth": "123"}
        self.state["attempts"] = [self.attempt]
        self.state["current"] = self.attempt
        self.run_dir = Path(self.plan["output_root"]) / "2026_unique_attempt"
        self.run_dir.mkdir(parents=True)

    def evidence(self, step=0):
        return {"step": step, "epoch": time.time(), "source": "tensorboard", "metrics": {}}

    def test_buffered_stdout_with_tensorboard_survives_startup_deadline(self):
        process = Mock()
        process.poll.side_effect = [None, None, 0]
        process.wait.return_value = 0
        with patch.object(supervisor, "runtime_check", side_effect=[self.evidence(0), self.evidence(1)]), \
             patch.object(supervisor, "process_alive", return_value=True), \
             patch.object(supervisor.time, "sleep"), \
             patch.object(supervisor, "terminate_group") as terminate:
            supervisor.monitor(self.session, self.plan, self.state, self.attempt, process)
        terminate.assert_not_called()
        self.assertEqual(self.console.read_text(), "")
        self.assertEqual(self.attempt["last_iteration"], 1)
        self.assertEqual(self.attempt["additional_updates_completed"], 2)
        self.assertNotIn("stop_reason", self.attempt)
        self.assertIn("update 1 via tensorboard", (self.session / "report.md").read_text())

    def test_startup_timeout_refreshes_artifacts_before_stopping(self):
        process = Mock()
        process.poll.side_effect = [None, None, 0]
        process.wait.return_value = 0
        with patch.object(supervisor, "runtime_check", side_effect=[{}, self.evidence(0), self.evidence(1)]), \
             patch.object(supervisor, "process_alive", return_value=True), \
             patch.object(supervisor.time, "sleep"), \
             patch.object(supervisor, "terminate_group") as terminate:
            supervisor.monitor(self.session, self.plan, self.state, self.attempt, process)
        terminate.assert_not_called()
        self.assertEqual(self.attempt["last_iteration"], 1)

    def test_genuinely_unobserved_startup_is_bounded_and_identifies_supervisor(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = -15
        with patch.object(supervisor, "runtime_check", return_value={}), \
             patch.object(supervisor, "process_alive", return_value=True), \
             patch.object(supervisor, "terminate_group") as terminate:
            supervisor.monitor(self.session, self.plan, self.state, self.attempt, process)
        terminate.assert_called_once()
        self.assertEqual(self.attempt["stop_origin"], "supervisor")
        self.assertEqual(self.attempt["stop_kind"], "startup_observation_timeout")
        self.assertIn("no training progress observed", self.attempt["stop_reason"])
        self.assertEqual(self.attempt["additional_updates_completed"], 0)

    def test_stalled_tensorboard_step_does_not_keep_run_alive(self):
        self.attempt.update(last_iteration=1, last_progress_epoch=time.time() - 100)
        self.attempt["progress_evidence"] = {**self.evidence(1), "epoch": time.time() - 100}
        with patch.object(supervisor, "runtime_check", return_value=self.evidence(1)):
            supervisor.refresh_progress(self.session, self.plan, self.state, self.attempt, force=True)
        self.assertEqual(supervisor.monitor_stop_reason(self.plan, self.state, self.attempt)[0], "progress_timeout")
        # The same evidence cannot reset the timer after a supervisor restart either.
        saved = json.loads(json.dumps(self.attempt))
        self.assertEqual(supervisor.monitor_stop_reason(self.plan, self.state, saved)[0], "progress_timeout")

    def test_budget_applies_even_with_fresh_progress(self):
        self.plan["training_budget_seconds"] = 1
        self.attempt.update(last_iteration=1, last_progress_epoch=time.time())
        self.assertEqual(supervisor.monitor_stop_reason(self.plan, self.state, self.attempt)[0], "budget")

    def test_console_progress_survives_broken_tensorboard(self):
        self.console.write_text("Learning iteration 0/4\n")
        with patch.object(supervisor, "runtime_check", side_effect=RuntimeError("partial event file")):
            supervisor.refresh_progress(self.session, self.plan, self.state, self.attempt, force=True)
        self.assertEqual(self.attempt["last_iteration"], 0)
        self.assertIn("partial event", self.attempt["progress_probe_error"])
        self.assertIsNone(supervisor.monitor_stop_reason(self.plan, self.state, self.attempt))

    def test_new_checkpoint_must_be_stable_and_old_or_other_run_files_are_ignored(self):
        old = self.run_dir / "model_900.pt"
        old.write_bytes(b"old")
        os.utime(old, (1, 1))
        unrelated = self.run_dir.parent / "2026_other_attempt"
        unrelated.mkdir()
        (unrelated / "model_999.pt").write_bytes(b"unrelated")
        checkpoint = self.run_dir / "model_0.pt"
        checkpoint.write_bytes(b"new checkpoint")
        with patch.object(supervisor, "runtime_check", side_effect=RuntimeError("unreadable events")):
            supervisor.refresh_progress(self.session, self.plan, self.state, self.attempt, force=True)
            self.assertIsNone(self.attempt["last_iteration"])
            supervisor.refresh_progress(self.session, self.plan, self.state, self.attempt, force=True)
        self.assertEqual(self.attempt["last_iteration"], 0)
        self.assertEqual(self.attempt["progress_evidence"]["source"], "checkpoint")
        self.assertIsNone(supervisor.monitor_stop_reason(self.plan, self.state, self.attempt))

    def test_ambiguous_output_directories_are_not_used(self):
        (self.run_dir.parent / "2027_unique_attempt").mkdir()
        with patch.object(supervisor, "runtime_check") as probe:
            supervisor.refresh_progress(self.session, self.plan, self.state, self.attempt, force=True)
        probe.assert_not_called()
        self.assertIsNone(self.attempt["last_iteration"])
        self.assertIn("Multiple output", self.attempt["progress_probe_error"])

    def test_video_command_is_explicit_and_headless_unbuffered(self):
        spec = {**self.attempt["spec"], "run_name": "test", "video": True}
        with patch.dict(os.environ, {"PYTHONUNBUFFERED": "0", "HEADLESS": "0"}):
            command, environment = supervisor.run_command(spec, self.plan, self.session)
        self.assertEqual(command[command.index("--viz") + 1], "kit")
        self.assertEqual(environment["HEADLESS"], "1")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
        command, _ = supervisor.run_command({**spec, "video": False}, self.plan, self.session)
        self.assertNotIn("--video", command)
        self.assertNotIn("--viz", command)

    def test_runtime_checks_are_bounded_in_training_environment(self):
        with patch.object(supervisor.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, '{"tensorboard":true,"video":false}', "")) as run:
            result = supervisor.runtime_check(self.plan, ["preflight"])
        self.assertTrue(result["tensorboard"])
        self.assertEqual(run.call_args.args[0][0], str(self.root / ".venv/bin/python"))
        self.assertEqual(run.call_args.kwargs["cwd"], str(self.root))
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_missing_video_dependencies_block_before_trainer_launch(self):
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        (self.root / ".git").mkdir()
        self.state["current"] = None
        self.state["attempts"] = []
        self.state["queue"][0]["video"] = True  # A queued retry may enable video.
        supervisor.save_state(self.session, self.state)
        old_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with patch.object(supervisor.shutil, "which", return_value="/bin/tool"), \
                 patch.object(supervisor, "runtime_check", side_effect=RuntimeError("No module named moviepy")) as probe, \
                 patch.object(supervisor.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "preflight failed before trainer launch"):
                    supervisor.campaign(self.session)
            launch.assert_not_called()
            self.assertIn("--video", probe.call_args.args[1])
        finally:
            os.chdir(old_cwd)
        state = supervisor.session_state(self.session)[1]
        self.assertEqual(state["failure_phase"], "preflight")
        self.assertEqual(state["attempts"], [])
        self.assertIn("Preflight failed", (self.session / "report.md").read_text())

    def test_no_video_preflight_does_not_import_video_packages(self):
        with patch.object(runtime_checks.importlib, "import_module") as importer, \
             patch.dict(sys.modules, {"moviepy": None, "imageio": None}):
            self.assertFalse(runtime_checks.preflight(False)["video"])
        importer.assert_called_once_with("tensorboard.backend.event_processing.event_accumulator")

    def test_preflight_cli_checks_environment_without_creating_or_launching_session(self):
        output = io.StringIO()
        command = ["supervisor.py", "preflight", "--plan", str(self.root / "plan.json")]
        with patch.object(sys, "argv", command), redirect_stdout(output), \
             patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}) as check, \
             patch.object(supervisor, "create_session", side_effect=AssertionError("created a session")), \
             patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("launched training")):
            self.assertEqual(supervisor.main(), 0)
        self.assertTrue(check.call_args.args[0]["eula_accepted"])
        self.assertEqual(json.loads(output.getvalue())["training_ref"], "release-r1")

    def test_preflight_cli_refuses_unaccepted_eula_before_import_checks(self):
        supervisor.eula_record_path().unlink()
        command = ["supervisor.py", "preflight", "--plan", str(self.root / "plan.json")]
        with patch.object(sys, "argv", command), redirect_stderr(io.StringIO()), \
             patch.object(supervisor, "check_preflight", side_effect=AssertionError("checked environment")):
            self.assertEqual(supervisor.main(), 2)

    def test_preflight_cli_preserves_actionable_failure(self):
        output = io.StringIO()
        command = ["supervisor.py", "preflight", "--plan", str(self.root / "plan.json")]
        with patch.object(sys, "argv", command), redirect_stderr(output), \
             patch.object(supervisor, "check_preflight", side_effect=RuntimeError("video encoder unavailable")):
            self.assertEqual(supervisor.main(), 2)
        self.assertIn("video encoder unavailable", output.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard integration needs tensorboard installed")
    def test_real_buffered_child_and_tensorboard_probe(self):
        (self.root / ".venv").symlink_to(sys.prefix, target_is_directory=True)
        ready = self.root / "ready"
        script = """
import sys, time
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
writer = EventFileWriter(sys.argv[1])
print('Learning iteration 0/4')  # Intentionally buffered in a pipe/file.
for step in (0, 1):
    writer.add_event(Event(wall_time=time.time(), step=step,
        summary=Summary(value=[Summary.Value(tag='Loss/value_function', simple_value=0.1)])))
writer.flush()
Path(sys.argv[2]).touch()
time.sleep(2)
writer.close()
"""
        environment = {"PATH": os.defpath, "HOME": str(self.root)}
        with self.console.open("wb") as stream:
            process = subprocess.Popen([sys.executable, "-c", script, str(self.run_dir), str(ready)],
                                       stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), self.console.read_text())
                self.assertNotIn("Learning iteration", self.console.read_text())
                self.attempt.update(pid=process.pid, process_birth=supervisor.process_birth(process.pid))
                real_sleep = time.sleep
                with patch.object(supervisor.time, "sleep", side_effect=lambda _: real_sleep(0.05)):
                    supervisor.monitor(self.session, self.plan, self.state, self.attempt, process)
                self.assertEqual(process.returncode, 0)
                self.assertNotIn("stop_reason", self.attempt)
                self.assertEqual(self.attempt["last_iteration"], 1)
                self.assertEqual(self.attempt["progress_evidence"]["source"], "tensorboard")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard integration needs tensorboard installed")
    def test_real_event_file_progress_and_partial_tail(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        writer = EventFileWriter(str(self.run_dir))
        for step in (0, 1):
            writer.add_event(Event(wall_time=time.time(), step=step,
                                   summary=Summary(value=[Summary.Value(tag="Loss/value_function", simple_value=0.2)])))
        writer.flush()
        writer.close()
        event_file = next(self.run_dir.glob("events.out.tfevents.*"))
        with event_file.open("ab") as stream:
            stream.write(b"partial")
        value = runtime_checks.tensorboard_progress(self.run_dir, self.attempt["started_epoch"], 0)
        self.assertEqual(value["step"], 1)
        self.assertEqual(value["source"], "tensorboard")
        self.assertEqual(runtime_checks.tensorboard_progress(self.run_dir, time.time() + 1, 0), {})
        self.assertEqual(runtime_checks.tensorboard_progress(self.run_dir, self.attempt["started_epoch"], 749), {})


if __name__ == "__main__":
    unittest.main()
