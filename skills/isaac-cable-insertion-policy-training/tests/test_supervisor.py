# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast process and recovery tests for the reusable supervisor."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supervisor  # noqa: E402
from storage import completed_result, records  # noqa: E402


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "isaac_ros_manipulation/isaaclab_training"
        self.root.mkdir(parents=True)
        (self.root / "pyproject.toml").write_text("# test project\n")
        (self.root / "uv.lock").write_text("# test lock\n")
        source = self.root / "src/isaaclab_training/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("# test task\n")
        self.original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.original_cwd)
        self.logs = Path(self.temp.name) / "logs"
        self.logs.mkdir()
        patcher = patch.object(supervisor, "LOGS", self.logs)
        patcher.start()
        self.addCleanup(patcher.stop)
        supervisor.record_eula_acceptance("explicit_user_reply")

    def plan(self, **changes: object) -> dict:
        value = {
            "schema_version": 1,
            "training_root": str(self.root),
            "max_completed_runs": 1,
            "decision_mode": "preplanned",
            "evaluation": {"enabled": False},
            "runs": [{
                "id": "baseline", "purpose": "smoke", "max_iterations": 1,
                "num_steps_per_env": 64, "video": False,
            }],
        }
        value.update(changes)
        return value

    def create(self, plan: dict | None = None) -> Path:
        path = Path(self.temp.name) / "plan.json"
        path.write_text(json.dumps(plan or self.plan()))
        return supervisor.create_session(path)

    def record_provenance(self, session, attempt):
        attempt["training_sha256"] = supervisor.training_fingerprint(self.root)
        params = supervisor.run_folder(session, attempt) / "params"
        params.mkdir(parents=True, exist_ok=True)
        attempt["resolved_configs"] = []
        for name in ("env.yaml", "agent.yaml"):
            path = params / name
            path.write_text("fixture: true\n")
            attempt["resolved_configs"].append({**supervisor.fingerprint(path),
                                                "path": str(path.relative_to(session))})

    @staticmethod
    def verified(session: Path, plan: dict, state: dict, attempt: dict) -> None:
        attempt["verification"] = {
            "checkpoint": str(session / "fake.pt"), "videos": [], "scalar_count": 1,
        }
        attempt["status"] = "verified" if attempt["exit_code"] == 0 else "artifact_verified"
        supervisor.emit(session, state, "artifacts_verified", run_id=attempt["run_id"])

    def test_remote_identity_strips_scp_and_url_credentials(self):
        for remote in ("git@example.com:organization/repository.git",
                       "ssh://git@example.com/organization/repository.git"):
            self.assertEqual(supervisor.safe_remote(remote), "ssh://example.com/organization/repository.git")
        self.assertEqual(supervisor.safe_remote("https://user:password@example.com/repo.git?token=private"),
                         "https://example.com/repo.git")
        self.assertEqual(supervisor.safe_remote("/work/local/repository"), "/work/local/repository")

    def test_log_directory_settings_cannot_diverge_from_trainer(self):
        with self.assertRaisesRegex(ValueError, "custom output_root"):
            supervisor.normalize_plan(self.plan(output_root=str(self.root / "custom")))
        for field in ("agent.experiment_name", "agent.run_name"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "supervisor owns"):
                supervisor.normalize_run({"id": "baseline", "overrides": {field: "custom"}})
        plan = supervisor.normalize_plan(self.plan())
        command, _ = supervisor.run_command({**plan["runs"][0], "run_name": "unique"}, plan, self.logs)
        experiment = command[command.index("--experiment_name") + 1]
        self.assertEqual(Path(plan["output_root"]), self.root / "logs/rsl_rl" / experiment)

    def test_start_end_video_schedule_and_curriculum_match(self) -> None:
        run = supervisor.normalize_run({
            "id": "fresh", "max_iterations": 3000, "num_steps_per_env": 64,
            "video_schedule": "start_end",
        })
        self.assertEqual(run["video_interval"], 191800)
        self.assertEqual(run["overrides"]["env.events.reset_plug_curriculum.params.num_steps_per_env"], 64)
        with self.assertRaisesRegex(ValueError, "curriculum num_steps_per_env"):
            supervisor.normalize_run({
                "id": "wrong", "max_iterations": 3000, "num_steps_per_env": 64,
                "overrides": {"env.events.reset_plug_curriculum.params.num_steps_per_env": 512},
            })
        range_run = supervisor.normalize_run({
            "id": "depth", "max_iterations": 3000, "num_steps_per_env": 64,
            "overrides": {"env.events.reset_plug_curriculum.params.approach_depth_range": [0.02, 0.06]},
        })
        plan = supervisor.normalize_plan(self.plan())
        command, _ = supervisor.run_command({**range_run, "run_name": "depth"}, plan, self.root)
        self.assertIn("env.events.reset_plug_curriculum.params.approach_depth_range=[0.02,0.06]", command)

    def test_standalone_release_task_defaults_and_minibatches(self) -> None:
        taskspace = supervisor.normalize_run({"id": "default"})
        self.assertEqual((taskspace["num_steps_per_env"], taskspace["max_iterations"]), (128, 1000))
        self.assertEqual(taskspace["video_interval"], 19200)
        anneal_key = "env.events.reset_plug_curriculum.params.anneal_end_iter"
        self.assertEqual(taskspace["overrides"][anneal_key], 333)
        joint = supervisor.normalize_run({"id": "joint"}, task="IsaacTraining-DisplayPortInsertion-Rizon4s-ROS-Inference")
        self.assertEqual((joint["num_steps_per_env"], joint["max_iterations"]), (512, 1500))
        self.assertEqual(joint["overrides"][anneal_key], 500)
        with self.assertRaisesRegex(ValueError, "anneal_end_iter"):
            supervisor.normalize_run({"id": "short", "max_iterations": 332})
        with self.assertRaisesRegex(ValueError, "num_mini_batches"):
            supervisor.normalize_run({"id": "tiny", "num_envs": 4})
        small = supervisor.normalize_run({"id": "tiny", "num_envs": 4,
                                          "overrides": {"agent.algorithm.num_mini_batches": 4}})
        self.assertEqual(small["num_envs"], 4)

    def test_release_curriculum_guard_matches_launched_command(self) -> None:
        key = "env.events.reset_plug_curriculum.params.anneal_end_iter"
        plan = supervisor.normalize_plan(self.plan())
        for iterations in (333, 1000):
            with self.subTest(iterations=iterations):
                run = supervisor.normalize_run({"id": "release", "max_iterations": iterations})
                command, _ = supervisor.run_command({**run, "run_name": "release"}, plan, self.root)
                self.assertIn(f"{key}=333", command)
                self.assertEqual(command[command.index("--max_iterations") + 1], str(iterations))
        raw = {"id": "short", "max_iterations": 100, "overrides": {key: 100}}
        run = supervisor.normalize_run(raw)
        command, _ = supervisor.run_command({**run, "run_name": "short"}, plan, self.root)
        self.assertIn(f"{key}=100", command)
        self.assertEqual(raw["overrides"], {key: 100})
        with self.assertRaisesRegex(ValueError, "anneal_end_iter"):
            supervisor.normalize_run({"id": "unfinished", "max_iterations": 1000,
                                      "overrides": {key: 2000}})
        smoke = supervisor.normalize_run({"id": "smoke", "purpose": "smoke", "max_iterations": 3})
        self.assertEqual(smoke["overrides"][key], 333)

    def test_example_plan_uses_current_release_defaults(self) -> None:
        example = json.loads((supervisor.SKILL / "assets/session-plan.example.json").read_text())
        example["training_root"] = str(self.root)
        run = supervisor.normalize_plan(example)["runs"][0]
        self.assertEqual(run["max_iterations"], 1000)
        self.assertEqual(run["num_steps_per_env"], 128)
        self.assertEqual(run["overrides"]["env.events.reset_plug_curriculum.params.anneal_end_iter"], 333)
        self.assertEqual(run["video_interval"], 127800)

    def test_old_source_plan_rejected_and_old_session_remains_read_only(self) -> None:
        old = self.plan()
        old["isaaclab_root"] = old.pop("training_root")
        with self.assertRaisesRegex(ValueError, "source-IsaacLab plans"):
            supervisor.normalize_plan(old)
        session = self.create()
        path = session / "session.json"
        value = json.loads(path.read_text())
        value["plan"]["isaaclab_root"] = value["plan"].pop("training_root")
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        with patch.object(sys, "argv", ["supervisor.py", "status", "--session", str(session)]), redirect_stdout(io.StringIO()):
            self.assertEqual(supervisor.main(), 0)
        with patch.object(sys, "argv", ["supervisor.py", "resume", "--session", str(session)]), redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.main(), 2)
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            supervisor.campaign(session)
        self.assertEqual(path.read_bytes(), before)

    def test_checkpoint_continuation_uses_additional_updates_for_verification(self) -> None:
        checkpoint = Path(self.temp.name) / "parent.pt"
        checkpoint.touch()
        run = supervisor.normalize_run({
            "id": "continued", "purpose": "smoke", "max_iterations": 100,
            "num_steps_per_env": 64, "video": False,
            "parent_checkpoint": str(checkpoint),
        })
        plan = supervisor.normalize_plan(self.plan(runs=[run]))
        with patch.object(supervisor.subprocess, "run", return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="749\n", stderr="",
        )) as read_parent:
            start = supervisor.parent_iteration(run, plan)
        self.assertEqual(start, 749)
        command = read_parent.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "checkpoint.py")
        self.assertEqual(command[2], str(checkpoint))

        session = self.create(self.plan(runs=[run]))
        plan, state = supervisor.session_state(session)
        run_dir = Path(plan["output_root"]) / "20261005_continued"
        params = run_dir / "params"
        params.mkdir(parents=True)
        for name in ("agent.yaml", "env.yaml"):
            (params / name).write_text("seed: 42\n")
        record = session / "attempts" / "continued"
        record.mkdir(parents=True)
        console = record / "console.log"
        console.write_text("Learning iteration 848/849\n")
        attempt = {
            "run_id": "continued", "run_name": "continued", "spec": run,
            "start_iteration": start, "exit_code": 0, "console": str(console),
        }
        self.assertEqual(supervisor.final_iteration(attempt), 848)
        verification = {"checkpoint": str(run_dir / "model_848.pt"), "videos": [], "scalar_count": 100}
        with patch.object(supervisor.subprocess, "run", return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(verification), stderr="",
        )) as verify:
            supervisor.verify_attempt(session, plan, state, attempt)
        command = verify.call_args.args[0]
        self.assertEqual(command[command.index("--expected-step") + 1], "848")
        self.assertEqual(command[command.index("--iterations") + 1], "100")
        self.assertEqual(attempt["verification"], verification)

    def test_checkpoint_continuation_evaluates_shifted_final_step(self) -> None:
        checkpoint = Path(self.temp.name) / "parent.pt"
        checkpoint.touch()
        run = {
            "id": "continued", "max_iterations": 100, "num_steps_per_env": 64,
            "video": False, "parent_checkpoint": str(checkpoint),
            "overrides": {"env.events.reset_plug_curriculum.params.anneal_end_iter": 1},
        }
        session = self.create(self.plan(
            runs=[run], evaluation={"enabled": True, "seeds": [7], "num_envs": 2,
                                    "checkpoint_steps": ["final"]},
        ))
        plan, state = supervisor.session_state(session)
        record = session / "attempts" / "continued"
        record.mkdir(parents=True)
        run_dir = Path(self.temp.name) / "output"
        run_dir.mkdir()
        (run_dir / "model_848.pt").touch()
        attempt = {
            "run_id": "continued", "run_dir": str(run_dir),
            "spec": plan["runs"][0], "start_iteration": 749,
        }
        self.record_provenance(session, attempt)

        def write_result(_session, _plan, _state, _attempt, command, _console, _identity):
            output = Path(command[command.index("--journal") + 1])
            supervisor.append_record(output, {"kind": "result", "seed": 7,
                "evaluation_id": command[command.index("--evaluation-id") + 1],
                "terminal_success_count": 1, "num_envs": 2})
            return 0

        with patch.object(supervisor, "evaluation_process", side_effect=write_result) as evaluate:
            supervisor.evaluate_attempt(session, plan, state, attempt)
        command = evaluate.call_args.args[4]
        self.assertEqual(command[command.index("--checkpoint") + 1], str(run_dir / "model_848.pt"))
        self.assertEqual(attempt["evaluations"]["848"]["rate"], 0.5)

    def test_successful_run_writes_report_and_no_duplicate_on_resume(self) -> None:
        session = self.create()
        script = "print('Learning iteration 0/1', flush=True); print('Training time: 1 seconds', flush=True)"
        with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
             patch.object(supervisor, "run_command", return_value=([sys.executable, "-c", script], {"PATH": os.defpath})), \
             patch.object(supervisor, "verify_attempt", side_effect=self.verified):
            supervisor.campaign(session)
            supervisor.campaign(session)
        state = supervisor.session_state(session)[1]
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["completed_runs"], 1)
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(state["attempts"][0]["exit_code"], 0)
        self.assertFalse((session / "provenance.tar.gz").exists())
        self.assertIn("reporting.py", state["code_sha256"])
        self.assertFalse((session / "code").exists())
        self.assertFalse((session / "source").exists())
        self.assertNotIn("source_snapshots", state)
        self.assertIn("baseline", (session / "report.md").read_text())
        attempt = state["attempts"][0]
        self.assertRegex(attempt["record_dir"], r"^runs/\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC__r01__baseline$")
        self.assertEqual([path.name for path in supervisor.run_folder(session, attempt).iterdir()], ["train.log"])

    def test_failure_is_preserved_and_corrected_attempt_can_run(self) -> None:
        session = self.create(self.plan(max_completed_runs=2))
        failure = "print('CUDA out of memory', flush=True); raise SystemExit(7)"
        success = "print('Learning iteration 0/1', flush=True)"
        with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
             patch.object(supervisor, "run_command", return_value=([sys.executable, "-c", failure], {"PATH": os.defpath})), \
             patch.object(sys, "argv", ["supervisor.py", "run", "--session", str(session)]):
            self.assertEqual(supervisor.main(), 2)
        state = supervisor.session_state(session)[1]
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["attempts"][0]["exit_code"], 7)
        self.assertIn("reduce --num_envs", state["attempts"][0]["diagnosis"])
        spec = Path(self.temp.name) / "retry.json"
        spec.write_text(json.dumps({
            "id": "retry128", "purpose": "smoke", "max_iterations": 1,
            "num_steps_per_env": 64, "num_envs": 128, "video": False,
        }))
        supervisor.queue_run(session, spec)
        with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
             patch.object(supervisor, "run_command", return_value=([sys.executable, "-c", success], {"PATH": os.defpath})), \
             patch.object(supervisor, "verify_attempt", side_effect=self.verified):
            supervisor.campaign(session)
        state = supervisor.session_state(session)[1]
        self.assertEqual(state["status"], "complete")
        self.assertEqual([item["run_id"] for item in state["attempts"]], ["baseline", "retry128"])
        self.assertEqual(state["attempts"][1]["spec"]["retry_of"], "baseline")
        self.assertEqual([item["run_number"] for item in state["attempts"]], [1, 2])

    def test_foreground_launcher_propagates_blocked_campaign(self):
        session = self.create()
        with patch.object(supervisor, "campaign", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "campaign blocked"):
                supervisor.launcher(session, "foreground")

    def test_missing_exit_marker_adopts_artifacts_without_inventing_exit_zero(self) -> None:
        session = self.create()
        plan, state = supervisor.session_state(session)
        run = state["queue"].pop(0)
        record = session / "attempts" / run["id"]
        record.mkdir(parents=True)
        console = record / "console.log"
        console.write_text("Learning iteration 0/1\nTraining time: 1 seconds\n")
        attempt = {
            "run_id": run["id"], "run_name": "missing-marker", "spec": run,
            "status": "running", "started_utc": supervisor.utc(),
            "started_epoch": supervisor.time.time() - 1,
            "console": str(console), "command": [], "exit_code": None,
        }
        state["attempts"] = [attempt]
        state["current"] = attempt
        supervisor.save_state(session, state)
        with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
             patch.object(supervisor, "verify_attempt", side_effect=self.verified), \
             patch.object(supervisor, "run_command", side_effect=AssertionError("duplicate launch")):
            supervisor.campaign(session)
        state = supervisor.session_state(session)[1]
        self.assertEqual(state["status"], "complete")
        self.assertIsNone(state["attempts"][0]["exit_code"])
        self.assertIn("unavailable", (session / "report.md").read_text())

    def test_recovered_run_honors_manual_gate_and_preplanned_progress(self) -> None:
        """Recovery and repeated resume must not spend another manual training run."""
        for mode in ("manual", "preplanned"):
            for status in ("running", "verified", "complete"):
                with self.subTest(mode=mode, status=status):
                    runs = [{"id": name, "purpose": "smoke", "max_iterations": 1, "video": False}
                            for name in ("first", "second")]
                    session = self.create(self.plan(runs=runs, max_completed_runs=2, decision_mode=mode))
                    plan, state = supervisor.session_state(session)
                    spec = state["queue"].pop(0)
                    console = session / "train.log"
                    console.write_text("Learning iteration 0/1\nTraining time: 1 seconds\n")
                    attempt = {"run_id": "first", "run_name": "first", "spec": spec,
                               "status": status, "exit_code": None, "console": str(console),
                               "started_epoch": supervisor.time.time() - 1}
                    state.update(attempts=[attempt], current=attempt, status="running",
                                 completed_runs=int(status == "complete"))
                    supervisor.save_state(session, state)
                    command = [sys.executable, "-c", "print('Learning iteration 0/1')"]
                    with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
                         patch.object(supervisor, "verify_attempt", side_effect=self.verified), \
                         patch.object(supervisor, "run_command", return_value=(command, {"PATH": os.defpath})) as launch:
                        supervisor.campaign(session)
                        supervisor.campaign(session)
                    state = supervisor.session_state(session)[1]
                    self.assertEqual(state["status"], "decision_pending" if mode == "manual" else "complete")
                    self.assertEqual(state["completed_runs"], 1 if mode == "manual" else 2)
                    self.assertEqual(launch.call_count, 0 if mode == "manual" else 1)
                    self.assertIsNone(state["current"])
                    if mode == "manual":
                        approved = Path(self.temp.name) / "approved.json"
                        approved.write_text(json.dumps({"id": "approved", "purpose": "smoke", "max_iterations": 1, "video": False}))
                        supervisor.queue_run(session, approved)
                        with patch.object(supervisor, "check_preflight", return_value={"training_ref": "release-r1"}), \
                             patch.object(supervisor, "verify_attempt", side_effect=self.verified), \
                             patch.object(supervisor, "run_command", return_value=(command, {"PATH": os.defpath})):
                            supervisor.campaign(session)
                        self.assertEqual(supervisor.session_state(session)[1]["attempts"][-1]["run_id"], "approved")

    def test_preflight_checks_public_origin_and_keeps_missing_remote_provenance(self) -> None:
        """The package owns .venv; the manipulation repository owns Git metadata."""
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        source = self.root / "src/isaaclab_training/__init__.py"

        def command_output(command, **kwargs):
            if command == ["git", "remote", "get-url", "origin"]:
                if remote is None:
                    raise subprocess.CalledProcessError(2, command)
                return subprocess.CompletedProcess(command, 0, stdout=remote, stderr="")
            if command == ["git", "rev-parse", "--show-toplevel"]:
                result = str(self.root.parent)
            elif command == ["git", "branch", "--show-current"]:
                result = "release-r1"
            else:
                result = "checked"
            return subprocess.CompletedProcess(command, 0, stdout=result, stderr="")

        # Exercise both SSH spellings from one canonical public repository URL.
        ssh_remote = "ssh://git@github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git"
        ssh_url = urlsplit(ssh_remote)
        scp_remote = f"{ssh_url.netloc}:{ssh_url.path.lstrip('/')}"
        for remote, accepted in [
            (None, True),
            ("https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git", True),
            (scp_remote, True),
            (ssh_remote, True),
            ("https://github.com/other/isaac_ros_manipulation.git", False),
        ]:
            with self.subTest(remote=remote):
                with patch.object(supervisor.shutil, "which", return_value="/bin/tool"), \
                     patch.object(supervisor, "runtime_check", side_effect=[
                         {"tensorboard": True, "video": False},
                         {"isaac_sim_version": "6.1.0.0", "isaaclab_training_source": str(source)},
                         {"checked_distributions": 276, "project_override_exceptions": []},
                     ]), patch.object(supervisor.subprocess, "run", side_effect=command_output):
                    if not accepted:
                        with self.assertRaisesRegex(RuntimeError, "training repository origin"):
                            supervisor.check_preflight(supervisor.normalize_plan(self.plan()))
                        continue
                    result = supervisor.check_preflight(supervisor.normalize_plan(self.plan()))
                self.assertEqual(result["training_remote"], supervisor.safe_remote(remote) if remote else "")
                self.assertEqual(result["training_ref"], "release-r1")
                self.assertEqual(result["training_repository"], str(self.root.parent))
                self.assertEqual(result["isaaclab_training_source"], str(source))
                self.assertFalse((self.root / ".git").exists())

    def test_evaluator_unknown_run_exits_before_simulation_imports(self) -> None:
        """A mistyped run ID produces an argparse error without loading Isaac Sim."""
        session = self.create()
        result = subprocess.run([
            sys.executable, str(supervisor.SKILL / "scripts/evaluate.py"),
            "--checkpoint", str(self.root / "unused.pt"), "--output", str(self.root / "unused.json"),
            "--session", str(session), "--run-id", "unknown", "--seed", "7",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("run ID 'unknown' was not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "unused.json").exists())

    def test_tensorboard_failure_is_a_structured_failed_verification(self) -> None:
        """Keep artifact evidence available when TensorBoard refuses to read data."""
        import verify_run
        module = MagicMock()
        module.EventAccumulator.return_value.Reload.side_effect = OSError("unreadable event data")
        modules = {"tensorboard.backend.event_processing.event_accumulator": module,
                   "imageio": None, "imageio.v2": None}
        console = self.root / "train.log"
        console.write_text("Training time: 1 second")
        (self.root / "model_0.pt").write_bytes(b"checkpoint")
        with patch.dict(sys.modules, modules):
            result = verify_run.verify(self.root, console, 0, 1, 200, 0)
        self.assertFalse(result["artifact_complete"])
        self.assertIsNone(result["last_step"])
        self.assertEqual(result["scalar_count"], 0)
        self.assertIn("cannot read TensorBoard scalars", result["problems"][0])
        self.assertIn("unreadable event data", result["problems"][0])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_auto_launcher_uses_detached_when_tmux_is_absent(self) -> None:
        session = self.create()
        with patch.object(supervisor.shutil, "which", return_value=None), \
             patch.object(supervisor.subprocess, "Popen") as spawned:
            self.assertEqual(supervisor.launcher(session, "auto"), "detached")
        self.assertTrue(spawned.called)
        self.assertEqual(supervisor.session_state(session)[1]["launcher"]["type"], "detached")

    def test_training_environment_excludes_shell_credentials(self) -> None:
        with patch.dict(os.environ, {"GITLAB_TOKEN": "secret", "CUDA_VISIBLE_DEVICES": "0",
                                     "LD_PRELOAD": "test-preload-value",
                                     "ISAACSIM_ASSET_REGION_PROFILE": "test-region",
                                     "ISAACSIM_ASSET_ROOT": "/assets"}):
            environment = supervisor.training_environment(supervisor.normalize_plan(self.plan(eula_accepted=True)))
        self.assertNotIn("GITLAB_TOKEN", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["OMNI_KIT_ACCEPT_EULA"], "Yes")
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("ISAACSIM_ASSET_REGION_PROFILE", environment)
        self.assertNotIn("ISAACSIM_ASSET_ROOT", environment)

    def test_start_requires_record_even_if_plan_claims_acceptance(self) -> None:
        supervisor.eula_record_path().unlink()
        with self.assertRaisesRegex(RuntimeError, "EULA acceptance is required"):
            self.create(self.plan(eula_accepted=True))
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_acceptance_is_reused_and_a_changed_scope_requires_a_new_answer(self) -> None:
        receipt = supervisor.eula_record_path()
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        session = self.create()
        plan = supervisor.session_state(session)[0]
        self.assertTrue(plan["eula_accepted"])
        self.assertEqual(plan["eula_confirmation"]["accepted_at_utc"], supervisor.eula_record()["accepted_at_utc"])
        stale = supervisor.read_json(receipt)
        stale["scope"] = "isaac-sim-older"
        receipt.write_text(json.dumps(stale))
        self.assertIsNone(supervisor.eula_record())
        with self.assertRaisesRegex(RuntimeError, "EULA acceptance is required"):
            self.create()

    def test_cli_will_not_record_acceptance_without_user_confirmation(self) -> None:
        receipt = supervisor.eula_record_path()
        receipt.unlink()
        with patch.object(sys, "argv", ["supervisor.py", "eula-accept"]), \
             patch.object(sys, "stdin", io.StringIO()), \
             redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.main(), 2)
        self.assertFalse(receipt.exists())
        with patch.object(sys, "argv", ["supervisor.py", "eula-accept", "--user-confirmed"]), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(supervisor.main(), 0)
        self.assertEqual(supervisor.eula_record()["source"], "explicit_user_reply")

    def test_revoked_acceptance_blocks_resume_before_preflight(self) -> None:
        session = self.create()
        supervisor.eula_record_path().unlink()
        with patch.object(supervisor, "check_preflight", side_effect=AssertionError("preflight ran")):
            with self.assertRaisesRegex(RuntimeError, "EULA acceptance is required"):
                supervisor.campaign(session)
        self.assertEqual(supervisor.session_state(session)[1]["status"], "blocked")

    def test_readable_names_collision_and_immutable_plan(self) -> None:
        with patch.object(supervisor, "utc", return_value="2026-10-05T16:09:04+00:00"):
            first = self.create(self.plan(name="DisplayPort Target 95"))
            second = self.create(self.plan(name="DisplayPort Target 95"))
        self.assertEqual(first.name, "2026-10-05_16-09-04_UTC__displayport-target-95")
        self.assertEqual(second.name, first.name + "__02")
        plan, state = supervisor.session_state(first)
        state["queue"].clear()
        supervisor.save_state(first, state)
        self.assertEqual(supervisor.session_state(first)[0], plan)
        self.assertEqual({p.name for p in first.iterdir()},
                         {"session.json", "events.jsonl", "report.md"})

    def test_resolved_configs_are_separate_per_run_and_survive_move(self) -> None:
        """Keep exact YAMLs with each attempt, without copying other native files."""
        session = self.create()
        plan, state = supervisor.session_state(session)
        for number in (1, 2):
            name = f"run{number}"
            native = Path(plan["output_root"]) / f"timestamp_{name}" / "params"
            native.mkdir(parents=True)
            (native / "agent.yaml").write_text(f"seed: {number}\nmax_iterations: 750\n")
            (native / "env.yaml").write_text(f"num_envs: {number * 128}\n")
            (native / "agent.pkl").write_bytes(b"not a requested YAML")
            attempt = {"run_id": name, "run_name": name, "spec": plan["runs"][0],
                       "record_dir": f"runs/{name}", "status": "complete", "exit_code": 0}
            supervisor.capture_configs(session, plan, attempt, required=True)
            supervisor.capture_configs(session, plan, attempt, required=True)
            state["attempts"].append(attempt)
            destination = supervisor.run_folder(session, attempt) / "params"
            self.assertEqual({p.name for p in destination.iterdir()}, {"agent.yaml", "env.yaml"})
            for config in attempt["resolved_configs"]:
                saved = supervisor.local_path(session, config["path"])
                self.assertFalse(Path(config["path"]).is_absolute())
                self.assertEqual(saved.read_bytes(), (native / saved.name).read_bytes())
                self.assertEqual(supervisor.fingerprint(saved)["sha256"], config["sha256"])
            self.assertEqual(len(attempt["resolved_configs"]), 2)
        supervisor.save_state(session, state)
        supervisor.write_report(session, state)
        report = (session / "report.md").read_text()
        self.assertIn("[agent.yaml](<runs/run2/params/agent.yaml>)", report)
        self.assertIn("[env.yaml](<runs/run2/params/env.yaml>)", report)
        self.assertNotIn("provenance.tar.gz", report)
        self.assertFalse((session / "provenance.tar.gz").exists())
        moved = session.with_name("moved-session")
        session.rename(moved)
        _, restored = supervisor.session_state(moved)
        for number, attempt in enumerate(restored["attempts"], 1):
            self.assertEqual((supervisor.run_folder(moved, attempt) / "params/agent.yaml").read_text(),
                             f"seed: {number}\nmax_iterations: 750\n")

    def test_required_configs_block_missing_yaml_and_failures_keep_available_yaml(self) -> None:
        """A missing config cannot pass verification; failed runs keep available evidence."""
        session = self.create()
        plan, _ = supervisor.session_state(session)
        attempt = {"run_id": "baseline", "run_name": "baseline", "record_dir": "runs/baseline"}
        native = Path(plan["output_root"]) / "timestamp_baseline" / "params"
        native.mkdir(parents=True)
        (native / "agent.yaml").write_text("seed: 42\n")
        with self.assertRaisesRegex(RuntimeError, "resolved config missing:.*env.yaml"):
            supervisor.capture_configs(session, plan, attempt, required=True)
        supervisor.capture_configs(session, plan, attempt)
        self.assertEqual([p.name for p in (session / "runs/baseline/params").iterdir()], ["agent.yaml"])
        self.assertEqual(len(attempt["resolved_configs"]), 1)

    def test_resumed_current_record_and_relative_paths_survive_move(self) -> None:
        session = self.create()
        _, state = supervisor.session_state(session)
        attempt = {"run_id": "baseline", "record_dir": "runs/readable", "console": "runs/readable/train.log"}
        state["attempts"] = [attempt]
        state["current"] = attempt
        supervisor.save_state(session, state)
        folder = supervisor.run_folder(session, attempt)
        folder.mkdir(parents=True)
        (folder / "train.log").write_text("evidence")
        moved = session.with_name("archived-session")
        session.rename(moved)
        _, restored = supervisor.session_state(moved)
        self.assertIs(restored["current"], restored["attempts"][0])
        self.assertEqual(supervisor.local_path(moved, restored["current"]["console"]).read_text(), "evidence")

    def evaluation_fixture(self):
        session = self.create(self.plan(
            runs=[{"id": "baseline", "max_iterations": 750, "num_steps_per_env": 64, "video": False,
                   "overrides": {"env.events.reset_plug_curriculum.params.anneal_end_iter": 500}}],
            evaluation={"enabled": True, "seeds": [7, 8], "num_envs": 2},
        ))
        plan, state = supervisor.session_state(session)
        native = Path(self.temp.name) / "native"
        native.mkdir()
        checkpoint = native / "model_749.pt"
        checkpoint.write_bytes(b"checkpoint-fixture")
        attempt = {"run_id": "baseline", "run_number": 1, "record_dir": "runs/readable",
                   "run_dir": str(native), "spec": plan["runs"][0], "status": "complete",
                   "verification": {"checkpoint": str(checkpoint), "videos": [], "exports": []},
                   "exit_code": 0, "additional_updates_completed": 750, "total_lineage_updates": 750}
        self.record_provenance(session, attempt)
        state["attempts"] = [attempt]
        state["status"] = "complete"
        state["completed_runs"] = 1
        supervisor.save_state(session, state)
        return session, plan, state, attempt

    @staticmethod
    def commit_evaluation(_session, _plan, _state, _attempt, command, _console, _identity):
        supervisor.append_record(Path(command[command.index("--journal") + 1]), {
            "kind": "result", "seed": int(command[command.index("--seed") + 1]),
            "evaluation_id": command[command.index("--evaluation-id") + 1],
            "terminal_success_count": 2, "num_envs": 2, "per_env_success": [1, 1],
        })
        return 0

    def test_interrupted_evaluation_reuses_durable_seed_without_relaunch(self) -> None:
        session, plan, state, attempt = self.evaluation_fixture()

        def interrupted(*args):
            self.commit_evaluation(*args)
            raise SystemExit("supervisor interrupted after result commit")

        with patch.object(supervisor, "evaluation_process", side_effect=interrupted):
            with self.assertRaises(SystemExit):
                supervisor.evaluate_attempt(session, plan, state, attempt)
        plan, state = supervisor.session_state(session)
        attempt = state["attempts"][0]
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            supervisor.evaluate_attempt(session, plan, state, attempt)
            supervisor.evaluate_attempt(session, plan, state, attempt)
        self.assertEqual(evaluate.call_count, 1)
        command = evaluate.call_args.args[4]
        self.assertEqual(command[command.index("--seed") + 1], "8")
        self.assertEqual(attempt["evaluations"]["749"]["successes"], 4)
        journal = supervisor.run_folder(session, attempt) / "evaluations.jsonl"
        self.assertEqual([item["seed"] for item in records(journal) if item["kind"] == "result"], [7, 8])

    def test_partial_journal_and_failed_process_are_not_success(self) -> None:
        journal = Path(self.temp.name) / "evaluations.jsonl"
        supervisor.append_record(journal, {"kind": "result", "evaluation_id": "fixture", "seed": 7})
        supervisor.append_record(journal, {"kind": "process", "evaluation_id": "fixture", "seed": 7, "exit_code": 2})
        with journal.open("ab") as stream:
            stream.write(b'{"kind":"res')
        self.assertIsNone(completed_result(journal, "fixture", 7))
        supervisor.append_record(journal, {"kind": "result", "evaluation_id": "fixture", "seed": 8})
        self.assertTrue(any(item["kind"] == "interrupted_record" for item in records(journal)))
        self.assertIsNotNone(completed_result(journal, "fixture", 8))

    def test_more_seeds_reuse_results_but_changed_checkpoint_does_not(self) -> None:
        session, plan, state, attempt = self.evaluation_fixture()
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertEqual(evaluate.call_count, 2)
            plan["evaluation"]["seeds"].append(9)
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(attempt["evaluations"]["749"]["episodes"], 6)
            Path(attempt["verification"]["checkpoint"]).write_bytes(b"different checkpoint")
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertEqual(evaluate.call_count, 6)

    def test_bad_checkpoint_request_does_not_block_corrected_evaluation(self):
        session, plan, state, attempt = self.evaluation_fixture()
        config = self.root / "evaluation.json"
        config.write_text(json.dumps({**plan["evaluation"], "checkpoint_steps": [749, 999999]}))
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "checkpoint missing"):
                supervisor.evaluate_saved(session, "baseline", config)
            self.assertNotIn("pending_evaluation", supervisor.session_state(session)[1])
            evaluate.assert_not_called()
            config.write_text(json.dumps(plan["evaluation"]))
            supervisor.evaluate_saved(session, "baseline", config)
            self.assertEqual(evaluate.call_count, 2)

    def test_failed_evaluation_can_be_cancelled_and_corrected(self):
        session, plan, state, attempt = self.evaluation_fixture()
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(plan["evaluation"]))
        with patch.object(supervisor, "evaluation_process", side_effect=RuntimeError("evaluation budget reached")):
            with self.assertRaisesRegex(RuntimeError, "budget reached"):
                supervisor.evaluate_saved(session, "baseline", config)
        config.write_text(json.dumps({**plan["evaluation"], "seeds": [9]}))
        with self.assertRaisesRegex(RuntimeError, "original config"):
            supervisor.evaluate_saved(session, "baseline", config)
        with patch.object(sys, "argv", ["supervisor.py", "cancel-evaluation", "--session", str(session)]):
            self.assertEqual(supervisor.main(), 0)
        state = supervisor.session_state(session)[1]
        self.assertNotIn("pending_evaluation", state)
        self.assertEqual(records(session / "events.jsonl")[-1]["error"], "evaluation budget reached")
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            supervisor.evaluate_saved(session, "baseline", config)
            self.assertEqual(evaluate.call_count, 1)

    def test_cancel_rejects_live_evaluator_then_reconciles_dead_job(self):
        session, plan, state, attempt = self.evaluation_fixture()
        state["pending_evaluation"] = {"run_id": "baseline", "evaluation": plan["evaluation"]}
        attempt["evaluation_job"] = {"identity": "fixture:seed7", "pid": os.getpid(),
                                     "birth": supervisor.process_birth(os.getpid()),
                                     "started_epoch": supervisor.time.time() - 1, "argv": []}
        supervisor.save_state(session, state)
        with self.assertRaisesRegex(RuntimeError, "evaluator is still active"):
            supervisor.cancel_evaluation(session)
        attempt["evaluation_job"]["birth"] = "not-this-process"
        supervisor.save_state(session, state)
        journal = supervisor.run_folder(session, attempt) / "evaluations.jsonl"
        supervisor.append_record(journal, {"kind": "result", "evaluation_id": "fixture", "seed": 7})
        with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            supervisor.cancel_evaluation(session)
        state = supervisor.session_state(session)[1]
        self.assertNotIn("evaluation_job", state["attempts"][0])
        self.assertNotIn("pending_evaluation", state)
        self.assertGreater(state["evaluation_seconds"], 0)
        self.assertIsNone(completed_result(journal, "fixture", 7))

    def test_evaluation_rejects_source_lock_and_config_drift_before_cache_reuse(self):
        session, plan, state, attempt = self.evaluation_fixture()
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertEqual(evaluate.call_count, 2)
            files = [self.root / "src/isaaclab_training/__init__.py", self.root / "uv.lock",
                     self.root / "pyproject.toml", supervisor.run_folder(session, attempt) / "params/env.yaml"]
            for path in files:
                original = path.read_bytes()
                with self.subTest(path=path):
                    path.write_bytes(original + b"# changed\n")
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        supervisor.evaluate_attempt(session, plan, state, attempt)
                    path.write_bytes(original)
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertEqual(evaluate.call_count, 2)
            attempt.pop("training_sha256")
            with self.assertRaisesRegex(RuntimeError, "no training fingerprint"):
                supervisor.evaluate_attempt(session, plan, state, attempt)

    def test_changed_provenance_has_distinct_evaluation_cache_identity(self):
        session, plan, state, attempt = self.evaluation_fixture()
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation) as evaluate:
            supervisor.evaluate_attempt(session, plan, state, attempt)
            old_identity = attempt["evaluations"]["749"]["evaluation_id"]
            # A separately recorded run can have the same checkpoint bytes/seeds
            # but different code. Its results must not reuse the old identity.
            (self.root / "src/isaaclab_training/__init__.py").write_text("# new task\n")
            attempt["training_sha256"] = supervisor.training_fingerprint(self.root)
            supervisor.evaluate_attempt(session, plan, state, attempt)
            self.assertNotEqual(old_identity, attempt["evaluations"]["749"]["evaluation_id"])
            self.assertEqual(evaluate.call_count, 4)

    def test_live_evaluator_is_adopted_without_duplicate_launch(self) -> None:
        session, plan, state, attempt = self.evaluation_fixture()
        attempt["evaluation_job"] = {"identity": "fixture", "pid": 123, "birth": "100",
                                     "started_epoch": supervisor.time.time() - 1, "argv": []}
        with patch.object(supervisor, "process_alive", side_effect=[True, False]), \
             patch.object(supervisor.time, "sleep"), \
             patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("duplicate evaluator")):
            code = supervisor.evaluation_process(session, plan, state, attempt, [],
                                                session / "evaluation.log", "fixture")
        self.assertIsNone(code)
        self.assertGreater(state["evaluation_seconds"], 0)

    def test_independent_validation_shares_logs_and_report(self) -> None:
        session, plan, state, attempt = self.evaluation_fixture()
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation):
            supervisor.evaluate_attempt(session, plan, state, attempt)
        config = Path(self.temp.name) / "holdout.json"
        config.write_text(json.dumps({"enabled": True, "group": "independent", "seeds": [7], "num_envs": 2}))
        with self.assertRaisesRegex(ValueError, "overlap"):
            supervisor.evaluate_saved(session, "baseline", config)
        config.write_text(json.dumps({"enabled": True, "group": "independent", "seeds": [9], "num_envs": 2}))
        with patch.object(supervisor, "evaluation_process", side_effect=self.commit_evaluation):
            supervisor.evaluate_saved(session, "baseline", config)
        report = (session / "report.md").read_text()
        self.assertIn("Independent validation: **100.00%** (2/2)", report)
        self.assertIn("R01 · baseline", report)
        self.assertLess(len(report.split()), 450)
        for value, expected in ((0, "0"), (None, "unavailable")):
            latest = supervisor.session_state(session)[1]
            latest["attempts"][0]["total_lineage_updates"] = value
            supervisor.write_report(session, latest)
            self.assertIn(f"total lineage updates={expected}.", (session / "report.md").read_text())
        self.assertEqual(len(list((session / "runs").rglob("evaluations.jsonl"))), 1)
        self.assertEqual(supervisor.session_state(session)[0], plan)

    def test_known_failed_exit_cannot_be_recovered_as_success(self) -> None:
        session, plan, state, attempt = self.evaluation_fixture()
        attempt.update(status="verifying", exit_code=7)
        state["current"] = attempt
        state["completed_runs"] = 0
        with patch.object(supervisor, "verify_attempt", side_effect=AssertionError("must not adopt failed run")):
            supervisor.reconcile_current(session, plan, state)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["completed_runs"], 0)


if __name__ == "__main__":
    unittest.main()
