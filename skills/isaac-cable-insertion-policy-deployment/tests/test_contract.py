# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with the Isaac Lab Python; no Docker, GPU, ROS, or Kit needed."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare")
deploy = load("deploy")
security = load("scene_security")
recorder = load("sim_record")


def supported_config():
    """Small saved-config fixture exercising the supported controller contract."""
    stiffness = [300, 300, 300, 30, 30, 30]
    controller = {
        "target_types": ["pose_rel"], "motion_control_axes_task": [1] * 6,
        "contact_wrench_control_axes_task": [0] * 6,
        "inertial_dynamics_decoupling": False, "partial_inertial_dynamics_decoupling": False,
        "gravity_compensation": False, "impedance_mode": "fixed", "nullspace_control": "none",
        "motion_stiffness_task": stiffness,
        "motion_damping_ratio_task": [d / (2 * math.sqrt(k)) for d, k in
                                      zip([35, 35, 35, 1.1, 1.1, 1.1], stiffness)],
    }
    env = {
        "policy_action_space": "task", "action_space": 6, "observation_space": 18,
        "obs_order": prepare.OBSERVATIONS, "num_arm_joints": 7,
        "success_pos_threshold": .003, "success_keypoint_scale": .15,
        "success_socket_offset": [.0375, 0, 0], "success_plug_offset": [0, 0, .0221],
        "grasp_offset": [.0025, 0, -.1875], "grasp_rot_offset": [0, 0, 0, 1],
        "success_plug_goal_rot_inv": [0, .70711, 0, .70711],
        "hand_hold_width": -.05, "hand_close_width": -.155,
        "sim": {"dt": 1 / 240}, "decimation": 8,
        "actions": {"arm_action": {
            "class_type": "module:DeployOperationalSpaceControllerAction", "body_name": "flange",
            "body_offset": None, "task_frame_rel_path": None, "clip": None,
            "joint_names": [f"joint{i}" for i in range(1, 8)],
            "position_scale": .025, "orientation_scale": .025, "controller_cfg": controller,
        }, "gripper_action": None},
        "observations": {"policy": {
            name: {"params": {"body_name": "flange"}} for name in prepare.OBSERVATIONS
        }},
        "scene": {
            name: {"spawn": {"usd_path": f"https://example.test/{filename}"},
                   "init_state": {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]}}
            for name, filename in {"robot": "rizon4s_with_grav.usd", "dp_plug": "displayport_plug.usd",
                                   "dp_socket": "displayport_socket_no_protrusions.usd"}.items()
        },
    }
    env["observations"]["policy"]["eef_pos"]["params"]["offset"] = [0, 0, .1925]
    env["observations"]["policy"]["socket_kp_pos"]["params"]["offset"] = [.0375, 0, 0]
    for name, function in zip(prepare.OBSERVATIONS,
                              ("eef_pos_w", "eef_rot_6d_w", "rigid_object_pos_w", "rigid_object_rot_6d_w")):
        term = env["observations"]["policy"][name]
        term["func"] = "isaaclab_training.mdp.observations:" + function
        term["params"]["asset_cfg"] = {"name": "robot" if name.startswith("eef_") else "dp_socket"}
    net = {"class_name": "RNNModel", "hidden_dims": [256, 128, 64], "activation": "elu",
           "obs_normalization": True, "rnn_type": "lstm", "rnn_hidden_dim": 256, "rnn_num_layers": 2}
    agent = {"class_name": "OnPolicyRunner", "clip_actions": 1.0,
             "actor": copy.deepcopy(net), "critic": copy.deepcopy(net)}
    agent["actor"]["distribution_cfg"] = {"class_name": "HeteroscedasticGaussianDistribution", "std_type": "log"}
    return env, agent


def read_document(document):
    loader = prepare.ConfigLoader(document)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


class ContractTests(unittest.TestCase):
    def test_changed_observation_sources_are_rejected(self):
        for name in prepare.OBSERVATIONS:
            for change in ("asset", "function"):
                with self.subTest(observation=name, change=change):
                    env, agent = supported_config()
                    term = env["observations"]["policy"][name]
                    if change == "asset":
                        term["params"]["asset_cfg"]["name"] = "dp_plug"
                    else:
                        term["func"] = "custom.observations:modified_pose"
                    with self.assertRaisesRegex(ValueError, "observation " + name):
                        prepare.validate_contract(env, agent)

    def test_prepare_exports_saved_native_actor_critic_settings(self):
        env, agent = supported_config()
        agent.update(seed=17, num_steps_per_env=128, max_iterations=1000)
        env["events"] = {
            f"{name}_physics_material": {"params": {
                "static_friction_range": [1, 1], "dynamic_friction_range": [1, 1],
                "restitution_range": [0, 0]}}
            for name in ("robot", "plug", "socket")
        }
        env["events"]["reset_plug_curriculum"] = {"params": {"anneal_end_iter": 333}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "params"
            params.mkdir()
            (params / "env.yaml").write_text(yaml.safe_dump(env))
            (params / "agent.yaml").write_text(yaml.safe_dump(agent))
            checkpoint = root / "model_999.pt"
            checkpoint.write_bytes(b"checkpoint loaded only by the exporter")
            output = root / "export-session"
            output.mkdir()
            args = argparse.Namespace(training_root=str(root), checkpoint=str(checkpoint),
                                      output=str(output), overrides=None)
            with patch.object(prepare, "check_export_environment", return_value={}), \
                 patch.object(prepare.importlib.metadata, "version", return_value="test"):
                record = prepare.prepare(args)
            command = record["export_command"]
            overrides = dict(item.split("=", 1) for item in command if item.startswith("agent."))
            self.assertFalse(any(key.startswith("agent.policy.") for key in overrides))
            for name in ("actor", "critic"):
                for key, value in agent[name].items():
                    if key != "distribution_cfg":
                        self.assertEqual(json.loads(overrides[f"agent.{name}.{key}"]), value)
            for key, value in agent["actor"]["distribution_cfg"].items():
                self.assertEqual(json.loads(overrides[f"agent.actor.distribution_cfg.{key}"]), value)
            self.assertEqual(command[command.index("--contract") + 1], "displayport_task_space")
            self.assertEqual(command[command.index("--validation_steps") + 1], "5")
            for name in ("env.yaml", "agent.yaml"):
                self.assertEqual((output / "params" / name).read_bytes(), (params / name).read_bytes())

    def test_native_network_shapes_remain_restricted(self):
        for name in ("actor", "critic"):
            with self.subTest(model=name):
                env, agent = supported_config()
                agent[name]["rnn_hidden_dim"] = 128
                with self.assertRaisesRegex(ValueError, f"agent.{name}.rnn_hidden_dim"):
                    prepare.validate_contract(env, agent)

    def test_prepared_grasp_checks_each_condition_independently(self):
        valid = {"grasp_error_m": .001, "grasp_rotation_error_deg": .2,
                 "clearance_m": .049, "finger_joint": -.055}
        recorder.validate_prepared_grasp(valid, -.155)
        for field, value in [("grasp_error_m", .004), ("grasp_rotation_error_deg", 6),
                             ("clearance_m", .039), ("clearance_m", .061),
                             ("finger_joint", -.155)]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(RuntimeError, "before policy enable"):
                    recorder.validate_prepared_grasp({**valid, field: value}, -.155)
        for field in valid:
            with self.subTest(nonfinite_field=field):
                with self.assertRaises(RuntimeError):
                    recorder.validate_prepared_grasp({**valid, field: float("nan")}, -.155)

    def test_modified_script_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            security.reviewed_controller_script("print('unreviewed')")

    def test_rgb_encoder_contract(self):
        import numpy as np
        for channels in (3, 4):
            self.assertEqual(len(recorder.rgb_frame_bytes(np.zeros((720, 1280, channels), dtype=np.uint8))),
                             720 * 1280 * 3)
        for frame in [np.zeros((720, 1280, 3), dtype=np.float32),
                      np.zeros((720, 1280, 2), dtype=np.uint8),
                      np.zeros((1280, 720, 3), dtype=np.uint8)]:
            with self.assertRaises((ValueError, RuntimeError)):
                recorder.rgb_frame_bytes(frame)

    def test_yaml_objects_cannot_execute(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "executed"
            document = "!!python/object/apply:os.system [" + json.dumps(f"touch {marker}") + "]"
            with self.assertRaises(yaml.constructor.ConstructorError):
                read_document(document)
            self.assertFalse(marker.exists())

    def test_tuple_and_slice_are_inert_data(self):
        document = "tuple: !!python/tuple [1, 2]\nslice: !!python/object/apply:builtins.slice [null, null, null]"
        self.assertEqual(read_document(document),
                         {"tuple": [1, 2], "slice": [None, None, None]})

    def test_supported_contract_then_wrong_action_frame(self):
        env, agent = supported_config()
        self.assertEqual(prepare.validate_contract(env, agent)["body_name"], "flange")
        env["actions"]["arm_action"]["body_offset"] = {"pos": [0, 0, .1925]}
        with self.assertRaisesRegex(ValueError, "body offset"):
            prepare.validate_contract(env, agent)

    def test_unsupported_controller_rejected(self):
        env, agent = supported_config()
        env["actions"]["arm_action"]["controller_cfg"]["gravity_compensation"] = True
        with self.assertRaisesRegex(ValueError, "gravity_compensation"):
            prepare.validate_contract(env, agent)

    def test_override_cannot_change_interface(self):
        with self.assertRaisesRegex(ValueError, "Unsupported export override"):
            prepare.extra_overrides('["env.actions.arm_action.body_name=tcp"]')


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        lab = self.root / "lab"
        (lab / ".venv/bin").mkdir(parents=True)
        (lab / "pyproject.toml").write_text('[project]\nname="isaaclab-training"\n')
        (lab / ".venv/bin/python").touch()
        checkpoint = lab / "model_749.pt"
        checkpoint.touch()
        self.args = argparse.Namespace(output=str(self.root / "session"), training_root=str(lab), uv=sys.executable,
            checkpoint=str(checkpoint), sudo_docker=False, accept_eula=True,
            runtime_image="test-ros", sim_image="test-sim", domain_id=211,
            install_sim_runtime=False, install_runtime=False)

    def test_simulator_uses_image_identity_and_scoped_mounts(self):
        session = deploy.Session(self.args)
        session.sim_image_id, session.runtime_image_id = "sha256:sim", "sha256:ros"
        commands = []
        def run(command, log, **kwargs):
            commands.append(command)
            if log == "sim-version.txt":
                (session.root / log).write_text("6.0.0-rc.59")
        with patch.object(session, "run", side_effect=run):
            session.create_simulator()
        command = commands[0]
        self.assertIn("sha256:sim", command)
        self.assertNotIn("test-sim", command)
        self.assertIn(f"{deploy.SCRIPTS}:/skill:ro", command)
        self.assertIn(f"{session.root}:{session.root}", command)
        self.assertIn("ROS_DOMAIN_ID=211", command)
        self.assertIn("ACCEPT_EULA=Y", command)
        # An exact option allowlist rejects any added privilege, capability,
        # device or namespace flag, including options passed with =value.
        options = {arg.split("=", 1)[0] for arg in command if arg.startswith("--")}
        self.assertEqual(options, {"--name", "--label", "--user", "--gpus",
                                   "--network", "--shm-size", "--entrypoint"})
        self.assertTrue(any("libx264" in " ".join(c) for c in commands))
        self.assertTrue(session.sim_created)

    def test_image_version_is_verified_before_encoding_or_export(self):
        session = deploy.Session(self.args)
        session.sim_image_id, session.runtime_image_id = "sha256:sim", "sha256:ros"
        def run(command, log, **kwargs):
            if log == "sim-version.txt":
                (session.root / log).write_text("6.1.0-rc.26")
        with patch.object(session, "run", side_effect=run) as calls:
            with self.assertRaisesRegex(ValueError, "requires Isaac Sim 6.0.0"):
                session.create_simulator()
        self.assertFalse(any(deploy.SIM_FFMPEG in str(c) for c in calls.call_args_list))

    def test_encoder_failure_is_detected_inside_simulator_container(self):
        session = deploy.Session(self.args)
        session.sim_image_id, session.runtime_image_id = "sha256:sim", "sha256:ros"
        def run(command, log, **kwargs):
            if log == "sim-version.txt":
                (session.root / log).write_text("6.0.0-rc.59")
            if "libx264" in " ".join(command):
                self.assertEqual(command[:3], ["docker", "exec", session.sim_container])
                raise RuntimeError("encoder unavailable")
        with patch.object(session, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "encoder unavailable"):
                session.create_simulator()

    def test_image_inspection_returns_immutable_identity(self):
        session = deploy.Session(self.args)
        with patch.object(session, "run"):
            (session.root / "sim-image.json").write_text(json.dumps([{"Id": "sha256:immutable"}]))
            self.assertEqual(session.inspect_image("mutable:tag", "sim-image.json"), "sha256:immutable")

    def test_compatibility_requires_semantic_pass(self):
        session = deploy.Session(self.args)
        (session.root / "compatibility.log").write_text("checker exited cleanly without a result")
        with patch.object(session, "start", return_value=Mock(poll=Mock(return_value=0))):
            with self.assertRaisesRegex(RuntimeError, "did not report PASSED"):
                session.check_compatibility()

    def test_compatibility_failure_is_not_accepted(self):
        session = deploy.Session(self.args)
        (session.root / "compatibility.log").write_text("System checking result: FAILED")
        with patch.object(session, "start", return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, "compatibility check failed"):
                session.check_compatibility()

    def test_compatibility_pass_restarts_same_container_to_clear_checker(self):
        session = deploy.Session(self.args)
        (session.root / "compatibility.log").write_text("System checking result: PASSED")
        proc = Mock()
        with patch.object(session, "start", return_value=proc), \
                patch.object(session, "run") as run, patch.object(session, "stop") as stop:
            session.check_compatibility()
        self.assertEqual(run.call_args_list[0].args[0],
                         ["docker", "stop", "--time", "10", session.sim_container])
        self.assertEqual(run.call_args_list[1].args[0], ["docker", "start", session.sim_container])
        stop.assert_called_once_with(proc)

    def test_cleanup_attempts_both_containers_when_ownership_repair_fails(self):
        session = deploy.Session(self.args)
        session.sim_created = session.created = True
        def run(command, *args, **kwargs):
            if session.sim_container in command and "chown" in command:
                raise RuntimeError("ownership repair failed")
        with patch.object(session, "run", side_effect=run) as calls:
            session.cleanup()
        for name in (session.sim_container, session.container):
            self.assertIn(["docker", "stop", "--time", "10", name],
                          [c.args[0] for c in calls.call_args_list])
        self.assertEqual(session.status, "failed during cleanup")

    def test_missing_params_rejected_before_runtime_commands(self):
        session = deploy.Session(self.args)
        with patch.object(session, "run") as run:
            with self.assertRaisesRegex(ValueError, "params/env.yaml"):
                session.preflight()
            run.assert_not_called()

    def test_existing_session_is_not_overwritten(self):
        output = Path(self.args.output)
        output.mkdir()
        marker = output / "existing.json"
        marker.write_text("keep me")
        with self.assertRaises(FileExistsError):
            deploy.Session(self.args)
        self.assertEqual(marker.read_text(), "keep me")

    def test_host_commands_do_not_expand_shell_syntax(self):
        session = deploy.Session(self.args)
        marker = self.root / "must-not-exist"
        argument = f"path with spaces/$(touch {marker});${{HOME}}"
        session.run([sys.executable, "-c", "import json,sys; print(json.dumps(sys.argv[1:]))", argument], "argv.log")
        self.assertEqual(json.loads((session.root / "argv.log").read_text()), [argument])
        self.assertFalse(marker.exists())

    def test_all_children_use_clean_environment(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/old/sim", "PYTHONPATH": "/old/python",
                                     "UV_PROJECT_ENVIRONMENT": "/old/venv", "SECRET_TEST_TOKEN": "private"}):
            session = deploy.Session(self.args)
            code = ("import json,os; names=('LD_LIBRARY_PATH','PYTHONPATH',"
                    "'UV_PROJECT_ENVIRONMENT','SECRET_TEST_TOKEN','PYTHONNOUSERSITE'); "
                    "print(json.dumps({name:os.environ[name] for name in names if name in os.environ}))")
            for name in ("run", "start"):
                command = [sys.executable, "-c", code]
                result = getattr(session, name)(command, name + ".log")
                if result is not None:
                    self.assertEqual(result.wait(timeout=10), 0)
                child = json.loads((session.root / (name + ".log")).read_text())
                for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "SECRET_TEST_TOKEN"):
                    self.assertNotIn(key, child)
                self.assertEqual(child["PYTHONNOUSERSITE"], "1")

    def test_training_plan_root_mismatch_rejected(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"training_root": "/wrong/root"}))
        self.args.training_plan = str(plan)
        with self.assertRaisesRegex(ValueError, "selected --training-root"):
            deploy.Session(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_export_preflight_without_source_version_or_deploy_runtime(self):
        params = Path(self.args.checkpoint).parent / "params"
        params.mkdir()
        for name in ("env.yaml", "agent.yaml"):
            (params / name).touch()
        self.args.export_only = True
        session = deploy.Session(self.args)
        with patch.object(session, "run") as run:
            session.preflight()
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(Path(self.args.training_root) / ".venv/bin/python"))
        self.assertTrue(command[1].endswith("training_support.py"))
        self.assertFalse((Path(self.args.training_root) / "VERSION").exists())

    def test_container_shell_roundtrips_literal_arguments(self):
        session = deploy.Session(self.args)
        original = ["python3", "/path with spaces/script.py", "a; $(false) `false` ${HOME}"]
        command = session.inside(original)
        self.assertEqual(command[:4], ["docker", "exec", session.container, "bash"])
        script = command[-1]
        self.assertTrue(script.startswith(deploy.ROS_ENV + "exec "))
        self.assertEqual(shlex.split(script[len(deploy.ROS_ENV + "exec "):]), original)


if __name__ == "__main__":
    unittest.main()
