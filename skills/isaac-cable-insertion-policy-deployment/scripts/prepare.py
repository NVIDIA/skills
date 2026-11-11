#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a supported training result and prepare a reproducible LEAPP export.

Run with the Isaac Lab environment's Python. This stage does not launch Kit or
load the pickle checkpoint. Python YAML tags are read as inert data only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

import yaml

from training_support import check_export_environment

TASK = "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference"
OBSERVATIONS = ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]


class ConfigLoader(yaml.SafeLoader):
    """Allow Isaac Lab's tuple/slice serialization without Python construction."""


for _tag in ("python/tuple", "python/object/apply:builtins.slice"):
    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:" + _tag,
        lambda loader, node: loader.construct_sequence(node),
    )
ConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
)


def read_config(path: Path) -> dict:
    loader = ConfigLoader(path.read_text())
    try:
        value = loader.get_single_data()
    finally:
        loader.dispose()
    if not isinstance(value, dict):
        raise ValueError(f"Expected configuration mapping: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(value, expected) -> bool:
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            equal(a, b) for a, b in zip(value, expected)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-8)
    return value == expected


def require(value, expected, label: str) -> None:
    if not equal(value, expected):
        raise ValueError(f"Unsupported {label}: {value!r}; supported value is {expected!r}")


def validate_contract(env: dict, agent: dict) -> dict:
    """Fail closed for interfaces not implemented by the prepared-grasp harness."""
    for name, expected in {"policy_action_space": "task", "action_space": 6,
                           "observation_space": 18, "obs_order": OBSERVATIONS,
                           "num_arm_joints": 7, "success_pos_threshold": 0.003,
                           "success_keypoint_scale": 0.15}.items():
        require(env.get(name), expected, name)
    arm = env["actions"]["arm_action"]
    require(arm["class_type"].split(":")[-1], "DeployOperationalSpaceControllerAction", "action type")
    require(arm.get("body_name"), "flange", "action body")
    require(arm.get("body_offset"), None, "action body offset")
    require(arm.get("task_frame_rel_path"), None, "action task frame")
    require(arm.get("clip"), None, "action clip")
    require(env["actions"].get("gripper_action"), None, "gripper action")
    joints = [f"joint{i}" for i in range(1, 8)]
    require(arm["joint_names"], joints, "arm joint names")
    for name in ("position_scale", "orientation_scale"):
        value = arm[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}: {value!r}")
    controller = arm["controller_cfg"]
    expected_controller = {
        "target_types": ["pose_rel"], "motion_control_axes_task": [1] * 6,
        "contact_wrench_control_axes_task": [0] * 6,
        "inertial_dynamics_decoupling": False, "partial_inertial_dynamics_decoupling": False,
        "gravity_compensation": False, "impedance_mode": "fixed", "nullspace_control": "none",
        "motion_stiffness_task": [300, 300, 300, 30, 30, 30],
    }
    for name, expected in expected_controller.items():
        require(controller.get(name), expected, "controller." + name)
    stiffness = controller["motion_stiffness_task"]
    damping = [2 * math.sqrt(k) * r for k, r in zip(stiffness, controller["motion_damping_ratio_task"])]
    require(damping, [35, 35, 35, 1.1, 1.1, 1.1], "controller damping")
    contract = {
        "socket_keypoint_offset": [0.0375, 0, 0], "plug_keypoint_offset": [0, 0, 0.0221],
        "grasp_offset": [0.0025, 0, -0.1875], "grasp_rot_offset": [0, 0, 0, 1],
        "success_plug_goal_rot_inv": [0, 0.70711, 0, 0.70711],
        "tcp_offset": [0, 0, 0.1925], "stiffness": stiffness, "damping": damping,
        "body_name": "flange", "body_offset": None,
        "hand_hold_width": -0.05, "hand_close_width": -0.155,
        "success_pos_threshold": env["success_pos_threshold"],
        "success_keypoint_scale": env["success_keypoint_scale"],
        "physics_dt": env["sim"]["dt"], "decimation": env["decimation"],
        "position_scale": arm["position_scale"], "orientation_scale": arm["orientation_scale"],
    }
    for source, key in (("success_socket_offset", "socket_keypoint_offset"),
                        ("success_plug_offset", "plug_keypoint_offset"),
                        ("grasp_offset", "grasp_offset"), ("grasp_rot_offset", "grasp_rot_offset"),
                        ("success_plug_goal_rot_inv", "success_plug_goal_rot_inv"),
                        ("hand_hold_width", "hand_hold_width"), ("hand_close_width", "hand_close_width")):
        require(env.get(source), contract[key], source)
    require(contract["physics_dt"], 1 / 240, "physics timestep")
    require(contract["decimation"], 8, "policy decimation")
    policy = env["observations"]["policy"]
    require(policy["eef_pos"]["params"]["offset"], contract["tcp_offset"], "observed TCP offset")
    require(policy["socket_kp_pos"]["params"]["offset"], contract["socket_keypoint_offset"], "socket observation offset")
    functions = ("eef_pos_w", "eef_rot_6d_w", "rigid_object_pos_w", "rigid_object_rot_6d_w")
    for name, function in zip(OBSERVATIONS, functions):
        observation = policy[name]
        require(observation.get("func"), f"isaaclab_training.mdp.observations:{function}",
                f"observation {name}.func")
        require(observation["params"].get("asset_cfg", {}).get("name"),
                "robot" if name.startswith("eef_") else "dp_socket", f"observation {name}.asset")
        for field in ("modifiers", "clip", "scale"):
            require(observation.get(field), None, f"observation {name}.{field}")
        require(observation.get("history_length", 0), 0, f"observation {name}.history_length")
        require(observation.get("delay_max_lag", 0), 0, f"observation {name}.delay_max_lag")
    for name in ("eef_pos", "eef_rot_6d"):
        require(policy[name]["params"]["body_name"], "flange", f"observation {name}.body")
    require(agent.get("class_name"), "OnPolicyRunner", "agent runner")
    require(agent.get("clip_actions"), 1.0, "agent.clip_actions")
    # The registered task and saved configuration use native actor/critic models.
    # Explicitly support this stock architecture, not arbitrary network shapes.
    for name in ("actor", "critic"):
        net = agent.get(name, {})
        for key, expected in {"class_name": "RNNModel", "hidden_dims": [256, 128, 64],
                              "activation": "elu", "obs_normalization": True,
                              "rnn_type": "lstm", "rnn_hidden_dim": 256, "rnn_num_layers": 2}.items():
            require(net.get(key), expected, f"agent.{name}.{key}")
    distribution = agent["actor"]["distribution_cfg"]
    require(distribution.get("class_name"), "HeteroscedasticGaussianDistribution", "actor distribution")
    require(distribution.get("std_type"), "log", "actor distribution std_type")
    require(env["scene"]["robot"]["init_state"]["pos"], [0, 0, 0], "robot base position")
    require(env["scene"]["robot"]["init_state"]["rot"], [0, 0, 0, 1], "robot base quaternion xyzw")
    for name, basename in {"robot": "rizon4s_with_grav.usd", "dp_plug": "displayport_plug.usd",
                           "dp_socket": "displayport_socket_no_protrusions.usd"}.items():
        require(env["scene"][name]["spawn"]["usd_path"].rsplit("/", 1)[-1], basename, f"{name} asset")
    return contract


def git_info(path: Path) -> dict:
    def command(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"path": str(path), "revision": command("rev-parse", "HEAD"),
            "tracked_changes": command("status", "--porcelain", "--untracked-files=no")}


def extra_overrides(value: str | None) -> list[str]:
    if value is None:
        return []
    candidate = Path(value)
    if not value.lstrip().startswith(("[", "{")) and candidate.is_file():
        value = candidate.read_text()
    data = json.loads(value)
    if not isinstance(data, list) or not all(isinstance(x, str) and "=" in x for x in data):
        raise ValueError("--overrides must contain a JSON list of Hydra key=value strings")
    # Export-time reset/reward overrides do not change the trained interface.
    allowed = ("env.rewards.", "env.events.reset_plug_curriculum.params.", "agent.algorithm.")
    for item in data:
        if not item.startswith(allowed) or "${" in item:
            raise ValueError(f"Unsupported export override: {item}. Interface changes require skill support.")
    return data


def prepare(args) -> dict:
    training_root = Path(args.training_root).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve(strict=True)
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("--checkpoint must be an explicit .pt checkpoint file")
    env_path = checkpoint.parent / "params/env.yaml"
    agent_path = checkpoint.parent / "params/agent.yaml"
    env, agent = read_config(env_path), read_config(agent_path)
    contract = validate_contract(env, agent)
    extras = extra_overrides(args.overrides)
    assets = {name: env["scene"][name] for name in ("robot", "dp_plug", "dp_socket")}
    materials = {name: env["events"][f"{name}_physics_material"]["params"] for name in ("robot", "plug", "socket")}
    for name, material in materials.items():
        for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
            values = material[key]
            if len(values) != 2 or values[0] != values[1]:
                raise ValueError(f"Randomized {name}.{key} requires an explicit simulation material selection")
    provenance = check_export_environment(training_root)
    versions = {name: importlib.metadata.version(name) for name in
                ("isaaclab", "isaaclab-training", "leapp", "rsl-rl-lib", "torch", "isaacsim")}
    overrides = [f"agent.num_steps_per_env={agent['num_steps_per_env']}",
                 f"agent.max_iterations={agent['max_iterations']}",
                 f"env.sim.dt={env['sim']['dt']}", f"env.decimation={env['decimation']}",
                 f"env.actions.arm_action.position_scale={contract['position_scale']}",
                 f"env.actions.arm_action.orientation_scale={contract['orientation_scale']}"]
    curriculum = env["events"]["reset_plug_curriculum"]["params"]
    for key, value in curriculum.items():
        if isinstance(value, (bool, int, float, str)) or value is None or isinstance(value, list):
            overrides.append(f"env.events.reset_plug_curriculum.params.{key}=" + json.dumps(value, separators=(",", ":")))
    # Supply validated saved inference settings through the native RSL-RL 5 API.
    for name in ("actor", "critic"):
        for key in ("class_name", "hidden_dims", "activation", "obs_normalization",
                    "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            overrides.append(f"agent.{name}.{key}=" + json.dumps(agent[name][key], separators=(",", ":")))
    for key in ("class_name", "std_type"):
        overrides.append(f"agent.actor.distribution_cfg.{key}="
                         + json.dumps(agent["actor"]["distribution_cfg"][key], separators=(",", ":")))
    overrides.extend(extras)
    export_dir = output / "export"
    command = [str(training_root / ".venv/bin/python"), str(Path(__file__).with_name("export_policy.py")),
               "--task", TASK, "--checkpoint", str(checkpoint), "--contract", "displayport_task_space",
               "--export_method", "onnx-dynamo", "--validation_steps", "5",
               "--export_save_path", str(export_dir), "--seed", str(agent["seed"]), *overrides]
    params = output / "params"
    params.mkdir(exist_ok=True)
    manifest = {"training_env_file": str(params / "env.yaml"), "training_env_sha256": sha256(env_path),
                "assets": assets, "materials": materials, "contract": contract}
    record = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "task": TASK, "contract": "displayport_task_space", "versions": versions,
              "repositories": [git_info(training_root)],
              "installation": provenance,
              "source_params": {"env": str(env_path), "agent": str(agent_path)},
              "params_sha256": {"env": sha256(env_path), "agent": sha256(agent_path)},
              "export_command": command, "export_cwd": str(training_root),
              "export_env": {"OMNI_KIT_ACCEPT_EULA": "Yes"},
              "scope": "Prepared-grasp insertion; stock task-space LSTM interface; no full manipulation mission",
              "notes": ["Export uses the verified training environment; deployment uses an Isaac Sim 6.0 Docker container.",
                        "Checkpoint pickle is trusted input and is loaded only by the subsequent exporter.",
                        "Saved YAML is not automatically consumed by the upstream exporter. Supported inference settings are explicitly supplied as Hydra overrides.",
                        "Reward and reset settings do not alter trained weights. Export validation does not measure policy success rate."]}
    for path, content in ((params / "env.yaml", env_path.read_bytes()),
                          (params / "agent.yaml", agent_path.read_bytes()),
                          (output / "asset-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
                          (output / "input.json", (json.dumps(record, indent=2) + "\n").encode()),
                          (output / "export-command.json", (json.dumps(command, indent=2) + "\n").encode())):
        if path.exists():
            raise ValueError(f"Refusing to overwrite prepared session file: {path}")
        path.write_bytes(content)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", "--isaaclab", dest="training_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Existing new session directory")
    parser.add_argument("--overrides", help="JSON list of supported additional Hydra overrides, or JSON file")
    args = parser.parse_args()
    try:
        record = prepare(args)
    except (ValueError, KeyError, OSError, yaml.YAMLError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({"output": str(Path(args.output).resolve()), "export_command": record["export_command"]}, indent=2))


if __name__ == "__main__":
    main()
