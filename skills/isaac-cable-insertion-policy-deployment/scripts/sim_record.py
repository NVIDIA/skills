#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record ROS LEAPP insertion from a prepared grasp in Isaac Sim 6.0.

Run inside the Isaac Sim Docker image with /isaac-sim/python.sh. The published NVIDIA tutorial USD
supplies its ROS ScriptNode; no source checkout or external launcher is needed.
The compatibility controller and flange action adapter reproduce the validated
training contract. This is an insertion smoke test, not a pickup mission or an
estimate of the training policy's statistical success rate.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scene_security import (
    SCENE_URL, SCENE_SHA256, SCRIPT_PATH, SCRIPT_SHA256, fetch_scene, local_scene_layer, verify_script, reviewed_controller_script, COMBINED_SCRIPT_SHA256,
)

ARM_PATH = "/World/envs/env_0/Robot"
PLUG_PATH = "/World/envs/env_0/DisplayPortPlug"
SIM_ROOT = Path("/isaac-sim")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=211)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--scene-url", default=SCENE_URL)
    parser.add_argument("--wall-timeout", type=float, default=300.0)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    args = parser.parse_args()
    for value in (args.output, args.asset_manifest, args.ffmpeg):
        if not value.is_absolute():
            parser.error(f"Absolute paths required: {value}")
    if not 0 <= args.domain_id <= 232:
        parser.error("Use ROS domain 0..232")
    if not 2 <= args.duration <= 120 or args.wall_timeout <= 0 or args.ready_timeout <= 0:
        parser.error("Duration must be 2..120 seconds and timeouts positive")
    if args.scene_url != SCENE_URL:
        parser.error("Only the reviewed, hash-pinned NVIDIA scene URL is supported")
    if not args.ffmpeg.is_file() or not os.access(args.ffmpeg, os.X_OK):
        parser.error("--ffmpeg must name an executable")
    version = (SIM_ROOT / "VERSION").read_text().strip()
    if not version.startswith("6.0.0"):
        parser.error(f"Isaac Sim 6.0 is required, got {version}")
    return args


def environment(domain):
    """Scene script settings documented by NVIDIA's DisplayPort launcher."""
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    defaults = {
        "ROS_DISTRO": "jazzy",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
        "DISPLAYPORT_PLAY_STEP_MODE": "play",
        "DISPLAYPORT_PLAY_GRIPPER_TARGET_MODE": "finger",
        "DISPLAYPORT_PLAY_GRIPPER_EFFORT_MODE": "finger",
        "DISPLAYPORT_PLAY_GRIPPER_COMMAND_PATH": "direct",
        "DISPLAYPORT_PLAY_ARM_COMMAND_PATH": "direct",
        "DISPLAYPORT_PLAY_DISABLE_ROBOT_GRAVITY": "1",
        "DISPLAYPORT_PLAY_JOINT_RESYNC_TOLERANCE_RAD": "0.05",
    }
    for name, value in defaults.items():
        os.environ[name] = value


def validate_contract(contract):
    """Reject manifests incompatible with the controller and adapter constants."""
    expected_vectors = {
        "stiffness": [300, 300, 300, 30, 30, 30],
        "damping": [35, 35, 35, 1.1, 1.1, 1.1],
        "tcp_offset": [0, 0, 0.1925],
        "grasp_rot_offset": [0, 0, 0, 1],
    }
    for name, expected in expected_vectors.items():
        actual = contract[name]
        if len(actual) != len(expected) or not all(
            math.isclose(value, target, abs_tol=1e-8, rel_tol=1e-6)
            for value, target in zip(actual, expected)
        ):
            raise ValueError(f"Unsupported training contract {name}: {actual}")
    if contract["body_name"] != "flange" or contract["body_offset"] is not None:
        raise ValueError("Only flange actions with no body offset are supported")
    if not math.isclose(contract["success_pos_threshold"], 0.003):
        raise ValueError("This recorder's insertion threshold is 0.003 m")


def validate_prepared_grasp(fixture, hand_close_width):
    """Require the same held-plug conditions during setup and the pre-enable hold."""
    if (
        not all(math.isfinite(fixture[key]) for key in (
            "grasp_error_m", "grasp_rotation_error_deg", "clearance_m", "finger_joint",
        ))
        or fixture["grasp_error_m"] > 0.003
        or fixture["grasp_rotation_error_deg"] > 5.0
        or not 0.04 <= fixture["clearance_m"] <= 0.06
        or fixture["finger_joint"] <= hand_close_width + 0.02
    ):
        raise RuntimeError("Prepared grasp failed verification before policy enable")


def pose_errors(plug_position, plug_quaternion, socket_mate_position, socket_quaternion, contract):
    """Return the training mate-position metric and its separate 7-point diagnostic.

    This implements the geometry described in DisplayportInsertionEnv's success
    calculation: seven points at the origin and +/- each axis. That environment
    currently determines success using mate-position distance only; the mean
    seven-point distance additionally detects orientation error.
    """
    from flange_action_adapter import multiply, rotate

    def normalized(quaternion):
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError("Invalid quaternion in success measurement")
        return tuple(value / norm for value in quaternion)

    plug_quaternion = normalized(plug_quaternion)
    socket_quaternion = normalized(socket_quaternion)
    plug_mate_rotation = multiply(
        plug_quaternion, normalized(contract["success_plug_goal_rot_inv"]),
    )
    rotated_offset = rotate(plug_quaternion, contract["plug_keypoint_offset"])
    plug_mate = [float(value) + float(offset) for value, offset in zip(plug_position, rotated_offset)]
    position_error = math.dist(plug_mate, socket_mate_position)
    scale = contract["success_keypoint_scale"]
    offsets = [(0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale),
               (-scale, 0, 0), (0, -scale, 0), (0, 0, -scale)]
    distances = []
    for offset in offsets:
        plug_delta = rotate(plug_mate_rotation, offset)
        socket_delta = rotate(socket_quaternion, offset)
        plug_point = [value + delta for value, delta in zip(plug_mate, plug_delta)]
        socket_point = [value + delta for value, delta in zip(socket_mate_position, socket_delta)]
        distances.append(math.dist(plug_point, socket_point))
    dot = abs(sum(a * b for a, b in zip(plug_mate_rotation, socket_quaternion)))
    return {
        "plug_keypoint": plug_mate,
        "position_error_m": position_error,
        "keypoint_error_m": sum(distances) / len(distances),
        "orientation_error_deg": math.degrees(2 * math.acos(min(1.0, dot))),
    }


def rgb_frame_bytes(data):
    """Validate renderer pixels before sending raw RGB bytes to the encoder."""
    import numpy as np

    if data.dtype != np.uint8 or data.shape not in ((720, 1280, 3), (720, 1280, 4)):
        raise ValueError(
            f"Expected uint8 720x1280 RGB(A), got dtype={data.dtype}, shape={data.shape}"
        )
    # The encoder receives pixels through stdin, never generated arguments/code.
    payload = np.ascontiguousarray(data[:, :, :3]).tobytes()
    if len(payload) != 720 * 1280 * 3:
        raise ValueError("Unexpected raw RGB frame byte length")
    return payload


def summarize_policy_sync(records, minimum_steps):
    """Reject missing, stale, future or skipped control-boundary responses."""
    if len(records) < minimum_steps:
        raise ValueError(f"Incomplete policy synchronization: {len(records)} < {minimum_steps}")
    for index, row in enumerate(records):
        stamp, command, wait = row["sim_time_s"], row["command_stamp_s"], row["wait_wall_s"]
        if not all(math.isfinite(value) for value in (stamp, command, wait)) or wait < 0:
            raise ValueError("Invalid policy synchronization timing")
        if abs(stamp - command) > 1e-6:
            raise ValueError("Policy response does not match its simulation boundary")
        if index and abs(stamp - records[index - 1]["sim_time_s"] - 1 / 30) > 1e-6:
            raise ValueError("Skipped or repeated policy synchronization boundary")
    return {"steps": len(records), "policy_hz": 30,
            "maximum_wait_wall_s": max(row["wait_wall_s"] for row in records)}


def run(app, args):
    # Isaac Sim's imports must follow SimulationApp construction.
    import carb
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.types import ArticulationAction
    from scipy.spatial.transform import Rotation
    from training_scene_assets import rebuild

    os.environ["CABLE_POLICY_SYNC_LOG"] = str(args.output / "policy-sync.jsonl")
    os.environ.pop("CABLE_POLICY_SYNC_FAILED", None)
    manifest = json.loads(args.asset_manifest.read_text())
    contract = manifest["contract"]
    validate_contract(contract)
    manager = omni.kit.app.get_app().get_extension_manager()
    # Disable executable USD content until the scene and ScriptNode are verified.
    carb.settings.get_settings().set("/app/omni.graph.scriptnode/opt_in", False)
    manager.set_extension_enabled_immediate("omni.graph.scriptnode", False)
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    for _ in range(20):
        app.update()
    cache_dir = args.asset_manifest.parent / "assets-cache"
    resolved_scene = local_scene_layer(args.verified_scene, cache_dir)
    if not omni.usd.get_context().open_stage(str(resolved_scene)):
        raise RuntimeError(f"Unable to open verified scene {resolved_scene}")
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Scene failed to load")
    rebuild(
        stage, args.asset_manifest, args.asset_manifest.parent / "assets-cache",
        args.output / "asset-verification.json",
    )

    script_attr = stage.GetPrimAtPath(SCRIPT_PATH).GetAttribute("inputs:script")
    script = reviewed_controller_script(verify_script(stage))
    script_attr.Set(script)
    # Only enable execution after SHA256 verification of the original scene and
    # embedded script, plus the two reviewed local controller compatibility edits.
    carb.settings.get_settings().set("/app/omni.graph.scriptnode/opt_in", True)
    manager.set_extension_enabled_immediate("omni.graph.scriptnode", True)

    # Preserve the official launcher's effective physics initialization. In this
    # scene it selects CPU physics despite GPU opinions in the authored USD.
    sim = SimulationContext.instance() or SimulationContext(set_defaults=False)
    if SimulationManager.get_physics_sim_view() is None:
        sim.initialize_physics()
        for _ in range(4):
            app.update()
    if SimulationManager.get_physics_sim_view() is None:
        raise RuntimeError("Isaac Sim physics initialization failed")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        app.update()
    # Complete renderer initialization before creating physics handles or the grasp.
    # Render-only updates warm the annotator without changing physical state.
    camera = rep.create.camera(
        position=(0.79, 0.47, 0.30), look_at=(0.475, 0.125, 0.10),
        focal_length=35, clipping_range=(0.01, 1000),
    )
    product = rep.create.render_product(camera, (1280, 720), force_new=True, name="CableInsertionVideo")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    product_path = str(getattr(product, "path", product))
    if not stage.GetPrimAtPath(product_path).IsValid():
        raise RuntimeError(f"Render product does not exist: {product_path}")
    render_deadline = time.monotonic() + args.ready_timeout
    warmup_frames = 0
    while True:
        sim.render()
        data = np.asarray(annotator.get_data())
        warmup_frames += 1
        if warmup_frames == 1:
            print("RENDER_WARMUP", product_path, "shape", data.shape, flush=True)
        if data.size:
            rgb_frame_bytes(data)
            print("RENDER_READY", product_path, "shape", data.shape, flush=True)
            break
        if time.monotonic() >= render_deadline or warmup_frames >= 180:
            raise RuntimeError(f"RGB annotator failed to initialize: shape={data.shape}, product={product_path}")

    robot = SingleArticulation(ARM_PATH, reset_xform_properties=False)
    robot.initialize()
    plug = SingleRigidPrim(PLUG_PATH, reset_xform_properties=False)
    plug.initialize()
    names = list(robot.dof_names)
    arm = np.array([names.index(f"joint{i}") for i in range(1, 8)], dtype=np.int32)
    controller = robot.get_articulation_controller()
    kp, kd = controller.get_gains()
    kp[arm], kd[arm] = 5000.0, 100.0
    controller.set_gains(kp, kd)
    # Public PhysX tensor APIs provide read-only kinematics without re-executing
    # the USD's Python ScriptNode in the host process.
    physics_view = SimulationManager.get_physics_sim_view().create_articulation_view(ARM_PATH)
    if physics_view.count != 1:
        raise RuntimeError("Expected one robot in the physics articulation view")
    if list(physics_view.shared_metatype.dof_names) != names:
        raise RuntimeError("Robot and physics view joint ordering differ")
    flange_index = list(physics_view.shared_metatype.link_names).index("flange")

    def tensor_array(value):
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def flange_pose():
        transforms = np.asarray(
            tensor_array(physics_view.get_link_transforms()), dtype=np.float64,
        ).reshape(physics_view.count, physics_view.max_links, 7)
        transform = transforms[0, flange_index]
        quaternion = transform[3:7].copy()
        norm = np.linalg.norm(quaternion)
        if not np.all(np.isfinite(transform)) or norm <= 0:
            raise RuntimeError("Invalid flange transform from PhysX")
        return transform[:3].copy(), quaternion / norm

    jacobian_shape = tensor_array(physics_view.get_jacobians()).shape
    if jacobian_shape[1] != int(physics_view.max_links) - 1 or flange_index < 1:
        raise RuntimeError(f"Expected fixed-base Jacobian tensor, got {jacobian_shape}")

    socket_initial = manifest["assets"]["dp_socket"]["init_state"]
    socket_rot = Rotation.from_quat(socket_initial["rot"])
    socket_pos = np.array(socket_initial["pos"]) + socket_rot.apply(contract["socket_keypoint_offset"])
    plug_rot = socket_rot * Rotation.from_quat(contract["success_plug_goal_rot_inv"]).inv()
    plug_offset = np.array(contract["plug_keypoint_offset"])
    grasp_offset = np.array(contract["grasp_offset"])
    grasp_rotation = Rotation.from_quat(contract["grasp_rot_offset"])
    plug_pos = socket_pos - plug_rot.apply(plug_offset) + [0, 0, 0.05]
    target_pos = plug_pos + plug_rot.apply(grasp_offset)
    target_rot = plug_rot * grasp_rotation
    q = robot.get_joint_positions().copy()
    initial_joints = manifest["assets"]["robot"]["init_state"]["joint_pos"]
    q[arm] = [initial_joints[names[index]] for index in arm]
    hold = contract["hand_hold_width"]
    for name, sign in {
        "finger_joint": 1, "left_inner_knuckle_joint": 1,
        "right_inner_knuckle_joint": 1, "right_outer_knuckle_joint": 1,
        "left_outer_finger_joint": -1, "right_outer_finger_joint": -1,
    }.items():
        q[names.index(name)] = sign * hold

    for iteration in range(180):
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        robot.apply_action(ArticulationAction(joint_positions=q))
        sim.step(render=False)
        pos, quat = flange_pose()
        error = np.r_[target_pos - pos, (target_rot * Rotation.from_quat(quat).inv()).as_rotvec()]
        if np.linalg.norm(error[:3]) < 0.0002 and np.linalg.norm(error[3:]) < 0.003:
            break
        jacobian = tensor_array(physics_view.get_jacobians())[0][flange_index - 1][:, arm]
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03**2 * np.eye(6), error)
        q[arm] += np.clip(delta, -0.15, 0.15)
    if np.linalg.norm(error[:3]) > 0.003 or np.linalg.norm(error[3:]) > 0.03:
        raise RuntimeError(f"IK fixture failed: {error}")
    print("FIXTURE_READY", iteration, error.tolist(), flush=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("cable_insertion_recording")
    goal_pub = node.create_publisher(PoseStamped, "/displayport_insertion/goal_pose", 10)
    gate_pub = node.create_publisher(Bool, "/displayport_task_space_policy/enable", 10)
    mode_pub = node.create_publisher(String, "/displayport_insertion/control_mode", 10)
    joint_pub = node.create_publisher(JointState, "/rizon_arm_command", 10)
    grip_pub = node.create_publisher(JointState, "/rizon_gripper_joint_command", 10)
    target_topic = "/displayport_task_space_policy/target_pose"
    target_count = 0
    last_target_wall = None

    def receive_target(_message):
        nonlocal target_count, last_target_wall
        target_count += 1
        last_target_wall = time.monotonic()

    node.create_subscription(PoseStamped, target_topic, receive_target, 10)

    def publish(enabled=False):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, socket_pos)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, socket_rot.as_quat())
        goal_pub.publish(msg)
        gate_pub.publish(Bool(data=enabled))
        mode_pub.publish(String(data="policy" if enabled else "joint"))
        if not enabled:
            command = JointState()
            command.name = [names[index] for index in arm]
            command.position = [float(q[index]) for index in arm]
            joint_pub.publish(command)
        command = JointState()
        command.name = ["finger_joint"]
        command.position = [float(contract["hand_close_width"])]
        grip_pub.publish(command)
        rclpy.spin_once(node, timeout_sec=0)

    encoder = None
    encoder_log = None
    trace = []
    frames = 0
    failure = None
    initial_fixture = None
    policy_sync = None
    wall = time.monotonic()
    try:
        ready_deadline = time.monotonic() + args.ready_timeout
        discovery_updates = 0
        while True:
            publish(False)
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros_like(q))
            robot.apply_action(ArticulationAction(joint_positions=q))
            app.update()
            discovery_updates += 1
            ready = (
                gate_pub.get_subscription_count() > 0
                and goal_pub.get_subscription_count() > 0
                and node.count_publishers(target_topic) > 0
            )
            if discovery_updates >= 90 and ready:
                break
            if time.monotonic() > ready_deadline:
                raise TimeoutError("ROS discovery timed out: start the ready LEAPP runtime and flange adapter")
        print("ROS_GRAPH_READY", flush=True)
        plug.initialize()
        pos, quat = flange_pose()
        actual_plug_rotation = Rotation.from_quat(quat) * grasp_rotation.inv()
        plug_pos = pos - actual_plug_rotation.apply(grasp_offset)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        plug.set_world_pose(plug_pos, np.roll(actual_plug_rotation.as_quat(), 1))
        plug.set_linear_velocity(np.zeros(3))
        plug.set_angular_velocity(np.zeros(3))
        # Release the dynamic plug into the grasp. No weld or post-setup teleport.
        for _ in range(10):
            publish(False)
            app.update()

        plug_position, plug_quaternion = plug.get_world_pose()
        flange_position, flange_quaternion = flange_pose()
        expected_rotation = Rotation.from_quat(flange_quaternion) * grasp_rotation.inv()
        expected_position = flange_position - expected_rotation.apply(grasp_offset)
        metrics = pose_errors(
            plug_position, np.roll(plug_quaternion, -1), socket_pos,
            socket_rot.as_quat(), contract,
        )
        initial_fixture = {
            "grasp_error_m": float(np.linalg.norm(plug_position - expected_position)),
            "grasp_rotation_error_deg": float((
                Rotation.from_quat(np.roll(plug_quaternion, -1)) * expected_rotation.inv()
            ).magnitude() * 180 / math.pi),
            "clearance_m": metrics["plug_keypoint"][2] - float(socket_pos[2]),
            "finger_joint": float(robot.get_joint_positions()[names.index("finger_joint")]),
        }
        print("GRASP_VERIFICATION", json.dumps(initial_fixture), flush=True)
        validate_prepared_grasp(initial_fixture, contract["hand_close_width"])

        encoder_log = (args.output / "encoder.log").open("w")
        encoder = subprocess.Popen(
            [str(args.ffmpeg), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "30", "-i", "-",
             "-an", "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(args.output / "video.mp4")],
            # Argument vector and validated absolute executable, never a shell.
            stdin=subprocess.PIPE, stderr=encoder_log, shell=False,
        )
        start = sim.current_time
        wall = time.monotonic()
        enabled_wall = None
        last_frame = -1
        print("RECORDING", flush=True)
        while sim.current_time - start < args.duration:
            if not app.is_running() or time.monotonic() - wall > args.wall_timeout:
                raise TimeoutError("Simulation closed or exceeded recording wall timeout")
            elapsed = sim.current_time - start
            enabled = elapsed >= 1.0
            if enabled and enabled_wall is None:
                enabled_wall = time.monotonic()
            publish(enabled)
            app.update()
            if os.environ.get("CABLE_POLICY_SYNC_FAILED"):
                raise RuntimeError(os.environ["CABLE_POLICY_SYNC_FAILED"])
            if enabled_wall is not None:
                since = last_target_wall if last_target_wall is not None else enabled_wall
                if time.monotonic() - since > 30:
                    raise TimeoutError("No recent ROS policy targets; inference or adapter is not healthy")
            pos, quat = plug.get_world_pose()
            metrics = pose_errors(
                pos, np.roll(quat, -1), socket_pos, socket_rot.as_quat(), contract,
            )
            flange_pos, flange_quat = flange_pose()
            expected_plug_rotation = Rotation.from_quat(flange_quat) * grasp_rotation.inv()
            trace.append({
                "sim_s": float(sim.current_time - start),
                **metrics,
                "error_m": metrics["position_error_m"],
                "flange": flange_pos.tolist(), "target_count": target_count,
                "finger_joint": float(robot.get_joint_positions()[names.index("finger_joint")]),
                "grasp_error_m": float(np.linalg.norm(pos - (flange_pos - expected_plug_rotation.apply(grasp_offset)))),
                "grasp_rotation_error_deg": float((
                    Rotation.from_quat(np.roll(quat, -1)) * expected_plug_rotation.inv()
                ).magnitude() * 180 / math.pi),
                "clearance_m": metrics["plug_keypoint"][2] - float(socket_pos[2]),
            })
            if not enabled:
                validate_prepared_grasp(trace[-1], contract["hand_close_width"])
            frame_number = int(elapsed * 30)
            if frame_number > last_frame:
                data = np.asarray(annotator.get_data())
                if data.size:
                    encoder.stdin.write(rgb_frame_bytes(data))
                    frames += 1
                    last_frame = frame_number
            if len(trace) % 60 == 0:
                print("PROGRESS", elapsed, "error_m", trace[-1]["error_m"], "targets", target_count, flush=True)
        if target_count < 10:
            raise RuntimeError(f"Too few policy targets to validate an insertion: {target_count}")
        if frames < args.duration * 25:
            raise RuntimeError(f"Incomplete video: {frames} frames for {args.duration}s")
        sync_rows = [json.loads(line) for line in
                     (args.output / "policy-sync.jsonl").read_text().splitlines()]
        # Allow initial ROS mode/gate delivery to take up to two policy periods.
        policy_sync = summarize_policy_sync(sync_rows, round((args.duration - 1) * 30) - 2)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            publish(False)
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0.01)
        finally:
            if encoder is not None:
                if encoder.stdin:
                    try:
                        encoder.stdin.close()
                    except BrokenPipeError:
                        failure = failure or "ffmpeg closed its input early"
                try:
                    encoder.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
                if encoder.returncode:
                    failure = failure or f"ffmpeg exited {encoder.returncode}"
            if encoder_log is not None:
                encoder_log.close()
            (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            terminal = trace[-1]["error_m"] if trace else None
            valid = failure is None and target_count >= 10 and frames >= args.duration * 25
            intervals = [
                (record["sim_s"] - previous["sim_s"], record["error_m"] < 0.003)
                for previous, record in zip(trace, trace[1:]) if record["sim_s"] > 1.0
            ]
            hold_seconds = sum(dt for dt, success in intervals if success)
            policy_seconds = sum(dt for dt, _ in intervals)
            trailing_hold = 0.0
            for dt, success in reversed(intervals):
                if not success:
                    break
                trailing_hold += dt
            result = {
                "valid": valid, "success": bool(valid and terminal < 0.003),
                "success_definition": "Final mate-position distance < 0.003 m (training success metric); one prepared-grasp insertion attempt",
                "keypoint_success": bool(valid and trace[-1]["keypoint_error_m"] < 0.003),
                "failure": failure, "requested_duration_s": args.duration,
                "recorded_sim_seconds": trace[-1]["sim_s"] if trace else 0,
                "final_error_m": terminal,
                "final_position_error_m": terminal,
                "final_keypoint_error_m": trace[-1]["keypoint_error_m"] if trace else None,
                "final_orientation_error_deg": trace[-1]["orientation_error_deg"] if trace else None,
                "minimum_error_m": min((item["error_m"] for item in trace), default=None),
                "within_threshold_seconds": hold_seconds,
                "within_threshold_fraction": hold_seconds / policy_seconds if policy_seconds else 0,
                "terminal_hold_seconds": trailing_hold,
                "target_count": target_count, "targets_received": target_count, "frames": frames, "video_fps": 30,
                "wall_seconds": time.monotonic() - wall,
                "policy_sync": policy_sync,
                "video": str(args.output / "video.mp4"),
                "controller": "Training OSC, fixed-base Jacobian corrected, flange-relative action adapter",
                "fixture": "5 cm above socket; plug held by contact, no weld; setup before recording",
                "initial_fixture": initial_fixture,
                "asset_manifest": str(args.asset_manifest),
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("RESULT", json.dumps(result), flush=True)
            node.destroy_node()
            timeline.stop()
    if failure:
        raise RuntimeError(failure)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    environment(args.domain_id)
    # Download and authenticate the executable scene asset before starting Kit.
    args.verified_scene = fetch_scene(args.asset_manifest.parent / "assets-cache", args.scene_url)
    runtime = {
        "python": sys.executable, "isaac_sim_root": str(SIM_ROOT),
        "isaac_sim_build": (SIM_ROOT / "VERSION").read_text().strip(),
        "scene_url": args.scene_url, "scene_sha256": SCENE_SHA256,
        "script_sha256": SCRIPT_SHA256, "executed_script_sha256": COMBINED_SCRIPT_SHA256, "ros_domain_id": args.domain_id,
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})
    exit_status = 0
    try:
        run(app, args)
    except BaseException as exc:
        exit_status = 1
        traceback.print_exc()
        result_path = args.output / "result.json"
        if not result_path.exists():
            result_path.write_text(json.dumps({
                "valid": False, "success": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }, indent=2) + "\n")
    finally:
        # Kit's default fast shutdown terminates the process; explicitly preserve
        # errors rather than allowing its default zero exit to mask exceptions.
        app.close(exit_code=exit_status)
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
