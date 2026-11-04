# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append to the NVIDIA tutorial USD's ScriptNode in memory.

The ScriptNode supplies np, ArticulationAction, and the helper functions used
below. prepare.py validates these fixed training gains and the assumptions:
no inertia decoupling/nullspace, gravity disabled, flange control. Inference
continues to run in ROS Node LEAPP/Triton, outside the simulator process.
"""


def _cartesian_joint_command(db, dt):
    state = db.per_instance_state
    record = state.latest_cartesian_record
    if record is None:
        return None
    robot = state.robot
    arm_names = [str(name) for name in _as_list(db.inputs.armJointNames)]
    indices = np.asarray([state.dof_index[name] for name in arm_names], dtype=np.int32)
    controller = robot.get_articulation_controller()
    if not getattr(state, "osc_active", False):
        kp, kd = controller.get_gains()
        state.osc_old_gains = (kp.copy(), kd.copy())
        kp, kd = kp.copy(), kd.copy()
        kp[indices], kd[indices] = 0, 0
        controller.set_gains(kp, kd)
        state.osc_active = True
        print("OSC_ACTIVE training gains; arm position drives disabled", flush=True)

    message = record["msg"]
    target = np.array([
        message.pose.position.x, message.pose.position.y, message.pose.position.z,
    ])
    target_quaternion = _normalize_quat_xyzw([
        message.pose.orientation.x, message.pose.orientation.y,
        message.pose.orientation.z, message.pose.orientation.w,
    ])
    position, quaternion = _get_link_pose_xyzw(robot, str(db.inputs.flangeFrameName))
    jacobian = _link_jacobian(robot, str(db.inputs.flangeFrameName), indices)
    quaternion_error = _quat_multiply_xyzw(
        target_quaternion, _quat_conjugate_xyzw(quaternion),
    )
    if quaternion_error[3] < 0:
        quaternion_error = -quaternion_error
    sine_half = float(np.linalg.norm(quaternion_error[:3]))
    if sine_half > 1e-9:
        rotation_error = quaternion_error[:3] * (
            2 * np.arctan2(sine_half, quaternion_error[3]) / sine_half
        )
    else:
        rotation_error = 2 * quaternion_error[:3]
    error = np.r_[target - position, rotation_error]
    velocity = jacobian @ np.asarray(_robot_get_joint_velocities(robot))[indices]
    wrench = (
        np.array([300, 300, 300, 30, 30, 30]) * error
        - np.array([35, 35, 35, 1.1, 1.1, 1.1]) * velocity
    )
    efforts = jacobian.T @ wrench
    limits = np.asarray(robot.dof_properties["maxEffort"])[indices]
    efforts = np.clip(efforts, -limits, limits)
    robot.apply_action(ArticulationAction(
        joint_efforts=efforts.astype(np.float32), joint_indices=indices,
    ))
    return [], [], []


_original_compute = compute


def compute(db):
    state = db.per_instance_state
    if getattr(state, "osc_active", False) and getattr(state, "control_mode", "joint") != "policy":
        controller = state.robot.get_articulation_controller()
        controller.set_gains(*state.osc_old_gains)
        indices = np.array([
            state.dof_index[str(name)] for name in _as_list(db.inputs.armJointNames)
        ], dtype=np.int32)
        state.robot.apply_action(ArticulationAction(
            joint_efforts=np.zeros(len(indices), dtype=np.float32), joint_indices=indices,
        ))
        state.osc_active = False
    result = _original_compute(db)
    _synchronize_policy(db)
    return result


def _synchronize_policy(db):
    """Hold physics at each policy boundary until ROS returns that step's action."""
    import json
    import os
    import time

    state = db.per_instance_state
    if getattr(state, "control_mode", "joint") != "policy":
        return
    sim_time = float(db.inputs.simulationTime)
    dt = float(db.inputs.deltaSimulationTime)
    tick = round(sim_time * 30)
    if abs(sim_time - tick / 30) > dt * 0.25:
        return
    if getattr(state, "cable_policy_tick", None) == tick:
        return
    state.cable_policy_tick = tick
    started = time.monotonic()
    deadline = started + 5.0
    stamp = None
    while True:
        record = state.latest_cartesian_record
        if record is not None:
            header = record["msg"].header.stamp
            stamp = header.sec + header.nanosec * 1e-9
            if stamp > sim_time + 1e-6:
                reason = f"Future policy timestamp {stamp:.6f} at simulation time {sim_time:.6f}"
                os.environ["CABLE_POLICY_SYNC_FAILED"] = reason
                raise RuntimeError(reason)
            if stamp >= sim_time - 1e-6:
                break
        if time.monotonic() >= deadline:
            reason = f"Policy response timed out at simulation time {sim_time:.6f}; latest {stamp}"
            os.environ["CABLE_POLICY_SYNC_FAILED"] = reason
            raise TimeoutError(reason)
        state.rclpy.spin_once(state.node, timeout_sec=0.01)
    # _original_compute publishes the observation/clock before returning. Apply
    # the response now, while this physics step is still held at that boundary.
    _cartesian_joint_command(db, dt)
    log_path = os.environ.get("CABLE_POLICY_SYNC_LOG")
    if log_path:
        if not hasattr(state, "cable_policy_log"):
            state.cable_policy_log = open(log_path, "w")
        state.cable_policy_log.write(json.dumps({
            "sim_time_s": sim_time, "command_stamp_s": stamp,
            "wait_wall_s": time.monotonic() - started,
        }) + "\n")
        state.cable_policy_log.flush()
