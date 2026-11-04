#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode flange-relative actions while preserving the policy's TCP observation.

Run in the ROS container after prepare.py validates the training contract:
arm_action.body_name=flange, body_offset=null, TCP offset=(0, 0, 0.1925).
The stock decoder must publish to a separate unused topic, so exactly this
adapter supplies /displayport_task_space_policy/target_pose.
"""

import argparse
import json
import math
from pathlib import Path


def multiply(a, b):
    """Quaternion product in ROS xyzw order."""
    x, y, z, w = a
    bx, by, bz, bw = b
    return (
        w * bx + x * bw + y * bz - z * by,
        w * by - x * bz + y * bw + z * bx,
        w * bz + x * by - y * bx + z * bw,
        w * bw - x * bx - y * by - z * bz,
    )


def rotate(quaternion, vector):
    x, y, z, w = quaternion
    return multiply(multiply(quaternion, (*vector, 0)), (-x, -y, -z, w))[:3]


def decode(message, scale):
    """Recover flange position, then apply a world-frame delta pose."""
    position = message.observation_pose.position
    rotation = message.observation_pose.orientation
    quaternion = (rotation.x, rotation.y, rotation.z, rotation.w)
    delta = message.delta_position
    axis_angle = message.delta_axis_angle
    values = (*quaternion, position.x, position.y, position.z,
              delta.x, delta.y, delta.z, axis_angle.x, axis_angle.y, axis_angle.z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite policy action or observation")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("Invalid zero observation quaternion")
    quaternion = tuple(value / norm for value in quaternion)
    offset = rotate(quaternion, (0, 0, 0.1925))
    vector = [scale * axis_angle.x, scale * axis_angle.y, scale * axis_angle.z]
    angle = math.sqrt(sum(value * value for value in vector))
    if angle > 1e-12:
        delta_rotation = tuple(value * math.sin(angle / 2) / angle for value in vector) + (math.cos(angle / 2),)
    else:
        delta_rotation = (0, 0, 0, 1)
    target_position = [
        position.x - offset[0] + scale * delta.x,
        position.y - offset[1] + scale * delta.y,
        position.z - offset[2] + scale * delta.z,
    ]
    return target_position, multiply(delta_rotation, quaternion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-blend", type=float, default=1.0)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    if not 0 < args.action_blend <= 1.0:
        parser.error("--action-blend must be in (0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from isaac_ros_deploy_interfaces.msg import CartesianPoseDeltaCommand

    rclpy.init()
    node = rclpy.create_node("flange_action_adapter")
    publisher = node.create_publisher(PoseStamped, "/displayport_task_space_policy/target_pose", 10)
    count = 0
    with args.output.open("w") as output:
        def callback(message):
            nonlocal count
            position, quaternion = decode(message, args.action_blend)
            target = PoseStamped()
            target.header = message.header
            target.pose.position.x, target.pose.position.y, target.pose.position.z = position
            target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = quaternion
            publisher.publish(target)
            count += 1
            output.write(json.dumps({
                "stamp_s": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                "position": position, "quaternion_xyzw": quaternion,
            }) + "\n")
            if count % 30 == 0:
                output.flush()
                print("ADAPTED_COMMANDS", count, flush=True)

        node.create_subscription(
            CartesianPoseDeltaCommand, "/displayport_task_space_policy/pose_delta_command", callback, 10,
        )
        print("FLANGE_ADAPTER_READY", flush=True)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
