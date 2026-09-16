# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore the simple small warehouse Carter Isaac Sim scene.

This script is sent to Isaac Sim through isaacsim_send.py. It runs inside the
Isaac Sim Python server, not on the host.
"""

import math
import os

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.storage.native import get_assets_root_path
import numpy as np
import omni.timeline
import omni.usd

DEFAULT_WAREHOUSE_USD_URI = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Environments/Simple_Warehouse/warehouse.usd"
)
WAREHOUSE_USD_URI = os.environ.get(
    "WAREHOUSE_USD_URI", DEFAULT_WAREHOUSE_USD_URI
)
ROBOT_PRIM = "/World/Nova_Carter_ROS"
ARTICULATION_ROOT = "/World/Nova_Carter_ROS/chassis_link"
SAFE_X = -3.3
SAFE_Y = 1.3
SAFE_Z = 0.0
SAFE_YAW = 0.0001
POSE_TOLERANCE_M = 0.05
CLOCK_GRAPH = "/ROS2ClockGraph"


async def ensure_stage():
    """Open and verify the exact warehouse URI selected by the runner."""
    ctx = omni.usd.get_context()
    stage_url = ctx.get_stage_url() or ""
    if stage_url != WAREHOUSE_USD_URI:
        warehouse_usd = WAREHOUSE_USD_URI
        result = (False, None)
        for attempt in range(1, 4):
            result = await stage_utils.open_stage_async(warehouse_usd)
            await app_utils.update_app_async(steps=120)
            print(f"opened_stage_attempt={attempt} result={result} url={warehouse_usd}")
            if result[0]:
                break
            await app_utils.update_app_async(steps=180)
        if not result[0]:
            raise RuntimeError(f"Failed to open warehouse stage: {warehouse_usd}")
    else:
        print(f"stage_already_open={stage_url}")
    stage_url = ctx.get_stage_url() or ""
    if stage_url != WAREHOUSE_USD_URI:
        raise RuntimeError(
            f"Active stage does not match requested warehouse URI: "
            f"active={stage_url!r} requested={WAREHOUSE_USD_URI!r}"
        )
    stage = stage_utils.get_current_stage()
    if stage is None:
        raise RuntimeError("No current USD stage after warehouse restore")
    default_prim = stage.GetDefaultPrim()
    print(
        "stage_verified="
        f"{stage_url} default_prim={default_prim.GetPath() if default_prim else None}"
    )
    return stage


async def ensure_robot(stage):
    """Add the Nova Carter ROS asset to the stage if it is missing."""
    if stage.GetPrimAtPath(ROBOT_PRIM).IsValid():
        print(f"robot_present={ROBOT_PRIM}")
    else:
        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Could not resolve Isaac Sim assets root")
        robot_usd = assets_root + "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"
        stage_utils.add_reference_to_stage(robot_usd, ROBOT_PRIM)
        await app_utils.update_app_async(steps=180)
        if not stage.GetPrimAtPath(ROBOT_PRIM).IsValid():
            raise RuntimeError(f"Failed to add robot reference at {ROBOT_PRIM}")
        print(f"robot_added={ROBOT_PRIM} usd={robot_usd}")

    if not stage.GetPrimAtPath(ARTICULATION_ROOT).IsValid():
        raise RuntimeError(f"Nova Carter articulation root missing: {ARTICULATION_ROOT}")
    print(f"robot_articulation_verified={ARTICULATION_ROOT}")


def configure_sensor_graphs():
    """Apply the sensor toggles used by the mission-client tutorial."""
    import omni.graph.core as og

    toggles = {
        f"{ROBOT_PRIM}/ros_lidars/front_2d_lidar_render_product.inputs:enabled": True,
        f"{ROBOT_PRIM}/ros_lidars/publish_front_2d_lidar_scan.inputs:topicName": "scan",
        f"{ROBOT_PRIM}/ros_lidars/front_3d_lidar_render_product.inputs:enabled": False,
        f"{ROBOT_PRIM}/front_hawk/left_camera_render_product.inputs:enabled": False,
        f"{ROBOT_PRIM}/front_hawk/right_camera_render_product.inputs:enabled": False,
    }
    for attr, value in toggles.items():
        try:
            og.Controller.set(og.Controller.attribute(attr), value)
            print(f"set_graph_attr={attr} value={value}")
        except Exception as exc:
            print(
                f"set_graph_attr_warning={attr}: {type(exc).__name__}: {exc}")


def ensure_ros_clock_graph(stage):
    """Publish Isaac simulation time for ROS nodes using use_sim_time."""
    if stage.GetPrimAtPath(CLOCK_GRAPH).IsValid():
        print(f"ros_clock_graph_present={CLOCK_GRAPH}")
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": CLOCK_GRAPH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
        },
    )
    print(f"ros_clock_graph_created={CLOCK_GRAPH}")


async def reset_robot_pose():
    """Reset Carter's articulation to a known collision-free start pose."""
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        await app_utils.update_app_async(steps=30)

    quat = np.array([[
        math.cos(SAFE_YAW / 2.0),
        0.0,
        0.0,
        math.sin(SAFE_YAW / 2.0),
    ]])
    pos = np.array([[SAFE_X, SAFE_Y, SAFE_Z]])
    robot = Articulation(ARTICULATION_ROOT)
    robot.set_world_poses(positions=pos, orientations=quat)
    await app_utils.update_app_async(steps=60)
    world_pos, world_quat = robot.get_world_poses()
    actual_pos = world_pos.numpy()[0]
    expected_pos = np.array([SAFE_X, SAFE_Y, SAFE_Z])
    if np.linalg.norm(actual_pos - expected_pos) > POSE_TOLERANCE_M:
        raise RuntimeError(
            "Carter reset pose verification failed: "
            f"expected={expected_pos.tolist()} actual={actual_pos.tolist()}"
        )
    if not timeline.is_playing():
        raise RuntimeError("Isaac timeline stopped during Carter reset")
    print(f"robot_reset={ARTICULATION_ROOT}")
    print(f"robot_world_pos={actual_pos.tolist()}")
    print(f"robot_world_quat={world_quat.numpy()[0].tolist()}")
    print("timeline_playing=True")


async def restore_camera():
    """Set a below-ceiling three-quarter view across the route area."""
    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=[-6.0, -8.0, 3.6],
        target=[-2.4, -2.0, 0.4],
    )
    await app_utils.update_app_async(steps=30)
    print("camera=below_ceiling_route_view")


async def main():
    """Restore all Isaac Sim scene state required for the showcase."""
    stage = await ensure_stage()
    await ensure_robot(stage)
    ensure_ros_clock_graph(stage)
    configure_sensor_graphs()
    await reset_robot_pose()
    await restore_camera()
    print(f"warehouse_usd_uri={WAREHOUSE_USD_URI}")
    print("small_warehouse_nova_carter_restore_complete=True")


await main()
