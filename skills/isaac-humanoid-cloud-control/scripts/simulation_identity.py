#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only Isaac Sim identity probe; run through the isaac-sim-remote adapter.

No arguments. Prints a JSON observation of the live stage, timeline and ROS
context. Remote execution fails if Kit or its initialized ROS context is absent.
"""

import json
import os

import omni.timeline
import omni.usd
import rclpy

stage = omni.usd.get_context().get_stage()
if stage is None or not rclpy.ok():
    raise RuntimeError("Isaac Sim stage and initialized ROS context are required")
print(json.dumps({
    "stage": stage.GetRootLayer().identifier,
    "playing": omni.timeline.get_timeline_interface().is_playing(),
    "domain": rclpy.get_default_context().get_domain_id(),
    "localhost_only": os.environ.get("ROS_LOCALHOST_ONLY"),
    "discovery_range": os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE"),
}))
