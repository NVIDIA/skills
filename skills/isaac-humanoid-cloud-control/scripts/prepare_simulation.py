#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the paused G1 stage through the loopback isaac-sim-remote adapter.

Usage:
    Send this file with the scene adapter's --file option after app readiness.
Arguments:
    None; uses the simulator process's isolated ROS environment.
Output:
    JSON readback of stage, ROS Context domains, timeline range and loop state.
Exit codes:
    Remote execution fails on an unexpected stage, playing timeline or bad readback.
"""

import json
import os

import omni.timeline
import omni.usd


SIM_STAGE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.0/Isaac/Samples/ROS2/Scenario/g1_ros_navigation.usd"
)


def prepare(context, timeline, environment):
    stage = context.get_stage()
    if stage is None or stage.GetRootLayer().identifier != SIM_STAGE:
        raise RuntimeError("the production G1 stage must be loaded")
    if context.get_stage_loading_status()[2] != 0 or timeline.is_playing():
        raise RuntimeError("stage loading must finish and the timeline must be paused")
    raw = environment.get("ROS_DOMAIN_ID", "")
    if not raw.isdecimal() or not 1 <= int(raw) <= 232:
        raise ValueError("set an isolated ROS_DOMAIN_ID between 1 and 232")
    if (environment.get("ROS_LOCALHOST_ONLY") != "1"
            or environment.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
            or environment.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp"):
        raise ValueError("the simulator requires loopback-only Fast DDS discovery")
    for key in ("ROS_STATIC_PEERS", "FASTRTPS_DEFAULT_PROFILES_FILE",
                "FASTDDS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI"):
        if environment.get(key):
            raise ValueError(f"custom discovery override is unsupported: {key}")
    domain = int(raw)
    contexts = []
    for prim in stage.Traverse():
        use_env = prim.GetAttribute("inputs:useDomainIDEnvVar")
        context_domain = prim.GetAttribute("inputs:domain_id")
        if use_env and context_domain:
            contexts.append((str(prim.GetPath()), use_env, context_domain))
    if not contexts:
        raise RuntimeError("no authored ROS Context domains were found")
    for path, use_env, context_domain in contexts:
        use_env.Set(True)
        context_domain.Set(domain)
        if use_env.Get() is not True or context_domain.Get() != domain:
            raise RuntimeError(f"ROS Context domain readback failed: {path}")
    timeline.set_end_time(86400.0)
    timeline.set_looping(False)
    timeline.commit()
    if (timeline.is_playing() or timeline.get_end_time() != 86400.0
            or timeline.is_looping()):
        raise RuntimeError("paused timeline range/loop readback failed")
    return {
        "stage": stage.GetRootLayer().identifier,
        "playing": False,
        "domain": domain,
        "ros_contexts": [path for path, _, _ in contexts],
        "end_time": timeline.get_end_time(),
        "looping": timeline.is_looping(),
    }


if not globals().get("_PREPARATION_IMPORT_ONLY", False):
    print(json.dumps(prepare(omni.usd.get_context(),
                             omni.timeline.get_timeline_interface(), os.environ)))
