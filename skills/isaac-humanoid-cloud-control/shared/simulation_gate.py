# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed simulation evidence policy; no ROS imports or side effects."""

SIM_STAGE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.0/Isaac/Samples/ROS2/Scenario/g1_ros_navigation.usd"
)
SIM_PLUGIN = "isaac_ros_deploy_ros2_control/TopicBasedSystemInterface"


def require_local_domain(environment) -> int:
    """Require explicit, non-default and loopback-only ROS discovery."""
    raw = environment.get("ROS_DOMAIN_ID", "")
    if not raw.isdecimal() or not 1 <= int(raw) <= 232:
        raise ValueError("set an isolated ROS_DOMAIN_ID between 1 and 232")
    if (environment.get("ROS_LOCALHOST_ONLY") != "1"
            or environment.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"):
        raise ValueError("loopback-only ROS discovery is required in every demo process")
    for key in ("ROS_STATIC_PEERS", "FASTRTPS_DEFAULT_PROFILES_FILE",
                "FASTDDS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI"):
        if environment.get(key):
            raise ValueError(f"custom discovery override is unsupported: {key}")
    if environment.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise ValueError("the reviewed simulation transport is rmw_fastrtps_cpp")
    return int(raw)


def require_simulation(identity, domain, components, providers, use_sim_time):
    """Reject absent, mixed, physical, or ambiguous runtime observations."""
    if (identity.get("stage") != SIM_STAGE or identity.get("playing") is not True
            or identity.get("domain") != domain
            or identity.get("localhost_only") != "1"
            or identity.get("discovery_range") != "LOCALHOST"):
        raise ValueError("live Isaac Sim stage, playing timeline and isolated domain must agree")
    if components != [("IsaacSimSystem", SIM_PLUGIN)]:
        raise ValueError("controller manager must expose only the Isaac Sim hardware interface")
    if providers != [("isaacsim_launcher", "/")]:
        raise ValueError("gantry service must have exactly one Isaac Sim launcher provider")
    if use_sim_time is not True:
        raise ValueError("controller manager must use simulation time")
