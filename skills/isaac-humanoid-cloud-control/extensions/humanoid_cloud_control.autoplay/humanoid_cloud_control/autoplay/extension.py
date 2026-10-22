# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A narrow, simulation-only Isaac Sim timeline starter."""

import asyncio
import os

import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd


class Extension(omni.ext.IExt):
    """Start the timeline after Kit has had a chance to load the requested USD."""

    def on_startup(self, ext_id: str) -> None:
        self._task = asyncio.ensure_future(self._play_when_ready())

    async def _play_when_ready(self) -> None:
        # Wait for the requested scenario and all of its payloads before a
        # grace period that allows the staged ROS launch to begin.
        context = omni.usd.get_context()
        for _ in range(3600):
            stage = context.get_stage()
            identifier = stage.GetRootLayer().identifier if stage else ""
            if "g1_ros_navigation" in identifier and context.get_stage_loading_status()[2] == 0:
                break
            await omni.kit.app.get_app().next_update_async()
        else:
            return

        # Some published USDs author the ROS 2 Context node with an explicit
        # domain of zero. Apply the demo's environment-selected domain to the
        # in-memory stage before the graph first evaluates.
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        for prim in stage.Traverse():
            use_env = prim.GetAttribute("inputs:useDomainIDEnvVar")
            context_domain = prim.GetAttribute("inputs:domain_id")
            if use_env and context_domain:
                use_env.Set(True)
                context_domain.Set(domain_id)

        for _ in range(900):
            await omni.kit.app.get_app().next_update_async()
        timeline = omni.timeline.get_timeline_interface()
        # The sample stage carries a short animation range. Extend it instead
        # of looping so ROS simulation time stays monotonic for long demos.
        timeline.set_end_time(86400.0)
        timeline.play()

    def on_shutdown(self) -> None:
        task = getattr(self, "_task", None)
        if task and not task.done():
            task.cancel()
