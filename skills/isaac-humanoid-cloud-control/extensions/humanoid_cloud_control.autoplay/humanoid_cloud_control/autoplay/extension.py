# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the G1 timeline without starting physics."""

import asyncio
import os
from pathlib import Path
import runpy

import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._task = asyncio.ensure_future(self._prepare_when_ready())

    async def _prepare_when_ready(self) -> None:
        helper = Path(__file__).resolve().parents[4] / "scripts/prepare_simulation.py"
        preparation = runpy.run_path(str(helper), init_globals={"_PREPARATION_IMPORT_ONLY": True})
        context = omni.usd.get_context()
        for _ in range(3600):
            stage = context.get_stage()
            identifier = stage.GetRootLayer().identifier if stage else ""
            if (identifier == preparation["SIM_STAGE"]
                    and context.get_stage_loading_status()[2] == 0):
                preparation["prepare"](context, omni.timeline.get_timeline_interface(), os.environ)
                return
            await omni.kit.app.get_app().next_update_async()
        raise RuntimeError("the production G1 stage did not finish loading")

    def on_shutdown(self) -> None:
        task = getattr(self, "_task", None)
        if task and not task.done():
            task.cancel()
