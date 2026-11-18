# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sidecar `nvidia-smi` sampling of power and clocks during the measured forward.

Copy into `$WORKDIR/bench/`. Start a `GpuSampler` just before the timed call and
stop it just after; it polls on its own thread, so nothing here runs inside the
timing window. Store the summary as the row's `gpu_telemetry`.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
from typing import Any

_QUERY = (
    "power.draw,"
    "clocks.current.sm,"
    "clocks.current.graphics,"
    "clocks.current.memory,"
    "pstate,"
    "clocks_event_reasons.active,"
    "clocks_event_reasons.gpu_idle,"
    "clocks_event_reasons.sw_power_cap,"
    "clocks_event_reasons.hw_slowdown,"
    "clocks_event_reasons.sw_thermal_slowdown,"
    "clocks_event_reasons.applications_clocks_setting"
)
_KEYS = [
    "power_draw_w",
    "clocks_sm_mhz",
    "clocks_graphics_mhz",
    "clocks_memory_mhz",
    "pstate",
    "event_reasons",
    "gpu_idle",
    "sw_power_cap",
    "hw_slowdown",
    "sw_thermal_slowdown",
    "applications_clocks",
]


def _parse_number(text: str) -> float | None:
    token = text.strip()
    if not token or token.upper() in {"[N/A]", "N/A", "NA"}:
        return None
    if "deprecated" in token.lower():
        return None
    try:
        return float(token.split()[0].replace(",", ""))
    except ValueError:
        return None


# A hung nvidia-smi must not hold the sampler thread past the sample it polls.
SMI_TIMEOUT_S = 10.0


def query_gpu_snapshot() -> dict[str, Any]:
    """One nvidia-smi row. Host-side; never call inside timed_forward."""
    raw = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
        text=True,
        shell=False,
        timeout=SMI_TIMEOUT_S,
    ).strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != len(_KEYS):
        raise RuntimeError(f"nvidia-smi field count {len(parts)} != {len(_KEYS)}: {raw!r}")
    row: dict[str, Any] = {}
    for key, value in zip(_KEYS, parts, strict=True):
        if key in {
            "pstate",
            "event_reasons",
            "gpu_idle",
            "sw_power_cap",
            "hw_slowdown",
            "sw_thermal_slowdown",
            "applications_clocks",
        }:
            row[key] = value
        else:
            row[key] = _parse_number(value)
    return row


def _busy(row: dict[str, Any]) -> bool:
    return str(row.get("gpu_idle", "")).lower() != "active"


def _minmaxmean(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def summarize_polls(polls: list[dict[str, Any]]) -> dict[str, Any]:
    busy = [p for p in polls if _busy(p)]
    throttle = []
    for key in ("sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "applications_clocks"):
        if any(str(p.get(key, "")).lower() == "active" for p in busy):
            throttle.append(key)
    return {
        "n_polls": len(polls),
        "n_busy": len(busy),
        "power_draw_w": _minmaxmean([p["power_draw_w"] for p in busy if p.get("power_draw_w") is not None]),
        "clocks_sm_mhz": _minmaxmean([p["clocks_sm_mhz"] for p in busy if p.get("clocks_sm_mhz") is not None]),
        "clocks_graphics_mhz": _minmaxmean(
            [p["clocks_graphics_mhz"] for p in busy if p.get("clocks_graphics_mhz") is not None]
        ),
        "clocks_memory_mhz": _minmaxmean(
            [p["clocks_memory_mhz"] for p in busy if p.get("clocks_memory_mhz") is not None]
        ),
        "throttle": throttle,
    }


class GpuSampler:
    """Sidecar nvidia-smi polls. Start just before timed_forward, stop after."""

    def __init__(self, interval_s: float = 0.2, stop_timeout_s: float = SMI_TIMEOUT_S + 1.0) -> None:
        self.interval_s = interval_s
        self.stop_timeout_s = stop_timeout_s
        self.polls: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # A thread left over from an earlier sample would append to this one's polls.
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GPU sampler is already running; call stop() first")
        self.polls = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.stop_timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("GPU sampler did not stop; nvidia-smi is not returning")
            self._thread = None
        return summarize_polls(self.polls)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.polls.append(query_gpu_snapshot())
            except (OSError, subprocess.SubprocessError, RuntimeError):
                pass
            self._stop.wait(self.interval_s)


def aggregate_file_telemetry(samples: dict[str, Any]) -> dict[str, Any]:
    """Min of mins, max of maxes, mean of means over ok rows."""
    rows = [row["gpu_telemetry"] for row in samples.values() if row.get("status") == "ok" and row.get("gpu_telemetry")]
    throttle: set[str] = set()
    n_busy_total = 0
    for tel in rows:
        n_busy_total += int(tel.get("n_busy") or 0)
        throttle.update(tel.get("throttle") or [])

    def collect(field: str) -> dict[str, float] | None:
        mins: list[float] = []
        maxs: list[float] = []
        means: list[float] = []
        for tel in rows:
            obj = tel.get(field)
            if not obj:
                continue
            if obj.get("min") is not None:
                mins.append(float(obj["min"]))
            if obj.get("max") is not None:
                maxs.append(float(obj["max"]))
            if obj.get("mean") is not None:
                means.append(float(obj["mean"]))
        if not means:
            return None
        return {"min": min(mins), "max": max(maxs), "mean": statistics.fmean(means)}

    return {
        "n_ok": len(rows),
        "n_busy_total": n_busy_total,
        "power_draw_w": collect("power_draw_w"),
        "clocks_sm_mhz": collect("clocks_sm_mhz"),
        "clocks_graphics_mhz": collect("clocks_graphics_mhz"),
        "clocks_memory_mhz": collect("clocks_memory_mhz"),
        "throttle": sorted(throttle),
    }
