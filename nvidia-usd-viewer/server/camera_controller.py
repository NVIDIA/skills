"""Orbit / pan / zoom camera controller.

Computes a 4×4 world-space camera transform from spherical coordinates
around a target point. Writes via the renderer's write_camera_transform().

Input events are forwarded by input_router from the NVST binary channel.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return 4×4 row-major world-from-camera matrix."""
    z = eye - target
    z_len = float(np.linalg.norm(z))
    z = (np.array([0.0, 0.0, 1.0]) if z_len < 1e-10 else z / z_len)
    x = np.cross(up, z)
    x_len = float(np.linalg.norm(x))
    x = (np.array([1.0, 0.0, 0.0]) if x_len < 1e-10 else x / x_len)
    y = np.cross(z, x)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = x
    mat[1, :3] = y
    mat[2, :3] = z
    mat[3, :3] = eye
    return mat


class CameraController:
    def __init__(self, width: int, height: int, write_fn: Callable[[np.ndarray], None]) -> None:
        self._write = write_fn
        self._target    = np.array([0.0, 0.0, 0.0])
        self._distance  = 500.0
        self._azimuth   = 0.0     # radians
        self._elevation = 0.3     # radians (slightly above horizon)
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._orbiting  = False
        self._panning   = False
        self._push()

    # ------------------------------------------------------------------
    # Input handlers (enqueued from input_router, executed on render thread)
    # ------------------------------------------------------------------

    def on_mouse_button(self, button: int, action: int, mods: int, x: float, y: float) -> None:
        pressed = (action == 1)
        if button == 0:
            self._orbiting = pressed
        elif button == 1:
            self._panning  = pressed
        if pressed:
            self._last_x, self._last_y = x, y
        else:
            self._last_x = self._last_y = None

    def on_mouse_move(self, x: float, y: float) -> None:
        if self._last_x is None:
            return
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x, self._last_y = x, y
        if self._orbiting:
            self._azimuth   -= dx * 0.005
            self._elevation  = float(np.clip(self._elevation - dy * 0.005, -1.5, 1.5))
            self._push()
        elif self._panning:
            mat   = self._matrix()
            right = mat[0, :3]
            up    = mat[1, :3]
            s = self._distance * 0.001
            self._target -= right * (dx * s)
            self._target += up    * (dy * s)
            self._push()

    def on_mouse_scroll(self, delta: float) -> None:
        self._distance = max(1.0, self._distance * (1.0 - delta * 0.1))
        self._push()

    def fit_to_bounds(self, center: np.ndarray, radius: float) -> None:
        self._target   = center.copy()
        self._distance = radius * 2.5
        self._push()

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "target":    self._target.tolist(),
            "distance":  self._distance,
            "azimuth":   self._azimuth,
            "elevation": self._elevation,
        }

    def set_state(self, state: dict) -> None:
        self._target    = np.array(state.get("target",    [0, 0, 0]), dtype=float)
        self._distance  = float(state.get("distance",  500.0))
        self._azimuth   = float(state.get("azimuth",   0.0))
        self._elevation = float(state.get("elevation", 0.3))
        self._push()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _matrix(self) -> np.ndarray:
        x = self._distance * math.cos(self._elevation) * math.sin(self._azimuth)
        y = self._distance * math.sin(self._elevation)
        z = self._distance * math.cos(self._elevation) * math.cos(self._azimuth)
        eye = self._target + np.array([x, y, z])
        return _look_at(eye, self._target, np.array([0.0, 1.0, 0.0]))

    def _push(self) -> None:
        self._write(self._matrix())
