"""RGBA8 → BGRA8 conversion.

ovstream expects BGRA8; ovrtx LdrColor output is RGBA8.
Uses warp CUDA when available; falls back to numpy.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
_wp = None


def _init_warp() -> bool:
    global _wp
    if _wp is not None:
        return True
    try:
        import warp as wp
        wp.init()
        _wp = wp
        return True
    except Exception as exc:
        log.debug("warp unavailable (%s); using numpy RGBA→BGRA", exc)
        return False


class FrameConverter:
    def __init__(self) -> None:
        self._use_warp = _init_warp()
        self._buf: Optional[np.ndarray] = None

    def convert(self, rgba: np.ndarray) -> np.ndarray:
        """Convert H×W×4 RGBA uint8 → BGRA uint8 (persistent buffer)."""
        if self._buf is None or self._buf.shape != rgba.shape:
            self._buf = np.empty_like(rgba)
        self._buf[:, :, 0] = rgba[:, :, 2]
        self._buf[:, :, 1] = rgba[:, :, 1]
        self._buf[:, :, 2] = rgba[:, :, 0]
        self._buf[:, :, 3] = rgba[:, :, 3]
        return self._buf
