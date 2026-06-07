"""Decodes NVST binary InputEvent structs and dispatches to camera/selection.

NVST forwards mouse, keyboard, wheel, and touch events as binary structs
through the ovstream input callback. This module decodes the subset needed
for orbit/pan/zoom camera control.

Refer to omniverse-realtime-viewer → references/viewer-input-routing/ for
the full input-routing and click-vs-drag dispatch contract.
"""
from __future__ import annotations

import logging
import struct
from typing import Callable

log = logging.getLogger(__name__)

# NVST InputEvent type bytes (subset used by the viewer camera).
_EVT_MOUSE_BUTTON = 0x01
_EVT_MOUSE_MOVE   = 0x02
_EVT_MOUSE_WHEEL  = 0x03


class InputRouter:
    def __init__(
        self,
        on_mouse_button: Callable[[int, int, int, float, float], None],
        on_mouse_move:   Callable[[float, float], None],
        on_mouse_scroll: Callable[[float], None],
        stream_width:    int,
        stream_height:   int,
    ) -> None:
        self._on_button = on_mouse_button
        self._on_move   = on_mouse_move
        self._on_scroll = on_mouse_scroll
        self._w = stream_width
        self._h = stream_height

    def handle(self, data: bytes) -> None:
        if not data:
            return
        evt = data[0]
        try:
            if evt == _EVT_MOUSE_BUTTON:
                self._mouse_button(data)
            elif evt == _EVT_MOUSE_MOVE:
                self._mouse_move(data)
            elif evt == _EVT_MOUSE_WHEEL:
                self._mouse_wheel(data)
        except struct.error:
            log.debug("InputEvent decode error (type=0x%02x len=%d)", evt, len(data))

    # type(1B) + button(1B) + action(1B) + mods(1B) + x_norm(4f) + y_norm(4f)
    def _mouse_button(self, d: bytes) -> None:
        if len(d) < 12:
            return
        _, button, action, mods, xn, yn = struct.unpack_from("<BBBBff", d)
        self._on_button(button, action, mods, xn * self._w, yn * self._h)

    # type(1B) + pad(3B) + x_norm(4f) + y_norm(4f)
    def _mouse_move(self, d: bytes) -> None:
        if len(d) < 12:
            return
        _, _, _, _, xn, yn = struct.unpack_from("<BBBBff", d)
        self._on_move(xn * self._w, yn * self._h)

    # type(1B) + pad(3B) + delta(4f)
    def _mouse_wheel(self, d: bytes) -> None:
        if len(d) < 8:
            return
        _, _, _, _, delta = struct.unpack_from("<BBBBf", d)
        self._on_scroll(delta)
