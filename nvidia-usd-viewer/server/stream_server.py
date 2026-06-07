"""Owns ovstream initialization, WebRTC server lifecycle, and frame submission.

All ovstream callbacks are called from StreamSDK internal threads. Never call
renderer or stage APIs directly from callbacks — enqueue work for the render loop.

Refer to omniverse-realtime-viewer → references/streaming-server/ and
references/streaming-lifecycle/ for the full lifecycle contract.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class StreamServer:
    def __init__(
        self,
        width: int,
        height: int,
        target_fps: int,
        signaling_port: int,
        public_ip: str,
        on_connection: Callable[[bool], None],
        on_message: Callable[[str], None],
        on_input: Callable[[bytes], None],
    ) -> None:
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.signaling_port = signaling_port
        self.public_ip = public_ip
        self._on_connection_cb = on_connection
        self._on_message_cb = on_message
        self._on_input_cb = on_input
        self._server = None
        self._connected = False
        self._initialized = False

    def initialize(self) -> None:
        import ovstream
        ovstream.initialize()
        self._initialized = True
        log.info("ovstream initialized")

    def start(self) -> None:
        import ovstream
        self._server = ovstream.Server(
            width=self.width,
            height=self.height,
            fps=self.target_fps,
            signaling_port=self.signaling_port,
            public_ip=self.public_ip,
        )
        # Register callbacks BEFORE start() to avoid missing early events.
        self._server.on_connection = self._handle_connection
        self._server.on_message    = self._handle_message
        self._server.on_input      = self._handle_input
        self._server.start()
        log.info("ovstream WebRTC started  signaling=%s:%d", self.public_ip, self.signaling_port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        if self._initialized:
            import ovstream
            ovstream.shutdown()
            self._initialized = False

    def stream_video(self, bgra: np.ndarray) -> None:
        """Submit a BGRA8 frame. No-op when no client is connected."""
        if self._server is None or not self._connected:
            return
        try:
            self._server.stream_video(bgra)
        except Exception as exc:
            log.debug("stream_video transient error: %s", exc)

    def send_message(self, msg: str) -> None:
        if self._server is None or not self._connected:
            return
        try:
            self._server.send_message(msg)
        except Exception as exc:
            log.debug("send_message error: %s", exc)

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _handle_connection(self, connected: bool) -> None:
        self._connected = connected
        log.info("Client %s", "connected" if connected else "disconnected")
        self._on_connection_cb(connected)

    def _handle_message(self, msg: str) -> None:
        self._on_message_cb(msg)

    def _handle_input(self, data: bytes) -> None:
        self._on_input_cb(data)
