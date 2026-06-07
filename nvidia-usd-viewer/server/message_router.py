"""Routes JSON data-channel messages to render-loop actions.

Message format:  { "event_type": "...", "payload": { ... } }

Refer to omniverse-realtime-viewer → references/streaming-messages/ for
the full data-channel protocol contract.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

log = logging.getLogger(__name__)


class MessageRouter:
    def __init__(
        self,
        send_fn:          Callable[[str], None],
        load_scene_fn:    Callable[[str], None],
        get_camera_fn:    Callable[[], dict],
        set_camera_fn:    Callable[[dict], None],
        get_hierarchy_fn: Callable[[], dict],
        get_settings_fn:  Callable[[], dict],
        set_settings_fn:  Callable[[dict], None],
    ) -> None:
        self._send        = send_fn
        self._load_scene  = load_scene_fn
        self._get_camera  = get_camera_fn
        self._set_camera  = set_camera_fn
        self._get_hier    = get_hierarchy_fn
        self._get_settings = get_settings_fn
        self._set_settings = set_settings_fn

        self._dispatch = {
            "openStageRequest":  self._on_open_stage,
            "getCameraState":    self._on_get_camera,
            "setCameraState":    self._on_set_camera,
            "getHierarchy":      self._on_get_hierarchy,
            "getRenderSettings": self._on_get_settings,
            "setRenderSettings": self._on_set_settings,
            "ping":              lambda p: self._emit("pong", {}),
        }

    def handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Invalid JSON from client")
            return
        handler = self._dispatch.get(msg.get("event_type", ""))
        if handler:
            try:
                handler(msg.get("payload", {}))
            except Exception:
                log.exception("Error in message handler")
        else:
            log.debug("Unknown event_type: %s", msg.get("event_type"))

    def send_initial_state(self) -> None:
        self._emit("cameraState",    self._get_camera())
        self._emit("renderSettings", self._get_settings())

    def send_scene_loaded(self, url: str) -> None:
        self._emit("stageLoaded", {"url": url})

    def send_error(self, message: str) -> None:
        self._emit("error", {"message": message})

    # ------------------------------------------------------------------

    def _on_open_stage(self, p: dict) -> None:
        url = p.get("url", "")
        if url:
            self._load_scene(url)

    def _on_get_camera(self, p: dict) -> None:
        self._emit("cameraState", self._get_camera())

    def _on_set_camera(self, p: dict) -> None:
        self._set_camera(p)
        self._emit("cameraState", self._get_camera())

    def _on_get_hierarchy(self, p: dict) -> None:
        self._emit("hierarchy", self._get_hier())

    def _on_get_settings(self, p: dict) -> None:
        self._emit("renderSettings", self._get_settings())

    def _on_set_settings(self, p: dict) -> None:
        self._set_settings(p)
        self._emit("renderSettings", self._get_settings())

    def _emit(self, event_type: str, payload: dict) -> None:
        self._send(json.dumps({"event_type": event_type, "payload": payload}))
