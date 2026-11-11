<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Streaming Message Server Handler Map

## Server Handler Map

Keep the transport adapter small: parse the inner app JSON, validate its shape,
select a handler, and queue slow queries or runtime commands. WebRTC's
`on_app_message` already removes the SDK envelope; SHM can feed the same decoder
with its raw app JSON. Stream-control envelopes do not belong in this dispatcher.

```python
import json


def decode_app_message(raw: str) -> dict | None:
    try:
        msg = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("event_type"), str):
        return None
    if not isinstance(msg.get("payload", {}), dict):
        return None
    return msg
```

```python
self._handlers = {
    "openStageRequest": self._handle_open_stage,
    "getChildrenRequest": self._handle_get_children,
    "getPropertiesRequest": self._handle_get_properties,
    "getPrimCountRequest": self._handle_get_prim_count,
    "getStatsRequest": self._handle_get_stats,
    "selectPrimsRequest": self._handle_select_prims,
    "makePrimsPickable": self._handle_make_pickable,
    "makePrimsSelectable": self._handle_make_pickable,
    "resetStage": self._handle_reset_stage,
    "resetStageRequest": self._handle_reset_stage,
    "loadingStateQuery": self._handle_loading_state_query,
    "getVariantsRequest": self._handle_get_variants,
    "setVariantRequest": self._handle_set_variant,
    "changeAOVRequest": self._handle_change_aov,
    "getAvailableAOVs": self._handle_get_available_aovs,
    "toggleSegView": self._handle_toggle_seg_view,
    "setCameraGizmo": self._handle_set_camera_gizmo,
}
```

WebRTC/native send helper (use raw `send_message` for a SHM adapter):

```python
def send_event(stream_server, event_type: str, payload: dict, *, reply=False) -> None:
    if not stream_server or (not reply and not stream_server.is_client_connected):
        return
    try:
        stream_server.send_app_message(json.dumps({"event_type": event_type, "payload": payload}, default=str))
    except Exception:
        if stream_server.is_client_connected:
            raise
        logger.debug("Dropping event after disconnect: %s", event_type, exc_info=True)
```

For a `MessageHandler` wrapper, apply the guard to the underlying ovstream server:

```python
def send_message(self, event_type: str, payload: Dict[str, Any], *, reply=False) -> None:
    send_event(self.server._stream_server, event_type, payload, reply=reply)
```

Pass `reply=True` for responses to app requests, which can arrive before the
connected callback. Route persistent send failures to the session error handler.
The wrapper name `send_message` is app-owned, not the raw
ovstream method.

If generating `server/pxr_worker.py`, make it a stateful JSON-lines worker with pure USD commands such as `load`, `get_children`, `get_properties`, `get_prim_count`, `get_variants`, and `get_root_prim_path`. It should not import ovstream or mutate renderer state.
