# Streaming Lifecycle

Use this reference to connect a browser, restore viewer state on reconnect, and
keep the stream responsive while scenes load.

## Own The Session

One browser provider owns the connection for a server endpoint. Keep its
connection effect dependent on endpoint configuration, with a stable or
ref-backed app-message dispatcher. UI state updates must not reconnect the
stream. Terminate on provider teardown or explicit disconnect, and handle
React's development mount/cleanup cycle without overlapping sessions.

On the server, register callbacks before `start()`:

```python
server = ovstream.Server(ovstream.ServerType.WEBRTC)
server.on_connection = on_connection
server.on_app_message = on_app_message
server.on_input = on_input
server.start(config)
```

Callbacks run on SDK threads. They validate and enqueue work; the render/runtime
owner performs scene, camera, selection, and simulation mutations. Use the
[app-message adapter](../streaming-messages/server-handler-map.md) for dispatch
and replies.

## Establish State Before Choosing A Scene

Treat connection, scene state, and decoded video as separate states:

| Signal | What the app can do |
|---|---|
| Confirmed SDK start success (see [client readiness](../streaming-client/README.md#appstreamer-direct-api)) | Send app commands and query current viewer state; a resolved `IN_PROGRESS` result is insufficient |
| Server state response | Restore the loaded scene, hierarchy, selection, AOV, and loading status |
| Advancing decoded frames | Mark the viewport live and dismiss the video-loading indicator |

On each connection, query server state, for example with `loadingStateQuery`,
and have the server publish the current snapshot. A connection callback can
also trigger that push; the query covers clients that missed it. Do not use a
fixed sleep as the data-channel readiness signal.

Restore the server's current scene before considering a client default. If the
server reports idle and the app offers a default scene, open it once. Subsequent
opens come from user actions. Treat a request for the already-loaded scene as a
state refresh unless the user explicitly requests reload.

Include `root_prim_path` in `openStageResult` so the hierarchy starts at the actual
root. Clear scene-dependent selection and inspector state when the scene changes,
and discard responses belonging to the previous scene or request.

An app request can arrive before `on_connection(True)`. Reply to received
requests without gating on that notification; guard unsolicited pushes with
`is_client_connected`. Keep request IDs intact so the browser can settle pending
actions and reject stale results.

## Keep Video And Controls Responsive

Load assets and perform expensive queries outside transport callbacks. Serialize
renderer mutation with rendering, while the streaming path continues submitting
an app-owned loading or last-good frame. Follow
[frame continuity](../streaming-server/frame-loop-and-continuity.md) for the
buffer and locking boundary.

Report loading progress and a final success or error. Preserve the previous
valid frame on failure, and let the user retry or choose another scene. A failed
app action should not tear down a healthy WebRTC session.

Use bounded reconnects and backoff. Surface persistent transport failures in the
UI, catch rejected app-message promises, and clear pending operations when their
session ends. Recover by requesting fresh server state, rather than replaying
stale scene mutations.

On shutdown, withdraw readiness and stop accepting runtime work. Stop frame
submissions and join any separate submission worker before closing transport
or releasing its frame buffers. The runtime owner finishes native work and
tears down the renderer; its completion or failure must reach the supervising
caller. Pair process-lifetime `ovstream.initialize()` with `ovstream.shutdown()`
after the final server closes; individual server restarts use `stop()` and
`close()` without shutting down the process runtime.

## Verify The User Path

- Server readiness follows the first valid rendered and converted frame,
  independent of whether a browser is connected.
- The browser has nonzero decoded dimensions and advancing frame count; a
  connected socket or a screenshot of the surrounding UI is insufficient.
- Camera input, selection, and an app request/reply work after first connection
  and reconnect.
- Scene switching keeps video/control traffic alive and ends with the matching
  hierarchy and loading status.

For endpoint/configuration issues, use [streaming client](../streaming-client/README.md)
and the selected upstream connection diagnostics. For deployment, use
[cloud deployment](../cloud-deployment/README.md).
