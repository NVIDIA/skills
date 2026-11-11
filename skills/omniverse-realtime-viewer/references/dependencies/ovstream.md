# ovstream Dependency

Use ovstream for browser streaming or local SHM delivery. An in-process desktop
viewer does not need a streaming transport. Follow [runtime dependency guidance](nvidia-runtime.md)
for acquisition and the selected release's upstream skills, headers, and examples
for transport configuration.

## Transport Boundaries

| Channel | Receive | Send | Payload |
|---|---|---|---|
| WebRTC/native application | `Server.on_app_message` | `Server.send_app_message` | Inner application JSON; SDK owns the transport envelope |
| SHM application | `Server.on_message` | `Server.send_message` | Application-defined raw messages |
| Viewport input | `Server.on_input` | Client native input API | `InputEvent` structs |

Keep stream-control messages in the SDK. Application callbacks validate and queue
commands; the runtime owner applies them. See [streaming messages](../streaming-messages/README.md)
for app dispatch and [streaming lifecycle](../streaming-lifecycle/README.md) for
connection timing.

SHM Python input uses the common client surface:

```python
client = ovstream.Client(ovstream.ClientType.SHM, stream_name="...")
client.send_input_event(event)
```

Native integrations use `ovstream/ovstream_client.h` and
`ovstream_client_send_input_event(client, event)`. Generate or verify FFI layouts
against the selected headers, including every member of tagged unions. Borrowed
client frames must be released before client teardown.

## Frame Ownership

For fixed-resolution streaming, `stream_video()` stages the submitted data into
server-owned storage before returning. The producer must finish writing before
submission and keep the source alive through that call. Reuse a persistent BGRA8
CUDA buffer to avoid per-frame allocation. Match the server's `cuda_device` and,
for WebRTC/native, `cuda_context` to the producer's allocation context.

Resizable WebRTC changes this ownership contract: the encoder can read the
application's buffer after return. It requires persistent maximum-size plain
CUDA allocations, not memory-pool allocations. Use the selected SDK's
`resizable-streams` guidance for that path; [viewport resize](../viewport-resize/README.md)
covers its integration with renderer, camera, and input state.

## Setup Verification

Use the package's native runtime unless deliberately building a matched native
integration. An unrelated library override can load a different ABI than the
Python bindings expect. Check the selected release's platform prerequisites
when import or initialization fails.

```bash
python3 -c "import ovstream; ovstream.initialize(); print(ovstream.get_version()); ovstream.shutdown()"
```

This checks loading and lifecycle only. Validate encoding with a real submitted
frame and decoded browser video using [streaming lifecycle](../streaming-lifecycle/README.md).

## XR mirrors

OVXR readback is borrowed storage with a separate expiry boundary. Follow
[OVXR frame-loop and readback](../ovxr-frame-loop-and-readback/README.md) and
[desktop mirror](../xr-desktop-mirror/README.md) to copy into application-owned
storage before publishing to an asynchronous consumer. Keep that storage alive
and synchronize reuse according to the selected ovstream release's contract.
SHM presents host pixels; resolve its current layout from matching headers and
carry any application-specific pixel-order metadata in the app protocol.
