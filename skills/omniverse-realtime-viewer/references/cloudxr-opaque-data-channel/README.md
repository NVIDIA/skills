# CloudXR messaging in an XR viewer

Read [library runtime-adapters and openxr-input-and-extensions](../dependencies/ovxr.md) for raw service/channel integration with separately supplied CloudXR headers. Those methods are not OVXR exports.

Viewer skills own [messages](../xr-client-message-contract/README.md), bounded dispatch and input normalization. Keep one SDK-owned session/frame loop; channel work produces immutable application intents.

Agree a channel identity and distinguish video, channel and dispatch readiness. Optional messaging failure need not stop rendering. Reconnect advances input epoch, cancels captures and converges canonical state.

Bound receive work/history and enforce serialized UTF-8 byte ceilings before sending. A measured 32 KiB budget on one deployment is an application qualification, not an extension-wide constant. Reject oversize replies clearly.

Use [spatial input](../xr-spatial-input/README.md) for source arbitration and [connection](../xr-client-connection/README.md) for diagnosis. A reply is not proof of submitted pixels or manipulation.
