# Build an XR viewer from libraries

Start with [the XR viewer recipe](../xr-viewer-recipe/README.md) and [dependencies](../dependencies/nvidia-runtime.md#ovxr). This compatibility route delegates SDK mechanics to [library skills](../dependencies/ovxr.md).

Use [bootstrap](library-only.md), [input scheduling](frame-input-order.md) and [settings/recovery](settings-and-lifetime.md) for viewer ownership. Add requested [input](../xr-spatial-input/README.md), [interaction state](../ovxr-python-interactions/README.md) and [UI](../ovxr-composition-ui/README.md).

For passthrough use [alpha](ar-compositing.md) and [supported AR controls](live-ar-controls.md). For a companion use [mirror lifetime](mirror-lifetime.md) and [desktop delivery](../xr-desktop-mirror/README.md). [Experience quality](../ovxr-composition-ui/experience-quality.md) and [validation](../ovxr-composition-ui-validation/README.md) define integrated acceptance.

Viewer skills own application state, safe scene composition, dispatch and delivery. Upstream owns setup/ABI, context/frames, anchors, borrowed handles, capture and runtime configuration.
