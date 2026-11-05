# Viewport Resize

Use this reference when an app needs to change rendered resolution. Ordinary
browser panel resizing only needs a fixed-resolution stream displayed with
`object-fit: contain`; it avoids renderer and encoder reallocations during layout
changes. Choose stream resolution at startup, or offer an explicit reconnect to
apply another size.

## Choose The Delivery Contract

For application-controlled live resize, read the selected ovstream release's
`resizable-streams` skill and example through
[dependency guidance](../dependencies/ovstream.md). Start with
`webrtc_resizable=True`; configured width/height reserve the maximum and each
submitted frame selects the active dimensions. Register `on_resize_request`
before start when using the SDK's browser resize negotiation. An app-owned
`resizeViewport` command can instead feed the same runtime queue.

This mode changes buffer ownership: the encoder can read a submitted buffer
after `stream_video()` returns. Use persistent maximum-size plain CUDA allocations
(`wp.ScopedMempool(device, False)` for Warp), preserving row pitch when rendering
into a smaller view. A rotating ring or producer CUDA event alone does not prove
encoder completion. Use the selected SDK's supported reuse protocol; if it does
not establish safe reuse for the application, retain fixed-resolution streaming.

## Apply One Resize Transaction

Coalesce layout requests and retain only the latest pending size. Validate
against the transport's minimum, maximum, and alignment requirements before
queuing the change. Keep the configured maximum realistic for the GPU's memory
budget.

The render/runtime owner applies the change between frames:

1. Update the displayed RenderProduct's `resolution` and camera aspect at one
   publication ordinal. Keep horizontal aperture and derive vertical aperture
   as `horizontal_aperture * height / width`.
2. Step that product at the published ordinal and check the actual mapped output
   dimensions before conversion and submission.
3. Update camera/input dimensions and acknowledge the effective size after the
   matching frame is produced. The browser confirms `videoWidth`/`videoHeight`
   before treating the new size as displayed.

For an attached OVStage, the following writes target queries selecting one
existing product and camera. `ordinal` comes from the runtime owner's clock:

```python
import numpy as np
import ovstage


def commit_resize(stage, product_query, camera_query, width, height,
                  horizontal_aperture, ordinal):
    dimensions = np.array([[width, height]], dtype=np.int32)
    dtype = ovstage.numpy_to_dldatatype(dimensions.dtype, lanes=2)
    resolution = ovstage.make_dltensor(
        dimensions, dtype=dtype, shape=[1], ndim=1,
    )
    stage.write_attribute(
        product_query, "resolution", ordinal, resolution, is_array=False,
    ).wait()
    aperture = np.array(
        [horizontal_aperture * height / width], dtype=np.float32,
    )
    stage.write_attribute(
        camera_query, "verticalAperture", ordinal, aperture, is_array=False,
    ).wait()
    stage.advance_write_floor(ordinal).wait()
```

Use this runtime write path or publish authored changes through the population
owner; editing an unrelated OpenUSD stage does not update the attached renderer.
Retain the effective size when rebuilding viewer camera/product state during
scene switches.

## Keep Interaction Aligned

Map input to the displayed product, including letterboxing when aspect ratios
differ. Native mouse MOVE events supply their coordinate extent in `data` and
`data2`; use it when normalizing resized input. Route picks to the same product
and publication as the displayed frame. Resize overlays and conversion targets
with that frame; no segmentation ID map is needed for native picking.

Validate shrinking and expanding, advancing decoded frames, stable object
proportions, and camera/picking at viewport edges. Include scene switching and
sustained streaming under the intended load to verify buffer reuse and memory
behavior.
