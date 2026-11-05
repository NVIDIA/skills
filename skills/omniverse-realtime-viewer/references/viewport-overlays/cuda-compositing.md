# CUDA Overlay Compositing

Use with [viewport overlays](README.md). This is app code to implement, not an
import from an unshipped overlay package. Resolve ovui, OVRTX, and Warp through
[dependencies](../dependencies/README.md); verify the selected release's
[headless export wrapper](https://github.com/NVIDIA-Omniverse/ovui/blob/main/ovui/python/omni/ui/standalone/headless_frame.py)
and native implementation before wiring its frame adapter.

## Pixel And Ownership Contract

- Base: a fully initialized, app-owned CUDA `uint8 [height, width, 4]` BGRA
  display buffer. Copy/convert OVRTX output using
  [streaming server](../streaming-server/README.md) or your local presentation
  adapter. Do not write into a mapped OVRTX render var or an encoder-owned frame.
- Overlay: a separate app-owned CUDA `uint8 [height, width, 4]` RGBA export
  buffer on the same GPU/context. Allocate once with
  `wp.empty((height, width, 4), dtype=wp.uint8, device=device)`.
  Pass `int(buffer.ptr)` and the actual byte row pitch
  `int(buffer.strides[0])` to the exporter. Pitch must be at least `width * 4`,
  and the allocation must cover every copied row. Check extent/format before
  each copy after any resize; never resize inside a blend call after copying.
- Both buffers must use the same image origin and compatible color encoding.
  Verify asymmetric corner markers to detect a vertical flip. Explicitly
  convert if the exporter and display buffer use different orientations.
- Export format `rgba8` specifies channel order and storage, not alpha
  convention. For straight alpha, `out = src * a + dst * (1-a)`; for
  premultiplied alpha, `out = src + dst * (1-a)`. Choose the mode from a pixel
  probe of the actual export. ImGui Vulkan drawing over transparent black can
  produce premultiplied RGB; do not multiply alpha twice.
- Verify background pixels are `(0, 0, 0, 0)`. An opaque framebuffer clear cannot
  be fixed by treating dark colors as transparent. Resolve the export/clear
  behavior with the selected ovui package before calling the overlay complete.
- The example blends display-encoded bytes over an opaque base and sets output
  alpha to 255. For color-critical compositing, convert both inputs to a common
  linear color space, blend there, then apply the display encoding. A
  transparent base needs the full output-alpha equation instead.

## Blend Kernel

This kernel supports straight or premultiplied RGBA over an opaque BGRA base.
The caller validates equal extents, device, orientation, and the alpha probe.

```python
import warp as wp

@wp.kernel
def blend_overlay(
    base_bgra: wp.array3d(dtype=wp.uint8),
    overlay_rgba: wp.array3d(dtype=wp.uint8),
    premultiplied: int,
):
    y, x = wp.tid()
    a = wp.int32(overlay_rgba[y, x, 3])
    if a == 0:
        return
    inv_a = wp.int32(255) - a
    source_scale = a
    if premultiplied != 0:
        source_scale = wp.int32(255)
    for c in range(3):
        # RGBA source -> BGRA destination.
        src = wp.int32(overlay_rgba[y, x, 2 - c])
        dst = wp.int32(base_bgra[y, x, c])
        value = (src * source_scale + dst * inv_a + wp.int32(127)) // wp.int32(255)
        base_bgra[y, x, c] = wp.uint8(wp.min(value, wp.int32(255)))
    base_bgra[y, x, 3] = wp.uint8(255)
```

For a byte-space probe, a straight half-alpha red pixel is approximately
`(255, 0, 0, 128)`; the equivalent premultiplied pixel is approximately
`(128, 0, 0, 128)`. Both should produce BGRA `(0, 0, 128, 255)` over black,
allowing rounding. Also test opaque red/blue and untouched transparent pixels.

## Export, Copy, Release, Blend

Implement the overlay tick adapter using the selected ovui release. The
following function assumes the owner has already ticked the overlay and that
`headless_frame` has been initialized. The explicit device synchronizations
provide a conservative correctness baseline; measure before replacing them
with release-supported stream/event dependencies.

```python
def composite_export(headless_frame, overlay_rgba, base_bgra, *, premultiplied):
    # All buffers are owned by the app and validated before entering this function.
    device = base_bgra.device
    if overlay_rgba.device != device or overlay_rgba.shape != base_bgra.shape:
        raise ValueError("Overlay and base must have matching devices and extents")
    height, width, channels = base_bgra.shape
    if channels != 4 or headless_frame.extent() != (width, height):
        raise ValueError("Export extent must match the four-channel display buffer")
    if headless_frame.format() != "rgba8":
        raise ValueError("Expected RGBA8 overlay export")

    # Some releases enqueue a stream-0 GPU wait; this is not a CPU timeout.
    if not headless_frame.wait_ready(10_000_000):
        return False
    try:
        copied = headless_frame.copy_to_linear(
            int(overlay_rgba.ptr), int(overlay_rgba.strides[0]),
            cuda_stream_handle=0,
        )
        if not copied:
            return False
    finally:
        # Finish the stream-0 wait/copy before releasing the Vulkan image,
        # including failure cleanup. A fatal CUDA error requires owner recovery.
        wp.synchronize_device(device)
        headless_frame.signal_consumed()
        wp.synchronize_device(device)

    wp.launch(
        blend_overlay, dim=(height, width),
        inputs=[base_bgra, overlay_rgba, int(premultiplied)], device=device,
    )
    wp.synchronize_device(device)
    return True
```

Check CUDA device/context compatibility before initializing interop. Native
`wait_ready` and `signal_consumed` implementations can enqueue semaphore work on
stream 0 even when a copy API accepts a stream argument. A nonzero copy stream
therefore does not by itself establish producer or release ordering. Inspect
upstream's native implementation before optimizing.

The numeric timeout argument is not proof of bounded execution: in some
releases it is not enforced by the native GPU wait. Treat producer failure as
an owner/session failure and use process supervision if a hard deadline is
required. Do not race teardown against an outstanding GPU semaphore wait.

On failed acquisition/copy, skip blending and preserve the fresh OVRTX base;
never blend uninitialized or previous-frame export memory. After acquiring an
export, release it exactly once in cleanup. Report persistent errors to the UI.
Retain both CUDA allocations and the submitted base frame until GPU work and
its downstream consumer have completed. For ovstream, use
[frame lifetime guidance](../streaming-server/frame-loop-and-continuity.md);
for local presentation, follow the selected image/texture adapter's completion
contract. Synchronizing the blend only proves the producer has finished, not
that an encoder or presenter has released the frame.
