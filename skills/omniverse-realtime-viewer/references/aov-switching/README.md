# AOV Switching

Use this reference to display color, HDR, normals, distance, or segmentation
outputs in a local or streamed viewer. The renderer produces the selected
output; the presentation shell continues to display the same image surface or
video track.

## Frame path

```text
RenderProduct orderedVars
  -> rendered output identified by full RenderVar path
  -> selected output mapped on a supported device
  -> display conversion into an app-owned image buffer
  -> local presentation or the existing ovstream video track
```

Snapshot the selected product and RenderVar once at the start of a frame. Use
that snapshot for pick enqueue, rendering, output lookup, and conversion so a UI
change cannot switch identities halfway through the frame. Picking remains a
native OVRTX query, independent of the displayed AOV.

Separate the choices the UI offers from the outputs requested for this frame.
For a single-image viewer, request the active beauty output (`LdrColor` or
`HdrColor`) and retain any non-color outputs used by other features. Publish
changes to the existing product's `orderedVars` through the runtime owner;
changing display mode does not require another product or scene replacement.
When a workflow needs both beauty outputs, validate feedback in the one actually
displayed rather than assuming all outputs contain identical overlays.

For example, add a distance RenderVar to a viewer that exposes a depth view:

```usda
def RenderVar "DistanceToImagePlane"
{
    uniform string sourceName = "DistanceToImagePlaneSD"
}
```

Reference its full path in the RenderProduct's `orderedVars`. Enable any
required rendering passes on that product using
[render-settings](../render-settings/README.md) and the selected SDK's AOV
contract. Avoid enabling unrelated passes for discovery.

## Output identity and availability

OVRTX keys `frame.render_vars` by the full RenderVar prim path. Its
`source_name` selects the conversion and display label; it is not necessarily a
unique ID. For multiple products, identify a choice by `(product_path,
render_var_path)`.

Keep available choices and the active choice on the runtime owner. A choice
becomes available only after mapping and conversion succeed at the expected
resolution. Keep copied metadata in this cache, not output handles or mapped
tensors. Rebuild it after scene replacement, layout/capability changes, or
resize. Switching among already-validated choices need not rediscover all of
them; keep the capability cache separate from the current `orderedVars` subset.

This app-side helper chooses among already-validated outputs. `available` maps
`(product_path, render_var_path)` to copied source-name strings:

```python
def choose_output(available, requested=None, current=None):
    if requested in available:
        return requested
    if current in available:
        return current
    return next(
        (key for key, source in available.items() if source == "LdrColor"),
        next(iter(available), None),
    )
```

An invalid user request should return an error without silently changing a
working selection. Use the helper to preserve the current choice or select a
valid default when availability changes. If none is available, retain the last
displayed frame, disable the selector, and show the load/output error.

## Map and convert

Allocate display buffers for the transport's resolution and ownership contract.
Each frame acquires a buffer that is free for writing; it must not overwrite
one still being read by an encoder or display consumer. Local and streamed
handoff lifetimes are described in [ovrtx-rendering](../ovrtx-rendering/README.md)
and [streaming-server](../streaming-server/README.md).

Prefer CUDA mapping when the selected output supports it. Discover mapping
devices from the selected release's output contract and a focused probe; do not
infer support from beauty or generalize one runtime's restriction to every
release. For an explicitly supported host path, wait, copy into owned storage
inside the mapping scope, then convert/upload into the transport's owned buffer.
Report the transfer path and measured cost. Reject unsupported choices rather
than silently claiming GPU-only or zero-copy delivery.

For a single-tensor CUDA-mappable AOV, the mapping is itself the DLPack producer:

```python
# product_path and var_path came from the frame's selection snapshot.
product = products[product_path]
for frame in product.frames:
    output = frame.render_vars[var_path]
    with output.map(device=Device.CUDA) as mapped:
        mapped.wait()
        source = wp.from_dlpack(mapped)
        # App conversion: validate source_name/dtype/shape, then write the
        # acquired display buffer using the rules below.
        convert_display(output.source_name, source, display_buffer, stream)
        wp.synchronize_stream(stream)
    # Conversion completed; the display/transport may now consume its buffer.
```

`convert_display` is an app-owned dispatcher, not an SDK API. Reject empty or
unexpected dimensions and unsupported dtype/source combinations. For
multi-tensor outputs, use the tensor name specified by that RenderVar's
contract (`mapped[tensor_name]`); do not guess names after failed conversions.

The example waits synchronously. To overlap GPU work, use
`mapped.wait_on(stream.cuda_stream)` and record consumer completion with
`mapped.unmap(stream=stream.cuda_stream)` before releasing the source, then
honor the destination consumer's readiness contract. See the rendering
reference for the full lifetime sequence.

## Display conversion

Image tensors are channel-last `[H,W,C]`; scalar images use `[H,W,1]`. Dispatch
on source name as well as dtype and shape: integer distance bit patterns are
not segmentation IDs. For ovstream, every conversion ends in BGRA8
`uint8 [H,W,4]`; use the local surface's required channel order for desktop UI.

| Source | Display conversion |
| --- | --- |
| `LdrColor` | RGBA8 to BGRA8 by swapping red and blue. |
| `HdrColor` | Decode linear floating-point values, apply exposure, tone mapping, and display transfer, then clamp to 8-bit. |
| `NormalSD` | Decode floating-point components; map `[-1,1]` to `[0,1]` for RGB display. |
| `DistanceToImagePlaneSD` | Axial distance visualization with a labeled distance range. |
| `DistanceToCameraSD` | Radial distance visualization with a labeled distance range. |
| `InstanceSegmentationSD` / `SemanticSegmentationSD` | Deterministic colors from integer IDs; preserve the original IDs for any data inspection. |
| `DiffuseAlbedoSD` | Use the RGBA8 path for 8-bit output, or apply a display transfer to linear floating-point color. |

When HDR is exposed as `uint16` containing fp16 bits, reinterpret those bits as
half floats before tone mapping. Likewise, normals or distances exposed as
packed `uint32` need float-bit reinterpretation. Numeric integer conversion or
scaling the integer value is not a decoder. Follow the selected output's
reported layout and SDK contract; prefer GPU conversion in the interactive
path when the output supports it.

Keep exposure and distance ranges stable while viewing an AOV so animation or
camera motion does not make the visualization flicker. Expose manual/automatic
range choices when users need them, and label axial versus radial depth
accurately. On conversion failure, keep the last valid image and report the
failed choice; use a validated fallback only with matching active-state UI.

## Lighting-contribution views

For diffuse, glossy, transmission, caustic, or custom light-path views, first read
OVRTX's current [LPE documentation](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/docs/sensors/cameras/light_path_expressions.rst)
and linked examples. Upstream owns expression syntax, output limits, supported
render modes, and completeness constraints such as radiance-cache contributions.
Request only the needed outputs and present them through this same AOV path.

Treat LPE outputs as linear HDR radiance in the reported rendering color space.
Reuse a stable, declared exposure/tone/display transform; retain unmodified
floating-point values for numerical comparisons. A scene with no matching paths
can legitimately produce a black contribution image. Validate with a scene known
to exercise the requested transport, not solely with changed menu state.

## UI and messages

Use the standard [streaming envelope](../streaming-messages/README.md):

```json
{"event_type":"changeAOVRequest","payload":{"aov":"NormalSD"}}
```

Source-name IDs are convenient for a single product with unique source names.
For duplicate sources or multiple products, extend the app protocol to carry
the full product/RenderVar identity instead of collapsing those outputs.

| Event | Behavior |
| --- | --- |
| `getAvailableAOVs` → `availableAOVsResult` | Return validated choices with IDs and labels. |
| `changeAOVRequest` → `activeAOVState` | Validate the requested choice; acknowledge the applied state or report rejection. |
| Initial connection, scene load, availability change | Push available choices and authoritative active state. |

Keep a requested choice pending until its matching output has been rendered,
mapped and converted successfully into an owned display buffer. Then commit the
active mode and image together and settle that exact request ID. Keep status
broadcasts separate from request completion. Fence replies by scene and request
identity so superseded choices cannot overwrite newer intent.

On failure or timeout, reject the matching request and preserve the accepted
mode/image; conversion into scratch storage must not corrupt the last good
buffer. Settle pending requests on scene replacement or disconnect too. Apply
empty availability lists so stale choices disappear, and update the label when
an explicit fallback is accepted. Preserve camera and selection state across
AOV changes.

## Verify the feature

Switch among the requested views while orbiting and selecting objects. Check
channel order, normals, distance labels/ranges, and readable HDR highlights.
Switch scenes and reconnect the browser: removed outputs must disappear, the
active choice must match the displayed image, and the same video track should
continue. A failed map/conversion must not submit uninitialized pixels.

## Processed GPU outputs

For custom post-processing, chained resources or lens sampling, read
[SPG processing](../spg-processing/README.md). For stylized image passes,
technical/X-Ray geometry strokes or explicit full wireframe, continue to
[stylized visualization](../stylized-visualization/README.md). Present the actual
published processed output through this viewer's existing owner and transport;
a menu selection or display conversion alone does not implement the effect.
