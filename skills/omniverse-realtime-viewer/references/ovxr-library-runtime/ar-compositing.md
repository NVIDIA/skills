# AR alpha and passthrough

Treat renderer alpha production, OVXR submission, and runtime blending as
separate boundaries. An alpha environment blend mode does not make an opaque
renderer output transparent, and a black pixel is not evidence of empty space.

## Diagnose at the producing boundary

Confirm the negotiated environment blend mode, the actual eye RenderProducts
and selected color RenderVars, then inspect a bounded RGBA sample before any
application compositor. Record the capture product, image domain, sequence and
whether overlays are included. A desktop window may discard alpha; its black
background cannot prove that the eye texture is opaque.

Renderer-global Carb settings and USD attributes on a RenderProduct use
different namespaces. A successful write to an unrecognized USD attribute can
be silently ignored. Check the installed renderer schema rather than replacing
slashes with colons in a Carb path. For the renderer schema exercised here:

| RenderProduct USD attribute | Alpha-output configuration |
| --- | --- |
| `omni:rtx:post:compositing:enabled` | true |
| `omni:rtx:post:compositing:doComposite` | false |
| `omni:rtx:post:compositing:outputAlpha` | true |
| `omni:rtx:post:compositing:blackBackground` | true |
| `omni:rtx:materials:refractionAsOpacity` | Material policy; verify glass/translucency behavior |
| `omni:rtx:pt:fractionalCutoutOpacity` | Cutout policy; measure render cost and edge quality |

In particular, `omni:rtx:post:backgroundZeroAlpha:*` is not a substitute
for the `compositing` schema names above. A valid global Carb setting can
still be an invalid spelling on a USD prim.

Use upstream session/context/publication guidance through
[the SDK route](../dependencies/ovxr.md) for product creation and initialization
boundaries. Apply the renderer policy above through a supported setup path;
verify actual alpha and client behavior separately from authored setting names.

For an explicitly requested live controls feature, the separate
[AR control policy](live-ar-controls.md) defines independent controls,
acknowledgements and pixel/tracking gates. An apply operation requires a
supported public path; arbitrary product writes are not a live-update API.

## Preserve alpha through composition

Preserve renderer alpha outside UI coverage. Match premultiplied/straight
alpha to the color blend equation and OpenXR layer flags. Do not threshold black
RGB or far-depth values to manufacture background alpha: geometry, transparency,
antialiased coverage, and lighting cannot be reconstructed from that test.
Remove diagnostic cutout passes after the renderer path works.

A DomeLight can illuminate the scene without being visible to the camera.
Choose background visibility separately from illumination. Matte/shadow-catcher
render settings apply to appropriately authored matte meshes; they do not turn
a DomeLight into a matte object. Do not blanket-mark scene meshes matte to hide
the environment.

Keep material/refraction and fractional-cutout policy explicit. Enabling all AR
options at once can change material appearance and frame time; compare matched
runs before assigning a cost to any one setting. Test an opaque dark object,
thin edges, a translucent object, and a matte receiver when used.

Bound alpha diagnostics by attempts/time as well as successful captures.
Zero-copy image access does not make a synchronous CUDA-to-host sample free.
Sample an owned image after source-copy completion where possible, keep work
off the XR owner, and disarm capture-only features afterward. A sparse histogram
must not become a routine full-raster scan. Retain the borrowed-image lease only
through GPU source reads, then analyze owned samples. If capture is create-time
only, a diagnostic-only context needs explicit cleanup through context refresh;
warn that this can require client reconnect. A shared mirror still needs capture
and must not trigger that cleanup. Coalesce lifecycle requests and reject
completion from workers belonging to destroyed contexts. Keep cleanup owed to
the original context until destruction succeeds, including failed consumer
shutdown. Bound lock acquisition and recheck cancellation/deadline before SDK
access; contain worker-start failure as a diagnostic failure, not a frame-loop
failure. A sparse histogram
can establish transparent background pixels; it cannot prove edge quality or
that partial coverage is absent throughout the image. Use headset observation
or correlated decoded output for the final visual claim.
