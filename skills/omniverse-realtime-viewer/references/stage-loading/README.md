# Stage Loading

Use this reference to compose a user USD asset with the camera and render
pipeline a viewer needs. Scene switching and load-state UX are covered by
[stage-management](../stage-management/README.md).

## Compose viewer-owned prims

Keep the source asset unchanged. A generated root layer sublayers it and adds a
viewer camera, RenderProduct, RenderVars, and optional render settings. Preserve
authored lights and material values; lighting controls and pick effects are
separate features, not load-time corrections.

Start with `LdrColor` for an interactive viewport. Add other outputs only for
requested features using [aov-switching](../aov-switching/README.md). Native
picking does not require a segmentation AOV.

Stage-level `upAxis` and `metersPerUnit` do not compose from a sublayer. Read
the original resolved source before building the wrapper; reading the wrapper
first can return fallback values instead of the source's metrics. In the
isolated USD metadata worker, use the effective [OpenUSD metrics](https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/pxr/usd/usdGeom/metrics.h)
so unauthored values follow USD's configured fallbacks:

```python
# Worker process only; do not import pxr into the renderer parent.
from pxr import Usd, UsdGeom

def read_source_metrics(source_path: str) -> dict:
    source = Usd.Stage.Open(source_path, load=Usd.Stage.LoadNone)
    if source is None:
        raise ValueError(f"Cannot open source: {source_path}")
    return {
        "up_axis": str(UsdGeom.GetStageUpAxis(source)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(source)),
    }
```

Return these copied scalars to the runtime owner with the source's load request.
Pass them explicitly to the wrapper builder and author them on its root layer:

```python
import math

CAMERA_PATH = "/Session/Cameras/Main"
RENDER_PRODUCT_PATH = "/Session/Render/Viewport"

def viewer_root_usda(scene_path: str, width: int, height: int, *,
                     up_axis: str, meters_per_unit: float) -> str:
    if up_axis not in ("Y", "Z"):
        raise ValueError("USD up-axis must be Y or Z")
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("metersPerUnit must be finite and positive")
    h_aperture = 20.955
    v_aperture = h_aperture * float(height) / float(width)
    scene_ref = scene_path.replace("\\", "/")
    return f"""#usda 1.0
(
    subLayers = [@{scene_ref}@]
    upAxis = "{up_axis}"
    metersPerUnit = {float(meters_per_unit)!r}
)
def Scope "Session"
{{
    def Scope "Cameras"
    {{
        def Camera "Main"
        {{
            float focalLength = 18.15
            float horizontalAperture = {h_aperture}
            float verticalAperture = {v_aperture}
            float2 clippingRange = (0.01, 10000000)
            token projection = "perspective"
            matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
    }}
    def Scope "Render"
    {{
        def RenderProduct "Viewport"
        {{
            rel camera = </Session/Cameras/Main>
            rel orderedVars = [</Session/Render/Vars/LdrColor>]
            uniform int2 resolution = ({int(width)}, {int(height)})
        }}
        def Scope "Vars"
        {{
            def RenderVar "LdrColor"
            {{
                uniform string sourceName = "LdrColor"
            }}
        }}
        def RenderSettings "Settings"
        {{
            rel products = </Session/Render/Viewport>
        }}
    }}
}}
"""
```

Pass an absolute scene path or resolvable asset URL to an inline root. For a
file-backed wrapper, relative paths are anchored at the wrapper's directory.
Keep downloaded asset directory structure intact so textures, MDL files, and
nested references retain their original anchors. See
[cloud-assets](../cloud-assets/README.md) for resolution and caching.

Use one USDA header/root metadata block per generated layer. Reuse prim-body
helpers or compose separate layers rather than nesting complete USDA documents.
Validate generated syntax in the isolated USD worker when developing the layer
builder.

## Populate, attach, and render

Register RTX schema paths before OVStage's first construction/population so
schema fallbacks are present from the first load. Populate a fresh Stage and
publish its ordinal before rendering:

```python
import os
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
import ovrtx
ovrtx.register_schema_paths()
import ovstage
from ovstage import PopulationDomain, population

stage = ovstage.Stage("viewer")
ordinal = 1
population.open_usd_from_string(
    stage, viewer_root_usda(str(stage_path), width, height, **source_metadata),
    ordinal=ordinal, domains=PopulationDomain.RENDERING,
)
stage.advance_write_floor(ordinal).wait()
renderer.attach_ovstage(stage)
products = renderer.step(
    render_products={RENDER_PRODUCT_PATH}, delta_time=1 / 60, ordinal=ordinal,
)
product = products[RENDER_PRODUCT_PATH]
```

This setup assumes the app has constructed `renderer` and chosen `stage_path`,
`width`, and `height`, and received `source_metadata` from the completed source
metrics worker request for that same asset. Fit or restore the camera through the runtime before the
first visible frame; the identity transform above is only its initial authored
value. Use [camera-controls](../camera-controls/README.md) for placement and
[ovrtx-rendering](../ovrtx-rendering/README.md) for output mapping and handoff.

For a file-backed root, use `population.open_usd()` instead. Reference edits,
time updates, and complete scene replacement follow
[ovstage-population](../ovstage-population/README.md).

## Resolution and scene metadata

Set RenderProduct resolution and derive camera vertical aperture from
`horizontalAperture * height / width`. A streamed viewer can keep a fixed render
size and use `object-fit: contain` in the browser. Implement live resolution
changes through the authored product and transport's resize contract when the
product requires them; CSS layout changes alone need not resize the renderer.

Derive hierarchy roots and bounds from the user asset, excluding viewer-owned
prims. Prefer its authored default prim when suitable, otherwise list its
scene roots. Pass these paths with load state rather than assuming `/World`.
Query metadata with OVStage or the isolated `pxr` worker; return copied values to
the UI. Obtain source up-axis and units before wrapper composition as above;
after authoring them on the root, the composed stage exposes those same metrics
to camera and simulation consumers. Re-read them for every newly loaded source.

## Diagnose a blank viewport

Capture `LdrColor` from the same product and ordinal used for display. A valid
direct frame localizes the problem to presentation or transport. Otherwise,
check the camera fit, product/camera relationship, full RenderVar path, source
composition, lighting, and material resolution. Confirm population completed
and the write floor was published before stepping.

See [render-settings](../render-settings/README.md) for render configuration and
[prim-pick-effects](../prim-pick-effects/README.md) for optional material effects.
