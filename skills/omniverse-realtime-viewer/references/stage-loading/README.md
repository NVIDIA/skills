# Stage Loading

## Triggers

Use this skill for load USD, RenderProduct, RenderVar, black frame, composite stage, session layer, camera aspect, OVStage population, `open_usd`, `add_usd_reference`, or Unable to find RenderProduct prim.

ovrtx needs a complete render pipeline in the stage: Camera -> RenderProduct -> RenderVar -> RenderSettings. Most user USD files do not include this, so viewers build a generated root/composite stage that sublayers the user scene and authors viewer-owned render prims. New ovrtx viewers populate that composed USD through OVStage, publish the ordinal, attach ovrtx, and render the committed publication.

For OVStage population, ovrtx render pipeline behavior, or release-specific
loading behavior not covered here, read `references/dependencies` for
acquisition guidance and supplemental dependency documentation.

## ovrtx Stage Loading APIs

Use OVStage population APIs for new viewer code:

- `ovstage.population.open_usd(stage, path, ordinal=N, domains=PopulationDomain.RENDERING)` opens a file/URL and populates the application-owned stage.
- `ovstage.population.open_usd_from_string(stage, usda, ordinal=N, domains=PopulationDomain.RENDERING)` opens generated inline USDA. Use this for viewer/session USD that sublayers a user scene and adds cameras, RenderProducts, RenderVars, and RenderSettings.
- `ovstage.population.add_usd_reference*()` edits the USD source after a root stage is open. Follow it with `ovstage.population.apply_usd_changes(stage, ordinal=N)`.
- `ovstage.population.remove_usd(stage, handle)` and `ovstage.population.reset_usd(stage)` edit or clear the USD source. Follow either with `apply_usd_changes(stage, ordinal=N)`.
- After each population or source-change operation, wait for completion, call `stage.advance_write_floor(N).wait()`, then render with `renderer.step(..., ordinal=N)` or call `renderer.update_from_stage(N)` if the app needs the renderer updated before stepping.

Direct `renderer.open_usd*`, `add_usd_reference*`, `remove_usd()`, and
`reset_stage()` are OVRTX standalone compatibility APIs. Keep them only for
standalone paths, explicit compatibility tests, or older pinned applications.
Do not use older implicit stage-addition or anonymous-layer staging patterns for
new ovrtx stage loading.

## Generated Root Stage Pattern

For local and streamed Omniverse Realtime Viewers, generate one root USDA layer that sublayers the user scene and contains only viewer camera, render product, render vars, and render settings. Do not inject lights here unless the user asked for viewer-controlled lighting and the app exposes a verified lighting capability or explicit reload/profile workflow.

```python
CAMERA_PATH = "/Session/Cameras/Main"
RENDER_PRODUCT_PATH = "/Session/Render/Viewport"

def viewer_root_usda(scene_path: str, width: int, height: int) -> str:
    h_aperture = 20.955
    v_aperture = h_aperture * float(height) / float(width)
    scene_ref = scene_path.replace("\\", "/")
    return f"""#usda 1.0
(
    subLayers = [
        @{scene_ref}@
    ]
    defaultPrim = "Session"
)
def Scope "Session" {{
    def Scope "Cameras" {{ def Camera "Main" {{
        float focalLength = 18.15
        float horizontalAperture = {h_aperture}
        float verticalAperture = {v_aperture}
        float2 clippingRange = (0.01, 10000000)
        token projection = "perspective"
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }} }}
    def Scope "Render" {{ def RenderProduct "Viewport" {{
        rel camera = </Session/Cameras/Main>
        rel orderedVars = [</Session/Render/Vars/LdrColor>]
        uniform int2 resolution = ({int(width)}, {int(height)})
    }}
    def Scope "Vars" {{
        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
    }} }}
}}
"""
```

Start basic generated viewers with `LdrColor` only. Add `HdrColor`, depth,
normal, segmentation, or other debug AOVs only when the app exposes an explicit
AOV/debug-view workflow and `aov-switching` plus `render-settings` have defined
the conversion and validation path. Native ovrtx picking does not need
`InstanceSegmentationSD`.

```python
import ovstage
from ovstage import PopulationDomain, population

stage = ovstage.Stage("viewer")
ordinal = 1
population.open_usd_from_string(
    stage,
    viewer_root_usda(str(stage_path), width, height),
    ordinal=ordinal,
    domains=PopulationDomain.RENDERING,
)
stage.advance_write_floor(ordinal).wait()
renderer.attach_ovstage(stage)

products = renderer.step(
    render_products={RENDER_PRODUCT_PATH},
    delta_time=1 / 60,
    ordinal=ordinal,
)
if hasattr(products, "__enter__"):
    with products as ctx:
        product = ctx[RENDER_PRODUCT_PATH]
else:
    product = products[RENDER_PRODUCT_PATH]
```

When the installed `renderer.step()` result is mapping-like but not a context
manager, consume it directly and still copy any frame data before the next step.

## Standalone Layer Versus Embedded Body

A generated root wrapper must contain exactly one `#usda 1.0` header and one
root metadata block. Do not concatenate a full standalone session layer string
inside another root layer; the second `#usda` header or metadata block creates
invalid USDA and can fail with a parse error near `(`/EOF.

Use two helpers when the same session content is needed in both forms:

```python
def build_session_body(...) -> str:
    return (
        'def Scope "Session"\n'
        '{\n'
        '    # Camera, RenderProduct, RenderVars, and RenderSettings.\n'
        '}\n'
    )


def build_session_layer(...) -> str:
    return (
        '#usda 1.0\n'
        '(\n'
        '    defaultPrim = "Session"\n'
        ')\n'
        + build_session_body(...)
    )


def build_root_layer(scene_ref: str, ...) -> str:
    return (
        '#usda 1.0\n'
        '(\n'
        f'    subLayers = [@{scene_ref}@]\n'
        '    defaultPrim = "Session"\n'
        ')\n'
        + build_session_body(...)
    )
```

Alternatively, keep the session layer separate and attach it with `subLayers` or
references. Never paste a standalone layer, including its header, into another
layer body.

## Direct Frame Validation

For local desktop viewers, always separate renderer validation from native UI
presentation validation. After populating and publishing the generated root
stage, step the same RenderProduct path and committed ordinal the viewport will
use and save a direct `LdrColor` artifact before debugging the window.

A nonblank direct `LdrColor` frame proves that the generated Camera ->
RenderProduct -> RenderVar wiring is basically working. If the native window is
still black or blank after that, the next suspect is the ovui presentation path,
not the render product path, camera relation, or USD composition.

If the direct frame is blank, continue debugging this skill's concerns: user
sublayer path, camera path, render product path, render var source name,
resolution, camera transform, stage lighting, material/plugin resolution, and
load-operation errors.

## Composite File Pattern

Streaming servers should prefer a wrapper `.usda` written beside the user stage. The wrapper sublayers the user scene, injects the server camera/render product/render vars, and is populated with `ovstage.population.open_usd(stage, composite_path, ordinal=N, domains=PopulationDomain.RENDERING)`. During scene switches, the next OVStage population open replaces the previous root source; do not reset first unless the user explicitly requested an empty stage.

The reference streaming server uses camera path `/OVCamera` and render product path `/Render/OVServer/ViewportTexture0`.

```python
OV_CAMERA_PRIM = "/OVCamera"
OV_RENDER_PRODUCT = "/Render/OVServer/ViewportTexture0"
CAMERA_HORIZONTAL_APERTURE = 20.955

def make_composite_stage(scene_url: str, width=1920, height=1080) -> str:
    scene_ref = scene_url.replace("\\", "/")
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    vertical_aperture = CAMERA_HORIZONTAL_APERTURE * float(safe_height) / float(safe_width)
    return f'''#usda 1.0
(
    subLayers = [
        @{scene_ref}@
    ]
)

def Camera "OVCamera"
{{
    float2 clippingRange = (1, 10000000)
    float focalLength = 18.15
    float horizontalAperture = {CAMERA_HORIZONTAL_APERTURE:.3f}
    float verticalAperture = {vertical_aperture:.4f}
    token projection = "perspective"
    double3 xformOp:translate = (-553.5, 246.6, -22.5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}}

def "Render"
{{
    def "OVServer"
    {{
        def RenderProduct "ViewportTexture0" (
            prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1"]
        )
        {{
            token omni:rtx:rendermode = "RealTimePathTracing"
            bool omni:rtx:pt:diAOV = 1
            bool omni:rtx:pt:giAOV = 1
            bool omni:rtx:pt:diffuseFilterAOV = 1
            bool omni:rtx:pt:reflectionsAOV = 1
            bool omni:rtx:pt:refractionFilterAOV = 1
            bool omni:rtx:pt:refractionsAOV = 1
            bool omni:rtx:pt:selfIllumAOV = 1
            bool omni:rtx:pt:volumesAOV = 1
            bool omni:rtx:pt:worldNormalsAOV = 1
            bool omni:rtx:pt:worldPosAOV = 1
            bool omni:rtx:pt:zDepthAOV = 1
            bool omni:rtx:pt:denoising:optix:denoiseAOVs = 1
            float omni:rtx:pt:zDepthMin = 0.1
            float omni:rtx:pt:zDepthMax = 10000
            int omni:rtx:pt:maxSamplesPerLaunch = 2073600
            float omni:rtx:rtpt:modulatingRoughnessThreshold = 0.08
            rel camera = <{OV_CAMERA_PRIM}>
            rel orderedVars = [
                </Render/Vars/LdrColor>,
            ]
            uniform int2 resolution = ({safe_width}, {safe_height})
        }}
    }}

    def "Vars"
    {{
        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
        def RenderVar "HdrColor"
        {{
            uniform string sourceName = "HdrColor"
        }}
        def RenderVar "Depth"
        {{
            uniform string sourceName = "DepthSD"
        }}
        def RenderVar "Normal"
        {{
            uniform string sourceName = "NormalSD"
        }}
        def RenderVar "InstanceSeg"
        {{
            uniform string sourceName = "InstanceSegmentationSD"
        }}
        def RenderVar "SemanticSeg"
        {{
            uniform string sourceName = "SemanticSegmentationSD"
        }}
        def RenderVar "Metallic"
        {{
            uniform string sourceName = "Metallic"
        }}
        def RenderVar "Roughness"
        {{
            uniform string sourceName = "Roughness"
        }}
        def RenderVar "Emissive"
        {{
            uniform string sourceName = "Emissive"
        }}
        def RenderVar "Diffuse"
        {{
            uniform string sourceName = "DiffuseAlbedoSD"
        }}
        def RenderVar "Specular"
        {{
            uniform string sourceName = "Specular"
        }}
        def RenderVar "AO"
        {{
            uniform string sourceName = "AmbientOcclusion"
        }}
        def RenderVar "DirectDiffuse"
        {{
            uniform string sourceName = "DirectDiffuse"
        }}
        def RenderVar "DirectSpecular"
        {{
            uniform string sourceName = "DirectSpecular"
        }}
        def RenderVar "IndirectDiffuse"
        {{
            uniform string sourceName = "IndirectDiffuse"
        }}
        def RenderVar "IndirectSpecular"
        {{
            uniform string sourceName = "IndirectSpecular"
        }}
        def RenderVar "MotionVectors"
        {{
            uniform string sourceName = "MotionVectors"
        }}
    }}

    def RenderSettings "OVRenderSettings"
    {{
        rel products = [<{OV_RENDER_PRODUCT}>]
    }}
}}

# Override EffectLayer shaders to disable selection glow.
# In ovrtx, no OmniGraph runtime drives EffectLayerMT.mdl's animation input.
# Setting Fader=0 forces a clean load-time non-highlighted state.
over "World"
{{
    over "Misc"
    {{
        over "Looks"
        {{
            over "Concrete_Rough"
            {{
                over "EffectLayer"
                {{
                    float inputs:Fader = 0
                }}
            }}
            over "Steel_Stainless"
            {{
                over "EffectLayer"
                {{
                    float inputs:Fader = 0
                }}
            }}
            over "MetallicGreen_OmniPbr"
            {{
                over "EffectLayer"
                {{
                    float inputs:Fader = 0
                }}
            }}
        }}
    }}
}}
'''
```

Pass the injected RenderProduct path to `renderer.step(..., ordinal=N)`.

The `EffectLayer` override block above is a material-effect example, not a
baseline stage-loading requirement. Official USD Viewer sample stages such as
`stage01`/`stage02` may need this fader override when rendered through ovrtx
without the OmniGraph runtime that normally drives EffectLayer animation. For a
general viewer, only generate equivalent `over` blocks when the active stage
actually contains compatible EffectLayer shader paths and the app intends to use
material-driven pick effects. Keep those overrides in the composite/session
layer before runtime effect writes. In standalone compatibility paths, direct
renderer effect writes can then use `PrimMode.EXISTING_ONLY` against those
preauthored attributes.

The OmniRtx API schemas and path-tracing AOV flags are required viewer-owned
render pipeline metadata. Recommended viewer implementations author the schemas
on the `RenderProduct`; if a target ovrtx build expects them on
`RenderSettings`, keep the same schema list and flag values on the render
settings prim instead of dropping them.

Do not use inline one-line prim bodies such as `def RenderVar "LdrColor" { uniform string sourceName = "LdrColor" }` or nested one-line override bodies such as `over "EffectLayer" { float inputs:Fader = 0 }`. Some ovrtx-bundled USD parser builds reject or misdiagnose these compact forms, especially when generated through Python strings with escaped braces. Use the multi-line brace form shown above for every generated `def`, `over`, and nested override block.

## Generated USDA Self-Check

Before calling `ovstage.population.open_usd*()` with a generated wrapper, or a
standalone compatibility `renderer.open_usd*()` path, validate the exact
generated text with OpenUSD in the selected `pxr` subprocess. This catches
malformed braces, bad asset references, and wrong value syntax before the
runtime enters its render/load path.

Use this validation for generated app scaffolds and tests. Keep it out of the
ovrtx render process when the app otherwise follows the pxr-subprocess isolation
contract.

## Initial Resolution And Aspect

The RenderProduct resolution and camera aperture must agree. Derive `verticalAperture` from `horizontalAperture * height / width` when creating session/composite camera data.

Browser-streamed Omniverse Realtime Viewer apps should use a fixed server render resolution, typically 1920x1080, and let the frontend display the video with `object-fit: contain`. CSS layout changes should not rebuild session/composite camera data.

Write the composite into the same directory as the user stage and reference the user stage by basename so relative textures, MDL files, and sublayers resolve:

```python
stage_dir = os.path.dirname(os.path.abspath(url))
stage_basename = os.path.basename(url)
stage_stem = os.path.splitext(stage_basename)[0]
composite_path = os.path.join(stage_dir, f"_ovrtx_composite_{stage_stem}.usda")
with open(composite_path, "w", encoding="utf-8") as f:
    f.write(make_composite_stage(stage_basename, width, height))

ordinal += 1
population.open_usd(stage, composite_path, ordinal=ordinal, domains=PopulationDomain.RENDERING)
stage.advance_write_floor(ordinal).wait()
products = renderer.step(
    render_products={OV_RENDER_PRODUCT},
    delta_time=1 / 60,
    ordinal=ordinal,
)
```

## Dynamic Scene Root

Do not assume the loaded scene root is `/World`. Some assets use roots such as `/stage`, and hardcoded `/World` paths break hierarchy, selection, and pickable-prim setup.

When opening the USD for metadata, detect and store both the root prim path and
the stage up-axis. The up-axis is viewer/session metadata: use it to configure
the viewer camera, never by rotating or rewriting the user USD.

1. Prefer `/World` if it exists.
2. Otherwise use `stage.GetDefaultPrim()` if valid.
3. Otherwise use the first pseudo-root child.

Pass `root_prim_path` through the load result so frontend hierarchy and selection
code can query the correct root. Keep `up_axis` in the server's active-stage
state and pass it to the camera controller before its initial fit or any input
handling.

### Implementation: pxr_worker subprocess

Do not import `pxr` (OpenUSD Python) in the main ovrtx process — it conflicts with ovrtx's bundled USD. Run all pxr queries in a separate subprocess:

```python
# pxr_worker.py — runs in subprocess, communicates via JSON over stdin/stdout
from pxr import UsdGeom


def cmd_get_root_prim_path():
    """Return root-path and camera metadata without mutating the opened USD stage."""
    if not _stage:
        return {"ok": False, "error": "no stage loaded"}

    up_axis = str(UsdGeom.GetStageUpAxis(_stage)).upper()
    # USD stages conventionally use Y or Z. Keep a deterministic viewer default
    # if a malformed or unsupported stage reports another token.
    if up_axis not in {"Y", "Z"}:
        up_axis = "Y"

    # 1. Prefer /World if it exists
    world = _stage.GetPrimAtPath("/World")
    if world.IsValid():
        return {"ok": True, "path": "/World", "up_axis": up_axis}

    # 2. Try DefaultPrim
    default_prim = _stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return {
            "ok": True,
            "path": str(default_prim.GetPath()),
            "up_axis": up_axis,
        }

    # 3. First pseudo-root child
    for child in _stage.GetPseudoRoot().GetChildren():
        return {
            "ok": True,
            "path": str(child.GetPath()),
            "up_axis": up_axis,
        }

    return {"ok": True, "path": "/", "up_axis": up_axis}
```

The main server proxies this through its existing
`PxrWorkerClient.get_root_prim_path()` command and, after each successful load,
caches `metadata["path"]` as `self.current_stage_root_path` and
`metadata["up_axis"]` as
`self.current_stage_up_axis`. Configure the server-owned `OrbitCamera` with that
axis before fit-to-stage, orbit, pan, or fly input. Do not let the browser choose
or override the axis.

### Protocol: Stage Metadata In `openStageResult`

The server includes `root_prim_path` and `up_axis` in every `openStageResult`
and `push_initial_state` message. The frontend may show this as diagnostic stage
metadata; camera behavior remains server-owned:

```python
self.send_message("openStageResult", {
    "url": active_url,
    "result": "success",
    "root_prim_path": server.current_stage_root_path,
    "up_axis": server.current_stage_up_axis,
})
# Send children from detected root, not hardcoded /World
children = server._pxr.get_children(root_path)
self.send_message("getChildrenResult", {
    "prim_path": root_path,
    "children": children,
})
```

## Skip-Reload Optimization

When the frontend sends `openStageRequest` for a stage that is already loaded (e.g., on reconnect or duplicate requests), skip the expensive renderer reload:

```python
def _stage_path_key(self, url: str) -> str:
    """Normalize path for comparison."""
    return os.path.normcase(os.path.abspath(url))

def _load_stage(self, url: str, force: bool = False) -> bool:
    # Skip if same stage already loaded
    if not force and self.current_stage_url:
        if self._stage_path_key(url) == self._stage_path_key(self.current_stage_url):
            logger.info("Stage already loaded, skipping reload: %s", url)
            return True
    # ... proceed with actual load
```

The `force=True` parameter is used by `_handle_reset_stage` so explicit resets always work. Without this optimization, duplicate or reconnect-time `openStageRequest` messages can trigger redundant reloads.

### Caveats

- Use `os.path.normcase(os.path.abspath(...))` for path comparison — not string equality.
- Keep `import os` at module level, never inside `_load_stage()`.
- The frontend should not send a default `openStageRequest` on connect. Let the server's delayed `push_initial_state` send the current `openStageResult` and root children after the data channel opens.
- The frontend should send `openStageRequest` only for explicit scene switches, file opens, resets, or reloads.
- A same-stage `openStageRequest` should fast-return success and current root state without starting reset or open population work.

## Do Not Block The Render Loop

Loading can be slow because USD composition, texture/material discovery, and
shader compilation may continue after OVStage population or standalone
`open_usd*` starts. In streaming apps, do not run the full load synchronously on
the WebRTC message handler or block frame production for more than a few
seconds.

Use a background loading thread plus a stage/runtime lock around OVStage
population, `apply_usd_changes`, write-floor advancement, renderer
attach/detach, and direct standalone renderer mutation. While the lock is held,
the render loop should skip `renderer.step()` and keep streaming the last good
frame.

## Rules

- `clippingRange` is `float2 clippingRange = (near, far)`, not separate `.near`/`.far` attributes.
- Use `ovstage.population.open_usd()` for file-backed root stages and
  `ovstage.population.open_usd_from_string()` for generated root USDA. Both
  replace the active populated root source.
- Use `ovstage.population.add_usd_reference*` only for additive content under a
  unique target path; keep the returned handle if you will remove it later and
  call `apply_usd_changes(stage, ordinal=N)` before publishing.
- Camera path must match the path used by `camera-controls` when writing `omni:xform`; the reference streaming camera is `/OVCamera`.
- The user scene is never modified by wrapper/session loading.
- Write composite files in the same directory as the user USD so relative textures, MDL files, and sublayers resolve correctly.
- Keep composite files alive for the server lifetime; population and renderer update paths can perform async texture/material loading after the open operation returns.
- If the user stage has a camera, copy focal length, horizontal/vertical aperture, clipping range, and transform for visual consistency.
- Use OVStage runtime writes for live camera `omni:xform`; the camera may only
  have authored `xformOp:*` in the USDA composition.
- Default basic viewer wrappers to `LdrColor` only. Add `HdrColor`, depth,
  normal, segmentation, or other debug AOVs only for an explicit AOV/debug-view
  feature with conversion and validation evidence.
- Add `InstanceSegmentationSD` only when a debug/display segmentation AOV is needed. Picking uses ovrtx pick queries and resolved path IDs; it does not require a token-map RenderVar.

## Failure Modes

- `Unable to find RenderProduct prim`: wrapper/session render pipeline missing or wrong render product path.
- Black frame: camera path invalid, resolution missing, or RenderVar sourceName wrong.
- Broken textures after wrapping: composite is in `/tmp` instead of beside the asset.
- Textures fail later after initial load: composite was deleted too early.

See also: `ovrtx-rendering`, `render-settings`, `camera-controls`, `stage-management`, `streaming-server`.
