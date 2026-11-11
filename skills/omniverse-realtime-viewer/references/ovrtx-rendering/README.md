# ovrtx Rendering

## Triggers

Use this skill for `ovrtx`, `renderer.step`, `step_async`, `attach_ovstage`, `update_from_stage`, committed render ordinals, `LdrColor`, AOVs, render vars, `HdrColor`, `NormalSD`, RenderProduct resolution, standalone `write_attribute`, `omni:xform`, `OVRTX_BIN_PATH`, magenta materials, or RenderApi errors.

ovrtx is a headless RTX renderer driven from Python. The app owns the render loop,
frame extraction, and any streaming/display handoff. For new ovrtx viewer
apps, OVStage owns scene population, hierarchy, runtime attributes, transforms,
material/effect writes, and write-floor publication; ovrtx attaches to that
stage and renders committed ordinals. Use direct ovrtx stage and attribute APIs
only for standalone compatibility paths or projects intentionally pinned to the
older renderer-owned stage model.

For ovrtx renderer behavior, Python/C API behavior, release notes, or behavior
not covered here, read `references/dependencies` for acquisition guidance and
supplemental dependency documentation.

## Core API

```python
import os
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
import ovrtx
# Stage-first attached mode: publish OVRTX schema paths before OVStage or any
# other OpenUSD consumer initializes. Importing ovrtx alone does not do this.
ovrtx.register_schema_paths()

import ovstage
from ovstage import PopulationDomain, population
from ovrtx import Renderer, RendererConfig

stage = ovstage.Stage("viewer")
ordinal = 1
population.open_usd(stage, "/path/to/composite.usda", ordinal=ordinal, domains=PopulationDomain.RENDERING)
stage.advance_write_floor(ordinal).wait()

# Create the renderer and attach the published stage.
renderer = Renderer(config=RendererConfig(
    sync_mode=True,
    active_cuda_gpus="0",
    keep_system_alive=True,
))
print(renderer.version)

renderer.attach_ovstage(stage)
products = renderer.step(
    render_products={"/Session/Render/Viewport"},
    delta_time=1.0 / 60.0,
    ordinal=ordinal,
)
product = products["/Session/Render/Viewport"]
```

`sync_mode=True` blocks until the GPU frame is complete. Async pipelines can use `False`, but then buffer lifetime and frame readiness need explicit care.

`renderer.attach_ovstage(stage)` attaches a live, externally owned
`ovstage.Stage`. Keep that Stage alive until `renderer.detach_ovstage()` or
renderer destruction. When a stage is attached, `renderer.step(..., ordinal=N)`
requires an ordinal and first updates the renderer through at least that
committed OVStage publication. A separate `renderer.update_from_stage(N)` is
only needed when the app must update renderer state before a step, such as
pre-step sensor or status work.

Call `ovrtx.register_schema_paths()` before constructing or first populating an
OVStage (and before any other OpenUSD runtime initializes). In current OVRTX,
this registers the RTX schema families with OVStage so populated schema
fallbacks resolve. An OVStage instance can be attached to only one live OVRTX
renderer. Renderer replacement must follow the
[attached-stage lifetime guidance](../ovstage-ovrtx-integration/README.md#lifetime-and-scene-changes),
including the payload limitation on affected builds.

`renderer.step()` returns a mapping-like `RenderProductSetOutputs`, indexed by
RenderProduct path; it is not a context manager. Map each needed RenderVar with
`with render_var.map(...) as mapping:` and copy the pixels inside that context.
Consume the results inside the frame loop and release references once the
required outputs have been copied.

`renderer.step_async()` enqueues the same frame work and returns an `Operation`. Poll `op.query_status()` from the runtime owner when you need the render loop to stay responsive, then call `op.wait()` before reading `RenderProductSetOutputs`. Do not mutate OVStage or the standalone renderer stage while a step operation is in flight.

## OVStage And Standalone Composition

ovrtx supports two scene-ownership modes:

- Attached mode for new apps: create an `ovstage.Stage`; load composed USD with
  `ovstage.population.open_usd(...)` or `open_usd_from_string(...)`; apply
  reference, reset, time, hierarchy, transform, material, and effect changes
  through OVStage; wait for the operation; advance the write floor; then render
  with `renderer.step(..., ordinal=N)`.
- Standalone compatibility mode: `renderer.open_usd*`, `add_usd_reference*`,
  `remove_usd()`, `reset_stage()`, `query_prims*`, `read_attribute*`,
  `write_attribute*`, `bind_attribute*`, and `map_attribute()` operate on a
  renderer-owned stage. These APIs remain useful for compatibility and focused
  standalone tests, but they are deprecated in ovrtx for production scene
  ownership.

Do not use older implicit stage-addition APIs as the main load path. Do not mix
OVStage writes with direct renderer stage writes for the same runtime state; pick
one owner per app path.

## Frame Extraction

```python
products = renderer.step(render_products={RENDER_PRODUCT_PATH}, delta_time=dt, ordinal=ordinal)
if RENDER_PRODUCT_PATH in products:
    product = products[RENDER_PRODUCT_PATH]
    for frame in product.frames:
        # Output identity is the full authored RenderVar prim path.
        ldr_path = "/Session/Render/Vars/LdrColor"
        if ldr_path in frame.render_vars:
            with frame.render_vars[ldr_path].map(device=Device.CUDA) as rv:
                rv.wait()  # Establish producer completion before Warp consumption.
                rgba_cuda = wp.from_dlpack(rv)  # borrowed H x W x 4 view
                wp.copy(owned_rgba_cuda, rgba_cuda)
                wp.synchronize_stream()  # Copy must finish before unmap.
            with frame.render_vars[ldr_path].map(device=Device.CPU) as rv:
                pixels = np.from_dlpack(rv).copy()  # H x W x 4 RGBA
```

CUDA mapping avoids a host transfer; CPU mapping copies to host. The mapped
object implements DLPack directly for a single-tensor output such as `LdrColor`.
For multi-tensor outputs, select names (`rv["Coordinates"]`, `rv["Intensity"]`)
and read scalar params through `rv.params`.

## Frame lifetime and synchronization

Keep step outputs alive while obtaining maps. DLPack consumers retain their
mapped backing storage, so copy frames into app-owned storage for UI and
transport queues rather than retaining renderer buffers.

GPU handoff has two completion points: wait for rendering before reading the
mapping, then finish the consumer's reads before releasing it. The example uses
`rv.wait()` and a consumer-stream wait for a simple synchronous copy. To keep
work asynchronous, use `rv.wait_on(consumer_stream_handle)` and
`rv.unmap(stream=consumer_stream_handle)` or `unmap(event=completion_event)` with
the actual consuming CUDA stream/event. Preserve destination-buffer ownership
until the display or encoder finishes with it.

For C consumers, `ovrtx_map_render_var_output()` exposes named `tensors[]` and
`params[]`; interpret them using the selected RenderVar's contract.

## Render Product Resolution

Set the viewer-owned RenderProduct `resolution` in the session or composite layer before loading the stage. The render product path must be the same path passed to `renderer.step(render_products={...})`.

Use a fixed render size with browser `object-fit: contain` by default. For an
explicit resize feature, update the authored resolution through the stage API
and follow `streaming-server/frame-loop-and-continuity.md` for the resizable
transport contract; verify actual output dimensions before submission.

## Render Vars And AOVs

The render product controls which AOVs ovrtx attempts to produce:

```usda
def RenderProduct "ViewportTexture0"
{
    rel camera = </OVCamera>
    rel orderedVars = [</Render/Vars/LdrColor>]
}

def RenderVar "LdrColor"
{
    uniform string sourceName = "LdrColor"
}
```

Basic generated viewers should request only `LdrColor` until an explicit
AOV/debug-view feature needs additional outputs. Add HDR, depth, normal,
segmentation, or other debug AOVs together with the conversion, UI, and
validation rules from `aov-switching` and `render-settings`.

`frame.render_vars` is keyed by the full authored `RenderVar` prim path, not by
the prim name or `sourceName`. For example, a `RenderVar "Normal"` at
`/Render/Vars/Normal` with `sourceName = "NormalSD"` appears under
`/Render/Vars/Normal`. Keep `source_name` only as the friendly AOV category
used for UI and conversion selection. Build a `source_name` →
`render_var_path` mapping from the frame outputs when the wrapper does not
author stable paths itself only if the viewer enforces one RenderVar per source
name. Otherwise, make the full `render_var_path` the UI option identity and
carry `source_name` only as display metadata.

Common displayable AOVs for viewer apps:

| AOV key | Tensor | Notes |
|---|---|---|
| `LdrColor` | `uint8 [H,W,4]` | Tonemapped RGBA; swap R/B for ovstream BGRA |
| `HdrColor` | `uint16 [H,W,4]` | Linear HDR exposed as fp16 bits in `uint16`; display with approximate tonemap |
| `NormalSD` | `float32 [H,W,4]` | World-space surface normals (XYZA) |
| `InstanceSegmentationSD` | `uint32 [H,W,1]` | Instance IDs; display/debug AOV, colorize for inspection |
| `SemanticSegmentationSD` | `uint32 [H,W,1]` | Semantic IDs; display/debug AOV, colorize for inspection |

All image tensors are channel-last `[H, W, C]`; scalar lanes are `[H, W, 1]`, not `[H, W]`. Request only the render vars the feature uses, and expose them only after they map to real non-empty data.

## Displaying other AOVs

Keep output identity as the RenderVar path and offer only outputs that map to
valid full-resolution data. Presence in `frame.render_vars` alone does not prove
that an output is displayable. Convert the selected output on GPU into the
transport's app-owned display buffer.

Use [aov-switching](../aov-switching/README.md) for availability state, dtype
conversion, and UI messages. Its output selection is independent of the chosen
local or streamed presentation shell.

## Runtime Attribute Writes

For new ovrtx applications, make OVStage the runtime data plane for
transforms, camera motion, material/effect controls, visibility changes, and
other stage-data state. Direct `renderer.write_attribute()` and
`bind_attribute()` are compatibility APIs for standalone renderer-owned stages,
not the source of truth when an OVStage is attached. Renderer-owned selection
outline group/style writes remain ovrtx responsibilities; route them through
`native-picking-selection`.

OVStage writes are ordinal-keyed. Wait for the write, advance the write floor,
then render that committed ordinal:

```python
import numpy as np
from ovstage import (
    AttributeSemantic,
    DLDataType,
    DLDataTypeCode,
    PrimMode,
    make_dltensor,
)

ordinal += 1
matrices = np.asarray([...], dtype=np.float64).reshape(len(paths), 16)
xform_tensor = make_dltensor(
    matrices,
    dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
    shape=[len(paths)],
)
stage.write_attribute(
    query,
    "omni:xform",
    ordinal=ordinal,
    tensors=xform_tensor,
    is_array=False,
    semantic=AttributeSemantic.MATRIX,
    prim_mode=PrimMode.UPSERT,
).wait()
stage.advance_write_floor(ordinal).wait()
products = renderer.step({RENDER_PRODUCT_PATH}, delta_time=dt, ordinal=ordinal)
```

`omni:xform` matrices are row-major `float64` matrices using the USD row-vector
convention, with translation in row `[3][0..2]`. In OVStage DLPack form, a
matrix write input is one 16-lane element per prim (`shape=[N]`, lanes `16`).
OVStage read/map outputs use scalar lanes and explicit `[N,4,4]` shape/strides;
map-creation dtypes remain lane-based. See [the data plane](../ovstage-data-plane/README.md).

For standalone compatibility paths, ovrtx consumes Fabric attributes such as
`omni:xform`; standard authored USD `xformOp:*` is not the live renderer update
path.

```python
import ovrtx

xforms = np.array([...], dtype=np.float64).reshape(n, 4, 4)
renderer.write_attribute(
    prim_paths=paths,
    attribute_name="omni:xform",
    tensor=xforms,
    semantic=ovrtx.Semantic.XFORM_MAT4x4,
    prim_mode=ovrtx.PrimMode.CREATE_NEW,
    data_access=0,  # 0=ASYNC for GPU tensors, 1=SYNC for CPU numpy
)
```

`ovrtx.PrimMode.EXISTING_ONLY` silently skips attributes not already registered. Use `ovrtx.PrimMode.CREATE_NEW` for compatibility camera, animation, and EffectLayer attributes that may not exist in Fabric yet. This is a different enum from `ovstage.PrimMode`; OVStage examples usually use `ovstage.PrimMode.UPSERT` for app-owned runtime state.

## Renderer Config And Process State

Set renderer-wide behavior at construction time:

- `sync_mode=True` keeps `step()` blocking until the frame is complete. Use `step_async()` for responsive loops that can poll an `Operation`.
- `active_cuda_gpus="0"` or `"0,1"` restricts which CUDA-visible devices ovrtx can use. Values are indices into `CUDA_VISIBLE_DEVICES`.
- `keep_system_alive=True` keeps the ovrtx system initialized across idle/reset periods. Choose it at renderer creation.

Set `enable_minimal`, `enable_realtimepathtracing` and `enable_pathtracing`
on the first renderer in the process. Conflicting explicit choices on later
instances fail. Selecting a RenderProduct mode does not disable the other
startup modes or recover their initialization cost. Spectral preparation
(`enable_spectral_rendering`) is also a first-renderer, process-lifetime choice.
MDL displacement (`enable_mdl_displacement`) is configured at renderer creation
and writes process-wide state; do not present it as an ordinary per-viewer live
toggle. Resolve defaults and supported combinations from upstream
[renderer configuration](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/docs/core/renderer_configuration.rst).

The ovrtx logging callback is process-global state, not renderer-owned state. Install it once during process startup if the app needs to route ovrtx logs into its logger. Avoid registering a different callback per request, per viewer tab, or per renderer instance; in multi-viewer processes the last process-global registration is the one that matters.

## First-Run Shader Compilation

The first `renderer.step()` on a cold machine or fresh cache can take 2-5 minutes while RTX pipelines and shaders compile. Treat this as a normal first-run cost before assuming the process is hung.

Use a longer timeout for first validation and CI runs, at least 300 seconds for the first rendered frame. Check the ovrtx log for shader or pipeline compilation progress before killing the process. GPU utilization can show 0% during parts of shader compilation because the work may be CPU-side, driver-side, or CUDA compute rather than graphics utilization.

Subsequent steps are usually fast once the shader cache is populated. If every run pays the full cold-start cost, check the local cache configuration in `dependencies` and make sure the cache directory is writable and persistent for the intended environment.

## Schema Registration And Path Dictionary

OVStage and standalone ovrtx runtime population are schema-driven. Built-in stage data such as Cameras, RenderProducts, RenderVars, `omni:xform`, pickability, and selection-outline attributes use supported schemas. If an app depends on custom authored attributes, register the schema before loading the stage, or opt in with root-layer `customLayerData.populateAllAuthoredAttributes = true` when broad authored-attribute population is acceptable. The broad flag can significantly increase memory usage.

Path and token IDs returned by C queries or pick-hit buffers are not user-facing
strings. In attached mode, decode OVStage query/read/write IDs with the
OVStage-owned path dictionary and decode ovrtx pick/render-output IDs with the
renderer-owned dictionary. Do not use one library's dictionary for IDs produced
by the other. Python pick-hit decoding uses `Renderer.resolve_prim_path_id()`.

For picking and selection-outline implementation, read `native-picking-selection`
before writing code. Use `Renderer.enqueue_pick_query()` with normalized
top-left RenderProduct bounds, `Renderer.resolve_prim_path_id(int_id)`,
`Renderer.set_selection_group_styles({group_id: SelectionGroupStyle(...)})`, and
`Renderer.set_selection_outline_group_strings(...)`. A consumed native query
produces `OVRTX_RENDER_VAR_PICK_HIT` automatically; do not author it in USD
`orderedVars`. Do not replace these with `hasattr`-driven picker fallbacks unless
native picking is unavailable and the user explicitly asks for a compatibility
path.

## Geometry Streaming

For USD assets in new ovrtx apps, prefer OVStage population and
`population.add_usd_reference*` composition. Use geometry streaming only for
runtime-generated geometry or high-frequency geometry updates that are owned by
the application. Keep schemas registered before streaming attributes, keep
stream ownership/lifetime explicit, and continue to render with ovrtx; do not
replace this with browser-side geometry rendering.

## Multi-GPU

`RendererConfig(active_cuda_gpus="0,1")` enables multiple CUDA-visible GPUs for rendering, subject to the installed ovrtx build and RenderProduct configuration. Per-RenderProduct `uint[] deviceIds = [...]` is an allow-list into `CUDA_VISIBLE_DEVICES`; use it when a product must run on a specific device.

For WebRTC streaming and Warp conversion, keep the selected display RenderProduct on the CUDA device used by the stream buffer or add an explicit copy. Picking currently requires the picking RenderProduct to run on CUDA-visible GPU `0`; pin pick products with `deviceIds = [0]` when multi-GPU rendering is enabled.

## Environment

```bash
export OVRTX_SKIP_USD_CHECK=1
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))')"
export LD_LIBRARY_PATH="$OVRTX_BIN_PATH/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If a separate USD build is present, ovrtx's bundled USD/plugins must resolve first or USD debug symbols can be registered twice (`SDF_ASSET` fatal errors).

## Import Order

Use one import discipline per process:

- Attached OVStage viewer (the default for new apps): set
  `OVRTX_SKIP_USD_CHECK`, import `ovrtx`, call
  `ovrtx.register_schema_paths()`, then create/populate OVStage before any
  other OpenUSD consumer initializes. This ordering wins over convenience
  import-order guidance for non-attached shells.
- Streaming/direct ovrtx server: set `OVRTX_SKIP_USD_CHECK`, construct `Renderer`, then import `pxr` if direct queries are needed.
- Verified in-process OpenUSD viewer/editor paths: import `pxr`
  first, then let local renderer/adapter code import ovrtx as documented in
  `local-viewer` and ovui guidance.
- Windows streaming path: keep `pxr` in a subprocess; do not import it in the ovrtx process.

The reason is USD plugin registry ownership. Mixing `usd-core` and ovrtx's bundled USD in the wrong order can cause MDL resolver failures, `_tf` DLL import failures, or duplicate symbol crashes.

## Minimal Render Example

```python
import os

os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

import numpy as np
import ovrtx
ovrtx.register_schema_paths()
import ovstage
from PIL import Image
from ovstage import PopulationDomain, population
from ovrtx import Renderer, RendererConfig, Device

stage = ovstage.Stage("viewer")
ordinal = 1
population.open_usd(stage, "/path/to/composite.usda", ordinal=ordinal, domains=PopulationDomain.RENDERING)
stage.advance_write_floor(ordinal).wait()
renderer = Renderer(config=RendererConfig(sync_mode=True))
renderer.attach_ovstage(stage)

for _ in range(60):
    products = renderer.step(
        render_products={"/Render/RenderProduct"},
        delta_time=1/60,
        ordinal=ordinal,
    )
    product = products["/Render/RenderProduct"]
    for frame in product.frames:
        ldr_path = "/Render/Vars/LdrColor"
        if ldr_path in frame.render_vars:
            with frame.render_vars[ldr_path].map(device=Device.CPU) as rv:
                rv.wait()
                Image.fromarray(np.from_dlpack(rv).copy()).save("output.png")
```

## MDL Material Resolution

ovrtx bundles MDL assets under `ovrtx/bin/library/mdl/`, including `Base/OmniPBR.mdl`, `Base/OmniGlass.mdl`, and `mdl/nvidia/core_definitions.mdl`. Without `OVRTX_BIN_PATH` pointing at `ovrtx/bin`, materials importing `::OmniPBR::OmniPBR` or `::nvidia::core_definitions::*` can render magenta.

UJITSO "multi-node material unsupported" warnings are informational if the full MDL compiler can find the library. They become a problem when `OVRTX_BIN_PATH` or `LD_LIBRARY_PATH` is wrong.

## Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `CRenderApi not found` | plugin tree missing | set `OVRTX_BIN_PATH` |
| `multiple debug symbol definitions for SDF_ASSET` | two USD instances | put ovrtx bundled libs first |
| `usd-core detected` | version check conflict | set `OVRTX_SKIP_USD_CHECK=1` |
| `Default.mdl` parse crash | renderer initialized after wrong USD registry | fix import/construction order |
| `RenderProductSetOutputs` has no `.get` | treated the result as a Python dict | use membership checks and `products[path]` |
| invalid output handle after returning frame data | renderer-backed output was retained past its valid lifetime | return an owned copy made inside the RenderVar map context |
| first `renderer.step()` appears hung | cold RTX shader or pipeline compilation | use a 300s+ first-run timeout and inspect ovrtx logs |
| `write_attribute` does nothing | used direct renderer writes in an attached-stage app, or missed compatibility prim mode | write through OVStage at a new ordinal; for standalone compatibility use `PrimMode.CREATE_NEW` |
| transform not visible | wrote authored `xformOp:*`, used wrong matrix layout, or rendered above the write floor | write `omni:xform` through OVStage as a row-major 16-lane matrix, advance the floor, then step with that ordinal |
| `Semantic.XFORM_MAT4x4` becomes `NONE` | imported adapter implementation module `_ovrtx` | `import ovrtx` directly |
| single-tensor examples fail or hide data | current render vars can be multi-tensor | use `np.from_dlpack(rv)` or named tensors such as `rv["TensorName"]` |
| AOV listed but blank | ovrtx produced empty tensor | enable PT flags, check the full RenderVar path and source name |
| `DepthSD` mapping fails | deprecated depth AOV | use `DistanceToImagePlaneSD` for axial depth or `DistanceToCameraSD` for radial distance; update the UI label accordingly |
| `Diffuse` empty/fails | wrong source name | use `DiffuseAlbedoSD` not `Diffuse` |
| `Roughness` mapping fails | unsupported in current ovrtx build | do not expose |
| red/blue swapped in browser | streamed RGBA directly | convert to BGRA before `stream_video()` |
| magenta materials | MDL resolver path missing | set `OVRTX_BIN_PATH` and plugin `LD_LIBRARY_PATH` |
| stale hangs | old GPU process | inspect `nvidia-smi`, kill stale Omniverse Realtime Viewer PIDs |

See also: `aov-switching`, `stage-loading`, `camera-controls`, `render-settings`, `selection-feedback`, `selection-animation`, `streaming-server`.

## Processed GPU outputs

For custom post-processing, chained resources or lens sampling, read
[SPG processing](../spg-processing/README.md). For stylized image passes,
technical/X-Ray geometry strokes or explicit full wireframe, continue to
[stylized visualization](../stylized-visualization/README.md). Present the actual
published processed output through this viewer's existing owner and transport;
a menu selection or display conversion alone does not implement the effect.
