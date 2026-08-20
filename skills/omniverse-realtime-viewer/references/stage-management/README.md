# Stage Management

## Triggers

Use this skill for switch scenes, load another file, change USD, asset browser, scene dropdown, persist across scenes, or reload stage.

Use this skill when the Omniverse Realtime Viewer needs multiple USD files, stage reload, reset, additive composition, or state that survives scene changes. New ovrtx viewers manage those changes as OVStage population/data-plane generations and render committed ordinals.

## Asset Discovery

Populate scene selectors from one of these sources:

- Local samples directory: scan for `.usd`, `.usda`, and `.usdc`; display basename, store absolute path.
- Cloud/cache source: resolve through `cloud-assets`, then list cached local files.
- User-provided path: validate with `Usd.Stage.Open()` for metadata queries before handing it to OVStage population.

Keep UI labels separate from load paths. Relative USD asset references require the stage file to remain in its original directory or in a cache preserving directory structure.

## Initial Stage Agreement

The server and frontend must agree on the initial stage. If the server starts with a stage already loaded, `push_initial_state()` should send `openStageResult` with the current stage URL and the frontend should accept that as authoritative. Do not let a frontend `useEffect([status])` blindly send `openStageRequest` for the dropdown default after every WebRTC reconnect; that can override the server-loaded stage.

If the frontend still sends an initial `openStageRequest`, the server must compare it with the already-loaded stage and return success immediately when they match.

```python
def same_stage(requested: str, current: str | None) -> bool:
    return bool(current) and os.path.normpath(requested) == os.path.normpath(current)
```

Skipping redundant reloads prevents unnecessary render interruption, CUDA context resets, long shader recompilation, and possible WebRTC disconnects.

## Stage Composition Policy

In ovrtx attached mode, stage replacement and additive composition are
OVStage population operations:

- Use `ovstage.population.open_usd(stage, path, ordinal=N, domains=PopulationDomain.RENDERING)` to replace the active root source with a file/URL-backed composed stage.
- Use `ovstage.population.open_usd_from_string(stage, usda, ordinal=N, domains=PopulationDomain.RENDERING)` for generated viewer/session USDA, commonly an inline root that sublayers the user scene.
- Use `ovstage.population.add_usd_reference*()` only for additive content under a unique target path. Keep the returned handle, call `ovstage.population.apply_usd_changes(stage, ordinal=N)`, wait, and advance the write floor.
- Use `ovstage.population.remove_usd(stage, handle)` to remove additive content, then `apply_usd_changes(stage, ordinal=N)`, wait, and advance the write floor.
- Use `ovstage.population.reset_usd(stage)` only for an explicit "clear scene" or shutdown flow; follow with `apply_usd_changes(stage, ordinal=N)` and publish the generation.

Direct `renderer.open_usd*`, `add_usd_reference*`, `remove_usd()`, and
`reset_stage()` are standalone compatibility APIs. Keep them off the production
path when an OVStage is attached.

Attach the renderer after the first successful population publication. Keep the
attachment across scene switches that reuse the same `ovstage.Stage`; detach and
reattach only when replacing or destroying the Stage instance.

## Hot-Swap Sequence

Run stage switching through the one runtime owner unless you have a dedicated
loading worker. Do not call `renderer.step()` while OVStage population,
`apply_usd_changes`, write-floor advancement, renderer `attach_ovstage`,
`detach_ovstage`, `update_from_stage`, or standalone renderer stage mutation is
active.

```python
def switch_scene(path: str):
    selection.clear()
    info_panel.hide()
    tree.reset()
    animator = None

    usd_stage = Usd.Stage.Open(path)          # hierarchy, bbox, material map
    camera_state = camera.snapshot()          # preserve if requested
    settings_state = settings.to_dict()
    ordinal = runtime.next_ordinal()

    # Replace-root population. path_or_composite(path) may be the user USD or a
    # generated wrapper USDA that sublayers the user USD and authors viewer prims.
    population.open_usd(runtime.stage, path_or_composite(path), ordinal=ordinal)
    runtime.stage.advance_write_floor(ordinal).wait()
    renderer.update_from_stage(ordinal)       # optional if the next step uses ordinal

    effect_ordinal = runtime.next_ordinal()
    reset_effect_layer_faders(runtime.stage, usd_stage, ordinal=effect_ordinal)
    runtime.stage.advance_write_floor(effect_ordinal).wait()

    material_map = build_prim_material_map(usd_stage)
    picker.rebuild(usd_stage)
    animator = build_animator(runtime.stage, usd_stage, pickable_paths)
    tree.attach_ovstage(usd_stage)

    settings.apply(settings_state, runtime.stage, usd_stage)
    camera.restore_or_fit(camera_state, usd_stage)
```

For generated viewer/session USDA that should not be written to disk:

```python
ordinal = runtime.next_ordinal()
population.open_usd_from_string(
    runtime.stage,
    make_viewer_root_usda(path, width, height),
    ordinal=ordinal,
)
runtime.stage.advance_write_floor(ordinal).wait()
```

For additive scene content:

```python
handle = population.add_usd_reference(
    runtime.stage,
    asset_path,
    "/Runtime/Assets/Asset_001",
)
ordinal = runtime.next_ordinal()
population.apply_usd_changes(runtime.stage, ordinal=ordinal)
runtime.stage.advance_write_floor(ordinal).wait()
# Later:
population.remove_usd(runtime.stage, handle)
ordinal = runtime.next_ordinal()
population.apply_usd_changes(runtime.stage, ordinal=ordinal)
runtime.stage.advance_write_floor(ordinal).wait()
```

## Async Operations

Python OVStage population `open_usd()` / `open_usd_from_string()` are blocking
convenience calls. Use the `_async` variants for non-blocking loads and poll or
wait for the returned `Operation` from the render/runtime owner:

```python
ordinal = runtime.next_ordinal()
op = population.open_usd_async(runtime.stage, path_or_composite(path), ordinal=ordinal)
while True:
    code, error_op_ids, _ = runtime.stage.wait_op(op.op_id, timeout=0)
    if code == ovstage.ErrorCode.TIMEOUT:
        stream_last_good_frame()
        continue
    if code == ovstage.ErrorCode.OK and not error_op_ids:
        break
    raise RuntimeError(op.error_message())

op.wait()
runtime.stage.advance_write_floor(ordinal).wait()
```

Apply the same pattern to async reset/reference population operations and
OVStage data-plane query/read/map operations. Wait with the API that enqueued
the work, then fetch/release the OVStage result handle before reading
dictionaries.

Do not treat an async enqueue or a C return value as proof that the stage is
loaded. Poll/query or wait for completion, advance the write floor, and update
or step the attached renderer with the committed ordinal before rebuilding pick
maps, hierarchy, material maps, animation bindings, or before reporting
`openStageResult: success`.

## Dynamic Root Prim

Never hardcode `/World` as the scene root. Many NVIDIA samples use `/World`, but other USD assets may use a different root such as `/stage`. Detect the root when opening the stage and pass it through stage-load state.

Root detection order:

1. Use `/World` when it exists.
2. Fall back to `stage.GetDefaultPrim()`.
3. Fall back to the first pseudo-root child that is not a viewer/session/render prim.

Include `root_prim_path` in `openStageResult` so the frontend knows where to start hierarchy queries. The stage tree, child queries, selection expansion, and `makePrimsSelectable` flow must use this dynamic root instead of a hardcoded `/World`.

## Preserve Camera

Use a policy, not an accident:

- `preserve`: keep azimuth/elevation/distance/target across stages.
- `fit`: compute bbox and frame the new scene.
- `stage-camera`: use the first authored camera if available, then fall back to bbox fit.

Camera state should be sanitized after restore. If a target or distance is non-finite, fall back to bbox center and a positive distance.

## Preserve Settings

Validated render settings and non-live profile defaults belong to app state, not the USD asset unless the user asks to author the file. Save settings JSON and re-apply only settings with a verified OVStage/session apply path after every replace-root population load and after additive composition changes that affect render settings.

```python
settings = RenderSettings.load("viewer_settings.json")
settings.apply_validated_settings(session_layer)
settings.save("viewer_settings.json")
```

Use `render-settings` for the schema and lighting controls.

## Reset, Reload, And Remove

`resetStageRequest` should reload the current scene from its source with
OVStage `population.open_usd()` or `open_usd_from_string()` and a scene-manager
`force=True` flag, publish the new ordinal, then rebuild all derived state:
hierarchy, pick buffers, material map, selection feedback, animator base
transforms, and info panel state. It does not need a response in the existing
protocol, but local UI should visually clear selection immediately.

Use OVStage `population.reset_usd()` only for an explicit "clear scene" source
state. Use `renderer.detach_ovstage()` or standalone `renderer.reset_stage()`
only for teardown/compatibility flows where the renderer should stop rendering
the attached stage. A reload of the current scene is not a clear; it is another
replace-root population load.

For additive references, remove only the handle returned by
`population.add_usd_reference*`, call `apply_usd_changes(stage, ordinal=N)`,
and publish that ordinal. Do not reset the entire stage to remove one additive
asset unless the intended result is to discard the entire root source and every
reference.

## Stage Switch Side Effects

After each new stage load:

- Write all EffectLayer shader `inputs:Fader` values to `0` through the
  generated session/composite layer or OVStage runtime data plane.
- Render at least two frames before trusting any display/debug segmentation AOV.
- Recompute pickable bbox data and descendant mesh expansion maps.
- Rebuild the stage tree/sidebar under the detected `root_prim_path`.
- Refit or restore camera before the next visible frame.
- Recreate transform/animation runtime state; do not reuse old OVStage queries,
  maps, direct renderer bindings, or path IDs across replace-root loads or
  renderer stage resets.

## Failure Modes

- Scene appears textureless after switching: composite/cache path broke relative asset resolution.
- Highlight starts glowing before selection: EffectLayer Faders were not reset in the session/composite layer or OVStage runtime state after reload.
- Picks return old prims: cached pick/path IDs survived a scene reload; clear ID maps and resolve new IDs through the current renderer path dictionary.
- Camera inside geometry: preserved distance/target does not fit the new scene; use bbox fit.
- Crash or hang on switch: `renderer.step()` ran concurrently with OVStage population/write-floor work or standalone renderer stage mutation.
- Success reported too early: async `Operation` was enqueued but not completed; poll or wait, advance the write floor, and step/update the renderer before rebuilding derived state.
- Wrong stage after reconnect: frontend requested its dropdown default instead of accepting the server's current stage from initial state.
- Long reload of the same scene: missing normalized-path check before starting a reload.
- Empty or wrong hierarchy for valid assets: code assumed `/World` even though the loaded stage used another root prim.

See also: `stage-loading`, `camera-controls`, `render-settings`, `object-selection`, `selection-feedback`, `selection-animation`, `stage-hierarchy`, `cloud-assets`.

## Adding This To An Existing Omniverse Realtime Viewer

- Add `server/scene_manager.py` or equivalent ownership around scene discovery, load, reset, and reload.
- Keep server state for current URL, loading state, hierarchy root, selection,
  camera policy, settings snapshot, active stage generation, and last committed
  OVStage ordinal.
- Add messages for `openStageRequest`, `openStageResult`, `resetStageRequest`, `loadingStateQuery`, and `loadingStateResponse`.
- Route all OVStage population, data-plane writes, write-floor advances, renderer
  updates, and renderer steps through the runtime thread.
- Modify `scene_loader.py` to rebuild viewer camera, RenderProduct, RenderVars, and optional wrapper files or inline root USDA per scene.
- Reapply validated render settings and camera policy after each load before the first visible frame.
- Clear selection, pick maps, info panels, hierarchy caches, highlight faders,
  OVStage queries/maps, direct renderer bindings, and animation state on switch.
- Frontend wires a scene picker or asset browser to `openStageRequest` and displays load/error state from responses.
- Persist cross-scene settings in an app JSON file, not in user USD assets.
- Push current scene, loading state, settings, selection, and root hierarchy to newly connected clients.
