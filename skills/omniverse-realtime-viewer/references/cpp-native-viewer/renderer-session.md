# C++ Renderer And Session Setup

## Renderer Setup

Create the renderer on its owner thread with sync mode and native selection outlines enabled. Register schema paths before any OVStage/OpenUSD initialization.
The C API uses typed `ovrtx_config_entry_t` arrays, not JSON strings.

```cpp
#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>

static ovrtx_renderer_t* createRenderer()
{
    const ovrtx_config_entry_t configEntries[] = {
        ovrtx_config_entry_sync_mode(true),
        ovrtx_config_entry_active_cuda_gpus(literal_to_ovx_string("0")),
        ovrtx_config_entry_selection_outline_enabled(true),
    };
    const ovrtx_config_t config = {
        configEntries,
        sizeof(configEntries) / sizeof(configEntries[0]),
    };

    if (!ok(ovrtx_register_schema_paths(&config))) {
        printLastOvrtxError("Failed to register RTX schema paths");
        return nullptr;
    }
    ovrtx_renderer_t* renderer = nullptr;
    const ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
    if (!ok(result) || !renderer) {
        printLastOvrtxError("Failed to create OVRTX renderer");
        return nullptr;
    }
    return renderer;
}
```

Available config entries include `ovrtx_config_entry_sync_mode(true)`,
`ovrtx_config_entry_active_cuda_gpus(...)`,
`ovrtx_config_entry_selection_outline_enabled(true)`, and
`ovrtx_create_renderer(...)`.

For a stage-first attached C/C++ viewer, call
`ovrtx_register_schema_paths(&config)` before the first OVStage initialization
or population (and before another OpenUSD consumer initializes). It publishes
the RTX schema definitions that OVStage needs to resolve populated fallbacks.
One live renderer may attach to an OVStage instance; detach before reassigning
that stage.

## Session Layer Pattern

Load an inline root USDA with the user stage as a sublayer plus viewer-owned
camera, render product, render vars, and render settings. The render var names
must map to real OVRTX source names, while output lookup uses the full authored
RenderVar path. `LdrColor` drives display, and
`ovrtx_pick_hit` is consumed only after a pick query.

Author the composite with USD layer authoring so source asset paths and
relationship targets are serialized correctly. Choose one app-owned set of
prim paths and use it consistently for definitions, camera/render relationships,
step requests and output lookup. A minimal layout is:

| Prim | Authored state |
|---|---|
| `/OVCamera` | Perspective camera, aperture matching the output aspect, clipping range and initial transform derived from scene bounds/units/up-axis |
| `/OVRenderProduct` | Camera relationship to `/OVCamera`, requested resolution, `uint[] deviceIds = [0]`, ordered var `/OVRenderProduct/LdrColor` |
| `/OVRenderProduct/LdrColor` | RenderVar with `sourceName = "LdrColor"` |
| `/OVRenderSettings` | Products relationship to `/OVRenderProduct` |

The composite sublayers the requested scene; it does not add lighting unless
viewer-controlled lights were requested. Keep composition and optional metadata
preparation outside the active UI/render loop. A C++ authoring helper or optional
isolated worker can prepare the layer; follow the selected dependency's OpenUSD
ABI boundary. No Python process is required by the native renderer.

## Populate And Attach

Initialize OVStage with `ovstage_initialize()`, then create an app-owned instance
with `ovstage_create_instance()`. Use the matched C package and the same loader
model as OVRTX, as described in `project-build.md`. Populate the composite into
that instance at an explicit ordinal; population and data-plane operations use
separate wait APIs:

```cpp
#include <ovstage/ovstage.h>
#include <iostream>

static bool waitStageOperation(
    ovstage_instance_t* stage, ovstage_enqueue_result_t op)
{
    if (op.status != OVSTAGE_OK) return false;
    ovstage_op_wait_result_t result{};
    const auto status = ovstage_wait_op(
        stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const bool success = status == OVSTAGE_OK && result.error_op_id_count == 0;
    for (size_t i = 0; i < result.error_op_id_count; ++i)
        std::cerr << fromOvxString(ovstage_get_last_op_error(stage, result.error_op_ids[i])) << "\n";
    ovstage_release_op(stage, op.op_index);
    return success;
}

static bool publishStage(ovstage_instance_t* stage, ovstage_ordinal_t ordinal)
{
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = ordinal;
    floor.scope = OVSTAGE_SCOPE_ALL;
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floor));
}

static bool populateComposite(
    ovstage_instance_t* stage, const std::string& usda,
    ovstage_ordinal_t ordinal)
{
    const auto op = ovstage_population_open_usd_from_string(
        stage, toOvxString(usda), ordinal, 0.0,
        OVSTAGE_POPULATION_DOMAIN_RENDERING);
    if (op.status != OVSTAGE_OK) return false;
    ovstage_population_op_wait_result_t result{};
    const auto status = ovstage_population_wait_op(
        stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    for (size_t i = 0; i < result.error_op_id_count; ++i)
        std::cerr << fromOvxString(ovstage_population_get_last_op_error(result.error_op_ids[i])) << "\n";
    return status == OVSTAGE_OK && result.error_op_id_count == 0
        && publishStage(stage, ordinal);
}
```

These helpers return failures to the owner, which publishes an error snapshot
and unwinds owned resources. Read operation diagnostics before releasing their
records. Keep the population string and write tensors alive until their waits
complete. After successful population/publication, attach with
`ovrtx_attach_ovstage(renderer, stage)` and check its result. Initialize the
camera and any app-owned runtime attributes through OVStage at the next ordinal
(strictly above the sealed write floor) before the first
published frame. Native frame ordering is in `presentation-loop.md`.

## Replace And Close

Prepare a candidate composite and metadata with a request generation. Keep the
active scene usable during preparation; only the latest requested candidate may
activate. At the owner-thread frame boundary, finish outstanding operations,
release maps/results/queries, detach the old stage, then attach the prepared
replacement. Clear stale picks, selected paths, bindings, and metadata on an
ordinary scene switch. For same-scene variant/visibility recomposition, preserve
only valid app-owned camera, accepted edits, and selection intent; reapply native
selection after the renderer has consumed the new population.

Keep the old stage until activation succeeds when implementing rollback. A
failed detach leaves the renderer unusable except for destruction. Follow
`ovstage-ovrtx-integration` and `stage-management` for lifetime and replacement
semantics; renderer reset is not scene replacement.

On close, stop accepting commands, cancel owned preparation work, finish the
owner's in-flight operation, release outputs and queries, detach, destroy the
renderer and stage, then shut down OVStage. Join the owner before destroying its
shared state. Release presentation resources and close the selected toolkit on
its required UI thread (for example, GL textures and GLFW/ImGui in the worked example).
