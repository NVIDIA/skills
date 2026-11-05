# ovstage + ovrtx Integration

Use this reference to connect an application-owned stage to the RTX renderer.
The stage owns runtime scene data; ovrtx owns render products, frame outputs,
renderer-native picks, and selection visualization.

## Render loop

Before creating or first populating the `ovstage.Stage`, call
`ovrtx.register_schema_paths()` in Python (or the matching C registration call)
before any OpenUSD consumer initializes. Create a Stage and attach it with
`renderer.attach_ovstage(stage)`. Population may precede or follow attachment,
but publish its initial ordinal before the first render step.
An OVStage instance can be attached to one live renderer only. Keep it alive
through attachment; renderer replacement follows the lifetime rules below.

On each logical update, commit stage work at ordinal `N`, then render the
committed publication:

```text
stage mutation or population at N
  -> wait for operations
  -> advance write floor to N
  -> renderer consumes N (explicit update and wait in the C API)
  -> renderer step for N
  -> map outputs and copy into app-owned display/transport buffers
```

Python `renderer.step(render_products, delta_time, ordinal=N)` performs the
attached-stage update before rendering, so most viewers do not need a separate
`renderer.update_from_stage(N)`. `renderer.update_from_stage(N)` remains useful
when an app needs renderer state synchronized before the next step. In attached
mode, `ordinal` is required for `step()` and invalid in standalone mode.

In C/C++, call `ovrtx_update_from_stage(renderer, N)` and wait for completion
before `ovrtx_step_with_stage(renderer, products, dt, N, &handle)`. Apply native
selection/pickability state after that update when population has introduced or
recreated prims. The C step does not perform the Python wrapper's automatic
update.

Do not render above the write floor. If an app publishes per frame, finish or
fence the prior render before publishing the next update. Do not mutate OVStage
or standalone renderer stage state while `renderer.step()` or `step_async()` is
in flight.

The renderer's pick queue and selection-outline configuration remain
renderer-owned: enqueue picks against the render product, resolve ovrtx pick IDs
through the renderer path dictionary, and discard results from an older stage
generation. Resolve OVStage query/read IDs through the OVStage path dictionary;
never decode one library's IDs with the other library's dictionary.

Keep absolute USD sample time separate from accumulated renderer step deltas.
Pause/seek must deliberately coordinate both; see
[population timing](../ovstage-population/README.md#usd-playback).

## Lifetime and scene changes

Keep the Stage alive while attached. Finish in-flight operations before
`detach_ovstage()`, release stage-specific queries and mappings, then destroy or
replace the Stage. If detach raises, destroy that renderer; it cannot step or
reattach after a failed detach. `renderer.reset()` resets sensor history and
simulation time; use [stage-management](../stage-management/README.md) to replace
scene contents.

### Replacing a renderer for a payload scene

On affected OVRTX builds, detaching or destroying a renderer can unload payloads
from its borrowed Stage. Reattaching that same Stage to a replacement renderer
can show a correct first frame, then lose payload geometry on the next
`apply_usd_changes`. Do not treat a successful first frame as proof that
same-Stage replacement is safe.

Prefer keeping the renderer/Stage pairing for payload scenes. If renderer
replacement is required on an affected or unverified build, follow the lifetime
rules above, then populate a **fresh Stage**
from the source and restore viewer-owned state through
[stage-management](../stage-management/README.md#replace-a-scene). Attach that
fresh Stage to the replacement renderer and validate payload contents both
before and after a subsequent USD update. Destroying the old renderer alone
does not avoid the affected cleanup path; skipping detach or attaching the
same Stage to two live renderers is not a workaround.

Follow the selected release's upstream
[Stage ownership contract](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/docs/core/ovstage_integration.rst).
Enable same-Stage renderer replacement only after verifying the exact installed
build preserves payloads through detach, reattach, and subsequent USD updates.

## Scene ownership

Use OVStage for scene population, queries, transforms, material/effect writes,
and cloning. Rendering, physics, sensors, and automation can then share the
application-owned stage and publication clock.

Standalone renderer stage APIs remain available for existing renderer-owned
apps. Keep one scene data plane for a given app path: attached OVStage work
uses OVStage mutation/query APIs, while renderer-native picks and outlines
remain OVRTX operations.

The browser, Electron, Tauri, and ovui shells consume copied runtime state and
rendered frames. Check first-frame readiness, camera edits, selection, property
queries, scene switching, and shutdown on the chosen shell.
