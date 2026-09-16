# ovstage + ovrtx Integration

Use this reference to connect an application-owned stage to the RTX renderer.
The stage owns runtime scene data; ovrtx owns render products, frame outputs,
renderer-native picks, and selection visualization.

## Render loop

Create and populate the `ovstage.Stage` first, publish its initial ordinal, then
attach the renderer with `renderer.attach_ovstage(stage)`. Attach once for a
Stage instance; detach before destroying or replacing that instance.

On each logical update, commit stage work at ordinal `N`, then render the
committed publication:

```text
stage mutation or population at N
  -> wait for operations
  -> advance write floor to N
  -> renderer update_from_stage(N) when a pre-step update is needed
  -> renderer step for N
  -> copy/map frame output before the next step
```

Python `renderer.step(render_products, delta_time, ordinal=N)` performs the
attached-stage update before rendering, so most viewers do not need a separate
`renderer.update_from_stage(N)`. `renderer.update_from_stage(N)` remains useful
when an app needs renderer state synchronized before the next step. In attached
mode, `ordinal` is required for `step()` and invalid in standalone mode.

Do not render above the write floor. If an app publishes per frame, finish or
fence the prior render before publishing the next update. Do not mutate OVStage
or standalone renderer stage state while `renderer.step()` or `step_async()` is
in flight.

The renderer's pick queue and selection-outline configuration remain
renderer-owned: enqueue picks against the render product, resolve ovrtx pick IDs
through the renderer path dictionary, and discard results from an older stage
generation. Resolve OVStage query/read IDs through the OVStage path dictionary;
never decode one library's IDs with the other library's dictionary.

## Compatibility Boundary

Standalone direct `renderer.open_usd*`, `add_usd_reference*`, `remove_usd()`,
`reset_stage()`, `query_prims*`, `read_attribute*`, `write_attribute*`,
`bind_attribute*`, and `map_attribute()` remain available for compatibility and
targeted standalone tests. New interchangeable viewer features should use
OVStage for population, queries, runtime transforms, material/effect writes, and
cloning so rendering, physics, sensors, and automation can consume the same
runtime stage.

## Rebuild contract

When implementing a viewer from these skills, keep the server/runtime loop in one
module with explicit stage and renderer ownership. Validate a first committed
frame, camera mutation, selection, hierarchy/property query, scene switch, and
shutdown/detach sequence. The browser, Electron, Tauri, and ovui layers remain
presentation/transport shells for the same runtime contract.
