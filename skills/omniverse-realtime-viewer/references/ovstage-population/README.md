# ovstage Population

Use this reference for loading a user USD scene, composing viewer-owned camera
and render setup, switching scenes, additive assets, references, variants, or
USD time updates.

## Authoring versus runtime

Keep the user asset immutable. Build a composed USD source that layers or
references the asset with viewer-owned camera, RenderProduct, RenderVar, and
RenderSettings data. Populate that composition once into the application-owned
stage. USD remains the authoring/composition boundary; ovstage is the runtime
representation consumed by the application and libraries.

For ovrtx viewer apps, use OVStage population as the default load path:

- `population.open_usd(stage, path, ordinal=N, domains=PopulationDomain.RENDERING)` for file/URL-backed wrapper stages.
- `population.open_usd_from_string(stage, usda, ordinal=N, domains=PopulationDomain.RENDERING)` for generated inline wrapper/session USDA.
- `population.add_usd_reference*()` / `remove_usd()` / `reset_usd()` for source edits after a root stage is open, followed by `population.apply_usd_changes(stage, ordinal=N)`.
- `population.update_from_usd_time(stage, ordinal=N, time_code=seconds_value)` for time-sampled USD playback.

After each population operation, wait for completion and call
`stage.advance_write_floor(N).wait()` before queries, attached renderer updates,
or `renderer.step(..., ordinal=N)`.

### Populate, then COMMIT — a reference add/remove is not live until committed

`population.add_usd_reference_from_string(...)` (and `add_usd_reference*` /
`remove_usd`) stage a source edit; they **do not take effect until you commit
them** with `population.apply_usd_changes(stage, ordinal=N)` followed by
`stage.advance_write_floor(N).wait()`. A reference added without that commit
**silently never populates into Fabric** — no error, the prim simply is not
there, so any later toggle or query against it no-ops. This is load-bearing for
incremental scene edits (e.g. adding a context asset behind a visibility toggle,
or swapping a geometry subtree):

```python
handle = population.add_usd_reference_from_string(stage, usda, prim_path)
ordinal = current_ordinal + 1
population.apply_usd_changes(stage, ordinal=ordinal)   # <- without this, the ref is a no-op
stage.advance_write_floor(ordinal).wait()
current_ordinal = ordinal                              # then queries / step @ordinal
```

Symmetrically, a `remove_usd` also needs `apply_usd_changes` + `advance_write_floor`
to actually drop the reference. Contrast with `omni:xform` and other data-plane
attribute writes, which take effect on their own write floor without
`apply_usd_changes` — the commit step is specific to **structural** (reference /
source) edits.

## Load and switch workflow

1. Stop accepting commands that depend on the old generation.
2. Clear selection, hierarchy caches, animation bases, and pending picks.
3. Allocate a new ordinal and apply the population/reset/reference operation.
4. Wait, advance the write floor, then rebuild query handles and derived DTOs.
5. Publish the new generation only after the first committed render is ready.

Do not use renderer reset/load calls as a scene-management mechanism in attached
OVStage apps. Treat a stage switch as a generation change: all path tokens,
query results, maps, renderer pick IDs, and asynchronous responses from the
prior generation are invalid. If the app replaces the entire `ovstage.Stage`
instance, detach the renderer before destroying the old instance and attach the
newly populated instance after publication.

## Generated-application checks

A reconstructed viewer must demonstrate: a user USD remains unchanged; the
viewer camera/render configuration comes from the composed source; scene switch
clears stale selection; and the first frame is rendered from a committed stage
publication.
