# ovstage Population

Use this reference to load composed USD, add or remove assets, and play USD
animation in an application-owned OVStage.

## Choose the operation

Keep the user asset unchanged. Compose it with viewer-owned cameras, render
products, render vars, and settings in a wrapper/session source. Register RTX
schemas with `ovrtx.register_schema_paths()` before constructing or populating
OVStage so USD schema fallbacks are available on the first load.

| Intent | Operation |
| --- | --- |
| Open, replace, or fully reload a scene | Populate a fresh Stage with `open_usd()` or `open_usd_from_string()`, then switch attachments using [stage-management](../stage-management/README.md). |
| Add an asset to the current scene | `add_usd_reference*()` at a unique target path; retain the returned handle. |
| Remove an added asset | `remove_usd(stage, handle)`. |
| Clear the current USD source | `reset_usd(stage)`. This preserves application-authored runtime data. |
| Play time-sampled USD | `update_from_usd_time(stage, ordinal=N, time=seconds)`. |
| Restore selected attributes from the retained USD composition | `repopulate_usd_with_desc(stage, ordinal, time, descs)`. |

`open_usd*` overlays populated data; it does not remove runtime prims or columns
absent from the new source. A fresh Stage gives complete scene replacement a
clear lifetime boundary. Repopulation retains source/session edits and reference
handles; use it for restoring selected runtime values, not replacing a scene.

## Populate and publish

```python
from ovstage import PopulationDomain, population

population.open_usd(
    stage, composite_path, ordinal=ordinal,
    domains=PopulationDomain.RENDERING,
)
stage.advance_write_floor(ordinal).wait()
```

For generated root USDA, use `open_usd_from_string(stage, usda, ordinal=ordinal,
domains=PopulationDomain.RENDERING)`. Blocking population calls complete before
returning; `_async` variants return an operation to poll or wait on. Publish the
write floor only after the operation completes. Queries and renderer steps then
consume that committed ordinal.

## Edit the composition

Reference add/remove and source reset change the USD composition. Apply those
changes before publishing:

```python
handle = population.add_usd_reference_from_string(stage, usda, prim_path)
ordinal = current_ordinal + 1
population.apply_usd_changes(stage, ordinal=ordinal)
stage.advance_write_floor(ordinal).wait()
current_ordinal = ordinal
```

Removal follows the same apply/publish sequence after
`population.remove_usd(stage, handle)`. Direct runtime attribute writes use the
[OVStage data plane](../ovstage-data-plane/README.md) and publish their ordinal
without `apply_usd_changes()`.

## USD playback

`time=` is absolute time in seconds, converted using the source stage's
`timeCodesPerSecond`. For population APIs with an optional time, the default is
`population.DEFAULT_TIME` (USD default time), distinct from numeric zero.

USD sampling time and renderer animation time are separate clocks. Populate at
the desired absolute USD time; pass elapsed seconds as `renderer.step`
`delta_time` for renderer-driven video, MDL animation, and animated splat/NuRec
content. A USD time change with zero step delta leaves that renderer animation
frozen. On seek, use the selected renderer reset/time contract when appropriate;
reset discards temporal history. See [attached timing](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/docs/core/ovstage_integration.rst).
Publish each playback update before rendering it.

See [OVStage + OVRTX integration](../ovstage-ovrtx-integration/README.md) for
attachment and frame scheduling, and [stage management](../stage-management/README.md)
for scene-switch UX and generation lifetime.
