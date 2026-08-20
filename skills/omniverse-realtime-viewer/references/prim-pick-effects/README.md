# Prim Pick Effects

## Triggers

Use this skill for picked prim effect, on pick write attribute, `inputs:Fader`,
EffectLayer, toggle visibility, custom MDL parameter, prim-to-material map,
selection glow attribute, or reversible pick-driven runtime state.

Use this when picking a prim should manipulate a USD/runtime attribute on that
prim or on a related material/shader prim. This skill is additive to native
selection outlines; do not replace outline selection with material effects.

For new OVStage-based viewers, pick effects write through the OVStage data plane
and the attached ovrtx renderer consumes the committed update. Direct
`Renderer.write_attribute()` or persistent OVRTX bindings for material,
visibility, or effect attributes are legacy/pre-OVStage patterns.

## Workflow

1. Resolve picked prim paths through `object-selection` or native OVRTX picking.
2. Update selection outlines through `selection-feedback`.
3. Map picked prims to effect targets when needed, such as material shader prims.
4. Compute desired effect state from the complete selected set.
5. Enqueue OVStage runtime writes for effect, material, visibility, or
   viewer-owned state changes.
6. Publish the write at a monotonic ordinal and let the renderer consume it in
   the normal render loop.
7. Clear or reset effect attributes on deselect and scene switch.

```text
pick hit
  -> selected path set
  -> native OVRTX outline groups
  -> effect target map
  -> OVStage attribute write at ordinal N
  -> renderer update_from_stage/step consumes N
```

## Runtime Write Pattern

Keep a small app-owned adapter around the pinned OVStage API. The exact method
names can differ by generated app; the contract is path validation,
stage-generation checking, monotonic ordinal publication, and copied tensor or
token values.

```python
def write_runtime_attr(runtime, paths, attr_name, values, *, prim_mode="existing"):
    if not paths:
        return None
    ordinal = runtime.next_ordinal()
    op = runtime.write_attribute(
        ordinal=ordinal,
        prim_paths=paths,
        attribute_name=attr_name,
        values=values,
        prim_mode=prim_mode,
    )
    op.wait()
    runtime.advance_write_floor(ordinal).wait()
    return ordinal
```

Use `prim_mode="existing"` when the composite/session layer already authored
the attribute. Use `prim_mode="create_new"` only for deliberate viewer-owned
runtime attributes and load-time resets. Never silently create material inputs
with invented names.

For repeated writes to the same target set, cache target path lists and value
buffers, but still publish writes through OVStage. Do not optimize by keeping
direct OVRTX material/effect bindings in new viewer code.

## Prim To Material Mapping

A picked mesh often does not own the effect attribute directly. For
material-driven effects, build a map from renderable prim path to material or
shader target:

```python
prim_to_effect_layer = {
    "/World/Mesh/Tray": "/World/Looks/Steel_Stainless/EffectLayer",
}
```

Prefer OVStage queries for candidate prims and attribute presence. Keep the
`stage-hierarchy` pxr fallback for relationship traversal such as inherited
material bindings until the runtime query path exposes those relationship values
with the needed fidelity.

## Shared-Material Awareness

Multiple prims can share one material and therefore one effect target. Never
turn off a target just because one prim was deselected; recompute active targets
from all currently selected prims.

```python
def active_effect_targets(selected_prims: set[str], prim_to_target: dict[str, str]) -> set[str]:
    return {
        target
        for prim in selected_prims
        for target in [prim_to_target.get(prim)]
        if target
    }

def update_pick_effects(selected_prims: set[str]) -> None:
    global active_targets
    next_targets = active_effect_targets(selected_prims, prim_to_effect_layer)

    for path in sorted(next_targets - active_targets):
        write_fader(path, 1.0)
    for path in sorted(active_targets - next_targets):
        write_fader(path, 0.0)

    active_targets = next_targets
```

This same rule applies to custom material parameters, display color ramps, and
any shared shader attribute.

## EffectLayer Fader Example

Some stages use EffectLayer shader prims with `float inputs:Fader = 0`
overrides in the composite/session layer. When the active stage exposes that
pattern and the user wants material-driven pick effects, a concrete target shape
is:

```text
/World/.../Looks/<MaterialName>/EffectLayer.inputs:Fader
```

Runtime toggle:

```python
import numpy as np

def write_fader(effect_layer_path: str, value: float) -> int | None:
    return write_runtime_attr(
        runtime,
        [effect_layer_path],
        "inputs:Fader",
        np.asarray([value], dtype=np.float32),
        prim_mode="existing",
    )
```

Load-time reset:

```python
layers = sorted(set(prim_to_effect_layer.values()))
write_runtime_attr(
    runtime,
    layers,
    "inputs:Fader",
    np.zeros((len(layers),), dtype=np.float32),
    prim_mode="create_new",
)
```

This glow is a pick effect, not the baseline selection signal. Keep native
selection outlines enabled so arbitrary scenes still show precise
selected-object boundaries when no EffectLayer material exists.

## Visibility Toggle Example

USD visibility is token-like. Preserve the previous value if the effect is
temporary, and write the new token through OVStage:

```python
write_runtime_attr(
    runtime,
    [picked_path],
    "visibility",
    ["invisible"],
    prim_mode="existing",
)

write_runtime_attr(
    runtime,
    [picked_path],
    "visibility",
    ["inherited"],
    prim_mode="existing",
)
```

Use this for explicit hide/show commands, not hover highlighting.

## Custom MDL Parameter Example

For app-authored materials with known shader inputs, write the input attribute
through OVStage:

```python
write_runtime_attr(
    runtime,
    [shader_path],
    "inputs:HoverAmount",
    np.asarray([0.65], dtype=np.float32),
    prim_mode="existing",
)
```

Only expose controls for attributes that exist in the active stage or are
deliberately authored by the viewer. Do not invent renderer-internal attribute
names.

## Scene Lifecycle

- Rebuild prim-to-material/effect maps after every scene load, reload, variant
  change, or material-map invalidation.
- Reset app-owned effect attributes to neutral values on stage load.
- Clear active target state on selection clear and scene switch.
- Serialize writes through the OVStage runtime owner; do not write while scene
  loading, reset, or generation replacement is active.
- Keep effect state separate from selection outline state.

## Focused Proof

For an app that includes pick effects, capture one compact proof with the picked
path, resolved effect target path, previous value, new value, OVStage write
ordinal, renderer-consumed ordinal, and reset/clear behavior. Do not add broad
pipeline artifacts unless the product explicitly requires them.

See also: `object-selection`, `selection-feedback`, `stage-queries`,
`stage-hierarchy`, `stage-attribute-reads`, `ovstage-data-plane`,
`ovstage-ovrtx-integration`.
