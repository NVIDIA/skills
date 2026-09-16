# ovstage Data Plane

Use this reference for hierarchy queries, property inspection, live transforms,
camera updates, render-product settings, material effects, cloning, and
CPU/GPU tensor exchange.

## Commands, queries, and DTOs

Keep a narrow runtime adapter. Commands receive canonical USD paths and values,
resolve them through the active stage path dictionary, assign an ordinal, and
return only an acknowledgement/generation. Queries return copied DTOs containing
a canonical path, value/schema summary, stage generation, and observed ordinal.

Never make renderer attribute APIs the source of scene truth. Use ovstage for
prim discovery, hierarchy, scalar/array reads, writes, maps, and clones. Keep
renderer APIs for renderer outputs, pick queues, and renderer-owned selection
configuration.

Direct `renderer.write_attribute()`, `bind_attribute()`, `map_attribute()`,
`query_prims()`, and `read_attribute()` are standalone compatibility APIs in
ovrtx. Do not call them to mutate transforms, material/effect values, or
stage-data viewer state while ovrtx is attached to an OVStage. Renderer-owned
picking and selection-outline configuration still belongs to ovrtx.

## Transform and interaction writes

Label every transform as authored, composed-local, or world intent. Convert
world-space interaction intent through the parent before writing runtime state.
For persistent product edits, use the app's non-destructive session/authoring
policy; do not silently overwrite the user asset.

Every render-visible write follows the runtime publication protocol: write at
`N`, wait, advance the floor, then render `N`. Rebuild cached material/prim maps
on a stage-generation change.

For `omni:xform`, use row-major `float64` matrices with translation in row
`[3][0..2]`. OVStage DLPack storage is one 16-lane element per prim
(`shape=[N]`, lanes `16`, `AttributeSemantic.MATRIX`), while older standalone
ovrtx Python compatibility snippets may show `(N, 4, 4)` NumPy matrices.

Use `PrimMode.UPSERT` for app-owned runtime attributes unless the feature
requires insert-only behavior. Use session/composite USD for persistent
non-destructive authored state; never silently author transform, material, or
viewer-setting state back into the user's USD asset.

## Tensor lifetime

Use the pinned runtime's DLPack contracts. Source buffers must survive until the
operation completes. Copy values that leave the runtime boundary; UI and WebRTC
messages never own a live stage tensor or mapped view.
