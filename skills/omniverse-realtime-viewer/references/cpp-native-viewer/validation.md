# C++ Validation

Validate the native executable through its actual selected native window. Compilation and
CPU checks establish API/math correctness; they do not prove rendered frames,
input routing, loader compatibility, or cleanup. Follow `references/validation.md` for the
shared evidence expectations.

## Runtime Invariants

- Use paired public OVRTX/OVStage packages with one loader model. Register RTX
  schemas before OVStage/OpenUSD initialization and respect the selected ABI or
  isolation boundary for optional composition tools.
- One runtime owner serializes population, writes, waits, publication, renderer
  updates, picks, steps and output mapping. The selected toolkit and presentation
  resources stay on their required UI thread, exchanging bounded commands and owned data.
- Wait writes, seal the ordinal, wait `ovrtx_update_from_stage()`, apply dirty
  native state, enqueue picking, then `ovrtx_step_with_stage()` at that ordinal.
- Map, copy, unmap and destroy results on every exit path. No mapped pointer or
  SDK handle crosses into the UI. `LdrColor` is uploaded as RGBA.
- Match output identity using the full authored RenderVar path; native picks use
  `OVRTX_RENDER_VAR_PICK_HIT`. Expose only AOVs with real full-resolution data.
- Use OVStage for live camera, transform and material writes; renderer-native
  selection groups and picks remain OVRTX operations.

## Native Acceptance

- [ ] Configure/build from the paired package helpers and launch on a real
  display with an NVIDIA GPU; capture the actual first presented frame.
- [ ] Exercise orbit, pan, wheel, fit and click picking with real pointer input.
  Check normalized coordinates at letterbox edges and after resize.
- [ ] Compare a known selected part's readable type/path and one numeric value
  against its USD source or an independent USD read. A Cube/Sphere must not
  appear as an opaque ID; distinguish correctly decoded metadata from a panel
  that merely updates. Use `references/stage-attribute-reads/README.md` for the owning
  dictionary and property-specific mapping.
- [ ] Check selected outline, tree and property state agree; deselection clears
  the previous outline. If variant/visibility recomposition retains selection,
  its outline must return without another click.
- [ ] For editing features, verify fractional values, a rotated/nonuniform live
  parent, reset-stack behavior, and parent edits after child edits. Check stage
  readback and rendered geometry, not only the control's displayed value.
- [ ] For gizmos/shortcuts, verify text input and modals consume keys, gizmos
  precede camera input, and releasing a drag outside the image ends the drag.
- [ ] Open another scene, rapidly replace it, and try an invalid file. Verify
  latest-request activation, cleared stale state and usable error recovery.
- [ ] Preserve source USD files; optional effects restore scene-owned baselines.
- [ ] Close through the window manager while rendering/loading, reopen, and
  verify owned renderer, stage, preparation jobs and toolkit presentation resources are released.

Report observed failures with a focused reproduction against the public SDK
contract. Keep API defects, environment issues and application omissions distinct
when deciding what to change in an integration recipe.

See also: `ovstage-ovrtx-integration`, `stage-management`, `viewer-input-routing`,
`native-picking-selection`, `selection-feedback`, and `prim-transform-safety`.
