# Pitfall: `class_filter` is not `component_types`

Reference for `skills/cdf-simulation`. Two taxonomies that look
interchangeable and are not; passing the wrong one produces zero crop files
while reporting success.

`job.scene.select_components(types=[...])`, `job.operations.add_tombstone(component_types=[...])`
(sidecar boards only -- see `payload-contract.md`, Row 4 "scoping"),
and `job.operations.add_excess(component_types=[...])` all key off the
literal component-type string from the board metadata (e.g. `"PASSIVE_SMD_0805"`).
`job.postprocess.crop(class_filter=[...])` keys off the **semantic class**
that `job.annotations.label_components(...)` derives for that component
(e.g. `"capacitor"` for the same part) -- a different, not-always-obvious
mapping this skill cannot predict for an arbitrary board without inspecting
the render's own `semantic_segmentation_labels_*.json` output first. Default
to `class_filter=None` (crop every labelled component already narrowed by
`select_components`) rather than guessing the derived class name. Only pass
an explicit `class_filter` when the prompt or board metadata names the
semantic class directly (not the component type).

## The Row 4/5 output contract

There is one job definition for each of rows 4 and 5, given by the
`SKILL.md` Row 4 and Row 5 templates. A record that leaves no non-empty file
under `cropped/` has failed.

**Row 4 — Structural defects (SMT):** scene/appearance as Row 3, then
exactly one of `add_shift(...)`, `add_tombstone(...)`, `add_sideflip(...)`
per the coverage rules in `fixed-defaults.md` (a non-empty selection
always yields at least one defect), `job.annotations.label_components(...)` +
`job.annotations.enable_operation_labels(namespace="defect")`, outputs
rgb + bounding_box_2d_tight + semantic_segmentation (colorized), then the
same crop block as Row 3's component-level case with `class_filter=None`
(structural defects are always reported as per-component crops, not
board-level).

**Row 5 — Texture defects (SMT solder):** scene/appearance as Row 3, then
`job.operations.solder_joints(solver="surface_evolver", source=...)`
followed by the mapped `add_*` solder-defect call(s) from the mapping in
`fixed-defaults.md` (skip/report any unsupported `bridge` portion per that
section), same annotations + outputs + crop block as Row 4.
