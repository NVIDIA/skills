# Reference flows

Reference for `skills/cdf-simulation`. The catalog rows in `SKILL.md`
describe the catalog's prompt shapes. The flows below describe complete
inspection jobs, each phrased as one prompt, with every unstated value fixed
here. A prompt that matches a flow compiles to the same `JobOrderPlan` as
the equivalent YAML config would, so the same capability is reachable from a
prompt and from a config.

Routing: **a flow wins over a catalog row whenever the prompt carries a flow
marker** -- a scan grid ("5x4 scan", "stitched"), a named refdes target
("shift on C33", "insufficient solder on J17"), "component-focus" /
"pin-focus" crops, or "missing-component defect on <refdes>". A prompt with
none of those markers is a catalog row and follows `SKILL.md` as before. Both
kinds resolve attachment, root and component types the same way
(`fixed-defaults.md`, "Board identity and paths").

## Explicit prompt arguments the flows understand

Beyond the two required inputs (`Attachment:`, `Output to`), a flow
prompt may carry these arguments. Each is optional; an absent one takes the
default in its row. Nothing else in the prompt is a parameter -- a number the
prompt does not state is never invented, it comes from the tables below.

| Argument | Form in the prompt | Maps to | Default |
|---|---|---|---|
| Scan grid | `a 5x4 scan` | `scan_grid(x_count=5, y_count=4)` and `limits(max_images=5*4)` | `1x1` (only the golden flow has a grid) |
| Resolution | `at 2000x2000`, `crops at 512x512` | `renderer(resolution=(2000, 2000))` | golden 1000x1000; focus flows 512x512 |
| Board side | `of the top side`, `from the bottom` | `camera(side=...)` on every pass, and for the golden flow the hole-plane block | top |
| Seed | `Seed 7.` | `CDFJob(random_seed=7)` | 0 |
| Named target | `shift on C33`, `insufficient solder on J17`, `missing-component defect on LS1` | `ComponentFilter(refdes=("C33",))` on that operation; the union of all named refdes is the scene selection | none -- every defect in a flow prompt names its refdes |
| Pin | `on J1 (pin SH)` | the `(refdes, pin)` pair in `roi_selection(pins=...)` | pin `"1"` |
| Solver | `with Surface Evolver solder` | `solder_joints(solver="surface_evolver")` | `surface_evolver` in every flow (the solder-defect adders require it) |

The output root is used twice: `compile_orders(output_dir=<root>,
serialization_path=<root>/stage.usda)`. Every flow serializes the composed
stage; the config side does the same through `execution.serialization_path`.

## Shared conventions (every flow)

These are the spellings the YAML interpreter produces. Use them verbatim;
each one spelled differently produces a different plan.

```python
import pcb_aoi  # noqa: F401  -- registers the "pcb-aoi" domain
from cdf import ComponentFilter, CDFJob

job = CDFJob(domain="pcb-aoi", random_seed=SEED)
job.scene.configure(usd_path=ATTACHMENT, root_path=BOARD_ROOT)      # root: the attachment's default prim
# job.scene.instances(prim_paths=INSTANCES)                          # panels only -- see "Panels"
job.scene.select_components(source="usd", component_filter=ComponentFilter(refdes=TARGETS))  # focus flows only
job.appearance.lighting(
    mode="white_light",
    preserve_scene_light_color=False,
    exposure_range=(0.0, 0.0),
    cone_angle_range=(120.0, 120.0),
    cone_softness_range=(1.0, 1.0),
    dome={"intensity": (1000.0, 1000.0), "visible": True},
)
job.appearance.components(vantablack=False)
job.operations.solder_joints(
    solver="surface_evolver",
    source="usd",
    component_filter=ComponentFilter(refdes=TARGETS),                # omit on the golden flow (all components)
    solver_options={"workers": 0, "solder_material_prim_path": SOLDER_MATERIAL},
)
...
job.annotations.label_components(source="usd", classes={TARGET_CLASS: list(TARGETS)})  # classes= only on focus flows
job.annotations.label_geometry(solder="solder", terminals="pin", pads="pad", ...)       # see each flow
job.annotations.enable_operation_labels(namespace="defect")                              # defect flows only

with job.capture.render_pass(PASS_NAME) as rp:
    rp.output_start(0)
    rp.renderer(mode="pathtracing", resolution=RESOLUTION, spp=1, total_spp=32, max_bounces=4, realtime_subframes=1)
    rp.camera(side=SIDE, rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0})
    rp.scan_grid(x_count=GX, y_count=GY)
    rp.limits(triggers=1, max_images=MAX_IMAGES)
    ...
plan = job.compile_orders(output_dir=OUTPUT_ROOT, serialization_path=OUTPUT_ROOT + "/stage.usda")
```

- `source="usd"` is written explicitly on `select_components`, `solder_joints`,
  `label_components` and as `eligibility_source="usd"` on `add_tombstone`; a
  bare call leaves `None` in the plan and the config side writes `"usd"`.
- `realtime_subframes=1` on every `renderer(...)`; `output_start(0)` and an
  explicit `limits(...)` on every pass.
- The camera rotation is the three-key dict above; `-90` about z is the AOI
  convention on all boards.
- Structural-operation magnitudes are the flow values below (they replace the
  catalog's `translate_range=0.2 / rotate_z_range=15.0` when a flow matches).
- **No `postprocess.crop` in any flow.** The focus flows (C, S, M, T) get their
  crops from the ROI capture pass (`roi_selection` + `limits`), not from the
  catalog rows' `postprocess.crop(...)` block; the golden flow crops nothing.
  A `postprocess.crop` order in a flow plan is a mistake.
- **Solder is always `solver="surface_evolver"`** in a flow, including the
  golden flow: the catalog Row 1 rule "golden board carries procedural solder"
  does not apply once a flow marker is present. Declaring the solver only
  checks that `SOLDER_EVOLVER_BIN` names an existing file; the binary is used
  by the executing container, not by the host that builds the plan. Never
  downgrade to `procedural` because the Evolver is absent where the plan is
  written -- that changes the job, and the run summary would be describing a
  different dataset than the one requested.
- `SOLDER_MATERIAL` is the `SolderPaste` material of the attachment:
  `<root>/Looks/SolderPaste` on a pcb-builder board (`/Board/Looks/SolderPaste`)
  and `/World/Looks/SolderPaste` on a panel export. Find it by searching the
  attachment (and its sublayers) for a `Material "SolderPaste"` prim; never
  assume the pcb-builder path on a board that does not have it.

### Panels

If the attachment composes several placements of one board as USD instances
(a panel export: `<root>/StepRepeats/SR_*` prims, each an
instance of the same prototype), register **all** of them, in stage order,
right after `scene.configure(...)`:

```python
job.scene.instances(prim_paths=("/World/<panel>_Carrier/StepRepeats/SR_0_0",
                                "/World/<panel>_Carrier/StepRepeats/SR_0_1"))
```

Detect this by opening the attachment: `[p.GetPath() for p in stage.Traverse() if p.IsInstance()]`
under the root, or the `instanceable = true` prims in the `.usda` text. A
partial list is rejected at execution ("duplicate component identity"), and
omitting the call on a panel is rejected too.

**The panel's `root_path` is the carrier, not the stage default prim.** On a
panel the default prim is `/World`, a container that also holds `/World/Looks`;
the board root the pipeline expects is the parent of the `StepRepeats` scope,
i.e. the Xform the instances live under (`/World/<panel>_Carrier`
on a panel export). Take the common parent of the instance prims' parent
scope. `/World` compiles but is not the board root. Every ROI target is then
rendered once per instance, so `max_images` for a focus flow on a panel is
`len(targets) * len(instances)` (six pins on a two-board panel = 12). A
scan-grid pass is different: it frames the whole panel, so its budget stays
`GX * GY` (a 5x4 golden scan of a two-board panel renders 20 tiles per pass, not 40).

## Flow G -- Golden board scan with stitched RGB and class mask

Prompt shape: `Generate a golden board dataset from the attached USD: a <X>x<Y>
scan of the <top|bottom> side at <W>x<H>, stitched RGB and stitched class mask.`
Applies to a single board or a panel.

| Fixed | Value |
|---|---|
| Components | none selected (`select_components` is **not** called: the golden flow labels the whole board) |
| Solder | `solder_joints(solver="surface_evolver", source="usd", solver_options=...)`, no filter, no defect -- **not** `procedural` (that is the catalog Row 1 rule, not this flow's) |
| Geometry labels | `label_geometry(solder="solder", terminals="pin", pads="pad", barrels="pad")` |
| Passes | `full_board` (rgb) and `full_board_roi` (colourized `semantic_segmentation`, `semantic_types=("class",)`, `visibility(hide=("solder",))`), both with `frame_padding=4, rename_to_grid_index=True` |
| Stitch | `stitch(source_pass="full_board", image_type="rgb", output_subdir="stitched", output_filename="full_board_rgb.png", blend="none")` and `stitch(source_pass="full_board_roi", image_type="semantic_segmentation", output_subdir="stitched", output_filename="full_board_semantic.png", blend="labels", color_map={"pin": (255, 0, 0, 255), "pad": (0, 255, 0, 255), "solder": (255, 255, 0, 255)})` |
| Bottom side | three additions, all required: `job.operations.hole_plane(side="bottom")` (declared next to `solder_joints`), `visibility(hide=("hole",), show=())` on `full_board`, `visibility(hide=("solder",), show=("hole",))` on `full_board_roi`, and the mask stitch's `color_map` gains `"hole": (0, 0, 255, 255)` (the hole plane is a labelled class on the bottom side) |
| No `label_components(classes=...)`, no `enable_operation_labels` | |

```python
job.annotations.label_components(source="usd")
job.annotations.label_geometry(solder="solder", terminals="pin", pads="pad", barrels="pad")
with job.capture.render_pass("full_board") as rp:
    rp.output_start(0)
    rp.renderer(mode="pathtracing", resolution=RESOLUTION, spp=1, total_spp=32, max_bounces=4, realtime_subframes=1)
    rp.camera(side=SIDE, rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0})
    rp.scan_grid(x_count=GX, y_count=GY)
    # bottom side only: rp.visibility(hide=("hole",), show=())
    rp.limits(triggers=1, max_images=GX * GY)
    rp.outputs(rgb=True, frame_padding=4, rename_to_grid_index=True)
with job.capture.render_pass("full_board_roi") as rp:
    rp.output_start(0)
    rp.renderer(...same...)
    rp.camera(...same...)
    rp.scan_grid(x_count=GX, y_count=GY)
    rp.visibility(hide=("solder",), show=())          # bottom side: show=("hole",)
    rp.limits(triggers=1, max_images=GX * GY)
    rp.outputs(rgb=False, semantic_segmentation=True, semantic_types=("class",),
               colorize_semantic_segmentation=True, frame_padding=4, rename_to_grid_index=True)
job.postprocess.stitch(...rgb...)
job.postprocess.stitch(...semantic, blend="labels", color_map=...)
```

Bottom side, the complete set of differences against the top-side build:

```python
job.operations.hole_plane(side="bottom")                     # after solder_joints(...)
# full_board pass:      rp.visibility(hide=("hole",), show=())
# full_board_roi pass:  rp.visibility(hide=("solder",), show=("hole",))
# camera(side="bottom", ...) on both passes
job.postprocess.stitch(..., blend="labels",
    color_map={"pin": (255, 0, 0, 255), "pad": (0, 255, 0, 255),
               "hole": (0, 0, 255, 255), "solder": (255, 255, 0, 255)})
```

## Flow C -- SMT component defects on named parts, component-focus crops

Prompt shape: `Apply component defects on the attached USD: shift on <R1>,
tombstone on <R2>, sideflip on <R3>, reverse polarity on <R4>, with Surface
Evolver solder on those parts; component-focus crops at 512x512.` Any subset
of the four defects is allowed; each names exactly one refdes.

| Operation | Fixed parameters |
|---|---|
| `add_shift` | `count=1, translate_range=0.5, rotate_z_range=5.0` -- **no `variant_index`** (uniform mode: the slide is visible, the rotation small) |
| `add_tombstone` | `count=1, angle_range=(75.0, 85.0), auto_filter=True, eligibility_source="usd"` |
| `add_sideflip` | `count=1, angle_range=(70.0, 85.0), variant_index=0, rotate_z_jitter_deg=10.0, slide_range_mm=0.25` |
| `add_reverse_polarity` | `count=1` |
| all four | `component_filter=ComponentFilter(refdes=(<its refdes>,))` |
| Selection / solder | `select_components(source="usd", component_filter=ComponentFilter(refdes=TARGETS))`; `solder_joints(..., component_filter=ComponentFilter(refdes=TARGETS))` |
| Labels | `label_components(source="usd", classes={"smt_component_defect_target": list(TARGETS)})`, `label_geometry(solder="solder", terminals="pin", pads="pad", component_filter=ComponentFilter(refdes=TARGETS))`, `enable_operation_labels(namespace="defect")` |
| Pass `component_focus` | top, 512x512, 1x1, `roi_selection(defects=(<the defect names, in the order shift, tombstone, sideflip, reverse_polarity>), padding=0.75)`, `limits(triggers=1, max_images=len(TARGETS))`, `outputs(rgb=True, semantic_segmentation=True, semantic_types=("defect",), colorize_semantic_segmentation=True, frame_padding=4)` |
| Never | `postprocess.crop(...)` (the ROI pass is the crop); a camera without `rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0}`; a `types=` selector instead of `ComponentFilter(refdes=...)` |

`TARGETS` is the refdes tuple in the order **tombstone, shift, sideflip,
reverse_polarity** targets (for example `FB3, C33, L2, D2`); the order does
not change what is rendered.

## Flow S -- SMT solder defects on named parts, component-focus crops

Prompt shape: `Apply solder defects on the attached USD: insufficient solder on
<R1>, excess solder on <R2>, interrupted solder on <R3>, missing solder on
<R4>; component-focus crops at 512x512.`

| Operation | Fixed parameters (`ratio=1.0` on all, `component_filter` = its refdes) |
|---|---|
| `add_insufficient` | `volume_ratio=0.4` |
| `add_excess` | `volume_ratio=2.0` |
| `add_interrupted` | `pad_exposure=0.4` |
| `add_missing_solder` | -- |
| Labels | class `smt_solder_defect_target`; geometry `solder/pin/pad` with the target filter; operation labels `defect` |
| Pass `component_focus` | top, 512x512, 1x1, `roi_selection(component_filter=ComponentFilter(refdes=TARGETS), padding=0.75)`, `max_images=len(TARGETS)`, outputs as Flow C |
| Never | `postprocess.crop(...)` (the ROI pass is the crop); a camera without `rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0}`; a `types=` selector instead of `ComponentFilter(refdes=...)` |

`solder_joints(...)` is declared before the adders; the compiler fixes the
application order anyway (solder generation, then defects), so the plan does
not depend on the declaration order.

## Flow M -- Missing component

Prompt shape: `Generate a missing-component defect on <R> of the attached USD,
with Surface Evolver solder; component-focus crop at 512x512.`

| Fixed | Value |
|---|---|
| Operation | `add_missing(ratio=1.0, component_filter=ComponentFilter(refdes=(R,)))` -- exclusive, no other structural operation |
| Labels | class `smt_missing_component_target`; geometry `solder/pin/pad` with the target filter; operation labels `defect` |
| Pass `component_focus` | top, 512x512, 1x1, `roi_selection(defects=("missing",), padding=0.75)`, `max_images=1`, `outputs(rgb=True, semantic_segmentation=True, semantic_types=("class", "defect"), colorize_semantic_segmentation=True, frame_padding=4)` |
| Never | `postprocess.crop(...)` (the ROI pass is the crop); a camera without `rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0}`; a `types=` selector instead of `ComponentFilter(refdes=...)` |

The missing pipeline writes a paired output per target (a present and an
absent frame); `max_images=1` still names one target.

## Flow T -- Through-hole solder defects on named pins, pin-focus crops

Prompt shape: `Apply through-hole solder defects on the attached USD:
insufficient solder on <R1>, bright solder on <R2>, excess solder on <R3>,
interrupted solder on <R4> (pin <P>), missing solder on <R5>, dry contact on
<R6>; pin-focus crops from the bottom at 512x512.` Applies to a single
board or a panel.

| Operation | Fixed parameters (`ratio=1.0` on all, `component_filter` = its refdes) |
|---|---|
| `add_insufficient` | `lead_climb=0.35, fullness=0.55` |
| `add_bright` | `lead_climb=0.75, fullness=0.45` |
| `add_excess` | `lead_climb=1.2, fullness=1.15` |
| `add_interrupted` | `pad_exposure=0.4` |
| `add_missing_solder` | -- |
| `add_dry_contact` | `gap_fraction=0.4, lead_climb=0.8, fullness=0.85` |
| Labels | class `tht_solder_defect_target`; `label_geometry(solder="solder", terminals="pin", pads="pad", barrels="pad", component_filter=ComponentFilter(refdes=TARGETS))`; operation labels `defect` |
| Pass `joint_focus` | **bottom**, 512x512, 1x1, `roi_selection(pins=((R1, "1"), ...), padding=0.75)` -- one pair per target, pin `"1"` unless the prompt says `(pin SH)`; `max_images=len(pins) * len(instances or 1)`; outputs `rgb + semantic_segmentation`, `semantic_types=("defect",)`, colourized, `frame_padding=4` |
| Never | `postprocess.crop(...)` (the ROI pass is the crop); a camera without `rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0}`; a `types=` selector instead of `ComponentFilter(refdes=...)` |

The ROI pin list follows the prompt's order of targets; the order does not
change what is rendered. Name
each pin once: on a two-board panel the six pins are rendered on both boards
by the instance registration, so the list stays six entries and only
`max_images` doubles to 12.
