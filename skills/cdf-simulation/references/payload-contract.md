# Payload contract

Reference for `skills/cdf-simulation`. This is the component-owner
contract: the parameters this component supports for each catalog row, how
each one is validated, and the exact shape this skill returns to its caller.
The skill never executes; everything below describes what it *emits* and
what the executing side must supply.

## What this skill emits

One matched prompt produces one **variant**, and one variant is one hand-off
record. A prompt that implies several variants ("shift, tombstone and
sideflip") produces several records, each marked independent -- never a loop
of `render()` calls inside one script (a second `render()` in one process
segfaults the Kit subprocess).

| Field | Type | Meaning |
|---|---|---|
| `row` | `1..5` | The catalog row matched in Step 1 |
| `build` | Python source | `import pcb_aoi` + `from cdf import CDFJob` + a `build_job() -> CDFJob` following the row's template in `SKILL.md` |
| `plan` | `JobOrderPlan`, serialized with `JobOrderCodec.dumps(plan)` | Output of `job.compile_orders(output_dir=...)` -- the transport-safe truth the Isaac child executes. `JobOrderPlan` holds `mappingproxy` and tuple values, so a bare `plan.model_dump()` is **not** JSON-serializable; the codec (`from cdf import JobOrderCodec`) is the representation the executor's own subprocess boundary uses, and `JobOrderCodec.loads(payload)` on the receiving side restores an equal plan |
| `resolved_inputs` | mapping | Every input below with its value **and its source** (`state file`, `prompt argument`, `agent context`, `prompt text`, `fixed default`) |
| `independent_launch` | `true` | Always; tells the executor this record needs its own process |
| `refusals` | list | Any clause of the prompt not served, with the response text from `SKILL.md` "When to refuse" (e.g. a `bridge` clause dropped from a Row 5 request) |

The record contains no rendered artifacts and makes no claim about them.

## What the caller supplies

| Caller-supplied | Required | Used for |
|---|---|---|
| Attachment path (board USD) | **yes -- no fallback.** Ask for a missing path when interaction is possible; stop non-interactive automation (`SKILL.md`, refusal case 3). Never substitute a sample board | `scene.configure(usd_path=...)` |
| Matching `<stem>.components.json` component sidecar | optional legacy input for standalone boards that rely on it, or an explicitly selected legacy source; never for instanced panels | Prefer embedded metadata with `source="usd"`. For a selected legacy source, call `scene.attach_sidecar(...)` before using its job-local name consistently in selection, solder, and annotations. Do not attach a file solely because it is nearby |
| Output root | **yes -- no fallback.** Ask for a missing path when interaction is possible; stop non-interactive automation. Never synthesize a directory | `compile_orders(output_dir=...)` and the capture pass `output_dir` |
| Fan-out | yes | One process per emitted record |
| Execution environment | yes | For Row 5: `manifold3d` **and** the `evolver` binary (or `SOLDER_EVOLVER_BIN`); their absence fails at `solder_joints(...)` declaration time, before compile |

## Supported parameters, by row

Reference flows (`reference-flows.md`) add explicit prompt arguments on top
of the rows below: seed, scan grid, board side, resolution, named refdes
targets and pin numbers, plus panel instance registration read from the
attachment. Their fixed values are listed per flow in that file; the
record's `resolved_inputs` names each one with source `prompt argument` or
`fixed default (flow <letter>)`.

Values come from the precedence order in `SKILL.md` "Inputs"; anything unset
takes the documented default from `fixed-defaults.md`. "Required" means the
API has no default and compile or declaration fails without it.

### Row 1 -- Golden board (THT)

| Parameter | Required | Default / rule |
|---|---|---|
| `usd_path`, `root_path` | yes | `usd_path` is the required attachment. Resolve `root_path` from the supplied stage by default, or the selected legacy sidecar for a sidecar-backed standalone board; for panels use the confirmed carrier root. An unreadable or unresolved attachment is refusal case 3. Same rule as `fixed-defaults.md`, "Board identity and paths" |
| `select_components(source=)` | yes | `"usd"` for embedded metadata by default, or the job-local name of the selected legacy component sidecar; use explicit `types=` resolved on that source |
| `solder_joints(solver=)` | yes | `"procedural"` -- the catalog's golden board; **no** `add_*` solder call may follow |
| `lighting` | no | `white_light`, exposure `(0,0)`, cone `(120,120)`/`(1,1)`, dome intensity `(1000,1000)` visible |
| `camera(side=, rotation_range=)` | yes | `"bottom"`, `{"z": -90}` |
| `renderer(...)` | yes | pathtracing, `(1920,1080)`, spp 1, total_spp 32, max_bounces 4 |
| `outputs` | no | rgb + bounding_box_2d_tight + semantic_segmentation (colorized) |
| `postprocess.crop` | forbidden | no crop chain on this row |

### Row 2 -- Variation type (THT pin geometry)

| Parameter | Required | Default / rule |
|---|---|---|
| `usd_path`, `root_path` | yes | a board authoring `pcba:mesh:role` lead meshes; root read from the attachment as in Row 1 |
| `pin_variation(components=)` | yes (one selector) | the refdes the prompt names, confirmed on the attachment; with none named, the attachment's through-hole parts by refdes; `package_types=` only when the prompt names a package type; no through-hole part on the attachment → stop and report it |
| exactly one of `tilt_range`, `offset_range`, `height_variation`, `cut_angle` | yes | `(5,15)` deg, `(0.1,0.4)` mm, `(0.3,1.0)` mm, `(5,15)` deg |
| `select_components` / `label_components` | forbidden | `pin_variation` targets leads directly |
| camera / renderer / outputs | as Row 1 | rgb only by default |

Validation: a board without `pkg_lead` meshes fails at execution with
"pin_variation found no through-hole leads"; the skill reports insufficient
THT metadata rather than retrying with invented parameters.

### Row 3 -- Good images (SMT)

| Parameter | Required | Default / rule |
|---|---|---|
| mode | yes | board-level (`"N good PCB images"`) or component-level (`"good images of <component>, max M"`) |
| `scan_grid(x_count=, y_count=)` | board-level | most-square factorisation with `x*y >= N`, reported (20 -> 5x4) |
| `select_components(types=)` | component-level | the named component type, which must resolve on the attachment (`fixed-defaults.md`, "Named components must resolve on the attachment"; otherwise refusal case 4) |
| `postprocess.crop(...)` | component-level | `method="bbox_tight"`, `types=["rgb","semantic_segmentation","component_instance"]`, **`class_filter=None`**, `offset=10`, `xform_depth=7` |
| `capture.limits(max_images=)` | component-level | the requested max |
| operations | forbidden | no defects on this row |

Validation: `class_filter` takes the derived semantic class, never a
component-type string -- the latter compiles, reports `"error": ""`, and
writes zero crop files.

### Row 4 -- Structural defects (SMT)

| Parameter | Required | Default / rule |
|---|---|---|
| defect type(s) | yes | one or more of `shift`, `tombstone`, `sideflip` |
| `ratio` | no | `1.0` for "full board coverage", else `0.2`. The executing side selects `max(1, int(pool * ratio))`, so a non-empty selection always yields at least one defect; `count=1` is the explicit spelling for a one-component selection. A record whose selection is empty is not emitted |
| `add_shift(translate_range=, rotate_z_range=)` | yes when shift | `0.2` mm, `15` deg |
| `add_tombstone(angle_range=)` | yes when tombstone | `(70,90)` deg |
| `add_sideflip(angle_range=)` | yes when sideflip | `(70,90)` deg |
| scoping | rule | the narrowed `select_components(types=)` is the scope of every pose operation. **Embedded-metadata board** (`pcba:*` schema, no sidecar): `add_tombstone(..., eligibility_source="usd")` and **no** operation-level `component_types` -- supplying it applies the operation to nothing. **Sidecar board**: `component_types=` on `add_tombstone` works and may accompany the narrowed selection; it never replaces it |
| `enable_operation_labels(namespace=)` | yes | `"defect"`, declared after `label_components` -- the defect class must reach the segmentation output |
| `postprocess.crop(...)` | yes | the Row 3 component-level block: `method="bbox_tight"`, `types=["rgb","semantic_segmentation","component_instance"]`, **`class_filter=None`**, `offset=10`, `xform_depth=7`, `output_subdir="cropped"`. Rows 4 and 5 are always reported as per-component defect crops; a record that leaves no file under `cropped/` is a failed record |
| `outputs` | no | rgb + bounding_box_2d_tight + semantic_segmentation (colorized) -- the tight boxes feed the crop |
| ordering | rule | `select_components` -> `add_*` -> `label_components` -> `enable_operation_labels` -> capture -> `postprocess.crop` (compile rejects annotation before selection) |

One picture carrying several defects is one record; one picture per defect
type is several records.

### Row 5 -- Texture defects (SMT solder, USD input)

| Parameter | Required | Default / rule |
|---|---|---|
| defect type(s) | yes | any of `excess`, `missing_solder`, `insufficient`, `bright`, `interrupted`, `dry_contact`; **`bridge` is refused** (partial fulfilment when mixed) |
| `solder_joints(solver=, source=)` | yes, once | `"surface_evolver"`; source is `"usd"` for embedded metadata by default, or the selected legacy sidecar's job-local name. `procedural` here is a validation error by design |
| `add_<type>(ratio=, component_types=)` | yes per type | ratio as Row 4 (`max(1, int(pool * ratio))` on the executing side; these calls take `ratio` only, no `count`); types as named (or the row's default scope), resolved on the attachment; works on both metadata sources |
| `enable_operation_labels(namespace=)` | yes | `"defect"`, as Row 4 |
| `postprocess.crop(...)` | yes | the Row 4 block, `class_filter=None`; a record with no file under `cropped/` is a failed record |
| `outputs` | no | as Row 4 |
| ordering | rule | `solder_joints` before any `add_*`; same select -> operations -> annotate -> capture -> crop order as Row 4 |

Validation: with `manifold3d` or the `evolver` binary missing,
`solder_joints(solver="surface_evolver")` raises at declaration, so a Row 5
record cannot even be produced in a bare environment; the skill reports the
missing prerequisite instead of downgrading the solver.

## Cross-row rules

- **Ask only for missing required inputs when interaction is possible.** Board
  USD and output root have no defaults. Every unset optional value takes its
  documented default and is listed in `resolved_inputs` with source
  `fixed default`.
- **Never override confirmed board metadata** with the `CAD_BOARD`
  values; a sidecar root or the embedded USD default prim wins, and the
  source used is reported.
- **Never substitute a missing attachment or output root.** Both are required
  caller inputs; ask when interaction is possible or stop non-interactive
  automation with refusal case 3. Never use a sample board or a synthesized
  directory in their place.
- **Never emit a defect record whose selection is empty**, and never
  describe a zero-defect outcome as success: a non-empty selection yields at
  least one defect on the executing side, and a clean board on a defect
  request is a failed record.
- **Never claim a step that does not exist**; an unsupported request is
  refused, not approximated.
- **Never execute.** The record ends at a clean `compile_orders(...)`.
