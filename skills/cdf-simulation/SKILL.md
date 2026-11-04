---
name: cdf-simulation
description: Component-owned guidance for CDF Simulation requests submitted directly or through Airflow-triggered DIG Flow Catalog prompts — golden THT board, THT pin-geometry variation, good/clean SMT images, structural pose defects, and solder texture defects, from a board USD. Turns one matched prompt into one validated `cdf.CDFJob` build and its compiled JobOrder payload; it never executes a render itself. Requires a caller-supplied board USD and output root, asks for either when missing from an interactive request, and resolves optional details with fixed, documented defaults.
license: Apache-2.0
owner: NVIDIA
service: cdf-simulation
metadata:
  version: 1.0.0
  author: NVIDIA CDF <cdf@nvidia.com>
  tags:
  - physical-ai
  - synthetic-data-generation
  - simulation
  - pcba
  - aoi
  - airflow
---

# cdf-simulation

Turns DIG Flow Catalog prompt templates and equivalent direct requests into
`cdf` jobs through the declarative Python API. This skill is the
**component-owned guidance** for `cdf-simulation`: a caller supplies a board
USD and output root, this skill builds and compiles the job, and the caller
can pass the payload to its own executor. Airflow may inject those required
inputs, or the caller may provide them directly. If either is missing and
interaction is possible, ask once. All optional ambiguity is resolved with
fixed, documented defaults. If a request truly cannot be served (an
unsupported defect type, a non-USD input), say so rather than guessing at a
fake capability or silently doing nothing.

**This skill does not execute.** It produces a validated `CDFJob` build and
its compiled JobOrder payload, then stops. An Airflow DAG or another executor
chosen by the caller performs rendering.

## Catalog coverage

This skill implements five DIG Flow Catalog rows, all taking a board USD as
input:

| # | Row | Status |
|---|---|---|
| 1 | Golden board (THT) | Implemented |
| 2 | Variation type (THT pin geometry) | Implemented |
| 3 | Good images (SMT) | Implemented |
| 4 | Structural defects (SMT) | Implemented |
| 5 | Texture defects (SMT solder) | Implemented, except `bridge` (unsupported — see below) |

### Reference flows

Beside the catalog rows, the skill serves the five **reference flows** in
[`references/reference-flows.md`](references/reference-flows.md): complete
inspection jobs, each phrased as one prompt whose compiled plan must equal
the equivalent YAML config. A prompt carrying a flow marker -- a scan grid
and "stitched", a defect on a **named refdes** ("shift on C33",
"insufficient solder on J17", "missing-component defect on LS1"),
"component-focus" or "pin-focus" crops -- is a flow, not a catalog row, and
takes its fixed values from that file (they differ from the catalog
defaults: shift is `translate_range=0.5, rotate_z_range=5.0` with no
`variant_index`, crops are ROI captures rather than `postprocess.crop`).

| Flow | Prompt marker |
|---|---|
| G golden scan | "golden board dataset ... NxM scan ... stitched" (single board or panel) |
| C component defects | "shift/tombstone/sideflip/reverse polarity on <refdes>" + "component-focus crops" |
| S SMT solder defects | "insufficient/excess/interrupted/missing solder on <refdes>" + "component-focus crops" |
| M missing component | "missing-component defect on <refdes>" |
| T THT solder defects | six THT solder defects on refdes + "pin-focus crops from the bottom" (single board or panel) |

## Fixed defaults

Every unspecified optional detail resolves to a documented default rather
than a question. The full table -- board identity and paths, the rule that a
named component must resolve on the attachment, THT board requirements,
pin-variation magnitude ranges, **the required structural-defect magnitudes**
(`translate_range`, `rotate_z_range`, `angle_range` -- the API has no
defaults for these and a catalog prompt never states them), the coverage
ratio, how multiple defect types combine, and the solder texture-defect
mapping -- is in
[`references/fixed-defaults.md`](references/fixed-defaults.md).

**Read it before building any script.** The values are not obvious and
several of them are load-bearing: the coverage ratio applies to structural
*and* solder defects, and the crop `class_filter` taxonomy is not the same as
`component_types`.

## Payload contract

What this skill returns to its caller, what it accepts, and how each value is
validated is in
[`references/payload-contract.md`](references/payload-contract.md). The short
version: the output is a `CDFJob` build plus the compiled `JobOrderPlan` from
`job.compile_orders(...)`, serialized with `JobOrderCodec.dumps(plan)` (a
bare `plan.model_dump()` is not JSON-serializable), together with a
resolved-inputs summary naming every default that applied. The caller supplies
the board USD and output root -- both **required**, with no fallback. This
skill supplies everything else from the catalog row and fixed defaults, and
marks each variant for an independent process.

## Instructions

Four steps, in order. Step 1 decides everything downstream, so do not start
writing a script before the row is settled.

1. **Match the prompt to exactly one catalog row or one reference flow.**
   Check the flow markers first (`references/reference-flows.md`, "Routing"):
   a named refdes target, a scan grid with "stitched", "component-focus" /
   "pin-focus" crops. With a marker present the prompt is that flow; without
   one it is a catalog row. A prompt matching **no row and no flow**
   explains the supported request shapes and asks the caller to choose one;
   it never assembles a best-effort substitute out of the pieces that do
   exist.
2. **Resolve every input from the precedence order in [Inputs](#inputs).**
   Anything still unset takes its documented default from
   [`references/fixed-defaults.md`](references/fixed-defaults.md).
3. **Build one `cdf.CDFJob` script**, following the row's shape in
   [Workflow](#workflow). One job per variant: the executing side must launch
   each variant as its own process, because sequential
   `job.capture.render(...)` calls in one process crash the second Kit
   subprocess -- so the payload says one variant, never a loop.
4. **Validate, then return** -- compile the job with
   `job.compile_orders(...)`, read the result, and return the build plus the
   compiled plan to the caller. Do not call `render()` here. Report what was
   assumed alongside the payload.

## Inputs

Each value is resolved from the first source that supplies it:

**state file -> explicit prompt arguments -> agent context -> user prompt text**

| Input | Required | Where it comes from if unset |
|---|---|---|
| Catalog row | yes | Matched from the prompt shape; no default. No match asks the caller to choose a supported shape |
| Board USD (attachment path) | yes, caller-provided | **No fallback.** Ask for a missing path when interaction is possible; for non-interactive automation, stop with refusal case 3. Never substitute a sample board |
| Output root | yes, caller-provided | **No fallback.** Ask for a missing path when interaction is possible; for non-interactive automation, stop with refusal case 3. Never synthesize a directory |
| Component sidecar | optional legacy input | Prefer embedded USD metadata with `source="usd"`. Attach a matching `<stem>.components.json` only when a legacy standalone board relies on it or the caller explicitly selects that source; file presence alone does not select it. Instanced panels require embedded metadata |
| Board `root_path` | no | Read from the supplied stage, or from the selected legacy sidecar for a sidecar-backed board. For instanced panels follow `references/reference-flows.md`, "Panels". Opening the attachment is required; an unreadable or unresolved board stops with refusal case 3. Never use another board's root as a fallback |
| Named component | from the prompt | Passed through as named when present in the selected metadata source (embedded USD by default, or an explicitly selected legacy sidecar -- list types, do not guess); a missing name is refusal case 4, never substituted. State the check in the run summary |
| Component types | no | The row's documented set |
| Seed, scan grid, board side, resolution, pin numbers (flows) | no | Explicit prompt arguments in the forms listed in `references/reference-flows.md` ("Seed 7", "a 5x4 scan", "of the bottom side", "at 2000x2000", "(pin SH)"); otherwise that file's per-flow defaults |
| Panel instances | from the attachment | When the attachment composes one board as several USD instances (`<root>/StepRepeats/SR_*`), register all of them with `job.scene.instances(...)`; image budgets multiply by the instance count (`references/reference-flows.md`, "Panels") |
| Coverage ratio | no | The single documented ratio, applied to structural *and* solder defects; the executing side selects `max(1, int(pool * ratio))`, so a non-empty selection always yields at least one defect |
| Defect magnitudes | no | The per-type ranges in `references/fixed-defaults.md` |
| `class_filter` | no | `None`. Never a component-type string -- different taxonomy |

State what was resolved and from where in the run summary so the caller can
see which defaults applied.

## Workflow

### 1. Match the prompt to a catalog row

Use the sample-prompt phrasing in the table under "Catalog coverage" and in
"Examples" below to classify the request into exactly one of rows 1-5.

If a prompt does not clearly match any of the rows or a reference flow,
explain the supported request shapes and ask the caller to choose one.

### 2. Resolve inputs deterministically

Extract from the prompt: attached USD path, output root, named component(s),
variation/defect type, "full board coverage" flag. Then **open the
attachment** and read two things from it: the board root and its component
types, using embedded metadata by default or the selected legacy sidecar for
a sidecar-backed standalone board. A named component that
appears in those types is used as named; a name that is absent is refusal
case 4 -- never a substitution. Apply the fixed defaults above for optional
details the prompt does not specify, and record both reads (root source,
component check) in the run summary.

The attachment path and the output root are the two inputs with **no
default**. If either is absent and interaction is possible, ask for it once.
For non-interactive automation, stop and report it (refusal case 3).
Never substitute a sample board or invent an output directory. Prefer embedded
USD metadata. Attach a component sidecar only when a legacy standalone board
relies on it or the caller explicitly selects it; do not switch sources merely
because a `.components.json` file is nearby. Keep all metadata-dependent calls
on that source. Instanced panels cannot use legacy component sidecars.

### 3. Build the `CDFJob` script

**`import pcb_aoi` before constructing the job.** The `"pcb-aoi"` domain
registers on import; without that import `CDFJob(domain="pcb-aoi", ...)`
raises `unknown domain 'pcb-aoi' (registered: none)` on the very first line.
Every template below assumes `import pcb_aoi` and `from cdf import CDFJob`
at the top of the script.

**Every row requires `job.capture.renderer(mode="pathtracing", resolution=..., spp=..., total_spp=..., max_bounces=...)` before `capture.camera`/`scan_grid`/`outputs` -- it is a required field, `compile_orders()` rejects a plan missing it ("capture render pass 'default' is missing: renderer"). Every template below assumes this call is present even where not shown explicitly.**

**`capture.camera(rotation_range=...)` is required, not optional** -- only
`side=` has a default. A top-side capture with no rotation asked for still
needs `rotation_range={"z": 0}`; omitting the kwarg is a `TypeError` before
anything is compiled.

**Several `operations.add_*` calls have required magnitude kwargs with no
defaults** (`add_shift` needs `translate_range` *and* `rotate_z_range`;
`add_tombstone` and `add_sideflip` each need `angle_range`). A catalog prompt
never states them, so they come from the fixed table in
[`references/fixed-defaults.md`](references/fixed-defaults.md) -- do not invent
a number and do not ask.

**Declaration order matters and is enforced.** `job.scene.select_components(...)`
must be declared before `job.annotations.label_components(...)` --
`compile_orders()` rejects the reverse order with
`annotations.components.apply requires component selection to be declared
first`. Every template below follows select → (operations) → annotate.

**`scene.select_components(source=..., types=None)` without `types=` needs
either USD-embedded component metadata or `source="components"` pointing at
an attached sidecar** -- bare `source="usd"` with no `types` and no sidecar
fails at runtime with `component selection needs explicit types or board
metadata`. Prefer confirmed embedded USD metadata. For a legacy standalone
board that requires a sidecar, attach its matching file with
`job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)`
and use `source="components"` consistently. Do not mix that source with
embedded metadata or attach a sidecar solely because the file exists.

Generate a standalone script with imports, `build_job() -> CDFJob`, an
`argparse` CLI accepting `--usd-path` and `--output-dir`, and `main()`, using
the per-row template below -- or, for a reference flow, the flow's template
and shared conventions
in `references/reference-flows.md` (explicit `source="usd"`, `output_start(0)`,
`limits(...)`, `realtime_subframes=1`, the three-key camera rotation), which
are the spellings the YAML interpreter produces. Place generated PCB scripts under `pcb_aoi/examples/` only if the run is
meant to be durable/reusable; for a one-off request, build and compile the
job in-memory, then emit the payload
([`references/payload-contract.md`](references/payload-contract.md))
without necessarily writing a file to disk. Execution belongs to the caller's
chosen executor, never this skill.

**Row 1 — Golden board (THT):** the template uses embedded USD metadata. For
a legacy sidecar-backed standalone board, attach its matching component
sidecar and use its job-local source name consistently instead of `"usd"`.

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path=...)  # root confirmed on the supplied stage
job.scene.select_components(types=[<the attachment's component family>], source="usd")
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
# The catalog's golden board carries PROCEDURAL solder: joints are generated,
# no defect is applied. `procedural` is valid here precisely because no
# add_* solder-defect call follows (those require surface_evolver).
job.operations.solder_joints(solver="procedural", component_types=[<the same family>], source="usd")  # component_types names the family the joints are generated for. With a sidecar attached, this reads source="components" too -- all three source= arguments in a row must name the same metadata source
job.annotations.label_components(source="usd")  # likewise source="components" when the sidecar is attached
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="bottom", rotation_range={"z": -90})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                     semantic_segmentation=True, colorize_semantic_segmentation=True)
# No postprocess.crop() call -- this row has no crop.
```

`camera_path`/`camera_xform_path`/`ring_light_root` are optional
(`scene.configure`'s default is `None`) -- omit them rather than assuming
the `/World/camera_light/...` Allegro rig exists verbatim on every board;
the pipeline falls back to a synthesized camera when unset.

**Row 2 — Variation type (THT pin geometry):**

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path=...)  # a board that authors pcba:mesh:role -- see below
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
job.operations.pin_variation(components=[<THT refdes>], offset_range=(0.1, 0.4))  # or tilt_range/cut_angle/height_variation
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="bottom", rotation_range={"z": -90})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True)
```

Row 2 needs a board whose through-hole leads are authored with the
`pcba:mesh:role` schema (`pkg_lead` meshes); a THT board without it cannot
run `pin_variation` at all (see "THT board requirements" in
`references/fixed-defaults.md`). `pin_variation(...)` targets leads directly
by `components=`/`package_types=` -- it does **not** need
`scene.select_components(...)` or `annotations.label_components(...)`
declared first (unlike every other row). Selector: the refdes the prompt
names, confirmed on the attachment; with no part named, see the selector
rule in `references/fixed-defaults.md`.

**Row 3 — Good images (SMT):** board-level (`"20 good PCB images"`) uses
`capture.scan_grid(x_count=X, y_count=Y)` where **`X * Y`** is the requested
count, not `N` in each axis -- `scan_grid(x_count=20, y_count=20)` is 400
renders, not 20. Pick the most square factorisation at or just above the ask
(20 -> 5x4; 30 -> 6x5; a prime like 17 -> 5x4 = 20 and say so), no
crop, `select_components(types=[<every component family the attachment
carries>], source="usd")` -- the families are read from the attachment
(`pcba:component:footprintId`), so on a one-family board the list has one
entry (e.g. `types: [C_0805_2012Metric]`); a bare
`select_components(source="usd")` with no `types=` compiles to a different
plan (`types: null`). Component-level (`"good images of
<component>, max 3"`) narrows first with
`select_components(types=[<component_type>], source="usd")`, then
after the same capture block adds
`job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"], class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")`
and caps `capture.limits(max_images=3)` (or the requested max). **`class_filter=None`, never the component-type string** -- see "Pitfall: `class_filter` is not `component_types`" below: `class_filter=["PASSIVE_SMD_0805"]` compiles and executes with `"error": ""` but silently produces zero crop files, because `annotations.label_components` assigns the semantic class `"capacitor"` to that component, not the literal type string. No solder or structural-defect operations for this row -- clean/golden only; output stops at the rendered/cropped RGB.

Board-level Row 3, complete (only the grid and the render size come from
the prompt):

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board")                       # the attachment's default prim
job.scene.select_components(types=[<all families of the attachment>], source="usd")
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
job.annotations.label_components(source="usd")                              # always: the good images ship their labels
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=X, y_count=Y)                                 # X * Y = the requested count
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                     semantic_segmentation=True, colorize_semantic_segmentation=True)
# no operations, no postprocess.crop
```

**Row 4 — Structural defects (SMT):** the same SMT scene as Row 3, with pose
operations declared after `select_components` and before
`label_components`, then operation labels and the per-component defect
crop. **Rows 4 and 5 always emit per-component defect crops** -- this is
the one output contract for both rows (`references/class-filter-taxonomy.md`):
`annotations.enable_operation_labels(namespace="defect")` puts the defect
class in the segmentation output, and the same `postprocess.crop(...)` block
as Row 3's component-level case follows the capture. A record with no file
under `cropped/` is a failed record, not a board-level variant. The
narrowed `select_components(types=[...])` is the scope of every pose
operation; whether `add_tombstone` also takes `component_types=` depends on
the attachment's metadata source -- see "Scoping by metadata source" below.

This legacy standalone-board example uses a component sidecar (the CAD_BOARD board used by
`pcb_aoi/examples/allegro_defect_job.py`); on an embedded-metadata board
use `source="usd"` throughout and the tombstone scoping below.

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_CAD_BOARD")
job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)  # sidecar board
job.scene.select_components(types=["PASSIVE_SMD_0805"], source="components")  # the scope of every pose op
job.appearance.lighting(...)                              # same block as Row 1
job.operations.add_tombstone(ratio=0.2, angle_range=(70.0, 90.0))
# job.operations.add_shift(ratio=0.2, translate_range=0.2, rotate_z_range=15.0)
# job.operations.add_sideflip(ratio=0.2, angle_range=(70.0, 90.0))
job.annotations.label_components(source="components")
job.annotations.enable_operation_labels(namespace="defect")
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                    semantic_segmentation=True, colorize_semantic_segmentation=True)
job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"],
                     class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")
```

One defect type per script when the prompt wants one image set per type (see
"Tell the executing side: one `render()` per process" below); all three in one
script only when the prompt wants a board carrying all three at once.

**Scoping by metadata source.** `add_shift` and `add_sideflip` have no
`component_types=` kwarg, so the narrowed `select_components(types=[...])`
is always their scope. `add_tombstone` does have one, and whether to use it
is decided by where the attachment's component metadata lives:

- **Embedded USD metadata** (`pcba:*` schema, no sidecar -- every pcb-builder
  export): narrow with `select_components(types=[...], source="usd")`, pass
  `eligibility_source="usd"`, and **omit** `component_types`. Supplying it
  applies the operation to nothing: the list resolves to prim paths the
  stage does not carry and the run fails with `tombstone selected from 1
  components but applied none of them`.
- **Sidecar board** (`<stem>.components.json` attached):
  `component_types=[...]` on `add_tombstone` works there and may accompany
  the narrowed selection. It is never a substitute for narrowing the
  selection, and it is never carried over to an embedded-metadata board.

**Row 5 — Texture defects (SMT solder):** as Row 4 -- same operation labels,
same outputs, same per-component defect crop -- but
`operations.solder_joints(...)` is declared once before any `add_*` solder
call, and the solver must actually be available in the executing environment.
The solder `add_*` calls take `ratio` only (no `count`) and their
`component_types=` works on both metadata sources.

The following is a legacy sidecar-backed standalone-board example. Prefer
`source="usd"` for embedded-metadata boards and omit `attach_sidecar(...)`.

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_CAD_BOARD")
job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)  # sidecar board
job.scene.select_components(types=["PASSIVE_SMD_0805"], source="components")   # embedded-metadata board: types=[<family>], source="usd"
job.appearance.lighting(...)                              # same block as Row 1
job.operations.solder_joints(solver="surface_evolver", component_types=["PASSIVE_SMD_0805"], source="components")  # component_types = the selected family
job.operations.add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])
# chain further supported types here, e.g.
# job.operations.add_missing_solder(ratio=0.2, component_types=["PASSIVE_SMD_0805"])
job.annotations.label_components(source="components")
job.annotations.enable_operation_labels(namespace="defect")
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                    semantic_segmentation=True, colorize_semantic_segmentation=True)
job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"],
                     class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")
```

`solder_joints(solver="surface_evolver")` resolves the solver **inside that
call**: with `manifold3d` or the native `evolver` binary missing it raises
there, before `compile_orders()` is reached. See
[`references/fixed-defaults.md`](references/fixed-defaults.md) — a Row 5 script
cannot even be dry-run validated in an environment that lacks them.

### Pitfall: `class_filter` is not `component_types`

`postprocess.crop(class_filter=...)` takes a derived semantic class, not a
component-type string. Given the latter it compiles, reports `"error": ""`,
and writes zero files. **Default to `class_filter=None`** and filter with
`types=`. Worked example and the two taxonomies side by side:
[`references/class-filter-taxonomy.md`](references/class-filter-taxonomy.md).

### 4. Validate, then return -- do not execute

Run `job.compile_orders(output_dir=...)`
to catch a malformed script before anyone spends Isaac boot time. A clean
compile is where this skill's job ends: return the `CDFJob` build, the
compiled `JobOrderPlan`, and the resolved-inputs summary to the
caller, in the shape
[`references/payload-contract.md`](references/payload-contract.md) defines.
Do **not** call `job.capture.render(...)` from here -- this guidance is
read, not run, and execution belongs to the caller's chosen executor. Report
what was returned; do not claim images were generated, because at this point
none have been.

**Tell the executing side: one `render()` per process.** Running several
variants (for example shift, tombstone and sideflip) sequentially in one
Python script crashes the second `render()`: the Kit subprocess segfaults
mid-run (`carb.crashreporter-breakpad` fatal trace), while each variant as
its own process runs cleanly. This matches documented Isaac/replicator
fragility around rebuilding the render graph mid-session. So when a prompt
implies several output variants ("generate shift, tombstone, and sideflip
defects"), emit one payload per variant and mark them as independent
launches -- never a loop of `render()` calls inside one script.

## Examples

| Prompt | Row | Key resolved values |
|---|---|---|
| "Generate a golden board image using the attached USD" | 1 | bottom camera, `solder_joints(solver="procedural")` with no defect calls, rgb+bbox+semantic, no crop |
| "Using the attached USD, generate images with tilt variation of J13" | 2 | `tilt_range=(5.0, 15.0)`, `components=["J13"]` |
| "Using the attached USD, generate images with cut variation of J13" | 2 | `cut_angle=(5.0, 15.0)` |
| "Generate 20 good PCB images using the attached USD" | 3 | board-level, `scan_grid(x_count=5, y_count=4)` = 20 images, no crop |
| "Generate good images of PASSIVE_SMD_0805, max 3" | 3 | `select_components(types=["PASSIVE_SMD_0805"])`, component-level crop, `max_images=3` (a component the attachment does not have is refused, case 4 -- never substituted) |
| "Using the attached USD, generate tombstone structural defects" | 4 | `add_tombstone(ratio=0.2, angle_range=(70.0, 90.0))` (no coverage phrase → default ratio, which still yields at least one defect on a non-empty selection; `angle_range` is required and comes from the fixed table); `enable_operation_labels` + per-component crop as on every row 4/5 record |
| "Generate shift structural defects for PASSIVE_SMD_0805, with full board coverage" | 4 | `select_components(types=["PASSIVE_SMD_0805"])` then `add_shift(ratio=1.0, translate_range=0.2, rotate_z_range=15.0)`; labels + crop |
| "Using the attached USD, generate excess solder on the passives" | 5 | `solder_joints(solver="surface_evolver")` + `add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])`; labels + crop |
| "Generate excess solder and missing solder defects for PASSIVE_SMD_0805" | 5 | one `solder_joints(...)` declaration, then `add_excess(ratio=0.2, ...)` and `add_missing_solder(ratio=0.2, ...)` chained (see "Combining multiple supported defect types") |
| "Using the attached USD, generate bridge defects on the ICs" | 5 | Fail loudly — bridge unsupported |

## When to refuse

Four cases follow. Each states plainly what was not done and why, produces no
substitute, and — where part of the request is servable — still delivers that
part.

**1. `bridge` mixed with supported solder defects.** Partial fulfilment, not
refusal: generate exactly the supported types named, drop the bridge clause,
report it.

> Requested: bridge and excess solder. Generated: excess solder only
> (`add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])`, 1 variant,
> output at `<dir>`). Not generated: bridge — solder shorting between adjacent
> pads/leads has no method in the current `cdf` API. Available texture
> defects: insufficient, excess, bright, interrupted, missing_solder,
> dry_contact. I did not add any of those to make up the count, and did not
> approximate a bridge from the ones that exist.

**2. A board whose real metadata is being overridden by the defaults.** The
`CAD_BOARD` values belong to the CAD_BOARD example board only.

> Not run. The attached USD carries its own metadata (a `<name>.components.json`
> sidecar, or embedded `pcba:*` metadata whose default prim is the board root)
> naming a different board root, so applying the `CAD_BOARD`
> values would silently render the wrong board identity — a mismatch that
> does not show up in the image. Re-run with that board's own `root_path`, or
> confirm the values are genuinely intended for it.

**3. No board USD, or no output root.** Both are required caller inputs.
An attachment that is not a USD stage (for example a photograph) counts as
no board USD. Ask the caller to provide the missing value when interaction is
possible. In non-interactive automation, stop; do not substitute a sample
board for the attachment or synthesize a directory for the output root.

> Not built. This request carries no board USD (or: no output root). Both are
> required caller inputs and have no defaults; images are not accepted as the
> board input. Provide a USD attachment path and the output root.

**4. A named component the attachment does not have.** A validation
failure, not a substitution. Check component types and refdes in the selected
metadata source (embedded USD by default, or the selected legacy sidecar); if the named one
is not there, build nothing.

> Not built. The prompt names `<name>`, which the attached board does not
> have (checked: <sidecar types | embedded component types>: `<list>`).
> Rendering another component in its place would deliver a dataset for the
> wrong part. Re-run naming one of the listed components, or attach the board
> that carries `<name>`.

What none of these do: assemble a lookalike out of the capabilities that do
exist, or report success for a run that did not happen. This skill returns
a payload and never renders, so it never has grounds to say images were
generated -- that report comes back from the executor, or not at all.

## Guardrails

- Ask only for a missing board USD or output root when interaction is
  possible. Apply fixed defaults for optional details and state what was
  assumed in the run summary.
- Never invent a component name, prim path, or API method that was not
  confirmed in this repo's own code (`pcb_aoi/sections.py`,
  `pcb_aoi/examples/*.py`, `cdf/API_SPEC.md`). If something needed does not
  exist (for example bridge defects), say so plainly.
- Never build on a board root that was not read from the attachment. The
  CAD_BOARD paths belong to the CAD_BOARD example board; an attachment
  that could not be opened is refusal case 3, not a reason to assume that
  board. Confirmed metadata includes the embedded USD default prim, not
  only a sidecar.
- Never substitute a component for one the prompt names. Check the
  attachment's component types first: pcb-builder footprint names
  (`C_0805_2012Metric`, `R_0603_1608Metric`, `SOT-723`) look unfamiliar and
  are real, and a name the attachment does not have is refusal case 4.
- Never substitute a sample board for a missing attachment, or a
  synthesized directory for a missing output root. Both are required caller
  inputs; ask when interaction is possible or stop non-interactive
  automation (refusal case 3).
- Never hand off a defect record whose selection is empty, and never
  describe a zero-defect outcome as the default doing its job: on a
  non-empty selection the executing side applies at least one defect
  (`max(1, int(pool * ratio))`), and a clean board on a defect request is a
  failed record.
- Never claim that a postprocessing step which does not exist in this repo
  was applied.
- Never execute from this skill. Compile, then return the build and the
  `JobOrderPlan` to the caller. A `render()` call here is a bug, not a
  shortcut.
- Never pass a component-type string to `postprocess.crop(class_filter=...)`
  — it is a different taxonomy than `component_types`/`types=` and will
  silently produce zero crop files with no error. Default to
  `class_filter=None`.

## Limitations

- **`bridge` solder defects are unsupported**; the other texture defects in
  row 5 are.
- **One variant per process, on the executing side.** Sequential
  `job.capture.render(...)` calls in a single Python process segfault the
  second Kit subprocess, so each payload this skill emits is one variant and
  is marked as an independent launch.
- **Guidance only.** This skill compiles and returns a payload; it does not
  render. The caller's executor owns execution, and this skill has no view of
  the run's outcome beyond what is reported back.
- **Requires a USD input.** Images or photographs are not accepted as the board input.
- **Two required caller inputs.** The board USD and output root may be supplied
  directly; non-interactive automation must inject both. Optional
  ambiguity resolves to the documented defaults.
- **The defaults are documented, not universal.** A different board needs its
  own confirmed metadata, not the `CAD_BOARD` values.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `"error": ""` but zero crop files | A component-type string was passed to `postprocess.crop(class_filter=...)`, which is a different taxonomy from `component_types` | Use `class_filter=None`; filter by `types=` instead |
| Second render segfaults (reported back by the executor) | Two `job.capture.render(...)` calls in one Python process | Emit one payload per variant, each marked as an independent launch; never a `render()` loop |
| This skill was asked to render | A caller treated component guidance as an executor | Compile and return the `JobOrderPlan`; execution belongs to the caller's chosen executor |
| `select_components` returns nothing | The sidecar and the board USD disagree, or the source is wrong for that board | Confirm the sidecar belongs to that board; check `source=` against the row |
| A defect request rendered a clean board with exit 0 | The selection was empty (`types=` matched nothing on that attachment); a non-empty selection always yields at least one defect | Check `types=` against the attachment's own metadata; do not hand off a plan whose selection is empty |
| Row 4/5 record with no file under `cropped/` | `enable_operation_labels(...)` or `postprocess.crop(...)` was left off the record, or `class_filter` carried a component-type string | Both calls are part of the Row 4/5 contract; `class_filter=None` |
| Compile error naming an unknown operation | An API method was invented rather than confirmed | Check `pcb_aoi/sections.py` and `cdf/API_SPEC.md`; if it is absent, the capability does not exist |
| Prompt matches nothing in the catalog | A request shape this skill does not cover | Explain the supported shapes; do not substitute |
| `pin_variation found no through-hole leads` | The board does not author `pcba:mesh:role` lead meshes | Report that the board's THT metadata is insufficient for pin-geometry variation; do not retry with invented parameters |
