# Fixed defaults

Reference for `skills/cdf-simulation`. Every optional value here is a
documented, consistent choice rather than a guess made fresh each run. Read
this before building any script. The board USD and output root are required
caller inputs and have no defaults; every run takes the caller-supplied
attachment, and a missing attachment is refusal case 3 in `SKILL.md`.

## Board identity and paths

Resolve `root_path` from the selected metadata source. **Inspecting
the attachment is a required step, not an option**: an attachment that was
never opened has no confirmed root.

1. Prefer embedded USD metadata: use the stage's confirmed board root.
   Read it from the file -- `Usd.Stage.Open(path).GetDefaultPrim().GetPath()`
   with `pxr` available, else the `defaultPrim = "..."` token in the
   `.usda` header (a `.usd`/`.usdc` attachment needs `pxr`). It is `/Board`
   on pcb-builder board exports.
   Exception -- a panel: when the stage composes several USD instances of one
   board under a `StepRepeats` scope, the default prim (`/World`) is only a
   container; the root is the carrier Xform that holds `StepRepeats`
   (`reference-flows.md`, "Panels");
2. for a legacy standalone board that relies on a selected
   `<stem>.components.json` sidecar, use the board root it names and confirm
   that it resolves in the supplied USD. A nearby sidecar does not override
   embedded metadata automatically;
3. if the attachment cannot be opened at all (path unreadable, sandbox
   forbids it, binary file and no `pxr`), **stop and report it** -- refusal
   case 3 with the reason "attachment root unresolvable" -- rather than
   guessing. A run that could not read its attachment has failed to resolve
   a required input; it must not hand off a plan built on another board's
   root.

`payload-contract.md` (Row 1) states the same order.

The CAD_BOARD example board used by every `pcb_aoi/examples/allegro_*.py`
example has these paths. Use them only when the attachment resolves to that
root; never as a stand-in for a board whose root was not read:

```python
root_path         = "/Board_CAD_BOARD"
camera_path       = "/World/camera_light/Camera"
camera_xform_path = "/World/camera_light"
ring_light_root   = "/World/camera_light/aoi_ring_light"
```

These four values are NOT mechanically derivable from the USD filename
(`CAD_BOARD_layout.usd` and `CAD_BOARD_aoi.usd` are different files but share
the same `root_path`). If the attached USD is a different board (it ships its
own `<name>.components.json` naming a different root, or its embedded
metadata resolves to a different default prim such as `/Board`), use that
board's own metadata instead and note the source in the run summary — do not
force the CAD_BOARD paths onto a board that carries its own. If inspection
does not resolve a unique board root, request an explicit root path rather
than selecting the first root prim.

Component source: prefer `source="usd"` for embedded USD metadata. Attach a
matching legacy component sidecar only when a standalone board relies on it
or the caller explicitly selects that source. Use
`job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)`
before referring to `source="components"`. Keep component selection, solder,
and annotations on that source; do not mix it with embedded metadata. File
presence alone does not choose a source. Instanced panels require embedded
USD metadata and cannot use legacy component sidecars.

## Named components must resolve on the attachment

A component the prompt names -- a type or footprint (`C_0805_2012Metric`) or a
refdes (`C22`) -- is used exactly as named, and only if the attached board
has it. There is no substitution. A name the attachment cannot resolve fails
validation (refusal case 4 in `SKILL.md`): say what was asked, list what the
attachment has, build nothing. Rendering some other component in its place
would hand back a successful dataset for the wrong part, and the mismatch
does not show in the images.

| Prompt | Scope used |
|---|---|
| Names a component the attachment's own metadata confirms (sidecar entry, or embedded component metadata) | That component, as named, however unfamiliar it looks (`C_0805_2012Metric`, `R_0603_1608Metric`, `SOT-723` ... are pcb-builder footprint names) |
| Names a component the attachment does not have | None: refusal case 4, nothing built |
| Names no component | The components the attachment itself carries (`SKILL.md`, Inputs, "Component types"): every SMT family for rows 3-5, the through-hole parts by refdes for row 2. On a one-family board that is that one family. If the attachment carries none of the row's kind, refuse rather than hand off an empty selection |

`PASSIVE_SMD_0805` is the SMT passive family of the CAD_BOARD example board.
It is a valid scope on that board; it is neither a default for a prompt that
names no component nor a fallback for a name another board does not have.

**Check the attachment, not an impression.** Enumerate its component types
before deciding either way:

- embedded source: the metadata attribute `pcba:component:footprintId`
  on each component prim of the composed stage, e.g. with `pxr`
  `{p.GetAttribute("pcba:component:footprintId").Get() for p in stage.Traverse() if p.HasAttribute("pcba:component:footprintId")}`.
  Without `pxr`, text-search the attachment's whole folder, not only the
  root file: a pcb-builder export keeps its components in a sublayer
  (`usd/components.usda`) that the root `.usda` merely references.
- selected legacy sidecar: the `type` (or `footprint`) field of each entry in
  that board's matching `<stem>.components.json`.

The run summary says which check was made ("`C_0805_2012Metric` found in
the attachment's component types, used as named"); a refusal lists the
types that were found.

## THT board requirements

Rows 1 and 2 both take a THT board, but they need different things from it:

- **Row 1 (Golden board)** works on any THT board whose components resolve
  as described above.
- **Row 2 (pin variation)** additionally needs the through-hole leads
  authored with the `pcba:mesh:role` schema (`pkg_lead` meshes, plus
  `pcba:board:holeRadiusMm`). On a board without it,
  `job.operations.pin_variation(...)` fails at execution time with
  `ValidationError: pin_variation found no through-hole leads: this board
  authors no 'pcba:mesh:role' = 'pkg_lead' meshes...`.

Do not assume an arbitrary attached USD authors the `pcba:mesh:role`
schema. If Row 2's `pin_variation` call fails with the "found no
through-hole leads" error above, report that the attached board's THT
metadata is insufficient for pin-geometry variation rather than retrying
with invented parameters.

## Coverage on small selections

A requested defect that modifies zero components is **not** an acceptable
outcome on a non-empty pool, and the executing side agrees: proportional
targeting selects `max(1, int(pool * ratio))`, clamped to the eligible pool
(pose operations in `scripts/sdg/standalone/defects_utils/selection.py`,
solder defects in `cdf/execution/isaac/job_order_runtime.py`), and a pose
operation that selects from a non-empty pool and applies to none of it
raises instead of exiting clean. So `ratio=0.2` (the no-coverage-phrase
default) on a one-component selection yields one defect, exactly as
`ratio=1.0` does; the skill states the expected count
(`max(1, int(1 * 0.2)) = 1`) in its summary. `count=1` on
`add_shift`/`add_tombstone`/`add_sideflip` is the explicit spelling of the
same thing for a one-component selection (the solder `add_*` calls take
`ratio` only). A clean render with exit 0 on a defect request therefore
means the selection matched nothing, which is a failed case, never a default
doing its job. On a board with tens of parts the same default selects 20%
of them, as intended.

## THT pin-variation magnitude ranges

`job.operations.pin_variation(...)` takes a `(lo, hi)` **range** for each
axis, not a single value or boolean — a bare "generate images with tilt
variation" names no magnitude, so use these fixed ranges (all inside
`PIN_VARIATION_BOUNDS`, chosen as representative mid-range values, not
derived from any specific board spec — note this in the run summary):

| Prompt term | API param | Fixed default range |
|---|---|---|
| protrusion | `height_variation` (mm) | `(0.3, 1.0)` |
| tilt | `tilt_range` (degrees) | `(5.0, 15.0)` |
| offset | `offset_range` (mm) | `(0.1, 0.4)` |
| cut | `cut_angle` (degrees) | `(5.0, 15.0)` |

Selector: `components=[...]` (explicit refdes list). Use the part the prompt
names, confirmed on the attachment; when the prompt names no part, use the
through-hole parts the attachment carries, read from it by refdes (on a
one-part board, that one refdes). Use `package_types=[...]` only when the
prompt names a package type. If the attachment has no through-hole part,
report that no THT target exists rather than guessing a refdes.

## Coverage ratio — applies to BOTH structural and solder defects

- `"full board coverage"` → `ratio=1.0` against whatever pool is currently
  selected (ratio is literally "fraction of the selected/eligible pool").
- A bare count-free request with no "full board coverage" phrase → a fixed
  moderate default of `ratio=0.2` (20% of the selected pool), matching the
  `ratio` scale used across the repo's own worked examples. **This default
  applies identically to every `add_*` call in this skill** — structural
  (`add_shift`/`add_tombstone`/`add_sideflip`) and solder
  (`add_excess`/`add_missing_solder`/etc.) alike; there is nothing
  solder-specific about it.
- `ratio` never rounds a non-empty pool down to zero: the executing side
  selects `max(1, int(pool * ratio))` for pose and solder defects alike. A
  request that resolves to a non-empty selection therefore always produces
  at least one defect -- state the expected count in the run summary. A
  selection that is empty (`types=` matched nothing on the attachment) is
  not served by any ratio; report it instead of handing off a plan that
  renders a clean board.

## Structural-defect magnitudes — required, and never in the prompt

`ratio` is not enough to build a row-4 call. Each structural operation has
**required** magnitude kwargs with no defaults in the API, and a catalog
prompt ("generate tombstone structural defects") never supplies them.
Omitting one is a `TypeError` at the call, before compilation. Use these
fixed values, drawn from the repo's own Allegro-board defect example
(`pcb_aoi/examples/allegro_defect_job.py`):

| Prompt term | API method | Fixed default magnitudes |
|---|---|---|
| shift / offset (pose) | `add_shift` | `translate_range=0.2` (mm), `rotate_z_range=15` (degrees) |
| tombstone | `add_tombstone` | `angle_range=(70.0, 90.0)` (degrees) |
| sideflip | `add_sideflip` | `angle_range=(70.0, 90.0)` (degrees) |

Other examples in the repo (`allegro_v7_defects_with_solder.py`,
`distributable_board_defects.py`) use larger values — `translate_range=1.0`,
`rotate_z_range=25.0`, sideflip `(60.0, 80.0)`. Those are for other boards;
this skill uses the set above for consistency, and the run summary should
say which set was applied.

**Only `add_tombstone` takes `component_types` directly, and only on a
sidecar board.** `add_shift` and `add_sideflip` do not — to scope either to
a named component, call `job.scene.select_components(types=[...], source=...)`
BEFORE the `add_shift`/`add_sideflip` call, narrowing the pool first. Do not
attempt to pass a `component_types=` kwarg to `add_shift`/`add_sideflip`; it
does not exist and will raise. On an embedded-metadata board omit
`component_types` from `add_tombstone` as well and scope it through the
narrowed selection with `eligibility_source="usd"`: supplying the list there
applies the operation to nothing. See `payload-contract.md`, Row 4,
"scoping".

## Combining multiple supported defect types in one request

A prompt can name more than one supported defect in the same request (e.g.
"generate excess solder and missing defects", or in principle two
structural defects together) — this is different from mixing a supported
type with unsupported `bridge` (see below, which is about dropping the
unsupported clause). For two or more **supported** types: declare
`solder_joints(...)` once (solder defects only need it declared a single
time, not once per defect call), then chain each mapped `add_*` call back to
back against the same `component_types` scope, each with its own
independent `ratio` resolved by the rule above. Order between the `add_*`
calls does not matter for solder defects (each targets independent joints);
for structural defects, `pin_variation`-style "last group wins" ordering
does not apply either — `add_shift`/`add_tombstone`/`add_sideflip` are
independent operations and can all be declared in one job.

**But read that together with "Tell the executing side: one `render()` per process"
in `SKILL.md`, because the two answer different questions.** Declaring several
structural `add_*` calls in one job puts all of those defects in the SAME
images: one job, one render, one process. That is right when the prompt wants a
board carrying several defects at once. It is NOT how you produce one image set
per defect type — that is several variants, and each variant needs its own
process, because a second `render()` in one process segfaults Kit. So: several
defects in one picture is one job; several pictures each with one defect is
several processes.

## Solder texture-defect mapping

| Prompt term | API method | Notes |
|---|---|---|
| "excess solder" | `add_excess(ratio, component_types=...)` | Direct match |
| "missing" (solder/texture) | `add_missing_solder(ratio, component_types=...)` | Direct match, no shape params |
| "insufficient" | `add_insufficient(ratio, ...)` | Direct match |
| "bright" | `add_bright(ratio, ...)` | THT-only |
| "interrupted" / "dewetting" | `add_interrupted(ratio, ...)` | Both THT/SMT |
| "dry contact" | `add_dry_contact(ratio, ...)` | THT-only |
| **"bridge"** | **No API method exists.** | **Fail loudly** — see below |

All solder defects require `job.operations.solder_joints(solver="surface_evolver", ...)`
declared first — `solver="procedural"` plus any solder-defect call is a
validation error by design (the API rejects `procedural` there
deliberately). Use `source="usd"` for embedded metadata. Use the attached
sidecar's source name for `solder_joints(...)` only when the job uses a
selected legacy component sidecar, consistent with selection and annotations.

**Row 1 is the one place `procedural` is the right solver.** The catalog's
golden board is "procedural solder, no defects", so the Row 1 template
declares `solder_joints(solver="procedural", ...)` and no `add_*` call —
which is exactly the combination the API accepts. A golden board rendered
without that declaration shows bare pads, which is not what the catalog row
describes. Do not pair `procedural` with any solder defect, and do not
switch Row 1 to `surface_evolver` just because Row 5 needs it: the
procedural solver has no environment prerequisites, which is also why Row 1
can be dry-run validated anywhere.

**`surface_evolver` needs TWO things the environment may not have, and BOTH
of them fail at the same, earliest possible stage: inside the
`job.operations.solder_joints(...)` call itself.** `pcb_aoi/sections.py`
calls `resolve_solver(solver)` in the body of `solder_joints`, so neither
missing dependency waits for `compile_orders()` or for `render()`. A Row 5
script cannot even be dry-run validated without them:

1. **The `manifold3d` Python package.** Without it, `job.operations.solder_joints(solver="surface_evolver", ...)`
   raises `ValidationError: unknown solder solver 'surface_evolver'
   (registered: ['none', 'procedural'])` **immediately at that call** — not
   at `render()`, not even at `compile_orders()`, but the moment
   `solder_joints(...)` itself is declared. The repo's `requirements.txt`
   lists `manifold3d`, but a minimal validation environment
   (`numpy`/`Pillow`/`pydantic`/`PyYAML` only, as in the CI `unit-tests`
   job) does **not** install it — so a Row 5 script that compiles cleanly in
   a full `cdf-simulation` environment will fail to even declare
   `solder_joints()` in a bare validation container unless `manifold3d` is
   installed too.
2. **The native `evolver` binary.** With `manifold3d` present but no binary,
   the same call raises `ValidationError: solder solver 'surface_evolver' is
   unavailable: Surface Evolver backend unimportable: Surface Evolver binary
   not found` — again at `solder_joints(...)`, not at `compile_orders()` and
   not at `render()`.

The shipped `cdf-simulation` container has both preinstalled. Outside it,
install `manifold3d` (`pip install manifold3d`) AND either build the binary
from `pcb_aoi/solder_evolver/third_party/evolver/build.sh` or point
`SOLDER_EVOLVER_BIN` at an already-built `evolver` binary. With *either* one
missing, even a compile-only dry-run validation of a Row 5 script is
impossible, not merely a render-time risk. Before running or even
*validating* any Row 5 job, confirm the executing environment has both — an
Airflow worker image or CI validation container without them will reject
every solder-defect script at declaration time, not just fail to render.

**Bridge defects are unsupported.** If a prompt asks for "bridge defects"
(alone or mixed with other solder-defect requests), do not generate
anything for the bridge portion and do not silently substitute
`add_interrupted` or any other method — respond:

> Bridge (solder short between adjacent pads/leads) is not supported by the
> current `cdf` API. Available solder defect types: insufficient, excess,
> bright, interrupted, missing_solder, dry_contact.

If the prompt mixed bridge with a supported type (e.g. "generate bridge and
excess solder defects"), still generate the supported part and report the
bridge portion as unsupported in the same response — do not fail the whole
request over one unsupported clause.
