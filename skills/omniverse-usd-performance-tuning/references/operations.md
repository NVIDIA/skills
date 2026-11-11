# Execute recorded targets

Execute the manifest's retained prims, dependencies, mesh paths and tolerances in its output
workspace. Process each distinct asset once per pass. An approved operation chain may supply
these records directly. Fill missing information without reselecting boundaries.
A request to validate before optimizing continues into authorized operations when preconditions
pass. Invalid or stale inputs block only dependent work.

## Validation scope

Use the [concept registry](usd-validation-runner/references/validator-concepts.json) and
[executor](usd-validation-runner/scripts/usd_validation_executor.py) to resolve checks for the
requested scope. Register standalone optimizer validators explicitly with
`usd_optimize.validators.register_all()`; importing the module alone does not register them. Use
`ValidationEngine(init_rules=False)` and enable only the resolved checks. Category discovery is
evidence of availability, not authorization for a full-category sweep.

The end-to-end workflow runs whole-stage source safety checks before asset authoring
and captures the source profile before optimization. Use its recorded results; do not
run those phases separately before invoking it. For standalone operations or
reconstruction outside that workflow, select every registry concept with
`role: safety_gate` and `scope_policy: whole_stage` before mutation. Resolve
composition and binding failures before dependent work. Asset authoring may supply
missing model kinds and organizational Scope types in the output; verify those corrections there.
Record pre-existing broken shader connections separately and reject any new connection failure. This registry query avoids a copied checklist.
On selected geometry targets, include `primvar_unused`, `primvar_indexability`, `vertex_weld`,
`topology_zero_area_faces` and `normals_winding` even when the structural assessment has no geometry
flags. Check `primitive_fit` on eligible converted CAD/BIM/MEP targets. A structural scan cannot
establish that mesh arrays are already clean.

Use flagged containment pairs for `spatial_occluded`, excluding transparent enclosures; use
flagged routing/overlap pairs for `spatial_overlapping` and `spatial_coinciding`. Check nominated
duplicate regions with `geom_duplicates`, adding `geom_duplicates_fuzzy` for near-match evidence.
For simulation, Boolean or printing targets, include `topology_manifold`, `normals_validity` and
`topology_general` at the affected targets. Honor recorded skipped targets and explain their
unverified coverage. A selected probe can time out or be unavailable; it must not disappear.

Scope costly checks to recorded candidate definitions or containment pairs. Preserve sampled,
timed-out, unavailable and not-performed coverage. Check affected authored composition after operations; deep geometric regression
checks belong in development tests.

Use `validate_concepts(stage_path, concepts, registry=load_registry())` for a cheap selected-rule
batch. For costly checks, call `run_scope_note` with the existing scope-note schema and a bounded
`subprocess_concept_runner(timeout_seconds=...)`. Import these functions from the shipped executor;
do not reimplement rule resolution. It distinguishes providers with identically named checkers.
Every selected concept needs an entry in the scope note's `targets`; its `concepts` list alone
does not execute checks. Record missing-runtime exceptions as `blocked_validation_runtime` and
timeouts as `timeout_recorded`. Keep their coverage incomplete even if other targets pass.

## Operation parameters

Consult this section while preparing the assessment's operation configuration.
The workflow confirms exact cached results before optimization. Apply the
[conditional execution order](#conditional-execution-order) to each owned interior.
An authorized standalone chain may use `deduplicateHierarchies` on selected regions
and local depth limits, but the successor worker does not accept that operation.
The successor worker preserves authored animation samples. `optimizeTimeSamples` is outside
that execution route because this route retains authored samples.

Read only the selected entries in the [operation catalog](operations/operations.json). Use
`curation`, `apply_authority`, `requires_confirmation` and `parameter_prerequisites`; read arguments
from the installed version's [upstream guide](upstreams/usd-optimize.md). Prefer canonical operations for matching findings. `documentary` entries require explicit user
intent; specialty operations need supporting findings. Check every operation key against
the verified `operationsAvailable` before execution.

| Authority | Decision |
|---|---|
| `auto` | Run applicable operations within authorized scope and preservation requirements. |
| `auto-within-tolerance` | Apply the recorded physical budget with a notice when the target and preservation intent support it. |
| `intent-gated` | Require the specific identity/appearance/loss decision; reuse an answer already supplied. |

For articulation, physics, metrology, simulation or behavior that depends on variant selections, use
the geometric error limit and functional requirements for that use case. The default
physical budget alone does not establish functional suitability. Animation and the currently selected variant do not establish a functional accuracy requirement. Preserve their transforms and authored state while applying the visual budget to eligible owned geometry. Physics and contact requirements apply to the retained geometry owner, without propagating through independent child assets to the factory assembly. The successor's supported
`fitPrimitives` configuration is appropriate for eligible `piping` and `large-spatial` targets with a notice, provided required identity and
appearance survive. `encapsulated-product` targets need their part identity and fitting intent
resolved. Inspect prismatic `generic` targets on their evidence.
The worker enables Sphere, Cone, Cube and Cylinder fitting and preserves nonconstant primvars and
subsets. Use USD Optimize's default fitting tolerances; the agent does not tune them
or convert a millimeter budget into relative fitting settings. Explicitly requested
`vertexTolerance` and `volumeTolerance` settings pass through unchanged.
`vertexTolerance` controls relative root-mean-square (RMS) surface error;
`volumeTolerance` controls relative volume error. Neither is a millimeter limit
or a maximum surface-deviation guarantee. Exact-geometry requirements exclude fitting;
resolve a stricter fitting requirement before enabling it in that scope.

A pipeline inherits each selected operation's requirements. Read prerequisite arrays or the
object's `fields` and `elicit_from_user` entries. Missing required preflight fields block that
operation; optional fields refine its parameters. Batch unresolved questions using the catalog's
defaults, conversion and skip option. When no prerequisites block exists, derive parameters from
user constraints, scoped evidence or the documented physical budget/upstream default. Ask for a
necessary value with no defensible default. Never invent an option ladder. Existing authorization remains effective
for the same scope and budget. Weld-only `meshCleanup` may run as `auto` within the authorized
scope; its catalog base remains `intent-gated` because topology-repair modes require their own
decision. Explain the proposed change, its affected prims and the loss it introduces when a new
decision is needed. Offer the catalog's skip option, or a plain instruction to skip that operation.
Only present applicable operations with supporting findings.

For small-mesh deletion, set `request.remove_small_geometry_mm` to the authorized
maximum dimension in millimetres (for example, `2` for 2 mm). Omit it to retain
small geometry. The worker runs native `removeSmallGeometry` before mesh cleanup,
fitting, reduction and merging, converts the threshold using stage units and the
largest occurrence scale, and excludes required interfaces. This threshold is
separate from surface-error tolerance. Lossless requests exclude deletion.

The following catalog operations need an authorized standalone chain with its own
scope and composition checks; they are outside the successor worker's supported operation set.
For `removePrims`, `deletePrims`, `removeUntypedPrims` and `deleteHiddenPrims`, inspect the exact
prim list, variants and runtime visibility. `boxClip` requires the region and keep-versus-clip
choice. `diceMeshes`, `manifoldMeshes`, `remeshMeshes` and `shrinkwrap` require the proposed
grid/voxel settings, topology change and a representative preview. `pythonScript` requires a
user-supplied or reviewed script and authorization to execute it. `removeAttributes` requires
the exact attributes and their downstream consumers. `moveMaterials` renames every material's
prim path; it requires the destination path and the bindings, references and layers that
address the current material paths. A `sparseMeshes` finding alone does not
authorize the resulting split or deletion.

### Physical geometry budget

The default physical budget for geometry reduction is **0.1 mm**. Record any user
override. Primitive fitting uses the native relative tolerances described above,
independently of this budget. Explicit exact-geometry requirements exclude
approximation in their scope.
Independently preserve materials, required topology, attributes, authored animation,
relationships, units and semantic identity; geometric tolerance does not relax them.

Convert with `tolerance_stage_units = mm_tolerance / (metersPerUnit * 1000)`.
Account for the largest occurrence scale in shared definitions' local budgets.
Use the budget for applicable native settings across the chain. Report configured
limits separately from independently measured deviation; normal execution trusts
supported native operations and does not re-prove their geometric algorithms.
For `decimateMeshes`, `maxMeanError` is a mean-error setting, not a maximum
pointwise deviation guarantee. Retain it as the quality budget and set
`reductionFactor: 0.0` to disable the default 50% keep quota. A triangle target is a ceiling, not
permission to disable fidelity limits. Record native results; do not increase error beyond an
authorized budget to reach a count. Rate-led reduction belongs to an explicitly requested
level-of-detail count specification. Set `pinBoundaries: true` for shared edges. `reductionFactor`
is the percentage of vertices kept after the operation's internal weld, not a percentage of authored
CAD vertices; compare face counts and measured geometry when evaluating reduction.

Interpret primitive-fit analysis from each primitive type's `meshCount` and
`nonconstPrimvarMeshCount`; top-level `composedCount` is not the candidate count. Reassess
fit candidates after removing UVs proven unused by all consumers. Preserve remaining
required primvars and subsets.

Primitive fitting defaults can discard nonconstant primvars and subsets. When they must survive, use
`ignoreNonConstPrimvars: false` and/or `ignoreSubsets: false` to skip affected meshes; these flags
restrict fitting. Normal primvars alone do not require that restriction. Choose settings that
preserve the required primvars, subsets and appearance.

Use `findCoincidingGeometry` and `findOverlappingMeshes` only on the nominated spatial scopes.
`findFlatHierarchies` identifies prims with large child lists; it does not identify
redundant single-child Xform chains and is not a prerequisite for `flattenHierarchy`.
`optimizePrimvars` can compact eligible primvar storage while retaining interpolation, indexing
and required values. Retain material and downstream primvar requirements in its settings.

## Editable operation inputs

Use the isolated working layers prepared by [asset authoring](asset-authoring.md).
Open the authored definition as a stage and select its owned geometry. Never
optimize through an instance proxy or edit an inherited persistent definition.

Inspect actual traversal when an operation succeeds without expected work: roots,
masks, classes, instances or undefined ancestors can exclude the target. Preserve
referenced `over` holders; promoting one to `def` can render an additional copy.
Convert a class to a concrete definition only in the editable asset that needs it.
Count geometry schemas separately: zero meshes does not mean an empty asset.

## Conditional execution order

- Probe occlusion with `findOccludedMeshes` without a deletion decision; delete discovered geometry only within the
  user's explicit purpose. Authorized deletion precedes work on retained geometry.
- Use `generateNormals` only when missing or invalid normals require it. Weld eligible vertices with `meshCleanup` before
  decimation; preserve UV seams and hard normals. Topology repair such as `makeManifold`
  is a separate decision from weld-only cleanup.
- Select `geometry_representation` from the asset's geometry and workload.
  For piping and conduit sections, choose `analytic_primitives`: clean, fit eligible
  pipes/conduits with native `fitPrimitives`, reduce remaining meshes, then merge
  compatible fittings, brackets and supports that remain meshes. Use the same order
  for other service sections whose geometry supports analytic shapes, such as round
  ducts. Native fitting determines which meshes qualify under the recorded appearance
  and accuracy requirements.
  A static section can remain one atomic asset containing both analytic primitives
  and merged meshes. Analytic shapes need neither independent placement nor separate
  child-asset boundaries. Factory visualization alone is not a reason to bypass fitting
  of pipe or conduit geometry. If a workload requirement excludes this representation,
  record that requirement in the asset decision's reason.
  For equipment interiors and fence modules, choose `merged_meshes` to consolidate
  static parts: clean and reduce, then merge recorded groups. Their merge inputs are
  excluded from fitting; other eligible meshes may still fit.
  Record the choice and reason with the asset decision. Preserve retained child assets,
  authored animation and required interfaces. Publish the optimized outputs using the
  [category-library workflow](asset-authoring.md#working-layers-and-publication).
- Keep merge last among geometry-changing operations. Do not weld or run `meshCleanup` on merged output; contacting surfaces can be lost. Reassess only remaining unmerged scopes and do not replay fitting or reduction for cleanup findings.
- After merging, run native `pruneLeaves`, then `flattenHierarchy` inside eligible
  static scopes of atomic and package definitions, then prune again. Removing empty
  siblings first exposes single-child chains to flattening. Flattening removes intermediate Xforms,
  including static non-identity transforms; pruning removes empty Xforms and Scopes.
  Keep the asset root, retained child assets, required editable prims, animated
  interfaces and addressed targets. Static editable Xforms can move to shorter
  paths while retaining their name and placement; the worker records those paths
  for publication and interface checks. Animated and externally addressed prims
  retain their paths. The worker derives scopes and updates geometry
  paths from native outputs; no per-Xform agent decisions are needed. Record native
  outcomes and removed prim counts. Static `purpose`, `refinementLevel` and
  `refinementEnableOverride` on geometry move with their prim; those properties alone
  do not protect ancestor Xforms. Authored animation and explicit interfaces still do.
- Run `computeExtents`, remove proven unused data, then clean up the assembly while retaining protected prims.
  Keep `pruneLeaves`'s `preserveUnloadedPayloads: true`. Never remove primvars without proving
  that material/shader consumers do not need them; textured/scanned assets retain required UVs. Check UsdShade connections, primvar readers and MDL
  consumers before removing a primvar with `removeUnusedUVs` or direct authoring; uncertain usage means retaining it.
- Apply an explicitly requested palette after merge on editable output layers, through
  [apply_palette.py](optimization-report/scripts/apply_palette.py). Its nearest-color behavior
  is not permission for a generic appearance change. Include referenced over-holder content
  and preserve its specifiers.

For `deduplicateGeometry` on the verified package family, tolerance is absolute in stage units. Use
`considerDeepTransforms: 0` unless the other setting has been qualified in development tests. Its
instanceable-reference default can make later decimation a no-op; optimize definitions before
instancing or use a verified non-instancing method.

## Interior merge

For native `merge`, read `boundary_path`, `protected_paths`, and the interior `operation_scope` from the target.
Use explicit `mesh_paths` and `output_path`; the scoped helper supplies native `meshPrimPaths`.
Different material bindings and face `GeomSubset` children in the `materialBind` family
can merge: native `merge` preserves their face assignments through material subsets.
Keep other child prims and explicitly addressed subsets outside the merge. The scope
helper groups eligible meshes by protected scope. Native merge determines compatibility,
handles supported primvar interpolation and indexing, and preserves material assignments.
Use the recorded mesh paths and exclusions; native merge may partition incompatible
mesh-wide attributes.

Use [component_scope.py](component_scope.py)'s `check_stage_scope(stage, target)` immediately
before merging. It returns the native merge configuration only when the selected meshes are
authored in the edit layer and the output stays inside the retained Xform or Scope. It checks
recorded protected paths, motion, physics and relationship targets. Assessment and
execution use the same definition-local `internal_paths`, including accepted
organizational ancestors inside the asset. Internal scope stops at retained child
assets; an organizational prim outside the asset cannot exempt its descendants.
Within that scope, source kinds, names and descriptive CAD material attributes do
not establish protected interfaces. Material bindings are preserved by native merge.
Record required semantic boundaries in `protected_paths`;
tools expand accepted internal subtrees without asking for every descendant path.
The frontier and report validator share these scope checks. A failed check blocks this merge;
report the offending paths instead of silently changing the target. An unscoped merge requires
the existing recorded `identity_change_authorization` or `intent_confirmed` evidence.

Read the installed `mergePoint` options before execution. The scoped helper uses Root Prim
(`mergePoint: 7`) with a `rootPath` relative to the selected meshes' common top-level root.
Stage mode can also stop at a composition root, so an absolute name can duplicate that prefix.
Keep the absolute destination in `operation_scope.output_path`; resolve and check the native
result against that destination before remapping source paths. Both inputs and output must
stay inside the retained boundary and any population mask.

Merge eligible interiors within the retained asset boundary with native `spatialMode: 0`
(the default). Do not compute an AABB surface-area ratio or impose a coherence cutoff.
When a loading, visibility or culling requirement calls for spatially bounded meshes,
use native clustering instead: put `spatialMode: 1`, `spatialThreshold` and
`spatialMaxSize` in the merge group's `operation_scope.spatial_merge` settings.
Read their units and semantics in the installed `merge` documentation. Do not invent
universal distance or size limits. `spatialMode: 2` caps vertex count;
`spatialMode: 3` merges shared seams and is not a general spatial partition.

For clustered merges, `operation_scope.output_path` is the containing prim for
native output meshes. The worker records all returned paths and removed source paths;
it does not infer which input belongs to each cluster. Subsequent assessment,
publication and reporting include every output. Retain material assignments,
protected paths and placement. Spatial clustering never expands the selected scope.

For `optimizeMaterials`, set `materialsPath` in the layer authoring the materials. Assembly
overrides cannot remove referenced library data.

## Point-instancer conversion when requested

Point-instancer conversion removes individual prim paths and requires explicit
authorization and supported scope/remapping handling. Native catalog availability
does not establish that the workflow can preserve those interfaces. Retain unsupported
content and report the limitation; do not infer emptiness from zero mesh count.
`organizePrototypes` concerns existing scene-graph prototypes, not point-instancer conversion.

## Invoke, schedule and persist

Run `scripts/successor_workflow.py --manifest MANIFEST --workspace OUTPUT --max-workers N`
with an assessed successor manifest and a fresh output directory. For a shared
library, add `--common-library LIBRARY_DIRECTORY`; supply its `index.json` during
assessment of each subsequent station and run one publisher at a time. The station
consumer is `OUTPUT/after.usdc`; publication layout is defined in
[asset authoring](asset-authoring.md#working-layers-and-publication).
The [compute adapter](../scripts/successor_execution.py) uses the existing
[native worker](usd-optimize-run-operations/scripts/native_worker.py) and
[batch runner](usd-optimize-run-operations/scripts/run_batch.py). Each unique
definition has an isolated writer, native execution evidence and output/composition checks. Explicit content
prerequisites determine scheduling; shared category membership does not serialize
geometry computation. Publication follows [asset authoring](asset-authoring.md).

Require evidence for cleanup, primitive fitting, reduction, merging and reuse on
each definition. Execute applicable work within the recorded budget; explain
unchanged or inapplicable results with inspected evidence. Inspect outputs and composition; unchanged counters alone do not establish completion.
The worker measures each written definition by geometry schema, including analytic
primitives. A mesh count of zero does not establish an empty target.

Use the compute adapter's resume support only while source, dependencies, runtime,
parameters and output fingerprints match. Failed checks invalidate dependent work;
independent ready jobs can continue. Retain failed written outputs for inspection.

Export edited layers with `Sdf.Layer.Export()` to new `.usdc` files. For compacting an existing output,
export to a sibling temporary file, verify it, then replace it atomically. `Save()`
can retain unused crate arrays. `Stage.Export()` flattens composition; the final
consumer must retain the agreed references and occurrence interfaces.
