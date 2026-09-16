# Validator Rule Reference

This table maps a reported **validator signal** to its **canonical concept**
and the **backing operation** that fixes it. It is the interpretation source for
turning findings into op candidates — it is **not** an execution allowlist and
it does **not** publish tiers.

**Single source of truth.** Validator *identity* (`module` + `class_name`),
*tier*, *scope policy*, and *preferred provider* live only in
`../../validator-concepts.json` (keyed by canonical concept). Do not restate
tier numbers here or in the runner README; if a tier matters, read it from the
registry. Execution goes through `scripts/usd_validation_executor.py`, which
resolves the canonical concept to a unique registered rule class and fails
closed on anything unknown or ambiguous. Never copy a runtime class name (e.g.
`IndexedPrimvarChecker`) or a category (`Geometry`, `Usd:Performance`) into a
scope note — class names are not unique across providers.

Usd Optimize validator mechanics and operation docs live upstream in
[usd-optimize](https://github.com/NVIDIA-Omniverse/usd-optimize/) and the
prebuilt Usd Optimize package. Resolve guidance from an extracted package
root via `$USD_OPTIMIZE_ROOT`. If no package root
exists, download/extract the published the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`)
package (direct archive URLs are in `references/upstreams/usd-optimize.md`) or
use the package path supplied by the user. To verify a rule's backing
operation, inspect upstream
`source/core/python/omni/scene/optimizer/validators/<module>.py`.

### Usd Optimize rules (default)

| Validator signal | Canonical concept | Backing op | Notes |
|------|------|-----------|-------|
| SceneOptimizerCoincidingGeometryChecker | `spatial_coinciding` | `findCoincidingGeometry` | Analysis-only; prefer `deduplicateGeometry` before destructive deletion. |
| SceneOptimizerColocatedVerticesChecker | `vertex_weld` | `meshCleanup` | Merges colocated vertices. |
| SceneOptimizerDuplicateFacesChecker | `topology_duplicate_faces` | `meshCleanup` | Removes duplicate faces. |
| SceneOptimizerDuplicateGeometryChecker | `geom_duplicates` | `deduplicateGeometry` | Converts identical meshes to USD instances; run per target or sample, never an unbounded whole-stage default. |
| SceneOptimizerDuplicateHierarchiesChecker | _(structural — no mesh concept)_ | `usd-hierarchy-dedupe-candidates` + `apply-restructure` | Use the hierarchy candidate finder + restructure gate, not a direct mesh op. |
| SceneOptimizerDuplicateMaterialsChecker | `material_duplicates` | `optimizeMaterials` | Merges duplicate material definitions. |
| SceneOptimizerEmptyLeafChecker | `structure_empty_leaf` | `pruneLeaves` | Removes leaf prims with no geometry. |
| SceneOptimizerFlatHierarchiesChecker | `structure_flat_hierarchy` | `findFlatHierarchies` → `flattenHierarchy` | Analysis-only signal; fix is the `flattenHierarchy` operation. |
| SceneOptimizerFuzzyDuplicateGeometryChecker | `geom_duplicates_fuzzy` | `deduplicateGeometry` | Same op, different threshold; run per target or sample. |
| SceneOptimizerIndexedPrimvarChecker | `primvar_indexability` | `optimizePrimvars` | Converts to indexed primvars when the result can change the op plan. |
| SceneOptimizerInvisiblePrimsChecker | `structure_invisible` | `removePrims` | Confirm intent before removing — invisibility may be deliberate. |
| SceneOptimizerIsolatedVerticesChecker | `topology_isolated_vertices` | `meshCleanup` | Removes isolated verts. |
| SceneOptimizerMeshDensityChecker | `perf_high_vertex_count` | `countVertices` | Informational; lossless reducers first, `decimateMeshes` only after the upfront tolerance prompt. |
| SceneOptimizerNonManifoldChecker | `topology_manifold` | `meshCleanup` | Skip for visualization-only workflows; run only for simulation-ready intent. |
| SceneOptimizerNormalsChecker | `normals_validity` | `generateNormals` | Regenerates missing/invalid normals; targeted check only. |
| SceneOptimizerPrimitiveFitChecker | `primitive_fit` | `fitPrimitives` | Bounded-loss; requires the tolerance prompt before applying. Highest-value reducer for converted CAD/BIM content. |
| SceneOptimizerRedundantTimeSamplesChecker | `perf_redundant_timesamples` | `optimizeTimeSamples` | Removes redundant samples on animated attributes. |
| SceneOptimizerRtxMeshCountChecker | `perf_rtx_mesh_count` | `rtxMeshCount` | Informational threshold check. Reduce via `deduplicateGeometry` + `flattenHierarchy` + `removeSmallGeometry`. |
| SceneOptimizerSmallMeshChecker | `perf_small_mesh` | `removeSmallGeometry` | Removes meshes below a screen-space threshold. |
| SceneOptimizerSparseMeshChecker | `perf_sparse_mesh` | `sparseMeshes` | Tune density thresholds. |
| SceneOptimizerUnusedUVsChecker | `primvar_unused` | `removeUnusedUVs` | Removes unbound UV sets when the result can change the op plan. |
| SceneOptimizerWindingsChecker | `normals_winding` | `meshCleanup` | Fixes inconsistent face winding. |
| SceneOptimizerZeroAreaFacesChecker | `topology_zero_area_faces` | `meshCleanup` | Removes degenerate faces. |
| SceneOptimizerZeroExtentChecker | `extents_zero` | `removeSmallGeometry` | Fix removes zero-extent meshes. Use `computeExtents` first when the cause is stale metadata. |

### Usd Optimize rules (expensive — only present with `--include-expensive`)

| Validator signal | Canonical concept | Backing op | Notes |
|------|------|-----------|-------|
| SceneOptimizerOccludedMeshesChecker | `spatial_occluded` | `findOccludedMeshes` → `removePrims` | **Two-step detect→act.** Analysis identifies fully-occluded prim paths; feed those to `removePrims`. Runs first in the Phase 4 op chain. Scope to SA `cross_component_pairs` that aren't explicitly transparent (`enclosure_opaque` true or unset; boundary pairs nominated via `candidate_source` hash OR semantics; bbox confirmation-only). Scoped probe runs without approval; only the removePrims deletion is intent-gated. |
| SceneOptimizerFindOverlappingMeshesChecker | `spatial_overlapping` | `findOverlappingMeshes` | Analysis-only. Fix: review and remove/merge in DCC. |

These expensive concepts are `gpu_bound` and Tier 3 in the registry; they must be
scoped to flagged pairs (`paths=` / `OpenMasked`) and run in bounded
subprocesses — never full-stage by default on large CAD/BIM/MEP assets.

### usd-validation-nvidia (OAV) base rules

The full list lives in the upstream `usd-validation-nvidia` package; we
mirror only the concepts that participate in the performance workflow. Many base
rules map onto a Usd Optimize operation — surface the equivalent op so the
user has an automated fix path even when the rule itself is upstream.

**Geometry rules with Usd Optimize operation equivalents:**

| OAV base rule | Canonical concept | Backing op | Notes |
|-----------|------|------------------|------|
| `ExtentsChecker` | `extents_general` | `computeExtents` | Broader than SO `ZeroExtentChecker`. |
| `IndexedPrimvarChecker` | `primvar_indexability` (oav impl) | `optimizePrimvars` | **OAV variant is the slow full audit.** Registry tiers the OAV implementation higher than the Usd Optimize triage one; the executor picks the Usd Optimize impl for performance tuning. |
| `WeldChecker` | `vertex_weld` | `meshCleanup` | Welds colocated verts. |
| `NormalsValidChecker` | `normals_validity` | `generateNormals` | Targeted check only. |
| `ZeroAreaFaceChecker` | `topology_zero_area_faces` | `meshCleanup` | — |
| `UnusedMeshTopologyChecker` | `topology_unused_mesh` | `meshCleanup` | Removes unreferenced points. |
| `ManifoldChecker` | `topology_manifold` | `meshCleanup` | Some topology repairs need DCC work; skip for visualization-only targets. |

**Stage / metadata / external references (safety gates — manual fix, no SO op):**

| OAV base rule | Canonical concept | Notes |
|-----------|------|------|
| `KindChecker` | `kind_metadata` | Fix via `prim.SetMetadata('kind', ...)`. |
| `DefaultPrimChecker` | `layout_default_prim` | Fix via `stage.SetDefaultPrim(...)`. |
| `StageMetadataChecker` | `stage_metadata` | Fix via `UsdGeom.SetStageUpAxis(...)`, etc. |
| `LayerSpecChecker` | `layer_spec_health` | Type/value mismatches in layer specs. |
| `MissingReferenceChecker` | `composition_missing_ref` | Unresolvable references — common on assets flattened elsewhere with absolute paths. High-priority gate for conversions. |
| `MaterialPathChecker` | `material_path` | `info:mdl:sourceAsset` pointing at missing files. |
| `UsdDanglingMaterialBinding` | `material_dangling_binding` | `material:binding` relationships whose targets do not resolve — often a correct leaf name under a stale or renamed root, so the fix is a re-target rather than a re-authoring. Evidence-poisoning: while the bindings dangle, the material axis of the evidence is void — `material_duplicates` reads a false 0 and any dedupe verdict computed against it is unreliable. Put the three-way choice from workflow.md 2c to the user: repair whole-stage BEFORE structuring (a bespoke `Sdf` re-target the agent authors directly — one pass, new output file, source never opened for write), waive and record `safety_gate.status: unresolved` in the report, or halt. |
| `TextureChecker` | `texture_bind` | Texture asset paths on shader inputs that the resolver cannot find. Same safety-gate routing as `material_dangling_binding` — repair, waive-and-record, or halt — but the evidence cost of waiving is lower: an unresolvable path still hashes consistently, so it does not falsify `material_duplicates` the way a dangling binding does. What waiving does cost is that the optimized output ships the same broken paths, so say so in the report. When the referenced file is genuinely absent rather than mis-pathed, authoring cannot fix it and waive-and-record is the honest branch. |
| `NormalMapTextureChecker` | `texture_normalmap` | `UsdUVTexture inputs:file` unresolvable. |
| `TypeChecker` | `type_metadata` | A defined prim with no type name. Fix via `prim.SetTypeName('Xform')` for a grouping prim, `'Scope'` for a pure namespace node. Prims under `/Render` are exempt and the rule skips them. Untyped prims are routine on converted CAD and they cost the descent: a group with no type reads as neither a Gprim nor an Xform, so boundary inference cannot tell a real assembly node from a stray one. |
| `PrimEncapsulationChecker` | `prim_encapsulation` | Two distinct failures: a Boundable nested under a Gprim (mesh inside mesh — the parent's extent cannot be computed and the two cannot have independent visibility), or a connectable (Shader, Material) parented outside a container-like connectable. Fix is a re-parent in the source DCC or a bespoke `Sdf` move; no op does it. Clear this before Phase 2f — the restructure carries the nesting into every prototype it authors. |
| `DanglingOverPrimChecker` | `layout_dangling_over` | A prim with only `over` specifiers that is not the target of any relationship or attribute connection. Fix is to give it a defining specifier (`def`/`class`) or delete it. On a converted or flattened stage it usually means a reference that no longer resolves, so check whether `composition_missing_ref` fired on the same run and treat them as one repair. Speculative-opinion overs are a legitimate authoring pattern; the rule cannot tell them apart, so confirm intent before deleting. |
| `UnicodeNameChecker` | `utf8_paths` | Prim, property, variant-set, or variant names that are not NFC-normalized, or siblings that collide once normalized. Fix is a rename to NFC. Renaming is identity-affecting, so it needs explicit user intent and a re-target of anything pointing at the old path. Colliding sibling names also poison name-based grouping: two names that render identically but hash differently split one repeated group into two, and the descent under-shares. |
| `SubdivisionSchemeChecker` | `subdivision_scheme` | `subdivisionScheme` not authored on a mesh. USD's default is `catmullClark`, which is wrong for tessellated CAD, so the renderer smooths geometry the converter meant to be faceted and every normals or decimation decision downstream is made against the wrong surface. Fix is to author the attribute: `none` when the mesh already carries normals, `catmullClark` when it does not and a smoothed surface is wanted. This rule ships its own `Suggestion` callables for both, but the executor path does not invoke fixers — author the attribute yourself. |
| `RigidBodyChecker` | `physics_rigid_body` | `UsdPhysics.RigidBodyAPI` applied to a non-Xformable prim (RB.003), to an instance proxy (RB.005), or to a prim with a skewed transform (RB.009). RB.005 is a hard constraint on the Phase 2g frontier: marking an ancestor of a rigid body `instanceable=true` makes it an instance proxy and breaks the body. Never choose a share level that encloses a `RigidBodyAPI` prim — share at the rigid-body/link level and reassemble with references, as `workflow.md` 2g already requires for articulated assets. Fix is authoring, not an op. |
| `ValidateTopologyChecker` | `topology_general` | Mesh topology invalid on one or more time samples — face-vertex indices out of range for the points array, or counts that do not add up. No op repairs it: `meshCleanup` welds and removes degenerates, it does not rebuild a broken index array. Fix is in the source DCC or the converter. Selected per target by the sim-ready row, not by the Always row, because its scope policy is `per_target_or_sample` (see the runner README). Treat a firing as a stop: geometry ops on a malformed mesh have undefined results. |

A safety gate never routes to an operation. Every one of them carries
`backing_op: null` in the registry and an integrity test enforces it, because a
gate that silently gained a backing op would start mutating a stage off the back
of a correctness check. Route a fired gate to guidance and an explicit user
decision instead. Waiving is a legitimate branch — detection is not mandatory
repair — but record it in the report's `safety_gate.status` field and state the
evidence cost in the same breath.

For OAV-equivalent fixes, label the op as a Usd Optimize operation (not the
validator's own `--fix` — this repo's validators don't ship a `--fix` mode).

For any signal not in this list, treat it as a **manual fix**. The evidence to
surface is the executor's own output: each entry in `validators[]` of the
`validation-report.json` that `run_scope_note` writes, with its
`canonical_name`, `status`, and `issues` count, plus the issue records the run
produced for that concept — message and prim path, grouped by rule, capped per
`runtime-artifact-token-budget.md`. There is no CSV on this path; the runtime's
CSV reporter and its `Suggestion` column are reachable only through the CLI the
executor's contract rules out. Don't invent fix commands, and don't assign a tier
here — if the concept matters, add it to `validator-concepts.json` and give it a
row above.

---
