<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hierarchy-Dedupe Rewrite Tool - Behavioral Specification

Status: draft
Audience: a coding agent or human implementing a hierarchy rewrite from the
read-only output of `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/usd-hierarchy-dedupe-candidates/references/instance-candidate-finder-spec.md`.

## 1. Purpose

Rewrite repeated local USD sub-hierarchies into shared prototype assets and
references. This is a USD authoring tool, not a Usd Optimize operation.

Use this behavior spec to author the rewrite with `pxr.Sdf`, `pxr.Usd`, and,
when running inside Kit, the same rule-pipeline discipline used by Isaac Sim
Asset Transformer: validate inputs, transform a copy, write explicit outputs,
then reload and validate.

## 2. Inputs

- `input_stage`: USD stage path that opens cleanly.
- `output_dir`: writable directory distinct from the input stage directory.
- `candidate_report`: output from the instance-candidate finder.
- `selected_groups`: candidate group ids or hashes approved by the user.
- `mode`: `internal_reference` or `external_prototype`.
- `material_policy`: `inline_local_external`, `preserve_external`, or
  `block_on_external`. Default: `inline_local_external`.
- `dry_run`: when true, emit a manifest without writing output layers.

The default mode is `external_prototype` for customer-facing digital twins
because it creates explicit assets that can be optimized, versioned, and
validated independently. `internal_reference` is acceptable for a single-file
experiment or when the user explicitly wants to keep one layer.

## 3. Preconditions

- Minimum USD validation passes on the input stage.
- The user has explicitly approved the selected candidate groups and output
  location.
- Every selected group has an `instanceability` verdict of `clean` or
  `review-required` with accepted findings. Do not auto-rewrite groups marked
  `blocked`.
- Candidate groups are non-overlapping after nested-group collapse. If two
  selected groups overlap, keep the parent group and drop the nested child
  unless the user explicitly chooses a narrower scope.
- The rewrite must run against the layer that owns the source prim specs. If a
  candidate's specs are spread across multiple layers, emit `blocked` and ask
  for a flatten/export step or an explicit edit-target plan.

## 4. Outputs

- A new assembly root USD file under `output_dir`.
- For `external_prototype`, one prototype USD file per selected group.
- For `internal_reference`, one prototype namespace inside the new root layer,
  such as `/__HierarchyPrototypes`.
- A manifest JSON with:
  - input stage path
  - output root path
  - selected candidate groups
  - prototype paths or prototype prim paths
  - material networks inlined into each prototype
  - rewritten instance sites
  - skipped candidates and reasons
  - validation result per written file

Never overwrite the input stage in place.

## 5. Rewrite Algorithm

1. Open the input stage and root layer.
2. Copy the input root layer to a new assembly layer.
3. For each selected candidate group:
   - Choose the first approved path as the prototype source unless the user
     selected a specific prototype.
   - Verify every candidate path still exists in the copied assembly layer.
   - Verify the owning layer for each candidate root. If ownership is
     ambiguous, block the group rather than guessing.
4. Materialize the prototype:
   - `external_prototype`: create a new layer, copy the prototype source spec
     to a stable root prim, set `defaultPrim`, save the layer.
   - `internal_reference`: copy the prototype source spec under the prototype
     namespace in the new assembly layer.
5. Resolve material-boundary dependencies for the prototype. See
   [§6 Material Inlining](#6-material-inlining).
6. Rewrite each duplicate instance site:
   - Keep the original root prim path and root placement opinions.
   - Remove authored children and descendant specs that would duplicate the
     prototype contribution.
   - Add a reference to the prototype. Use a relative asset path for external
     prototypes when possible.
   - Set `instanceable = true` only when the candidate report found no local
     child overrides or cross-boundary relationships that would make the site
     invalid as an instance.
7. Save the new assembly layer and prototype layers.
8. Reopen the new assembly root from disk and run the minimum USD validation
   reference owned by `usd-validation-runner`.

Use `Sdf.CopySpec` for spec copying and `Usd.Prim.GetReferences().AddReference`
or direct `Sdf.Reference` list edits for references. Do not flatten the whole
stage unless the user has accepted the loss of composition structure.

## 5a. Nested prototype library is the default (not flat / outermost-only)

Externalized sharing is authored **bottom-up as a nested library**: author each
leaf/subcomponent prototype once, and have **parent prototypes *reference* their
child prototypes rather than inlining them**. Flat or outermost-only sharing
(every shared prototype a self-contained copy of its whole subtree) re-stores the
shared children once per parent and is **explicitly insufficient** — on a large
data-center assembly asset it left disk *larger* than the original, where the
nested library instead recovered a substantial disk reduction. The
`nested_parent_proto` manifest field records the parent→child link for each
nested prototype.

- **Inclusion floor.** Only units at/above the band-resolved minimum-prim floor
  (`MINP`, evidence-seeded ≈20 prims with occurrence ≥2) join the nested library.
- **Sub-floor leaves stay inline on purpose.** Tiny recurring leaves (screws,
  connectors) are **kept inline** (`kept_inline_for_merge`) so a later
  within-prototype mesh-merge can fuse them; instancing them finely bakes in a
  granularity the merge pass would have to tear back down (see the
  instancing-granularity-vs-merge rule). They are the irreducible cross-module
  residual, not a defect.
- **Variant / outlier behavior.** When a structural group is mostly identical
  with a few real value-variants (e.g. `[17, 1]`), author **one prototype for the
  identical majority** and keep the outlier distinct — then **recurse only into
  the outlier's differing branches**, instancing the sub-modules it shares with
  the majority so only its genuine difference stays distinct. Do not author N
  distinct prototypes for N near-identical copies, and do not merge the real
  variant into the majority.

## 5b. Read the existing structure as input (resume, don't restart)

Many inputs arrive **already partway down** the hierarchy — already-instanced
scenes, BIM/CAD exports carrying authored prototypes, references, and `kind`.
Before proposing a rewrite, **inspect the existing composition first**: existing
instancing, prototypes, references, and `kind`. Treat **existing prototypes as
the candidate set at that level** and apply the same value-variant grouping
(one prototype per genuine variant) to them — including **collapsing prototypes
that are byte-identical but were authored separately**. This is the *same*
descent and the *same* dedup entered at the level the asset is already at, not a
separate code path; the boundary and stop rules are unchanged from there.

## 6. Material Inlining

Cross-boundary material relationships are common in CAD and digital twin
assets: duplicate equipment, furniture, or HVAC assemblies often bind geometry
inside the candidate subtree to materials under a shared `/Looks`,
`/Materials`, or similar scope outside that subtree. If those relationships are
left pointing at the source stage, the prototype is harder to validate,
version, move, and optimize independently.

When `material_policy=inline_local_external` (the default), the rewrite tool
must inline local material dependencies into each prototype:

1. For the canonical source subtree, collect authored material bindings and
   UsdShade connections whose targets are outside the selected subtree.
2. Treat material targets as inlineable when the target prim is part of the
   input stage or package and is not an explicit external material-library
   dependency.
3. Build the material-network closure for each inlineable material: the
   Material prim, Shader and NodeGraph descendants, and connected shader or
   nodegraph prims required by that network.
4. Copy each material network into the prototype, preferably under a stable
   child scope such as `/<PrototypeRoot>/Looks`.
5. Rewrite copied geometry bindings and copied shader connections so they
   target the inlined material-network paths.
6. Preserve texture and other asset-valued inputs, but validate that they still
   resolve from the prototype layer. If a relative asset path would stop
   resolving, rewrite it relative to the prototype layer or mark the group
   `blocked` until the dependency move is explicit.

Do not decide material equivalence by material prim name alone. If different
copies bind to different material paths, compare the material-network closure
or split the candidate group. If the material networks differ, skip the group
or leave the affected sites uninstanceable; do not silently collapse distinct
looks.

When `material_policy=preserve_external`, keep external material targets and
record them in the manifest. When `material_policy=block_on_external`, block
any selected group with material bindings or shader connections that cross the
prototype boundary.

## 7. Safety Rules

- Do not collapse candidates based only on display names. Display names can be
  used for sorting or labels, but content identity must come from the candidate
  hash and optional value checks.
- Preserve root transforms at each duplicate site. Root placement is per
  instance; descendant transforms are prototype content.
- Preserve authored metadata on the duplicate root unless it conflicts with the
  reference arc or instanceability.
- Do not rewrite non-material relationships or attribute connections that
  target paths outside the candidate subtree unless the candidate report
  explicitly marks the group as accepted after review.
- If a duplicate has authored child overrides, either keep it uninstanceable
  with a normal reference or skip it. Do not mark it instanceable and silently
  drop overrides.
- Validate every written file before reporting success.

## 8. Reporting

Report:

- number of candidate groups selected
- prototype files or prims written
- material networks inlined or preserved as external dependencies
- duplicate sites rewritten
- sites left uninstanceable and why
- candidates skipped and why
- validation status
- estimated prim-count reduction from the candidate report, clearly labeled as
  an estimate until post-write profiling confirms it

## 9. Relationship to Usd Optimize

After the hierarchy rewrite, Usd Optimize can still be used on the resulting
prototype assets. **Open each prototype as its own root layer** (the edit-target
invariant — see `restructure-mode.md`); SO's edit target must *be* that file's
bytes, never the composed assembly.

- **Per-prototype op chain: `meshCleanup → deduplicateGeometry → computeExtents`.**
  Run it inside each prototype asset (mesh-level dedup is last-mile cleanup, not
  the structuring move).
- **Within-prototype mesh merge (draw-call / scene-graph reduction), when intended.** A mesh merge
  fuses many small meshes into one. It is a **draw-call / scene-graph**
  win, NOT a disk win — merge concatenates geometry (bytes ~= sum, and
  the crate already byte-dedups within a layer, so it can be *worse* for instanced
  geometry). Run merge as a **within-prototype** operation (`merge once, benefit N
  times` across every instance), never *across* an instance boundary. Op-chain
  pattern: **`merge` (within-prototype) → vertex weld where geometry is contiguous
  → `computeExtents`**. The weld tail is **conditional** (a no-op for dispersed
  meshes), must respect **UV seams and hard normals** (weld only coincident verts
  within tolerance), and any bytes it reclaims are credited to **the disk tier via the
  measured weld/dedup source (`disk_win_source: vertex_weld`)** — **never** attributed
  to the merge. See the **merge-eligibility guard** below before fusing anything.
  The **(scope × material) grouping mechanic, the GeomSubset fallback, and the
  archetype-gated merge depth** that execute the manifest `merge` disposition live
  in the dedicated `mesh-merge-rewrite-spec.md` (sibling to
  `point-instancer-rewrite-spec.md`); this section owns the per-prototype op-chain
  placement and the eligibility guard it cites.
- **`pruneLeaves` is stage-level cleanup, not part of the per-prototype chain.**
  Guard it against **unloaded payloads**: a prim whose payload is not loaded
  composes no children, so it presents as an empty leaf and is silently pruned.
  Never run `pruneLeaves` over prims with unloaded payloads (load them first, or
  scope the op away). See `operation-safety.md` § Caveat: `pruneLeaves` on unloaded
  payloads, and `ref-remap-mode.md` § Stage-Level Cleanup.
- **Persist with a compacting `Sdf.Layer.Export(tmp) + atomic replace`, not
  `layer.Save()`.** `Save()` appends without garbage-collecting dedup-orphaned
  arrays and silently grows the file; `Export` recompresses and GCs.
- Run `optimizeMaterials` and other lossless cleanup on the prototype assets or
  new assembly root as appropriate.
- **Lossless dead-data removal (own pass after the geometry chain).** The
  geometry chain shares duplicates but does not shrink disk; the disk lever is
  removing data nothing consumes. The most common case is an **unused UV set**:
  when no material samples a texture coordinate, `primvars:st` (and its indices)
  is dead weight, and primvars usually dominate the bytes — removing it can be the
  single biggest lossless saving. This step is **fail-closed**: delete a primvar
  only after **proving zero consumers** (no `UsdUVTexture` / `UsdPrimvarReader`,
  no MDL, no shader input reads it) and gate by archetype — a **textured or
  scanned asset keeps its UVs**. Persist with the same `Export`-compact step.

### Merge-eligibility guard (bounds coherence)

Only merge **spatially-coherent clusters** of meshes. Do **NOT** merge spatially
**dispersed** geometry: fusing dispersed meshes produces one oversized/overlapping
AABB that **degrades BVH/raytracing** — false ray–box hits and worse culling — so
the runtime gets *slower* even though the draw-call count dropped.

Gate the decision on `merge_bounds_coherence` = merged-prim AABB surface area ÷
Σ(member AABB surface area). A value near `1` means the members were adjacent (a
real draw-call win); a value far above `1` means they were dispersed. **Do not
merge when it would exceed `K` (default `2.0`).** This is the same threshold the
report scoring enforces (the `MERGE_BOUNDS_COHERENCE_MAX` constant in the optimization-report scorer):
a merge that lands `merge_bounds_coherence > K` earns **no scene-graph credit**, so a
dispersed merge is both wrong to perform and unscored.

Merge also requires **weak/none identity** (the disposition matrix's two
identity-destroying rows). Never merge a strong-identity, addressable
`component`/`subcomponent` — that destroys per-part selectability/serviceability
and fails the preservation gate (`merge_identity_class` must be `weak` or
`none`). `merge` (`mergeStaticMeshes`) is a cataloged, intent-gated Usd Optimize
op; it stays available — this guidance bounds *when* it is eligible, it does not
remove it.
