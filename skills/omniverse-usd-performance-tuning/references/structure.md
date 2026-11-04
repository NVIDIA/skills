# Discover asset boundaries

Use the intended workload to decide what must remain independently placeable,
replaceable, configurable or movable. Factory-scale use is the default when no
other workload is specified. The [factory structuring guide](https://docs.omniverse.nvidia.com/vfi/latest/guide/factory-level-structuring.html)
provides the equipment, assembly and editable-interface model.

## Inspect and select asset boundaries

Open the selected configuration with payloads loaded. Include hidden geometry;
visibility can change with animation. Record missing dependencies as incomplete
inspection. Do not enumerate alternate variants. Preserve the source layers and
the variant selections recorded at the start of the run.

Tools summarize relevant prims children before parents, including referenced and
instanced interiors. Reuse cached summaries for shared definitions while retaining
occurrence context. Use schema, topology and valid bounds to narrow expensive
comparisons. Missing bounds are unknown, not evidence that geometry is absent.

### Boundary decision checklist

An asset boundary selects a prim whose assigned content will be extracted into a
separately referenceable asset definition. Set `disposition: "retain"` to select
that boundary; this does not preserve its internal hierarchy unchanged. Set
`disposition: "internal"` to include the region within its enclosing asset.
Preserving an editable prim or property does not itself select an asset boundary.

Use the candidate summary to answer this checklist. Record one boundary decision
and concise reason per reviewed subtree or group; do not write an explanation for
each mesh. Missing optional metadata does not prevent a decision.

| Question | Evidence to use | Decision |
|---|---|---|
| Which equipment, service-system section or architectural element is used as a unit? | Prim name/path, `displayName`, `assetInfo`, specific CAD/BIM attributes and assembly context | Select its root as an asset boundary when the workload calls for independent placement, replacement or configuration. Keep the complete equipment boundary when selecting its internal mechanical bodies as child assets. Use existing meaningful service sections for factory visualization. |
| Which internal mechanical bodies need separate boundaries? | Joint body targets, animated relative motion and descriptive mechanical hierarchy identifying particular robot links, sliding rails or other bodies, including static poses | Keep complete equipment as a reusable asset. Select its meaningful rigid bodies as child assets, including static source poses. Both equipment assemblies and their bodies can share definitions; each occurrence preserves its own body placements. Merge compatible static geometry within each body. A suggestive part name, arbitrary CAD grouping or repeated meshes alone does not establish a mechanical body. |
| Which additional internal prims or properties must remain available? | The exact path and its workload-required placement, visibility or identification use | For additional edits inside a selected body, record the exact prim/property path and explicitly requested or directly implied workload use in `required_occurrence_edits`. Child-asset occurrence roots already provide placement; do not duplicate declarations solely to preserve their positions. Preserve an internal frame when the requested control belongs within a body. Identification requires a stated selection or lookup use. Part names alone do not require extra controls. |
| Does this asset contain the intended geometry? | Assigned geometry count, selected child assets, unassigned geometry and flagged misplaced/combined parts | Keep adequate subtrees. Request regrouping or native splitting only for an identified ownership problem. |
| Does its geometry call for fitting before merging, or consolidation of its static interior? | Pipe/conduit or other simple-shape geometry, equipment context and workload requirements | Apply the [representation decision](operations.md#conditional-execution-order) and record `geometry_representation`; being static or atomic does not settle this choice. |
| What repeats, and what differs? | `product_review_groups` with authored identification clues and complete member paths; member labels and child summaries; confirmed comparison results | Review a product group together and submit common boundary decisions through `source_prim_paths`; separate exceptions. Review corresponding interiors once when accepted product identity or content confirmation establishes correspondence. Ask which cited model, part or configuration values distinguish the interchangeable product; name those values in the existing reason for `product_identity`. Descriptions echoed in exporter metadata remain one clue. Use content confirmation when product identity is unresolved; see [persistent reuse](library-reuse.md). |
| Which equipment category should this asset be published in? | Prim names, `displayName`, `assetInfo`, named CAD/BIM attributes and assembly context | Choose the most specific supported equipment category. An unknown model does not require a general category. |
| Is an equipment family already supported by explicit evidence? | Product/model identifiers or explicit copy relationships | Optionally record `equipment_family`; otherwise omit it. Explain the basis in the existing reason. Family membership neither creates a boundary nor confirms geometry reuse. |
| Does this asset contain child assets selected by boundary decisions, and does it also own geometry outside those child assets? | Selected child-asset IDs and assigned geometry | Let tools derive the [atomic, package or aggregate structure](asset-authoring.md). Resolve contradictions or incomplete ownership. Child prims alone are not child assets. |

Review parents before children. Select the complete equipment, meaningful
service-system section or architectural element with `retain`. Use
`decision_scope: "prim"` when internal child boundaries remain undecided and review
them next; use `subtree` when interior review is complete. Record required internal
edits in the same decision. If children were reviewed first, use
`evidence.enclosing_boundaries` to select the complete equipment root before
completing assessment. Select that root using the equipment evidence in the
checklist and its required use as a unit; enclosing organizational Xforms do not
become assets merely because they contain selected children. An asset without child assets is atomic. Atomic means no child assets,
not one mesh or one physical part. Use `internal/prim` to continue through an
organizational prim when a meaningful asset boundary has not yet been found.
Do not automatically descend to individually named parts. Tools classify child
assets before parents and handle placement, transform baking and geometry comparisons.

Interpret German and English CAD labels, including compounds and umlaut spellings
(`Tuer`, `Schraenke`), using assembly context. Keep original product identifiers,
dimensions and handedness significant; manufacturer and copy labels nominate
related equipment, not identical geometry.

Combine labels with hierarchy context: opaque children beneath a product-labelled
ancestor may be ordinary internal geometry; related product-labelled siblings may
be separate occurrences. A level or discipline label does not establish equipment.
Similar dimensions nominate repetition but do not prove matching geometry. Compare
asset-local dimensions with the same units, purposes and USD time code, separately
from world-space position. Rotated copies can have different world-axis-aligned
bounds. Missing bounds mean unknown; do not demand a geometry scan to fill every row.

In `candidates.json`, `evidence.product_review_groups` batches members with shared
product IDs, authored labels or alphanumeric label tokens. Page it with
`successor_assessment.py show --workspace OUTPUT --candidate ID --field evidence.product_review_groups`.
Use each group’s `source_prim_paths` for a common decision when its member evidence
supports the same conclusion. Groups may overlap; submit each prim once and split
exceptions into separate decisions. Interpret station and occurrence tokens from
the observed naming convention while preserving model, dimensions, cut length,
revision and handedness. Scripts keep complete labels and do not strip suffixes.
`evidence.members` supplies the member context and children; `repetition_nominations`
and `confirmed_reuse` distinguish candidates from confirmed matches. Schema-based review groups are explicitly marked
`review_batch_only_not_equivalence`. Read each member's labels and exceptions;
a representative's counts or children do not describe every member. Early reuse
nominations are not confirmed matches. Fingerprints validate evidence freshness;
they are not explanations of duplication. Request specific property values or
spatial facts only when they could change a decision. Visual inspection is not
required or available to the target agent.

For example, a factory-visualization workload could
use this hierarchy. Labels describe structural type, not USD model kind:

```text
Building - aggregate
└─ Level 01 - aggregate, if independently loaded or delivered
   ├─ Chiller - package
   │  ├─ Pump assembly - atomic
   │  └─ Internal geometry
   ├─ Complete robot - aggregate
   │  ├─ Base link - atomic
   │  └─ Moving links - atomic
   ├─ Piping section - atomic
   │  └─ Static pipe and fitting geometry
   ├─ Electrical conduit system - aggregate
   │  ├─ South service zone - atomic
   │  └─ North service zone - atomic
   ├─ Wall section - atomic
   ├─ Floor section - atomic
   └─ Ceiling section - atomic
```

A complete robot with geometry outside its selected child assets is a package.

Use existing meaningful service-system and architectural sections for factory
visualization. Static pipes and fittings within a section are internal geometry
unless the workload establishes a separate asset requirement. If a section also
contains a valve selected as a child asset for required independent configuration,
that section becomes a package. Apply the same criteria to conduit and duct
sections. Do not invent partitions or regroup adequate subtrees to match the example.
A level, discipline or product-family label alone does not establish an asset
boundary; assess what its subtree represents in the workload.

For each selected asset boundary, record its independent use, supporting prim or property
paths and a concise reason. Distinguish observed facts, user requirements and
domain inference. Account for each service-system section and architectural
element under its meaningful owner; do not collect unrelated infrastructure into
one package merely because it shares a building ancestor.

Tools collect authored animation and addressed relationship/connection targets.
Review unresolved semantic questions using the identified parts and their assembly
context; record additional requirements by exact prim/property path and use.
Use the boundary checklist to select mechanical bodies, including static poses.
Authored animation remains preserved by the existing tools regardless of
additional editability decisions.
An internal attachment frame or required identification prim remains within its
asset; geometry beneath it may be optimized while that prim stays addressable.
Record an unresolved requirement when a required use cannot be satisfied.
[Asset authoring](asset-authoring.md) owns structural type, USD model kind and
preservation behavior. Tools handle placement and animation preservation.

## Reconstruct ownership

Keep the existing hierarchy when it already represents the chosen owners. Regrouping
is not a prerequisite for optimization or library publication. Request reconstruction
only when source organization prevents the intended geometry ownership:

| Source condition | Action |
|---|---|
| A subtree already represents the asset | Keep it |
| Its geometry is scattered under unrelated parents | Regroup the prims under the accepted owner |
| One mesh contains geometry belonging to different assets | Use a supported native `splitMeshes` mode, then assign the resulting pieces |

For misplaced parts, inspect attachment relationships, assembly metadata and
mounting context before spatial evidence. Bounds shortlist possible owners;
proximity alone does not settle ownership. A fastener connecting two bodies has
one geometry owner and can retain relationships to both bodies.

For regrouping, `owner_prim_path` names the physical owner. When that owner is
missing, set `create_owner: true` in the reconstruction proposal to create one
`Xform` immediately beneath an existing defined, editable parent. This does not
create missing ancestor chains or overwrite a prim. Leave the flag absent for an
existing owner. The tools preserve placement and inherited state, reject cycles
and namespace collisions, then return the new owner for explicit boundary and
structural classification. Creating an Xform alone does not establish an asset.

After submitting reconstruction decisions, run
`scripts/successor_reconstruction.py --workspace OUTPUT`.
The controller batches disjoint regroup requests into one authored output.
Overlapping or dependent requests remain sequential; each request retains its
outcome. It invokes [native splitting](usd-optimize-run-operations/scripts/native_reconstruction.py)
or [prim regrouping](usd-optimize-run-operations/scripts/prim_regroup.py) and returns
a new assessment workspace. The regroup adapter calls
`regroup(source_layer, output_layer, memberships, create_owners=...)`, where
`memberships` maps disjoint source prim roots to destination owner paths. It copies
the local layer stack into an isolated output, retains references and payloads,
and moves subtrees with `Usd.NamespaceEditor`; it does not flatten the composed
stage. The result contains `status`, input/output paths, `root_path_map` and
`created_owner_paths`. The controller translates the agent's proposals into these
arguments; agents do not maintain descendant-by-descendant mappings. Continue in that returned
workspace; the previous one is superseded. It retains the original source and
ordered root-path mappings for profiling and remapping required interfaces.
Regroup preservation is tested in development; execution does not produce or
replay per-property preservation checks. Later phases check operation records and
written composition as described in [verification](verification.md#written-output-checks).
Read the returned output paths and operation results. Do not assume native
output names or use mesh-count reduction to judge a split. Disconnected pieces
and material subsets do not automatically become separate assets.

In decision JSON, use `operation: "splitMeshes"` with `parameters: {"splitOn": 0}`
for disconnected vertices, or `parameters: {"splitOn": 1}` for material GeomSubsets.
The controller derives the prim paths from `source_prim_paths`.
It operates on editable polygon meshes and preserves ancestor animation. Animated
mesh data, subdivision surfaces, nested meshes and unresolved relationship remapping
remain unsupported. Another native mode needs its own preservation checks; do not
write a custom face-cutting algorithm.
If the available modes cannot separate the intended parts, record the precise
unsupported case after testing applicable options. Reinspect changed regions and
remap required properties before proceeding. A failed reconstruction leaves its
dependent work incomplete while independent regions can continue.

## Confirm reuse

When `persistent_indexes` are supplied, inspect `persistent_nominations` beside
candidate evidence. Each row names the source prim, matching product/model
identifier and published layer/prim entrypoint. A child's nomination applies to
that child, not its containing equipment. These are unconfirmed clues; retain
independent boundaries and required edits. Record accepted product identity with exact source facts and a substitution reason when supported; otherwise preparation confirms published source identity regardless of category; see [persistent reuse](library-reuse.md). If the workload explicitly permits small constant-colour rounding substitutions, set `allow_constant_colour_rounding: true`; preparation records the accepted difference and preserves occurrence placement. For source-content matching without that authorization, effective appearance must match exactly.


Authored clues nominate candidates. Accepted product identity authorizes representation
substitution, not a claim of identical source geometry or corresponding internal mesh
paths. Content-based reuse follows the existing source-identity checks and explicitly
authorized enclosing-child normalization. See [persistent reuse](library-reuse.md)
for confirmation requirements.

Resolve reuse of child assets selected by boundary decisions before merging the remaining
interior geometry. Repeated meshes alone do not establish asset boundaries or
exclude geometry from merging. Internal references and instances can remain
inside an atomic asset.

## Record decisions

Select the stage entry point as the root asset boundary. For a factory or building layout,
record `structural_type: "aggregate"` with the layout reason: the root composes
equipment and building assets without owning geometry. A spatial subset of a
layout, such as one floor, roof or site zone, is still a layout. Assign its geometry
to the physical owners selected above; do not change the layout root to a package
just to absorb that geometry. An equipment-only
source can instead have an atomic or package root according to its interior.
Organizational children need no separate asset boundary. Record regrouping or splitting for misplaced geometry
before marking its terminal wrapper internal.

Use `internal` for an organizational prim and `retain` for an asset. With
`decision_scope: "prim"` (the default), its interior remains unresolved: review continues through its child groups.
Use `decision_scope: "subtree"` with `retain` to select an asset and complete its
interior review together, or with `internal` to complete a region inside its owner. Record the evidence and reason for that scope;
do not use it to skip units whose independent use is still undecided.
`source_prim_paths` can group disjoint subtrees with the same reviewed conclusion.
Selected asset boundaries and protected interfaces still receive their own
review. Required editable prims remain authored even when their boundary decision
is internal. Terminal geometry wrappers require no separate mesh decisions.

Run `scripts/successor_assessment.py inspect --source INPUT --workspace OUTPUT --request REQUEST_JSON`. `REQUEST_JSON` is a JSON file containing the workload and operation policy.
Record the workload and [physical geometry budget](operations.md#physical-geometry-budget),
including any user override or explicit exact-geometry requirement. The tool selects
the successor policy and writes the current source context, inventory, grouped boundary
reviews, definitions, occurrences and unresolved decisions.

The CLI returns counts and artifact paths; complete evidence remains in `candidates.json`. List candidate groups with `scripts/successor_assessment.py show --workspace OUTPUT --offset 0 --limit 10`. Use `show --workspace OUTPUT --candidate ID` to list its fields, then `--field evidence.all_members` for its summary or `--field evidence.members --offset 0 --limit 10` for members. Follow `next_offset` to inspect the remaining members; a page is not evidence for unreviewed members. Dot-separated fields can select nested evidence, including a member index.

Submit decisions with `scripts/successor_assessment.py decide --workspace OUTPUT --decisions DECISIONS`.
Use `decision-help.json` to fill `decision-template.json` under `decision-schema.json`.
Copy current source context and candidate fingerprints, then judge reviewed members;
the template supplies no boundary answers. Submit a separate file to survive template
regeneration. The adapter validates current evidence/configuration and
preserves earlier decisions. During boundary review, unchanged source evidence is reused;
current decisions and required edits are evaluated again. Progress reports the assessment
pass, stage scan counts, and evidence reuse separately from boundary review.
Tools own membership and generated paths. Agent records supply disposition,
independent use, reason, evidence paths, semantic category or unknown, and
required occurrence edits, `geometry_representation` for the selected operation order (omission uses `merged_meshes`), optional `product_identity` for accepted interchangeable products (key, reason, asset-root frame and exact evidence; format in `decision-help.json`), plus optional `equipment_family` when assigned using the
checklist above. Unreviewed members stay pending. Source or configuration changes
invalidate corresponding evidence.

Continue with [asset authoring](asset-authoring.md) when ownership and interior
review are complete. Classification cannot substitute for reviewing the contents.
