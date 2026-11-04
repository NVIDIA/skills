# Classify and author assets

Classify selected assets children before parents, after their workload-required independent use has
justified a boundary under [structure assessment](structure.md). Having child assets
and no owned geometry determines aggregate structure; it does not justify making
an organizational prim a separate asset. These workflow conventions use
the terminology of [annotated asset structures](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/asset-structure-principles.html#annotated-asset-structures):

| Structure | Owned content |
|---|---|
| Atomic | Geometry with no selected child asset boundaries after interior review |
| Package | Owned geometry together with selected child assets |
| Aggregate | Referenced child assets and assembly opinions; no owned geometry |

Every geometry occurrence belongs to one atomic asset or package. An aggregate's
referenced descendants are expected; unassigned geometry is not. An atomic asset
may use internal geometry references and shared material libraries.

## Model hierarchy and occurrence opinions

Choose `kind` for the composed role, separately from structural type. Equipment
can be a `component` even when its structure is aggregate. Significant parts
inside a component are `subcomponent` occurrences; their standalone definitions
can remain components. Organizational ancestors of models need `group` or a
derived kind such as `assembly`, without becoming separate published assets.
Authoring clears obsolete copied `kind` metadata on copied internal prims, including protected coordinate frames without
geometry, then applies the selected asset and interface kinds. Preserving a prim
and its transforms does not require preserving its imported model kind. Do not turn
every imported model label into `subcomponent`. Selected child assets and referenced
contents keep their declared roles; source layers remain unchanged.

Keep editable mechanisms in the equipment hierarchy, with references to atomic
geometry below their interfaces. Preserve occurrence placement, visibility,
authored animation and domain relationships. Shared geometry must not turn a
required editable descendant into an instance proxy. Instance eligible atomic asset occurrences even when
the current assembly uses a definition only once. Placement remains editable on
the occurrence root; parents with independently editable children remain non-instanceable.

Imported references and instances describe source composition, not required asset boundaries.
Preparation resolves selected interiors into editable definitions while preserving selected child
assets and required internal prims. Assessment merge groups are provisional; preparation
recomputes them on those editable inputs before optimization. Accepted product identities
are checked against supplied libraries before preparation. Confirmed parent-child groups
reuse published definitions directly. Explicit variant contexts are applied during ordinary
input preparation. Other definitions continue through preparation and
the existing reuse checks. The workflow records how many worker inputs were avoided.

The authoring and optimization tools handle transform composition and placement.
They may bake static internal transforms into merged geometry; those original
transform attributes need not survive. Retain transform operations and their order
where authored animation or a required editable attribute depends on them, including
reset-stack behavior and authored time samples. The agent records the required
movement/edit paths, not a transform-preservation checklist.

Record each edit's existing prim/property `path`, human-readable `use` and
`edit_class` (`placement`, `visibility` or `identification`). Placement and
visibility require the corresponding USD capability and an editable output prim.
Identification preserves the named prim/property for its selection or lookup use
and existing identifying opinions; it imposes no transform-frame or editability
requirement. Occurrence labels can differ while sharing geometry. Geometry beneath
a required prim can merge within it. A Mesh used as a placement frame keeps that
frame; fitting remains available beneath a preserved Xform. Preserve empty
attachment frames without adding geometry. Detailed pose/visibility trials and
unrelated-occurrence isolation checks belong in development fixtures, not repeated
whole-scene runtime trials. Unsupported required interfaces remain incomplete.

## Shared materials and semantic libraries

Resolve effective material bindings in occurrence context, including GeomSubsets,
inherited bindings and binding purposes. Keep shader connections, primvars and
resources intact. The workflow creates the shared material library automatically
before geometry workers run: it extracts used networks and their shader dependencies,
removes unused networks, and invokes native `optimizeMaterials` for compatible
duplicates. Asset definitions reference the resulting material prims; do not
create separate copies of their shader networks for each asset. Distinct material
properties and required metadata remain distinct. One writer coordinates this work.

Publication remaps those references to the delivered library. The published
manifest's `output.material_library` and the report identify its final location.
Published material-library filenames use `materials_` followed by the existing
unique publication suffix. Preserve source material prim names and `displayName`
labels when present. A CAD colour label such as `Color_D2D2FF` identifies colour;
it does not establish a physical material such as steel or plastic. Keep those
labels when colour is the only supplied identification. Material names and labels
do not determine equivalence; native material optimization compares the networks.
Check the reported location when inspecting the delivery, rather than the temporary
`working/materials.usdc` file. Resource localization follows
[the persistent-library requirements](library-reuse.md); this does not imply
independent lookup of equivalent materials across unrelated scenes.

Publish multiple separately referenceable definitions in a library for each
established semantic category: robots, fences, pumps, chairs and similar types.
Use the general library when equipment type cannot be established. Optional
`equipment_family` labels group supported product/model relationships within a
category; omit them when evidence is absent. They do not change library partitioning,
entrypoints or geometry-equivalence decisions. Category, structural type and kind
are independent fields. Distinct models in a category retain distinct entrypoints.
Keep the consumer assembly separate from category libraries.

For persistent reuse, follow [the library index requirements](library-reuse.md).
Confirm exact cache hits before optimization across the supplied indexes,
independent of category. The
[lookup helper](usd-optimize-run-operations/scripts/persistent_library.py) returns
nominations requiring confirmation; equal names or bounds do not authorize reuse.
Keep inherited entries read-only and preserve their local coordinate frames.
Changed geometry creates a new output version instead of silently changing other
consumers. Use [remote access](remote-assets.md) for resolver and localization work.

## Working layers and publication

Express geometry in the selected asset's local coordinate frame and retain its
occurrence transform. Replace the original rendered subtree with its reference;
do not leave a second rendered copy. `Sdf.CopySpec` copies authored specs, so the
authoring tool must first resolve the selected composition. Give each definition
units, axis and a usable default prim.

Author an immutable input and separate output for each unique definition. Parent
working layers reference immutable child inputs, so a worker never reads another
worker's partially written output. These temporary files do not determine delivery
packaging. Record output entrypoints before dispatch.

[Category planning](usd-optimize-run-operations/scripts/category_publication.py)
separates independent computation from publication. After native execution and output checks pass,
[publication](usd-optimize-run-operations/scripts/semantic_library.py) rewrites
references to verified outputs and writes each category through one writer.
Within that unpublished layer, the [native sharing helper](usd-optimize-run-operations/scripts/shared_geometry.py)
can share identical remaining geometry across definitions with different child
poses. It resolves explicit owned occurrence paths to editable meshes or
existing sources beneath each asset's `Prototypes` class, submitting each authored source
once. Native `deduplicateGeometry` and scoped `organizePrototypes` place confirmed
sources under the category's `/Library/Prototypes` class. The publisher updates
occurrence path mappings and the written library fingerprints. Editable ancestors,
addressed mesh interfaces, material bindings and independently referenced entrypoints
remain intact. These internal geometry references do not introduce asset boundaries.
This pass compares eligible whole meshes within one writable category layer. It does not recover repeated parts
inside a merged mesh or share geometry across category libraries. Inherited entries remain read-only.
For a common library, publish unique category files under `assets/`, materials
under `materials/`, and supporting layers under `dependencies/`. Preserve existing
files and atomically replace the cumulative `index.json` after new files are
installed. Keep each station’s `after.usdc`, reports and working files in its output
directory. References are relative to final file locations. A failed publication
leaves the previous index usable; unindexed new files may remain. Standalone
publication commits its output directory together. Reopen the consumer and check references, binding remaps, placement and ownership;
a successful copy alone does not establish valid composition.

For portable delivery, follow [the library dependency requirements](library-reuse.md).
Move a successfully published dependency snapshot and reopen it to check references,
materials and resources. Otherwise retain resolvable persistent references and report them.
