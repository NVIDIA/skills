# Persistent library reuse

Set `persistent_indexes` in the workflow request to the library index files to inspect.
The tools load them once per batch. Existing libraries remain immutable.

## Execution order

During assessment, explicit product/model identifiers, authored names and CAD label
tokens nominate published entries in `candidates.json`. Existing indexes without
label keys retain content-based lookup. These clues support grouped review; they do not automatically confirm reuse or establish asset boundaries. On retained assets, record `product_identity` only when cited values identify the interchangeable product and its relevant configuration for the workload. Identify the actual model, part, revision, dimensions or handedness values in the existing reason; a generic assurance of interchangeability is not an explanation. A genuine product identifier may occur in a prim name, `displayName`, `assetInfo` or a custom attribute. A generic description repeated in exporter metadata remains one clue, not independent corroboration. Use the nominated library product key when accepting its identity; otherwise preserve the distinguishing values in the key. Manufacturer alone, family membership, bounds alone, or an equipment category or description do not establish product identity; omit `product_identity` and use existing content matching. Missing identifiers and different category labels do not exclude reuse.

Before optimization, [persistent confirmation](usd-optimize-run-operations/scripts/persistent_reuse.py)
first confirms an accepted product key against published product metadata, without hashing geometry content. The identity record asserts corresponding asset-local frames and explains the accepted representation substitution; occurrence placement does not change the key. Publication integrity, units, required relative interfaces and authored animation still apply. The result is `product_identity_hit`, not a claim of identical source geometry.

If no accepted product entry confirms, prepared-input confirmation uses the existing original-source content and root-frame lookup. Newly published assets retain these content identities alongside any accepted product metadata, so a different product key does not prevent exact or normalized reuse. Reuse the existing published
definition regardless of current decimation settings, tolerance or representation
preferences. An explicit library update is separate from scene optimization.
Confirmation checks completed publication, required relative prim paths, dependency
fingerprints and composed reopening. A hit reports the existing disposition
`exact_cache_hit` and skips optimization. Current published geometry supplies output
accounting; historical mesh maps are unnecessary.

When the workload explicitly permits small constant-colour rounding substitutions,
set `allow_constant_colour_rounding: true` in the request. This also enables source
identity comparison for one static enclosing Xform child whose name and transform
represent occurrence placement. Root display names remain on occurrences; material-subset
labels may differ when their exact face memberships and effective materials agree.
Geometry and internal transforms still match exactly;
required internal editable prims, child asset boundaries and animation exclude this
comparison. Constant RGB values in [0,1] are compared in the same 8-bit interval
(`floor(channel * 255 + 1e-5) / 255`); textures, opacity and other shader inputs remain
exact. Matching material display labels and HOOPS material-description metadata are
not required. Do not enable this option from a generic optimization request.

A confirmed `normalized_cache_hit` keeps the existing library appearance, skips native
optimization and records the authorization, original/canonical colour values and
placement correction in its reuse receipt. The workflow preserves the occurrence
root placement and applies the frame correction above surviving geometry. A failed
match follows ordinary optimization; the agent must not assert equivalence from names.

The workflow also consolidates accepted product identities within the current run, preserving required interfaces, animation and occurrence placement. For content-based reuse, it consolidates matching leaf inputs when source identity,
root frame, units, operation scopes and required editable interfaces agree. Exact
identity includes authored animation, relationships and appearance; it supports
multiple children without colour-rounding authorization. Authorized normalized
identity follows the enclosing-child and appearance rules above. One representative
is optimized and published while each occurrence retains its placement and interfaces.
Cached parent selections take precedence. `within_run_reuse` receipts record original
and representative definition IDs and the identity basis. Supplied-library reporting
counts confirmed definition lookups separately from distinct referenced library entries.

## Index and USD metadata

Index format 2 contains one entry per independently published asset: library identifier,
prim path, exact cache key, optional normalized source key, optional accepted product key and product identification clues
(explicit IDs and namespaced label/token keys). Library records hold file locations, fingerprints, coordinate/time conventions
and dependency identifiers. Each dependency location and fingerprint is stored once.
Internal shared mesh sources do not create asset index entries.

The authoritative `usdPerformance:reuse` customData dictionary on each definition root
holds lookup keys, entry identity, category, asset structure, source signatures and publication completion. Accepted product identity, cited source facts and substitution reason are retained in `product_identity_json`; the index projects only its key. Opted-in normalized identities retain their source frame and compact colour decision evidence before geometry optimization. Original
product metadata remains in USD. Operation history, assessments, measurements and
source-to-output maps stay in run artifacts; they are not persistent lookup data.
`persistent_library.py::rebuild_index()` reconstructs the index from published USD and
its dependencies without input stages, prior reports or optimization.

Readers accept format 1 through the same source-identity checks. Libraries lacking source signatures require republication from saved source matching records; their combined policy hashes cannot recover source identity. Republished
legacy definitions receive compact metadata in an owned copy. Prior files are never
rewritten.

## Placement and delivery

Preserve local coordinate frames and current required occurrence edits. Root and
ancestor transforms, including animation and transform-stack resets, participate in
exact input identity. Arbitrary hierarchy remapping and independently re-tessellated
shape recognition are outside this matcher.

Confirmed current-format entries remain external and read-only by default, with relative library and index paths. Retain the directory layout when moving the delivery; dependencies inside supplied libraries must also be relocatable. Set
`portable_output: true` for a localized dependency snapshot. Publish library files and
their index together, with stable definition entrypoints. Reopen consumers and check
composition and placement through the ordinary workflow checks.

MDL modules remain at their original absolute locations in nonportable outputs.
USD dependency tools do not inspect imports or texture paths inside MDL modules.
The index records this incomplete resource closure; exact cache confirmation excludes
those definitions. Portable publication with MDL dependencies remains incomplete until
an MDL-aware collector can preserve that closure.
