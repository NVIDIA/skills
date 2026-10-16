# DEFT OD AOI data contract

Read this before freezing the application policy or admitting new training
images.

## Normalized roles

The application consumes four normalized COCO roles:

| Role | Boxes | May enter training | Purpose |
|---|---:|---:|---|
| KPI | labeled or empty | never | gap queries and checkpoint selection |
| Test | labeled or empty | never | report-only measurement |
| Real | at least one per image | after retrieval and admission | positive mining |
| Clean | exactly zero per image when nonempty | after retrieval and admission | negative mining |

Every COCO document declares exactly one foreground category named `defect`.
Background is implicit. Clean images remain explicit `images` rows with zero
annotations; files absent from the COCO document do not train the detector.
Every bbox must have a nonnegative origin, positive dimensions, and remain
fully within its image dimensions. Boundary overflow is an input-contract
violation; source ground truth is rejected rather than clipped or rewritten.

Each image resolves through `source_path` when present, otherwise
`images_dir/file_name`. Resolved identities must be disjoint across KPI,
test, real, and clean roles. The initializer verifies paths, IDs, boxes,
category names, role-specific annotation counts, and cross-role overlap before
freezing hashes in the policy.

## Canonical KPI metadata

When synthesis is enabled, the resolved KPI image path and the matching
strict-gap `filepath` must resolve to the same underlying file. Using the same
path works, as does a symbolic link to that file.

Every annotated KPI image must carry the exact fields `dataset_id`,
`texture_id`, and `defect_class`, either directly on its image record or under
`deft_od_aoi`. `dataset_id` is the synthesis route/dataset identity;
`texture_id` is the namespaced product or texture identity; and `defect_class`
is the source defect label. Retrieval does not accept alternate field names.
Initialization validates this contract before any embedding or inference work.
Boxless KPI images used only for clean/background queries may omit
`defect_class`. Their `dataset_id` and `texture_id` are not required for query
formation, but should be preserved when available for dataset and object-type
tracking.

Real query artifacts preserve the three canonical fields and add a derived
`pocket` value of `dataset_id/texture_id/defect_class` for near-miss accounting.
Admission preview `per_dataset` counts also use canonical `dataset_id`, so they
report synthesis-route identity rather than optional source provenance.

When synthesis is enabled, each eligible KPI annotation/image pair must also
resolve a real pixel `fn_mask_source`. Synthesis metadata may appear directly
on the image or annotation record or under `deft_od_aoi`; a box never
substitutes for the mask.

The real role must start with at least one boxed image. The clean role may start
with zero images; initialization emits a warning and a typed `UNAVAILABLE`
clean retrieval capability. The clean retrieval role is independent from the
AnomalyGenNext synthesis reference pool.

`synthesis.routes` is an allowlist: a valid ID absent from it skips synthesis
but still participates in the separate real-data DEFT path. A missing ID is
malformed metadata, not a synthesis opt-out. For routed FNs,
`texture_id+defect_class` must match a type declared by that route's recipe and
defect specification.

AnomalyGen clean-reference images come from the synthesis pool and are not the
DEFT `clean` retrieval role. Missing references skip only affected synthesis
FNs; they do not disable real-defect or clean-negative retrieval.

## Artifact-scoped role states

`role_status` is scoped to the artifact that contains it; it is not one shared
status enum. In `query_manifest.json`, it describes retrieval-producer
availability (`READY`, `EXHAUSTED`, or `NO_QUERIES`) and may carry exclusion
counts. In round-robin selection and admission reports, it describes
post-selection outcomes (`SELECTED` or `NO_MATCHES`) and carries
`selected_count`. Consumers must interpret the field using the containing
artifact's contract rather than mixing the two state machines.

Initialization rejects routed boxed annotations with missing route metadata or
a nonexistent `fn_mask_source`. It deliberately does not inspect mask contents;
synthesis preparation owns per-FN mask eligibility.

## Cumulative admission

`admit_deft_od_aoi_coco.py` recomputes similarity from the frozen embeddings,
deduplicates selected crops to source images, rejects previously admitted
sources, and publishes a new binary COCO. Pass `--previous-coco` from
iteration 2 onward. Clean negatives are capped by
`routing.clean_cumulative_cap_per_real`. When synthesis is enabled, pass both
the generated binary COCO and its image root. On the second, synthesis-only
admission pass, also pass `--synthetic-only` with the same-iteration real
admission `train.json` as `--previous-coco`; the preview records
`mining_admission: skipped`. Without that explicit flag, mining admission still
runs. Synthetic admission is capped by
`synthesis.cumulative_fraction_of_total_defects` so that synthetic defects
occupy at most that fraction of the combined real and synthetic defect pool.
Existing records remain unchanged. The legacy
`cumulative_fraction_of_real_defects` key remains readable with its original
synthetic-to-real meaning. Admission also emits `admission_preview.json`. When
overfetched crops do not contain enough novel parent images for a branch target,
it admits the available parents and records the shortfall instead of failing the
iteration.
Real retrieval may become exhausted while clean retrieval continues until the
existing cumulative real count's clean allowance is full.
`admission_report.json.retrieval_admission` records requested parent-image
targets, counts after previously admitted sources are removed, and final
counts after policy caps for both real and clean retrieval. For real retrieval,
the deduplicated count also excludes parents selected for an earlier reason in
the same iteration. This makes source deduplication and clean cumulative-cap
truncation visible without recording the excluded filenames.

The initial admission report calculates remaining capacity after admitting
real/clean images, and emits `SYNTHETIC_ADMISSION_CAP_ZERO` when it is zero.
The stage controller then records a budget skip instead of requesting
iteration synthesis. Missing or invalid capacity evidence is an error.
Positive capacity permits synthesis, but post-generation admission still
enforces quality and cumulative limits; `SYNTHETIC_ADMISSION_CAPPED` reports
otherwise-eligible generated images excluded by the cap. Training is allowed
only when the iteration adds images. Cap formulas and whole-FN allocation
are unchanged; a zero cumulative real count still permits no synthetic
admissions.
