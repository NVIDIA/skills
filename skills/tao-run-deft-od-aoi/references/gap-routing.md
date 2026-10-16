# Gap matching and retrieval routing

Read this when preparing KPI gap jobs or interpreting retrieval requests.

## Dual gap passes

Run `tao-analyze-gaps-od-map` action `object_detection` twice over the same
KPI ground truth and predictions:

| Pass | Confidence | Consumed rows | Route |
|---|---:|---|---|
| Loose | 0.3 | false positives | near miss → real; background-like → clean |
| Strict | 0.8 | false negatives | real and optional synthesis |

Both passes use same-class one-to-one matching at IoU 0.5. An unmatched loose
prediction below IoU 0.05 is background-like. From 0.05 inclusive to 0.5
exclusive it is a near miss. A strict FN may represent a missed object or an
object covered only by a lower-confidence prediction.

The inference threshold remains 0.001 so gap analysis, rather than inference,
owns the loose and strict operating points.

## SigLIP retrieval

Build the candidate cache once. `retrieval.preprocessing.profile` independently
selects the embedding input geometry:

- `square_context` (default) expands the
  longer defect-box edge by the context scale, mean-pad to a square at image
  boundaries, mean-pad non-square clean-grid cells, then bicubic-resize defect,
  query, and clean-grid crops to
  `retrieval.output_size` (default `224`); this setting is read and validated
  only for this profile;
- `tight_context` preserves the newer behavior: expand both box dimensions by
  the context scale, clip the resulting aspect-ratio-preserving rectangle to
  the image, and keep each crop's native dimensions. `retrieval.output_size`
  has no effect on this profile.

For either profile:

- each real annotation produces a 1.5× contextual crop;
- each verified-clean image produces the whole image and a 2×2 grid;
- both roles use the same frozen SigLIP encoder.

COCO and gap boxes use the displayed-image coordinate frame. Retrieval applies
the image's EXIF orientation before reading dimensions or cropping pixels, so
the coordinates must not be transformed a second time.

Query preparation validates the candidate manifest before reuse. The frozen
preprocessing profile, model and model path, context scale, clean-grid layout,
and square output size must match the current policy. A legacy manifest without
`preprocessing_profile` is treated as `tight_context`, matching the geometry
that produced it.

`retrieval.selection.strategy` independently selects how those embeddings are
ranked:

- `round_robin_similarity` (default) ranks candidates per query and advances
  one rank depth at a time in stable query order. Strict-FN and near-miss
  quotas are isolated by `(dataset_id, texture_id, defect_class)` pocket, while
  background-clean queries share the same deterministic round progression;
- `max_similarity` retains global selection by each candidate's maximum cosine
  similarity to any query.

Round-robin selection bypasses the global mining action but runs as an explicit
retrieval step before the stage commit. It reads the complete candidate and
query embedding tables and materializes the same role-specific
`mine_<role>/final_unique_files.parquet` contract as max-similarity. Admission
consumes those committed selections instead of recomputing them. Prior sources,
exact candidate filepaths in `exclude_<role>_candidates.parquet`, duplicate
parents, and candidates below `minimum_similarity` are excluded.
The exclusion parquet is an optional reserved extension point: when present it
must use canonical candidate paths and its declared count must match.
Each pocket reranks with admitted parents excluded through the frozen
`round_robin_refill_overfetch` depths, screens invalid boxes and visual
duplicates against the cumulative admission index, admits every available
novel parent, and records any positive-quota shortfall without failing the
iteration. The following retrieval round handles convergence when no candidates
remain.
Query rounds use the stable gap-derived
`routing_order_key`, not transient hashed query identifiers.
Each near-miss pocket requests `near_miss_real_factor` candidates per query,
bounded by the single canonical `near_miss_real_cap_per_pocket` setting.
Each strict round-robin query uses its gap-row `real_factor`, then the explicit
`--real-factor`, then `routing.round_robin_real_factor_default` (`3` by
default); all strict queries in one pocket must agree. Max-similarity retains
the `routing.real_mine_factor_min` fallback (`1` by default). Annotated KPI
images must provide the canonical `dataset_id`, `texture_id`, and
`defect_class` fields. Optional provenance fields do not define pockets.

For each iteration, crop strict FNs and near-miss FPs as real queries, and
background-like loose FPs as clean queries. Embed queries with the identical
encoder. Invoke `tao-mine-od-images` for max-similarity or
`deft_od_aoi_round_robin_selection.py` for round-robin.
Context crops smaller than 8 pixels on either edge are expanded around the
requested defect center within image bounds. Only source images narrower than
8 pixels require mean-color padding; ordinary crop dimensions remain unchanged.

For `max_similarity`, retrieval is global within the real or clean role.
Provenance metadata does not partition the index. Empty query roles emit no
action. Admission recomputes
maximum cosine similarity from the frozen embeddings, applies the frozen
minimum, deduplicates parent images, and enforces cumulative caps. The mining
request overfetches crop candidates by the frozen factor, while admission
measures capacity in unique parent images after cumulative exclusions. Parent
budgets are targets rather than hard minimums: admission takes up to each
target and writes `admission_preview.json` with desired parents, mined crops,
unique and novel parents, shortfalls, branch counts, and selected parents by
source dataset. A completely empty first admission still fails.

Strict-FN real budgets use the selected factor within the frozen `1..6` range.
Near-miss budgets are computed independently for each frozen
`dataset_id/texture_id/defect_class` pocket as `min(2 * queries, 20)` by default;
this cap is not a per-dataset allocation quota. Clean retrieval remains global
and is capped only by its factor and the cumulative clean-to-real bound.

Pass the prior cumulative COCO when preparing iteration 2 and later so every
already admitted source is expanded to its candidate-crop exclusion set.
Role-specific exclusion parquets are reconciled against candidate crop or
source paths and are never ignored. A role with zero remaining candidates is
typed `EXHAUSTED` while other roles continue. When every role is exhausted and
no synthesis work is pending, the query manifest marks convergence and the
workflow completes without submitting an empty mining request.

For either strategy, candidates may remain while every similarity is below the
frozen minimum. This is reported per role as `NO_MATCHES`, distinct from an
exhausted candidate pool. It does not fail the retrieval stage: synthesis may
still add training images, and a loop without synthesis converges instead of
retraining an unchanged cumulative dataset.

The initial `-1.0` similarity threshold is an explicit calibration policy,
not evidence that all candidates are equally useful. Review retrieval outputs
before freezing a stricter value for a later run.
