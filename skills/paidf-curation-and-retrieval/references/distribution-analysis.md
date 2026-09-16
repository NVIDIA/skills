# Distribution Analysis

Analyze completed split output into a dataset-distribution report and a compact
augmentation signal. Use this only after `metas/v0/*.json` exists.

## Inputs

- Per-clip metadata under `{output_clip_path}/metas/v0/`.
- Optional `v0/all_window_captions.json` for bulk caption inspection.
- User-approved targets or a matched domain catalog.
- The exact target `osmo-data-enrichment` recipe, when augmentation is planned.

## Procedure

1. Read each metadata JSON and flatten one row per clip.
2. Extract `qwen_type_classification`, all non-enhanced `windows[]` keys ending
   in `_caption`, `motion_score.global_mean`, `aesthetic_score`, and
   `qwen_filter_score` when present.
3. Label only domain-relevant dimensions. Typical dimensions are
   `time_of_day`, `weather_condition`, `scene_type`, `event_type`,
   `camera_angle`, and `density`. Use `unknown` when evidence is insufficient
   and report the unknown rate.
4. Count observed labels and compare them with explicit user targets. If none
   exist, derive targets from the matched catalog; use uniform known-label
   targets only as a disclosed fallback.
5. Compute per-dimension coverage as
   `1 - JSD(observed_distribution, target_distribution)`. Flag a gap when
   `observed_pct < target_pct - 5.0`. Report the harmonic mean of dimension
   coverage scores as `overall_diversity_score`.
6. Separate appearance gaps that augmentation can address from gaps requiring
   source footage. Cosmos Transfer can alter weather, lighting, time of day,
   and surface appearance; it cannot reliably add/remove actors, synthesize
   events, or change camera geometry.
7. Write the two outputs below beside the pipeline output, then validate every
   count, percentage, path, and probability sum.

## Human-readable output

Write `curation_distribution_report.md` with:

- dataset/run identity, clip count, and overall diversity score;
- the largest observed-versus-target gaps;
- one observed/target/delta table per analyzed dimension;
- augmentable gaps with the selected recipe;
- non-augmentable gaps with concrete source-footage actions; and
- unresolved assumptions and recommended next steps.

Keep the report specific to the current run. Do not embed historical benchmark
runs or copy large sample reports into skill content.

## Machine-readable output

Write `distribution_analysis.yaml` with this compact contract:

```yaml
dataset:
  name: "<dataset_name>_<run_timestamp>"
  total_clips: 0
  source_pipeline_run: "<run_id>"
  output_dir: "<absolute_output_root>"
dimensions:
  <dimension>:
    observed: {<label>: 0}
    target: {<label>: 0.0}
    coverage_score: 0.0
    unknown_count: 0
    gaps:
      - label: "<label>"
        observed_pct: 0.0
        target_pct: 0.0
        deficit: 0.0
recommended_augmentation_variables:
  recipe: "<recipe>"
  <recipe_dimension>: {<allowed_label>: 0.0}
overall_diversity_score: 0.0
recommendations: []
non_augmentable_gaps: []
```

`distribution_analysis.yaml` is a generated run artifact, not a checked-in
skill file.

## Augmentation label contract

Before writing `recommended_augmentation_variables`:

1. Identify the exact consuming recipe.
2. Read that recipe's authoritative
   `augmentation.yaml` `template_generation.variables` mapping.
3. Use the recipe's dimension names and allowed labels exactly; do not rely on
   copied vocabularies in this skill.
4. Drop labels with no recipe equivalent and disclose the loss.
5. Redistribute dropped weight proportionally and require each dimension to
   sum to `1.0`.
6. If no recipe exists, omit augmentation weights and recommend source footage.

Do not copy weights directly into another repository or submit enrichment
without an explicit user request and review of the target configuration.

## Validation checklist

- [ ] Every input clip is counted once; unreadable metadata is reported.
- [ ] Caption fields are selected dynamically from the configured algorithm.
- [ ] Unknown labels and multi-label handling are explicit.
- [ ] Targets are user-approved, catalog-derived, or disclosed as fallback.
- [ ] Coverage calculations and the 5% gap threshold are reproducible.
- [ ] Recipe labels come from the current authoritative recipe configuration.
- [ ] Every augmentation dimension sums to `1.0`.
- [ ] Event, density, and camera-geometry gaps are not misrepresented as
      appearance augmentation.
- [ ] Output paths contain no credentials or sensitive values.

For balanced re-curation use `distribution-aware-curation.md`; for a narrow
intersection slice use `restrictive-curation.md`; for prompt/taxonomy discovery
use `context-understanding.md`. These are route pointers, not nested links.
