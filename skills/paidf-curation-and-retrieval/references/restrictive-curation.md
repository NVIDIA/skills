# Restrictive Curation (Intersection-Filter Mode)

End-to-end workflow where the user wants the output dataset to contain
**only clips matching a specific intersection of conditions** (for
example: "only daytime collisions", "only night-time pedestrian events
in rain"). The agent translates the request into pipeline-level
filters where possible, applies a post-curation intersection filter
for any dimensions the pipeline cannot enforce, and produces a slice
report instead of a balancing report.

This is the opposite intent from `distribution-aware-curation.md`:
that workflow shapes a *distribution across labels*; this workflow
*excludes* every clip that does not satisfy the user's intersection.

---

## When to use this playbook (vs. distribution-aware-curation)

Use restrictive curation when the user's language signals
*exclusion* / *only* / *just*:

| User says ... | Playbook |
|--------------|----------|
| "Balance the dataset across day / night / dawn." | `distribution-aware-curation.md` |
| "Make sure night and rain are at least 15%." | `distribution-aware-curation.md` |
| "I want a safety-weighted dataset." | `distribution-aware-curation.md` |
| "Only daytime collisions." | **this file** |
| "Just pedestrian events at night, drop everything else." | **this file** |
| "Filter to clips where weather=rain AND event=collision." | **this file** |
| "Build a fine-tuning slice for one specific scenario." | **this file** |

If intent is ambiguous, ask one disambiguation question:

> "Do you want a *balanced* dataset (every condition represented in
> some proportion) or a *filtered slice* (only clips matching your
> intersection, everything else discarded)?"

---

## Best-practice guardrails (and when to bypass them)

`distribution-aware-curation.md` enforces these guardrails by default:

- Edge-case coverage: `night`, `fog`, `snow`, `rain` >= 10% each.
- Class balance: no class > 5x the smallest.
- Rare-event floor: safety-critical events >= 10-15%.

**Restrictive curation deliberately violates these.** A "daytime
collisions only" dataset will have 0% night, 0% non-collision, and is
intended to. Before executing, the agent MUST surface the
consequences and get explicit confirmation:

> "This filter will retain only clips matching `time_of_day = day` AND
> `event_type IN {vehicle_to_vehicle_collision, ...}`. Expected
> outcome:
>
> - ~80% of curated clips will be discarded.
> - The resulting dataset will not generalize to night, dawn, or
>   non-collision scenes.
> - This is appropriate for fine-tuning a specialist model, not for
>   training a general-purpose model.
>
> Confirm to proceed?"

Do not run the filter without an explicit "yes" / "proceed" from the
user.

---

## Phase 1: Capture the filter

Convert the user's natural-language request into a structured
intersection filter and save it next to the run config.

### Filter schema (`restrictive_filter.yaml`)

```yaml
filter:
  mode: "intersection"            # all conditions must match (AND across dimensions)
  approved_by: "user"
  description: "Daytime vehicle-to-vehicle collisions only"

  include:                        # OR within a dimension, AND across dimensions
    time_of_day: ["day"]
    event_type:
      - "vehicle_to_vehicle_collision"
      - "vehicle_to_motorcycle_collision"
      - "vehicle_to_pedestrian_collision"
    weather_condition: ["clear", "cloudy"]   # exclude rain/snow/fog
  exclude:                        # explicit exclusions (applied after include)
    scene_type: ["indoor"]
```

Semantics:

- A clip is **kept** iff for every dimension in `include`, the clip's
  label is in the listed set, AND for every dimension in `exclude`,
  the clip's label is NOT in the listed set.
- Empty / omitted dimensions are unconstrained.
- Within a dimension, list entries are OR-ed.

Pre-flight validation (before running anything heavy):

1. Run KPI baseline analysis (see `context-understanding.md`) and
   confirm at least one of the requested labels exists in the raw
   data per dimension. Warn loudly if a requested label is at 0% in
   the KPI sample, because the full curation will likely also yield
   zero.
2. Estimate slice size: `slice_pct ~= product(include_pct[dim])` over
   the included dimensions, treating each KPI dimension as
   approximately independent. Surface the estimate:

> "KPI baseline shows ~80% day, ~20% collisions, ~85%
> clear-or-cloudy. Estimated slice size: ~14% of curated output. On a
> 500-video raw set yielding ~320 curated clips, expect roughly 40-50
> clips after filtering. Continue?"

If the estimated slice is below ~30 clips, recommend either relaxing
the filter or sourcing more raw footage.

---

## Phase 2: Push the filter down to the pipeline (when possible)

Filtering at the pipeline layer is preferred over post-curation
filtering because it skips expensive captioning / embedding work on
clips that will be discarded anyway. Some dimensions can be enforced
upstream; others cannot.

| Dimension | Pipeline-level enforcement | Notes |
|-----------|---------------------------|-------|
| `event_type` | `video_classifier: true` + `video_classifier_use_custom_categories: true` + `video_classifier_allow: [...]` | Drops clips whose classifier label is not in the allow list; runs after split, before captioning. |
| `event_type` (negative) | `video_classifier_block: [...]` | Inverse: drop clips matching the block list. |
| `quality` floor | `aesthetic_threshold`, `motion_filter` | Standard quality filters, not new. |
| `time_of_day` | None (no upstream classifier for time-of-day) | Filter post-curation from caption keywords. |
| `weather_condition` | None | Filter post-curation from caption keywords. |
| `scene_type` | Partially via `video_classifier` if categories include it | Otherwise post-curation. |
| Domain-specific labels | Partially via `video_classifier` custom categories | Otherwise post-curation. |

### Concrete pipeline override (cookbook-style `split.yaml`)

For "daytime collisions only" the agent should emit a split-config
override that pushes `event_type` filtering into the pipeline:

```yaml
pipeline: "split"
input_video_path: "<raw_path>"
output_clip_path: "<output_path>"

# ... standard split / caption / embed config ...

# -- Restrictive event filter (push event_type intersection into pipeline) --
video_classifier: true
video_classifier_use_custom_categories: true
video_classifier_allow:
  - "vehicle_to_vehicle_collision"
  - "vehicle_to_motorcycle_collision"
  - "vehicle_to_pedestrian_collision"
video_classifier_rejection_threshold: 0.5
```

Notes:

- `video_classifier_use_custom_categories: true` is REQUIRED
  whenever an allow / block list is supplied.
- The classifier label space must match labels used elsewhere in the
  filter; reuse names from the cookbook's
  `classification_events.yaml` to stay consistent.
- For `time_of_day` / `weather_condition` filtering inside the
  pipeline, an option of last resort is to write a custom
  `captioning_prompt_text` that emits a deterministic token (e.g.
  `EXCLUDE_REASON: night`) when the clip does not match, then
  post-process. This is brittle and rarely worth doing -- the
  post-curation script in Phase 3 is simpler.

---

## Phase 3: Apply the post-curation intersection filter

For dimensions the pipeline could not enforce (`time_of_day`,
`weather_condition`, etc.), filter the curated output by reusing the
keyword labeling from `distribution-analysis.md` Phase 2.

### Procedure

1. Read all `metas/v0/*.json` from the curation output.
2. For each clip, label every dimension named in
   `restrictive_filter.yaml` using the same keyword extraction logic
   as `distribution-analysis.md` (caption text +
   `qwen_type_classification`).
3. Apply the filter:
   ```text
   for clip in curated_clips:
     keep = True
     for dim, allowed in filter.include.items():
       if clip.label[dim] not in allowed:
         keep = False
         break
     if keep:
       for dim, blocked in filter.exclude.items():
         if clip.label[dim] in blocked:
           keep = False
           break
     if keep:
       retained.append(clip)
   ```
4. Move or symlink retained clips into a `04_restricted_output/`
   directory alongside the original `metas/v0/`.
5. Record per-clip drop reasons (which dimension failed) in
   `restrictive_filter_log.csv` for the report.

### What to do with quality scores

Unlike distribution-aware curation, restrictive curation does NOT
drop clips for quality reasons unless the user asks. If two clips
both match the filter, both are kept; quality is reported but not
used as a tiebreaker. Rationale: the user's intent is "give me
everything that matches," not "give me a curated balanced subset."

If the user explicitly asks for a quality floor on top of the filter
("daytime collisions, but only the high-quality ones"), apply
`aesthetic_threshold` / `motion_global_mean_threshold` upstream
rather than dropping post-hoc, so the pipeline does not waste GPU
time on clips that will be discarded for either reason.

---

## Phase 4: Slice report

Produce one output file per restrictive run. The schema differs from
the distribution-aware report: there is no "balancing actions"
section because no balancing was performed.

### `restrictive_curation_report.md`

```markdown
# Restrictive Curation Report

**Dataset:** `<dataset_name>_<run_timestamp>`
**Filter:** <one-line summary, e.g. "time_of_day=day AND event_type IN {collision*}">
**Mode:** intersection
**Pipeline run:** `<output_dir>`

---

## Filter Applied

| Dimension | Mode | Labels |
|-----------|------|--------|
| `time_of_day` | include | day |
| `event_type` | include | vehicle_to_vehicle_collision, vehicle_to_motorcycle_collision, vehicle_to_pedestrian_collision |
| `weather_condition` | include | clear, cloudy |
| `scene_type` | exclude | indoor |

## Slice Size

- Curated clips (pre-filter): <N_curated>
- Retained clips (post-filter): <N_retained>
- Retention rate: <pct>%
- Pipeline-level drops (video_classifier): <N_pipeline_drops>
- Post-curation drops: <N_post_drops>

## Drop Reasons (post-curation)

| Dimension that failed | Clips dropped |
|----------------------|---------------|
| `time_of_day` | <N> |
| `weather_condition` | <N> |
| `scene_type` | <N> |

## Slice Coverage

How well the retained clips cover the *included* labels (within the
slice). This replaces "distribution comparison" -- there is no
target distribution to compare against, only a sanity check that
every included label has at least some representation.

| Dimension | Label | Retained count | Retained share within slice |
|-----------|-------|----------------|----------------------------|
| `event_type` | vehicle_to_vehicle_collision | <N> | <pct>% |
| `event_type` | vehicle_to_motorcycle_collision | <N> | <pct>% |
| ... | ... | ... | ... |

**Empty buckets** (included label with zero retained clips): list
them explicitly so the user can decide whether to source more data
or relax the filter.

## Recommendations

- If `<N_retained>` is below the user's training-set minimum,
  recommend (a) relaxing one dimension, (b) sourcing more raw
  footage, or (c) running augmentation on the slice.
- If any included label has zero retained clips, the upstream raw
  data does not contain that combination -- relaxing or sourcing is
  required; balancing cannot help.
- For augmentation on a restrictive slice, point users at
  `osmo-data-enrichment` with weights derived from
  `distribution-analysis.md` run *on the slice itself*.
```

---

## Example agent-user interaction

```text
User:  "I want only daytime clips with vehicle collisions for fine-tuning."

Agent: "That's a restrictive (intersection-filter) curation, not a
       balanced one. To confirm:

       - include time_of_day = day
       - include event_type IN {vehicle_to_vehicle_collision,
         vehicle_to_motorcycle_collision, vehicle_to_pedestrian_collision}
       - everything else is discarded

       From the KPI sample (15 clips): ~80% day, ~20% collisions.
       Estimated slice: ~16% of curated output, roughly 50 clips
       from your 500-video raw set.

       Consequences: the resulting dataset will not generalize to
       night, dawn, or non-collision scenes. This is fine for a
       specialist fine-tune, not for a general model.

       Proceed?"

User:  "Yes."

Agent: [Emits a cookbook-style split.yaml with
        video_classifier_allow set so the pipeline drops
        non-collision clips during processing.]
       [Runs the pipeline on 500 videos.]
       [Pipeline drops ~256 non-collision clips upstream;
        outputs 64 collision clips with metadata.]
       [Applies post-curation filter for time_of_day=day from
        caption keywords; drops 12 night/dawn clips.]
       [Writes 04_restricted_output/ (52 clips) and
        restrictive_curation_report.md.]

       "Restrictive curation complete:

       - 500 raw videos -> 320 split clips -> 64 after pipeline
         classifier -> 52 after post-curation time_of_day filter
       - All retained clips match the filter (intersection of day +
         collision)
       - Empty bucket: vehicle_to_pedestrian_collision (0 clips
         retained). Raw data did not contain a daytime
         pedestrian-collision example. Source more footage or
         relax to include night.
       - Full report: restrictive_curation_report.md"
```

---

## Cross-references

- **Balancing intent (the opposite playbook):**
  `distribution-aware-curation.md` -- when the user wants a
  *balanced* output, not a *filtered slice*.
- **Dimension labeling (shared with both playbooks):**
  `distribution-analysis.md` Phase 2 -- the keyword extraction logic
  used to label `time_of_day`, `weather_condition`, etc.
- **KPI baseline:** `context-understanding.md` -- prerequisite for
  the slice-size estimate in Phase 1.
- **Pipeline-level event filtering:** `video-curation.md` ->
  Qwen Video Classifier section, and the `traffic-video-analytics`
  cookbook for a concrete `video_classifier_allow` example.
- **Pipeline execution:** `running-pipelines.md` -- how to run the curation
  pipeline once the override config is emitted.
