# Distribution-Aware Curation

End-to-end workflow where the agent analyzes KPI sample data, proposes
target distributions grounded in model-training best practices, curates
raw data with distribution awareness, and produces a final report with
gap analysis and augmentation recommendations.

> **Want a filtered slice instead of a balanced dataset?**
> If the user says *only*, *just*, *exclude*, or *filter to* a
> specific intersection (for example, "only daytime collisions"),
> use `restrictive-curation.md` instead. That playbook is the
> opposite intent: it discards everything that does not match the
> user's intersection rather than balancing across labels. The
> guardrails below (rare-event floor, edge-case coverage, class
> balance) do not apply to that mode.

---

## Prerequisites

- Completed KPI analysis split run (see `context-understanding.md` Phase 1-2)
- KPI captions available at `{output_clip_path}/metas/v0/*.json`
- Raw (uncurated) video dataset available for the full curation run
- Domain catalog matched or event taxonomy synthesized

---

## Phase 1: Baseline Distribution

Run immediately after KPI analysis completes. The goal is to understand
what the raw data actually contains before proposing any targets.

### Procedure

1. Read all KPI sample captions from `metas/v0/*.json`.
2. For each clip, extract dimensions using the same keyword-based
   labeling from `distribution-analysis.md` Phase 2:
   - `time_of_day`: day, night, dawn_dusk
   - `weather_condition`: clear, cloudy, rain, snow, fog
   - `event_type`: from the domain catalog or synthesized taxonomy
   - `density`: low, medium, high (domain-dependent)
   - `camera_angle`: overhead, high-angle, eye-level, low-angle
   - Additional domain-specific dimensions as needed
3. Count labels per dimension and compute percentages.
4. Save `kpi_baseline_distribution.yaml` in the KPI run directory:

```yaml
baseline:
  source: "kpi_analysis"
  total_clips: 15
  dimensions:
    time_of_day:
      day: 12
      night: 2
      dawn_dusk: 1
    weather_condition:
      clear: 10
      cloudy: 3
      rain: 2
    event_type:
      normal_traffic_flow: 8
      vehicle_to_vehicle_collision: 3
      stalled_vehicle: 2
      pedestrian_in_roadway: 2
```

5. Present the baseline to the user with a brief summary:
   "KPI analysis of 15 clips shows 80% daytime, 13% nighttime. Event
   distribution is 53% normal, 20% collisions. Would you like
   distribution-aware curation for the full dataset?"

---

## Phase 2: Target Proposal

If the user wants distribution-aware curation, propose target
distributions. The agent MUST consider model-training best practices
when formulating proposals.

### Model-Training Best Practices

Apply these principles when proposing targets:

| Principle | Guideline | Rationale |
|-----------|-----------|-----------|
| Class balance | No class should exceed 5x the smallest class | Prevents model bias toward majority classes |
| Rare-event floor | Safety-critical events: minimum 10-15% representation | Improves recall for high-consequence events |
| Edge-case coverage | Night, fog, snow, rain: minimum 10% each if present in raw data | Models must generalize across conditions |
| Baseline grounding | Targets cannot exceed what raw data contains | Cannot curate clips that do not exist |
| Diminishing returns | Beyond 40% for any single class, additional clips add less value | Reallocate budget to underrepresented classes |
| Diversity over volume | A balanced 200-clip dataset often outperforms an imbalanced 500-clip set | Distribution matters more than raw count |

### Proposal Options

Present these options to the user in natural language. Adapt the
specific percentages to the baseline and domain.

**Option A: Balanced**
Uniform distribution across all labels per dimension. Best when the
raw data has enough variety to support it.

**Option B: Baseline-preserving with minimum thresholds**
Keep the natural distribution but enforce floors: no label below 10%
(or 15% for safety-critical events). Best when the natural distribution
reflects real-world frequency and the user wants to preserve that
signal while ensuring minimum coverage.

**Option C: Safety-weighted**
Overweight safety-critical events and adverse conditions (night, rain,
fog) relative to routine observations. Typical split: 40% anomalous /
60% normal (vs. the natural ~80% normal / ~20% anomalous). Best for
safety-focused models where recall on rare events is paramount.

**Option D: Custom**
User specifies their own targets per dimension. The agent validates
against baseline feasibility and warns if targets are unreachable.

### Agent Interaction

Present the baseline, then the options:

> "Based on KPI analysis, your raw data is 80% daytime / 13% night /
> 7% dawn. For event types, 53% is normal traffic.
>
> I can curate the full dataset with distribution awareness. Options:
>
> A) **Balanced**: ~33% day / ~33% night / ~33% dawn per time_of_day.
>    This requires enough night/dawn clips in the raw data.
>
> B) **Baseline-preserving with floors**: Keep the natural skew but
>    ensure night >= 15% and dawn >= 10%. Anomalous events >= 15%.
>
> C) **Safety-weighted**: Boost anomalous events to 40%, night to 25%,
>    adverse weather to 20%. Reduces normal/daytime/clear proportionally.
>
> D) **Custom**: Tell me your target percentages per dimension.
>
> Which approach, or would you like to adjust?"

After the user selects, record the approved targets in
`curation_targets.yaml`:

```yaml
targets:
  strategy: "safety_weighted"
  approved_by: "user"
  dimensions:
    time_of_day:
      day: 0.50
      night: 0.30
      dawn_dusk: 0.20
    weather_condition:
      clear: 0.40
      cloudy: 0.25
      rain: 0.20
      fog: 0.10
      snow: 0.05
    event_type:
      normal_traffic_flow: 0.40
      vehicle_to_vehicle_collision: 0.20
      vehicle_to_motorcycle_collision: 0.15
      stalled_vehicle: 0.10
      pedestrian_in_roadway: 0.10
      collision_aftermath: 0.05
```

---

## Phase 3: Distribution-Aware Curation

### Strategy Selection

The agent selects the best balancing mechanism based on data
characteristics. Present the chosen strategy to the user before
executing.

```text
                         Raw data volume
                              |
                   +----------+----------+
                   |                     |
              Abundant                 Scarce
           (>3x target count)     (<2x target count)
                   |                     |
          Post-curation            Multi-pass with
            filtering             adjusted thresholds
                   |                     |
                   +----------+----------+
                              |
                     Hybrid (per-dimension)
```

**Post-curation filtering** (preferred when raw data is abundant):
1. Run the full curation pipeline on all raw data with standard
   quality thresholds.
2. After curation, classify every output clip across target dimensions.
3. For each dimension, compute how many clips fall into each label.
4. Selectively drop clips from overrepresented labels until the
   distribution matches targets (within tolerance).
5. Prioritize dropping lower-quality clips first (lower aesthetic
   score, lower motion score) to retain the best examples.

**Multi-pass with adjusted thresholds** (preferred when data is scarce):
1. First pass: curate with standard thresholds. Classify output clips.
2. Identify underrepresented labels.
3. Second pass on remaining raw videos: relax quality thresholds
   (lower `aesthetic_threshold`; for motion, either lower
   `motion_global_mean_threshold` or switch `motion_filter` to
   `"score-only"` / `"disable"`) specifically for clips that match
   underrepresented labels.
4. This recovers borderline clips that would normally be filtered out
   but are valuable for distribution balance.

   **Fixed-camera note:** For fixed-mounted camera data (fixed-camera video,
   intersection monitoring), `motion_filter` measures scene activity,
   not camera shake. Low-motion clips represent quiet periods (empty
   roads, idle warehouse aisles) that are often underrepresented but
   important for baseline coverage. Consider using `"score-only"` to
   record motion scores without filtering, or `"disable"` entirely,
   then use the scores during post-curation balancing.

**Hybrid** (default recommendation):
Apply post-curation filtering for dimensions with abundant data,
multi-pass for dimensions where data is scarce. The agent decides
per-dimension.

### Execution Steps

1. **Run full curation pipeline** on raw data using the standard
   config (see `running-pipelines.md`). All stages enabled as appropriate.

2. **Classify curated output.** Read all clip metadata from
   `metas/v0/` and classify each clip across every target
   dimension (same procedure as `distribution-analysis.md` Phase 2).

3. **Compute current distribution** and compare to approved targets.

4. **Apply balancing strategy:**

   For post-curation filtering:
   ```text
   for each dimension:
     for each overrepresented label:
       excess = observed_count - target_count
       sort clips by quality_score ascending
       mark lowest-quality `excess` clips for removal
   ```
   Use `quality_score = aesthetic_score * motion_score.global_mean` when
   both fields exist. If `motion_score` is absent or not a map, fall back
   to `aesthetic_score`; if both are absent, retain the clip unless another
   rule requires removal.
   Resolve conflicts (clip marked for removal by one dimension but
   needed by another) by keeping the clip -- prefer retention over
   removal when dimensions conflict.

   For multi-pass:
   ```text
   for each underrepresented label:
     deficit = target_count - observed_count
     re-run pipeline on unprocessed raw videos with:
       - relaxed aesthetic threshold (e.g., 0.3 -> 0.2)
       - relaxed motion threshold (e.g., 0.1 -> 0.05)
     classify new clips, keep those matching the underrepresented label
     stop when deficit is filled or raw data is exhausted
   ```

5. **Produce balanced output set.** Move or symlink retained clips
   into a `04_balanced_output/` directory alongside the original
   `metas/v0/`.

6. Inform the user of the result before proceeding to the report.

---

## Phase 4: Final Distribution Report

Produce two output files in the pipeline output directory.

### `distribution_analysis.yaml`

Use the existing schema from `distribution-analysis.md`. The YAML
captures the final (post-balancing) distribution, not the pre-balancing
one.

### `curation_distribution_report.md`

Extended version of the standard report template. Includes all sections
from `distribution-analysis.md` plus the following additions.

#### Extended Template

````markdown
# Curation Distribution Report

**Dataset:** `<dataset_name>_<run_timestamp>`
**Clips analyzed:** <total_clips_before_balancing>
**Clips retained:** <total_clips_after_balancing>
**Overall diversity score:** <score> / 1.00
**Strategy:** <balanced | baseline_preserving | safety_weighted | custom>
**Pipeline run:** `<output_dir>`

---

## Executive Summary

<1-3 sentence overview: what strategy was applied, how distribution
improved, what gaps remain.>

**Top improvements:**

1. **<dimension> / <label>** -- was <X>%, now <Y>% (target: <Z>%)
2. ...

**Remaining gaps:**

1. **<dimension> / <label>** -- achieved <X>%, target <Y>% (shortfall: <Z>%)
2. ...

---

## Distribution Comparison

### <dimension_name>

| Label | Baseline (KPI) | Pre-balance | Target | Achieved | Delta |
|-------|---------------|-------------|--------|----------|-------|
| <label> | <kpi_pct>% | <pre_pct>% | <target_pct>% | <achieved_pct>% | <+/- delta>% |
| ... | ... | ... | ... | ... | ... |

**Coverage score:** <before_balancing> -> <after_balancing>

<Repeat for each dimension.>

---

## Balancing Actions Taken

| Action | Clips affected | Rationale |
|--------|---------------|-----------|
| Dropped <N> daytime/clear clips | <N> | Overrepresented (was 80%, target 50%) |
| Retained <N> borderline night clips | <N> | Underrepresented, relaxed aesthetic threshold |
| ... | ... | ... |

**Total clips:** <before> -> <after> (<dropped> dropped, <recovered> recovered via relaxed thresholds)

---

## Gap Analysis

### Gaps addressable by augmentation

| Dimension | Label | Achieved | Target | Shortfall | Recipe | Action |
|-----------|-------|----------|--------|-----------|--------|--------|
| <dim> | <label> | <pct>% | <target>% | <delta>% | `<recipe>` | Augment; weights in YAML |

### Gaps requiring new footage

| Dimension | Label | Achieved | Target | Shortfall | Why not augmentable | Action |
|-----------|-------|----------|--------|-----------|--------------------|---------|
| <dim> | <label> | <pct>% | <target>% | <delta>% | <reason> | Source <N> more clips |

---

## Augmentation Signal

The procedure generates `distribution_analysis.yaml` in the pipeline output
directory. It is a run artifact, not a checked-in skill file. Its weights
account for what curation already achieved and represent the remaining gap
only.

---

## Next Steps

<Agent's recommendations: run augmentation, source more footage,
re-curate with different strategy, or accept current distribution.>
````

---

## Balancing Strategy Decision Tree

Use this to select the right strategy for each dimension:

| Condition | Strategy | Rationale |
|-----------|----------|-----------|
| Raw clips >= 3x target count for all labels | Post-curation filtering | Enough data to be selective |
| Raw clips < 2x target count for some labels | Multi-pass (for those labels) | Need to recover borderline clips |
| Some dimensions abundant, others scarce | Hybrid | Per-dimension optimization |
| User requests maximum retention | Multi-pass only | Avoid dropping any clips |
| User requests highest quality | Post-curation filtering only | Drop low-quality clips aggressively |

---

## Conflict Resolution

When balancing across multiple dimensions simultaneously, conflicts
arise (e.g., a clip is "night + clear" -- dropping it helps balance
time_of_day but hurts weather_condition).

Resolution rules:
1. **Prefer retention over removal.** If a clip is needed by any
   dimension, keep it.
2. **Prioritize safety-critical dimensions.** Event type balance
   takes precedence over environmental condition balance.
3. **Use quality as tiebreaker.** When two clips serve the same
   dimensional role, keep the one with higher quality scores.
4. **Accept imperfect balance.** A 5% deviation from target is
   acceptable if it avoids losing valuable clips.

---

## Example Agent-User Interaction

```text
Agent: "KPI analysis complete. I analyzed 15 sample clips from your
       traffic intersection dataset.

       Baseline distribution:
       - Time: 80% day, 13% night, 7% dawn/dusk
       - Weather: 67% clear, 20% cloudy, 13% rain, 0% fog/snow
       - Events: 53% normal, 20% collision, 13% stalled, 13% pedestrian

       Your raw dataset has 500 videos. Would you like me to curate
       with distribution awareness? I can balance the output to improve
       model training effectiveness."

User:  "Yes, go ahead. I want good coverage of nighttime and collisions."

Agent: "Based on your priorities and model-training best practices,
       I recommend Option C (Safety-weighted):

       - Time: 50% day, 30% night, 20% dawn/dusk
       - Weather: 40% clear, 25% cloudy, 20% rain, 10% fog, 5% snow
       - Events: 40% normal, 25% collision, 15% stalled, 10% pedestrian,
         10% other anomalous

       Night is boosted to 30% (from 13%) and collisions to 25% (from 20%).
       This requires at least ~150 night clips in your raw data.

       Shall I proceed, or adjust the targets?"

User:  "Looks good, proceed."

Agent: [Runs full curation pipeline on 500 videos]
       [Classifies 320 curated clips across dimensions]
       [Applies hybrid balancing: post-filter for time/weather,
        multi-pass for events]

       "Curation complete. Results:

       - 320 clips curated, 245 retained after balancing
       - Time: 51% day, 28% night, 21% dawn (target: 50/30/20)
       - Weather: 42% clear, 24% cloudy, 19% rain, 10% fog, 5% snow
       - Events: 41% normal, 23% collision, 14% stalled, 12% pedestrian

       Night is 2% short of target -- your raw data had limited night
       footage. I recommend augmenting 5-10 night clips via
       osmo-data-enrichment (traffic_intersection recipe).

       Collision coverage is close to target. No augmentation needed.

       Full report saved to curation_distribution_report.md and
       distribution_analysis.yaml in the output directory."
```

---

## Cross-References

- **Distribution analysis (post-hoc only):** `distribution-analysis.md`
  -- use when you already have curated output and want to analyze its
  distribution without re-curating.
- **Restrictive curation (intersection filter):**
  `restrictive-curation.md` -- use when the user wants a filtered
  slice (e.g. "only daytime collisions") rather than a balanced
  distribution.
- **KPI analysis:** `context-understanding.md` -- prerequisite workflow
  that produces the baseline distribution.
- **Augmentation label contract:** `distribution-analysis.md` section
  "Augmentation Label Contract" -- required when generating YAML weights
  for `osmo-data-enrichment`.
- **Pipeline execution:** `running-pipelines.md` -- how to run the curation pipeline.
