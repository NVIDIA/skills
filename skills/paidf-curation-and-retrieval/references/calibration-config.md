# Calibration Configuration (No KPI Samples)

Mandatory interview workflow when the agent must emit a pipeline
config but no KPI run output exists and no KPI sample videos are
available for inspection. The agent CANNOT skip this interview by
inferring defaults from a one-line user request -- too many
pipeline decisions silently bake in guessed thresholds.

This document is binding: **the agent must complete Phase 1 before
emitting any `*.yaml` pipeline config when the calibration workflow
applies (i.e. there is no KPI baseline to start from).**

---

## When this applies

The calibration workflow applies when ALL of these are true:

1. The user wants a config / wants to run a pipeline.
2. There is no KPI run output directory the agent can read
   (no `metas/v0/*.json` from a prior discovery run).
3. The user has not provided sample videos the agent can inspect
   with `ffprobe` or run a small KPI discovery pass on.

If KPI sample videos ARE available -- even just one or two -- prefer
`context-understanding.md` Phase 1 "With KPI Videos" instead. A
30-minute KPI discovery run produces dramatically better defaults
than this interview.

If KPI run output already exists, read it first and use the
distribution-aware playbook (`distribution-aware-curation.md`) or
the standard `configuration-decision-tree.md`.

---

## Trigger phrases (examples)

Activate this workflow only when the user wants a Cosmos Curator
pipeline config **and** there is no KPI run output **and** no
inspectable sample videos. Domain-specific examples:

- "Configure cosmos-curator for this video folder; I have no KPI
  output and cannot share sample clips yet."
- "Generate a split pipeline YAML for traffic-video-analytics
  footage with no prior discovery run."
- "Start PAIDF curation with no KPI samples to inspect."

Do **not** activate this workflow (route elsewhere) when:

- KPI run output exists (`metas/v0` JSON from a discovery run) —
  use [distribution-aware-curation.md](distribution-aware-curation.md)
  or [configuration-decision-tree.md](configuration-decision-tree.md).
- Sample videos are available for `ffprobe` or a short KPI pass —
  use [context-understanding.md](context-understanding.md)
  Phase 1 "With KPI Videos".
- The user only asks generic first-time setup ("what do I run?")
  or "configure the pipeline" without stating that KPI output and
  samples are both missing — ask whether KPI output or samples
  exist, then route. Do not treat those phrases as calibration
  triggers by themselves.

If the prompt matches the no-KPI-no-samples case, run Phase 1
below before emitting anything.

---

## Phase 1: Mandatory interview

The agent MUST get answers (or an explicit "use defaults" with
disclosure) for every question in this section before writing the
config. Group the questions in one or two messages -- do not
ask them one at a time.

### A. Inputs (blocking, no safe default)

| # | Question | Why it blocks |
|---|---------|---------------|
| A1 | Where are the videos? Local path or S3 URI? | `input_video_path` must be set. |
| A2 | Roughly how many videos and what total duration? (10s / 100s / 1000s of videos; minutes / hours / days of footage) | Drives `limit`, batch sizes, expected runtime, and whether to start with calibration. |
| A3 | Can you point at one or two representative files so I can run `ffprobe` on them, or share their resolution / FPS / codec / typical duration? | Drives splitting parameters, captioning sampling FPS, and whether super-resolution is worth enabling. |
| A4 | Camera type -- fixed-mounted (roadside / stationary dashcam), handheld / body-worn, in-vehicle moving, or mixed? | Drives `splitting_algorithm` (fixed-stride vs TransNetV2) and motion filter strategy (motion = activity vs camera shake). |
| A5 | Where should output go? Local path or S3? Are presigned S3 URLs needed? | `output_clip_path`, `input_presigned_s3_url`, `output_presigned_s3_url`. |

### B. Domain (blocking; catalog match can satisfy this)

| # | Question | Why it blocks |
|---|---------|---------------|
| B1 | What domain is this? (Traffic / fixed-camera video / warehouse / construction / retail / parking / logistics / workplace safety / autonomous-vehicle / general) | Drives `captioning_prompt_variant`, classifier event taxonomy, and whether a built-in catalog applies. |
| B2 | Are there specific observable events you care about identifying? (e.g. collisions, falls, spills, blocked access, unsafe proximity, item handling) | Drives whether to enable `video_classifier` and what `video_classifier_allow` should be. |

Try a catalog match against `.agents/references/catalogs/*.yaml`
following the procedure in `context-understanding.md`. If a catalog
matches with >= 3 keyword overlaps, the agent can skip B2 and use
the catalog's `events` list as the starting taxonomy (still confirm
with the user).

### C. Goal & scope (blocking)

| # | Question | Why it blocks |
|---|---------|---------------|
| C1 | What is the end goal? (Captioning only / captioning + filtering / dataset for fine-tuning / dedup of an existing set / sharding for training) | Drives which pipelines to run (`split` only, or `split` -> `dedup` -> `shard`). |
| C2 | Do you want a balanced dataset across conditions, or a filtered slice (only X-type clips), or just "all valid clips with captions"? | Drives whether to follow `distribution-aware-curation.md`, `restrictive-curation.md`, or neither. |
| C3 | Do you need embeddings? (Required for dedup; optional otherwise.) | Drives `generate_embeddings`, `embedding_algorithm`. Defer to `false` if uncertain -- can be added later. |

### D. Hardware (blocking)

| # | Question | Why it blocks |
|---|---------|---------------|
| D1 | GPU type and count? (e.g. 1x L40S, 2x H100, 8x A100) | Drives captioning algorithm choice (FP8 vs full precision, vLLM async vs single-GPU), batch sizes, and worker counts. |
| D2 | Available system RAM? | Drives `SHM_SIZE` (Docker `--shm-size`). Default `24gb`; reduce to ~8-16gb on small hosts; raise to 32-64gb only if RAM permits. |
| D3 | Disk space available for output? | Quick sanity check: split output is roughly 1-3x input size before compression. |

### E. Calibration vs. full run (blocking)

| # | Question | Why it blocks |
|---|---------|---------------|
| E1 | For the first run, do you want a calibration pass (5-10 videos, no filters, ~10 minutes on one GPU) before committing to the full dataset? | The recommended default is **yes**. The calibration pass replaces the missing KPI step. See Phase 3. |

### F. Optional / skippable (only ask if relevant)

These are recommended-but-skippable. The agent surfaces explicit
defaults and the user can say "use defaults":

| # | Question | Default if skipped |
|---|---------|-------------------|
| F1 | Quality strictness: strict / moderate / permissive | `permissive` -- all filters disabled for the calibration run; tighten after inspecting output. |
| F2 | Are there overlaid TV graphics / news-style banners / watermarks? | `false` -- `artificial_text_filter: "disable"`. Enable only if the user confirms. |
| F3 | Do clips have unusually low resolution that would benefit from super-resolution upscaling? | `false` -- `super_resolution: false`. Adds significant GPU cost; only enable when source res is < 540p AND a 720p/1080p output is required. |
| F4 | Multi-camera mode? (Synchronized clips from multiple angles of the same scene.) | `false` -- `multi_cam: false`. |
| F5 | Custom captioning prompt vs catalog default? | Catalog default. |

---

## Phase 2: Emit the config with mandatory disclosure

After the interview, emit the config AND a structured disclosure
block that lists every field as one of:

- **inferred** -- derived from a user answer or `ffprobe` output.
- **catalog** -- inherited from a built-in domain catalog match.
- **defaulted** -- safe default applied because the user did not
  answer or said "use defaults"; flagged for review.
- **deferred** -- intentionally left at upstream parser default,
  not relevant for the calibration run.

The disclosure schema is markdown, sits next to the config, and is
read back to the user before any pipeline run starts.

### Disclosure template (`calibration_disclosure.md`)

````markdown
# Calibration Config Disclosure

**Config emitted:** `<path/to/config.yaml>`
**Calibration mode:** <yes | no>
**Estimated calibration runtime:** <X minutes on Y GPUs>

## Inferred from your input

| Field | Value | Source |
|-------|-------|--------|
| `input_video_path` | `/data/traffic-cams` | A1 |
| `splitting_algorithm` | `fixed-stride` | A4 (fixed-camera video) |
| `fixed_stride_split_duration` | `30` | A3 (12-min average duration) |
| `captioning_algorithm` | `qwen3_vl_30b_fp8` | D1 (1x L40S, 48GB) |
| `captioning_prompt_variant` | `default` | catalog: traffic_safety |

## Defaulted -- review before production

| Field | Default | Why defaulted | Action |
|-------|---------|---------------|--------|
| `motion_global_mean_threshold` | `null` (disabled) | No KPI baseline; calibration run uses `motion_filter: score-only`. | Set after inspecting `motion_score.global_mean` percentiles in calibration output. |
| `aesthetic_threshold` | `null` (disabled) | No KPI baseline; calibration run scores but does not drop. | Set to ~4.5 after inspecting aesthetic-score distribution in calibration output. |
| `captioning_sampling_fps` | `2.0` | Upstream default; appropriate for most footage at 24-30 fps. | Reduce to 1.0 for low-motion footage; raise to 4.0 only for very fast scenes. |

## Deferred -- not running in calibration

| Field | Value | Why deferred |
|-------|-------|--------------|
| `video_classifier` | `false` | Wait until taxonomy is confirmed against calibration captions. |
| `enhance_captions` | `false` | Adds an LLM pass; only worth running once the base captioning quality is verified. |
| `super_resolution` | `false` | Source resolution adequate (1280x720 confirmed via ffprobe). |
| `generate_embeddings` | `false` | No dedup planned for first pass. Re-enable when running `dedup`. |

## Open questions you did not answer

| Question | Default applied | When to revisit |
|----------|----------------|-----------------|
| F2 (overlaid graphics) | filter disabled | If calibration captions complain about banners or scoreboards, enable `artificial_text_filter`. |

## Required follow-ups

1. Run the calibration pass:
   ```bash
   make run-pipeline CONFIG_FILE=<path/to/config.yaml>
   ```
2. Inspect output: `<output_clip_path>/metas/v0/*.json`,
   `<output_clip_path>/summary.json`.
3. Return to me with: a) whether captions are domain-appropriate,
   b) typical motion / aesthetic score ranges, c) whether any
   pipeline stage failed or warned.
4. I will then promote the config to a full run and apply the
   threshold tuning above.
````

The disclosure is non-optional. If the user says "just give me the
config, skip the disclosure," the agent should still emit it but
trim the prose -- the table of "defaulted" fields must remain so
the user can see what was guessed.

---

## Phase 3: Calibration run (the substitute for KPI when none exists)

The calibration run replaces the missing KPI step. It is small,
cheap, and produces enough output for the agent and user to tune
real thresholds.

### Calibration config delta

Start from `configs/split_calibration.yaml` (committed alongside this
document) and override:

| Field | Value | Rationale |
|-------|-------|-----------|
| `limit` | `5` to `10` | Process only a handful of videos. |
| `splitting_algorithm` | `fixed-stride` | Complete coverage; same reasoning as KPI discovery. |
| `fixed_stride_split_duration` | `30` | 30-second clips give dense coverage of every video. |
| `motion_filter` | `"score-only"` | Score every clip; do not drop. Lets the user see the motion distribution. |
| `aesthetic_threshold` | `null` | Score every clip; do not drop. |
| `vlm_filter` | `"disable"` | No semantic filtering yet. |
| `video_classifier` | `false` | Taxonomy not validated yet. |
| `artificial_text_filter` | `"disable"` | Defer until user confirms presence. |
| `enhance_captions` | `false` | Defer until base captions are verified. |
| `generate_embeddings` | `false` | Defer until dedup is needed. |
| `perf_profile` | `true` | Surface per-stage timing for the disclosure follow-up. |

This config is intentionally lean: only captioning runs, with
filters in score-only or disable mode so nothing is dropped that
the user might want to look at.

### Post-calibration agent procedure

After the calibration run completes, the agent does the work the
KPI run would have done:

1. Read `metas/v0/*.json` and aggregate per-stage scores.
2. Compute distributions:
   - `motion_score.global_mean` -- 5th, 25th, 50th, 75th, 95th
     percentiles. Recommend `motion_global_mean_threshold` set near
     the 10th percentile of "kept" clips, i.e. drop the bottom decile.
   - `aesthetic_score` (if `clip_extraction_target_res` produced
     scoreable frames) -- recommend `aesthetic_threshold` near the
     10-15th percentile.
3. Read 5-10 captions end-to-end. Verify the prompt variant
   produced domain-appropriate output. If captions look generic,
   propose switching to a custom prompt or a closer catalog.
4. Read `summary.json` for stage failure counts; if any stage
   failed > 5%, surface it as a follow-up rather than silently
   moving to production.
5. Emit a "promotion config" -- the calibration config with `limit`
   removed, filters tightened to the recommended thresholds, and
   any catalog updates applied. Reuse the same disclosure schema.

---

## Phase 4: Promotion to full run

Only after Phase 3 has produced reviewed thresholds may the agent
emit a full-run config. The promotion config differs from the
calibration config in:

- `limit: 0` (process all videos)
- `motion_filter: "enable"` with the tuned threshold
- `aesthetic_threshold` set to the tuned value
- `video_classifier: true` if the user confirmed an event
  taxonomy in Phase 3
- `generate_embeddings: true` if dedup or shard pipelines will run
  downstream
- `enhance_captions: true` if base captions were validated and the
  user wants the LLM refinement pass

The disclosure for the promotion config should reference the
calibration run output directory so the user can see where the
tuning came from.

---

## Anti-patterns

The agent must avoid these calibration-setup mistakes:

| Anti-pattern | Why it is wrong |
|--------------|-----------------|
| Emitting a full-run config from a one-line user request | Bakes in guessed thresholds across 6+ filter knobs. |
| Skipping the calibration run "because the user is in a hurry" | A 10-minute calibration always saves more time than it costs (catches misconfigured paths, wrong codec, etc.). |
| Defaulting `super_resolution: true` "to be safe" | Adds significant GPU cost; only useful when source resolution is genuinely too low for downstream training. |
| Defaulting `video_classifier: true` with a guessed allow list | Drops clips silently; the user may not realize the classifier is filtering. |
| Defaulting `motion_filter: "enable"` with the upstream threshold on fixed-camera video | Drops every "quiet period" clip even though those are often the baseline-coverage clips the user wanted. Use `"score-only"` or `"disable"` until calibrated. |
| Setting `enhance_captions: true` before validating base captions | Wastes LLM tokens refining captions whose underlying VLM choice is wrong. |
| Omitting the disclosure block | Hides which fields are guessed; the user has no signal that the config is unreviewed. |

---

## Cross-references

- **Config template:** `configs/split_calibration.yaml` -- safe-defaults
  YAML the agent should copy as the calibration starting point.
- **With KPI samples:** `context-understanding.md` Phase 1 -- prefer
  this whenever even a few sample videos are available.
- **Decision tree (post-calibration):** `configuration-decision-tree.md`
  -- the standard knobs for promoting the config from calibration
  to production.
- **Run mechanics:** `running-pipelines.md` -- how to actually invoke
  `make run-pipeline`, set credentials, monitor progress.
- **If the goal is a balanced dataset:**
  `distribution-aware-curation.md`.
- **If the goal is a filtered slice:** `restrictive-curation.md`.
