# Compare Profiles

## When to Use

Use when comparing matching baseline/after profiles; do not use without paired profile-stage JSON.

## Instructions

See `references/_shared/standard-instructions.md`.

## Pre-flight Checklist

Before computing the comparison verdict, re-read and confirm:

- [ ] Verdict thresholds — see
  [compare-profiles.md § Verdict Thresholds](../compare-profiles.md#verdict-thresholds)
  for improvement/regression bands.
- [ ] `runtime-artifact-token-budget.md` — don't dump raw profile data.
- [ ] Both baseline and after profiles used same measurement method.
## Output Format

See `references/_shared/standard-output-format.md`.

Use this reference after running `profile-stage` both before and after optimization.
It compares the two result sets and reports whether the changes helped, hurt,
or had no measurable effect.

## Runtime context header (every verdict)

Before reporting the verdict, prepend the **compact one-liner** from
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md` (Format B). The verdict is only
reproducible against the runtime that produced it; users reading the verdict
later need to know which Kit / Usd Optimize / usd-validation-nvidia versions
were in effect. Read from the `runtime_context` block in
`<output_path>/setup-preflight.json` (canonical location; see
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md` *Where artifacts live*).

```
[Kit: {runtime_context.kit.application} {runtime_context.kit.version}  |  SO: {runtime_context.usdOptimize.version}  |  AV: {runtime_context.assetValidator.version}]
```

If a profile capture spans more than one runtime (rare — usually means the
baseline was captured before an environment switch), refuse to compare and
ask the user to either re-capture both profiles on the same runtime or
explicitly opt into a cross-runtime comparison. Record the chosen runtime in
the comparison output regardless.

## Purpose

Quantify before/after performance deltas, classify improvements and
regressions, and produce an evidence-backed verdict for the optimization flow.

## Required Inputs

This is the authoritative input list for `compare-profiles`; the docs-class
[compare-profiles.md](../compare-profiles.md#required-inputs) points here.

Two profile results (JSON from `profile-stage`):

- `baseline` — captured before optimization.
- `optimized` (after) — captured after optimization.

Plus:

- Matching profile mode: quick vs quick or full vs full.
- Same hardware and runtime environment for full-mode comparisons, unless the
  user explicitly accepts a cross-runtime comparison.
- Knowledge of the operation chain, restructure step, or validation-driven fix
  applied between the two captures.

## Examples

- "Compare these quick profile JSON files and flag regressions."
- "Did the optimized Kit trace improve runtime frame cost?"

## Comparison metrics

### Quick mode comparisons

| Metric | Improvement means | Regression means |
|--------|-------------------|------------------|
| cold_open_ms | Faster composition | More composition overhead |
| warm_open_ms | Faster cached open with sufficient confidence | Slower cached open only when measured with the same low-noise protocol |
| traverse_ms | Simpler authored scene graph | More authored prims to visit (excludes prototypes) |
| traverse_full_ms | Lower total traversal cost | More prims including prototypes (diagnostic; not a regression if prototype growth is expected from deduplication) |
| attribute_resolution_ms | Fewer/simpler attrs | More fallback opinions |
| transform_ms | Shallower/simpler xforms | Deeper nesting |
| prim_count | Fewer prims (instancing) | Overs or prototype growth |
| prim_count_authored | Fewer authored prims | Authored scene grew (unexpected) |
| layer_count | Fewer layers (packaging) | Layer explosion |

### Full mode comparisons (in addition to quick)

| Metric | Improvement means | Regression means |
|--------|-------------------|------------------|
| fps_mean | More frames per second | Heavier rendering |
| frame_time_mean_ms | Shorter frames | Longer frames |
| hydra_sync_ms | Faster scene population | More Hydra work |
| rtx_render_ms | Faster GPU rendering | Heavier GPU load |
| stage_load_ms | Faster initial load | Slower load |

## Significance thresholds

See [compare-profiles.md § Verdict Thresholds](../compare-profiles.md#verdict-thresholds)
for the canonical improvement / neutral / regression / critical-regression bands
(>5% gain, ±5% neutral, >5% regression, >20% critical) and the "report absolute
values and percentages together" rule.

## Warm-load confidence

Treat `warm_open_ms` as regression evidence only when both inputs followed
`profile-stage`'s Stage-open Timing Protocol: same mode/runtime, at least five
warm samples, and bounded sample spread. If sample metadata is missing, the
after capture ran in the same process that performed optimization, spread is
high, or the delta is within the measured spread, classify the warm-open row as
`neutral` and note that warm-load evidence is inconclusive. Do not list
`warm_open_ms` under `regressions` unless it is corroborated by cold-open,
traversal, layer, prim, or other structural evidence.

## Output

```json
{
  "verdict": "improved | neutral | regressed | mixed",
  "summary": "Load time improved 2.3x. Prim count reduced 29%. No regressions.",
  "metrics": [
    { "name": "cold_open_ms", "before": 545.3, "after": 235.6, "change_pct": -56.8, "verdict": "improved" },
    { "name": "prim_count", "before": 2742, "after": 1941, "change_pct": -29.2, "verdict": "improved" },
    { "name": "total_size_kb", "before": 4957, "after": 4722, "change_pct": -4.7, "verdict": "neutral" }
  ],
  "regressions": [],
  "recommendations": []
}
```

## Regression handling

If any metric regressed >5%:

1. Report which metric regressed and by how much.
2. Correlate with what changed — did file size grow? Did prim count increase?
3. Check for known causes:
   - Size regression after Usd Optimize operations → likely USDC `Layer.Save()` bloat
     (see `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/usd-edit-target-planner/references/output-saving.md`).
   - Load time regression after adding instancing → unexpected, investigate
     prototype count vs instance count ratio.
   - Prim count increase after deduplication → expected (prototype prims added),
     not a regression if instances compensate.
4. Recommend whether to keep the optimization, revert, or adjust.

## Integration with the optimization flow

The full flow with profiling:

```
omniverse-usd-performance-tuning
→ profile-stage (BASELINE)
→ usd-structure-assessment
→ usd-validation-runner (master router; uses skills/omniverse-usd-performance-tuning/references/usd-validation-runner/README.md for tier detail and selected-probe policy)
→ restructure-decision (Phase 2e gate)
→ instancing-readiness (if applicable)
→ Usd Optimize operations / instancing
→ apply-restructure (Phase 5 ref-remap)
→ profile-stage (AFTER)
→ compare-profiles
→ optimization-report
→ report to user with evidence from the generated report
```

(See `skills/omniverse-usd-performance-tuning/references/workflow.md`
for the full canonical 7-phase flow.)

## Rules

- Always compare same mode (quick vs quick, full vs full).
- Always compare same hardware / environment for full mode.
- Report absolute numbers AND percentages — "2.3x faster" is more useful
  than "-56.8%" alone.
- If the optimization verdict is "regressed" or "mixed", do not present it
  as a success. Be honest — the user needs to decide whether to keep or revert.
- A neutral result is not a failure — it means the scene didn't have the
  problem this optimization targets.

## Limitations

- Does not collect profile data; use `profile-stage` first.
- Cannot prove causality without knowing which operations changed the stage.
- Full mode comparisons are unreliable across different GPUs, drivers, or Kit runtimes.

## Troubleshooting

- If modes differ, rerun one capture so both inputs are quick or both are full.
- If a full-mode result looks noisy, repeat the capture and separate startup zones from steady-state zones.
- If a metric is missing from one result, report it as unavailable rather than assuming no change.

## Startup vs steady-state separation

When comparing full mode results, separate zones into:

- **Startup zones** (fire count = 1 or only during init): `createContext`,
  `Collect physical devices`, `compileShaderGroupForDevice`, `Acquire MdlTranslator`,
  `createExtensionManager`, `initialize`, etc.
- **Steady-state zones** (fire count = N, once per frame): `App Update`,
  `hydraRenderViews`, `Renderer::renderViews`, `SceneRenderer-rtx: render`, etc.

Classification rule: if a zone's count matches the frame count (±10%), it's
steady-state. If count = 1 or is proportional to startup (extensions loaded,
devices enumerated), it's startup.

A startup regression combined with a steady-state improvement is a **tradeoff,
not a regression**. Report it as:

> "Scene open is Xms slower (prototype setup), but each frame renders Y% faster.
> Net positive after Z frames of rendering."

Only flag as a true regression if steady-state zones also got slower.
