---
agent_context: usd-performance-workflow
agent_routes:
  - omniverse-usd-performance-tuning
agent_next:
  - compare-profiles/README.md
freshness: 2026-05-20
version: "0.1.0"
---

# Compare Profiles Contract

Use this page as the docs-class summary for `compare-profiles`. The executable
workflow reference remains
`skills/omniverse-usd-performance-tuning/references/compare-profiles/README.md`.

## Required Inputs

See [compare-profiles/README.md § Required Inputs](compare-profiles/README.md#required-inputs)
for the authoritative input list (paired baseline/after `profile-stage` JSON,
matching mode, same hardware/runtime for full mode, and the change applied
between captures).

## Verdict Thresholds

- **Improvement:** metric improved by more than 5%.
- **Neutral:** metric changed within plus or minus 5%.
- **Regression:** metric worsened by more than 5%.
- **Critical regression:** metric worsened by more than 20%.

Report absolute values and percentages together. A neutral result is not a
failure; it means the measured scene did not materially change for that metric.

## Structural-Only Runs

When the run used quick structural signals and no meaningful before/after timing
or frame metrics were captured, set `workflow_mode: structural_only` in the
report — do **not** invent a verdict value. The `verdict` stays within its enum
(`improved | neutral | regressed | mixed`); use `neutral` when no measured metric
materially changed. The report's `notes` field must say which runtime or access
blocker prevented a stronger performance verdict and must recommend the next
profile capture needed to graduate it.

## Terminal Report Requirement

End-to-end optimization work finishes with both:

- a JSON report that conforms to the `optimization-report` reference's schema (`scripts/optimization-report.schema.json`)
- a Markdown companion summary generated from the same evidence

Do not substitute a chat-only recap or an unrelated `SUMMARY.md` for the
terminal optimization report.

## Regression Handling

See [compare-profiles/README.md § Regression handling](compare-profiles/README.md#regression-handling)
for the authoritative regression-handling steps (name the metric and quantify the
change, correlate with what changed, check known causes, and decide keep/revert/adjust).
