# Skill Benchmark: cdf-event-video-generation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-event-video-generation`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 15 evaluation tasks (15 positive)
- Dataset digest: `sha256:777897553a6f3d3a1d11c7a4c5e3dfebbcb28e750b5631f1caf902969d0f7cb3` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 81.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 100.0% (+20.0 points) | 60.0% → 96.7% (+36.7 points) |
| Correctness | 36.0% → 97.3% (+61.3 points) | 30.7% → 82.7% (+52.0 points) |
| Discoverability | 77.5% — baseline ran, but no comparable score was available; uplift unavailable | 76.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.0% → 83.9% (+38.9 points) | 36.6% → 67.2% (+30.6 points) |
| Efficiency | 84.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,878,043 | 12,044,434 | -9,166,391 | -76.10% | skill 15/15; base 15/15 |
| claude-code | evg-anomaly-distribution-positive | 207,193 | 254,498 | -47,305 | -18.59% | skill 1/1; base 1/1 |
| claude-code | evg-defect-routing-negative | 296,226 | 1,092,690 | -796,464 | -72.89% | skill 1/1; base 1/1 |
| claude-code | evg-e2e-external-positive | 256,727 | 284,462 | -27,735 | -9.75% | skill 1/1; base 1/1 |
| claude-code | evg-e2e-internal-positive | 201,979 | 5,458,092 | -5,256,113 | -96.30% | skill 1/1; base 1/1 |
| claude-code | evg-extra-distribution-key-negative | 163,141 | 286,466 | -123,325 | -43.05% | skill 1/1; base 1/1 |
| claude-code | evg-generation-only-unsupported | 71,250 | 276,702 | -205,452 | -74.25% | skill 1/1; base 1/1 |
| claude-code | evg-local-upload-positive | 196,787 | 194,356 | +2,431 | +1.25% | skill 1/1; base 1/1 |
| claude-code | evg-missing-output-clarification | 113,262 | 420,115 | -306,853 | -73.04% | skill 1/1; base 1/1 |
| claude-code | evg-no-dataset-negative | 112,605 | 153,524 | -40,919 | -26.65% | skill 1/1; base 1/1 |
| claude-code | evg-no-secret-leak-security | 29,706 | 29,465 | +241 | +0.82% | skill 1/1; base 1/1 |
| claude-code | evg-not-iaa-attribute-search-routing-negative | 322,419 | 1,936,481 | -1,614,062 | -83.35% | skill 1/1; base 1/1 |
| claude-code | evg-not-iaa-clothing-routing-negative | 282,439 | 357,718 | -75,279 | -21.04% | skill 1/1; base 1/1 |
| claude-code | evg-platform-selection-positive | 70,794 | 120,534 | -49,740 | -41.27% | skill 1/1; base 1/1 |
| claude-code | evg-status-results-positive | 340,358 | 966,040 | -625,682 | -64.77% | skill 1/1; base 1/1 |
| claude-code | evg-vda-routing-negative | 213,157 | 213,291 | -134 | -0.06% | skill 1/1; base 1/1 |
| codex | All cases | 2,551,059 | 3,691,775 | -1,140,716 | -30.90% | skill 15/15; base 15/15 |
| codex | evg-anomaly-distribution-positive | 91,207 | 13,733 | +77,474 | +564.14% | skill 1/1; base 1/1 |
| codex | evg-defect-routing-negative | 330,443 | 389,782 | -59,339 | -15.22% | skill 1/1; base 1/1 |
| codex | evg-e2e-external-positive | 212,302 | 162,216 | +50,086 | +30.88% | skill 1/1; base 1/1 |
| codex | evg-e2e-internal-positive | 89,793 | 360,202 | -270,409 | -75.07% | skill 1/1; base 1/1 |
| codex | evg-extra-distribution-key-negative | 531,435 | 85,076 | +446,359 | +524.66% | skill 1/1; base 1/1 |
| codex | evg-generation-only-unsupported | 119,284 | 98,847 | +20,437 | +20.68% | skill 1/1; base 1/1 |
| codex | evg-local-upload-positive | 131,280 | 72,492 | +58,788 | +81.10% | skill 1/1; base 1/1 |
| codex | evg-missing-output-clarification | 92,636 | 335,387 | -242,751 | -72.38% | skill 1/1; base 1/1 |
| codex | evg-no-dataset-negative | 145,232 | 423,093 | -277,861 | -65.67% | skill 1/1; base 1/1 |
| codex | evg-no-secret-leak-security | 30,173 | 13,370 | +16,803 | +125.68% | skill 1/1; base 1/1 |
| codex | evg-not-iaa-attribute-search-routing-negative | 147,275 | 29,723 | +117,552 | +395.49% | skill 1/1; base 1/1 |
| codex | evg-not-iaa-clothing-routing-negative | 190,094 | 342,111 | -152,017 | -44.43% | skill 1/1; base 1/1 |
| codex | evg-platform-selection-positive | 91,390 | 71,099 | +20,291 | +28.54% | skill 1/1; base 1/1 |
| codex | evg-status-results-positive | 216,473 | 1,195,058 | -978,585 | -81.89% | skill 1/1; base 1/1 |
| codex | evg-vda-routing-negative | 132,042 | 99,586 | +32,456 | +32.59% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,429,102 | 15,736,209 | -10,307,107 | -65.50% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 31 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cdf-event-video-generation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cdf-event-video-generation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (7457 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-event-video-generation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in airflow-direct-api.md (`skills/cdf-event-video-generation/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cdf-event-video-generation/SKILL.md`)
- 26 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
