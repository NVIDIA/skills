# Skill Benchmark: doca-flow-dpa-perf

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-flow-dpa-perf`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:fd774d6fab89dd6f3570c10603513afadc65b0e6a50895a9b48522234f3547e6` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.9% — baseline ran, but no comparable score was available; uplift unavailable | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 45.0% → 100.0% (+55.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.3% → 98.1% (+56.8 points) | 63.8% → 100.0% (+36.2 points) |
| Efficiency | 81.1% — baseline ran, but no comparable score was available; uplift unavailable | 80.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 837,022 | 488,800 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-flow-dpa-perf.explicit-positive.v1 | 263,507 | 211,146 | +52,361 | +24.80% | skill 1/1; base 1/1 |
| claude-code | doca-flow-dpa-perf.implicit-positive.v1 | 543,202 | 185,578 | +357,624 | +192.71% | skill 1/1; base 1/1 |
| claude-code | doca-flow-dpa-perf.implicit-positive.v2 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | doca-flow-dpa-perf.negative.v1 | 30,313 | 92,076 | -61,763 | -67.08% | skill 1/1; base 1/1 |
| codex | All cases | 469,399 | 196,844 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-flow-dpa-perf.explicit-positive.v1 | 140,344 | 58,287 | +82,057 | +140.78% | skill 1/1; base 1/1 |
| codex | doca-flow-dpa-perf.implicit-positive.v1 | 303,330 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | doca-flow-dpa-perf.implicit-positive.v2 | N/A | 106,332 | N/A | N/A | skill 0/1; base 1/1 |
| codex | doca-flow-dpa-perf.negative.v1 | 25,725 | 32,225 | -6,500 | -20.17% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,306,421 | 685,644 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/tools/doca-flow-dpa-perf/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-flow-dpa-perf`)
- **LOW** QUALITY/quality_discoverability: Description very long (1000 chars, recommend 50-150) (`skills/tools/doca-flow-dpa-perf/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/tools/doca-flow-dpa-perf/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/tools/doca-flow-dpa-perf/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
