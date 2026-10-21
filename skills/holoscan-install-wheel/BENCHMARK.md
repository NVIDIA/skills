# Skill Benchmark: holoscan-install-wheel

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-install-wheel`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:18935ebfa4d03f6845b5bc03c7446f438da336bc4a6046bfe090eb4e18405337` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 83.3% (-8.4 points) | 83.3% → 91.7% (+8.4 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 72.1% → 95.0% (+22.9 points) | 75.0% → 92.5% (+17.5 points) |
| Efficiency | 98.6% — baseline ran, but no comparable score was available; uplift unavailable | 97.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 889,856 | 1,281,266 | -391,410 | -30.55% | skill 6/6; base 6/6 |
| claude-code | holoscan-install-wheel-5x-001 | 63,306 | 151,311 | -88,005 | -58.16% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-wheel-5x-002 | 60,471 | 290,619 | -230,148 | -79.19% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-wheel-5x-003 | 60,378 | 186,965 | -126,587 | -67.71% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-wheel-5x-004 | 62,185 | 312,173 | -249,988 | -80.08% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-wheel-5x-005 | 583,114 | 218,130 | +364,984 | +167.32% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-wheel-5x-006 | 60,402 | 122,068 | -61,666 | -50.52% | skill 1/1; base 1/1 |
| codex | All cases | 452,778 | 340,146 | +112,632 | +33.11% | skill 6/6; base 6/6 |
| codex | holoscan-install-wheel-5x-001 | 29,391 | 14,739 | +14,652 | +99.41% | skill 1/1; base 1/1 |
| codex | holoscan-install-wheel-5x-002 | 29,498 | 14,092 | +15,406 | +109.32% | skill 1/1; base 1/1 |
| codex | holoscan-install-wheel-5x-003 | 29,193 | 14,060 | +15,133 | +107.63% | skill 1/1; base 1/1 |
| codex | holoscan-install-wheel-5x-004 | 29,294 | 14,112 | +15,182 | +107.58% | skill 1/1; base 1/1 |
| codex | holoscan-install-wheel-5x-005 | 306,532 | 268,937 | +37,595 | +13.98% | skill 1/1; base 1/1 |
| codex | holoscan-install-wheel-5x-006 | 28,870 | 14,206 | +14,664 | +103.22% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,342,634 | 1,621,412 | -278,778 | -17.19% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`team-skills/holoscan/holoscan-sdk/holoscan-install-wheel/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`team-skills/holoscan/holoscan-sdk/holoscan-install-wheel/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description doesn't mention WHEN to use this skill (`team-skills/holoscan/holoscan-sdk/holoscan-install-wheel/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`team-skills/holoscan/holoscan-sdk/holoscan-install-wheel/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`team-skills/holoscan/holoscan-sdk/holoscan-install-wheel/SKILL.md`)
- 2 additional finding(s) are available in the full evaluation artifacts.

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
