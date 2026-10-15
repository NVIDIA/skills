# Skill Benchmark: rtx-remix-modding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `rtx-remix-modding`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (11 positive, 3 negative)
- Dataset digest: `sha256:3a20325201aaaf0e423b220c0c53c7f51ea67a26cd886388f07b2c9084f665b9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 82.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 92.9% (-7.1 points) | 96.4% → 92.9% (-3.5 points) |
| Correctness | 34.3% → 72.9% (+38.6 points) | 31.4% → 74.3% (+42.9 points) |
| Discoverability | 76.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.8% → 73.1% (+22.3 points) | 51.4% → 72.4% (+21.0 points) |
| Efficiency | 97.1% — baseline ran, but no comparable score was available; uplift unavailable | 97.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,119,981 | 1,665,625 | -545,644 | -32.76% | skill 14/14; base 14/14 |
| claude-code | 1 | 94,479 | 119,967 | -25,488 | -21.25% | skill 1/1; base 1/1 |
| claude-code | 10 | 30,081 | 28,955 | +1,126 | +3.89% | skill 1/1; base 1/1 |
| claude-code | 11 | 61,379 | 152,499 | -91,120 | -59.75% | skill 1/1; base 1/1 |
| claude-code | 12 | 31,375 | 29,020 | +2,355 | +8.12% | skill 1/1; base 1/1 |
| claude-code | 13 | 65,351 | 30,695 | +34,656 | +112.90% | skill 1/1; base 1/1 |
| claude-code | 14 | 31,857 | 30,998 | +859 | +2.77% | skill 1/1; base 1/1 |
| claude-code | 2 | 62,164 | 119,337 | -57,173 | -47.91% | skill 1/1; base 1/1 |
| claude-code | 3 | 63,014 | 214,931 | -151,917 | -70.68% | skill 1/1; base 1/1 |
| claude-code | 4 | 61,391 | 248,054 | -186,663 | -75.25% | skill 1/1; base 1/1 |
| claude-code | 5 | 61,322 | 150,081 | -88,759 | -59.14% | skill 1/1; base 1/1 |
| claude-code | 6 | 61,913 | 29,811 | +32,102 | +107.69% | skill 1/1; base 1/1 |
| claude-code | 7 | 118,463 | 148,920 | -30,457 | -20.45% | skill 1/1; base 1/1 |
| claude-code | 8 | 317,383 | 302,686 | +14,697 | +4.86% | skill 1/1; base 1/1 |
| claude-code | 9 | 59,809 | 59,671 | +138 | +0.23% | skill 1/1; base 1/1 |
| codex | All cases | 785,910 | 1,040,450 | -254,540 | -24.46% | skill 14/14; base 14/14 |
| codex | 1 | 62,585 | 83,292 | -20,707 | -24.86% | skill 1/1; base 1/1 |
| codex | 10 | 13,683 | 18,033 | -4,350 | -24.12% | skill 1/1; base 1/1 |
| codex | 11 | 29,160 | 84,574 | -55,414 | -65.52% | skill 1/1; base 1/1 |
| codex | 12 | 29,619 | 13,505 | +16,114 | +119.32% | skill 1/1; base 1/1 |
| codex | 13 | 28,922 | 13,391 | +15,531 | +115.98% | skill 1/1; base 1/1 |
| codex | 14 | 29,413 | 13,440 | +15,973 | +118.85% | skill 1/1; base 1/1 |
| codex | 2 | 29,262 | 84,188 | -54,926 | -65.24% | skill 1/1; base 1/1 |
| codex | 3 | 29,114 | 98,415 | -69,301 | -70.42% | skill 1/1; base 1/1 |
| codex | 4 | 29,533 | 119,200 | -89,667 | -75.22% | skill 1/1; base 1/1 |
| codex | 5 | 29,198 | 68,701 | -39,503 | -57.50% | skill 1/1; base 1/1 |
| codex | 6 | 29,065 | 29,440 | -375 | -1.27% | skill 1/1; base 1/1 |
| codex | 7 | 29,491 | 55,045 | -25,554 | -46.42% | skill 1/1; base 1/1 |
| codex | 8 | 402,936 | 345,182 | +57,754 | +16.73% | skill 1/1; base 1/1 |
| codex | 9 | 13,929 | 14,044 | -115 | -0.82% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,905,891 | 2,706,075 | -800,184 | -29.57% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/rtx-remix-modding/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/rtx-remix-modding/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/rtx-remix-modding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/rtx-remix-modding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/rtx-remix-modding/SKILL.md`)

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
