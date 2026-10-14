# Skill Benchmark: doca-dpa

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-dpa`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:99ac207da052504949cf6f114412975da1858f1bcea2374f3fac8c667696bb9c` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.7% — baseline ran, but no comparable score was available; uplift unavailable | 97.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 53.8% → 89.4% (+35.6 points) | 60.0% → 100.0% (+40.0 points) |
| Efficiency | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,201,395 | 594,450 | +606,945 | +102.10% | skill 4/4; base 4/4 |
| claude-code | doca-dpa.explicit-positive.v1 | 356,484 | 215,339 | +141,145 | +65.55% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.implicit-positive.v1 | 450,002 | 126,595 | +323,407 | +255.47% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.implicit-positive.v2 | 363,522 | 125,801 | +237,721 | +188.97% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.negative.v1 | 31,387 | 126,715 | -95,328 | -75.23% | skill 1/1; base 1/1 |
| codex | All cases | 514,136 | 215,299 | +298,837 | +138.80% | skill 4/4; base 4/4 |
| codex | doca-dpa.explicit-positive.v1 | 171,079 | 54,248 | +116,831 | +215.36% | skill 1/1; base 1/1 |
| codex | doca-dpa.implicit-positive.v1 | 154,613 | 36,602 | +118,011 | +322.42% | skill 1/1; base 1/1 |
| codex | doca-dpa.implicit-positive.v2 | 143,971 | 56,476 | +87,495 | +154.92% | skill 1/1; base 1/1 |
| codex | doca-dpa.negative.v1 | 44,473 | 67,973 | -23,500 | -34.57% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,715,531 | 809,749 | +905,782 | +111.86% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_prompt_size_limit: A Tier 2 cluster exceeds the LLM prompt character limit. (`skills/libs/doca-dpa`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-dpa/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/libs/doca-dpa`)
- **LOW** QUALITY/quality_discoverability: Description very long (904 chars, recommend 50-150) (`skills/libs/doca-dpa/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/libs/doca-dpa/SKILL.md`)
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
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
