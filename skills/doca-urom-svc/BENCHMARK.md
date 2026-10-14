# Skill Benchmark: doca-urom-svc

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-urom-svc`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:ea16cc085828d2133e3ded6b24d1d0cc8cb949802c1ddac4c03c2ddb9c1c05f9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 97.3% — baseline ran, but no comparable score was available; uplift unavailable | 98.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 70.0% → 100.0% (+30.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 60.0% → 100.0% (+40.0 points) | 48.8% → 100.0% (+51.2 points) |
| Efficiency | 86.7% — baseline ran, but no comparable score was available; uplift unavailable | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,029,689 | 496,175 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-urom-svc.explicit-positive.v1 | 286,205 | 182,540 | +103,665 | +56.79% | skill 1/1; base 1/1 |
| claude-code | doca-urom-svc.implicit-positive.v1 | 384,732 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | doca-urom-svc.implicit-positive.v2 | N/A | 189,495 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | doca-urom-svc.negative.v1 | 358,752 | 124,140 | +234,612 | +188.99% | skill 1/1; base 1/1 |
| codex | All cases | 537,200 | 162,581 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-urom-svc.explicit-positive.v1 | 199,626 | 38,625 | +161,001 | +416.83% | skill 1/1; base 1/1 |
| codex | doca-urom-svc.implicit-positive.v1 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | doca-urom-svc.implicit-positive.v2 | 299,069 | 109,590 | +189,479 | +172.90% | skill 1/1; base 1/1 |
| codex | doca-urom-svc.negative.v1 | 38,505 | 14,366 | +24,139 | +168.03% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,566,889 | 658,756 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/services/doca-urom-svc`)
- **LOW** QUALITY/quality_discoverability: Description very long (663 chars, recommend 50-150) (`skills/services/doca-urom-svc/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/services/doca-urom-svc/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/services/doca-urom-svc/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/services/doca-urom-svc/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
