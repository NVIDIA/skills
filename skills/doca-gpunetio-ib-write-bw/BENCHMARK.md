# Skill Benchmark: doca-gpunetio-ib-write-bw

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-gpunetio-ib-write-bw`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:3bd3f0ca1fe371c64bb864f4382cd582807ddac7beebb2e09da24dd432e32662` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | 95.3% — baseline ran, but no comparable score was available; uplift unavailable | 96.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 30.0% → 100.0% (+70.0 points) | 63.3% → 100.0% (+36.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.7% → 87.5% (+55.8 points) | 30.8% → 93.8% (+63.0 points) |
| Efficiency | 89.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 913,305 | 980,448 | N/A | N/A | skill 4/4; base 6/6 |
| claude-code | doca-gpunetio-ib-write-bw.explicit-positive.v1 | 269,387 | 632,612 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | doca-gpunetio-ib-write-bw.implicit-positive.v1 | 344,182 | 95,628 | +248,554 | +259.92% | skill 1/1; base 1/1 |
| claude-code | doca-gpunetio-ib-write-bw.implicit-positive.v2 | 267,005 | 127,925 | +139,080 | +108.72% | skill 1/1; base 1/1 |
| claude-code | doca-gpunetio-ib-write-bw.negative.v1 | 32,731 | 124,283 | -91,552 | -73.66% | skill 1/1; base 1/1 |
| codex | All cases | 405,475 | 410,169 | N/A | N/A | skill 4/4; base 6/6 |
| codex | doca-gpunetio-ib-write-bw.explicit-positive.v1 | 153,862 | 34,241 | +119,621 | +349.35% | skill 1/1; base 1/1 |
| codex | doca-gpunetio-ib-write-bw.implicit-positive.v1 | 121,065 | 14,651 | +106,414 | +726.33% | skill 1/1; base 1/1 |
| codex | doca-gpunetio-ib-write-bw.implicit-positive.v2 | 111,233 | 223,349 | N/A | N/A | skill 1/1; base 3/3 |
| codex | doca-gpunetio-ib-write-bw.negative.v1 | 19,315 | 137,928 | -118,613 | -86.00% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,318,780 | 1,390,617 | N/A | N/A | skill 8/8; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/tools/doca-gpunetio-ib-write-bw/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/tools/doca-gpunetio-ib-write-bw/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-gpunetio-ib-write-bw`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: No confirmation (`TASKS.md:100`)
- **LOW** QUALITY/quality_discoverability: Description very long (985 chars, recommend 50-150) (`skills/tools/doca-gpunetio-ib-write-bw/SKILL.md`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
