# Skill Benchmark: doca-aes-gcm

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-aes-gcm`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:b714f4675f43669d5298e2ed02dbee85b5b8d394ced607da4e7f4da7e23663cb` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 97.8% — baseline ran, but no comparable score was available; uplift unavailable | 95.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 75.0% → 100.0% (+25.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 72.1% → 97.5% (+25.4 points) |
| Efficiency | 88.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 465,329 | 528,413 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-aes-gcm.explicit-positive.v1 | 214,999 | 184,304 | +30,695 | +16.65% | skill 1/1; base 1/1 |
| claude-code | doca-aes-gcm.implicit-positive.v1 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | doca-aes-gcm.implicit-positive.v2 | 218,452 | 218,319 | +133 | +0.06% | skill 1/1; base 1/1 |
| claude-code | doca-aes-gcm.negative.v1 | 31,878 | 125,790 | -93,912 | -74.66% | skill 1/1; base 1/1 |
| codex | All cases | 475,180 | 195,059 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-aes-gcm.explicit-positive.v1 | 116,086 | 52,949 | +63,137 | +119.24% | skill 1/1; base 1/1 |
| codex | doca-aes-gcm.implicit-positive.v1 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | doca-aes-gcm.implicit-positive.v2 | 313,918 | 122,622 | +191,296 | +156.00% | skill 1/1; base 1/1 |
| codex | doca-aes-gcm.negative.v1 | 45,176 | 19,488 | +25,688 | +131.81% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 940,509 | 723,472 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/libs/doca-aes-gcm/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-aes-gcm/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/libs/doca-aes-gcm`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: Sudo    (`BENCHMARK.md:87`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: Sudo    (`BENCHMARK.md:88`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
