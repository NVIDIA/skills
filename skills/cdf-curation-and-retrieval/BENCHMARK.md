# Skill Benchmark: cdf-curation-and-retrieval

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-curation-and-retrieval`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:68c9c65d696cee768d1ab51c11a9e58f26d2fa9369aed6853ef5a7d51c69abe8` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 85.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 84.6% → 100.0% (+15.4 points) | 100.0% → 96.2% (-3.8 points) |
| Correctness | 43.1% → 100.0% (+56.9 points) | 52.3% → 92.3% (+40.0 points) |
| Discoverability | 86.2% — baseline ran, but no comparable score was available; uplift unavailable | 66.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.1% → 89.0% (+43.9 points) | 41.2% → 75.5% (+34.3 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 96.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,845,953 | 2,903,814 | -57,861 | -1.99% | skill 13/13; base 13/13 |
| claude-code | 1 | 117,484 | 281,990 | -164,506 | -58.34% | skill 1/1; base 1/1 |
| claude-code | 11 | 109,584 | 122,056 | -12,472 | -10.22% | skill 1/1; base 1/1 |
| claude-code | 12 | 30,544 | 30,280 | +264 | +0.87% | skill 1/1; base 1/1 |
| claude-code | 13 | 61,860 | 30,689 | +31,171 | +101.57% | skill 1/1; base 1/1 |
| claude-code | 14 | 246,336 | 150,240 | +96,096 | +63.96% | skill 1/1; base 1/1 |
| claude-code | 15 | 291,458 | 211,382 | +80,076 | +37.88% | skill 1/1; base 1/1 |
| claude-code | 16 | 160,435 | 122,511 | +37,924 | +30.96% | skill 1/1; base 1/1 |
| claude-code | 2 | 250,343 | 180,841 | +69,502 | +38.43% | skill 1/1; base 1/1 |
| claude-code | 3 | 960,007 | 632,407 | +327,600 | +51.80% | skill 1/1; base 1/1 |
| claude-code | 4 | 116,918 | 313,950 | -197,032 | -62.76% | skill 1/1; base 1/1 |
| claude-code | 5 | 189,807 | 547,212 | -357,405 | -65.31% | skill 1/1; base 1/1 |
| claude-code | 6 | 197,169 | 123,313 | +73,856 | +59.89% | skill 1/1; base 1/1 |
| claude-code | 9 | 114,008 | 156,943 | -42,935 | -27.36% | skill 1/1; base 1/1 |
| codex | All cases | 1,527,696 | 1,377,724 | +149,972 | +10.89% | skill 13/13; base 13/13 |
| codex | 1 | 92,466 | 38,997 | +53,469 | +137.11% | skill 1/1; base 1/1 |
| codex | 11 | 193,496 | 99,109 | +94,387 | +95.24% | skill 1/1; base 1/1 |
| codex | 12 | 13,711 | 13,626 | +85 | +0.62% | skill 1/1; base 1/1 |
| codex | 13 | 14,355 | 118,752 | -104,397 | -87.91% | skill 1/1; base 1/1 |
| codex | 14 | 14,488 | 23,993 | -9,505 | -39.62% | skill 1/1; base 1/1 |
| codex | 15 | 88,433 | 84,294 | +4,139 | +4.91% | skill 1/1; base 1/1 |
| codex | 16 | 223,428 | 116,397 | +107,031 | +91.95% | skill 1/1; base 1/1 |
| codex | 2 | 90,154 | 100,075 | -9,921 | -9.91% | skill 1/1; base 1/1 |
| codex | 3 | 440,834 | 285,710 | +155,124 | +54.29% | skill 1/1; base 1/1 |
| codex | 4 | 72,470 | 69,683 | +2,787 | +4.00% | skill 1/1; base 1/1 |
| codex | 5 | 119,008 | 254,483 | -135,475 | -53.24% | skill 1/1; base 1/1 |
| codex | 6 | 91,202 | 70,588 | +20,614 | +29.20% | skill 1/1; base 1/1 |
| codex | 9 | 73,651 | 102,017 | -28,366 | -27.81% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,373,649 | 4,281,538 | +92,111 | +2.15% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in calibration-config.md (`skills/cdf-curation-and-retrieval/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-curation-and-retrieval/SKILL.md (`BENCHMARK.md:105`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-curation-and-retrieval/SKILL.md (`BENCHMARK.md:106`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-curation-and-retrieval/SKILL.md (`BENCHMARK.md:107`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-curation-and-retrieval/SKILL.md (`BENCHMARK.md:108`)
- 3 additional finding(s) are available in the full evaluation artifacts.

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
