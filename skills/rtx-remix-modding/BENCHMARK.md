# Skill Benchmark: rtx-remix-modding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `rtx-remix-modding`
- Evaluation date: 2026-10-06
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 11 evaluation tasks (8 positive, 3 negative)
- Dataset digest: `sha256:7a32fc881c330ecfeaa36d0afd37f0d220e5dc84fac8aefef0c1bcb3db46b52b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 86.2% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 95.5% → 90.9% (-4.6 points) |
| Correctness | 29.1% → 65.5% (+36.4 points) | 16.4% → 72.7% (+56.3 points) |
| Discoverability | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 53.9% → 75.9% (+22.0 points) | 43.3% → 79.1% (+35.8 points) |
| Efficiency | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 97.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,013,190 | 1,561,033 | -547,843 | -35.09% | skill 11/11; base 11/11 |
| claude-code | 1 | 61,139 | 151,684 | -90,545 | -59.69% | skill 1/1; base 1/1 |
| claude-code | 10 | 30,071 | 30,206 | -135 | -0.45% | skill 1/1; base 1/1 |
| claude-code | 11 | 62,630 | 403,844 | -341,214 | -84.49% | skill 1/1; base 1/1 |
| claude-code | 2 | 61,519 | 182,621 | -121,102 | -66.31% | skill 1/1; base 1/1 |
| claude-code | 3 | 62,745 | 180,960 | -118,215 | -65.33% | skill 1/1; base 1/1 |
| claude-code | 4 | 62,289 | 126,147 | -63,858 | -50.62% | skill 1/1; base 1/1 |
| claude-code | 5 | 62,293 | 181,532 | -119,239 | -65.68% | skill 1/1; base 1/1 |
| claude-code | 6 | 61,579 | 29,652 | +31,927 | +107.67% | skill 1/1; base 1/1 |
| claude-code | 7 | 149,862 | 118,266 | +31,596 | +26.72% | skill 1/1; base 1/1 |
| claude-code | 8 | 368,311 | 124,857 | +243,454 | +194.99% | skill 1/1; base 1/1 |
| claude-code | 9 | 30,752 | 31,264 | -512 | -1.64% | skill 1/1; base 1/1 |
| codex | All cases | 681,803 | 915,665 | -233,862 | -25.54% | skill 11/11; base 11/11 |
| codex | 1 | 29,557 | 128,273 | -98,716 | -76.96% | skill 1/1; base 1/1 |
| codex | 10 | 13,728 | 18,080 | -4,352 | -24.07% | skill 1/1; base 1/1 |
| codex | 11 | 29,025 | 70,121 | -41,096 | -58.61% | skill 1/1; base 1/1 |
| codex | 2 | 29,155 | 122,389 | -93,234 | -76.18% | skill 1/1; base 1/1 |
| codex | 3 | 28,976 | 86,634 | -57,658 | -66.55% | skill 1/1; base 1/1 |
| codex | 4 | 29,166 | 85,929 | -56,763 | -66.06% | skill 1/1; base 1/1 |
| codex | 5 | 29,199 | 69,273 | -40,074 | -57.85% | skill 1/1; base 1/1 |
| codex | 6 | 28,916 | 29,461 | -545 | -1.85% | skill 1/1; base 1/1 |
| codex | 7 | 29,349 | 40,953 | -11,604 | -28.33% | skill 1/1; base 1/1 |
| codex | 8 | 420,537 | 250,413 | +170,124 | +67.94% | skill 1/1; base 1/1 |
| codex | 9 | 14,195 | 14,139 | +56 | +0.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,694,993 | 2,476,698 | -781,705 | -31.56% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

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
