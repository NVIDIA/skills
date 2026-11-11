# Skill Benchmark: rtx-remix-modding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `rtx-remix-modding`
- Evaluation date: 2026-10-20
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
| Overall | 93.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 96.4% → 96.4% (±0.0 points) |
| Correctness | 31.4% → 78.6% (+47.2 points) | 30.0% → 67.1% (+37.1 points) |
| Discoverability | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.7% → 93.2% (+42.5 points) | 41.7% → 67.4% (+25.7 points) |
| Efficiency | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 98.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,215,532 | 1,750,031 | -534,499 | -30.54% | skill 14/14; base 14/14 |
| claude-code | 1 | 95,878 | 181,968 | -86,090 | -47.31% | skill 1/1; base 1/1 |
| claude-code | 10 | 30,109 | 29,921 | +188 | +0.63% | skill 1/1; base 1/1 |
| claude-code | 11 | 61,916 | 219,441 | -157,525 | -71.78% | skill 1/1; base 1/1 |
| claude-code | 12 | 63,200 | 30,404 | +32,796 | +107.87% | skill 1/1; base 1/1 |
| claude-code | 13 | 64,827 | 31,275 | +33,552 | +107.28% | skill 1/1; base 1/1 |
| claude-code | 14 | 92,003 | 31,258 | +60,745 | +194.33% | skill 1/1; base 1/1 |
| claude-code | 2 | 61,810 | 120,148 | -58,338 | -48.56% | skill 1/1; base 1/1 |
| claude-code | 3 | 62,985 | 183,962 | -120,977 | -65.76% | skill 1/1; base 1/1 |
| claude-code | 4 | 63,702 | 154,200 | -90,498 | -58.69% | skill 1/1; base 1/1 |
| claude-code | 5 | 62,872 | 151,411 | -88,539 | -58.48% | skill 1/1; base 1/1 |
| claude-code | 6 | 61,849 | 29,736 | +32,113 | +107.99% | skill 1/1; base 1/1 |
| claude-code | 7 | 185,269 | 118,627 | +66,642 | +56.18% | skill 1/1; base 1/1 |
| claude-code | 8 | 216,303 | 345,401 | -129,098 | -37.38% | skill 1/1; base 1/1 |
| claude-code | 9 | 92,809 | 122,279 | -29,470 | -24.10% | skill 1/1; base 1/1 |
| codex | All cases | 632,481 | 1,020,221 | -387,740 | -38.01% | skill 14/14; base 14/14 |
| codex | 1 | 29,344 | 117,332 | -87,988 | -74.99% | skill 1/1; base 1/1 |
| codex | 10 | 13,677 | 13,719 | -42 | -0.31% | skill 1/1; base 1/1 |
| codex | 11 | 29,103 | 84,654 | -55,551 | -65.62% | skill 1/1; base 1/1 |
| codex | 12 | 29,603 | 13,529 | +16,074 | +118.81% | skill 1/1; base 1/1 |
| codex | 13 | 29,638 | 13,589 | +16,049 | +118.10% | skill 1/1; base 1/1 |
| codex | 14 | 29,065 | 13,591 | +15,474 | +113.85% | skill 1/1; base 1/1 |
| codex | 2 | 29,190 | 83,774 | -54,584 | -65.16% | skill 1/1; base 1/1 |
| codex | 3 | 29,151 | 83,205 | -54,054 | -64.96% | skill 1/1; base 1/1 |
| codex | 4 | 29,368 | 130,843 | -101,475 | -77.55% | skill 1/1; base 1/1 |
| codex | 5 | 29,334 | 82,255 | -52,921 | -64.34% | skill 1/1; base 1/1 |
| codex | 6 | 29,145 | 29,504 | -359 | -1.22% | skill 1/1; base 1/1 |
| codex | 7 | 29,450 | 55,054 | -25,604 | -46.51% | skill 1/1; base 1/1 |
| codex | 8 | 281,993 | 285,175 | -3,182 | -1.12% | skill 1/1; base 1/1 |
| codex | 9 | 14,420 | 13,997 | +423 | +3.02% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,848,013 | 2,770,252 | -922,239 | -33.29% | skill 28/28; base 28/28 |

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
