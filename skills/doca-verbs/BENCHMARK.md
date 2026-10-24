# Skill Benchmark: doca-verbs

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-verbs`
- Evaluation date: 2026-10-24
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:68fee5caecf9a6d68a1653e3eadd6daf86fb1f79f00fb1f99fffcc2003f1a39b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 100.0% (+60.0 points) | 85.0% → 100.0% (+15.0 points) |
| Discoverability | 93.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.5% → 74.8% (+37.3 points) | 60.0% → 95.6% (+35.6 points) |
| Efficiency | 84.4% — baseline ran, but no comparable score was available; uplift unavailable | 75.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,099,408 | 459,339 | +640,069 | +139.35% | skill 4/4; base 4/4 |
| claude-code | doca-verbs.explicit-positive.v1 | 472,239 | 92,008 | +380,231 | +413.26% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.implicit-positive.v1 | 264,376 | 89,156 | +175,220 | +196.53% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.implicit-positive.v2 | 194,306 | 125,515 | +68,791 | +54.81% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.negative.v1 | 168,487 | 152,660 | +15,827 | +10.37% | skill 1/1; base 1/1 |
| codex | All cases | 297,879 | 202,982 | +94,897 | +46.75% | skill 4/4; base 4/4 |
| codex | doca-verbs.explicit-positive.v1 | 54,968 | 52,378 | +2,590 | +4.94% | skill 1/1; base 1/1 |
| codex | doca-verbs.implicit-positive.v1 | 81,600 | 53,569 | +28,031 | +52.33% | skill 1/1; base 1/1 |
| codex | doca-verbs.implicit-positive.v2 | 105,212 | 30,633 | +74,579 | +243.46% | skill 1/1; base 1/1 |
| codex | doca-verbs.negative.v1 | 56,099 | 66,402 | -10,303 | -15.52% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,397,287 | 662,321 | +734,966 | +110.97% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/libs/doca-verbs`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in command-appendix.md (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/libs/doca-verbs/SKILL.md`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
