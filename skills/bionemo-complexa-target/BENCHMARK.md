# Skill Benchmark: complexa-target

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `complexa-target`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:16323930db417ff3f6af55867160aa4eae9f6fbe5929408117c4a7df6d112418` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 25.0% → 50.0% (+25.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.5% → 100.0% (+2.5 points) | 97.5% → 100.0% (+2.5 points) |
| Efficiency | 78.4% — baseline ran, but no comparable score was available; uplift unavailable | 74.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,821,072 | 2,886,799 | -65,727 | -2.28% | skill 4/4; base 4/4 |
| claude-code | complexa-target-001 | 918,193 | 624,617 | +293,576 | +47.00% | skill 1/1; base 1/1 |
| claude-code | complexa-target-002 | 864,450 | 853,654 | +10,796 | +1.26% | skill 1/1; base 1/1 |
| claude-code | complexa-target-003 | 798,839 | 1,170,824 | -371,985 | -31.77% | skill 1/1; base 1/1 |
| claude-code | complexa-target-004 | 239,590 | 237,704 | +1,886 | +0.79% | skill 1/1; base 1/1 |
| codex | All cases | 875,155 | 789,823 | +85,332 | +10.80% | skill 4/4; base 4/4 |
| codex | complexa-target-001 | 326,186 | 227,088 | +99,098 | +43.64% | skill 1/1; base 1/1 |
| codex | complexa-target-002 | 294,178 | 244,017 | +50,161 | +20.56% | skill 1/1; base 1/1 |
| codex | complexa-target-003 | 167,803 | 193,006 | -25,203 | -13.06% | skill 1/1; base 1/1 |
| codex | complexa-target-004 | 86,988 | 125,712 | -38,724 | -30.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,696,227 | 3,676,622 | +19,605 | +0.53% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/complexa-target/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (AE4): analysis-evasion: Suspicious Unicode normalization or mixed-script content (`SKILL.md:1`)
- **LOW** QUALITY/quality_discoverability: Description very long (390 chars, recommend 50-150) (`skills/complexa-target/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/complexa-target/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/complexa-target/SKILL.md`)
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
