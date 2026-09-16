# Skill Benchmark: kermt-finetune

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kermt-finetune`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:78e80c1a73622d5a07c1526f6270c42770347f5faa77969107e6cc60de50bc41` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 84.9% — baseline ran, but no comparable score was available; uplift unavailable | 69.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 88.5% → 100.0% (+11.5 points) | 27.3% → 71.4% (+44.1 points) |
| Correctness | 23.1% → 96.0% (+72.9 points) | 78.2% → 91.4% (+13.2 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 79.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 18.9% → 51.0% (+32.1 points) | 30.7% → 35.7% (+5.0 points) |
| Efficiency | 79.8% — baseline ran, but no comparable score was available; uplift unavailable | 69.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,624,153 | 5,610,370 | N/A | N/A | skill 5/5; base 13/13 |
| claude-code | kermt-finetune-001 | 268,233 | 530,319 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-finetune-002 | 460,964 | 908,098 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-finetune-003 | 324,432 | 861,553 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-finetune-004 | 30,364 | 30,003 | +361 | +1.20% | skill 1/1; base 1/1 |
| claude-code | kermt-finetune-005 | 540,160 | 3,280,397 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 1,907,233 | 6,598,000 | N/A | N/A | skill 7/7; base 11/11 |
| codex | kermt-finetune-001 | 126,075 | 990,399 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-finetune-002 | 395,517 | 798,736 | -403,219 | -50.48% | skill 1/1; base 1/1 |
| codex | kermt-finetune-003 | 190,354 | 1,944,827 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-finetune-004 | 13,917 | 18,278 | -4,361 | -23.86% | skill 1/1; base 1/1 |
| codex | kermt-finetune-005 | 1,181,370 | 2,845,760 | -1,664,390 | -58.49% | skill 3/3; base 3/3 |
| ALL AGENTS | Dataset aggregate | 3,531,386 | 12,208,370 | N/A | N/A | skill 12/12; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 43 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/kermt-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/kermt-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kermt-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/kermt-finetune/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/kermt-finetune/SKILL.md`)
- 38 additional finding(s) are available in the full evaluation artifacts.

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
