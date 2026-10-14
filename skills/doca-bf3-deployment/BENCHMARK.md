# Skill Benchmark: doca-bf3-deployment

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-bf3-deployment`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (2 positive, 1 negative)
- Dataset digest: `sha256:c3d6c876b2c75b4892b8cad068e3cffbe5d78b34b73f1ee3a8be5c76fc675e88` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 73.3% → 100.0% (+26.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 76.1% → 100.0% (+23.9 points) | 60.0% → 98.3% (+38.3 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 553,135 | 605,997 | -52,862 | -8.72% | skill 3/3; base 3/3 |
| claude-code | doca-bf3-deployment.explicit-positive.v1 | 200,352 | 225,130 | -24,778 | -11.01% | skill 1/1; base 1/1 |
| claude-code | doca-bf3-deployment.implicit-positive.v1 | 260,359 | 259,288 | +1,071 | +0.41% | skill 1/1; base 1/1 |
| claude-code | doca-bf3-deployment.negative.v1 | 92,424 | 121,579 | -29,155 | -23.98% | skill 1/1; base 1/1 |
| codex | All cases | 383,051 | 576,943 | -193,892 | -33.61% | skill 3/3; base 3/3 |
| codex | doca-bf3-deployment.explicit-positive.v1 | 166,119 | 278,586 | -112,467 | -40.37% | skill 1/1; base 1/1 |
| codex | doca-bf3-deployment.implicit-positive.v1 | 134,037 | 225,761 | -91,724 | -40.63% | skill 1/1; base 1/1 |
| codex | doca-bf3-deployment.negative.v1 | 82,895 | 72,596 | +10,299 | +14.19% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 936,186 | 1,182,940 | -246,754 | -20.86% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 184-224)
  vs "## debug" in TASKS.md (lines 296-328) (`CAPABILITIES.md:184`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/doca-bf3-deployment/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in details.md (`skills/doca-bf3-deployment/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (1008 chars, recommend 50-150) (`skills/doca-bf3-deployment/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/doca-bf3-deployment/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
