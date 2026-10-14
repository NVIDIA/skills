# Skill Benchmark: doca-container-deployment

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-container-deployment`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:f620767bb542265dce8251dde87797f9c87f96fcfb08bea2d513dddfb5ebd0e1` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 93.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 100.0% (+50.0 points) | 65.0% → 100.0% (+35.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 49.2% → 100.0% (+50.8 points) | 54.6% → 98.8% (+44.2 points) |
| Efficiency | 63.3% — baseline ran, but no comparable score was available; uplift unavailable | 73.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 800,507 | 813,748 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-container-deployment.explicit-positive.v1 | 276,740 | 378,971 | -102,231 | -26.98% | skill 1/1; base 1/1 |
| claude-code | doca-container-deployment.implicit-positive.v1 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | doca-container-deployment.implicit-positive.v2 | 493,425 | 277,697 | +215,728 | +77.68% | skill 1/1; base 1/1 |
| claude-code | doca-container-deployment.negative.v1 | 30,342 | 157,080 | -126,738 | -80.68% | skill 1/1; base 1/1 |
| codex | All cases | 435,638 | 157,672 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-container-deployment.explicit-positive.v1 | 136,954 | 25,451 | +111,503 | +438.11% | skill 1/1; base 1/1 |
| codex | doca-container-deployment.implicit-positive.v1 | 269,223 | 102,970 | +166,253 | +161.46% | skill 1/1; base 1/1 |
| codex | doca-container-deployment.implicit-positive.v2 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | doca-container-deployment.negative.v1 | 29,461 | 29,251 | +210 | +0.72% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,236,145 | 971,420 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within TASKS.md:
  "## test" in TASKS.md (lines 305-309)
  vs "## test" in TASKS.md (lines 391-400) (`TASKS.md:305`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 199-308)
  vs "## run" in TASKS.md (lines 232-286)
  vs "## debug" in TASKS.md (lines 415-503)
  vs "## Command appendix" in TASKS.md (lines 533-545) (`CAPABILITIES.md:199`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/doca-container-deployment/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in details.md (`skills/doca-container-deployment/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (970 chars, recommend 50-150) (`skills/doca-container-deployment/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
