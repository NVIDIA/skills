# Skill Benchmark: i4h-workflow-dataset-teleop

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `i4h-workflow-dataset-teleop`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:7e5d819eae66b55ccd3c97e712d134cb73d73d8e5a746f5c03faf7907c3d98c9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 66.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 88.9% → 66.7% (-22.2 points) | 71.4% → 83.3% (+11.9 points) |
| Correctness | 4.4% → 66.7% (+62.3 points) | 31.4% → 93.3% (+61.9 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 75.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 3.3% → 21.7% (+18.4 points) | 10.9% → 36.3% (+25.4 points) |
| Efficiency | 83.3% — baseline ran, but no comparable score was available; uplift unavailable | 71.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,038,495 | 4,435,856 | N/A | N/A | skill 3/3; base 9/9 |
| claude-code | i4h-workflow-dataset-teleop-g1-keyboard-five | 272,645 | 2,956,670 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-dataset-teleop-specialty-endoluminal-patient-twin | 308,462 | 724,135 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-dataset-teleop-table-keyboard-five | 457,388 | 755,051 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 3,869,512 | 2,533,623 | N/A | N/A | skill 3/3; base 7/7 |
| codex | i4h-workflow-dataset-teleop-g1-keyboard-five | 1,327,813 | 114,717 | +1,213,096 | +1057.47% | skill 1/1; base 1/1 |
| codex | i4h-workflow-dataset-teleop-specialty-endoluminal-patient-twin | 878,644 | 1,000,629 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-dataset-teleop-table-keyboard-five | 1,663,055 | 1,418,277 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 4,908,007 | 6,969,479 | N/A | N/A | skill 6/6; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Unknown (SQP-2): The skill instructs the agent to automatically clone a remote git repository into the user's home directory (~/$I4H_REPO (`SKILL.md:33`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-dataset-teleop/SKILL.md`)

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
