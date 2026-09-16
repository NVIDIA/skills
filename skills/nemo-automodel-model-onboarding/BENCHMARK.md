# Skill Benchmark: nemo-automodel-model-onboarding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-automodel-model-onboarding`
- Evaluation date: 2026-09-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:814bfc7c94da8ea6fd1a065d7f3f0c4fcda9ef918f1da7eb46630700f241fc25` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 66.7% → 100.0% (+33.3 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.4% → 93.5% (+46.1 points) | 62.2% → 89.2% (+27.0 points) |
| Efficiency | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 98.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 207,704 | 2,909,039 | -2,701,335 | -92.86% | skill 3/3; base 3/3 |
| claude-code | nemo-automodel-model-onboarding-001-new-dense-llm | 69,440 | 967,449 | -898,009 | -92.82% | skill 1/1; base 1/1 |
| claude-code | nemo-automodel-model-onboarding-002-moe-state-dict | 68,988 | 351,542 | -282,554 | -80.38% | skill 1/1; base 1/1 |
| claude-code | nemo-automodel-model-onboarding-003-vlm-onboarding | 69,276 | 1,590,048 | -1,520,772 | -95.64% | skill 1/1; base 1/1 |
| codex | All cases | 160,716 | 1,776,597 | N/A | N/A | skill 3/3; base 4/4 |
| codex | nemo-automodel-model-onboarding-001-new-dense-llm | 47,120 | 776,645 | -729,525 | -93.93% | skill 1/1; base 1/1 |
| codex | nemo-automodel-model-onboarding-002-moe-state-dict | 30,458 | 102,750 | -72,292 | -70.36% | skill 1/1; base 1/1 |
| codex | nemo-automodel-model-onboarding-003-vlm-onboarding | 83,138 | 897,202 | N/A | N/A | skill 1/1; base 2/2 |
| ALL AGENTS | Dataset aggregate | 368,420 | 4,685,636 | N/A | N/A | skill 6/6; base 7/7 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5622 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-automodel-model-onboarding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/nemo-automodel-model-onboarding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/nemo-automodel-model-onboarding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No troubleshooting section documented (`skills/nemo-automodel-model-onboarding/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nemo-automodel-model-onboarding/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
