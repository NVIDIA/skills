# Skill Benchmark: using-ovnewton

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `using-ovnewton`
- Evaluation date: 2026-10-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (5 positive, 2 negative)
- Dataset digest: `sha256:207f3dfeb5504f9b90e549d26ffcd798dc3df9d4764d50d0f11043cc5e892ab3` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 78.6% → 92.9% (+14.3 points) | 85.7% → 85.7% (±0.0 points) |
| Correctness | 54.3% → 100.0% (+45.7 points) | 77.1% → 100.0% (+22.9 points) |
| Discoverability | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.7% → 89.6% (+48.9 points) | 61.4% → 91.8% (+30.4 points) |
| Efficiency | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 71.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,694,920 | 5,952,635 | -4,257,715 | -71.53% | skill 7/7; base 7/7 |
| claude-code | using-ovnewton-basic | 314,959 | 2,518,542 | -2,203,583 | -87.49% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-existing-stage | 60,415 | 93,620 | -33,205 | -35.47% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-newton-only | 997,072 | 1,337,150 | -340,078 | -25.43% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-omniverse-authoring | 31,517 | 31,781 | -264 | -0.83% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-rendering | 63,382 | 65,544 | -2,162 | -3.30% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-stage-edits | 61,947 | 156,244 | -94,297 | -60.35% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-vbd | 165,628 | 1,749,754 | -1,584,126 | -90.53% | skill 1/1; base 1/1 |
| codex | All cases | 624,157 | 1,133,966 | -509,809 | -44.96% | skill 7/7; base 7/7 |
| codex | using-ovnewton-basic | 136,252 | 322,768 | -186,516 | -57.79% | skill 1/1; base 1/1 |
| codex | using-ovnewton-existing-stage | 60,172 | 124,936 | -64,764 | -51.84% | skill 1/1; base 1/1 |
| codex | using-ovnewton-newton-only | 236,246 | 186,003 | +50,243 | +27.01% | skill 1/1; base 1/1 |
| codex | using-ovnewton-omniverse-authoring | 45,433 | 33,122 | +12,311 | +37.17% | skill 1/1; base 1/1 |
| codex | using-ovnewton-rendering | 57,311 | 50,820 | +6,491 | +12.77% | skill 1/1; base 1/1 |
| codex | using-ovnewton-stage-edits | 45,529 | 83,955 | -38,426 | -45.77% | skill 1/1; base 1/1 |
| codex | using-ovnewton-vbd | 43,214 | 332,362 | -289,148 | -87.00% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,319,077 | 7,086,601 | -4,767,524 | -67.28% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`SKILL.md:57`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`SKILL.md:57`)
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
