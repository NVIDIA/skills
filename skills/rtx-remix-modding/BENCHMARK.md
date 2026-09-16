# Skill Benchmark: rtx-remix-modding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `rtx-remix-modding`
- Evaluation date: 2026-09-23
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 10 evaluation tasks (7 positive, 3 negative)
- Dataset digest: `sha256:29a10a044caef4ddb28554a7c700cd5179001fc4eac72b52505b4f948df94241` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 80.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 83.3% (-4.2 points) |
| Correctness | 25.0% → 70.9% (+45.9 points) | 18.0% → 58.3% (+40.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 32.1% → 82.3% (+50.2 points) | 34.0% → 69.0% (+35.0 points) |
| Efficiency | 98.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 842,223 | 2,269,699 | N/A | N/A | skill 11/11; base 16/16 |
| claude-code | 1 | 62,075 | 400,387 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 10 | 60,251 | 30,000 | N/A | N/A | skill 2/2; base 1/1 |
| claude-code | 2 | 61,586 | 90,438 | -28,852 | -31.90% | skill 1/1; base 1/1 |
| claude-code | 3 | 62,843 | 425,499 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 4 | 65,544 | 124,242 | -58,698 | -47.24% | skill 1/1; base 1/1 |
| claude-code | 5 | 62,448 | 152,730 | -90,282 | -59.11% | skill 1/1; base 1/1 |
| claude-code | 6 | 62,033 | 29,816 | +32,217 | +108.05% | skill 1/1; base 1/1 |
| claude-code | 7 | 62,776 | 331,004 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 8 | 312,411 | 654,478 | -342,067 | -52.27% | skill 1/1; base 1/1 |
| claude-code | 9 | 30,256 | 31,105 | -849 | -2.73% | skill 1/1; base 1/1 |
| codex | All cases | 1,579,075 | 2,088,000 | N/A | N/A | skill 12/12; base 20/20 |
| codex | 1 | 28,825 | 283,307 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 10 | 17,901 | 13,652 | +4,249 | +31.12% | skill 1/1; base 1/1 |
| codex | 2 | 29,429 | 222,184 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 3 | 28,843 | 265,820 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 4 | 29,177 | 119,307 | -90,130 | -75.54% | skill 1/1; base 1/1 |
| codex | 5 | 28,768 | 69,117 | -40,349 | -58.38% | skill 1/1; base 1/1 |
| codex | 6 | 28,756 | 29,588 | -832 | -2.81% | skill 1/1; base 1/1 |
| codex | 7 | 29,110 | 122,283 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 8 | 1,344,356 | 948,260 | +396,096 | +41.77% | skill 3/3; base 3/3 |
| codex | 9 | 13,910 | 14,482 | -572 | -3.95% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,421,298 | 4,357,699 | N/A | N/A | skill 23/23; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

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
