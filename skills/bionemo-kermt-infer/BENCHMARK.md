# Skill Benchmark: kermt-infer

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kermt-infer`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:517c8a976ea37ee217ad5779f5e01f7533ffb45540b1496c22de3dd14d114490` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 78.6% — baseline ran, but no comparable score was available; uplift unavailable | 80.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 70.0% → 100.0% (+30.0 points) |
| Correctness | 25.0% → 85.0% (+60.0 points) | 50.0% → 90.0% (+40.0 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.9% → 35.0% (+9.1 points) | 22.3% → 34.4% (+12.1 points) |
| Efficiency | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,527,449 | 1,249,375 | N/A | N/A | skill 4/4; base 8/8 |
| claude-code | kermt-infer-001 | 310,888 | 430,603 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-infer-002 | 404,604 | 440,486 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-infer-003 | 306,093 | 155,002 | +151,091 | +97.48% | skill 1/1; base 1/1 |
| claude-code | kermt-infer-004 | 505,864 | 223,284 | +282,580 | +126.56% | skill 1/1; base 1/1 |
| codex | All cases | 1,510,773 | 3,889,875 | N/A | N/A | skill 4/4; base 10/10 |
| codex | kermt-infer-001 | 103,536 | 224,502 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-infer-002 | 86,842 | 1,236,389 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-infer-003 | 79,261 | 256,670 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-infer-004 | 1,241,134 | 2,172,314 | -931,180 | -42.87% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,038,222 | 5,139,250 | N/A | N/A | skill 8/8; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 58 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/kermt-infer/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/kermt-infer/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kermt-infer/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/kermt-infer/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/kermt-infer/SKILL.md`)
- 53 additional finding(s) are available in the full evaluation artifacts.

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
