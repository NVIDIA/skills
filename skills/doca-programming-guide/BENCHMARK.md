# Skill Benchmark: doca-programming-guide

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-programming-guide`
- Evaluation date: 2026-10-06
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:2e27fd3409307342645dce6375a022958ae921dbff162c787a794aee65a7587a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.6% — baseline ran, but no comparable score was available; uplift unavailable | 93.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 80.0% → 95.0% (+15.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.7% → 84.2% (+42.5 points) | 59.6% → 87.5% (+27.9 points) |
| Efficiency | 68.9% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,089,645 | 649,277 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-programming-guide.explicit-positive.v1 | 270,265 | 182,332 | +87,933 | +48.23% | skill 1/1; base 1/1 |
| claude-code | doca-programming-guide.implicit-positive.v1 | 508,681 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | doca-programming-guide.implicit-positive.v2 | N/A | 219,462 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | doca-programming-guide.negative.v1 | 310,699 | 247,483 | +63,216 | +25.54% | skill 1/1; base 1/1 |
| codex | All cases | 427,689 | 238,144 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-programming-guide.explicit-positive.v1 | 146,728 | 63,523 | +83,205 | +130.98% | skill 1/1; base 1/1 |
| codex | doca-programming-guide.implicit-positive.v1 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | doca-programming-guide.implicit-positive.v2 | 238,951 | 105,094 | +133,857 | +127.37% | skill 1/1; base 1/1 |
| codex | doca-programming-guide.negative.v1 | 42,010 | 69,527 | -27,517 | -39.58% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,517,334 | 887,421 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/doca-programming-guide/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`TASKS.md:132`)
- **LOW** QUALITY/quality_discoverability: Description very long (713 chars, recommend 50-150) (`skills/doca-programming-guide/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/doca-programming-guide/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/doca-programming-guide/SKILL.md`)
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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
