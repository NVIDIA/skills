# Skill Benchmark: compileiq-author-objective

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-author-objective`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:41f4cb8b11ebfdd0634e651d5b1521635fec3258bc30530fdfd787b6b6ad72d0` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-author-objective | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-author-objective | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-author-objective | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-author-objective | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-author-objective | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-author-objective`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 90.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 81.3% → 100.0% (+18.7 points) |
| Correctness | 67.5% → 97.5% (+30.0 points) | 87.5% → 100.0% (+12.5 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.6% → 82.8% (+47.2 points) | 50.0% → 88.4% (+38.4 points) |
| Efficiency | 80.9% — baseline ran, but no comparable score was available; uplift unavailable | 65.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.5% → 93.7% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.9% → 87.3% (+4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,485,837 | 1,252,944 | +232,893 | +18.59% | skill 8/8; base 8/8 |
| claude-code | compileiq-author-objective-canary | 157,657 | 147,605 | +10,052 | +6.81% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-fa4-cutedsl | 342,111 | 161,032 | +181,079 | +112.45% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-flashinfer | 159,427 | 112,931 | +46,496 | +41.17% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-legacy-nvcc-review | 282,737 | 227,273 | +55,464 | +24.40% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-mixed-space | 113,552 | 147,472 | -33,920 | -23.00% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-negative-scipy-objective | 35,817 | 35,932 | -115 | -0.32% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-negative-triton-autotune | 37,899 | 38,329 | -430 | -1.12% | skill 1/1; base 1/1 |
| claude-code | compileiq-author-objective-triton-same-score | 356,637 | 382,370 | -25,733 | -6.73% | skill 1/1; base 1/1 |
| codex | All cases | 560,699 | 1,864,520 | -1,303,821 | -69.93% | skill 8/8; base 8/8 |
| codex | compileiq-author-objective-canary | 34,804 | 51,486 | -16,682 | -32.40% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-fa4-cutedsl | 103,872 | 1,106,248 | -1,002,376 | -90.61% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-flashinfer | 50,823 | 35,503 | +15,320 | +43.15% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-legacy-nvcc-review | 72,609 | 75,220 | -2,611 | -3.47% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-mixed-space | 34,650 | 35,093 | -443 | -1.26% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-negative-scipy-objective | 18,099 | 17,928 | +171 | +0.95% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-negative-triton-autotune | 22,805 | 27,070 | -4,265 | -15.76% | skill 1/1; base 1/1 |
| codex | compileiq-author-objective-triton-same-score | 223,037 | 515,972 | -292,935 | -56.77% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,046,536 | 3,117,464 | -1,070,928 | -34.35% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-author-objective/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-author-objective/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (619 chars, recommend 50-150) (`skills/compileiq-author-objective/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/compileiq-author-objective/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/compileiq-author-objective/SKILL.md`)
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
