# Skill Benchmark: make-data-pipeline

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `make-data-pipeline`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (8 positive, 2 negative)
- Dataset digest: `sha256:28abb53723ba2e11254de9d0adb3e076e5715b7723e28b1222f17e0dc540a82c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| make-data-pipeline | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| make-data-pipeline | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| make-data-pipeline | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| make-data-pipeline | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| make-data-pipeline | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `make-data-pipeline`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 90.1% — baseline ran, but no comparable score was available; uplift unavailable | 94.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 92.0% → 100.0% (+8.0 points) | 88.0% → 98.0% (+10.0 points) |
| Discoverability | 53.8% — baseline ran, but no comparable score was available; uplift unavailable | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 91.7% → 100.0% (+8.3 points) | 90.0% → 100.0% (+10.0 points) |
| Efficiency | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 96.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.1% → 98.0% (+0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.1% → 92.5% (-0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,386,569 | 1,355,898 | +30,671 | +2.26% | skill 10/10; base 10/10 |
| claude-code | make-data-pipeline-001-coverage | 156,441 | 144,154 | +12,287 | +8.52% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-002-level1-gate | 141,965 | 141,342 | +623 | +0.44% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-003-runtime-args | 183,277 | 106,175 | +77,102 | +72.62% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-004-score-gate | 194,172 | 218,868 | -24,696 | -11.28% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-005-demo-is-not-level2 | 106,956 | 178,963 | -72,007 | -40.24% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-006-tolerance | 106,363 | 106,140 | +223 | +0.21% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-007-higher-score | 107,424 | 106,906 | +518 | +0.48% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-008-module-swap-negative | 141,322 | 106,002 | +35,320 | +33.32% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-009-benchmark-negative | 142,529 | 141,597 | +932 | +0.66% | skill 1/1; base 1/1 |
| claude-code | make-data-pipeline-010-revision | 106,120 | 105,751 | +369 | +0.35% | skill 1/1; base 1/1 |
| codex | All cases | 754,275 | 724,071 | +30,204 | +4.17% | skill 10/10; base 10/10 |
| codex | make-data-pipeline-001-coverage | 81,178 | 86,291 | -5,113 | -5.93% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-002-level1-gate | 93,922 | 68,282 | +25,640 | +37.55% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-003-runtime-args | 94,659 | 84,306 | +10,353 | +12.28% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-004-score-gate | 98,884 | 102,206 | -3,322 | -3.25% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-005-demo-is-not-level2 | 76,904 | 83,443 | -6,539 | -7.84% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-006-tolerance | 54,835 | 54,214 | +621 | +1.15% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-007-higher-score | 54,605 | 54,114 | +491 | +0.91% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-008-module-swap-negative | 54,141 | 54,084 | +57 | +0.11% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-009-benchmark-negative | 67,946 | 69,494 | -1,548 | -2.23% | skill 1/1; base 1/1 |
| codex | make-data-pipeline-010-revision | 77,201 | 67,637 | +9,564 | +14.14% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,140,844 | 2,079,969 | +60,875 | +2.93% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/make-data-pipeline/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (317 chars, recommend 50-150) (`skills/make-data-pipeline/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/make-data-pipeline/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/make-data-pipeline/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/make-data-pipeline/SKILL.md`)
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
