# Skill Benchmark: newton-performance

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `newton-performance`
- Validation status: `passed`
- Evaluation date: 2026-11-11
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 12 evaluation tasks (11 positive, 1 negative)
- Dataset digest: `sha256:6ecce6eb8f7b83630e185404459abc12d2a3d0ce0ab5c386ad9dbe1f77ef09f9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| newton-performance | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| newton-performance | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| newton-performance | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| newton-performance | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| newton-performance | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `newton-performance`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 87.3% — baseline ran, but no comparable score was available; uplift unavailable | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 88.3% → 91.7% (+3.4 points) | 93.3% → 98.3% (+5.0 points) |
| Discoverability | 65.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 87.5% → 83.8% (-3.7 points) | 94.2% → 95.8% (+1.6 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 54.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 97.6% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 85.5% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,256,391 | 1,016,996 | +239,395 | +23.54% | skill 12/12; base 12/12 |
| claude-code | newton-performance-blocker | 187,700 | 144,854 | +42,846 | +29.58% | skill 1/1; base 1/1 |
| claude-code | newton-performance-context-many-worlds | 150,891 | 110,428 | +40,463 | +36.64% | skill 1/1; base 1/1 |
| claude-code | newton-performance-coupled-sweep | 113,462 | 110,372 | +3,090 | +2.80% | skill 1/1; base 1/1 |
| claude-code | newton-performance-explicit-review | 76,185 | 37,891 | +38,294 | +101.06% | skill 1/1; base 1/1 |
| claude-code | newton-performance-implicit-slow-motion | 37,150 | 36,931 | +219 | +0.59% | skill 1/1; base 1/1 |
| claude-code | newton-performance-implicit-startup | 74,826 | 37,608 | +37,218 | +98.96% | skill 1/1; base 1/1 |
| claude-code | newton-performance-multiturn-into-scope | 149,285 | 72,355 | +76,930 | +106.32% | skill 1/1; base 1/1 |
| claude-code | newton-performance-multiturn-out-of-scope | 36,220 | 107,889 | -71,669 | -66.43% | skill 1/1; base 1/1 |
| claude-code | newton-performance-negative-correctness | 109,246 | 109,220 | +26 | +0.02% | skill 1/1; base 1/1 |
| claude-code | newton-performance-negative-solver-api | 142,752 | 71,261 | +71,491 | +100.32% | skill 1/1; base 1/1 |
| claude-code | newton-performance-negative-usage | 106,087 | 105,972 | +115 | +0.11% | skill 1/1; base 1/1 |
| claude-code | newton-performance-negative-warp-kernel | 72,587 | 72,215 | +372 | +0.52% | skill 1/1; base 1/1 |
| codex | All cases | 663,094 | 413,055 | +250,039 | +60.53% | skill 12/12; base 12/12 |
| codex | newton-performance-blocker | 89,357 | 58,119 | +31,238 | +53.75% | skill 1/1; base 1/1 |
| codex | newton-performance-context-many-worlds | 149,774 | 56,632 | +93,142 | +164.47% | skill 1/1; base 1/1 |
| codex | newton-performance-coupled-sweep | 68,463 | 58,925 | +9,538 | +16.19% | skill 1/1; base 1/1 |
| codex | newton-performance-explicit-review | 59,220 | 27,088 | +32,132 | +118.62% | skill 1/1; base 1/1 |
| codex | newton-performance-implicit-slow-motion | 45,124 | 30,648 | +14,476 | +47.23% | skill 1/1; base 1/1 |
| codex | newton-performance-implicit-startup | 45,450 | 13,816 | +31,634 | +228.97% | skill 1/1; base 1/1 |
| codex | newton-performance-multiturn-into-scope | 66,054 | 26,310 | +39,744 | +151.06% | skill 1/1; base 1/1 |
| codex | newton-performance-multiturn-out-of-scope | 21,308 | 18,111 | +3,197 | +17.65% | skill 1/1; base 1/1 |
| codex | newton-performance-negative-correctness | 39,037 | 53,005 | -13,968 | -26.35% | skill 1/1; base 1/1 |
| codex | newton-performance-negative-solver-api | 30,522 | 30,295 | +227 | +0.75% | skill 1/1; base 1/1 |
| codex | newton-performance-negative-usage | 17,537 | 17,478 | +59 | +0.34% | skill 1/1; base 1/1 |
| codex | newton-performance-negative-warp-kernel | 31,248 | 22,628 | +8,620 | +38.09% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,919,485 | 1,430,051 | +489,434 | +34.22% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/newton-performance/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/newton-performance/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/newton-performance/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/newton-performance/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/newton-performance/SKILL.md`)
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
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
