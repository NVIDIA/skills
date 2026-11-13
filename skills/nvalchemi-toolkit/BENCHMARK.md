# Skill Benchmark: nvalchemi-toolkit

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvalchemi-toolkit`
- Validation status: `passed`
- Evaluation date: 2026-11-13
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (7 positive, 2 negative)
- Dataset digest: `sha256:c096747a831a786dceff6fcf5d46191b44347068431663c2a7e66efbf5593c7a` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvalchemi-toolkit | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvalchemi-toolkit`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 92.4% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 82.2% → 95.6% (+13.4 points) | 88.9% → 97.8% (+8.9 points) |
| Discoverability | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 82.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 73.0% → 88.8% (+15.8 points) | 78.5% → 88.2% (+9.7 points) |
| Efficiency | 82.0% — baseline ran, but no comparable score was available; uplift unavailable | 77.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 80.9% → 85.0% (+4.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.3% → 84.6% (+1.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,717,204 | 7,133,763 | -3,416,559 | -47.89% | skill 9/9; base 9/9 |
| claude-code | comp-01 | 487,046 | 1,539,010 | -1,051,964 | -68.35% | skill 1/1; base 1/1 |
| claude-code | imp-01 | 678,417 | 1,022,654 | -344,237 | -33.66% | skill 1/1; base 1/1 |
| claude-code | imp-02 | 938,266 | 2,312,577 | -1,374,311 | -59.43% | skill 1/1; base 1/1 |
| claude-code | model-01 | 599,446 | 807,908 | -208,462 | -25.80% | skill 1/1; base 1/1 |
| claude-code | neg-01 | 184,939 | 181,871 | +3,068 | +1.69% | skill 1/1; base 1/1 |
| claude-code | neg-03 | 269,177 | 267,605 | +1,572 | +0.59% | skill 1/1; base 1/1 |
| claude-code | scope-01 | 241,406 | 279,572 | -38,166 | -13.65% | skill 1/1; base 1/1 |
| claude-code | scope-02 | 238,990 | 611,212 | -372,222 | -60.90% | skill 1/1; base 1/1 |
| claude-code | scope-03 | 79,517 | 111,354 | -31,837 | -28.59% | skill 1/1; base 1/1 |
| codex | All cases | 5,202,602 | 7,665,006 | -2,462,404 | -32.13% | skill 9/9; base 9/9 |
| codex | comp-01 | 915,913 | 1,224,314 | -308,401 | -25.19% | skill 1/1; base 1/1 |
| codex | imp-01 | 1,269,600 | 1,448,475 | -178,875 | -12.35% | skill 1/1; base 1/1 |
| codex | imp-02 | 1,890,904 | 1,409,250 | +481,654 | +34.18% | skill 1/1; base 1/1 |
| codex | model-01 | 280,676 | 472,826 | -192,150 | -40.64% | skill 1/1; base 1/1 |
| codex | neg-01 | 74,582 | 72,159 | +2,423 | +3.36% | skill 1/1; base 1/1 |
| codex | neg-03 | 548,029 | 406,804 | +141,225 | +34.72% | skill 1/1; base 1/1 |
| codex | scope-01 | 68,047 | 47,641 | +20,406 | +42.83% | skill 1/1; base 1/1 |
| codex | scope-02 | 105,310 | 2,538,169 | -2,432,859 | -95.85% | skill 1/1; base 1/1 |
| codex | scope-03 | 49,541 | 45,368 | +4,173 | +9.20% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,919,806 | 14,798,769 | -5,878,963 | -39.73% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvalchemi-toolkit/SKILL.md`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:86`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:218`)
- **LOW** QUALITY/quality_discoverability: Description very long (703 chars, recommend 50-150) (`skills/nvalchemi-toolkit/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nvalchemi-toolkit/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
