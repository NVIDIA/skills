# Skill Benchmark: daqiri-onboarding

> **Publication eligible**

## Publication Recommendation

- Evaluation source: self-assessed
- Quality assessment: SELF-ASSESSED
- Reported execution: completed
- Reported evaluation verdict: PASS
- CI checks allow publication with self-assessed evaluation evidence.
- CI did not run Tier 3. Scores, comparisons, and evaluation verdicts below are producer-reported. Scoring definitions reference Skill Evaluator 1.6.9 and do not set the CI publication gate.

Scoring definitions below are those of Skill Evaluator 1.6.9; older reports may use different formulas. Supplied scores are displayed unchanged.

## Evaluation Metadata

- Skill: `daqiri-onboarding`
- Validation status: `passed`
- Evaluation date: 2026-11-09
- Evaluator version: `daqiri-paired-cli-byoe-v4-external-minimal`
- Agents: Codex (`gpt-6.1-sol`), Claude (`claude-opus-5-5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: not recorded (legacy or non-live result)
- Attempts per task: 1
- Environment: not recorded (legacy or non-live result)
- Tier 2 evidence: required for publication
- Tier 3 evidence: self-assessed; CI did not run live evaluation

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| daqiri-onboarding | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| daqiri-onboarding | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| daqiri-onboarding | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| daqiri-onboarding | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| daqiri-onboarding | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `daqiri-onboarding`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Codex (Baseline → Skill Uplift) | Claude (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Efficiency | Not available | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Self-assessed evaluation | **SELF-ASSESSED** | Reported execution: completed; CI did not run Tier 3 |

Tier 3 validation was skipped and executed 0 checks.
- CI did not run Tier 3; evaluation source: self-assessed.

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/daqiri-onboarding/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/daqiri-onboarding/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in onboarding-flow.md (`skills/daqiri-onboarding/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/daqiri-onboarding/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/daqiri-onboarding/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
- Reference quality rule (not the CI publication gate): PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override that reference quality rule.
- The 100% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `assertion_pass_rate` (Assertion Pass Rate): additional evaluator signal.
- `task_success` (Task Success): additional evaluator signal.
- `scope_preservation` (Scope Preservation): additional evaluator signal.
- `skill_activation_alignment` (Skill Activation Alignment): additional evaluator signal.

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
