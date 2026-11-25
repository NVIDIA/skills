# Skill Benchmark: doca-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-setup`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:169f7c0489e994502eee43ed5716bf51cebe677b1bd8f3a68ecc6278038d804e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-setup | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-setup | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-setup | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-setup | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-setup | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-setup`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 98.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 55.0% → 100.0% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 52.1% → 100.0% (+47.9 points) | 83.3% → 100.0% (+16.7 points) |
| Efficiency | 93.7% — baseline ran, but no comparable score was available; uplift unavailable | 73.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 94.0% (-4.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.0% → 86.2% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 580,409 | 324,438 | +255,971 | +78.90% | skill 4/4; base 4/4 |
| claude-code | doca-setup.explicit-positive.v1 | 224,965 | 36,766 | +188,199 | +511.88% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v1 | 123,245 | 107,807 | +15,438 | +14.32% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v2 | 123,061 | 73,000 | +50,061 | +68.58% | skill 1/1; base 1/1 |
| claude-code | doca-setup.negative.v1 | 109,138 | 106,865 | +2,273 | +2.13% | skill 1/1; base 1/1 |
| codex | All cases | 431,863 | 123,756 | +308,107 | +248.96% | skill 4/4; base 4/4 |
| codex | doca-setup.explicit-positive.v1 | 161,252 | 40,489 | +120,763 | +298.26% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v1 | 133,584 | 35,072 | +98,512 | +280.89% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v2 | 106,891 | 34,920 | +71,971 | +206.10% | skill 1/1; base 1/1 |
| codex | doca-setup.negative.v1 | 30,136 | 13,275 | +16,861 | +127.01% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,012,272 | 448,194 | +564,078 | +125.86% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 81-81)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 94-94) (`CAPABILITIES.md:81`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across TASKS.md and references/command-appendix.md:
  "## Deferred task verbs" in TASKS.md (lines 263-268)
  vs "## Deferred task verbs" in references/command-appendix.md (lines 33-38) (`TASKS.md:263`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and TASKS.md:
  "# DOCA setup" in SKILL.md (lines 1-42)
  vs "## Instructions" in SKILL.md (lines 75-113)
  vs "## When to load this skill" in SKILL.md (lines 144-159)
  vs "## What this skill provides" in SKILL.md (lines 160-168)
  vs "## Related skills" in SKILL.md (lines 210-216)
  vs "# DOCA setup workflows" in TASKS.md (lines 1-24) (`SKILL.md:1`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in command-appendix.md (`skills/doca-setup/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/doca-setup/SKILL.md`)
- 12 additional finding(s) are available in the full evaluation artifacts.

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
