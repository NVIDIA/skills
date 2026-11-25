# Skill Benchmark: doca-dts

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-dts`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:1f0a5dec87b17e5f794f9aa7e1c80561854711f5e5b6fdd87c357e0f4a3b0a8c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-dts | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dts | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dts | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dts | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dts | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-dts`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 83.3% → 91.7% (+8.4 points) |
| Correctness | 50.0% → 83.3% (+33.3 points) | 66.7% → 90.0% (+23.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 15.3% → 68.8% (+53.5 points) | 47.2% → 78.5% (+31.3 points) |
| Efficiency | 73.2% — baseline ran, but no comparable score was available; uplift unavailable | 65.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 92.7% (-5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 79.2% → 80.1% (+0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,058,488 | 509,884 | +548,604 | +107.59% | skill 6/6; base 6/6 |
| claude-code | doca-dts.explicit-positive.v1 | 232,909 | 36,640 | +196,269 | +535.67% | skill 1/1; base 1/1 |
| claude-code | doca-dts.implicit-positive.v1 | 220,637 | 72,129 | +148,508 | +205.89% | skill 1/1; base 1/1 |
| claude-code | doca-dts.implicit-positive.v2 | 271,368 | 108,165 | +163,203 | +150.88% | skill 1/1; base 1/1 |
| claude-code | doca-dts.implicit-positive.v3 | 220,810 | 108,808 | +112,002 | +102.94% | skill 1/1; base 1/1 |
| claude-code | doca-dts.negative.v1 | 73,609 | 73,103 | +506 | +0.69% | skill 1/1; base 1/1 |
| claude-code | doca-dts.negative.v2 | 39,155 | 111,039 | -71,884 | -64.74% | skill 1/1; base 1/1 |
| codex | All cases | 664,824 | 482,382 | +182,442 | +37.82% | skill 6/6; base 6/6 |
| codex | doca-dts.explicit-positive.v1 | 140,607 | 165,444 | -24,837 | -15.01% | skill 1/1; base 1/1 |
| codex | doca-dts.implicit-positive.v1 | 115,612 | 30,765 | +84,847 | +275.79% | skill 1/1; base 1/1 |
| codex | doca-dts.implicit-positive.v2 | 79,084 | 34,860 | +44,224 | +126.86% | skill 1/1; base 1/1 |
| codex | doca-dts.implicit-positive.v3 | 153,526 | 39,398 | +114,128 | +289.68% | skill 1/1; base 1/1 |
| codex | doca-dts.negative.v1 | 47,261 | 69,183 | -21,922 | -31.69% | skill 1/1; base 1/1 |
| codex | doca-dts.negative.v2 | 128,734 | 142,732 | -13,998 | -9.81% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,723,312 | 992,266 | +731,046 | +73.67% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 20 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/services/doca-dts/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/services/doca-dts/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/services/doca-dts/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/services/doca-dts`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/services/doca-dts/SKILL.md`)
- 15 additional finding(s) are available in the full evaluation artifacts.

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
