# Skill Benchmark: doca-firefly

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-firefly`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:507d57859a365707cbf6ff537ee239128427ca02a501116cbf00d3ab681150f7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-firefly | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-firefly | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-firefly | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-firefly | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-firefly | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-firefly`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 90.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 70.0% → 85.0% (+15.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 56.3% → 100.0% (+43.7 points) |
| Efficiency | 82.2% — baseline ran, but no comparable score was available; uplift unavailable | 73.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 92.2% (-5.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.8% → 84.8% (±0.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 758,878 | 289,883 | +468,995 | +161.79% | skill 4/4; base 4/4 |
| claude-code | doca-firefly.explicit-positive.v1 | 272,403 | 73,304 | +199,099 | +271.61% | skill 1/1; base 1/1 |
| claude-code | doca-firefly.implicit-positive.v1 | 281,421 | 73,231 | +208,190 | +284.29% | skill 1/1; base 1/1 |
| claude-code | doca-firefly.implicit-positive.v2 | 168,502 | 72,010 | +96,492 | +134.00% | skill 1/1; base 1/1 |
| claude-code | doca-firefly.negative.v1 | 36,552 | 71,338 | -34,786 | -48.76% | skill 1/1; base 1/1 |
| codex | All cases | 387,276 | 198,773 | +188,503 | +94.83% | skill 4/4; base 4/4 |
| codex | doca-firefly.explicit-positive.v1 | 133,403 | 48,095 | +85,308 | +177.37% | skill 1/1; base 1/1 |
| codex | doca-firefly.implicit-positive.v1 | 107,086 | 57,999 | +49,087 | +84.63% | skill 1/1; base 1/1 |
| codex | doca-firefly.implicit-positive.v2 | 107,977 | 70,762 | +37,215 | +52.59% | skill 1/1; base 1/1 |
| codex | doca-firefly.negative.v1 | 38,810 | 21,917 | +16,893 | +77.08% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,146,154 | 488,656 | +657,498 | +134.55% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "## Pattern overview" in CAPABILITIES.md (lines 17-46)
  vs "### Service shape" in CAPABILITIES.md (lines 49-78)
  vs "## Deployment shape" in CAPABILITIES.md (lines 79-101)
  vs "### Four-axis PTP configuration" in CAPABILITIES.md (lines 103-108)
  vs "### Pairing with time-sensitive consumers" in CAPABILITIES.md (lines 124-142)
  vs "## Version compatibility" in CAPABILITIES.md (lines 155-191)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 192-211)
  vs "## Safety policy" in CAPABILITIES.md (lines 265-316)
  vs "## Instructions" in SKILL.md (lines 65-80)
  vs "## Example questions this skill answers well" in SKILL.md (lines 85-136)
  vs "## Audience" in SKILL.md (lines 137-167)
  vs "## When to load this skill" in SKILL.md (lines 168-209)
  vs "## Related skills" in SKILL.md (lines 281-325)
  vs "## configure" in TASKS.md (lines 21-132)
  vs "## build" in TASKS.md (lines 136-165)
  vs "## run" in TASKS.md (lines 211-257)
  vs "## run" in TASKS.md (lines 258-262)
  vs "## debug" in TASKS.md (lines 340-399)
  vs "## Deferred task verbs" in TASKS.md (lines 467-501)
  vs "## Cross-cutting" in TASKS.md (lines 502-521) (`CAPABILITIES.md:17`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/services/doca-firefly/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/services/doca-firefly/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/services/doca-firefly/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/services/doca-firefly`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
