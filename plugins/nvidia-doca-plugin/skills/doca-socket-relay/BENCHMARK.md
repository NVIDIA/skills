# Skill Benchmark: doca-socket-relay

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-socket-relay`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:17dfd2642bf754704fb0dd534cd92cbf58fdf6040a4bec732e227ff10f358941` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-socket-relay | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-socket-relay | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-socket-relay | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-socket-relay | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-socket-relay | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-socket-relay`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 93.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 85.0% → 95.0% (+10.0 points) |
| Discoverability | Not available | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 62.5% → 100.0% (+37.5 points) |
| Efficiency | Not available | 77.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.3% → 87.2% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 638,563 | 358,733 | +279,830 | +78.01% | skill 4/4; base 4/4 |
| claude-code | doca-socket-relay.explicit-positive.v1 | 228,298 | 71,569 | +156,729 | +218.99% | skill 1/1; base 1/1 |
| claude-code | doca-socket-relay.implicit-positive.v1 | 172,333 | 108,647 | +63,686 | +58.62% | skill 1/1; base 1/1 |
| claude-code | doca-socket-relay.implicit-positive.v2 | 165,376 | 107,633 | +57,743 | +53.65% | skill 1/1; base 1/1 |
| claude-code | doca-socket-relay.negative.v1 | 72,556 | 70,884 | +1,672 | +2.36% | skill 1/1; base 1/1 |
| codex | All cases | 365,440 | 185,058 | +180,382 | +97.47% | skill 4/4; base 4/4 |
| codex | doca-socket-relay.explicit-positive.v1 | 126,010 | 52,146 | +73,864 | +141.65% | skill 1/1; base 1/1 |
| codex | doca-socket-relay.implicit-positive.v1 | 99,842 | 42,443 | +57,399 | +135.24% | skill 1/1; base 1/1 |
| codex | doca-socket-relay.implicit-positive.v2 | 125,916 | 45,202 | +80,714 | +178.56% | skill 1/1; base 1/1 |
| codex | doca-socket-relay.negative.v1 | 13,672 | 45,267 | -31,595 | -69.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,004,003 | 543,791 | +460,212 | +84.63% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Pattern overview" in CAPABILITIES.md (lines 32-48)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 94-109)
  vs "## Safety policy" in CAPABILITIES.md (lines 315-373) (`CAPABILITIES.md:32`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "# DOCA Socket Relay — Capabilities" in CAPABILITIES.md (lines 1-18)
  vs "## Pattern overview" in CAPABILITIES.md (lines 23-30)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 63-72)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 77-81)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 156-237)
  vs "## Instructions" in SKILL.md (lines 32-61)
  vs "## Example questions this skill answers well" in SKILL.md (lines 62-119)
  vs "## When to load this skill" in SKILL.md (lines 120-151)
  vs "## What this skill provides" in SKILL.md (lines 152-185)
  vs "## Related skills" in SKILL.md (lines 225-277)
  vs "# DOCA Socket Relay — Tasks" in TASKS.md (lines 1-28)
  vs "## configure" in TASKS.md (lines 41-104)
  vs "## run" in TASKS.md (lines 249-308)
  vs "## debug" in TASKS.md (lines 405-489)
  vs "## Command appendix" in TASKS.md (lines 574-582) (`CAPABILITIES.md:1`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/tools/doca-socket-relay/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-socket-relay/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-socket-relay`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
