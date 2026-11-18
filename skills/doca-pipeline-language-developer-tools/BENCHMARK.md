# Skill Benchmark: doca-pipeline-language-developer-tools

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-pipeline-language-developer-tools`
- Validation status: `passed`
- Evaluation date: 2026-11-08
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (3 positive, 2 negative)
- Dataset digest: `sha256:b8bbbcc052479714d9ea60279e534806882df2d3c9535b6b49a539ae925b6c5d` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-pipeline-language-developer-tools | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| doca-pipeline-language-developer-tools | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| doca-pipeline-language-developer-tools | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| doca-pipeline-language-developer-tools | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| doca-pipeline-language-developer-tools | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `doca-pipeline-language-developer-tools`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 44.0% → 100.0% (+56.0 points) | 60.0% → 100.0% (+40.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.0% → 95.0% (+70.0 points) | 35.0% → 100.0% (+65.0 points) |
| Efficiency | 87.3% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 96.3% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.3% → 88.0% (+3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 721,572 | 575,432 | +146,140 | +25.40% | skill 5/5; base 5/5 |
| claude-code | doca-pipeline-language-developer-tools.explicit-positive.v1 | 245,120 | 108,548 | +136,572 | +125.82% | skill 1/1; base 1/1 |
| claude-code | doca-pipeline-language-developer-tools.implicit-positive.v1 | 202,949 | 107,811 | +95,138 | +88.25% | skill 1/1; base 1/1 |
| claude-code | doca-pipeline-language-developer-tools.implicit-positive.v2 | 200,656 | 106,831 | +93,825 | +87.83% | skill 1/1; base 1/1 |
| claude-code | doca-pipeline-language-developer-tools.negative.v1 | 36,638 | 144,207 | -107,569 | -74.59% | skill 1/1; base 1/1 |
| claude-code | doca-pipeline-language-developer-tools.negative.v2 | 36,209 | 108,035 | -71,826 | -66.48% | skill 1/1; base 1/1 |
| codex | All cases | 347,932 | 254,844 | +93,088 | +36.53% | skill 5/5; base 5/5 |
| codex | doca-pipeline-language-developer-tools.explicit-positive.v1 | 105,350 | 38,943 | +66,407 | +170.52% | skill 1/1; base 1/1 |
| codex | doca-pipeline-language-developer-tools.implicit-positive.v1 | 81,008 | 39,105 | +41,903 | +107.16% | skill 1/1; base 1/1 |
| codex | doca-pipeline-language-developer-tools.implicit-positive.v2 | 101,430 | 58,540 | +42,890 | +73.27% | skill 1/1; base 1/1 |
| codex | doca-pipeline-language-developer-tools.negative.v1 | 30,135 | 52,056 | -21,921 | -42.11% | skill 1/1; base 1/1 |
| codex | doca-pipeline-language-developer-tools.negative.v2 | 30,009 | 66,200 | -36,191 | -54.67% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,069,504 | 830,276 | +239,228 | +28.81% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "# DOCA Pipeline Language Developer Tools — Capabilities" in CAPABILITIES.md (lines 1-19)
  vs "## Pattern overview" in CAPABILITIES.md (lines 24-31)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 56-66)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 67-74)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 79-86)
  vs "## Version compatibility" in CAPABILITIES.md (lines 112-121)
  vs "## Observability" in CAPABILITIES.md (lines 135-146)
  vs "## Scope boundaries" in SKILL.md (lines 3-15)
  vs "## Examples" in SKILL.md (lines 16-31)
  vs "## Quick facts" in SKILL.md (lines 32-71)
  vs "## Instructions" in SKILL.md (lines 72-102)
  vs "## Example questions this skill answers well" in SKILL.md (lines 103-133)
  vs "## When to load this skill" in SKILL.md (lines 134-160)
  vs "## What this skill provides" in SKILL.md (lines 161-189)
  vs "## Loading order" in SKILL.md (lines 213-224)
  vs "## Related skills" in SKILL.md (lines 225-248)
  vs "## debug" in TASKS.md (lines 77-129)
  vs "## Cross-cutting" in TASKS.md (lines 289-308) (`CAPABILITIES.md:1`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Pattern overview" in CAPABILITIES.md (lines 33-49)
  vs "## Safety policy" in CAPABILITIES.md (lines 147-163) (`CAPABILITIES.md:33`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/tools/doca-pipeline-language-developer-tools/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-pipeline-language-developer-tools/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-pipeline-language-developer-tools`)
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
