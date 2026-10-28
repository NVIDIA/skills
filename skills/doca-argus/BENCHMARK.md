# Skill Benchmark: doca-argus

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-argus`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:e2b075fe2be9760c3a342f783c65ee6bbfa9912f5f08a84aadaa925b77cbe4e4` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-argus | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-argus | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-argus | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-argus | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-argus | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `doca-argus`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.6% — baseline ran, but no comparable score was available; uplift unavailable | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 70.0% → 100.0% (+30.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 62.5% → 100.0% (+37.5 points) |
| Efficiency | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | 76.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 94.1% (-4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.3% → 86.2% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 663,468 | 286,962 | +376,506 | +131.20% | skill 4/4; base 4/4 |
| claude-code | doca-argus.explicit-positive.v1 | 210,209 | 71,287 | +138,922 | +194.88% | skill 1/1; base 1/1 |
| claude-code | doca-argus.implicit-positive.v1 | 208,115 | 107,500 | +100,615 | +93.60% | skill 1/1; base 1/1 |
| claude-code | doca-argus.implicit-positive.v2 | 208,473 | 71,983 | +136,490 | +189.61% | skill 1/1; base 1/1 |
| claude-code | doca-argus.negative.v1 | 36,671 | 36,192 | +479 | +1.32% | skill 1/1; base 1/1 |
| codex | All cases | 383,418 | 183,925 | +199,493 | +108.46% | skill 4/4; base 4/4 |
| codex | doca-argus.explicit-positive.v1 | 120,155 | 66,829 | +53,326 | +79.79% | skill 1/1; base 1/1 |
| codex | doca-argus.implicit-positive.v1 | 101,549 | 59,167 | +42,382 | +71.63% | skill 1/1; base 1/1 |
| codex | doca-argus.implicit-positive.v2 | 127,475 | 26,687 | +100,788 | +377.67% | skill 1/1; base 1/1 |
| codex | doca-argus.negative.v1 | 34,239 | 31,242 | +2,997 | +9.59% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,046,886 | 470,887 | +575,999 | +122.32% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/services/doca-argus`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "## Pattern overview" in CAPABILITIES.md (lines 17-51)
  vs "### Service shape" in CAPABILITIES.md (lines 54-91)
  vs "## Deployment shape" in CAPABILITIES.md (lines 92-117)
  vs "## Safety policy" in CAPABILITIES.md (lines 289-377)
  vs "## Examples" in SKILL.md (lines 3-11)
  vs "## Instructions" in SKILL.md (lines 12-54)
  vs "## Example questions this skill answers well" in SKILL.md (lines 55-106)
  vs "## Audience" in SKILL.md (lines 107-142)
  vs "## When to load this skill" in SKILL.md (lines 143-178)
  vs "## What this skill provides" in SKILL.md (lines 179-209)
  vs "## Loading order" in SKILL.md (lines 249-263)
  vs "## Related skills" in SKILL.md (lines 264-312)
  vs "## configure" in TASKS.md (lines 23-105)
  vs "## build" in TASKS.md (lines 109-142)
  vs "## Command appendix" in TASKS.md (lines 456-471)
  vs "## Deferred task verbs" in TASKS.md (lines 479-517)
  vs "## Cross-cutting" in TASKS.md (lines 518-548) (`CAPABILITIES.md:17`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/services/doca-argus/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/services/doca-argus`)
- **LOW** QUALITY/quality_discoverability: Description very long (977 chars, recommend 50-150) (`skills/services/doca-argus/SKILL.md`)
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
