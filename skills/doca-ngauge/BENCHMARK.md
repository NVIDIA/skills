# Skill Benchmark: doca-ngauge

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-ngauge`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (3 positive, 2 negative)
- Dataset digest: `sha256:1a4e76c3f1c9facd6fd004fd66cee224aecadc6badf7aa601ab818f934fa77ac` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-ngauge | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-ngauge | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-ngauge | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-ngauge | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-ngauge | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `doca-ngauge`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 91.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 48.0% → 100.0% (+52.0 points) | 92.0% → 100.0% (+8.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.0% → 95.0% (+55.0 points) | 65.0% → 95.0% (+30.0 points) |
| Efficiency | 91.2% — baseline ran, but no comparable score was available; uplift unavailable | 69.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 95.7% (-2.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 85.4% → 86.4% (+1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 644,068 | 395,429 | +248,639 | +62.88% | skill 5/5; base 5/5 |
| claude-code | doca-ngauge.explicit-positive.v1 | 163,393 | 108,249 | +55,144 | +50.94% | skill 1/1; base 1/1 |
| claude-code | doca-ngauge.implicit-positive.v1 | 163,188 | 108,250 | +54,938 | +50.75% | skill 1/1; base 1/1 |
| claude-code | doca-ngauge.implicit-positive.v2 | 244,908 | 108,001 | +136,907 | +126.76% | skill 1/1; base 1/1 |
| claude-code | doca-ngauge.negative.v1 | 36,481 | 35,537 | +944 | +2.66% | skill 1/1; base 1/1 |
| claude-code | doca-ngauge.negative.v2 | 36,098 | 35,392 | +706 | +1.99% | skill 1/1; base 1/1 |
| codex | All cases | 317,285 | 213,879 | +103,406 | +48.35% | skill 5/5; base 5/5 |
| codex | doca-ngauge.explicit-positive.v1 | 77,255 | 68,735 | +8,520 | +12.40% | skill 1/1; base 1/1 |
| codex | doca-ngauge.implicit-positive.v1 | 75,533 | 54,632 | +20,901 | +38.26% | skill 1/1; base 1/1 |
| codex | doca-ngauge.implicit-positive.v2 | 94,232 | 25,502 | +68,730 | +269.51% | skill 1/1; base 1/1 |
| codex | doca-ngauge.negative.v1 | 26,604 | 30,194 | -3,590 | -11.89% | skill 1/1; base 1/1 |
| codex | doca-ngauge.negative.v2 | 43,661 | 34,816 | +8,845 | +25.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 961,353 | 609,308 | +352,045 | +57.78% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "# DOCA ngauge — Capabilities" in CAPABILITIES.md (lines 1-18)
  vs "## Pattern overview" in CAPABILITIES.md (lines 19-44)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 54-59)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 63-68)
  vs "## Examples" in SKILL.md (lines 15-27)
  vs "## Quick facts" in SKILL.md (lines 28-81)
  vs "## Instructions" in SKILL.md (lines 82-108)
  vs "## Example questions this skill answers well" in SKILL.md (lines 109-139)
  vs "## When to load this skill" in SKILL.md (lines 140-163)
  vs "## What this skill provides" in SKILL.md (lines 164-188)
  vs "## configure" in TASKS.md (lines 28-78)
  vs "## run" in TASKS.md (lines 97-162)
  vs "## debug" in TASKS.md (lines 189-220) (`CAPABILITIES.md:1`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and TASKS.md:
  "## Related skills" in SKILL.md (lines 225-259)
  vs "## Cross-cutting" in TASKS.md (lines 307-329) (`SKILL.md:225`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-ngauge/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-ngauge`)
- **MEDIUM** SECURITY/analysis-evasion (AE4): analysis-evasion: Suspicious Unicode normalization or mixed-script content (`CAPABILITIES.md:1`)
- 23 additional finding(s) are available in the full evaluation artifacts.

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
