# Skill Benchmark: doca-bf3-deployment

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-bf3-deployment`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 3 evaluation tasks (2 positive, 1 negative)
- Dataset digest: `sha256:c3d6c876b2c75b4892b8cad068e3cffbe5d78b34b73f1ee3a8be5c76fc675e88` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-bf3-deployment | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-bf3-deployment | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-bf3-deployment | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-bf3-deployment | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-bf3-deployment | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-bf3-deployment`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 89.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 66.7% → 100.0% (+33.3 points) | 73.3% → 80.0% (+6.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.8% → 100.0% (+22.2 points) | 77.8% → 100.0% (+22.2 points) |
| Efficiency | 83.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.4% → 93.8% (-3.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 81.7% → 82.6% (+0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 662,031 | 554,766 | +107,265 | +19.34% | skill 3/3; base 3/3 |
| claude-code | doca-bf3-deployment.explicit-positive.v1 | 205,781 | 145,015 | +60,766 | +41.90% | skill 1/1; base 1/1 |
| claude-code | doca-bf3-deployment.implicit-positive.v1 | 302,589 | 301,282 | +1,307 | +0.43% | skill 1/1; base 1/1 |
| claude-code | doca-bf3-deployment.negative.v1 | 153,661 | 108,469 | +45,192 | +41.66% | skill 1/1; base 1/1 |
| codex | All cases | 260,512 | 218,054 | +42,458 | +19.47% | skill 3/3; base 3/3 |
| codex | doca-bf3-deployment.explicit-positive.v1 | 105,378 | 73,181 | +32,197 | +44.00% | skill 1/1; base 1/1 |
| codex | doca-bf3-deployment.implicit-positive.v1 | 102,735 | 89,176 | +13,559 | +15.20% | skill 1/1; base 1/1 |
| codex | doca-bf3-deployment.negative.v1 | 52,399 | 55,697 | -3,298 | -5.92% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 922,543 | 772,820 | +149,723 | +19.37% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 184-224)
  vs "## debug" in TASKS.md (lines 296-328) (`CAPABILITIES.md:184`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md and references/details.md:
  "# DOCA BlueField-3 (BF3) deployment — Capabilities" in CAPABILITIES.md (lines 1-26)
  vs "## Pattern overview" in CAPABILITIES.md (lines 27-63)
  vs "## Instructions" in SKILL.md (lines 13-48)
  vs "## Audience" in SKILL.md (lines 49-89)
  vs "## When to load this skill" in SKILL.md (lines 90-139)
  vs "## What this skill provides" in SKILL.md (lines 140-183)
  vs "# DOCA BlueField-3 (BF3) deployment — Tasks" in TASKS.md (lines 1-28)
  vs "## build" in TASKS.md (lines 94-116)
  vs "## Deferred task verbs" in TASKS.md (lines 348-385)
  vs "## Related skills" in references/details.md (lines 109-150) (`CAPABILITIES.md:1`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/doca-bf3-deployment/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in details.md (`skills/doca-bf3-deployment/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/doca-bf3-deployment/SKILL.md`)
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
