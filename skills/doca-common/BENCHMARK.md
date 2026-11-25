# Skill Benchmark: doca-common

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-common`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:9e22539842a371b035b6820995478cb465ac90fc3e65d8d974e6798810076bb7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-common | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-common | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-common | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-common | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-common | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-common`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 90.0% (+40.0 points) | 65.0% → 100.0% (+35.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.0% → 64.6% (+39.6 points) | 41.7% → 95.8% (+54.1 points) |
| Efficiency | 79.5% — baseline ran, but no comparable score was available; uplift unavailable | 69.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 92.3% (-6.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.5% → 84.3% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 789,521 | 287,483 | +502,038 | +174.63% | skill 4/4; base 4/4 |
| claude-code | doca-common.explicit-positive.v1 | 213,116 | 71,857 | +141,259 | +196.58% | skill 1/1; base 1/1 |
| claude-code | doca-common.implicit-positive.v1 | 212,856 | 107,460 | +105,396 | +98.08% | skill 1/1; base 1/1 |
| claude-code | doca-common.implicit-positive.v2 | 326,798 | 72,647 | +254,151 | +349.84% | skill 1/1; base 1/1 |
| claude-code | doca-common.negative.v1 | 36,751 | 35,519 | +1,232 | +3.47% | skill 1/1; base 1/1 |
| codex | All cases | 368,517 | 153,231 | +215,286 | +140.50% | skill 4/4; base 4/4 |
| codex | doca-common.explicit-positive.v1 | 103,321 | 48,466 | +54,855 | +113.18% | skill 1/1; base 1/1 |
| codex | doca-common.implicit-positive.v1 | 75,702 | 27,075 | +48,627 | +179.60% | skill 1/1; base 1/1 |
| codex | doca-common.implicit-positive.v2 | 154,941 | 47,578 | +107,363 | +225.66% | skill 1/1; base 1/1 |
| codex | doca-common.negative.v1 | 34,553 | 30,112 | +4,441 | +14.75% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,158,038 | 440,714 | +717,324 | +162.76% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## buf" in CAPABILITIES.md (lines 155-159)
  vs "## buf" in CAPABILITIES.md (lines 160-160)
  vs "## dev" in CAPABILITIES.md (lines 332-336) (`CAPABILITIES.md:155`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## dev" in CAPABILITIES.md (lines 307-311)
  vs "## dev" in CAPABILITIES.md (lines 312-318) (`CAPABILITIES.md:307`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found within TASKS.md:
  "## configure" in TASKS.md (lines 63-110)
  vs "## use" in TASKS.md (lines 367-419) (`TASKS.md:63`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md:
  "## Pattern overview" in CAPABILITIES.md (lines 27-34)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 65-78)
  vs "## Instructions" in SKILL.md (lines 29-52)
  vs "## Example questions this skill answers well" in SKILL.md (lines 53-107)
  vs "## When to load this skill" in SKILL.md (lines 140-173) (`CAPABILITIES.md:27`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-common/SKILL.md`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
