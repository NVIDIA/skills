# Skill Benchmark: doca-perftest

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-perftest`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:43ee35f3cdeb638111792c9166ce0f14e4d89def25865872b518f9029087d974` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-perftest | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-perftest | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-perftest | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-perftest | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-perftest | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `doca-perftest`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 78.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 52.0% → 96.0% (+44.0 points) | 60.0% → 88.0% (+28.0 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.0% → 65.0% (+35.0 points) | 35.0% → 65.0% (+30.0 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 59.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 97.2% (-0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.5% → 85.9% (+1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 598,414 | 402,074 | +196,340 | +48.83% | skill 5/5; base 5/5 |
| claude-code | doca-perftest.explicit-positive.v1 | 113,761 | 36,385 | +77,376 | +212.66% | skill 1/1; base 1/1 |
| claude-code | doca-perftest.implicit-positive.v1 | 149,523 | 39,015 | +110,508 | +283.24% | skill 1/1; base 1/1 |
| claude-code | doca-perftest.implicit-positive.v2 | 189,247 | 71,892 | +117,355 | +163.24% | skill 1/1; base 1/1 |
| claude-code | doca-perftest.negative.v1 | 36,963 | 36,614 | +349 | +0.95% | skill 1/1; base 1/1 |
| claude-code | doca-perftest.negative.v2 | 108,920 | 218,168 | -109,248 | -50.08% | skill 1/1; base 1/1 |
| codex | All cases | 249,551 | 224,244 | +25,307 | +11.29% | skill 5/5; base 5/5 |
| codex | doca-perftest.explicit-positive.v1 | 62,669 | 56,580 | +6,089 | +10.76% | skill 1/1; base 1/1 |
| codex | doca-perftest.implicit-positive.v1 | 62,682 | 36,627 | +26,055 | +71.14% | skill 1/1; base 1/1 |
| codex | doca-perftest.implicit-positive.v2 | 54,245 | 36,016 | +18,229 | +50.61% | skill 1/1; base 1/1 |
| codex | doca-perftest.negative.v1 | 43,571 | 47,606 | -4,035 | -8.48% | skill 1/1; base 1/1 |
| codex | doca-perftest.negative.v2 | 26,384 | 47,415 | -21,031 | -44.36% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 847,965 | 626,318 | +221,647 | +35.39% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Drivers" in CAPABILITIES.md (lines 23-29)
  vs "## select-driver" in TASKS.md (lines 31-37) (`CAPABILITIES.md:23`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "## Examples" in SKILL.md (lines 22-27)
  vs "## Example questions this skill answers well" in SKILL.md (lines 36-41) (`SKILL.md:22`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/tools/doca-perftest/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-perftest/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-perftest`)
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
