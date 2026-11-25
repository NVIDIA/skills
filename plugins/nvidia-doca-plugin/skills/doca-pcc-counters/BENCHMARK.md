# Skill Benchmark: doca-pcc-counters

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-pcc-counters`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:d4cf11046a7a83cd5f672ca04e9d52a3d7ec7e75196f5e27915e2f5df99a3fc7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-pcc-counters | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-pcc-counters | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-pcc-counters | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-pcc-counters | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-pcc-counters | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-pcc-counters`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 85.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 10.0% → 100.0% (+90.0 points) | 65.0% → 100.0% (+35.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.0% → 87.5% (+62.5 points) | 68.8% → 62.5% (-6.3 points) |
| Efficiency | 97.0% — baseline ran, but no comparable score was available; uplift unavailable | 70.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 93.9% (-4.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.7% → 85.3% (-5.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 532,845 | 461,955 | +70,890 | +15.35% | skill 4/4; base 4/4 |
| claude-code | doca-pcc-counters.explicit-positive.v1 | 169,590 | 106,507 | +63,083 | +59.23% | skill 1/1; base 1/1 |
| claude-code | doca-pcc-counters.implicit-positive.v1 | 163,282 | 141,812 | +21,470 | +15.14% | skill 1/1; base 1/1 |
| claude-code | doca-pcc-counters.implicit-positive.v2 | 163,524 | 106,306 | +57,218 | +53.82% | skill 1/1; base 1/1 |
| claude-code | doca-pcc-counters.negative.v1 | 36,449 | 107,330 | -70,881 | -66.04% | skill 1/1; base 1/1 |
| codex | All cases | 321,342 | 122,434 | +198,908 | +162.46% | skill 4/4; base 4/4 |
| codex | doca-pcc-counters.explicit-positive.v1 | 126,652 | 39,910 | +86,742 | +217.34% | skill 1/1; base 1/1 |
| codex | doca-pcc-counters.implicit-positive.v1 | 84,245 | 47,154 | +37,091 | +78.66% | skill 1/1; base 1/1 |
| codex | doca-pcc-counters.implicit-positive.v2 | 79,795 | 21,951 | +57,844 | +263.51% | skill 1/1; base 1/1 |
| codex | doca-pcc-counters.negative.v1 | 30,650 | 13,419 | +17,231 | +128.41% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 854,187 | 584,389 | +269,798 | +46.17% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 33 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "# DOCA PCC Counters (`pcc_counters.sh`) — Capabilities" in CAPABILITIES.md (lines 1-18)
  vs "## Pattern overview" in CAPABILITIES.md (lines 19-44)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 53-56)
  vs "## Version compatibility" in CAPABILITIES.md (lines 102-128)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 129-176)
  vs "## Examples" in SKILL.md (lines 5-24)
  vs "## Instructions" in SKILL.md (lines 25-56)
  vs "## Example questions this skill answers well" in SKILL.md (lines 57-92)
  vs "## Audience" in SKILL.md (lines 93-126)
  vs "## What this skill provides" in SKILL.md (lines 149-185)
  vs "## Related skills" in SKILL.md (lines 224-254)
  vs "# DOCA PCC Counters (`pcc_counters.sh`) — Tasks" in TASKS.md (lines 1-22)
  vs "## configure" in TASKS.md (lines 23-70)
  vs "## run" in TASKS.md (lines 109-160)
  vs "## debug" in TASKS.md (lines 183-226)
  vs "## Deferred task verbs" in TASKS.md (lines 227-252)
  vs "## Command appendix" in TASKS.md (lines 253-284) (`CAPABILITIES.md:1`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-pcc-counters/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-pcc-counters`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo          (`BENCHMARK.md:130`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo          (`BENCHMARK.md:131`)
- 29 additional finding(s) are available in the full evaluation artifacts.

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
