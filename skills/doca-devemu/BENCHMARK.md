# Skill Benchmark: doca-devemu

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-devemu`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:bddc3102319aeedfab57309e910e03e8c1185f748c84845eadf3d5b4a3799de2` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-devemu | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-devemu | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-devemu | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-devemu | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-devemu | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-devemu`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 98.6% — baseline ran, but no comparable score was available; uplift unavailable | 92.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 80.0% → 95.0% (+15.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 87.5% → 100.0% (+12.5 points) |
| Efficiency | 92.9% — baseline ran, but no comparable score was available; uplift unavailable | 71.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 85.7% (-12.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.5% → 84.8% (+2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 622,042 | 327,317 | +294,725 | +90.04% | skill 4/4; base 4/4 |
| claude-code | doca-devemu.explicit-positive.v1 | 195,264 | 37,606 | +157,658 | +419.24% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v1 | 194,959 | 73,720 | +121,239 | +164.46% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v2 | 194,920 | 108,830 | +86,090 | +79.11% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.negative.v1 | 36,899 | 107,161 | -70,262 | -65.57% | skill 1/1; base 1/1 |
| codex | All cases | 336,084 | 210,038 | +126,046 | +60.01% | skill 4/4; base 4/4 |
| codex | doca-devemu.explicit-positive.v1 | 98,035 | 60,868 | +37,167 | +61.06% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v1 | 115,068 | 39,557 | +75,511 | +190.89% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v2 | 79,781 | 30,581 | +49,200 | +160.88% | skill 1/1; base 1/1 |
| codex | doca-devemu.negative.v1 | 43,200 | 79,032 | -35,832 | -45.34% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 958,126 | 537,355 | +420,771 | +78.30% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Observability" in CAPABILITIES.md (lines 323-330)
  vs "## Observability" in CAPABILITIES.md (lines 339-348)
  vs "## Usage notes (DOCA 3.5)" in TASKS.md (lines 72-75) (`CAPABILITIES.md:323`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Capabilities and modes" in CAPABILITIES.md (lines 108-108)
  vs "## Usage notes (DOCA 3.5)" in TASKS.md (lines 79-96) (`CAPABILITIES.md:108`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md and references/details.md:
  "## Pattern overview" in CAPABILITIES.md (lines 30-37)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 73-76)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 77-85)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 93-102)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 113-114)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 118-122)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 260-266)
  vs "## Observability" in CAPABILITIES.md (lines 290-322)
  vs "## Safety policy" in CAPABILITIES.md (lines 366-377)
  vs "## Safety policy" in CAPABILITIES.md (lines 381-387)
  vs "## Safety policy" in CAPABILITIES.md (lines 388-400)
  vs "## Safety policy" in CAPABILITIES.md (lines 426-434)
  vs "## Deferred topic boundaries" in CAPABILITIES.md (lines 449-507)
  vs "## Examples" in SKILL.md (lines 5-13)
  vs "## Instructions" in SKILL.md (lines 14-50)
  vs "## Audience" in SKILL.md (lines 51-83)
  vs "## When to load this skill" in SKILL.md (lines 84-126)
  vs "## What this skill provides" in SKILL.md (lines 127-165)
  vs "## Loading order" in SKILL.md (lines 166-197)
  vs "## Related skills" in SKILL.md (lines 198-251)
  vs "## configure" in TASKS.md (lines 109-117)
  vs "## configure" in TASKS.md (lines 138-165)
  vs "## configure" in TASKS.md (lines 187-196)
  vs "## configure" in TASKS.md (lines 229-243)
  vs "## run" in TASKS.md (lines 341-344)
  vs "## run" in TASKS.md (lines 346-391)
  vs "## test" in TASKS.md (lines 401-407)
  vs "## test" in TASKS.md (lines 420-463)
  vs "## test" in TASKS.md (lines 465-472)
  vs "## debug" in TASKS.md (lines 508-532)
  vs "## debug" in TASKS.md (lines 557-583)
  vs "## Example questions this skill answers well" in references/details.md (lines 10-75)
  vs "## Related skills" in references/details.md (lines 125-176) (`CAPABILITIES.md:30`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and references/details.md:
  "## Pattern overview" in CAPABILITIES.md (lines 39-58)
  vs "## Safety policy" in CAPABILITIES.md (lines 401-411)
  vs "## Related skills" in references/details.md (lines 177-184) (`CAPABILITIES.md:39`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-devemu/SKILL.md`)
- 17 additional finding(s) are available in the full evaluation artifacts.

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
