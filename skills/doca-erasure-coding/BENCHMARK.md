# Skill Benchmark: doca-erasure-coding

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-erasure-coding`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:ce6ee874a1ed1103e7f48d2a35930d6eafc4d66d818d35f6cca641d5b505d03e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-erasure-coding | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-erasure-coding | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-erasure-coding | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-erasure-coding | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-erasure-coding | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-erasure-coding`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 62.5% → 100.0% (+37.5 points) |
| Correctness | 45.0% → 100.0% (+55.0 points) | 50.0% → 100.0% (+50.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 43.8% → 100.0% (+56.2 points) | 54.2% → 79.2% (+25.0 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 72.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.6% → 92.3% (-4.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.4% → 85.8% (-0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 829,006 | 755,789 | +73,217 | +9.69% | skill 4/4; base 4/4 |
| claude-code | doca-erasure-coding.explicit-positive.v1 | 342,085 | 107,362 | +234,723 | +218.63% | skill 1/1; base 1/1 |
| claude-code | doca-erasure-coding.implicit-positive.v1 | 228,939 | 383,479 | -154,540 | -40.30% | skill 1/1; base 1/1 |
| claude-code | doca-erasure-coding.implicit-positive.v2 | 221,775 | 229,517 | -7,742 | -3.37% | skill 1/1; base 1/1 |
| claude-code | doca-erasure-coding.negative.v1 | 36,207 | 35,431 | +776 | +2.19% | skill 1/1; base 1/1 |
| codex | All cases | 372,069 | 284,424 | +87,645 | +30.81% | skill 4/4; base 4/4 |
| codex | doca-erasure-coding.explicit-positive.v1 | 123,542 | 124,042 | -500 | -0.40% | skill 1/1; base 1/1 |
| codex | doca-erasure-coding.implicit-positive.v1 | 104,211 | 61,863 | +42,348 | +68.45% | skill 1/1; base 1/1 |
| codex | doca-erasure-coding.implicit-positive.v2 | 105,268 | 71,826 | +33,442 | +46.56% | skill 1/1; base 1/1 |
| codex | doca-erasure-coding.negative.v1 | 39,048 | 26,693 | +12,355 | +46.29% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,201,075 | 1,040,213 | +160,862 | +15.46% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Capabilities and modes" in CAPABILITIES.md (lines 58-69)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 103-130)
  vs "## Safety policy" in CAPABILITIES.md (lines 319-345) (`CAPABILITIES.md:58`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "## Pattern overview" in CAPABILITIES.md (lines 16-19)
  vs "## Pattern overview" in CAPABILITIES.md (lines 20-26)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 51-57)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 75-79)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 84-89)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 90-100)
  vs "## Deferred topic boundaries" in CAPABILITIES.md (lines 378-415)
  vs "## Instructions" in SKILL.md (lines 22-37)
  vs "## Example questions this skill answers well" in SKILL.md (lines 52-55)
  vs "## Example questions this skill answers well" in SKILL.md (lines 56-123)
  vs "## Audience" in SKILL.md (lines 127-157)
  vs "## When to load this skill" in SKILL.md (lines 158-216)
  vs "## What this skill provides" in SKILL.md (lines 217-245)
  vs "## configure" in TASKS.md (lines 29-32)
  vs "## configure" in TASKS.md (lines 34-125)
  vs "## modify" in TASKS.md (lines 169-177) (`CAPABILITIES.md:16`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 234-241)
  vs "## test" in TASKS.md (lines 247-309)
  vs "## test" in TASKS.md (lines 311-321)
  vs "## debug" in TASKS.md (lines 362-398) (`CAPABILITIES.md:234`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-erasure-coding/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/libs/doca-erasure-coding/SKILL.md`)
- 14 additional finding(s) are available in the full evaluation artifacts.

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
