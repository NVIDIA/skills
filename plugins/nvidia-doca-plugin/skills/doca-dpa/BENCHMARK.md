# Skill Benchmark: doca-dpa

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-dpa`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:99ac207da052504949cf6f114412975da1858f1bcea2374f3fac8c667696bb9c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-dpa | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dpa | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dpa | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dpa | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-dpa | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-dpa`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 92.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 60.0% → 95.0% (+35.0 points) | 90.0% → 85.0% (-5.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 81.3% (+31.3 points) | 70.8% → 91.7% (+20.9 points) |
| Efficiency | 85.9% — baseline ran, but no comparable score was available; uplift unavailable | 61.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 93.5% (-4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 78.9% → 79.4% (+0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 812,930 | 327,887 | +485,043 | +147.93% | skill 4/4; base 4/4 |
| claude-code | doca-dpa.explicit-positive.v1 | 210,788 | 37,001 | +173,787 | +469.68% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.implicit-positive.v1 | 261,487 | 109,827 | +151,660 | +138.09% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.implicit-positive.v2 | 304,217 | 72,672 | +231,545 | +318.62% | skill 1/1; base 1/1 |
| claude-code | doca-dpa.negative.v1 | 36,438 | 108,387 | -71,949 | -66.38% | skill 1/1; base 1/1 |
| codex | All cases | 469,407 | 301,871 | +167,536 | +55.50% | skill 4/4; base 4/4 |
| codex | doca-dpa.explicit-positive.v1 | 117,145 | 159,323 | -42,178 | -26.47% | skill 1/1; base 1/1 |
| codex | doca-dpa.implicit-positive.v1 | 191,647 | 47,383 | +144,264 | +304.46% | skill 1/1; base 1/1 |
| codex | doca-dpa.implicit-positive.v2 | 113,083 | 39,269 | +73,814 | +187.97% | skill 1/1; base 1/1 |
| codex | doca-dpa.negative.v1 | 47,532 | 55,896 | -8,364 | -14.96% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,282,337 | 629,758 | +652,579 | +103.62% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Capabilities and modes" in CAPABILITIES.md (lines 124-124)
  vs "## verbs" in CAPABILITIES.md (lines 375-375) (`CAPABILITIES.md:124`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## comms" in CAPABILITIES.md (lines 228-251)
  vs "## verbs" in CAPABILITIES.md (lines 343-357) (`CAPABILITIES.md:228`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Capabilities and modes" in CAPABILITIES.md (lines 125-125)
  vs "## comms" in CAPABILITIES.md (lines 201-227) (`CAPABILITIES.md:125`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## comms" in CAPABILITIES.md (lines 182-186)
  vs "## build" in TASKS.md (lines 113-117)
  vs "## build" in TASKS.md (lines 118-125)
  vs "## build" in TASKS.md (lines 126-135)
  vs "## comms" in TASKS.md (lines 420-424)
  vs "## verbs" in TASKS.md (lines 568-573)
  vs "## Command appendix" in TASKS.md (lines 731-740) (`CAPABILITIES.md:182`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-dpa/SKILL.md`)
- 12 additional finding(s) are available in the full evaluation artifacts.

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
