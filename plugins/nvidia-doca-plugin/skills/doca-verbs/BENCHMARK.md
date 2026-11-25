# Skill Benchmark: doca-verbs

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-verbs`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:68fee5caecf9a6d68a1653e3eadd6daf86fb1f79f00fb1f99fffcc2003f1a39b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-verbs | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-verbs | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-verbs | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-verbs | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| doca-verbs | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `doca-verbs`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 100.0% → 100.0% (±0.0 points) | Not available |
| Correctness | 40.0% → 95.0% (+55.0 points) | Not available |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 31.3% → 87.5% (+56.2 points) | Not available |
| Efficiency | 90.3% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 89.0% (-9.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 561,008 | 358,619 | +202,389 | +56.44% | skill 4/4; base 4/4 |
| claude-code | doca-verbs.explicit-positive.v1 | 150,230 | 71,911 | +78,319 | +108.91% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.implicit-positive.v1 | 150,130 | 143,016 | +7,114 | +4.97% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.implicit-positive.v2 | 224,692 | 72,589 | +152,103 | +209.54% | skill 1/1; base 1/1 |
| claude-code | doca-verbs.negative.v1 | 35,956 | 71,103 | -35,147 | -49.43% | skill 1/1; base 1/1 |
| codex | All cases | 269,363 | 220,856 | +48,507 | +21.96% | skill 4/4; base 4/4 |
| codex | doca-verbs.explicit-positive.v1 | 50,188 | 73,181 | -22,993 | -31.42% | skill 1/1; base 1/1 |
| codex | doca-verbs.implicit-positive.v1 | 82,484 | 62,027 | +20,457 | +32.98% | skill 1/1; base 1/1 |
| codex | doca-verbs.implicit-positive.v2 | 92,316 | 38,859 | +53,457 | +137.57% | skill 1/1; base 1/1 |
| codex | doca-verbs.negative.v1 | 44,375 | 46,789 | -2,414 | -5.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 830,371 | 579,475 | +250,896 | +43.30% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md and references/deferred-topics.md:
  "## Capabilities and modes" in CAPABILITIES.md (lines 37-44)
  vs "## Version compatibility" in CAPABILITIES.md (lines 182-219)
  vs "## Examples" in SKILL.md (lines 5-28)
  vs "## STOP — most RDMA tasks do NOT belong here" in SKILL.md (lines 34-61)
  vs "## The decision this skill exists to gate" in SKILL.md (lines 87-118)
  vs "## When to load this skill" in SKILL.md (lines 141-171)
  vs "## What this skill provides" in SKILL.md (lines 172-196)
  vs "## Related skills" in SKILL.md (lines 242-290)
  vs "## configure" in TASKS.md (lines 65-163)
  vs "## Deferred topic boundaries" in references/deferred-topics.md (lines 1-41) (`CAPABILITIES.md:37`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in command-appendix.md (`skills/libs/doca-verbs/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/libs/doca-verbs/SKILL.md`)
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
