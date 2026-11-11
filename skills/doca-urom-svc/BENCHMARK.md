# Skill Benchmark: doca-urom-svc

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-urom-svc`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:ea16cc085828d2133e3ded6b24d1d0cc8cb949802c1ddac4c03c2ddb9c1c05f9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| doca-urom-svc | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-urom-svc | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-urom-svc | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-urom-svc | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| doca-urom-svc | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `doca-urom-svc`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 98.4% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 87.5% → 93.8% (+6.3 points) |
| Efficiency | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 76.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 89.8% (-8.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.0% → 83.8% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 676,599 | 253,595 | +423,004 | +166.80% | skill 4/4; base 4/4 |
| claude-code | doca-urom-svc.explicit-positive.v1 | 187,525 | 36,922 | +150,603 | +407.90% | skill 1/1; base 1/1 |
| claude-code | doca-urom-svc.implicit-positive.v1 | 187,659 | 73,198 | +114,461 | +156.37% | skill 1/1; base 1/1 |
| claude-code | doca-urom-svc.implicit-positive.v2 | 265,218 | 72,033 | +193,185 | +268.19% | skill 1/1; base 1/1 |
| claude-code | doca-urom-svc.negative.v1 | 36,197 | 71,442 | -35,245 | -49.33% | skill 1/1; base 1/1 |
| codex | All cases | 426,999 | 208,225 | +218,774 | +105.07% | skill 4/4; base 4/4 |
| codex | doca-urom-svc.explicit-positive.v1 | 164,601 | 75,599 | +89,002 | +117.73% | skill 1/1; base 1/1 |
| codex | doca-urom-svc.implicit-positive.v1 | 135,362 | 39,613 | +95,749 | +241.71% | skill 1/1; base 1/1 |
| codex | doca-urom-svc.implicit-positive.v2 | 96,390 | 47,845 | +48,545 | +101.46% | skill 1/1; base 1/1 |
| codex | doca-urom-svc.negative.v1 | 30,646 | 45,168 | -14,522 | -32.15% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,103,598 | 461,820 | +641,778 | +138.97% | skill 8/8; base 8/8 |

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

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/services/doca-urom-svc`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "## Pattern overview" in CAPABILITIES.md (lines 23-30)
  vs "## Pattern overview" in CAPABILITIES.md (lines 32-49)
  vs "### Service shape" in CAPABILITIES.md (lines 57-98)
  vs "### Publisher / executor paired-contract model" in CAPABILITIES.md (lines 99-122)
  vs "## Deployment shape" in CAPABILITIES.md (lines 123-153)
  vs "### Configuration axes" in CAPABILITIES.md (lines 162-167)
  vs "### Pairing surface — host library plus RDMA substrate" in CAPABILITIES.md (lines 184-201)
  vs "## Version compatibility" in CAPABILITIES.md (lines 202-257)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 267-272)
  vs "## Safety policy" in CAPABILITIES.md (lines 351-451)
  vs "# DOCA UROM Service" in SKILL.md (lines 1-8)
  vs "## Instructions" in SKILL.md (lines 18-45)
  vs "## Example questions this skill answers well" in SKILL.md (lines 50-111)
  vs "## Audience" in SKILL.md (lines 115-155)
  vs "## When to load this skill" in SKILL.md (lines 156-202)
  vs "## What this skill provides" in SKILL.md (lines 203-236)
  vs "## What this skill deliberately does not ship" in SKILL.md (lines 237-282)
  vs "## Related skills" in SKILL.md (lines 300-361)
  vs "## configure" in TASKS.md (lines 24-134)
  vs "## run" in TASKS.md (lines 280-286)
  vs "## debug" in TASKS.md (lines 369-441)
  vs "## Cross-cutting" in TASKS.md (lines 550-571) (`CAPABILITIES.md:23`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/services/doca-urom-svc/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/services/doca-urom-svc`)
- **LOW** QUALITY/quality_discoverability: Description very long (663 chars, recommend 50-150) (`skills/services/doca-urom-svc/SKILL.md`)
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
