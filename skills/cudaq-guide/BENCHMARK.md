# Skill Benchmark: cudaq-guide

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudaq-guide`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 13 evaluation tasks (10 positive, 3 negative)
- Dataset digest: `sha256:7884d2878ed008a3969a7668a02fb2f9fda8e70495368a69ca8bf4ce1f6071dd` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cudaq-guide | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cudaq-guide`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 86.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 86.2% → 98.5% (+12.3 points) | 89.2% → 92.3% (+3.1 points) |
| Discoverability | 96.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 51.3% → 87.8% (+36.5 points) | 54.5% → 76.3% (+21.8 points) |
| Efficiency | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 66.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 96.8% (-1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.2% → 87.8% (+0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,481,300 | 789,012 | +692,288 | +87.74% | skill 13/13; base 13/13 |
| claude-code | cudaq-guide-applications-001 | 114,052 | 35,921 | +78,131 | +217.51% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-adjoint-001 | 156,496 | 71,436 | +85,060 | +119.07% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-api-001 | 119,297 | 35,531 | +83,766 | +235.75% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-constraint-001 | 110,692 | 108,076 | +2,616 | +2.42% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-gpu-sim-001 | 110,531 | 36,318 | +74,213 | +204.34% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-install-001 | 114,060 | 36,596 | +77,464 | +211.67% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-001 | 35,033 | 35,019 | +14 | +0.04% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-002 | 35,378 | 35,462 | -84 | -0.24% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-003 | 107,075 | 106,823 | +252 | +0.24% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-parallelize-001 | 112,905 | 36,427 | +76,478 | +209.95% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-qpu-001 | 150,348 | 36,633 | +113,715 | +310.42% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-route-qiskit-001 | 74,529 | 72,392 | +2,137 | +2.95% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-test-program-001 | 240,904 | 142,378 | +98,526 | +69.20% | skill 1/1; base 1/1 |
| codex | All cases | 968,418 | 744,788 | +223,630 | +30.03% | skill 13/13; base 13/13 |
| codex | cudaq-guide-applications-001 | 65,741 | 28,006 | +37,735 | +134.74% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-adjoint-001 | 54,199 | 38,662 | +15,537 | +40.19% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-api-001 | 49,848 | 17,572 | +32,276 | +183.68% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-constraint-001 | 62,648 | 46,925 | +15,723 | +33.51% | skill 1/1; base 1/1 |
| codex | cudaq-guide-gpu-sim-001 | 66,773 | 21,785 | +44,988 | +206.51% | skill 1/1; base 1/1 |
| codex | cudaq-guide-install-001 | 66,144 | 47,291 | +18,853 | +39.87% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-001 | 13,227 | 13,176 | +51 | +0.39% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-002 | 13,371 | 13,279 | +92 | +0.69% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-003 | 45,286 | 49,995 | -4,709 | -9.42% | skill 1/1; base 1/1 |
| codex | cudaq-guide-parallelize-001 | 79,206 | 31,002 | +48,204 | +155.49% | skill 1/1; base 1/1 |
| codex | cudaq-guide-qpu-001 | 87,100 | 35,199 | +51,901 | +147.45% | skill 1/1; base 1/1 |
| codex | cudaq-guide-route-qiskit-001 | 49,199 | 31,016 | +18,183 | +58.62% | skill 1/1; base 1/1 |
| codex | cudaq-guide-test-program-001 | 315,676 | 370,880 | -55,204 | -14.88% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,449,718 | 1,533,800 | +915,918 | +59.72% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/onboarding.md:
  "## Troubleshooting" in SKILL.md (lines 78-92)
  vs "## Platform Troubleshooting" in references/onboarding.md (lines 207-215) (`SKILL.md:78`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/onboarding.md:
  "## Purpose" in SKILL.md (lines 3-9)
  vs "## Instructions" in SKILL.md (lines 18-28)
  vs "## Routing by Argument" in SKILL.md (lines 29-41)
  vs "## Menu" in SKILL.md (lines 42-59)
  vs "# CUDA-Q Onboarding Reference" in references/onboarding.md (lines 1-5)
  vs "## Invocation Examples" in references/onboarding.md (lines 191-206) (`SKILL.md:3`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'author' is ignored; use 'metadata.author' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cudaq-guide/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
