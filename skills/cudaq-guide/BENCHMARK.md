# Skill Benchmark: cudaq-guide

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudaq-guide`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 14 evaluation tasks (11 positive, 3 negative)
- Dataset digest: `sha256:e587d102c08982b984d330ae5fd7786631b7c327cf05f97b946576160ba2e91b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cudaq-guide | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-guide | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

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
| Overall | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 92.9% → 92.9% (±0.0 points) | 96.4% → 100.0% (+3.6 points) |
| Correctness | 82.9% → 98.6% (+15.7 points) | 84.3% → 91.4% (+7.1 points) |
| Discoverability | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 94.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 53.9% → 92.3% (+38.4 points) | 54.5% → 72.0% (+17.5 points) |
| Efficiency | 82.5% — baseline ran, but no comparable score was available; uplift unavailable | 67.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 96.7% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.2% → 87.8% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,781,721 | 980,867 | +800,854 | +81.65% | skill 14/14; base 14/14 |
| claude-code | cudaq-guide-applications-001 | 114,678 | 35,800 | +78,878 | +220.33% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-adjoint-001 | 152,084 | 71,490 | +80,594 | +112.73% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-api-001 | 119,393 | 35,805 | +83,588 | +233.45% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-constraint-001 | 110,618 | 36,052 | +74,566 | +206.83% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-gpu-sim-001 | 110,858 | 36,165 | +74,693 | +206.53% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-install-001 | 114,888 | 71,534 | +43,354 | +60.61% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-001 | 35,059 | 35,010 | +49 | +0.14% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-002 | 35,365 | 35,672 | -307 | -0.86% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-003 | 107,244 | 143,565 | -36,321 | -25.30% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-parallelize-001 | 112,517 | 36,815 | +75,702 | +205.63% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-parallelize-002 | 155,115 | 154,285 | +830 | +0.54% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-qpu-001 | 150,336 | 36,482 | +113,854 | +312.08% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-route-qiskit-001 | 112,073 | 72,524 | +39,549 | +54.53% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-test-program-001 | 351,493 | 179,668 | +171,825 | +95.63% | skill 1/1; base 1/1 |
| codex | All cases | 1,015,279 | 678,173 | +337,106 | +49.71% | skill 14/14; base 14/14 |
| codex | cudaq-guide-applications-001 | 69,484 | 28,791 | +40,693 | +141.34% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-adjoint-001 | 51,410 | 26,025 | +25,385 | +97.54% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-api-001 | 49,534 | 21,657 | +27,877 | +128.72% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-constraint-001 | 62,553 | 34,462 | +28,091 | +81.51% | skill 1/1; base 1/1 |
| codex | cudaq-guide-gpu-sim-001 | 66,706 | 17,572 | +49,134 | +279.62% | skill 1/1; base 1/1 |
| codex | cudaq-guide-install-001 | 62,366 | 38,657 | +23,709 | +61.33% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-001 | 13,223 | 13,176 | +47 | +0.36% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-002 | 13,330 | 13,287 | +43 | +0.32% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-003 | 50,293 | 44,874 | +5,419 | +12.08% | skill 1/1; base 1/1 |
| codex | cudaq-guide-parallelize-001 | 61,029 | 30,410 | +30,619 | +100.69% | skill 1/1; base 1/1 |
| codex | cudaq-guide-parallelize-002 | 80,299 | 49,673 | +30,626 | +61.66% | skill 1/1; base 1/1 |
| codex | cudaq-guide-qpu-001 | 86,889 | 33,622 | +53,267 | +158.43% | skill 1/1; base 1/1 |
| codex | cudaq-guide-route-qiskit-001 | 45,307 | 30,429 | +14,878 | +48.89% | skill 1/1; base 1/1 |
| codex | cudaq-guide-test-program-001 | 302,856 | 295,538 | +7,318 | +2.48% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,797,000 | 1,659,040 | +1,137,960 | +68.59% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/onboarding.md:
  "## Troubleshooting" in SKILL.md (lines 78-92)
  vs "## Platform Troubleshooting" in references/onboarding.md (lines 242-250) (`SKILL.md:78`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/onboarding.md:
  "## Purpose" in SKILL.md (lines 3-9)
  vs "## Instructions" in SKILL.md (lines 18-28)
  vs "## Routing by Argument" in SKILL.md (lines 29-41)
  vs "## Menu" in SKILL.md (lines 42-59)
  vs "# CUDA-Q Onboarding Reference" in references/onboarding.md (lines 1-5)
  vs "## Invocation Examples" in references/onboarding.md (lines 226-241) (`SKILL.md:3`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'author' is ignored; use 'metadata.author' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cudaq-guide/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
