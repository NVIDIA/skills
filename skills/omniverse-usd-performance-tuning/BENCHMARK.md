# Skill Benchmark: omniverse-usd-performance-tuning

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `omniverse-usd-performance-tuning`
- Validation status: `passed`
- Evaluation date: 2026-11-02
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (9 positive, 1 negative)
- Dataset digest: `sha256:9f32956658415b5e96a803c6cda4eff0f866c64232bb4bd88176cfef273b0ad5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| omniverse-usd-performance-tuning | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

- Controls for `omniverse-usd-performance-tuning`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 75.8% — baseline ran, but no comparable score was available; uplift unavailable | 78.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 90.0% (-10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 60.0% → 76.0% (+16.0 points) | 66.0% → 84.0% (+18.0 points) |
| Discoverability | 78.9% — baseline ran, but no comparable score was available; uplift unavailable | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.3% → 55.5% (+15.2 points) | 45.8% → 57.9% (+12.1 points) |
| Efficiency | 78.6% — baseline ran, but no comparable score was available; uplift unavailable | 74.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.4% → 85.7% (-2.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 85.0% → 79.2% (-5.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 7,414,490 | 3,447,634 | +3,966,856 | +115.06% | skill 10/10; base 10/10 |
| claude-code | usd-performance-broad-optimization-flow | 494,353 | 586,825 | -92,472 | -15.76% | skill 1/1; base 1/1 |
| claude-code | usd-performance-destructive-op-approval | 375,980 | 605,643 | -229,663 | -37.92% | skill 1/1; base 1/1 |
| claude-code | usd-performance-expensive-validation-approval | 4,379,560 | 490,583 | +3,888,977 | +792.73% | skill 1/1; base 1/1 |
| claude-code | usd-performance-no-overwrite-source | 414,410 | 208,228 | +206,182 | +99.02% | skill 1/1; base 1/1 |
| claude-code | usd-performance-quality-budget-not-triangle-budget | 444,638 | 298,689 | +145,949 | +48.86% | skill 1/1; base 1/1 |
| claude-code | usd-performance-report-schema-conformance | 150,133 | 187,484 | -37,351 | -19.92% | skill 1/1; base 1/1 |
| claude-code | usd-performance-runtime-preflight-gate | 433,568 | 603,928 | -170,360 | -28.21% | skill 1/1; base 1/1 |
| claude-code | usd-performance-structural-only-report | 541,085 | 286,130 | +254,955 | +89.10% | skill 1/1; base 1/1 |
| claude-code | usd-performance-viewer-build-distractor-negative | 35,751 | 35,744 | +7 | +0.02% | skill 1/1; base 1/1 |
| claude-code | usd-performance-zero-work-operation | 145,012 | 144,380 | +632 | +0.44% | skill 1/1; base 1/1 |
| codex | All cases | 4,676,096 | 2,322,905 | +2,353,191 | +101.30% | skill 10/10; base 10/10 |
| codex | usd-performance-broad-optimization-flow | 412,119 | 358,167 | +53,952 | +15.06% | skill 1/1; base 1/1 |
| codex | usd-performance-destructive-op-approval | 276,085 | 278,140 | -2,055 | -0.74% | skill 1/1; base 1/1 |
| codex | usd-performance-expensive-validation-approval | 767,016 | 474,406 | +292,610 | +61.68% | skill 1/1; base 1/1 |
| codex | usd-performance-no-overwrite-source | 292,841 | 186,489 | +106,352 | +57.03% | skill 1/1; base 1/1 |
| codex | usd-performance-quality-budget-not-triangle-budget | 412,686 | 252,324 | +160,362 | +63.55% | skill 1/1; base 1/1 |
| codex | usd-performance-report-schema-conformance | 611,473 | 182,857 | +428,616 | +234.40% | skill 1/1; base 1/1 |
| codex | usd-performance-runtime-preflight-gate | 732,768 | 218,931 | +513,837 | +234.70% | skill 1/1; base 1/1 |
| codex | usd-performance-structural-only-report | 548,884 | 257,121 | +291,763 | +113.47% | skill 1/1; base 1/1 |
| codex | usd-performance-viewer-build-distractor-negative | 37,576 | 26,621 | +10,955 | +41.15% | skill 1/1; base 1/1 |
| codex | usd-performance-zero-work-operation | 584,648 | 87,849 | +496,799 | +565.51% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,090,586 | 5,770,539 | +6,320,047 | +109.52% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/optimization-report/scripts/descent_convergence_common.py and references/optimization-report/scripts/validate_report.py:
  "_is_number()" in references/optimization-report/scripts/descent_convergence_common.py (lines 65-67)
  vs "_is_number()" in references/optimization-report/scripts/validate_report.py (lines 175-177) (`references/optimization-report/scripts/descent_convergence_common.py:65`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/omniverse-usd-performance-tuning/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/omniverse-usd-performance-tuning/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in asset-authoring.md (`skills/omniverse-usd-performance-tuning/SKILL.md`)
- **LOW** QUALITY/quality_efficiency: Non-doc file in references/: component_scope.py (`skills/omniverse-usd-performance-tuning/SKILL.md`)
- 20 additional finding(s) are available in the full evaluation artifacts.

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
