# Skill Benchmark: omniverse-usd-performance-tuning

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `omniverse-usd-performance-tuning`
- Validation status: `passed`
- Evaluation date: 2026-11-03
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
| Overall | 79.4% — baseline ran, but no comparable score was available; uplift unavailable | 78.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 90.0% (-10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 60.0% → 86.0% (+26.0 points) | 60.0% → 82.0% (+22.0 points) |
| Discoverability | 82.8% — baseline ran, but no comparable score was available; uplift unavailable | 76.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.6% → 60.4% (+23.8 points) | 36.6% → 57.2% (+20.6 points) |
| Efficiency | 77.6% — baseline ran, but no comparable score was available; uplift unavailable | 79.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 89.8% → 82.4% (-7.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.1% → 81.2% (-4.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,706,896 | 3,093,234 | +3,613,662 | +116.82% | skill 10/10; base 10/10 |
| claude-code | usd-performance-broad-optimization-flow | 678,645 | 532,071 | +146,574 | +27.55% | skill 1/1; base 1/1 |
| claude-code | usd-performance-destructive-op-approval | 309,506 | 496,764 | -187,258 | -37.70% | skill 1/1; base 1/1 |
| claude-code | usd-performance-expensive-validation-approval | 1,670,524 | 346,372 | +1,324,152 | +382.29% | skill 1/1; base 1/1 |
| claude-code | usd-performance-no-overwrite-source | 418,774 | 351,316 | +67,458 | +19.20% | skill 1/1; base 1/1 |
| claude-code | usd-performance-quality-budget-not-triangle-budget | 543,103 | 223,488 | +319,615 | +143.01% | skill 1/1; base 1/1 |
| claude-code | usd-performance-report-schema-conformance | 1,289,622 | 187,833 | +1,101,789 | +586.58% | skill 1/1; base 1/1 |
| claude-code | usd-performance-runtime-preflight-gate | 635,290 | 430,375 | +204,915 | +47.61% | skill 1/1; base 1/1 |
| claude-code | usd-performance-structural-only-report | 980,372 | 308,288 | +672,084 | +218.01% | skill 1/1; base 1/1 |
| claude-code | usd-performance-viewer-build-distractor-negative | 35,986 | 35,825 | +161 | +0.45% | skill 1/1; base 1/1 |
| claude-code | usd-performance-zero-work-operation | 145,074 | 180,902 | -35,828 | -19.81% | skill 1/1; base 1/1 |
| codex | All cases | 4,825,372 | 2,202,842 | +2,622,530 | +119.05% | skill 10/10; base 10/10 |
| codex | usd-performance-broad-optimization-flow | 380,616 | 327,709 | +52,907 | +16.14% | skill 1/1; base 1/1 |
| codex | usd-performance-destructive-op-approval | 248,592 | 251,168 | -2,576 | -1.03% | skill 1/1; base 1/1 |
| codex | usd-performance-expensive-validation-approval | 930,589 | 314,874 | +615,715 | +195.54% | skill 1/1; base 1/1 |
| codex | usd-performance-no-overwrite-source | 335,037 | 300,793 | +34,244 | +11.38% | skill 1/1; base 1/1 |
| codex | usd-performance-quality-budget-not-triangle-budget | 426,719 | 235,931 | +190,788 | +80.87% | skill 1/1; base 1/1 |
| codex | usd-performance-report-schema-conformance | 529,614 | 181,246 | +348,368 | +192.21% | skill 1/1; base 1/1 |
| codex | usd-performance-runtime-preflight-gate | 593,564 | 251,186 | +342,378 | +136.30% | skill 1/1; base 1/1 |
| codex | usd-performance-structural-only-report | 1,024,821 | 232,254 | +792,567 | +341.25% | skill 1/1; base 1/1 |
| codex | usd-performance-viewer-build-distractor-negative | 33,326 | 21,207 | +12,119 | +57.15% | skill 1/1; base 1/1 |
| codex | usd-performance-zero-work-operation | 322,494 | 86,474 | +236,020 | +272.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 11,532,268 | 5,296,076 | +6,236,192 | +117.75% | skill 20/20; base 20/20 |

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
