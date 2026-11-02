# Skill Benchmark: omniverse-usd-performance-tuning

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `omniverse-usd-performance-tuning`
- Validation status: `passed`
- Evaluation date: 2026-10-29
- Evaluator version: `1.6.4`
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
| omniverse-usd-performance-tuning | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| omniverse-usd-performance-tuning | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |

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
| Overall | 78.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.0% → 90.0% (±0.0 points) |
| Correctness | 54.0% → 82.0% (+28.0 points) | 54.0% → 86.0% (+32.0 points) |
| Discoverability | 75.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.9% → 53.4% (+19.5 points) | 40.8% → 56.6% (+15.8 points) |
| Efficiency | 82.1% — baseline ran, but no comparable score was available; uplift unavailable | 75.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.9% → 83.7% (-5.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.0% → 81.0% (-5.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 7,858,336 | 2,941,214 | +4,917,122 | +167.18% | skill 10/10; base 10/10 |
| claude-code | usd-performance-broad-optimization-flow | 513,656 | 427,077 | +86,579 | +20.27% | skill 1/1; base 1/1 |
| claude-code | usd-performance-destructive-op-approval | 358,842 | 351,550 | +7,292 | +2.07% | skill 1/1; base 1/1 |
| claude-code | usd-performance-expensive-validation-approval | 3,603,605 | 606,570 | +2,997,035 | +494.10% | skill 1/1; base 1/1 |
| claude-code | usd-performance-no-overwrite-source | 481,096 | 206,081 | +275,015 | +133.45% | skill 1/1; base 1/1 |
| claude-code | usd-performance-quality-budget-not-triangle-budget | 546,672 | 317,331 | +229,341 | +72.27% | skill 1/1; base 1/1 |
| claude-code | usd-performance-report-schema-conformance | 149,706 | 148,979 | +727 | +0.49% | skill 1/1; base 1/1 |
| claude-code | usd-performance-runtime-preflight-gate | 468,684 | 453,015 | +15,669 | +3.46% | skill 1/1; base 1/1 |
| claude-code | usd-performance-structural-only-report | 1,554,990 | 214,353 | +1,340,637 | +625.43% | skill 1/1; base 1/1 |
| claude-code | usd-performance-viewer-build-distractor-negative | 35,815 | 35,684 | +131 | +0.37% | skill 1/1; base 1/1 |
| claude-code | usd-performance-zero-work-operation | 145,270 | 180,574 | -35,304 | -19.55% | skill 1/1; base 1/1 |
| codex | All cases | 4,936,831 | 2,218,354 | +2,718,477 | +122.54% | skill 10/10; base 10/10 |
| codex | usd-performance-broad-optimization-flow | 396,546 | 369,682 | +26,864 | +7.27% | skill 1/1; base 1/1 |
| codex | usd-performance-destructive-op-approval | 287,056 | 270,586 | +16,470 | +6.09% | skill 1/1; base 1/1 |
| codex | usd-performance-expensive-validation-approval | 795,380 | 244,976 | +550,404 | +224.68% | skill 1/1; base 1/1 |
| codex | usd-performance-no-overwrite-source | 281,248 | 232,537 | +48,711 | +20.95% | skill 1/1; base 1/1 |
| codex | usd-performance-quality-budget-not-triangle-budget | 423,815 | 308,264 | +115,551 | +37.48% | skill 1/1; base 1/1 |
| codex | usd-performance-report-schema-conformance | 677,593 | 205,495 | +472,098 | +229.74% | skill 1/1; base 1/1 |
| codex | usd-performance-runtime-preflight-gate | 589,780 | 168,312 | +421,468 | +250.41% | skill 1/1; base 1/1 |
| codex | usd-performance-structural-only-report | 1,164,628 | 313,361 | +851,267 | +271.66% | skill 1/1; base 1/1 |
| codex | usd-performance-viewer-build-distractor-negative | 33,369 | 17,775 | +15,594 | +87.73% | skill 1/1; base 1/1 |
| codex | usd-performance-zero-work-operation | 287,416 | 87,366 | +200,050 | +228.98% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,795,167 | 5,159,568 | +7,635,599 | +147.99% | skill 20/20; base 20/20 |

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
