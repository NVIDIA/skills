# Skill Benchmark: nvflare-convert-huggingface

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvflare-convert-huggingface`
- Validation status: `passed`
- Evaluation date: 2026-11-02
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 21 evaluation tasks (21 positive)
- Dataset digest: `sha256:3794dc432acf265997fa5eb68a48b58af6f70a01c5578fbd5eb674f88b7ac22f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvflare-convert-huggingface | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvflare-convert-huggingface | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvflare-convert-huggingface | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvflare-convert-huggingface | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvflare-convert-huggingface | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

- Controls for `nvflare-convert-huggingface`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 85.9% — baseline ran, but no comparable score was available; uplift unavailable | 76.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 45.2% → 100.0% (+54.8 points) | 57.1% → 73.8% (+16.7 points) |
| Correctness | 71.4% → 93.3% (+21.9 points) | 70.5% → 90.5% (+20.0 points) |
| Discoverability | 87.1% — baseline ran, but no comparable score was available; uplift unavailable | 75.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.4% → 69.1% (+21.7 points) | 50.3% → 66.9% (+16.6 points) |
| Efficiency | 79.8% — baseline ran, but no comparable score was available; uplift unavailable | 75.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 82.7% → 85.8% (+3.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 77.6% → 82.2% (+4.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 12,940,204 | 25,314,229 | -12,374,025 | -48.88% | skill 21/21; base 21/21 |
| claude-code | huggingface-combined-fedstats-workflow-boundary | 116,899 | 142,379 | -25,480 | -17.90% | skill 1/1; base 1/1 |
| claude-code | huggingface-convert-basic | 915,261 | 1,698,303 | -783,042 | -46.11% | skill 1/1; base 1/1 |
| claude-code | huggingface-convert-custom-aggregation | 1,763,199 | 2,365,661 | -602,462 | -25.47% | skill 1/1; base 1/1 |
| claude-code | huggingface-convert-external-data-path | 1,076,266 | 1,514,228 | -437,962 | -28.92% | skill 1/1; base 1/1 |
| claude-code | huggingface-convert-peft-sft | 913,258 | 2,903,649 | -1,990,391 | -68.55% | skill 1/1; base 1/1 |
| claude-code | huggingface-convert-relative-data-path | 778,007 | 1,152,744 | -374,737 | -32.51% | skill 1/1; base 1/1 |
| claude-code | huggingface-ddp-contract | 965,391 | 3,063,379 | -2,097,988 | -68.49% | skill 1/1; base 1/1 |
| claude-code | huggingface-factory-distributed-direct-return | 812,479 | 1,680,971 | -868,492 | -51.67% | skill 1/1; base 1/1 |
| claude-code | huggingface-factory-evaluated-callback-metrics | 663,344 | 1,325,837 | -662,493 | -49.97% | skill 1/1; base 1/1 |
| claude-code | huggingface-factory-peft-sft-local-return | 1,215,268 | 1,118,926 | +96,342 | +8.61% | skill 1/1; base 1/1 |
| claude-code | huggingface-factory-train-only | 930,521 | 1,484,883 | -554,362 | -37.33% | skill 1/1; base 1/1 |
| claude-code | huggingface-global-negative-serving | 279,787 | 509,915 | -230,128 | -45.13% | skill 1/1; base 1/1 |
| claude-code | huggingface-injection-resistance | 734,080 | 665,349 | +68,731 | +10.33% | skill 1/1; base 1/1 |
| claude-code | huggingface-loss-only-selection-metric | 213,644 | 107,351 | +106,293 | +99.01% | skill 1/1; base 1/1 |
| claude-code | huggingface-lower-is-better-metric | 213,960 | 109,224 | +104,736 | +95.89% | skill 1/1; base 1/1 |
| claude-code | huggingface-negative-dual-trainer | 114,584 | 142,769 | -28,185 | -19.74% | skill 1/1; base 1/1 |
| claude-code | huggingface-negative-lightning | 146,426 | 1,020,756 | -874,330 | -85.66% | skill 1/1; base 1/1 |
| claude-code | huggingface-negative-manual-pytorch | 110,068 | 70,570 | +39,498 | +55.97% | skill 1/1; base 1/1 |
| claude-code | huggingface-offline-model-cache-miss | 697,374 | 2,278,403 | -1,581,029 | -69.39% | skill 1/1; base 1/1 |
| claude-code | huggingface-train-only-disables-model-selection | 203,800 | 1,177,832 | -974,032 | -82.70% | skill 1/1; base 1/1 |
| claude-code | huggingface-unsupported-privacy-request | 76,588 | 781,100 | -704,512 | -90.19% | skill 1/1; base 1/1 |
| codex | All cases | 11,098,180 | 20,544,760 | -9,446,580 | -45.98% | skill 21/21; base 21/21 |
| codex | huggingface-combined-fedstats-workflow-boundary | 29,570 | 54,971 | -25,401 | -46.21% | skill 1/1; base 1/1 |
| codex | huggingface-convert-basic | 889,478 | 1,621,259 | -731,781 | -45.14% | skill 1/1; base 1/1 |
| codex | huggingface-convert-custom-aggregation | 1,146,977 | 1,202,845 | -55,868 | -4.64% | skill 1/1; base 1/1 |
| codex | huggingface-convert-external-data-path | 916,050 | 2,045,881 | -1,129,831 | -55.22% | skill 1/1; base 1/1 |
| codex | huggingface-convert-peft-sft | 970,147 | 3,007,634 | -2,037,487 | -67.74% | skill 1/1; base 1/1 |
| codex | huggingface-convert-relative-data-path | 1,178,539 | 1,160,702 | +17,837 | +1.54% | skill 1/1; base 1/1 |
| codex | huggingface-ddp-contract | 807,108 | 1,748,111 | -941,003 | -53.83% | skill 1/1; base 1/1 |
| codex | huggingface-factory-distributed-direct-return | 584,373 | 1,329,827 | -745,454 | -56.06% | skill 1/1; base 1/1 |
| codex | huggingface-factory-evaluated-callback-metrics | 568,892 | 965,253 | -396,361 | -41.06% | skill 1/1; base 1/1 |
| codex | huggingface-factory-peft-sft-local-return | 638,361 | 369,921 | +268,440 | +72.57% | skill 1/1; base 1/1 |
| codex | huggingface-factory-train-only | 767,410 | 1,544,532 | -777,122 | -50.31% | skill 1/1; base 1/1 |
| codex | huggingface-global-negative-serving | 544,071 | 1,303,528 | -759,457 | -58.26% | skill 1/1; base 1/1 |
| codex | huggingface-injection-resistance | 877,916 | 1,264,752 | -386,836 | -30.59% | skill 1/1; base 1/1 |
| codex | huggingface-loss-only-selection-metric | 95,847 | 35,736 | +60,111 | +168.21% | skill 1/1; base 1/1 |
| codex | huggingface-lower-is-better-metric | 71,474 | 40,466 | +31,008 | +76.63% | skill 1/1; base 1/1 |
| codex | huggingface-negative-dual-trainer | 46,856 | 41,061 | +5,795 | +14.11% | skill 1/1; base 1/1 |
| codex | huggingface-negative-lightning | 45,484 | 57,122 | -11,638 | -20.37% | skill 1/1; base 1/1 |
| codex | huggingface-negative-manual-pytorch | 41,158 | 41,070 | +88 | +0.21% | skill 1/1; base 1/1 |
| codex | huggingface-offline-model-cache-miss | 745,903 | 1,233,599 | -487,696 | -39.53% | skill 1/1; base 1/1 |
| codex | huggingface-train-only-disables-model-selection | 68,828 | 40,964 | +27,864 | +68.02% | skill 1/1; base 1/1 |
| codex | huggingface-unsupported-privacy-request | 63,738 | 1,435,526 | -1,371,788 | -95.56% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 24,038,384 | 45,858,989 | -21,820,605 | -47.58% | skill 42/42; base 42/42 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 21 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvflare-convert-huggingface/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvflare-convert-huggingface/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nvflare-convert-huggingface/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (387 chars, recommend 50-150) (`skills/nvflare-convert-huggingface/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/nvflare-convert-huggingface/SKILL.md`)
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
