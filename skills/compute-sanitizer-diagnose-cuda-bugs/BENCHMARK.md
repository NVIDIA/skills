# Skill Benchmark: compute-sanitizer-diagnose-cuda-bugs

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compute-sanitizer-diagnose-cuda-bugs`
- Evaluation date: 2026-10-13
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 16 evaluation tasks (15 positive, 1 negative)
- Dataset digest: `sha256:8a351af431b170a1edcef4f1cfc3449b1209d407dc0915bcf32f226b22c78ff8` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 96.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 96.3% → 98.8% (+2.5 points) | 96.3% → 100.0% (+3.7 points) |
| Discoverability | 93.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 81.3% → 96.4% (+15.1 points) | 79.0% → 94.4% (+15.4 points) |
| Efficiency | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,610,130 | 735,129 | +875,001 | +119.03% | skill 16/16; base 16/16 |
| claude-code | cs-boundary-global-race | 31,721 | 32,233 | -512 | -1.59% | skill 1/1; base 1/1 |
| claude-code | cs-boundary-host-race | 106,439 | 34,324 | +72,115 | +210.10% | skill 1/1; base 1/1 |
| claude-code | cs-flag-error-exitcode | 64,972 | 29,920 | +35,052 | +117.15% | skill 1/1; base 1/1 |
| claude-code | cs-flag-leak-check | 104,909 | 30,339 | +74,570 | +245.79% | skill 1/1; base 1/1 |
| claude-code | cs-flag-num-cuda-barriers | 103,284 | 60,457 | +42,827 | +70.84% | skill 1/1; base 1/1 |
| claude-code | cs-flag-padding | 105,167 | 31,748 | +73,419 | +231.26% | skill 1/1; base 1/1 |
| claude-code | cs-flag-track-unused | 103,975 | 61,472 | +42,503 | +69.14% | skill 1/1; base 1/1 |
| claude-code | cs-injection-sanitizer-output | 105,438 | 30,426 | +75,012 | +246.54% | skill 1/1; base 1/1 |
| claude-code | cs-injection-source-comment | 106,542 | 30,443 | +76,099 | +249.97% | skill 1/1; base 1/1 |
| claude-code | cs-negative-host-logic | 119,655 | 118,601 | +1,054 | +0.89% | skill 1/1; base 1/1 |
| claude-code | cs-route-initcheck-uninit | 103,146 | 29,935 | +73,211 | +244.57% | skill 1/1; base 1/1 |
| claude-code | cs-route-memcheck-oob | 105,610 | 31,400 | +74,210 | +236.34% | skill 1/1; base 1/1 |
| claude-code | cs-route-racecheck-shared | 104,629 | 30,502 | +74,127 | +243.02% | skill 1/1; base 1/1 |
| claude-code | cs-route-synccheck-barrier | 103,642 | 30,021 | +73,621 | +245.23% | skill 1/1; base 1/1 |
| claude-code | cs-trap-clean-memcheck | 66,750 | 31,518 | +35,232 | +111.78% | skill 1/1; base 1/1 |
| claude-code | cs-trap-misleading-race | 174,251 | 121,790 | +52,461 | +43.07% | skill 1/1; base 1/1 |
| codex | All cases | 869,274 | 604,678 | +264,596 | +43.76% | skill 16/16; base 16/16 |
| codex | cs-boundary-global-race | 35,353 | 28,354 | +6,999 | +24.68% | skill 1/1; base 1/1 |
| codex | cs-boundary-host-race | 31,346 | 25,210 | +6,136 | +24.34% | skill 1/1; base 1/1 |
| codex | cs-flag-error-exitcode | 30,377 | 17,869 | +12,508 | +70.00% | skill 1/1; base 1/1 |
| codex | cs-flag-leak-check | 68,003 | 35,792 | +32,211 | +89.99% | skill 1/1; base 1/1 |
| codex | cs-flag-num-cuda-barriers | 83,791 | 30,828 | +52,963 | +171.80% | skill 1/1; base 1/1 |
| codex | cs-flag-padding | 87,893 | 27,006 | +60,887 | +225.46% | skill 1/1; base 1/1 |
| codex | cs-flag-track-unused | 74,450 | 51,580 | +22,870 | +44.34% | skill 1/1; base 1/1 |
| codex | cs-injection-sanitizer-output | 30,664 | 13,642 | +17,022 | +124.78% | skill 1/1; base 1/1 |
| codex | cs-injection-source-comment | 30,963 | 13,690 | +17,273 | +126.17% | skill 1/1; base 1/1 |
| codex | cs-negative-host-logic | 56,800 | 83,240 | -26,440 | -31.76% | skill 1/1; base 1/1 |
| codex | cs-route-initcheck-uninit | 48,845 | 17,925 | +30,920 | +172.50% | skill 1/1; base 1/1 |
| codex | cs-route-memcheck-oob | 50,631 | 25,382 | +25,249 | +99.48% | skill 1/1; base 1/1 |
| codex | cs-route-racecheck-shared | 49,763 | 29,328 | +20,435 | +69.68% | skill 1/1; base 1/1 |
| codex | cs-route-synccheck-barrier | 49,031 | 19,260 | +29,771 | +154.57% | skill 1/1; base 1/1 |
| codex | cs-trap-clean-memcheck | 30,689 | 70,490 | -39,801 | -56.46% | skill 1/1; base 1/1 |
| codex | cs-trap-misleading-race | 110,675 | 115,082 | -4,407 | -3.83% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,479,404 | 1,339,807 | +1,139,597 | +85.06% | skill 32/32; base 32/32 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 16 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_discoverability: Description very long (830 chars, recommend 50-150) (`skills/compute-sanitizer-diagnose-cuda-bugs/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Output Format section does not define a concrete structure (`skills/compute-sanitizer-diagnose-cuda-bugs/SKILL.md`)

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
