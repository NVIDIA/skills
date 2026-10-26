# Skill Benchmark: runai-get-workload-health-summary

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `runai-get-workload-health-summary`
- Evaluation date: 2026-10-25
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 30 evaluation tasks (22 positive, 8 negative)
- Dataset digest: `sha256:6e09ea63475e9a74d6b84883683da186393aaa90546908bea0e670ac7be05515` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 84.4% — baseline ran, but no comparable score was available; uplift unavailable | 85.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 67.3% → 84.0% (+16.7 points) | 70.0% → 90.7% (+20.7 points) |
| Discoverability | 75.5% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 44.3% → 64.9% (+20.6 points) | 48.8% → 79.0% (+30.2 points) |
| Efficiency | 97.4% — baseline ran, but no comparable score was available; uplift unavailable | 66.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,713,769 | 5,007,438 | -293,669 | -5.86% | skill 30/30; base 30/30 |
| claude-code | A1-overview-failed-node-correlation | 230,570 | 241,498 | -10,928 | -4.53% | skill 1/1; base 1/1 |
| claude-code | A10-drilldown-workload-id-degraded | 210,001 | 107,368 | +102,633 | +95.59% | skill 1/1; base 1/1 |
| claude-code | A11-drilldown-name-project-resolution | 118,408 | 185,494 | -67,086 | -36.17% | skill 1/1; base 1/1 |
| claude-code | A12-drilldown-inference-metrics | 254,069 | 142,714 | +111,355 | +78.03% | skill 1/1; base 1/1 |
| claude-code | A3-overview-project-idle | 187,918 | 148,601 | +39,317 | +26.46% | skill 1/1; base 1/1 |
| claude-code | A4-overview-department-failed | 151,175 | 287,059 | -135,884 | -47.34% | skill 1/1; base 1/1 |
| claude-code | A6-overview-image-pattern | 315,097 | 264,137 | +50,960 | +19.29% | skill 1/1; base 1/1 |
| claude-code | A7-overview-restart-sort | 109,364 | 148,123 | -38,759 | -26.17% | skill 1/1; base 1/1 |
| claude-code | A8-overview-limit-cap | 67,530 | 67,312 | +218 | +0.32% | skill 1/1; base 1/1 |
| claude-code | B1-known-failed-root-cause-routes-out | 106,118 | 106,016 | +102 | +0.10% | skill 1/1; base 1/1 |
| claude-code | B3-inference-latency-root-cause-routes-out | 141,590 | 233,347 | -91,757 | -39.32% | skill 1/1; base 1/1 |
| claude-code | B4-quota-fairness-routes-out | 177,752 | 285,787 | -108,035 | -37.80% | skill 1/1; base 1/1 |
| claude-code | B5-unscoped-cluster-dump-asks-filter | 71,746 | 108,702 | -36,956 | -34.00% | skill 1/1; base 1/1 |
| claude-code | B6-bulk-delete-safety | 147,992 | 148,979 | -987 | -0.66% | skill 1/1; base 1/1 |
| claude-code | C1-healthy-no-handoff | 73,704 | 214,594 | -140,890 | -65.65% | skill 1/1; base 1/1 |
| claude-code | C4-scheduling-only-pending-boundary | 180,550 | 417,318 | -236,768 | -56.74% | skill 1/1; base 1/1 |
| claude-code | C5-missing-policy-graceful | 208,477 | 106,703 | +101,774 | +95.38% | skill 1/1; base 1/1 |
| claude-code | D1-cluster-health-routes-elsewhere | 204,103 | 314,711 | -110,608 | -35.15% | skill 1/1; base 1/1 |
| claude-code | D2-utilization-overview-boundary | 105,482 | 146,304 | -40,822 | -27.90% | skill 1/1; base 1/1 |
| claude-code | D3-successful-workload-no-failure | 115,324 | 68,352 | +46,972 | +68.72% | skill 1/1; base 1/1 |
| claude-code | D6-unrelated-request | 101,762 | 138,249 | -36,487 | -26.39% | skill 1/1; base 1/1 |
| claude-code | safety-cluster-mismatch | 117,767 | 105,947 | +11,820 | +11.16% | skill 1/1; base 1/1 |
| claude-code | safety-correlation-budget | 173,632 | 174,585 | -953 | -0.55% | skill 1/1; base 1/1 |
| claude-code | safety-untrusted-tool-output | 115,464 | 67,074 | +48,390 | +72.14% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-count-only | 115,559 | 67,261 | +48,298 | +71.81% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-failed-snapshot | 209,987 | 141,775 | +68,212 | +48.11% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-image-pull-snapshot | 210,035 | 142,287 | +67,748 | +47.61% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-node-overview | 213,322 | 252,183 | -38,861 | -15.41% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-project-overview | 68,488 | 68,022 | +466 | +0.69% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-running-snapshot | 210,783 | 106,936 | +103,847 | +97.11% | skill 1/1; base 1/1 |
| codex | All cases | 4,060,623 | 4,201,942 | -141,319 | -3.36% | skill 30/30; base 30/30 |
| codex | A1-overview-failed-node-correlation | 268,619 | 134,720 | +133,899 | +99.39% | skill 1/1; base 1/1 |
| codex | A10-drilldown-workload-id-degraded | 213,585 | 126,919 | +86,666 | +68.28% | skill 1/1; base 1/1 |
| codex | A11-drilldown-name-project-resolution | 71,993 | 129,824 | -57,831 | -44.55% | skill 1/1; base 1/1 |
| codex | A12-drilldown-inference-metrics | 203,272 | 128,043 | +75,229 | +58.75% | skill 1/1; base 1/1 |
| codex | A3-overview-project-idle | 93,940 | 78,504 | +15,436 | +19.66% | skill 1/1; base 1/1 |
| codex | A4-overview-department-failed | 234,859 | 240,048 | -5,189 | -2.16% | skill 1/1; base 1/1 |
| codex | A6-overview-image-pattern | 409,784 | 601,011 | -191,227 | -31.82% | skill 1/1; base 1/1 |
| codex | A7-overview-restart-sort | 73,640 | 96,626 | -22,986 | -23.79% | skill 1/1; base 1/1 |
| codex | A8-overview-limit-cap | 284,181 | 30,465 | +253,716 | +832.81% | skill 1/1; base 1/1 |
| codex | B1-known-failed-root-cause-routes-out | 61,853 | 45,862 | +15,991 | +34.87% | skill 1/1; base 1/1 |
| codex | B3-inference-latency-root-cause-routes-out | 144,507 | 177,515 | -33,008 | -18.59% | skill 1/1; base 1/1 |
| codex | B4-quota-fairness-routes-out | 143,005 | 236,781 | -93,776 | -39.60% | skill 1/1; base 1/1 |
| codex | B5-unscoped-cluster-dump-asks-filter | 33,021 | 32,932 | +89 | +0.27% | skill 1/1; base 1/1 |
| codex | B6-bulk-delete-safety | 32,887 | 72,709 | -39,822 | -54.77% | skill 1/1; base 1/1 |
| codex | C1-healthy-no-handoff | 32,803 | 159,853 | -127,050 | -79.48% | skill 1/1; base 1/1 |
| codex | C4-scheduling-only-pending-boundary | 30,612 | 161,402 | -130,790 | -81.03% | skill 1/1; base 1/1 |
| codex | C5-missing-policy-graceful | 171,475 | 126,865 | +44,610 | +35.16% | skill 1/1; base 1/1 |
| codex | D1-cluster-health-routes-elsewhere | 232,376 | 179,851 | +52,525 | +29.20% | skill 1/1; base 1/1 |
| codex | D2-utilization-overview-boundary | 63,493 | 61,153 | +2,340 | +3.83% | skill 1/1; base 1/1 |
| codex | D3-successful-workload-no-failure | 51,197 | 30,374 | +20,823 | +68.56% | skill 1/1; base 1/1 |
| codex | D6-unrelated-request | 62,079 | 45,819 | +16,260 | +35.49% | skill 1/1; base 1/1 |
| codex | safety-cluster-mismatch | 51,334 | 230,148 | -178,814 | -77.70% | skill 1/1; base 1/1 |
| codex | safety-correlation-budget | 78,873 | 109,911 | -31,038 | -28.24% | skill 1/1; base 1/1 |
| codex | safety-untrusted-tool-output | 51,033 | 30,242 | +20,791 | +68.75% | skill 1/1; base 1/1 |
| codex | v2-recorded-count-only | 72,656 | 30,248 | +42,408 | +140.20% | skill 1/1; base 1/1 |
| codex | v2-recorded-failed-snapshot | 234,079 | 212,821 | +21,258 | +9.99% | skill 1/1; base 1/1 |
| codex | v2-recorded-image-pull-snapshot | 212,442 | 193,328 | +19,114 | +9.89% | skill 1/1; base 1/1 |
| codex | v2-recorded-node-overview | 137,640 | 239,829 | -102,189 | -42.61% | skill 1/1; base 1/1 |
| codex | v2-recorded-project-overview | 74,057 | 79,351 | -5,294 | -6.67% | skill 1/1; base 1/1 |
| codex | v2-recorded-running-snapshot | 235,328 | 178,788 | +56,540 | +31.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,774,392 | 9,209,380 | -434,988 | -4.72% | skill 60/60; base 60/60 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 30 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/runai-get-workload-health-summary`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/runai-get-workload-health-summary/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/runai-get-workload-health-summary/SKILL.md`)
- **LOW** SCHEMA/unexpected_file: Unexpected 'LICENSE' in skill root (`skills/runai-get-workload-health-summary/LICENSE`)

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
