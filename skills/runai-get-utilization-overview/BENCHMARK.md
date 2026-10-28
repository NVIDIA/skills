# Skill Benchmark: runai-get-utilization-overview

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `runai-get-utilization-overview`
- Evaluation date: 2026-10-25
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 22 evaluation tasks (16 positive, 6 negative)
- Dataset digest: `sha256:a7626a9135219d3b9fa142abfafe70d7b324df6a04228e5890cf22c718eb975b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 69.1% → 89.1% (+20.0 points) | 69.1% → 87.3% (+18.2 points) |
| Discoverability | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.9% → 81.4% (+22.5 points) | 61.8% → 85.3% (+23.5 points) |
| Efficiency | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 62.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,549,377 | 3,294,332 | +1,255,045 | +38.10% | skill 22/22; base 22/22 |
| claude-code | A1-idle-holders-7d | 275,255 | 74,922 | +200,333 | +267.39% | skill 1/1; base 1/1 |
| claude-code | A2-fractional-holder | 169,034 | 165,299 | +3,735 | +2.26% | skill 1/1; base 1/1 |
| claude-code | A4-idle-gpus-right-now | 73,588 | 71,211 | +2,377 | +3.34% | skill 1/1; base 1/1 |
| claude-code | A5-utilization-by-department | 171,591 | 109,591 | +62,000 | +56.57% | skill 1/1; base 1/1 |
| claude-code | A6-node-pool-waste | 263,730 | 106,396 | +157,334 | +147.88% | skill 1/1; base 1/1 |
| claude-code | A7-drill-vision-train | 168,465 | 110,478 | +57,987 | +52.49% | skill 1/1; base 1/1 |
| claude-code | A8-gpu-hours-last-24h | 273,066 | 115,440 | +157,626 | +136.54% | skill 1/1; base 1/1 |
| claude-code | B1-quota-over-borrowing-routes-out | 104,523 | 69,117 | +35,406 | +51.23% | skill 1/1; base 1/1 |
| claude-code | B2-cluster-health-routes-out | 152,718 | 148,903 | +3,815 | +2.56% | skill 1/1; base 1/1 |
| claude-code | B4-one-workload-health-routes-out | 108,188 | 148,243 | -40,055 | -27.02% | skill 1/1; base 1/1 |
| claude-code | C1-no-cluster-id-asks | 117,521 | 171,968 | -54,447 | -31.66% | skill 1/1; base 1/1 |
| claude-code | C10-find-and-stop-mixed | 168,492 | 112,517 | +55,975 | +49.75% | skill 1/1; base 1/1 |
| claude-code | C2-missing-utilization-unknown | 172,234 | 153,313 | +18,921 | +12.34% | skill 1/1; base 1/1 |
| claude-code | C3-bursty-not-idle | 168,775 | 161,139 | +7,636 | +4.74% | skill 1/1; base 1/1 |
| claude-code | C4-memory-loaded-not-busy | 168,258 | 149,064 | +19,194 | +12.88% | skill 1/1; base 1/1 |
| claude-code | C5-live-outside-accounting | 217,401 | 188,033 | +29,368 | +15.62% | skill 1/1; base 1/1 |
| claude-code | C7-nvidia-smi-request | 111,692 | 107,558 | +4,134 | +3.84% | skill 1/1; base 1/1 |
| claude-code | C8-window-longer-than-30-days | 283,057 | 204,790 | +78,267 | +38.22% | skill 1/1; base 1/1 |
| claude-code | C9-old-cluster-no-memory-metric | 265,984 | 153,145 | +112,839 | +73.68% | skill 1/1; base 1/1 |
| claude-code | D1-stop-idle-notebooks | 171,554 | 149,045 | +22,509 | +15.10% | skill 1/1; base 1/1 |
| claude-code | D4-chargeback-cost | 563,261 | 336,817 | +226,444 | +67.23% | skill 1/1; base 1/1 |
| claude-code | D5-week-over-week-trend | 380,990 | 287,343 | +93,647 | +32.59% | skill 1/1; base 1/1 |
| codex | All cases | 5,528,171 | 2,965,372 | +2,562,799 | +86.42% | skill 22/22; base 22/22 |
| codex | A1-idle-holders-7d | 1,387,176 | 33,306 | +1,353,870 | +4064.94% | skill 1/1; base 1/1 |
| codex | A2-fractional-holder | 155,357 | 106,625 | +48,732 | +45.70% | skill 1/1; base 1/1 |
| codex | A4-idle-gpus-right-now | 77,797 | 33,349 | +44,448 | +133.28% | skill 1/1; base 1/1 |
| codex | A5-utilization-by-department | 217,399 | 89,801 | +127,598 | +142.09% | skill 1/1; base 1/1 |
| codex | A6-node-pool-waste | 381,248 | 33,084 | +348,164 | +1052.36% | skill 1/1; base 1/1 |
| codex | A7-drill-vision-train | 155,590 | 89,688 | +65,902 | +73.48% | skill 1/1; base 1/1 |
| codex | A8-gpu-hours-last-24h | 752,909 | 135,206 | +617,703 | +456.86% | skill 1/1; base 1/1 |
| codex | B1-quota-over-borrowing-routes-out | 32,588 | 32,131 | +457 | +1.42% | skill 1/1; base 1/1 |
| codex | B2-cluster-health-routes-out | 175,035 | 150,556 | +24,479 | +16.26% | skill 1/1; base 1/1 |
| codex | B4-one-workload-health-routes-out | 63,976 | 79,023 | -15,047 | -19.04% | skill 1/1; base 1/1 |
| codex | C1-no-cluster-id-asks | 77,403 | 33,663 | +43,740 | +129.93% | skill 1/1; base 1/1 |
| codex | C10-find-and-stop-mixed | 124,974 | 47,478 | +77,496 | +163.23% | skill 1/1; base 1/1 |
| codex | C2-missing-utilization-unknown | 151,317 | 91,813 | +59,504 | +64.81% | skill 1/1; base 1/1 |
| codex | C3-bursty-not-idle | 155,598 | 71,679 | +83,919 | +117.08% | skill 1/1; base 1/1 |
| codex | C4-memory-loaded-not-busy | 152,918 | 94,330 | +58,588 | +62.11% | skill 1/1; base 1/1 |
| codex | C5-live-outside-accounting | 123,199 | 197,867 | -74,668 | -37.74% | skill 1/1; base 1/1 |
| codex | C7-nvidia-smi-request | 47,892 | 47,627 | +265 | +0.56% | skill 1/1; base 1/1 |
| codex | C8-window-longer-than-30-days | 684,431 | 896,462 | -212,031 | -23.65% | skill 1/1; base 1/1 |
| codex | C9-old-cluster-no-memory-metric | 296,988 | 31,584 | +265,404 | +840.31% | skill 1/1; base 1/1 |
| codex | D1-stop-idle-notebooks | 34,500 | 81,258 | -46,758 | -57.54% | skill 1/1; base 1/1 |
| codex | D4-chargeback-cost | 245,311 | 329,825 | -84,514 | -25.62% | skill 1/1; base 1/1 |
| codex | D5-week-over-week-trend | 34,565 | 259,017 | -224,452 | -86.66% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,077,548 | 6,259,704 | +3,817,844 | +60.99% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_discoverability: Description very long (887 chars, recommend 50-150) (`skills/runai-get-utilization-overview/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/runai-get-utilization-overview/SKILL.md`)
- **LOW** SCHEMA/unexpected_file: Unexpected 'LICENSE' in skill root (`skills/runai-get-utilization-overview/LICENSE`)

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
