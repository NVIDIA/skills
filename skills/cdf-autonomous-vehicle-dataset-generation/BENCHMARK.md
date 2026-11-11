# Skill Benchmark: cdf-autonomous-vehicle-dataset-generation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-autonomous-vehicle-dataset-generation`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:35d93cc2c87bd5329bf1ee47eda7b0e50d848bad825d5883757c2e3e00736988` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 87.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 100.0% (+33.3 points) | 100.0% → 83.3% (-16.7 points) |
| Correctness | 23.3% → 100.0% (+76.7 points) | 23.3% → 100.0% (+76.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.5% → 88.1% (+68.6 points) | 16.3% → 73.1% (+56.8 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,594,568 | 6,523,558 | -4,928,990 | -75.56% | skill 6/6; base 6/6 |
| claude-code | av-negative-event-generation | 394,494 | 1,442,819 | -1,048,325 | -72.66% | skill 1/1; base 1/1 |
| claude-code | av-negative-standalone-vda | 30,277 | 3,518,523 | -3,488,246 | -99.14% | skill 1/1; base 1/1 |
| claude-code | day-starter-payload | 333,568 | 415,117 | -81,549 | -19.64% | skill 1/1; base 1/1 |
| claude-code | rain-starter-payload | 272,662 | 418,388 | -145,726 | -34.83% | skill 1/1; base 1/1 |
| claude-code | retrieval-starter-payload | 224,159 | 243,141 | -18,982 | -7.81% | skill 1/1; base 1/1 |
| claude-code | snow-starter-payload | 339,408 | 485,570 | -146,162 | -30.10% | skill 1/1; base 1/1 |
| codex | All cases | 1,328,416 | 1,047,120 | +281,296 | +26.86% | skill 6/6; base 6/6 |
| codex | av-negative-event-generation | 46,672 | 30,008 | +16,664 | +55.53% | skill 1/1; base 1/1 |
| codex | av-negative-standalone-vda | 641,980 | 427,058 | +214,922 | +50.33% | skill 1/1; base 1/1 |
| codex | day-starter-payload | 221,722 | 115,015 | +106,707 | +92.78% | skill 1/1; base 1/1 |
| codex | rain-starter-payload | 166,152 | 85,370 | +80,782 | +94.63% | skill 1/1; base 1/1 |
| codex | retrieval-starter-payload | 129,701 | 147,911 | -18,210 | -12.31% | skill 1/1; base 1/1 |
| codex | snow-starter-payload | 122,189 | 241,758 | -119,569 | -49.46% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,922,984 | 7,570,678 | -4,647,694 | -61.39% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in airflow-direct-api.md (`skills/cdf-autonomous-vehicle-dataset-generation/SKILL.md`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl --fail --silent --show-error --connect-timeout 10 --max-time 60 \
        -H 'Content-Type: application/json' --data-binary  (`references/airflow-direct-api.md:30`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl --silent --show-error --max-time 30 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  -o "$RUN_FILE" -w '%{http_code}' \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID_PATH")
printf 'Run loo (`references/airflow-direct-api.md:59`)
- **LOW** QUALITY/quality_discoverability: Description very long (305 chars, recommend 50-150) (`skills/cdf-autonomous-vehicle-dataset-generation/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/cdf-autonomous-vehicle-dataset-generation/SKILL.md`)
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
