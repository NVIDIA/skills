# Skill Benchmark: nemo-relay-install

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-relay-install`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 15 evaluation tasks (14 positive, 1 negative)
- Dataset digest: `sha256:ac9a9cc622e67035055f7153d383fd5ea4ef6039a1dfb0eeb0bd7180f89dc9c1` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.2% — baseline ran, but no comparable score was available; uplift unavailable | 69.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 70.0% → 76.7% (+6.7 points) | 73.3% → 76.7% (+3.4 points) |
| Correctness | 50.7% → 82.7% (+32.0 points) | 52.0% → 81.3% (+29.3 points) |
| Discoverability | 92.8% — baseline ran, but no comparable score was available; uplift unavailable | 71.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.9% → 59.0% (+23.1 points) | 27.1% → 53.1% (+26.0 points) |
| Efficiency | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 65.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,062,774 | 2,610,570 | +1,452,204 | +55.63% | skill 15/15; base 15/15 |
| claude-code | nemo-relay-install-001 | 162,809 | 257,414 | -94,605 | -36.75% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-002 | 160,303 | 122,622 | +37,681 | +30.73% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-003 | 327,890 | 261,602 | +66,288 | +25.34% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-004 | 473,425 | 352,027 | +121,398 | +34.49% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-005 | 673,325 | 151,777 | +521,548 | +343.63% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-006 | 92,180 | 292,456 | -200,276 | -68.48% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-007 | 92,183 | 90,628 | +1,555 | +1.72% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-008 | 166,602 | 123,547 | +43,055 | +34.85% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-009 | 170,504 | 90,667 | +79,837 | +88.06% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-010 | 294,998 | 127,425 | +167,573 | +131.51% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-011 | 157,017 | 152,321 | +4,696 | +3.08% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-012 | 451,265 | 156,993 | +294,272 | +187.44% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-013 | 125,842 | 153,099 | -27,257 | -17.80% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-014 | 321,309 | 121,598 | +199,711 | +164.24% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-015 | 393,122 | 156,394 | +236,728 | +151.37% | skill 1/1; base 1/1 |
| codex | All cases | 5,499,543 | 4,294,538 | +1,205,005 | +28.06% | skill 15/15; base 15/15 |
| codex | nemo-relay-install-001 | 85,962 | 34,662 | +51,300 | +148.00% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-002 | 136,675 | 98,264 | +38,411 | +39.09% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-003 | 149,777 | 107,805 | +41,972 | +38.93% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-004 | 465,913 | 146,084 | +319,829 | +218.93% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-005 | 368,161 | 324,015 | +44,146 | +13.62% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-006 | 27,977 | 449,507 | -421,530 | -93.78% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-007 | 144,075 | 92,781 | +51,294 | +55.29% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-008 | 13,735 | 18,352 | -4,617 | -25.16% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-009 | 891,238 | 908,199 | -16,961 | -1.87% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-010 | 91,681 | 155,406 | -63,725 | -41.01% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-011 | 124,309 | 132,452 | -8,143 | -6.15% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-012 | 1,514,917 | 810,088 | +704,829 | +87.01% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-013 | 350,101 | 128,955 | +221,146 | +171.49% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-014 | 958,599 | 750,465 | +208,134 | +27.73% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-015 | 176,423 | 137,503 | +38,920 | +28.30% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 9,562,317 | 6,905,108 | +2,657,209 | +38.48% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: https://api.github.com/ (`SKILL.md:45`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
