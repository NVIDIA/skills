# Skill Benchmark: i4h-workflow-train-rl

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `i4h-workflow-train-rl`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 9 evaluation tasks (9 positive)
- Dataset digest: `sha256:eaab434eabd6b5ea5e26065aef290992f9fb29cbadea578299cbb70d56828b94` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | Not available | 60.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 79.2% → 37.5% (-41.7 points) |
| Correctness | Not available | 27.5% → 81.7% (+54.2 points) |
| Discoverability | Not available | 77.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 10.6% → 33.4% (+22.8 points) |
| Efficiency | Not available | 71.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 13,429,024 | 21,169,179 | N/A | N/A | skill 10/27; base 26/27 |
| claude-code | i4h-workflow-train-rl-new-instrument-reach | 8,327,735 | 4,750,947 | +3,576,788 | +75.29% | skill 2/2; base 2/2 |
| claude-code | i4h-workflow-train-rl-readme-export-instrument-reach | 266,558 | 792,997 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-readme-train-instrument-reach | 262,197 | 3,070,667 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-specialty-trocar-full-lifecycle | 390,207 | 921,972 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-specialty-trocar-pretrained-model | 538,042 | 3,762,151 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-specialty-ultrasound-full-lifecycle | 2,235,510 | 5,391,470 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-trocar-eval | 462,234 | 816,591 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-trocar-post-training | 426,732 | 622,930 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-train-rl-ultrasound-export | 519,809 | 1,039,454 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 15,261,479 | 10,269,959 | N/A | N/A | skill 12/12; base 24/24 |
| codex | i4h-workflow-train-rl-new-instrument-reach | 2,253,704 | 385,724 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-train-rl-readme-export-instrument-reach | 1,534,112 | 1,147,733 | N/A | N/A | skill 2/2; base 3/3 |
| codex | i4h-workflow-train-rl-readme-train-instrument-reach | 1,060,001 | 1,013,591 | +46,410 | +4.58% | skill 2/2; base 2/2 |
| codex | i4h-workflow-train-rl-specialty-trocar-full-lifecycle | 1,186,917 | 316,706 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-train-rl-specialty-trocar-pretrained-model | 2,557,737 | 3,939,996 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-train-rl-specialty-ultrasound-full-lifecycle | 3,201,349 | 1,682,531 | N/A | N/A | skill 2/2; base 3/3 |
| codex | i4h-workflow-train-rl-trocar-eval | 139,660 | 413,257 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-train-rl-trocar-post-training | 2,368,347 | 414,219 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-train-rl-ultrasound-export | 959,652 | 956,202 | +3,450 | +0.36% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 28,690,503 | 31,439,138 | N/A | N/A | skill 22/39; base 50/51 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 1 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-train-rl/SKILL.md`)

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
