# Skill Benchmark: i4h-workflow-create

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `i4h-workflow-create`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (5 positive)
- Dataset digest: `sha256:5cc46d1f852070edff4a32347f96e7887fbdf3302ea6f9adbee8c105bd2f2e8a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 80.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 79.2% → 100.0% (+20.8 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 35.0% → 80.0% (+45.0 points) | 34.6% → 72.0% (+37.4 points) |
| Discoverability | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 13.8% → 37.0% (+23.2 points) | 24.8% → 34.0% (+9.2 points) |
| Efficiency | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 97.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,255,623 | 7,645,488 | N/A | N/A | skill 5/5; base 12/12 |
| claude-code | i4h-workflow-create-existing-global-id | 226,097 | 1,377,501 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-create-generic-fast | 297,088 | 1,246,881 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-create-missing-specialty | 92,904 | 4,361,294 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | i4h-workflow-create-readme-blank | 262,562 | 251,779 | +10,783 | +4.28% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-create-readme-robotic-instrument-reach | 376,972 | 408,033 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 423,283 | 1,055,531 | N/A | N/A | skill 5/5; base 11/11 |
| codex | i4h-workflow-create-existing-global-id | 97,892 | 254,511 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-create-generic-fast | 92,316 | 267,136 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-create-missing-specialty | 28,288 | 306,233 | N/A | N/A | skill 1/1; base 3/3 |
| codex | i4h-workflow-create-readme-blank | 96,551 | 113,548 | -16,997 | -14.97% | skill 1/1; base 1/1 |
| codex | i4h-workflow-create-readme-robotic-instrument-reach | 108,236 | 114,103 | -5,867 | -5.14% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,678,906 | 8,701,019 | N/A | N/A | skill 10/10; base 23/23 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 1 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-create/SKILL.md`)

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
