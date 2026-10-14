# Skill Benchmark: doca-rmax

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-rmax`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:6a4d01baa73511220efb9c0ead385bb4eff7924fe876469a2bf3c1cbfebf8a79` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 97.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 45.0% → 100.0% (+55.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.3% → 100.0% (+53.7 points) | 36.3% → 100.0% (+63.7 points) |
| Efficiency | 76.9% — baseline ran, but no comparable score was available; uplift unavailable | 94.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,155,055 | 651,397 | +503,658 | +77.32% | skill 4/4; base 4/4 |
| claude-code | doca-rmax.explicit-positive.v1 | 226,804 | 247,596 | -20,792 | -8.40% | skill 1/1; base 1/1 |
| claude-code | doca-rmax.implicit-positive.v1 | 314,679 | 154,367 | +160,312 | +103.85% | skill 1/1; base 1/1 |
| claude-code | doca-rmax.implicit-positive.v2 | 208,462 | 93,025 | +115,437 | +124.09% | skill 1/1; base 1/1 |
| claude-code | doca-rmax.negative.v1 | 405,110 | 156,409 | +248,701 | +159.01% | skill 1/1; base 1/1 |
| codex | All cases | 530,450 | 118,678 | +411,772 | +346.97% | skill 4/4; base 4/4 |
| codex | doca-rmax.explicit-positive.v1 | 136,483 | 26,453 | +110,030 | +415.95% | skill 1/1; base 1/1 |
| codex | doca-rmax.implicit-positive.v1 | 146,285 | 14,695 | +131,590 | +895.47% | skill 1/1; base 1/1 |
| codex | doca-rmax.implicit-positive.v2 | 146,309 | 27,726 | +118,583 | +427.70% | skill 1/1; base 1/1 |
| codex | doca-rmax.negative.v1 | 101,373 | 49,804 | +51,569 | +103.54% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,685,505 | 770,075 | +915,430 | +118.88% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 20 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-rmax/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/libs/doca-rmax`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`BENCHMARK.md:86`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`BENCHMARK.md:87`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`BENCHMARK.md:88`)
- 15 additional finding(s) are available in the full evaluation artifacts.

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
