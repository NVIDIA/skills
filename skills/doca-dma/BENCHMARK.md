# Skill Benchmark: doca-dma

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-dma`
- Evaluation date: 2026-10-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:58f730de82eaf2a44e485bfea8650a85067792eab4dc2ddae21139b7669af3bb` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 97.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 75.0% → 100.0% (+25.0 points) | 85.0% → 100.0% (+15.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.5% → 95.8% (+48.3 points) | 64.4% → 98.1% (+33.7 points) |
| Efficiency | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 96.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 691,026 | 789,322 | -98,296 | -12.45% | skill 4/4; base 4/4 |
| claude-code | doca-dma.explicit-positive.v1 | 210,042 | 313,587 | -103,545 | -33.02% | skill 1/1; base 1/1 |
| claude-code | doca-dma.implicit-positive.v1 | 253,941 | 95,573 | +158,368 | +165.70% | skill 1/1; base 1/1 |
| claude-code | doca-dma.implicit-positive.v2 | 196,740 | 256,012 | -59,272 | -23.15% | skill 1/1; base 1/1 |
| claude-code | doca-dma.negative.v1 | 30,303 | 124,150 | -93,847 | -75.59% | skill 1/1; base 1/1 |
| codex | All cases | 359,331 | 291,805 | +67,526 | +23.14% | skill 4/4; base 4/4 |
| codex | doca-dma.explicit-positive.v1 | 118,331 | 43,956 | +74,375 | +169.20% | skill 1/1; base 1/1 |
| codex | doca-dma.implicit-positive.v1 | 112,917 | 33,092 | +79,825 | +241.22% | skill 1/1; base 1/1 |
| codex | doca-dma.implicit-positive.v2 | 86,610 | 155,328 | -68,718 | -44.24% | skill 1/1; base 1/1 |
| codex | doca-dma.negative.v1 | 41,473 | 59,429 | -17,956 | -30.21% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,050,357 | 1,081,127 | -30,770 | -2.85% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/libs/doca-dma`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: Sudo   (`BENCHMARK.md:85`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: Sudo   (`TASKS.md:355`)
- **LOW** QUALITY/quality_discoverability: Description very long (998 chars, recommend 50-150) (`skills/libs/doca-dma/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/libs/doca-dma/SKILL.md`)
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
