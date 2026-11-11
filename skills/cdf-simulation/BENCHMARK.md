# Skill Benchmark: cdf-simulation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-simulation`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (8 positive)
- Dataset digest: `sha256:2e5e11881e77f0623a60210ec155e7d091f33ef15aa5dd0ec3a558c1188ab83c` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 12.5% → 92.5% (+80.0 points) | 0.0% → 55.0% (+55.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 28.2% → 61.9% (+33.7 points) | 17.7% → 50.0% (+32.3 points) |
| Efficiency | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,052,388 | 1,920,460 | +1,131,928 | +58.94% | skill 8/8; base 8/8 |
| claude-code | 1 | 619,240 | 365,300 | +253,940 | +69.52% | skill 1/1; base 1/1 |
| claude-code | 2 | 551,000 | 332,051 | +218,949 | +65.94% | skill 1/1; base 1/1 |
| claude-code | 3 | 555,637 | 151,973 | +403,664 | +265.62% | skill 1/1; base 1/1 |
| claude-code | 4 | 73,199 | 155,996 | -82,797 | -53.08% | skill 1/1; base 1/1 |
| claude-code | 5 | 72,211 | 387,582 | -315,371 | -81.37% | skill 1/1; base 1/1 |
| claude-code | 6 | 343,252 | 122,141 | +221,111 | +181.03% | skill 1/1; base 1/1 |
| claude-code | 7 | 73,748 | 219,393 | -145,645 | -66.39% | skill 1/1; base 1/1 |
| claude-code | 8 | 764,101 | 186,024 | +578,077 | +310.75% | skill 1/1; base 1/1 |
| codex | All cases | 1,725,170 | 1,241,620 | +483,550 | +38.95% | skill 8/8; base 8/8 |
| codex | 1 | 219,491 | 102,251 | +117,240 | +114.66% | skill 1/1; base 1/1 |
| codex | 2 | 210,877 | 114,885 | +95,992 | +83.55% | skill 1/1; base 1/1 |
| codex | 3 | 468,757 | 112,124 | +356,633 | +318.07% | skill 1/1; base 1/1 |
| codex | 4 | 92,944 | 80,484 | +12,460 | +15.48% | skill 1/1; base 1/1 |
| codex | 5 | 150,137 | 54,924 | +95,213 | +173.35% | skill 1/1; base 1/1 |
| codex | 6 | 230,828 | 391,586 | -160,758 | -41.05% | skill 1/1; base 1/1 |
| codex | 7 | 123,774 | 96,685 | +27,089 | +28.02% | skill 1/1; base 1/1 |
| codex | 8 | 228,362 | 288,681 | -60,319 | -20.89% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,777,558 | 3,162,080 | +1,615,478 | +51.09% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (9434 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-simulation/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 583 lines (limit: 500) (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (575 chars, recommend 50-150) (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cdf-simulation/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
