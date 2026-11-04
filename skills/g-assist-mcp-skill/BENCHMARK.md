# Skill Benchmark: g-assist-mcp-skill

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `g-assist-mcp-skill`
- Evaluation date: 2026-09-30
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 16 evaluation tasks (11 positive, 5 negative)
- Dataset digest: `sha256:005ae43f3adb9fe981b27b5096c41379327675753c3c913e0df8734f6d3159f2` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 69.5% → 97.5% (+28.0 points) | 60.0% → 100.0% (+40.0 points) |
| Discoverability | 85.5% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 61.7% → 84.5% (+22.8 points) | 51.9% → 90.0% (+38.1 points) |
| Efficiency | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,528,370 | 1,874,259 | N/A | N/A | skill 16/16; base 21/21 |
| claude-code | g-assist-mcp-skill-001 | 101,334 | 328,623 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | g-assist-mcp-skill-002 | 65,844 | 241,302 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | g-assist-mcp-skill-003 | 106,536 | 121,924 | -15,388 | -12.62% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-004 | 60,840 | 30,211 | +30,629 | +101.38% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-005 | 29,542 | 29,379 | +163 | +0.55% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-006 | 66,787 | 245,203 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | g-assist-mcp-skill-007 | 244,171 | 60,168 | +184,003 | +305.82% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-008 | 67,581 | 61,844 | +5,737 | +9.28% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-009 | 101,917 | 60,829 | +41,088 | +67.55% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-010 | 66,514 | 151,041 | -84,527 | -55.96% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-011 | 65,545 | 59,938 | +5,607 | +9.35% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-neg-001 | 29,034 | 28,955 | +79 | +0.27% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-neg-002 | 182,458 | 121,391 | +61,067 | +50.31% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-neg-003 | 183,617 | 183,777 | -160 | -0.09% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-neg-004 | 91,026 | 88,951 | +2,075 | +2.33% | skill 1/1; base 1/1 |
| claude-code | g-assist-mcp-skill-neg-005 | 65,624 | 60,723 | +4,901 | +8.07% | skill 1/1; base 1/1 |
| codex | All cases | 854,089 | 1,237,266 | N/A | N/A | skill 16/16; base 21/21 |
| codex | g-assist-mcp-skill-001 | 63,169 | 157,026 | N/A | N/A | skill 1/1; base 3/3 |
| codex | g-assist-mcp-skill-002 | 64,251 | 42,220 | +22,031 | +52.18% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-003 | 63,451 | 41,068 | +22,383 | +54.50% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-004 | 30,180 | 13,441 | +16,739 | +124.54% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-005 | 30,016 | 13,501 | +16,515 | +122.32% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-006 | 47,016 | 183,210 | N/A | N/A | skill 1/1; base 2/2 |
| codex | g-assist-mcp-skill-007 | 46,482 | 84,383 | -37,901 | -44.92% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-008 | 65,038 | 41,952 | +23,086 | +55.03% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-009 | 63,166 | 41,650 | +21,516 | +51.66% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-010 | 30,473 | 237,826 | N/A | N/A | skill 1/1; base 3/3 |
| codex | g-assist-mcp-skill-011 | 63,268 | 40,742 | +22,526 | +55.29% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-neg-001 | 13,265 | 13,193 | +72 | +0.55% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-neg-002 | 100,808 | 111,836 | -11,028 | -9.86% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-neg-003 | 85,308 | 118,907 | -33,599 | -28.26% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-neg-004 | 41,357 | 40,890 | +467 | +1.14% | skill 1/1; base 1/1 |
| codex | g-assist-mcp-skill-neg-005 | 46,841 | 55,421 | -8,580 | -15.48% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,382,459 | 3,111,525 | N/A | N/A | skill 32/32; base 42/42 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 1 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 16 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: no
   approval (`SKILL.md:119`)

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
