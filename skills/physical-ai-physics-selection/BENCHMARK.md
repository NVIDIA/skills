# Skill Benchmark: physical-ai-physics-selection

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `physical-ai-physics-selection`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (16 positive, 2 negative)
- Dataset digest: `sha256:8f634bf9512031e90d3fd187a56f4049871355b21e5f96efbbe09abb60fa9858` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.3% — baseline ran, but no comparable score was available; uplift unavailable | 84.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 88.9% → 94.4% (+5.5 points) | 94.4% → 94.4% (±0.0 points) |
| Correctness | 88.9% → 100.0% (+11.1 points) | 97.8% → 98.9% (+1.1 points) |
| Discoverability | 98.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 78.5% → 98.1% (+19.6 points) | 90.3% → 95.6% (+5.3 points) |
| Efficiency | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 43.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,425,174 | 5,094,765 | N/A | N/A | skill 17/18; base 17/18 |
| claude-code | physical-ai-physics-selection-analysis-only | 100,308 | 93,327 | +6,981 | +7.48% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-cable-shared-stage | 99,077 | 257,155 | -158,078 | -61.47% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-cpu-only | 1,894,103 | 1,885,677 | +8,426 | +0.45% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-drake-analysis | 104,637 | 31,871 | +72,766 | +228.31% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-existing-stage | 224,179 | 184,719 | +39,460 | +21.36% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-framework-owned | 137,647 | 124,838 | +12,809 | +10.26% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-genesis-hardware | 316,402 | 225,922 | +90,480 | +40.05% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-gradient-required | 99,489 | 33,733 | +65,756 | +194.93% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-jax-tpu | 100,687 | 62,432 | +38,255 | +61.27% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-new-manipulation-learning | 100,577 | 35,766 | +64,811 | +181.21% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-physx-live-edits | 142,583 | 32,533 | +110,050 | +338.27% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-preserve-bullet | 356,997 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | physical-ai-physics-selection-preserve-mujoco | N/A | 366,991 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | physical-ai-physics-selection-standalone-urdf | 102,420 | 370,302 | -267,882 | -72.34% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-unverified-performance | 97,036 | 29,618 | +67,418 | +227.63% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-usd-is-not-enough | 99,825 | 119,338 | -19,513 | -16.35% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-vsim-robotics | 349,122 | 1,210,598 | -861,476 | -71.16% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-windows-host | 100,085 | 29,945 | +70,140 | +234.23% | skill 1/1; base 1/1 |
| codex | All cases | 2,402,713 | 1,570,171 | N/A | N/A | skill 17/18; base 17/18 |
| codex | physical-ai-physics-selection-analysis-only | 34,256 | 45,297 | -11,041 | -24.37% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-cable-shared-stage | 75,052 | 44,437 | +30,615 | +68.90% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-cpu-only | 1,269,781 | 721,876 | +547,905 | +75.90% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-drake-analysis | 62,540 | 30,170 | +32,370 | +107.29% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-existing-stage | 47,012 | 103,738 | -56,726 | -54.68% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-framework-owned | 286,702 | 97,222 | +189,480 | +194.89% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-genesis-hardware | 56,033 | 37,629 | +18,404 | +48.91% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-gradient-required | 61,836 | 63,291 | -1,455 | -2.30% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-jax-tpu | 45,192 | 25,818 | +19,374 | +75.04% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-new-manipulation-learning | 112,003 | 80,895 | +31,108 | +38.45% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-physx-live-edits | 40,664 | 25,212 | +15,452 | +61.29% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-preserve-bullet | 103,675 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | physical-ai-physics-selection-preserve-mujoco | N/A | 137,127 | N/A | N/A | skill 0/1; base 1/1 |
| codex | physical-ai-physics-selection-standalone-urdf | 28,787 | 38,218 | -9,431 | -24.68% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-unverified-performance | 28,755 | 13,553 | +15,202 | +112.17% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-usd-is-not-enough | 33,002 | 24,366 | +8,636 | +35.44% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-vsim-robotics | 77,337 | 47,353 | +29,984 | +63.32% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-windows-host | 40,086 | 33,969 | +6,117 | +18.01% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,827,887 | 6,664,936 | N/A | N/A | skill 34/36; base 34/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/physical-ai-physics-selection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/physical-ai-physics-selection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/physical-ai-physics-selection/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in routing-and-runtime.md (`skills/physical-ai-physics-selection/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/physical-ai-physics-selection/SKILL.md`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
