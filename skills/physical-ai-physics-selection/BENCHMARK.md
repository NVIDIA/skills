# Skill Benchmark: physical-ai-physics-selection

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `physical-ai-physics-selection`
- Evaluation date: 2026-10-14
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
| Overall | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 94.4% (+2.7 points) | 94.4% → 94.4% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 97.8% → 100.0% (+2.2 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.2% → 95.3% (+18.1 points) | 87.6% → 97.2% (+9.6 points) |
| Efficiency | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 36.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,600,301 | 11,200,487 | N/A | N/A | skill 17/18; base 17/18 |
| claude-code | physical-ai-physics-selection-analysis-only | 103,284 | 92,922 | +10,362 | +11.15% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-cable-shared-stage | 104,450 | 92,245 | +12,205 | +13.23% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-cpu-only | 1,542,854 | 7,201,948 | -5,659,094 | -78.58% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-drake-analysis | 104,443 | 31,291 | +73,152 | +233.78% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-existing-stage | 184,006 | 691,883 | -507,877 | -73.41% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-framework-owned | 179,296 | 92,772 | +86,524 | +93.27% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-genesis-hardware | 549,259 | 842,997 | -293,738 | -34.84% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-gradient-required | 103,707 | 157,674 | -53,967 | -34.23% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-jax-tpu | 104,741 | 31,321 | +73,420 | +234.41% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-new-manipulation-learning | 105,402 | 97,829 | +7,573 | +7.74% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-physx-live-edits | 144,257 | 32,566 | +111,691 | +342.97% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-preserve-bullet | N/A | 336,384 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | physical-ai-physics-selection-preserve-mujoco | 443,651 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | physical-ai-physics-selection-standalone-urdf | 100,614 | 254,050 | -153,436 | -60.40% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-unverified-performance | 62,706 | 30,113 | +32,593 | +108.24% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-usd-is-not-enough | 99,848 | 91,540 | +8,308 | +9.08% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-vsim-robotics | 568,026 | 1,092,776 | -524,750 | -48.02% | skill 1/1; base 1/1 |
| claude-code | physical-ai-physics-selection-windows-host | 99,757 | 30,176 | +69,581 | +230.58% | skill 1/1; base 1/1 |
| codex | All cases | 3,042,611 | 972,920 | N/A | N/A | skill 17/18; base 17/18 |
| codex | physical-ai-physics-selection-analysis-only | 47,909 | 33,343 | +14,566 | +43.69% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-cable-shared-stage | 92,405 | 39,279 | +53,126 | +135.25% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-cpu-only | 1,824,496 | 242,096 | +1,582,400 | +653.63% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-drake-analysis | 72,065 | 37,456 | +34,609 | +92.40% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-existing-stage | 42,738 | 94,531 | -51,793 | -54.79% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-framework-owned | 147,476 | 84,957 | +62,519 | +73.59% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-genesis-hardware | 88,699 | 44,242 | +44,457 | +100.49% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-gradient-required | 51,398 | 43,764 | +7,634 | +17.44% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-jax-tpu | 52,378 | 27,409 | +24,969 | +91.10% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-new-manipulation-learning | 80,299 | 54,657 | +25,642 | +46.91% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-physx-live-edits | 47,672 | 24,828 | +22,844 | +92.01% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-preserve-bullet | 113,235 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | physical-ai-physics-selection-preserve-mujoco | N/A | 82,371 | N/A | N/A | skill 0/1; base 1/1 |
| codex | physical-ai-physics-selection-standalone-urdf | 47,183 | 39,087 | +8,096 | +20.71% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-unverified-performance | 28,842 | 13,611 | +15,231 | +111.90% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-usd-is-not-enough | 40,519 | 18,045 | +22,474 | +124.54% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-vsim-robotics | 207,028 | 70,320 | +136,708 | +194.41% | skill 1/1; base 1/1 |
| codex | physical-ai-physics-selection-windows-host | 58,269 | 22,924 | +35,345 | +154.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,642,912 | 12,173,407 | N/A | N/A | skill 34/36; base 34/36 |

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
