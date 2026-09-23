# Skill Benchmark: nvmolkit-usage

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvmolkit-usage`
- Evaluation date: 2026-09-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (12 positive)
- Dataset digest: `sha256:ce8098e0dd2fc0698933b7d4d303fe13bdd1d9be7e87d142bdfc44baae7a9f90` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 92.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 91.7% (+25.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 95.0% → 100.0% (+5.0 points) | 96.7% → 96.7% (±0.0 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 91.4% → 92.6% (+1.2 points) | 82.9% → 85.4% (+2.5 points) |
| Efficiency | 79.5% — baseline ran, but no comparable score was available; uplift unavailable | 84.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,493,519 | 7,705,207 | -1,211,688 | -15.73% | skill 12/12; base 12/12 |
| claude-code | nvmolkit-usage-001 | 155,960 | 499,586 | -343,626 | -68.78% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-002 | 1,444,045 | 251,086 | +1,192,959 | +475.12% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-003 | 360,067 | 1,104,712 | -744,645 | -67.41% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-004 | 397,086 | 997,159 | -600,073 | -60.18% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-005 | 1,032,516 | 813,303 | +219,213 | +26.95% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-006 | 201,515 | 899,766 | -698,251 | -77.60% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-007 | 354,878 | 826,660 | -471,782 | -57.07% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-008 | 1,120,265 | 332,364 | +787,901 | +237.06% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-009 | 68,769 | 328,077 | -259,308 | -79.04% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-010 | 587,346 | 1,087,974 | -500,628 | -46.01% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-011 | 69,433 | 62,607 | +6,826 | +10.90% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-012 | 701,639 | 501,913 | +199,726 | +39.79% | skill 1/1; base 1/1 |
| codex | All cases | 943,466 | 734,327 | +209,139 | +28.48% | skill 12/12; base 12/12 |
| codex | nvmolkit-usage-001 | 42,367 | 42,161 | +206 | +0.49% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-002 | 41,746 | 56,617 | -14,871 | -26.27% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-003 | 74,002 | 77,309 | -3,307 | -4.28% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-004 | 93,170 | 35,798 | +57,372 | +160.27% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-005 | 132,697 | 34,827 | +97,870 | +281.02% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-006 | 94,449 | 28,675 | +65,774 | +229.38% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-007 | 71,719 | 64,534 | +7,185 | +11.13% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-008 | 142,524 | 89,073 | +53,451 | +60.01% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-009 | 30,802 | 20,599 | +10,203 | +49.53% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-010 | 118,378 | 29,444 | +88,934 | +302.04% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-011 | 30,667 | 18,603 | +12,064 | +64.85% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-012 | 70,945 | 236,687 | -165,742 | -70.03% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,436,985 | 8,439,534 | -1,002,549 | -11.88% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 1 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/nvmolkit-usage/SKILL.md`)

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
