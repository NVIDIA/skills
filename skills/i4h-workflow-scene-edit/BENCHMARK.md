# Skill Benchmark: i4h-workflow-scene-edit

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `i4h-workflow-scene-edit`
- Evaluation date: 2026-10-01
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (12 positive)
- Dataset digest: `sha256:2eb07663502630273976ed2758a993f26bbb4e8c8f3da3251a802b98aa12329c` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 62.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 33.3% → 45.8% (+12.5 points) |
| Correctness | Not available | 38.3% → 76.7% (+38.4 points) |
| Discoverability | Not available | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 15.4% → 39.1% (+23.7 points) |
| Efficiency | Not available | 75.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 12,267,301 | 11,916,188 | N/A | N/A | skill 10/12; base 10/12 |
| claude-code | i4h-workflow-scene-edit-compound-live-cadence | 452,931 | 221,284 | +231,647 | +104.68% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-deterministic-bake | 396,175 | 155,704 | +240,471 | +154.44% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-g1-head-camera-regression | 826,911 | 288,534 | +538,377 | +186.59% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-live-session-no-bake | 484,026 | 187,854 | +296,172 | +157.66% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-readme-add-modes | N/A | 1,674,874 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-readme-add-robotic-policy-mode | 2,681,268 | N/A | N/A | N/A | skill 1/1; base 0/0 |
| claude-code | i4h-workflow-scene-edit-readme-define-behavior | N/A | N/A | N/A | N/A | skill 0/1; base 0/0 |
| claude-code | i4h-workflow-scene-edit-readme-define-goal | 1,307,277 | 2,552,340 | -1,245,063 | -48.78% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-readme-g1-surgical-layout | 1,035,329 | 470,114 | +565,215 | +120.23% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-readme-robotic-instrument-reach | 1,273,330 | 778,268 | +495,062 | +63.61% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-reuse-known-scene-scale | 409,796 | 4,849,820 | -4,440,024 | -91.55% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-scene-edit-table-red-cube | 3,400,258 | 737,396 | +2,662,862 | +361.12% | skill 1/1; base 1/1 |
| codex | All cases | 23,431,914 | 6,891,601 | N/A | N/A | skill 10/12; base 10/12 |
| codex | i4h-workflow-scene-edit-compound-live-cadence | 1,582,864 | 570,548 | +1,012,316 | +177.43% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-deterministic-bake | 456,832 | 593,856 | -137,024 | -23.07% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-g1-head-camera-regression | 1,604,360 | 630,519 | +973,841 | +154.45% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-live-session-no-bake | 354,214 | 86,515 | +267,699 | +309.42% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-readme-add-modes | 5,106,088 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | i4h-workflow-scene-edit-readme-add-robotic-policy-mode | N/A | 410,779 | N/A | N/A | skill 0/1; base 1/1 |
| codex | i4h-workflow-scene-edit-readme-define-behavior | 3,755,528 | 2,085,695 | +1,669,833 | +80.06% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-readme-define-goal | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-scene-edit-readme-g1-surgical-layout | 3,433,351 | 1,274,077 | +2,159,274 | +169.48% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-readme-robotic-instrument-reach | 3,527,894 | 838,869 | +2,689,025 | +320.55% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-reuse-known-scene-scale | 2,695,989 | 141,676 | +2,554,313 | +1802.93% | skill 1/1; base 1/1 |
| codex | i4h-workflow-scene-edit-table-red-cube | 914,794 | 259,067 | +655,727 | +253.11% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 35,699,215 | 18,807,789 | N/A | N/A | skill 20/24; base 20/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** PII/gps_coordinates: GPS coordinates (location information) (`references/existing-scene-assets.md:22`)
- **MEDIUM** PII/gps_coordinates: GPS coordinates (location information) (`references/existing-scene-assets.md:23`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-scene-edit/SKILL.md`)

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
