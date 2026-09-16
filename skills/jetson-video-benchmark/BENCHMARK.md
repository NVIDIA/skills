# Skill Benchmark: jetson-video-benchmark

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `jetson-video-benchmark`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:d1c525d9b65879a764e8676b2177c71573dee5cbfbceb816d55ca67074b7086b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 82.1% → 100.0% (+17.9 points) |
| Correctness | Not available | 52.9% → 100.0% (+47.1 points) |
| Discoverability | Not available | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 34.8% → 93.0% (+58.2 points) |
| Efficiency | Not available | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 811,096 | 1,310,555 | N/A | N/A | skill 7/7; base 10/21 |
| claude-code | benchmark-4k-planning-estimate | 141,466 | 72,101 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | benchmark-camera-direction-medium-quality | 150,509 | 65,461 | +85,048 | +129.92% | skill 1/1; base 1/1 |
| claude-code | benchmark-documented-estimate | 186,708 | 359,806 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | benchmark-natural-camera-capacity-planning | 109,049 | 123,731 | -14,682 | -11.87% | skill 1/1; base 1/1 |
| claude-code | benchmark-no-content | 63,049 | 599,533 | N/A | N/A | skill 1/1; base 1/3 |
| claude-code | benchmark-quality-boundary | 60,960 | 59,664 | +1,296 | +2.17% | skill 1/1; base 1/1 |
| claude-code | benchmark-undocumented-p4 | 99,355 | 30,259 | +69,096 | +228.35% | skill 1/1; base 1/1 |
| codex | All cases | 543,961 | 1,572,817 | N/A | N/A | skill 7/7; base 14/14 |
| codex | benchmark-4k-planning-estimate | 65,929 | 45,385 | +20,544 | +45.27% | skill 1/1; base 1/1 |
| codex | benchmark-camera-direction-medium-quality | 127,179 | 134,210 | N/A | N/A | skill 1/1; base 3/3 |
| codex | benchmark-documented-estimate | 65,755 | 159,060 | N/A | N/A | skill 1/1; base 2/2 |
| codex | benchmark-natural-camera-capacity-planning | 172,004 | 138,724 | N/A | N/A | skill 1/1; base 3/3 |
| codex | benchmark-no-content | 28,517 | 1,032,822 | N/A | N/A | skill 1/1; base 3/3 |
| codex | benchmark-quality-boundary | 28,338 | 33,628 | -5,290 | -15.73% | skill 1/1; base 1/1 |
| codex | benchmark-undocumented-p4 | 56,239 | 28,988 | +27,251 | +94.01% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,355,057 | 2,883,372 | N/A | N/A | skill 14/14; base 24/35 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in benchmark-output-contract.md (`skills/jetson-video-benchmark/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/jetson-video-benchmark/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/jetson-video-benchmark/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/jetson-video-benchmark/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (389 chars, recommend 50-150) (`skills/jetson-video-benchmark/SKILL.md`)
- 4 additional finding(s) are available in the full evaluation artifacts.

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
