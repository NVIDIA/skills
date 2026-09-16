# Skill Benchmark: jetson-video-capability

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `jetson-video-capability`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (5 positive)
- Dataset digest: `sha256:4954ec1cc526996669be4072c5d1a20bbbed003c6d01b75c541838c4a1161eb5` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 96.0% (+6.0 points) | 86.7% → 96.0% (+9.3 points) |
| Discoverability | 96.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.3% → 63.0% (+7.7 points) | 69.4% → 75.3% (+5.9 points) |
| Efficiency | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 78.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 564,852 | 427,829 | N/A | N/A | skill 5/5; base 6/6 |
| claude-code | capability-av1 | 149,204 | 91,965 | +57,239 | +62.24% | skill 1/1; base 1/1 |
| claude-code | capability-codec-not-buffer-sharing | 62,366 | 31,795 | +30,571 | +96.15% | skill 1/1; base 1/1 |
| claude-code | capability-decoders | 145,329 | 153,869 | -8,540 | -5.55% | skill 1/1; base 1/1 |
| claude-code | capability-drm-boundary | 61,468 | 60,589 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | capability-encoders | 146,485 | 89,611 | +56,874 | +63.47% | skill 1/1; base 1/1 |
| codex | All cases | 292,165 | 227,303 | N/A | N/A | skill 5/5; base 6/6 |
| codex | capability-av1 | 40,021 | 48,594 | -8,573 | -17.64% | skill 1/1; base 1/1 |
| codex | capability-codec-not-buffer-sharing | 53,247 | 31,946 | +21,301 | +66.68% | skill 1/1; base 1/1 |
| codex | capability-decoders | 123,782 | 67,516 | N/A | N/A | skill 1/1; base 2/2 |
| codex | capability-drm-boundary | 28,711 | 18,017 | +10,694 | +59.36% | skill 1/1; base 1/1 |
| codex | capability-encoders | 46,404 | 61,230 | -14,826 | -24.21% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 857,017 | 655,132 | N/A | N/A | skill 10/10; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in capability-queries.md (`skills/jetson-video-capability/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/jetson-video-capability/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/jetson-video-capability/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without asking (`references/surface-selection-contract.md:28`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/jetson-video-capability/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
