# Skill Benchmark: nemo-fabric-build-adapter

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-build-adapter`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (5 positive, 2 negative)
- Dataset digest: `sha256:b0644fe48c189bd6fcbf458a022b58e155d5b001adaf4b153c9ea36fd4cae5f7` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 78.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 57.1% → 85.7% (+28.6 points) |
| Correctness | Not available | 71.4% → 82.9% (+11.5 points) |
| Discoverability | Not available | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 52.1% → 70.2% (+18.1 points) |
| Efficiency | Not available | 70.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 39,627,147 | 4,796,580 | +34,830,567 | +726.15% | skill 7/7; base 7/7 |
| claude-code | nemo-fabric-build-adapter-001-new-python-adapter | 5,168,425 | 1,954,587 | +3,213,838 | +164.43% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-002-native-openai-streaming | 68,179 | 299,848 | -231,669 | -77.26% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-003-consumer-integration-negative | 5,350,629 | 247,023 | +5,103,606 | +2066.04% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 16,317,781 | 185,832 | +16,131,949 | +8680.93% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-005-system-instruction-composition | 12,348,832 | 1,166,696 | +11,182,136 | +958.44% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-006-relay-request-correlation | 231,605 | 846,176 | -614,571 | -72.63% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-007-warm-session-continuation | 141,696 | 96,418 | +45,278 | +46.96% | skill 1/1; base 1/1 |
| codex | All cases | 5,166,302 | 12,163,513 | -6,997,211 | -57.53% | skill 7/7; base 7/7 |
| codex | nemo-fabric-build-adapter-001-new-python-adapter | 1,312,179 | 571,346 | +740,833 | +129.66% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-002-native-openai-streaming | 75,895 | 112,232 | -36,337 | -32.38% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-003-consumer-integration-negative | 115,342 | 155,665 | -40,323 | -25.90% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 3,149,737 | 7,639,888 | -4,490,151 | -58.77% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-005-system-instruction-composition | 320,386 | 3,501,113 | -3,180,727 | -90.85% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-006-relay-request-correlation | 99,951 | 99,565 | +386 | +0.39% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-007-warm-session-continuation | 92,812 | 83,704 | +9,108 | +10.88% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 44,793,449 | 16,960,093 | +27,833,356 | +164.11% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-fabric-build-adapter/SKILL.md`)
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
