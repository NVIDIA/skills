# Skill Benchmark: nemo-fabric-integrate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-integrate`
- Evaluation date: 2026-10-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:4e12e11bb429ad195c00ceab9cd53bbf6351f32173f03b0ae827531ba2441a9e` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 70.8% — baseline ran, but no comparable score was available; uplift unavailable | 83.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 58.3% (-41.7 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 30.0% → 86.7% (+56.7 points) | 33.3% → 93.3% (+60.0 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 27.9% → 67.1% (+39.2 points) | 30.6% → 71.6% (+41.0 points) |
| Efficiency | 62.1% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 9,966,672 | 6,465,088 | +3,501,584 | +54.16% | skill 6/6; base 6/6 |
| claude-code | nemo-fabric-integrate-001-python-service | 2,936,794 | 278,491 | +2,658,303 | +954.54% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-002-evaluation-platform | 3,248,168 | 248,458 | +2,999,710 | +1207.33% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-003-adapter-contribution-negative | 767,353 | 184,815 | +582,538 | +315.20% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-004-overview-negative | 68,694 | 60,320 | +8,374 | +13.88% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-005-streaming-boundary | 1,986,509 | 280,668 | +1,705,841 | +607.78% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-006-system-instruction-modes | 959,154 | 5,412,336 | -4,453,182 | -82.28% | skill 1/1; base 1/1 |
| codex | All cases | 5,858,731 | 4,264,627 | +1,594,104 | +37.38% | skill 6/6; base 6/6 |
| codex | nemo-fabric-integrate-001-python-service | 116,700 | 112,759 | +3,941 | +3.50% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-002-evaluation-platform | 739,994 | 98,305 | +641,689 | +652.75% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-003-adapter-contribution-negative | 4,366,445 | 3,831,200 | +535,245 | +13.97% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-004-overview-negative | 29,815 | 25,087 | +4,728 | +18.85% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-005-streaming-boundary | 500,887 | 97,944 | +402,943 | +411.40% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-006-system-instruction-modes | 104,890 | 99,332 | +5,558 | +5.60% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 15,825,403 | 10,729,715 | +5,095,688 | +47.49% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (5945 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-fabric-integrate/SKILL.md`)
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
