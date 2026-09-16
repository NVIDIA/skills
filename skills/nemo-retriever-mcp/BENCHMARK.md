# Skill Benchmark: nemo-retriever-mcp

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-retriever-mcp`
- Evaluation date: 2026-09-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:66c211f3143896abe07a31fd20994bea25e2de26ead9f3ede9af351fcbc969e5` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.1% — baseline ran, but no comparable score was available; uplift unavailable | 69.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.8% → 100.0% (+6.2 points) | 30.0% → 90.0% (+60.0 points) |
| Correctness | 17.5% → 60.0% (+42.5 points) | 16.0% → 44.0% (+28.0 points) |
| Discoverability | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 75.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.3% → 50.6% (+11.3 points) | 28.5% → 44.3% (+15.8 points) |
| Efficiency | 86.4% — baseline ran, but no comparable score was available; uplift unavailable | 92.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 970,886 | 1,055,581 | N/A | N/A | skill 4/4; base 8/8 |
| claude-code | nemo-retriever-mcp-001 | 120,998 | 272,278 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemo-retriever-mcp-002 | 395,568 | 597,886 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemo-retriever-mcp-003 | 424,583 | 155,744 | +268,839 | +172.62% | skill 1/1; base 1/1 |
| claude-code | nemo-retriever-mcp-004 | 29,737 | 29,673 | +64 | +0.22% | skill 1/1; base 1/1 |
| codex | All cases | 1,296,562 | 5,600,373 | N/A | N/A | skill 5/5; base 10/10 |
| codex | nemo-retriever-mcp-001 | 169,544 | 551,996 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemo-retriever-mcp-002 | 881,093 | 1,542,171 | N/A | N/A | skill 2/2; base 3/3 |
| codex | nemo-retriever-mcp-003 | 232,325 | 3,492,642 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemo-retriever-mcp-004 | 13,600 | 13,564 | +36 | +0.27% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,267,448 | 6,655,954 | N/A | N/A | skill 9/9; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nemo-retriever-mcp/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-retriever-mcp/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nemo-retriever-mcp/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-retriever-mcp/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-retriever-mcp/SKILL.md`)
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
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
