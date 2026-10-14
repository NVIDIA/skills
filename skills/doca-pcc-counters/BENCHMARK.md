# Skill Benchmark: doca-pcc-counters

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-pcc-counters`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:d4cf11046a7a83cd5f672ca04e9d52a3d7ec7e75196f5e27915e2f5df99a3fc7` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.9% — baseline ran, but no comparable score was available; uplift unavailable | 98.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 23.3% → 100.0% (+76.7 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.7% → 87.5% (+45.8 points) | 30.0% → 100.0% (+70.0 points) |
| Efficiency | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 96.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 473,061 | 1,177,416 | N/A | N/A | skill 4/4; base 6/6 |
| claude-code | doca-pcc-counters.explicit-positive.v1 | 146,573 | 688,958 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | doca-pcc-counters.implicit-positive.v1 | 146,556 | 238,041 | -91,485 | -38.43% | skill 1/1; base 1/1 |
| claude-code | doca-pcc-counters.implicit-positive.v2 | 146,735 | 123,501 | +23,234 | +18.81% | skill 1/1; base 1/1 |
| claude-code | doca-pcc-counters.negative.v1 | 33,197 | 126,916 | -93,719 | -73.84% | skill 1/1; base 1/1 |
| codex | All cases | 240,377 | 292,014 | N/A | N/A | skill 4/4; base 5/5 |
| codex | doca-pcc-counters.explicit-positive.v1 | 67,973 | 182,494 | N/A | N/A | skill 1/1; base 2/2 |
| codex | doca-pcc-counters.implicit-positive.v1 | 67,860 | 68,461 | -601 | -0.88% | skill 1/1; base 1/1 |
| codex | doca-pcc-counters.implicit-positive.v2 | 68,069 | 27,189 | +40,880 | +150.35% | skill 1/1; base 1/1 |
| codex | doca-pcc-counters.negative.v1 | 36,475 | 13,870 | +22,605 | +162.98% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 713,438 | 1,469,430 | N/A | N/A | skill 8/8; base 11/11 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 30 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/tools/doca-pcc-counters`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/tools/doca-pcc-counters`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`CAPABILITIES.md:60`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`CAPABILITIES.md:61`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`CAPABILITIES.md:144`)
- 26 additional finding(s) are available in the full evaluation artifacts.

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
