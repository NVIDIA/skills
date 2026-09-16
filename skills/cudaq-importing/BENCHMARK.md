# Skill Benchmark: cudaq-importing

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudaq-importing`
- Evaluation date: 2026-09-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:85489ea0e566f515affb2b77d430f181d4180dffc638531af9c99f85f899a055` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.9% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 93.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 79.2% → 89.8% (+10.6 points) | 76.7% → 93.1% (+16.4 points) |
| Efficiency | 76.3% — baseline ran, but no comparable score was available; uplift unavailable | 86.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,549,818 | 486,841 | +1,062,977 | +218.34% | skill 4/4; base 4/4 |
| claude-code | cudaq-importing-001 | 676,591 | 159,112 | +517,479 | +325.23% | skill 1/1; base 1/1 |
| claude-code | cudaq-importing-002 | 183,676 | 69,436 | +114,240 | +164.53% | skill 1/1; base 1/1 |
| claude-code | cudaq-importing-003 | 655,970 | 224,089 | +431,881 | +192.73% | skill 1/1; base 1/1 |
| claude-code | cudaq-importing-004 | 33,581 | 34,204 | -623 | -1.82% | skill 1/1; base 1/1 |
| codex | All cases | 308,702 | 117,968 | +190,734 | +161.68% | skill 4/4; base 4/4 |
| codex | cudaq-importing-001 | 79,475 | 14,241 | +65,234 | +458.07% | skill 1/1; base 1/1 |
| codex | cudaq-importing-002 | 105,600 | 33,611 | +71,989 | +214.18% | skill 1/1; base 1/1 |
| codex | cudaq-importing-003 | 80,533 | 33,761 | +46,772 | +138.54% | skill 1/1; base 1/1 |
| codex | cudaq-importing-004 | 43,094 | 36,355 | +6,739 | +18.54% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,858,520 | 604,809 | +1,253,711 | +207.29% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'author' is ignored; use 'metadata.author' (`skills/cudaq-importing/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/cudaq-importing/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cudaq-importing/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cudaq-importing/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudaq-importing/SKILL.md`)
- 2 additional finding(s) are available in the full evaluation artifacts.

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
