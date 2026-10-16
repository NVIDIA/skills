# Skill Benchmark: tilegym-converting-cutile-to-triton

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tilegym-converting-cutile-to-triton`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (1 positive, 4 negative)
- Dataset digest: `sha256:6eb6dce8f7e1fd54fc084bc411e537b99c4ddcfd0385c53ed32c220626278056` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 98.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 82.5% → 100.0% (+17.5 points) | 90.0% → 99.0% (+9.0 points) |
| Efficiency | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 97.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 188,706 | 211,123 | -22,417 | -10.62% | skill 5/5; base 5/5 |
| claude-code | 01-overview-cutile-to-triton | 67,321 | 89,258 | -21,937 | -24.58% | skill 1/1; base 1/1 |
| claude-code | 02-swiftui-animation-negative | 31,092 | 31,113 | -21 | -0.07% | skill 1/1; base 1/1 |
| claude-code | 03-kafka-rebalance-negative | 31,121 | 31,056 | +65 | +0.21% | skill 1/1; base 1/1 |
| claude-code | 04-css-grid-negative | 29,952 | 29,587 | +365 | +1.23% | skill 1/1; base 1/1 |
| claude-code | 05-mongodb-aggregation-negative | 29,220 | 30,109 | -889 | -2.95% | skill 1/1; base 1/1 |
| codex | All cases | 171,192 | 109,094 | +62,098 | +56.92% | skill 5/5; base 5/5 |
| codex | 01-overview-cutile-to-triton | 101,158 | 39,762 | +61,396 | +154.41% | skill 1/1; base 1/1 |
| codex | 02-swiftui-animation-negative | 19,576 | 22,783 | -3,207 | -14.08% | skill 1/1; base 1/1 |
| codex | 03-kafka-rebalance-negative | 18,745 | 15,022 | +3,723 | +24.78% | skill 1/1; base 1/1 |
| codex | 04-css-grid-negative | 13,665 | 13,553 | +112 | +0.83% | skill 1/1; base 1/1 |
| codex | 05-mongodb-aggregation-negative | 18,048 | 17,974 | +74 | +0.41% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 359,898 | 320,217 | +39,681 | +12.39% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within translations/workflow.md:
  "### TMA Setup (Required Once)" in translations/workflow.md (lines 208-218)
  vs "# TMA allocator (required once per kernel launch context)" in translations/workflow.md (lines 362-368) (`translations/workflow.md:208`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in debugging.md (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/tilegym-converting-cutile-to-triton/SKILL.md (`BENCHMARK.md:71`)
- **MEDIUM** SECURITY/Unknown (AE4): analysis-evasion: Suspicious Unicode normalization or mixed-script content (`references/optimizing-reference.md:1`)
- 20 additional finding(s) are available in the full evaluation artifacts.

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
