# Skill Benchmark: holoscan-install-source

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-install-source`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:8a153da53159948d77200bc314d7c3d7e7dba6b6df680fcd0eb31f5109fdb051` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 97.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 100.0% (+33.3 points) | 83.3% → 100.0% (+16.7 points) |
| Correctness | 83.3% → 100.0% (+16.7 points) | 100.0% → 96.7% (-3.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 65.8% → 90.0% (+24.2 points) | 78.3% → 95.0% (+16.7 points) |
| Efficiency | 98.5% — baseline ran, but no comparable score was available; uplift unavailable | 98.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 477,734 | 1,539,621 | -1,061,887 | -68.97% | skill 6/6; base 6/6 |
| claude-code | holoscan-install-source-5x-001 | 62,817 | 426,384 | -363,567 | -85.27% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-source-5x-002 | 62,766 | 184,777 | -122,011 | -66.03% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-source-5x-003 | 62,846 | 186,712 | -123,866 | -66.34% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-source-5x-004 | 162,349 | 431,392 | -269,043 | -62.37% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-source-5x-005 | 63,402 | 154,090 | -90,688 | -58.85% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-source-5x-006 | 63,554 | 156,266 | -92,712 | -59.33% | skill 1/1; base 1/1 |
| codex | All cases | 238,954 | 383,825 | -144,871 | -37.74% | skill 6/6; base 6/6 |
| codex | holoscan-install-source-5x-001 | 29,589 | 13,979 | +15,610 | +111.67% | skill 1/1; base 1/1 |
| codex | holoscan-install-source-5x-002 | 28,973 | 13,816 | +15,157 | +109.71% | skill 1/1; base 1/1 |
| codex | holoscan-install-source-5x-003 | 29,432 | 14,251 | +15,181 | +106.53% | skill 1/1; base 1/1 |
| codex | holoscan-install-source-5x-004 | 92,332 | 313,651 | -221,319 | -70.56% | skill 1/1; base 1/1 |
| codex | holoscan-install-source-5x-005 | 29,033 | 14,145 | +14,888 | +105.25% | skill 1/1; base 1/1 |
| codex | holoscan-install-source-5x-006 | 29,595 | 13,983 | +15,612 | +111.65% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 716,688 | 1,923,446 | -1,206,758 | -62.74% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`team-skills/holoscan/holoscan-sdk/holoscan-install-source/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`team-skills/holoscan/holoscan-sdk/holoscan-install-source/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`team-skills/holoscan/holoscan-sdk/holoscan-install-source/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`team-skills/holoscan/holoscan-sdk/holoscan-install-source/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`team-skills/holoscan/holoscan-sdk/holoscan-install-source/SKILL.md`)
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
