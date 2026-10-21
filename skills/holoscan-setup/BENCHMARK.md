# Skill Benchmark: holoscan-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-setup`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:eb042e5df52f57f4abef43703a6a63b1a537a38068aadd09374d605aebbdf208` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 96.0% → 100.0% (+4.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 82.0% → 82.0% (±0.0 points) | 68.0% → 83.0% (+15.0 points) |
| Efficiency | 98.7% — baseline ran, but no comparable score was available; uplift unavailable | 98.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 497,697 | 1,006,939 | -509,242 | -50.57% | skill 5/5; base 5/5 |
| claude-code | holoscan-setup-5x-001 | 62,658 | 188,559 | -125,901 | -66.77% | skill 1/1; base 1/1 |
| claude-code | holoscan-setup-5x-002 | 125,757 | 184,072 | -58,315 | -31.68% | skill 1/1; base 1/1 |
| claude-code | holoscan-setup-5x-003 | 60,055 | 327,067 | -267,012 | -81.64% | skill 1/1; base 1/1 |
| claude-code | holoscan-setup-5x-004 | 62,380 | 123,964 | -61,584 | -49.68% | skill 1/1; base 1/1 |
| claude-code | holoscan-setup-5x-005 | 186,847 | 183,277 | +3,570 | +1.95% | skill 1/1; base 1/1 |
| codex | All cases | 313,462 | 302,067 | +11,395 | +3.77% | skill 5/5; base 5/5 |
| codex | holoscan-setup-5x-001 | 28,676 | 14,283 | +14,393 | +100.77% | skill 1/1; base 1/1 |
| codex | holoscan-setup-5x-002 | 28,831 | 14,053 | +14,778 | +105.16% | skill 1/1; base 1/1 |
| codex | holoscan-setup-5x-003 | 28,631 | 13,933 | +14,698 | +105.49% | skill 1/1; base 1/1 |
| codex | holoscan-setup-5x-004 | 28,571 | 14,195 | +14,376 | +101.28% | skill 1/1; base 1/1 |
| codex | holoscan-setup-5x-005 | 198,753 | 245,603 | -46,850 | -19.08% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 811,159 | 1,309,006 | -497,847 | -38.03% | skill 10/10; base 10/10 |

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

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`team-skills/holoscan/holoscan-sdk/holoscan-setup/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`team-skills/holoscan/holoscan-sdk/holoscan-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`team-skills/holoscan/holoscan-sdk/holoscan-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`team-skills/holoscan/holoscan-sdk/holoscan-setup/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description doesn't mention WHEN to use this skill (`team-skills/holoscan/holoscan-sdk/holoscan-setup/SKILL.md`)
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
