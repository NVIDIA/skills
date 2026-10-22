# Skill Benchmark: using-ovnewton

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `using-ovnewton`
- Evaluation date: 2026-10-19
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (5 positive, 2 negative)
- Dataset digest: `sha256:229164b02575b0e70ff657dae1616a1289f9db36e537a0fbb9465d6694922319` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.2% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 57.1% → 100.0% (+42.9 points) | 100.0% → 85.7% (-14.3 points) |
| Correctness | 71.4% → 94.3% (+22.9 points) | 85.7% → 88.6% (+2.9 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.0% → 87.9% (+32.9 points) | 58.2% → 79.3% (+21.1 points) |
| Efficiency | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 69.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,485,116 | 7,071,032 | -5,585,916 | -79.00% | skill 7/7; base 7/7 |
| claude-code | using-ovnewton-basic | 215,800 | 2,920,898 | -2,705,098 | -92.61% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-existing-stage | 63,726 | 160,755 | -97,029 | -60.36% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-newton-only | 986,312 | 1,140,901 | -154,589 | -13.55% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-omniverse-authoring | 31,404 | 35,063 | -3,659 | -10.44% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-rendering | 62,969 | 919,649 | -856,680 | -93.15% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-stage-edits | 62,110 | 139,415 | -77,305 | -55.45% | skill 1/1; base 1/1 |
| claude-code | using-ovnewton-vbd | 62,795 | 1,754,351 | -1,691,556 | -96.42% | skill 1/1; base 1/1 |
| codex | All cases | 800,187 | 1,376,483 | -576,296 | -41.87% | skill 7/7; base 7/7 |
| codex | using-ovnewton-basic | 169,217 | 655,583 | -486,366 | -74.19% | skill 1/1; base 1/1 |
| codex | using-ovnewton-existing-stage | 113,244 | 92,087 | +21,157 | +22.98% | skill 1/1; base 1/1 |
| codex | using-ovnewton-newton-only | 372,250 | 320,070 | +52,180 | +16.30% | skill 1/1; base 1/1 |
| codex | using-ovnewton-omniverse-authoring | 19,430 | 19,614 | -184 | -0.94% | skill 1/1; base 1/1 |
| codex | using-ovnewton-rendering | 41,084 | 58,568 | -17,484 | -29.85% | skill 1/1; base 1/1 |
| codex | using-ovnewton-stage-edits | 40,854 | 84,070 | -43,216 | -51.40% | skill 1/1; base 1/1 |
| codex | using-ovnewton-vbd | 44,108 | 146,491 | -102,383 | -69.89% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,285,303 | 8,447,515 | -6,162,212 | -72.95% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/using-ovnewton/SKILL.md`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`SKILL.md:69`)
- **LOW** QUALITY/quality_discoverability: Description very long (243 chars, recommend 50-150) (`skills/using-ovnewton/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
