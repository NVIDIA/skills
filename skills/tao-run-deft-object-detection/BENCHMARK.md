# Skill Benchmark: tao-run-deft-object-detection

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-object-detection`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:d3fce2a83efb9ec178ecc5fc8a20f8c2280adffa3b279916a073015372c25237` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 83.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 100.0% (+16.7 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 5.0% → 95.0% (+90.0 points) | 24.0% → 100.0% (+76.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 42.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 10.4% → 86.0% (+75.6 points) | 17.3% → 80.5% (+63.2 points) |
| Efficiency | 82.6% — baseline ran, but no comparable score was available; uplift unavailable | 94.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 562,944 | 2,793,793 | N/A | N/A | skill 4/4; base 12/12 |
| claude-code | tao-run-deft-object-detection-basic | 107,777 | 531,281 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-object-detection-resume-state | 285,769 | 716,791 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-object-detection-run-loop | 69,514 | 95,207 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-object-detection-stage-skills | 99,884 | 1,450,514 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 365,449 | 417,290 | N/A | N/A | skill 4/4; base 10/10 |
| codex | tao-run-deft-object-detection-basic | 13,944 | 13,786 | +158 | +1.15% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-resume-state | 305,336 | 320,957 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-object-detection-run-loop | 14,410 | 41,751 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-object-detection-stage-skills | 31,759 | 40,796 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 928,393 | 3,211,083 | N/A | N/A | skill 8/8; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 117 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/scripts-and-agents.md:
  "## Stage Reference Modules" in SKILL.md (lines 111-130)
  vs "## Stage Reference Modules" in references/scripts-and-agents.md (lines 106-119) (`SKILL.md:111`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- 113 additional finding(s) are available in the full evaluation artifacts.

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
