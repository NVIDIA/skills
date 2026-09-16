# Skill Benchmark: tao-run-deft-cr-its-mining

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-cr-its-mining`
- Evaluation date: 2026-09-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:e2415bd85ab17e2af8a9afff1301e63b0145eceba60a95906283869558f70186` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 59.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 1.7% → 100.0% (+98.3 points) | 40.0% → 53.3% (+13.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 7.3% → 93.1% (+85.8 points) | 42.7% → 46.0% (+3.3 points) |
| Efficiency | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 97.9% → 98.9% (+1.0 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,073,272 | 1,843,618 | N/A | N/A | skill 4/4; base 12/12 |
| claude-code | tao-run-deft-cr-its-mining-basic | 71,928 | 90,508 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-cr-its-mining-multi-question | 337,992 | 549,881 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-cr-its-mining-plugin-paths | 411,625 | 581,372 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-cr-its-mining-resume | 251,727 | 621,857 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 85,146 | 86,989 | N/A | N/A | skill 6/6; base 5/5 |
| codex | tao-run-deft-cr-its-mining-basic | 14,086 | 13,560 | +526 | +3.88% | skill 1/1; base 1/1 |
| codex | tao-run-deft-cr-its-mining-multi-question | 14,072 | 13,911 | +161 | +1.16% | skill 1/1; base 1/1 |
| codex | tao-run-deft-cr-its-mining-plugin-paths | 14,665 | 45,618 | N/A | N/A | skill 1/1; base 2/2 |
| codex | tao-run-deft-cr-its-mining-resume | 42,323 | 13,900 | N/A | N/A | skill 3/3; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,158,418 | 1,930,607 | N/A | N/A | skill 10/10; base 17/17 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 47 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/restore_docker_mount_permissions.py and scripts/workflow_common.py:
  "absolute_path()" in scripts/restore_docker_mount_permissions.py (lines 17-19)
  vs "absolute_path()" in scripts/workflow_common.py (lines 35-37) (`scripts/restore_docker_mount_permissions.py:17`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/mining-loop.md:
  "## Completion Criteria" in SKILL.md (lines 173-192)
  vs "## Completion Criteria" in references/mining-loop.md (lines 311-329) (`SKILL.md:173`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/prepare_gap_analysis_predictions.py and scripts/workflow_common.py:
  "load_json_array()" in scripts/prepare_gap_analysis_predictions.py (lines 16-25)
  vs "load_json_array()" in scripts/workflow_common.py (lines 48-57)
  vs "read_jsonl()" in scripts/workflow_common.py (lines 101-116) (`scripts/prepare_gap_analysis_predictions.py:16`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-cr-its-mining/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/applications/tao-run-deft-cr-its-mining/SKILL.md`)
- 45 additional finding(s) are available in the full evaluation artifacts.

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
