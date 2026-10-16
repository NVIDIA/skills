# Skill Benchmark: tao-run-deft-cr-its-mining

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-cr-its-mining`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:e2415bd85ab17e2af8a9afff1301e63b0145eceba60a95906283869558f70186` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 61.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 15.0% → 60.0% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 7.3% → 96.3% (+89.0 points) | 22.1% → 47.7% (+25.6 points) |
| Efficiency | 78.0% — baseline ran, but no comparable score was available; uplift unavailable | 99.6% → 98.9% (-0.7 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,455,519 | 895,270 | +560,249 | +62.58% | skill 4/4; base 4/4 |
| claude-code | tao-run-deft-cr-its-mining-basic | 72,574 | 30,141 | +42,433 | +140.78% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-cr-its-mining-multi-question | 513,393 | 183,958 | +329,435 | +179.08% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-cr-its-mining-plugin-paths | 657,055 | 372,511 | +284,544 | +76.39% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-cr-its-mining-resume | 212,497 | 308,660 | -96,163 | -31.15% | skill 1/1; base 1/1 |
| codex | All cases | 56,439 | 54,935 | +1,504 | +2.74% | skill 4/4; base 4/4 |
| codex | tao-run-deft-cr-its-mining-basic | 13,893 | 13,466 | +427 | +3.17% | skill 1/1; base 1/1 |
| codex | tao-run-deft-cr-its-mining-multi-question | 14,094 | 13,571 | +523 | +3.85% | skill 1/1; base 1/1 |
| codex | tao-run-deft-cr-its-mining-plugin-paths | 14,231 | 13,891 | +340 | +2.45% | skill 1/1; base 1/1 |
| codex | tao-run-deft-cr-its-mining-resume | 14,221 | 14,007 | +214 | +1.53% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,511,958 | 950,205 | +561,753 | +59.12% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 75 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/restore_docker_mount_permissions.py and scripts/workflow_common.py:
  "absolute_path()" in scripts/restore_docker_mount_permissions.py (lines 17-19)
  vs "absolute_path()" in scripts/workflow_common.py (lines 35-37) (`scripts/restore_docker_mount_permissions.py:17`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/mining-loop.md:
  "## Completion Criteria" in SKILL.md (lines 177-197)
  vs "## Completion Criteria" in references/mining-loop.md (lines 403-422) (`SKILL.md:177`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/prepare_gap_analysis_predictions.py and scripts/validate_cosmos_embed_output.py and scripts/workflow_common.py:
  "load_json_array()" in scripts/prepare_gap_analysis_predictions.py (lines 16-25)
  vs "load_json_object()" in scripts/validate_cosmos_embed_output.py (lines 84-90)
  vs "load_json_array()" in scripts/workflow_common.py (lines 48-57)
  vs "load_json_object()" in scripts/workflow_common.py (lines 60-66)
  vs "read_jsonl()" in scripts/workflow_common.py (lines 110-125) (`scripts/prepare_gap_analysis_predictions.py:16`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-cr-its-mining/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/applications/tao-run-deft-cr-its-mining/SKILL.md`)
- 73 additional finding(s) are available in the full evaluation artifacts.

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
