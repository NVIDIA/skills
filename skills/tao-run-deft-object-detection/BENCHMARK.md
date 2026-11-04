# Skill Benchmark: tao-run-deft-object-detection

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-object-detection`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:d3fce2a83efb9ec178ecc5fc8a20f8c2280adffa3b279916a073015372c25237` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 81.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 10.0% → 100.0% (+90.0 points) | 20.0% → 75.0% (+55.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 5.1% → 87.6% (+82.5 points) | 21.4% → 67.0% (+45.6 points) |
| Efficiency | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 566,219 | 1,367,224 | -801,005 | -58.59% | skill 4/4; base 4/4 |
| claude-code | tao-run-deft-object-detection-basic | 102,497 | 243,352 | -140,855 | -57.88% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-resume-state | 296,096 | 249,754 | +46,342 | +18.56% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-run-loop | 67,481 | 326,395 | -258,914 | -79.33% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-stage-skills | 100,145 | 547,723 | -447,578 | -81.72% | skill 1/1; base 1/1 |
| codex | All cases | 262,171 | 163,074 | +99,097 | +60.77% | skill 4/4; base 4/4 |
| codex | tao-run-deft-object-detection-basic | 30,978 | 13,544 | +17,434 | +128.72% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-resume-state | 185,925 | 122,085 | +63,840 | +52.29% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-run-loop | 31,182 | 13,893 | +17,289 | +124.44% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-stage-skills | 14,086 | 13,552 | +534 | +3.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 828,390 | 1,530,298 | -701,908 | -45.87% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 130 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/applications/tao-run-deft-object-detection`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within references/pipeline-and-state.md:
  "## Pipeline" in references/pipeline-and-state.md (lines 23-26)
  vs "## Stage Execution" in references/pipeline-and-state.md (lines 160-170) (`references/pipeline-and-state.md:23`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/applications/tao-run-deft-object-detection/SKILL.md`)
- 127 additional finding(s) are available in the full evaluation artifacts.

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
