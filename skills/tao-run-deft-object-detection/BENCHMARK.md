# Skill Benchmark: tao-run-deft-object-detection

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-object-detection`
- Validation status: `passed`
- Evaluation date: 2026-11-24
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:d3fce2a83efb9ec178ecc5fc8a20f8c2280adffa3b279916a073015372c25237` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| tao-run-deft-object-detection | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-deft-object-detection | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-deft-object-detection | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-deft-object-detection | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-deft-object-detection | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `tao-run-deft-object-detection`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 63.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 15.0% → 90.0% (+75.0 points) | 20.0% → 70.0% (+50.0 points) |
| Discoverability | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 23.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 14.4% → 62.5% (+48.1 points) | 16.2% → 27.8% (+11.6 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 96.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 96.0% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.2% → 93.1% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 394,191 | 464,629 | -70,438 | -15.16% | skill 4/4; base 4/4 |
| claude-code | tao-run-deft-object-detection-basic | 77,846 | 35,655 | +42,191 | +118.33% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-resume-state | 158,695 | 181,174 | -22,479 | -12.41% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-run-loop | 36,876 | 35,849 | +1,027 | +2.86% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-object-detection-stage-skills | 120,774 | 211,951 | -91,177 | -43.02% | skill 1/1; base 1/1 |
| codex | All cases | 139,370 | 100,935 | +38,435 | +38.08% | skill 4/4; base 4/4 |
| codex | tao-run-deft-object-detection-basic | 13,724 | 13,337 | +387 | +2.90% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-resume-state | 98,138 | 60,656 | +37,482 | +61.79% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-run-loop | 13,932 | 13,640 | +292 | +2.14% | skill 1/1; base 1/1 |
| codex | tao-run-deft-object-detection-stage-skills | 13,576 | 13,302 | +274 | +2.06% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 533,561 | 565,564 | -32,003 | -5.66% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 97 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/scripts-and-agents.md:
  "## Stage Reference Modules" in SKILL.md (lines 117-138)
  vs "## Stage Reference Modules" in references/scripts-and-agents.md (lines 133-146) (`SKILL.md:117`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/tao-analyze-detection-kpi.md and scripts/summarize_kpi.py:
  "#   docker rm "deft_${PHASE}_kpi"" in references/tao-analyze-detection-kpi.md (lines 157-182)
  vs "## Outputs" in references/tao-analyze-detection-kpi.md (lines 183-202)
  vs "(module docstring)" in scripts/summarize_kpi.py (lines 1-34) (`references/tao-analyze-detection-kpi.md:157`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within references/pipeline-and-state.md:
  "## Pipeline" in references/pipeline-and-state.md (lines 23-26)
  vs "## Stage Execution" in references/pipeline-and-state.md (lines 160-170) (`references/pipeline-and-state.md:23`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/grounding-dino.md and references/pipeline-and-state.md:
  "## Iteration train" in references/grounding-dino.md (lines 205-252)
  vs "### Iteration N — seven stages, in order" in references/pipeline-and-state.md (lines 81-92) (`references/grounding-dino.md:205`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across references/pipeline-and-state.md and references/tao-analyze-detection-kpi.md and references/tao-analyze-gaps-od-map.md:
  "### Iteration N — seven stages, in order" in references/pipeline-and-state.md (lines 45-45)
  vs "## When to invoke" in references/tao-analyze-detection-kpi.md (lines 5-11)
  vs "## When to invoke" in references/tao-analyze-gaps-od-map.md (lines 5-8) (`references/pipeline-and-state.md:45`)
- 97 additional finding(s) are available in the full evaluation artifacts.

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
