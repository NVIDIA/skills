# Skill Benchmark: vss-summarize-video

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-summarize-video`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 3 evaluation tasks (2 positive, 1 negative)
- Dataset digest: `sha256:2d3aa44da1d687045c1aa735919e28dc2ffee16c19aa3676de506299a16876bc` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| vss-summarize-video | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-summarize-video | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-summarize-video | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-summarize-video | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-summarize-video | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `vss-summarize-video`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 78.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 83.3% → 100.0% (+16.7 points) |
| Correctness | 26.7% → 93.3% (+66.6 points) | 13.3% → 53.3% (+40.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.3% → 51.1% (+20.8 points) | 30.3% → 47.0% (+16.7 points) |
| Efficiency | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 96.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 96.0% (-2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.5% → 91.9% (-0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 386,819 | 359,045 | +27,774 | +7.74% | skill 3/3; base 3/3 |
| claude-code | summarize-video | 121,898 | 144,347 | -22,449 | -15.55% | skill 1/1; base 1/1 |
| claude-code | summarize-video-negative-archive-search | 143,590 | 71,458 | +72,132 | +100.94% | skill 1/1; base 1/1 |
| claude-code | summarize-video-no-lvs-vlm-fallback | 121,331 | 143,240 | -21,909 | -15.30% | skill 1/1; base 1/1 |
| codex | All cases | 152,454 | 188,736 | -36,282 | -19.22% | skill 3/3; base 3/3 |
| codex | summarize-video | 49,073 | 106,201 | -57,128 | -53.79% | skill 1/1; base 1/1 |
| codex | summarize-video-negative-archive-search | 57,534 | 27,335 | +30,199 | +110.48% | skill 1/1; base 1/1 |
| codex | summarize-video-no-lvs-vlm-fallback | 45,847 | 55,200 | -9,353 | -16.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 539,273 | 547,781 | -8,508 | -1.55% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 7 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/cli_usage.md:
  "### Stage 4: Submit Once Through the CLI" in SKILL.md (lines 261-269)
  vs "## Output and exits" in references/cli_usage.md (lines 136-145) (`SKILL.md:261`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/hitl-prompts.md:
  "### Stage 4: Submit Once Through the CLI" in SKILL.md (lines 253-255)
  vs "### HITL: collect scenario and events first (REQUIRED — do not skip)" in references/hitl-prompts.md (lines 76-78) (`SKILL.md:253`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/cli_usage.md and references/deploy-lvs-service.md and references/end-to-end-example.md and references/hitl-prompts.md and references/integrate-lvs-service.md and references/video-summarization-api.md and references/video-summarization-deployment.md and references/video-summarization-environment-variables.md:
  "## When to Use" in SKILL.md (lines 25-35)
  vs "## Required References" in SKILL.md (lines 36-74)
  vs "## The one job command" in references/cli_usage.md (lines 33-67)
  vs "## Container Image" in references/deploy-lvs-service.md (lines 3-15)
  vs "### Submit one summarize job" in references/end-to-end-example.md (lines 93-101)
  vs "### HITL: collect scenario and events first (REQUIRED — do not skip)" in references/hitl-prompts.md (lines 4-8)
  vs "### HITL: collect scenario and events first (REQUIRED — do not skip)" in references/hitl-prompts.md (lines 9-14)
  vs "### HITL: collect scenario and events first (REQUIRED — do not skip)" in references/hitl-prompts.md (lines 17-29)
  vs "### HITL: collect scenario and events first (REQUIRED — do not skip)" in references/hitl-prompts.md (lines 30-31)
  vs "## Overview" in references/integrate-lvs-service.md (lines 3-11)
  vs "### Inputs" in references/integrate-lvs-service.md (lines 34-63)
  vs "## API Schema" in references/integrate-lvs-service.md (lines 83-124)
  vs "## Environment Variables" in references/integrate-lvs-service.md (lines 125-152)
  vs "## Example Compose Snippet" in references/integrate-lvs-service.md (lines 191-235)
  vs "# Video Summarization API Reference" in references/video-summarization-api.md (lines 1-11)
  vs "## File Summarization" in references/video-summarization-api.md (lines 129-131)
  vs "## File Summarization" in references/video-summarization-api.md (lines 145-155)
  vs "## File Summarization" in references/video-summarization-api.md (lines 186-189)
  vs "# Video Summarization Deployment Reference" in references/video-summarization-deployment.md (lines 1-5)
  vs "## Current VSS Docker Compose Shape" in references/video-summarization-deployment.md (lines 6-31)
  vs "## Required Inputs" in references/video-summarization-deployment.md (lines 84-96)
  vs "## Required Inputs" in references/video-summarization-deployment.md (lines 99-101)
  vs "## Database Backend Selection" in references/video-summarization-deployment.md (lines 239-248)
  vs "# Video Summarization Environment Variables" in references/video-summarization-environment-variables.md (lines 1-6)
  vs "## Profile Env" in references/video-summarization-environment-variables.md (lines 29-36)
  vs "## Profile Env" in references/video-summarization-environment-variables.md (lines 45-59)
  vs "## Profile Env" in references/video-summarization-environment-variables.md (lines 61-76)
  vs "## Profile Env" in references/video-summarization-environment-variables.md (lines 77-85)
  vs "## Runtime Rules" in references/video-summarization-environment-variables.md (lines 203-213) (`SKILL.md:25`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/video-summarization-deployment.md and references/video-summarization-environment-variables.md:
  "## Required Inputs" in references/video-summarization-deployment.md (lines 63-71)
  vs "## Profile Env" in references/video-summarization-environment-variables.md (lines 8-11) (`references/video-summarization-deployment.md:63`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/video-summarization-debugging.md and references/video-summarization-deployment.md:
  "## Fast Status" in references/video-summarization-debugging.md (lines 6-18)
  vs "## Verify Running Service" in references/video-summarization-deployment.md (lines 32-45) (`references/video-summarization-debugging.md:6`)
- 15 additional finding(s) are available in the full evaluation artifacts.

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
