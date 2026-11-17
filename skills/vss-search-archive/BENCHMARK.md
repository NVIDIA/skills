# Skill Benchmark: vss-search-archive

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-search-archive`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 18 evaluation tasks (14 positive, 4 negative)
- Dataset digest: `sha256:c7f67a32c256d3ea604292404038de74527eaf4d41c696cd5ec21622f8de6d1a` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| vss-search-archive | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-search-archive | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-search-archive | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-search-archive | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-search-archive | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `vss-search-archive`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 88.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 97.2% → 100.0% (+2.8 points) |
| Correctness | 37.8% → 95.6% (+57.8 points) | 65.6% → 88.9% (+23.3 points) |
| Discoverability | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 29.6% → 66.2% (+36.6 points) | 57.4% → 67.1% (+9.7 points) |
| Efficiency | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 96.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.8% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.1% → 92.4% (+1.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,779,632 | 1,973,295 | -193,663 | -9.81% | skill 18/18; base 18/18 |
| claude-code | search-archive | 186,476 | 107,612 | +78,864 | +73.29% | skill 1/1; base 1/1 |
| claude-code | search-archive-attribute-flags | 241,688 | 145,071 | +96,617 | +66.60% | skill 1/1; base 1/1 |
| claude-code | search-archive-confirm-verification | 295,538 | 107,118 | +188,420 | +175.90% | skill 1/1; base 1/1 |
| claude-code | search-archive-delete-only | 35,652 | 143,288 | -107,636 | -75.12% | skill 1/1; base 1/1 |
| claude-code | search-archive-exit2-help | 76,185 | 35,496 | +40,689 | +114.63% | skill 1/1; base 1/1 |
| claude-code | search-archive-exit6-partial | 76,066 | 179,729 | -103,663 | -57.68% | skill 1/1; base 1/1 |
| claude-code | search-archive-exit6-without-data | 75,559 | 106,813 | -31,254 | -29.26% | skill 1/1; base 1/1 |
| claude-code | search-archive-fusion-decompose | 76,328 | 106,251 | -29,923 | -28.16% | skill 1/1; base 1/1 |
| claude-code | search-archive-implicit | 76,632 | 144,576 | -67,944 | -47.00% | skill 1/1; base 1/1 |
| claude-code | search-archive-ingest-only | 35,539 | 144,211 | -108,672 | -75.36% | skill 1/1; base 1/1 |
| claude-code | search-archive-missing-source | 75,613 | 35,227 | +40,386 | +114.65% | skill 1/1; base 1/1 |
| claude-code | search-archive-mixed-upload-and-stream | 76,317 | 143,867 | -67,550 | -46.95% | skill 1/1; base 1/1 |
| claude-code | search-archive-negative-direct-video-qa | 71,374 | 106,270 | -34,896 | -32.84% | skill 1/1; base 1/1 |
| claude-code | search-archive-negative-summary | 35,448 | 70,725 | -35,277 | -49.88% | skill 1/1; base 1/1 |
| claude-code | search-archive-object-flags | 116,447 | 182,092 | -65,645 | -36.05% | skill 1/1; base 1/1 |
| claude-code | search-archive-partially-verified | 75,648 | 35,158 | +40,490 | +115.17% | skill 1/1; base 1/1 |
| claude-code | search-archive-results-first | 76,850 | 35,857 | +40,993 | +114.32% | skill 1/1; base 1/1 |
| claude-code | search-archive-rtsp-live-stream | 76,272 | 143,934 | -67,662 | -47.01% | skill 1/1; base 1/1 |
| codex | All cases | 778,984 | 941,051 | -162,067 | -17.22% | skill 18/18; base 18/18 |
| codex | search-archive | 57,569 | 56,215 | +1,354 | +2.41% | skill 1/1; base 1/1 |
| codex | search-archive-attribute-flags | 46,985 | 87,685 | -40,700 | -46.42% | skill 1/1; base 1/1 |
| codex | search-archive-confirm-verification | 65,549 | 26,960 | +38,589 | +143.13% | skill 1/1; base 1/1 |
| codex | search-archive-delete-only | 41,397 | 59,096 | -17,699 | -29.95% | skill 1/1; base 1/1 |
| codex | search-archive-exit2-help | 46,911 | 13,271 | +33,640 | +253.49% | skill 1/1; base 1/1 |
| codex | search-archive-exit6-partial | 62,548 | 49,214 | +13,334 | +27.09% | skill 1/1; base 1/1 |
| codex | search-archive-exit6-without-data | 29,589 | 69,880 | -40,291 | -57.66% | skill 1/1; base 1/1 |
| codex | search-archive-fusion-decompose | 64,716 | 89,851 | -25,135 | -27.97% | skill 1/1; base 1/1 |
| codex | search-archive-implicit | 47,539 | 69,302 | -21,763 | -31.40% | skill 1/1; base 1/1 |
| codex | search-archive-ingest-only | 42,005 | 55,845 | -13,840 | -24.78% | skill 1/1; base 1/1 |
| codex | search-archive-missing-source | 29,569 | 13,248 | +16,321 | +123.20% | skill 1/1; base 1/1 |
| codex | search-archive-mixed-upload-and-stream | 47,019 | 34,752 | +12,267 | +35.30% | skill 1/1; base 1/1 |
| codex | search-archive-negative-direct-video-qa | 13,440 | 45,171 | -31,731 | -70.25% | skill 1/1; base 1/1 |
| codex | search-archive-negative-summary | 13,382 | 60,150 | -46,768 | -77.75% | skill 1/1; base 1/1 |
| codex | search-archive-object-flags | 47,010 | 70,307 | -23,297 | -33.14% | skill 1/1; base 1/1 |
| codex | search-archive-partially-verified | 29,544 | 13,236 | +16,308 | +123.21% | skill 1/1; base 1/1 |
| codex | search-archive-results-first | 47,141 | 59,754 | -12,613 | -21.11% | skill 1/1; base 1/1 |
| codex | search-archive-rtsp-live-stream | 47,071 | 67,114 | -20,043 | -29.86% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,558,616 | 2,914,346 | -355,730 | -12.21% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/cli_usage.md:
  "## Search workflow" in SKILL.md (lines 103-108)
  vs "## Output and exits" in references/cli_usage.md (lines 139-143)
  vs "## Output and exits" in references/cli_usage.md (lines 147-160) (`SKILL.md:103`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/operations/vss-search-archive/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/operations/vss-search-archive/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in cli_usage.md (`skills/operations/vss-search-archive/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/operations/vss-search-archive`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
