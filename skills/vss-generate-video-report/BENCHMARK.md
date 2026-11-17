# Skill Benchmark: vss-generate-video-report

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-generate-video-report`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 14 evaluation tasks (13 positive, 1 negative)
- Dataset digest: `sha256:7df8115f049bfd086f9d9ee077273beec9ef5f0167a6b6f16ae646ea847d284b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| vss-generate-video-report | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-generate-video-report | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-generate-video-report | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-generate-video-report | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-generate-video-report | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `vss-generate-video-report`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 83.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 12.9% → 94.3% (+81.4 points) | 37.1% → 94.3% (+57.2 points) |
| Discoverability | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 9.1% → 74.3% (+65.2 points) | 28.6% → 77.6% (+49.0 points) |
| Efficiency | 72.8% — baseline ran, but no comparable score was available; uplift unavailable | 95.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 94.1% (-4.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.2% → 91.1% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,820,532 | 1,958,474 | -137,942 | -7.04% | skill 14/14; base 14/14 |
| claude-code | summarize-without-report-non-activation | 145,063 | 106,285 | +38,778 | +36.48% | skill 1/1; base 1/1 |
| claude-code | video-report-empty-range-policy | 190,704 | 70,620 | +120,084 | +170.04% | skill 1/1; base 1/1 |
| claude-code | video-report-hitl-harness-fallback | 161,112 | 142,153 | +18,959 | +13.34% | skill 1/1; base 1/1 |
| claude-code | video-report-hitl-multi-edit-loop | 149,186 | 142,025 | +7,161 | +5.04% | skill 1/1; base 1/1 |
| claude-code | video-report-hitl-off-policy | 184,260 | 177,400 | +6,860 | +3.87% | skill 1/1; base 1/1 |
| claude-code | video-report-hitl-on-policy | 184,568 | 179,034 | +5,534 | +3.09% | skill 1/1; base 1/1 |
| claude-code | video-report-hitl-stall-guard | 86,345 | 142,063 | -55,718 | -39.22% | skill 1/1; base 1/1 |
| claude-code | video-report-local-or-base64-path | 153,132 | 216,304 | -63,172 | -29.21% | skill 1/1; base 1/1 |
| claude-code | video-report-long-video-lvs-handoff | 124,320 | 178,173 | -53,853 | -30.23% | skill 1/1; base 1/1 |
| claude-code | video-report-profile-agnostic-mode-a | 123,697 | 178,551 | -54,854 | -30.72% | skill 1/1; base 1/1 |
| claude-code | video-report-routing-mode-a | 35,236 | 35,216 | +20 | +0.06% | skill 1/1; base 1/1 |
| claude-code | video-report-routing-mode-b | 35,147 | 70,390 | -35,243 | -50.07% | skill 1/1; base 1/1 |
| claude-code | video-report-step3-no-vss-agent-fallback | 123,170 | 142,240 | -19,070 | -13.41% | skill 1/1; base 1/1 |
| claude-code | video-report-vlm-unclear-options | 124,592 | 178,020 | -53,428 | -30.01% | skill 1/1; base 1/1 |
| codex | All cases | 937,753 | 654,384 | +283,369 | +43.30% | skill 14/14; base 14/14 |
| codex | summarize-without-report-non-activation | 13,276 | 13,167 | +109 | +0.83% | skill 1/1; base 1/1 |
| codex | video-report-empty-range-policy | 47,634 | 53,716 | -6,082 | -11.32% | skill 1/1; base 1/1 |
| codex | video-report-hitl-harness-fallback | 85,966 | 54,522 | +31,444 | +57.67% | skill 1/1; base 1/1 |
| codex | video-report-hitl-multi-edit-loop | 83,694 | 58,499 | +25,195 | +43.07% | skill 1/1; base 1/1 |
| codex | video-report-hitl-off-policy | 65,588 | 54,474 | +11,114 | +20.40% | skill 1/1; base 1/1 |
| codex | video-report-hitl-on-policy | 64,126 | 54,653 | +9,473 | +17.33% | skill 1/1; base 1/1 |
| codex | video-report-hitl-stall-guard | 66,019 | 59,144 | +6,875 | +11.62% | skill 1/1; base 1/1 |
| codex | video-report-local-or-base64-path | 100,639 | 41,038 | +59,601 | +145.23% | skill 1/1; base 1/1 |
| codex | video-report-long-video-lvs-handoff | 86,560 | 55,772 | +30,788 | +55.20% | skill 1/1; base 1/1 |
| codex | video-report-profile-agnostic-mode-a | 107,567 | 52,797 | +54,770 | +103.74% | skill 1/1; base 1/1 |
| codex | video-report-routing-mode-a | 13,263 | 13,319 | -56 | -0.42% | skill 1/1; base 1/1 |
| codex | video-report-routing-mode-b | 29,600 | 13,223 | +16,377 | +123.85% | skill 1/1; base 1/1 |
| codex | video-report-step3-no-vss-agent-fallback | 87,922 | 75,476 | +12,446 | +16.49% | skill 1/1; base 1/1 |
| codex | video-report-vlm-unclear-options | 85,899 | 54,584 | +31,315 | +57.37% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,758,285 | 2,612,858 | +145,427 | +5.57% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within references/report-types/sop-compliance.md:
  "# consumes the VA_MCP_URL it validated (no re-derivation, no Docker fallback)." in references/report-types/sop-compliance.md (lines 25-32)
  vs "# Fresh shell: paste the Endpoint resolution hand-off here too — this block requires its VA_MCP_URL." in references/report-types/sop-compliance.md (lines 55-62) (`references/report-types/sop-compliance.md:25`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/operations/vss-generate-video-report/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (9004 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/operations/vss-generate-video-report/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/operations/vss-generate-video-report`)
- **MEDIUM** SECURITY/skillspector_execution_failure: skillspector scan did not execute reliably (agent-control analysis limitation): references/report-types/sop-compliance.md [static; static_parse_limit; analyzers=static_patterns_tool_misuse]: A security-relevant expression could not be fully resolved by the bounded static parser. SkillSpector cannot fully inspect this code pattern yet. Check: printf invocation; location: reported source range; unchecked scope: command word. Relevant parser input not fully checked.; references/report-types/video-analysis.md [static; static_parse_limit; analyzers=static_patterns_tool_misuse]: A security-relevant expression could not be fully resolved by the bounded static parser. SkillSpector cannot fully inspect this code pattern yet. Check: printf invocation; location: reported source range; unchecked scope: command word. Relevant parser input not fully checked. (`.`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
