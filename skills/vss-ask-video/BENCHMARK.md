# Skill Benchmark: vss-ask-video

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-ask-video`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 17 evaluation tasks (16 positive, 1 negative)
- Dataset digest: `sha256:830efbe7c931304d5f442008bdbca6a2d7b5552ad59745dfd4190886fe681b13` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| vss-ask-video | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-ask-video | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-ask-video | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-ask-video | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| vss-ask-video | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `vss-ask-video`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.1% → 100.0% (+2.9 points) | 91.2% → 100.0% (+8.8 points) |
| Correctness | 44.7% → 98.8% (+54.1 points) | 35.3% → 84.7% (+49.4 points) |
| Discoverability | 87.5% — baseline ran, but no comparable score was available; uplift unavailable | 81.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.7% → 74.0% (+32.3 points) | 43.1% → 58.3% (+15.2 points) |
| Efficiency | 97.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 95.5% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.8% → 91.8% (-1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,673,966 | 1,971,679 | -297,713 | -15.10% | skill 17/17; base 17/17 |
| claude-code | archive-search-non-activation | 156,290 | 107,915 | +48,375 | +44.83% | skill 1/1; base 1/1 |
| claude-code | direct-file-vlm | 122,371 | 220,345 | -97,974 | -44.46% | skill 1/1; base 1/1 |
| claude-code | exact-stored-job | 121,306 | 143,153 | -21,847 | -15.26% | skill 1/1; base 1/1 |
| claude-code | explicit-fresh-window | 121,922 | 107,468 | +14,454 | +13.45% | skill 1/1; base 1/1 |
| claude-code | hot-context-sufficient | 35,306 | 35,041 | +265 | +0.76% | skill 1/1; base 1/1 |
| claude-code | introspection-disabled | 121,994 | 107,205 | +14,789 | +13.80% | skill 1/1; base 1/1 |
| claude-code | introspection-partial | 78,550 | 35,076 | +43,474 | +123.94% | skill 1/1; base 1/1 |
| claude-code | introspection-unconfigured | 121,706 | 145,183 | -23,477 | -16.17% | skill 1/1; base 1/1 |
| claude-code | invalid-child-identity | 78,303 | 179,485 | -101,182 | -56.37% | skill 1/1; base 1/1 |
| claude-code | markdown-pointer-introspection-enabled | 122,033 | 143,348 | -21,315 | -14.87% | skill 1/1; base 1/1 |
| claude-code | markdown-sufficient | 35,313 | 106,579 | -71,266 | -66.87% | skill 1/1; base 1/1 |
| claude-code | no-markdown-introspection-enabled | 121,935 | 106,783 | +15,152 | +14.19% | skill 1/1; base 1/1 |
| claude-code | no-memory-grounded-window | 79,418 | 142,454 | -63,036 | -44.25% | skill 1/1; base 1/1 |
| claude-code | no-memory-without-scope | 78,806 | 71,033 | +7,773 | +10.94% | skill 1/1; base 1/1 |
| claude-code | separate-shell-cli | 78,470 | 70,631 | +7,839 | +11.10% | skill 1/1; base 1/1 |
| claude-code | whole-recording-long | 78,623 | 143,371 | -64,748 | -45.16% | skill 1/1; base 1/1 |
| claude-code | whole-recording-short | 121,620 | 106,609 | +15,011 | +14.08% | skill 1/1; base 1/1 |
| codex | All cases | 976,464 | 707,814 | +268,650 | +37.95% | skill 17/17; base 17/17 |
| codex | archive-search-non-activation | 77,003 | 42,070 | +34,933 | +83.04% | skill 1/1; base 1/1 |
| codex | direct-file-vlm | 46,473 | 190,767 | -144,294 | -75.64% | skill 1/1; base 1/1 |
| codex | exact-stored-job | 46,244 | 46,080 | +164 | +0.36% | skill 1/1; base 1/1 |
| codex | explicit-fresh-window | 63,286 | 40,826 | +22,460 | +55.01% | skill 1/1; base 1/1 |
| codex | hot-context-sufficient | 13,314 | 13,136 | +178 | +1.36% | skill 1/1; base 1/1 |
| codex | introspection-disabled | 66,326 | 40,899 | +25,427 | +62.17% | skill 1/1; base 1/1 |
| codex | introspection-partial | 63,687 | 13,216 | +50,471 | +381.89% | skill 1/1; base 1/1 |
| codex | introspection-unconfigured | 66,478 | 41,103 | +25,375 | +61.74% | skill 1/1; base 1/1 |
| codex | invalid-child-identity | 47,952 | 60,214 | -12,262 | -20.36% | skill 1/1; base 1/1 |
| codex | markdown-pointer-introspection-enabled | 86,752 | 55,037 | +31,715 | +57.62% | skill 1/1; base 1/1 |
| codex | markdown-sufficient | 13,367 | 13,205 | +162 | +1.23% | skill 1/1; base 1/1 |
| codex | no-markdown-introspection-enabled | 86,648 | 41,067 | +45,581 | +110.99% | skill 1/1; base 1/1 |
| codex | no-memory-grounded-window | 67,503 | 13,334 | +54,169 | +406.25% | skill 1/1; base 1/1 |
| codex | no-memory-without-scope | 29,758 | 13,205 | +16,553 | +125.35% | skill 1/1; base 1/1 |
| codex | separate-shell-cli | 89,156 | 29,950 | +59,206 | +197.68% | skill 1/1; base 1/1 |
| codex | whole-recording-long | 66,256 | 26,846 | +39,410 | +146.80% | skill 1/1; base 1/1 |
| codex | whole-recording-short | 46,261 | 26,859 | +19,402 | +72.24% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,650,430 | 2,679,493 | -29,063 | -1.08% | skill 34/34; base 34/34 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5290 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/operations/vss-ask-video/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/operations/vss-ask-video`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/operations/vss-ask-video/SKILL.md`)
- **MEDIUM** SECURITY/Unrestricted Tool Access (EA1): Excessive Agency: execute every tool (`SKILL.md:167`)
- **LOW** QUALITY/quality_discoverability: Description very long (448 chars, recommend 50-150) (`skills/operations/vss-ask-video/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
