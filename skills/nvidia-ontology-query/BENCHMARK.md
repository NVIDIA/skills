# Skill Benchmark: nvidia-ontology-query

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-query`
- Validation status: `passed`
- Evaluation date: 2026-11-19
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 15 evaluation tasks (15 positive)
- Dataset digest: `sha256:9ebf688eb0f22bd40dbdb3d21e5ecc3404a2c9abcd0177e05700ab3fe9fa3ed7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-query | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nvidia-ontology-query`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.7% → 100.0% (+3.3 points) | 100.0% → 96.7% (-3.3 points) |
| Correctness | 53.3% → 98.7% (+45.4 points) | 58.7% → 94.7% (+36.0 points) |
| Discoverability | 96.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 53.3% → 91.1% (+37.8 points) | 57.5% → 88.3% (+30.8 points) |
| Efficiency | 92.2% — baseline ran, but no comparable score was available; uplift unavailable | 87.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 96.1% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.6% → 90.3% (-2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,768,208 | 1,212,654 | +555,554 | +45.81% | skill 15/15; base 15/15 |
| claude-code | query-eval-001-mcp-first | 114,065 | 143,599 | -29,534 | -20.57% | skill 1/1; base 1/1 |
| claude-code | query-eval-002-no-fastapi-public | 74,303 | 107,257 | -32,954 | -30.72% | skill 1/1; base 1/1 |
| claude-code | query-eval-003-conversation-409 | 108,294 | 143,106 | -34,812 | -24.33% | skill 1/1; base 1/1 |
| claude-code | query-eval-004-sso-vs-api-token | 114,077 | 70,870 | +43,207 | +60.97% | skill 1/1; base 1/1 |
| claude-code | query-eval-005-no-raw-schema-tools | 73,816 | 35,257 | +38,559 | +109.37% | skill 1/1; base 1/1 |
| claude-code | query-eval-006-cuopt-not-in-repo | 377,147 | 143,917 | +233,230 | +162.06% | skill 1/1; base 1/1 |
| claude-code | query-eval-007-readiness-before-retry | 73,803 | 70,864 | +2,939 | +4.15% | skill 1/1; base 1/1 |
| claude-code | query-eval-008-quantity-not-ratio | 114,672 | 71,148 | +43,524 | +61.17% | skill 1/1; base 1/1 |
| claude-code | query-eval-009-run-metric-fanout | 113,913 | 35,640 | +78,273 | +219.62% | skill 1/1; base 1/1 |
| claude-code | query-eval-010-scenario-lineage | 114,295 | 107,278 | +7,017 | +6.54% | skill 1/1; base 1/1 |
| claude-code | query-eval-011-prose-versus-rows | 113,961 | 70,955 | +43,006 | +60.61% | skill 1/1; base 1/1 |
| claude-code | query-eval-012-source-role | 114,318 | 71,340 | +42,978 | +60.24% | skill 1/1; base 1/1 |
| claude-code | query-eval-013-truncated-result | 113,750 | 35,485 | +78,265 | +220.56% | skill 1/1; base 1/1 |
| claude-code | query-eval-014-information-answer | 73,847 | 70,641 | +3,206 | +4.54% | skill 1/1; base 1/1 |
| claude-code | query-eval-015-unanswered-message | 73,947 | 35,297 | +38,650 | +109.50% | skill 1/1; base 1/1 |
| codex | All cases | 1,701,754 | 468,887 | +1,232,867 | +262.93% | skill 15/15; base 15/15 |
| codex | query-eval-001-mcp-first | 50,004 | 57,374 | -7,370 | -12.85% | skill 1/1; base 1/1 |
| codex | query-eval-002-no-fastapi-public | 28,848 | 13,278 | +15,570 | +117.26% | skill 1/1; base 1/1 |
| codex | query-eval-003-conversation-409 | 74,308 | 54,531 | +19,777 | +36.27% | skill 1/1; base 1/1 |
| codex | query-eval-004-sso-vs-api-token | 48,938 | 34,832 | +14,106 | +40.50% | skill 1/1; base 1/1 |
| codex | query-eval-005-no-raw-schema-tools | 28,703 | 13,294 | +15,409 | +115.91% | skill 1/1; base 1/1 |
| codex | query-eval-006-cuopt-not-in-repo | 1,106,238 | 45,088 | +1,061,150 | +2353.51% | skill 1/1; base 1/1 |
| codex | query-eval-007-readiness-before-retry | 28,679 | 13,176 | +15,503 | +117.66% | skill 1/1; base 1/1 |
| codex | query-eval-008-quantity-not-ratio | 44,886 | 13,245 | +31,641 | +238.89% | skill 1/1; base 1/1 |
| codex | query-eval-009-run-metric-fanout | 45,171 | 40,598 | +4,573 | +11.26% | skill 1/1; base 1/1 |
| codex | query-eval-010-scenario-lineage | 44,969 | 40,513 | +4,456 | +11.00% | skill 1/1; base 1/1 |
| codex | query-eval-011-prose-versus-rows | 44,902 | 13,201 | +31,701 | +240.14% | skill 1/1; base 1/1 |
| codex | query-eval-012-source-role | 44,991 | 13,259 | +31,732 | +239.32% | skill 1/1; base 1/1 |
| codex | query-eval-013-truncated-result | 44,888 | 13,256 | +31,632 | +238.62% | skill 1/1; base 1/1 |
| codex | query-eval-014-information-answer | 33,095 | 62,392 | -29,297 | -46.96% | skill 1/1; base 1/1 |
| codex | query-eval-015-unanswered-message | 33,134 | 40,850 | -7,716 | -18.89% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,469,962 | 1,681,541 | +1,788,421 | +106.36% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without checking (`SKILL.md:99`)
- **LOW** QUALITY/quality_discoverability: Description very long (274 chars, recommend 50-150) (`skills/nvidia-ontology-query/SKILL.md`)

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
