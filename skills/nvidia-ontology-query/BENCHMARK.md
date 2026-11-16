# Skill Benchmark: nvidia-ontology-query

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-query`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:314823168b2eada85a48a450189c9d2cc1d374eed343c106147bd2b63080bf72` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-query | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-query | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

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
| Overall | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 85.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 52.3% → 98.5% (+46.2 points) | 53.9% → 78.5% (+24.6 points) |
| Discoverability | 82.3% — baseline ran, but no comparable score was available; uplift unavailable | 77.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 54.2% → 81.4% (+27.2 points) | 59.0% → 80.1% (+21.1 points) |
| Efficiency | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.8% → 96.6% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.6% → 92.3% (-0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,575,273 | 961,353 | +613,920 | +63.86% | skill 13/13; base 13/13 |
| claude-code | query-eval-001-mcp-first | 113,356 | 143,452 | -30,096 | -20.98% | skill 1/1; base 1/1 |
| claude-code | query-eval-002-no-fastapi-public | 74,425 | 70,970 | +3,455 | +4.87% | skill 1/1; base 1/1 |
| claude-code | query-eval-003-conversation-409 | 110,419 | 142,457 | -32,038 | -22.49% | skill 1/1; base 1/1 |
| claude-code | query-eval-004-sso-vs-api-token | 112,773 | 70,674 | +42,099 | +59.57% | skill 1/1; base 1/1 |
| claude-code | query-eval-005-no-raw-schema-tools | 73,483 | 35,337 | +38,146 | +107.95% | skill 1/1; base 1/1 |
| claude-code | query-eval-006-cuopt-not-in-repo | 457,091 | 107,318 | +349,773 | +325.92% | skill 1/1; base 1/1 |
| claude-code | query-eval-007-readiness-before-retry | 73,809 | 35,310 | +38,499 | +109.03% | skill 1/1; base 1/1 |
| claude-code | query-eval-008-quantity-not-ratio | 35,546 | 71,140 | -35,594 | -50.03% | skill 1/1; base 1/1 |
| claude-code | query-eval-009-run-metric-fanout | 113,097 | 35,740 | +77,357 | +216.44% | skill 1/1; base 1/1 |
| claude-code | query-eval-010-scenario-lineage | 113,502 | 107,232 | +6,270 | +5.85% | skill 1/1; base 1/1 |
| claude-code | query-eval-011-prose-versus-rows | 113,272 | 35,221 | +78,051 | +221.60% | skill 1/1; base 1/1 |
| claude-code | query-eval-012-source-role | 71,435 | 71,044 | +391 | +0.55% | skill 1/1; base 1/1 |
| claude-code | query-eval-013-truncated-result | 113,065 | 35,458 | +77,607 | +218.87% | skill 1/1; base 1/1 |
| codex | All cases | 526,096 | 410,591 | +115,505 | +28.13% | skill 13/13; base 13/13 |
| codex | query-eval-001-mcp-first | 48,618 | 60,316 | -11,698 | -19.39% | skill 1/1; base 1/1 |
| codex | query-eval-002-no-fastapi-public | 13,328 | 13,297 | +31 | +0.23% | skill 1/1; base 1/1 |
| codex | query-eval-003-conversation-409 | 72,867 | 54,538 | +18,329 | +33.61% | skill 1/1; base 1/1 |
| codex | query-eval-004-sso-vs-api-token | 48,334 | 49,435 | -1,101 | -2.23% | skill 1/1; base 1/1 |
| codex | query-eval-005-no-raw-schema-tools | 28,395 | 40,334 | -11,939 | -29.60% | skill 1/1; base 1/1 |
| codex | query-eval-006-cuopt-not-in-repo | 50,871 | 58,736 | -7,865 | -13.39% | skill 1/1; base 1/1 |
| codex | query-eval-007-readiness-before-retry | 13,313 | 13,257 | +56 | +0.42% | skill 1/1; base 1/1 |
| codex | query-eval-008-quantity-not-ratio | 44,493 | 13,260 | +31,233 | +235.54% | skill 1/1; base 1/1 |
| codex | query-eval-009-run-metric-fanout | 44,555 | 13,458 | +31,097 | +231.07% | skill 1/1; base 1/1 |
| codex | query-eval-010-scenario-lineage | 44,353 | 27,010 | +17,343 | +64.21% | skill 1/1; base 1/1 |
| codex | query-eval-011-prose-versus-rows | 44,327 | 13,204 | +31,123 | +235.71% | skill 1/1; base 1/1 |
| codex | query-eval-012-source-role | 28,391 | 40,510 | -12,119 | -29.92% | skill 1/1; base 1/1 |
| codex | query-eval-013-truncated-result | 44,251 | 13,236 | +31,015 | +234.32% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,101,369 | 1,371,944 | +729,425 | +53.17% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: Valid semantic version: 0.2.1
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 4 files for PII
- Code Integrity & Hygiene: Checking 3 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 4 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 3 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvidia-ontology-query': 160 char description
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent claude-code

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
