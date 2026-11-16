# Skill Benchmark: nvidia-ontology-management

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-management`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:4c8455d88fc45fd16ded90344c740ea7149fe29edbca7f112b346c21177d23fa` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-management | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvidia-ontology-management`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.4% — baseline ran, but no comparable score was available; uplift unavailable | 82.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 96.2% → 84.6% (-11.6 points) |
| Correctness | 40.0% → 95.4% (+55.4 points) | 44.6% → 84.6% (+40.0 points) |
| Discoverability | 90.8% — baseline ran, but no comparable score was available; uplift unavailable | 94.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 23.7% → 70.5% (+46.8 points) | 47.4% → 61.5% (+14.1 points) |
| Efficiency | 85.2% — baseline ran, but no comparable score was available; uplift unavailable | 85.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 96.2% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.7% → 91.1% (-0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,824,072 | 1,606,351 | +217,721 | +13.55% | skill 13/13; base 13/13 |
| claude-code | management-eval-001-mcp-readonly | 158,099 | 107,426 | +50,673 | +47.17% | skill 1/1; base 1/1 |
| claude-code | management-eval-002-validate-422 | 111,861 | 142,707 | -30,846 | -21.61% | skill 1/1; base 1/1 |
| claude-code | management-eval-003-reset-is-destructive | 111,863 | 142,244 | -30,381 | -21.36% | skill 1/1; base 1/1 |
| claude-code | management-eval-004-token-acts-as-owner | 155,734 | 142,618 | +13,116 | +9.20% | skill 1/1; base 1/1 |
| claude-code | management-eval-005-import-replace | 192,127 | 142,590 | +49,537 | +34.74% | skill 1/1; base 1/1 |
| claude-code | management-eval-006-lineage-semantic-layer | 221,503 | 106,401 | +115,102 | +108.18% | skill 1/1; base 1/1 |
| claude-code | management-eval-007-patch-vs-put-sql-attribute | 113,535 | 178,348 | -64,813 | -36.34% | skill 1/1; base 1/1 |
| claude-code | management-eval-008-run-grain-measure | 112,808 | 72,042 | +40,766 | +56.59% | skill 1/1; base 1/1 |
| claude-code | management-eval-009-relationship-evidence | 152,112 | 142,264 | +9,848 | +6.92% | skill 1/1; base 1/1 |
| claude-code | management-eval-010-import-exact-readback | 112,957 | 106,863 | +6,094 | +5.70% | skill 1/1; base 1/1 |
| claude-code | management-eval-011-result-publication-mode | 110,623 | 108,031 | +2,592 | +2.40% | skill 1/1; base 1/1 |
| claude-code | management-eval-012-promotion-readback-failure | 112,878 | 106,888 | +5,990 | +5.60% | skill 1/1; base 1/1 |
| claude-code | management-eval-013-composed-lifecycle | 157,972 | 107,929 | +50,043 | +46.37% | skill 1/1; base 1/1 |
| codex | All cases | 1,002,425 | 583,965 | +418,460 | +71.66% | skill 13/13; base 13/13 |
| codex | management-eval-001-mcp-readonly | 81,638 | 56,170 | +25,468 | +45.34% | skill 1/1; base 1/1 |
| codex | management-eval-002-validate-422 | 79,056 | 54,223 | +24,833 | +45.80% | skill 1/1; base 1/1 |
| codex | management-eval-003-reset-is-destructive | 48,409 | 61,974 | -13,565 | -21.89% | skill 1/1; base 1/1 |
| codex | management-eval-004-token-acts-as-owner | 46,516 | 60,949 | -14,433 | -23.68% | skill 1/1; base 1/1 |
| codex | management-eval-005-import-replace | 52,776 | 41,663 | +11,113 | +26.67% | skill 1/1; base 1/1 |
| codex | management-eval-006-lineage-semantic-layer | 178,136 | 27,009 | +151,127 | +559.54% | skill 1/1; base 1/1 |
| codex | management-eval-007-patch-vs-put-sql-attribute | 48,253 | 52,789 | -4,536 | -8.59% | skill 1/1; base 1/1 |
| codex | management-eval-008-run-grain-measure | 44,155 | 17,847 | +26,308 | +147.41% | skill 1/1; base 1/1 |
| codex | management-eval-009-relationship-evidence | 44,842 | 40,375 | +4,467 | +11.06% | skill 1/1; base 1/1 |
| codex | management-eval-010-import-exact-readback | 49,007 | 33,499 | +15,508 | +46.29% | skill 1/1; base 1/1 |
| codex | management-eval-011-result-publication-mode | 44,140 | 41,260 | +2,880 | +6.98% | skill 1/1; base 1/1 |
| codex | management-eval-012-promotion-readback-failure | 44,694 | 41,000 | +3,694 | +9.01% | skill 1/1; base 1/1 |
| codex | management-eval-013-composed-lifecycle | 240,803 | 55,207 | +185,596 | +336.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,826,497 | 2,190,316 | +636,181 | +29.05% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/write-api.md:
  "## Bulk import / export" in SKILL.md (lines 97-109)
  vs "## Model interchange" in references/write-api.md (lines 59-68) (`SKILL.md:97`)
- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in modeling.md (`skills/nvidia-ontology-management/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nvidia-ontology-management/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
