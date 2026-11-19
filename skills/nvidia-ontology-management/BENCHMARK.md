# Skill Benchmark: nvidia-ontology-management

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-management`
- Validation status: `passed`
- Evaluation date: 2026-11-19
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 14 evaluation tasks (14 positive)
- Dataset digest: `sha256:2a7e02bc91822e5415a1671f8a353b25b4cc6fe394c0a319553f56aff937e157` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-management | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

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
| Overall | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 96.4% (-3.6 points) | 100.0% → 92.9% (-7.1 points) |
| Correctness | 40.0% → 97.1% (+57.1 points) | 45.7% → 84.3% (+38.6 points) |
| Discoverability | 91.4% — baseline ran, but no comparable score was available; uplift unavailable | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 28.6% → 81.0% (+52.4 points) | 48.8% → 67.9% (+19.1 points) |
| Efficiency | 86.7% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.0% (-2.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.4% → 91.3% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,100,518 | 1,642,922 | +457,596 | +27.85% | skill 14/14; base 14/14 |
| claude-code | management-eval-001-mcp-readonly | 154,595 | 107,432 | +47,163 | +43.90% | skill 1/1; base 1/1 |
| claude-code | management-eval-002-validate-422 | 148,941 | 142,861 | +6,080 | +4.26% | skill 1/1; base 1/1 |
| claude-code | management-eval-003-reset-is-destructive | 152,153 | 178,413 | -26,260 | -14.72% | skill 1/1; base 1/1 |
| claude-code | management-eval-004-token-acts-as-owner | 109,910 | 142,693 | -32,783 | -22.97% | skill 1/1; base 1/1 |
| claude-code | management-eval-005-import-replace | 190,655 | 142,456 | +48,199 | +33.83% | skill 1/1; base 1/1 |
| claude-code | management-eval-006-lineage-semantic-layer | 261,816 | 106,309 | +155,507 | +146.28% | skill 1/1; base 1/1 |
| claude-code | management-eval-007-patch-vs-put-sql-attribute | 113,643 | 70,556 | +43,087 | +61.07% | skill 1/1; base 1/1 |
| claude-code | management-eval-008-run-grain-measure | 113,247 | 107,763 | +5,484 | +5.09% | skill 1/1; base 1/1 |
| claude-code | management-eval-009-relationship-evidence | 156,698 | 179,263 | -22,565 | -12.59% | skill 1/1; base 1/1 |
| claude-code | management-eval-010-import-exact-readback | 114,454 | 107,101 | +7,353 | +6.87% | skill 1/1; base 1/1 |
| claude-code | management-eval-011-result-publication-mode | 150,513 | 107,873 | +42,640 | +39.53% | skill 1/1; base 1/1 |
| claude-code | management-eval-012-promotion-readback-failure | 114,523 | 106,871 | +7,652 | +7.16% | skill 1/1; base 1/1 |
| claude-code | management-eval-013-composed-lifecycle | 204,860 | 107,802 | +97,058 | +90.03% | skill 1/1; base 1/1 |
| claude-code | management-eval-014-negative-probe-without-sql | 114,510 | 35,529 | +78,981 | +222.30% | skill 1/1; base 1/1 |
| codex | All cases | 829,742 | 633,911 | +195,831 | +30.89% | skill 14/14; base 14/14 |
| codex | management-eval-001-mcp-readonly | 100,740 | 41,889 | +58,851 | +140.49% | skill 1/1; base 1/1 |
| codex | management-eval-002-validate-422 | 72,405 | 42,538 | +29,867 | +70.21% | skill 1/1; base 1/1 |
| codex | management-eval-003-reset-is-destructive | 48,990 | 63,042 | -14,052 | -22.29% | skill 1/1; base 1/1 |
| codex | management-eval-004-token-acts-as-owner | 52,917 | 40,896 | +12,021 | +29.39% | skill 1/1; base 1/1 |
| codex | management-eval-005-import-replace | 53,409 | 40,973 | +12,436 | +30.35% | skill 1/1; base 1/1 |
| codex | management-eval-006-lineage-semantic-layer | 59,128 | 40,595 | +18,533 | +45.65% | skill 1/1; base 1/1 |
| codex | management-eval-007-patch-vs-put-sql-attribute | 49,037 | 60,623 | -11,586 | -19.11% | skill 1/1; base 1/1 |
| codex | management-eval-008-run-grain-measure | 44,230 | 13,478 | +30,752 | +228.16% | skill 1/1; base 1/1 |
| codex | management-eval-009-relationship-evidence | 43,964 | 54,301 | -10,337 | -19.04% | skill 1/1; base 1/1 |
| codex | management-eval-010-import-exact-readback | 49,183 | 69,527 | -20,344 | -29.26% | skill 1/1; base 1/1 |
| codex | management-eval-011-result-publication-mode | 44,245 | 41,090 | +3,155 | +7.68% | skill 1/1; base 1/1 |
| codex | management-eval-012-promotion-readback-failure | 44,897 | 54,588 | -9,691 | -17.75% | skill 1/1; base 1/1 |
| codex | management-eval-013-composed-lifecycle | 121,457 | 57,071 | +64,386 | +112.82% | skill 1/1; base 1/1 |
| codex | management-eval-014-negative-probe-without-sql | 45,140 | 13,300 | +31,840 | +239.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,930,260 | 2,276,833 | +653,427 | +28.70% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/write-api.md:
  "## Bulk import / export" in SKILL.md (lines 100-112)
  vs "## Model interchange" in references/write-api.md (lines 59-68) (`SKILL.md:100`)
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
