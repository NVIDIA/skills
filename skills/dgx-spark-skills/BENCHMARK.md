# Skill Benchmark: dgx-spark-skills

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `dgx-spark-skills`
- Validation status: `passed`
- Evaluation date: 2026-10-27
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 22 evaluation tasks (20 positive, 2 negative)
- Dataset digest: `sha256:b21a9c4ba8dfb32650eaf8b0098d878db91519bcc9af9e2b6c550198cbcab099` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| dgx-spark-skills | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| dgx-spark-skills | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| dgx-spark-skills | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| dgx-spark-skills | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| dgx-spark-skills | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `dgx-spark-skills`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 97.7% (-2.3 points) | 100.0% → 95.5% (-4.5 points) |
| Correctness | 76.4% → 97.3% (+20.9 points) | 77.3% → 88.2% (+10.9 points) |
| Discoverability | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 59.6% → 89.9% (+30.3 points) | 61.1% → 78.1% (+17.0 points) |
| Efficiency | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 75.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.7% → 93.6% (-4.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.7% → 87.8% (-0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,364,009 | 1,437,016 | +2,926,993 | +203.69% | skill 22/22; base 22/22 |
| claude-code | apache-spark-out-of-scope | 35,468 | 35,266 | +202 | +0.57% | skill 1/1; base 1/1 |
| claude-code | confirmed-64gb-memory-configuration | 116,557 | 36,170 | +80,387 | +222.25% | skill 1/1; base 1/1 |
| claude-code | contextual-inventory-reuse | 115,948 | 35,244 | +80,704 | +228.99% | skill 1/1; base 1/1 |
| claude-code | field-diagnostics-planning | 379,139 | 36,886 | +342,253 | +927.87% | skill 1/1; base 1/1 |
| claude-code | field-diagnostics-result-boundary | 260,144 | 35,371 | +224,773 | +635.47% | skill 1/1; base 1/1 |
| claude-code | founders-exact-identity-match | 158,474 | 36,113 | +122,361 | +338.83% | skill 1/1; base 1/1 |
| claude-code | github-spark-out-of-scope | 36,206 | 35,908 | +298 | +0.83% | skill 1/1; base 1/1 |
| claude-code | identify-system-guidance | 159,034 | 36,238 | +122,796 | +338.86% | skill 1/1; base 1/1 |
| claude-code | known-issue-oem-version-boundary | 207,924 | 36,155 | +171,769 | +475.09% | skill 1/1; base 1/1 |
| claude-code | known-issue-unified-memory | 203,297 | 72,793 | +130,504 | +179.28% | skill 1/1; base 1/1 |
| claude-code | oem-capacity-context | 157,192 | 35,831 | +121,361 | +338.70% | skill 1/1; base 1/1 |
| claude-code | oem-recovery-source-boundary | 203,346 | 36,003 | +167,343 | +464.80% | skill 1/1; base 1/1 |
| claude-code | playbook-selection | 357,734 | 332,802 | +24,932 | +7.49% | skill 1/1; base 1/1 |
| claude-code | playbook-stale-execution-source | 295,043 | 107,228 | +187,815 | +175.15% | skill 1/1; base 1/1 |
| claude-code | support-forum-source-boundary | 406,599 | 201,821 | +204,778 | +101.47% | skill 1/1; base 1/1 |
| claude-code | support-preparation | 162,019 | 36,558 | +125,461 | +343.18% | skill 1/1; base 1/1 |
| claude-code | support-redaction-check-only | 168,483 | 36,386 | +132,097 | +363.04% | skill 1/1; base 1/1 |
| claude-code | support-redaction-temporary-file-notice | 167,056 | 35,692 | +131,364 | +368.05% | skill 1/1; base 1/1 |
| claude-code | sync-cluster-scope | 198,590 | 35,759 | +162,831 | +455.36% | skill 1/1; base 1/1 |
| claude-code | sync-custom-app-draft | 201,497 | 36,416 | +165,081 | +453.32% | skill 1/1; base 1/1 |
| claude-code | sync-remote-access | 159,030 | 36,557 | +122,473 | +335.02% | skill 1/1; base 1/1 |
| claude-code | update-guidance | 215,229 | 109,819 | +105,410 | +95.99% | skill 1/1; base 1/1 |
| codex | All cases | 1,381,766 | 567,572 | +814,194 | +143.45% | skill 22/22; base 22/22 |
| codex | apache-spark-out-of-scope | 17,654 | 17,674 | -20 | -0.11% | skill 1/1; base 1/1 |
| codex | confirmed-64gb-memory-configuration | 45,639 | 13,545 | +32,094 | +236.94% | skill 1/1; base 1/1 |
| codex | contextual-inventory-reuse | 13,375 | 13,277 | +98 | +0.74% | skill 1/1; base 1/1 |
| codex | field-diagnostics-planning | 81,497 | 38,736 | +42,761 | +110.39% | skill 1/1; base 1/1 |
| codex | field-diagnostics-result-boundary | 85,292 | 29,852 | +55,440 | +185.72% | skill 1/1; base 1/1 |
| codex | founders-exact-identity-match | 65,145 | 13,508 | +51,637 | +382.27% | skill 1/1; base 1/1 |
| codex | github-spark-out-of-scope | 30,139 | 30,147 | -8 | -0.03% | skill 1/1; base 1/1 |
| codex | identify-system-guidance | 68,481 | 25,933 | +42,548 | +164.07% | skill 1/1; base 1/1 |
| codex | known-issue-oem-version-boundary | 70,469 | 30,335 | +40,134 | +132.30% | skill 1/1; base 1/1 |
| codex | known-issue-unified-memory | 68,689 | 28,252 | +40,437 | +143.13% | skill 1/1; base 1/1 |
| codex | oem-capacity-context | 61,946 | 37,078 | +24,868 | +67.07% | skill 1/1; base 1/1 |
| codex | oem-recovery-source-boundary | 71,000 | 30,440 | +40,560 | +133.25% | skill 1/1; base 1/1 |
| codex | playbook-selection | 148,381 | 38,954 | +109,427 | +280.91% | skill 1/1; base 1/1 |
| codex | playbook-stale-execution-source | 54,283 | 13,310 | +40,973 | +307.84% | skill 1/1; base 1/1 |
| codex | support-forum-source-boundary | 118,573 | 41,007 | +77,566 | +189.15% | skill 1/1; base 1/1 |
| codex | support-preparation | 67,314 | 22,111 | +45,203 | +204.44% | skill 1/1; base 1/1 |
| codex | support-redaction-check-only | 13,642 | 13,620 | +22 | +0.16% | skill 1/1; base 1/1 |
| codex | support-redaction-temporary-file-notice | 13,789 | 13,510 | +279 | +2.07% | skill 1/1; base 1/1 |
| codex | sync-cluster-scope | 69,442 | 25,871 | +43,571 | +168.42% | skill 1/1; base 1/1 |
| codex | sync-custom-app-draft | 65,040 | 24,422 | +40,618 | +166.32% | skill 1/1; base 1/1 |
| codex | sync-remote-access | 74,478 | 35,527 | +38,951 | +109.64% | skill 1/1; base 1/1 |
| codex | update-guidance | 77,498 | 30,463 | +47,035 | +154.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,745,775 | 2,004,588 | +3,741,187 | +186.63% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 45 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/dgx-spark-skills`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/dgx-spark-skills/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/dgx-spark-skills/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/dgx-spark-skills/SKILL.md`)
- **MEDIUM** SECURITY/MCP Least Privilege (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- 41 additional finding(s) are available in the full evaluation artifacts.

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
