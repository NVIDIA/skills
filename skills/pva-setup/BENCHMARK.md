# Skill Benchmark: pva-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `pva-setup`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.3`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:cd27b6bb41c731994e49ddbe8b4df25c30cf9919ed71f166965f5bb79a16765e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| pva-setup | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| pva-setup | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| pva-setup | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| pva-setup | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| pva-setup | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `pva-setup`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 68.0% → 100.0% (+32.0 points) | 60.0% → 100.0% (+40.0 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 63.7% → 100.0% (+36.3 points) | 63.7% → 100.0% (+36.3 points) |
| Efficiency | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 96.4% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.3% → 90.9% (-1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 699,736 | 584,002 | +115,734 | +19.82% | skill 5/5; base 5/5 |
| claude-code | setup-driveos-out-of-scope | 74,447 | 35,362 | +39,085 | +110.53% | skill 1/1; base 1/1 |
| claude-code | setup-missing-installation-source | 239,660 | 185,209 | +54,451 | +29.40% | skill 1/1; base 1/1 |
| claude-code | setup-negative | 35,458 | 35,240 | +218 | +0.62% | skill 1/1; base 1/1 |
| claude-code | setup-release-mismatch-plan | 154,806 | 182,358 | -27,552 | -15.11% | skill 1/1; base 1/1 |
| claude-code | setup-solutions-default-source-plan | 195,365 | 145,833 | +49,532 | +33.96% | skill 1/1; base 1/1 |
| codex | All cases | 392,641 | 351,123 | +41,518 | +11.82% | skill 5/5; base 5/5 |
| codex | setup-driveos-out-of-scope | 28,816 | 54,843 | -26,027 | -47.46% | skill 1/1; base 1/1 |
| codex | setup-missing-installation-source | 127,666 | 88,815 | +38,851 | +43.74% | skill 1/1; base 1/1 |
| codex | setup-negative | 13,391 | 13,320 | +71 | +0.53% | skill 1/1; base 1/1 |
| codex | setup-release-mismatch-plan | 98,850 | 106,629 | -7,779 | -7.30% | skill 1/1; base 1/1 |
| codex | setup-solutions-default-source-plan | 123,918 | 87,516 | +36,402 | +41.59% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,092,377 | 935,125 | +157,252 | +16.82% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/pva-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/pva-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/pva-setup/SKILL.md`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`SKILL.md:82`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/pva-setup/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
