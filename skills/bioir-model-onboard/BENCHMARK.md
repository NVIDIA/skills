# Skill Benchmark: model-onboard

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `model-onboard`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (7 positive, 2 negative)
- Dataset digest: `sha256:5a0da6e54e8b513fed09cf61c70a0c16dbe1009d7c1e1d13947071f80a5e2caf` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| model-onboard | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| model-onboard | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| model-onboard | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| model-onboard | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| model-onboard | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `model-onboard`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 95.6% → 95.6% (±0.0 points) | 80.0% → 88.9% (+8.9 points) |
| Discoverability | 55.7% — baseline ran, but no comparable score was available; uplift unavailable | 68.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 92.6% → 92.6% (±0.0 points) | 80.6% → 86.1% (+5.5 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 96.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 98.1% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.1% → 92.6% (-0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,163,926 | 1,140,793 | +23,133 | +2.03% | skill 9/9; base 9/9 |
| claude-code | model-onboard-001-feasibility | 147,371 | 145,816 | +1,555 | +1.07% | skill 1/1; base 1/1 |
| claude-code | model-onboard-002-checkpoint-accounting | 106,922 | 106,613 | +309 | +0.29% | skill 1/1; base 1/1 |
| claude-code | model-onboard-003-bin-edges | 106,598 | 106,313 | +285 | +0.27% | skill 1/1; base 1/1 |
| claude-code | model-onboard-004-structure-score | 106,452 | 142,298 | -35,846 | -25.19% | skill 1/1; base 1/1 |
| claude-code | model-onboard-005-kernel-shapes | 106,753 | 106,652 | +101 | +0.09% | skill 1/1; base 1/1 |
| claude-code | model-onboard-006-hub-registration | 271,426 | 180,119 | +91,307 | +50.69% | skill 1/1; base 1/1 |
| claude-code | model-onboard-007-module-swap-negative | 105,963 | 105,697 | +266 | +0.25% | skill 1/1; base 1/1 |
| claude-code | model-onboard-008-benchmark-order | 106,517 | 106,404 | +113 | +0.11% | skill 1/1; base 1/1 |
| claude-code | model-onboard-009-pipeline-only-negative | 105,924 | 140,881 | -34,957 | -24.81% | skill 1/1; base 1/1 |
| codex | All cases | 698,793 | 615,282 | +83,511 | +13.57% | skill 9/9; base 9/9 |
| codex | model-onboard-001-feasibility | 73,589 | 102,811 | -29,222 | -28.42% | skill 1/1; base 1/1 |
| codex | model-onboard-002-checkpoint-accounting | 85,504 | 55,756 | +29,748 | +53.35% | skill 1/1; base 1/1 |
| codex | model-onboard-003-bin-edges | 57,749 | 54,335 | +3,414 | +6.28% | skill 1/1; base 1/1 |
| codex | model-onboard-004-structure-score | 70,372 | 68,719 | +1,653 | +2.41% | skill 1/1; base 1/1 |
| codex | model-onboard-005-kernel-shapes | 108,037 | 69,667 | +38,370 | +55.08% | skill 1/1; base 1/1 |
| codex | model-onboard-006-hub-registration | 112,472 | 100,888 | +11,584 | +11.48% | skill 1/1; base 1/1 |
| codex | model-onboard-007-module-swap-negative | 68,279 | 53,948 | +14,331 | +26.56% | skill 1/1; base 1/1 |
| codex | model-onboard-008-benchmark-order | 68,502 | 55,266 | +13,236 | +23.95% | skill 1/1; base 1/1 |
| codex | model-onboard-009-pipeline-only-negative | 54,289 | 53,892 | +397 | +0.74% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,862,719 | 1,756,075 | +106,644 | +6.07% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/model-onboard/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/model-onboard/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/model-onboard/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/bench-perf-oss/SKILL.md (`SKILL.md:84`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/model-onboard/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
